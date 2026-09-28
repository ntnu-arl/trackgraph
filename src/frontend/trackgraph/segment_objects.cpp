/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */
#include <kimera_pgmo/mesh_delta.h>

#include "hydra/utils/mesh_utilities.h"
#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

void TrackGraphSegmentUpdater::remapExistingObjectConnections(
    const kimera_pgmo::MeshOffsetInfo& offsets, DynamicSceneGraph& graph) {
  std::vector<uint64_t> to_erase;
  std::vector<uint64_t> frozen_frontier;
  for (const auto object_uid : active_frontier_) {
    auto record_iter = active_objects_.find(object_uid);
    if (record_iter == active_objects_.end()) {
      to_erase.push_back(object_uid);
      continue;
    }

    auto& record = record_iter->second;
    if (!graph.hasNode(record.node_id)) {
      to_erase.push_back(object_uid);
      continue;
    }

    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    kimera_pgmo::MeshOffsetInfo::RemapStats support_stats;
    offsets.remapVertexIndices(attrs.mesh_connections, &support_stats);
    refreshTrackedSupportState(offsets, attrs, &record);
    if (support_stats.all_archived || attrs.mesh_connections.empty()) {
      frozen_frontier.push_back(object_uid);
    }
  }

  for (const auto object_uid : to_erase) {
    active_frontier_.erase(object_uid);
    auto object_it = active_objects_.find(object_uid);
    if (object_it != active_objects_.end()) {
      for (const auto track_id : getAssociatedTrackIds(object_it->second)) {
        track_to_object_uid_.erase(track_id);
      }
      active_objects_.erase(object_it);
    }
  }

  for (const auto object_uid : frozen_frontier) {
    active_frontier_.erase(object_uid);
  }
}

void TrackGraphSegmentUpdater::markMissingTracksInactive(
    uint64_t /*timestamp_ns*/,
    const std::unordered_map<uint32_t, TrackObservation>& observations,
    DynamicSceneGraph& graph) {
  if (!config.mark_missing_tracks_inactive) {
    return;
  }

  for (auto& [object_uid, record] : active_objects_) {
    (void)object_uid;
    bool any_track_observed = false;
    forEachAssociatedTrackId(record, [&](uint32_t track_id) {
      if (any_track_observed) {
        return;
      }
      if (observations.count(track_id)) {
        any_track_observed = true;
      }
    });

    if (any_track_observed) {
      record.untracked_updates = 0;
      continue;
    }

    record.is_tracked = false;
    ++record.untracked_updates;
    if (!graph.hasNode(record.node_id)) {
      continue;
    }

    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    markTrackedMetadata(attrs, record);
    backend_sync_node_ids_.insert(record.node_id);
  }
}

void TrackGraphSegmentUpdater::updateExistingObject(
    uint64_t timestamp_ns,
    uint32_t track_id,
    bool tracking_is_keyframe,
    const TrackObservation& observation,
    const TrackVertexEvidence& evidence,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  const auto uid_iter = track_to_object_uid_.find(track_id);
  if (uid_iter == track_to_object_uid_.end()) {
    return;
  }

  const auto record_iter = active_objects_.find(uid_iter->second);
  if (record_iter == active_objects_.end()) {
    return;
  }

  auto& record = record_iter->second;
  const bool was_tracked = record.is_tracked;
  const auto previous_untracked_updates = record.untracked_updates;
  const bool source_track_color_changed = rememberSourceTrackColor(record, observation);
  record.is_tracked = true;
  record.last_observed_ns = timestamp_ns;
  record.untracked_updates = 0;

  if (!graph.hasNode(record.node_id)) {
    return;
  }

  auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
  attrs.last_update_time_ns = timestamp_ns;
  const auto prev_feature = attrs.semantic_feature;
  setTrackedPrototypeIfAvailable(observation, attrs);
  if (isValidSemanticFeature(attrs.semantic_feature) &&
      (!isValidSemanticFeature(prev_feature) ||
       !attrs.semantic_feature.isApprox(prev_feature))) {
    bumpPrototypeGeneration(record);
  }

  ++record.frames_observed;
  if (tracking_is_keyframe) {
    ++record.keyframes_observed;
  }

  const auto prev_support_size = attrs.mesh_connections.size();
  mergeUniqueIndices(attrs.mesh_connections, evidence.confirmed_vertex_indices);
  if (attrs.mesh_connections.size() != prev_support_size) {
    bumpSupportGeneration(record);
  }
  refreshTrackedSupportState(offsets, attrs, &record);
  if (attrs.is_active) {
    active_frontier_.insert(record.object_uid);
  } else {
    active_frontier_.erase(record.object_uid);
  }

  if (!was_tracked || previous_untracked_updates != 0u || source_track_color_changed) {
    markTrackedMetadata(attrs, record);
    backend_sync_node_ids_.insert(record.node_id);
  }

  if (evidence.confirmed_vertex_indices.size() < config.min_points_update) {
    return;
  }

  if (!attrs.mesh_connections.empty()) {
    ScopedTimer geometry_timer(kTrackedObjectGeometryTimer, timestamp_ns);
    updateTrackedObjectGeometry(
        *graph.mesh(), attrs, getAssociatedTrackIds(record), config.bounding_box_type);
  }

  ++record.num_track_updates;
}

void TrackGraphSegmentUpdater::applyOwnedWinnerSupport(
    uint64_t timestamp_ns,
    const WinnerVertexMap& owned_winner_vertices_by_track,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  if (owned_winner_vertices_by_track.empty()) {
    return;
  }

  for (const auto& [track_id, vertex_indices] : owned_winner_vertices_by_track) {
    if (vertex_indices.empty()) {
      continue;
    }

    const auto uid_iter = track_to_object_uid_.find(track_id);
    if (uid_iter == track_to_object_uid_.end() || uid_iter->second == 0u) {
      continue;
    }

    auto record_iter = active_objects_.find(uid_iter->second);
    if (record_iter == active_objects_.end()) {
      continue;
    }

    auto& record = record_iter->second;
    if (!graph.hasNode(record.node_id)) {
      continue;
    }

    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    if (!getTrackedObjectNodeAttributes(attrs)) {
      continue;
    }

    const auto prev_support_size = attrs.mesh_connections.size();
    mergeUniqueIndices(attrs.mesh_connections, vertex_indices);
    if (attrs.mesh_connections.size() == prev_support_size) {
      continue;
    }

    bumpSupportGeneration(record);
    attrs.last_update_time_ns = timestamp_ns;
    refreshTrackedSupportState(offsets, attrs, &record);
    if (attrs.is_active) {
      active_frontier_.insert(record.object_uid);
    } else {
      active_frontier_.erase(record.object_uid);
    }

    if (graph.hasMesh() && !attrs.mesh_connections.empty()) {
      ScopedTimer geometry_timer(kTrackedObjectGeometryTimer, timestamp_ns);
      updateTrackedObjectGeometry(*graph.mesh(),
                                  attrs,
                                  getAssociatedTrackIds(record),
                                  config.bounding_box_type);
    }

    markTrackedMetadata(attrs, record);
    backend_sync_node_ids_.insert(record.node_id);
  }
}

void TrackGraphSegmentUpdater::createObjectForTrack(
    uint64_t timestamp_ns,
    uint32_t track_id,
    bool tracking_is_keyframe,
    const TrackObservation& observation,
    const TrackVertexEvidence& evidence,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph,
    bool use_raw_winner_creation_support,
    const ProvisionalTrackedObjectRecord* provisional) {
  const auto& creation_vertex_indices = use_raw_winner_creation_support
                                            ? evidence.global_vertex_indices
                                            : evidence.confirmed_vertex_indices;
  if (creation_vertex_indices.size() < config.min_points_create) {
    return;
  }

  ObjectRecord record;
  record.object_uid = next_object_uid_++;
  record.node_id = next_node_id_;
  record.source_track_id = track_id;
  record.source_track_ids = {track_id};
  rememberSourceTrackColor(record, observation);
  record.first_observed_ns =
      provisional ? provisional->first_observed_ns : timestamp_ns;
  record.last_observed_ns = timestamp_ns;
  record.num_track_updates = 1;
  record.frames_observed = provisional ? provisional->frames_observed : 1u;
  // Source tracks are born on keyframes upstream. If Hydra delays object creation
  // until a later propagation frame because mesh support is initially too small,
  // preserve that source-track keyframe provenance instead of starting at zero.
  const auto provisional_keyframes =
      provisional ? provisional->keyframes_observed
                  : static_cast<uint32_t>(tracking_is_keyframe ? 1u : 0u);
  record.keyframes_observed = std::max<uint32_t>(1u, provisional_keyframes);
  record.untracked_updates = 0;
  record.state_generation = 1;
  record.support_generation = provisional ? provisional->support_generation : 1u;
  record.prototype_generation = provisional ? provisional->prototype_generation : 1u;
  record.is_tracked = true;

  auto attrs = std::make_unique<TrackedObjectNodeAttributes>();
  attrs->last_update_time_ns = timestamp_ns;
  attrs->semantic_label = SemanticNodeAttributes::NO_SEMANTIC_LABEL;
  attrs->name = "track_" + std::to_string(track_id);
  attrs->color = getObjectCreationColor(observation);
  setTrackedPrototypeIfAvailable(observation, *attrs);
  if (!isValidSemanticFeature(attrs->semantic_feature) && provisional &&
      isValidSemanticFeature(provisional->semantic_feature)) {
    attrs->semantic_feature = provisional->semantic_feature;
  }
  attrs->mesh_connections.insert(attrs->mesh_connections.begin(),
                                 creation_vertex_indices.begin(),
                                 creation_vertex_indices.end());
  refreshTrackedSupportState(offsets, *attrs, &record);
  syncTrackedNodeIdentity(*attrs, record);
  {
    ScopedTimer geometry_timer(kTrackedObjectGeometryTimer, timestamp_ns);
    updateTrackedObjectGeometry(
        *graph.mesh(), *attrs, getAssociatedTrackIds(record), config.bounding_box_type);
  }
  markTrackedMetadata(*attrs, record);

  graph.emplaceNode(config.layer_id, next_node_id_, std::move(attrs));
  track_to_object_uid_[track_id] = record.object_uid;
  active_objects_[record.object_uid] = record;
  if (graph.getNode(next_node_id_).attributes<ObjectNodeAttributes>().is_active) {
    active_frontier_.insert(record.object_uid);
  }
  provisional_tracks_.erase(track_id);
  ++next_node_id_;
}

bool TrackGraphSegmentUpdater::adoptTrackIntoExistingRoot(
    uint64_t timestamp_ns,
    uint32_t track_id,
    bool tracking_is_keyframe,
    const TrackObservation& observation,
    const TrackVertexEvidence& evidence,
    uint64_t target_root_uid,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  if (target_root_uid == 0u) {
    return false;
  }

  track_to_object_uid_[track_id] = target_root_uid;
  auto record_iter = active_objects_.find(target_root_uid);
  if (record_iter == active_objects_.end()) {
    return false;
  }

  auto& record = record_iter->second;
  if (record.source_track_ids.empty() && record.source_track_id != 0u) {
    record.source_track_ids.push_back(record.source_track_id);
  }
  appendUniqueTrackIds(record.source_track_ids, {track_id});
  rememberSourceTrackColor(record, observation);
  const auto provisional_iter = provisional_tracks_.find(track_id);
  if (provisional_iter != provisional_tracks_.end()) {
    const auto& provisional = provisional_iter->second;
    record.first_observed_ns =
        std::min(record.first_observed_ns, provisional.first_observed_ns);
    record.last_observed_ns =
        std::max(record.last_observed_ns, provisional.last_observed_ns);
    record.frames_observed += provisional.frames_observed;
    record.keyframes_observed += provisional.keyframes_observed;
    record.support_generation =
        std::max(record.support_generation, provisional.support_generation);
    record.prototype_generation =
        std::max(record.prototype_generation, provisional.prototype_generation);
  }
  updateExistingObject(timestamp_ns,
                       track_id,
                       tracking_is_keyframe,
                       observation,
                       evidence,
                       offsets,
                       graph);
  if (graph.hasNode(record.node_id)) {
    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    syncTrackedNodeIdentity(attrs, record);
    markTrackedMetadata(attrs, record);
  }
  provisional_tracks_.erase(track_id);
  return true;
}

void TrackGraphSegmentUpdater::freezeInactiveArchivedObjects(DynamicSceneGraph& graph) {
  std::vector<uint64_t> to_freeze;
  for (const auto& [object_uid, record] : active_objects_) {
    if (!graph.hasNode(record.node_id)) {
      continue;
    }

    const auto& attrs =
        graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    if (!record.is_tracked && !attrs.is_active) {
      to_freeze.push_back(object_uid);
    }
  }

  for (const auto object_uid : to_freeze) {
    auto iter = active_objects_.find(object_uid);
    if (iter == active_objects_.end()) {
      continue;
    }

    if (graph.hasNode(iter->second.node_id)) {
      auto& attrs =
          graph.getNode(iter->second.node_id).attributes<ObjectNodeAttributes>();
      markTrackedMetadata(attrs, iter->second);
      backend_sync_node_ids_.insert(iter->second.node_id);
    }

    iter->second.historical_revived_candidate_override = false;
    frozen_candidates_[object_uid] = makeHistoricalRecord(iter->second);
    active_frontier_.erase(object_uid);
    for (const auto track_id : getAssociatedTrackIds(iter->second)) {
      track_to_object_uid_.erase(track_id);
    }
    active_objects_.erase(iter);
  }
}

}  // namespace hydra
