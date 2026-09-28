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
#include "hydra/utils/mesh_utilities.h"
#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

void TrackGraphSegmentUpdater::applyActiveReidResults(
    const ActiveSegmentReid::ReidentificationResult& result,
    uint64_t timestamp_ns,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  for (const auto& [duplicate_object_uid, existing_object_uid] : result.merges) {
    if (duplicate_object_uid == existing_object_uid) {
      continue;
    }

    mergeTrackedObjectIntoExisting(
        existing_object_uid, duplicate_object_uid, timestamp_ns, offsets, graph);
  }
}

bool TrackGraphSegmentUpdater::mergeTrackedObjectIntoExisting(
    uint64_t existing_object_uid,
    uint64_t duplicate_object_uid,
    uint64_t timestamp_ns,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph,
    bool enable_historical_candidate_override) {
  if (existing_object_uid == duplicate_object_uid) {
    return false;
  }

  const auto duplicate_record_iter = active_objects_.find(duplicate_object_uid);
  if (duplicate_record_iter == active_objects_.end()) {
    return false;
  }

  auto existing_record_iter = active_objects_.find(existing_object_uid);
  if (existing_record_iter == active_objects_.end()) {
    auto frozen_iter = frozen_candidates_.find(existing_object_uid);
    if (frozen_iter == frozen_candidates_.end()) {
      return false;
    }

    active_objects_[existing_object_uid] = makeActiveRecord(frozen_iter->second);
    if (graph.hasNode(frozen_iter->second.node_id) &&
        graph.getNode(frozen_iter->second.node_id)
            .attributes<ObjectNodeAttributes>()
            .is_active) {
      active_frontier_.insert(existing_object_uid);
    }
    frozen_candidates_.erase(frozen_iter);
    existing_record_iter = active_objects_.find(existing_object_uid);
  }

  auto& existing_record = existing_record_iter->second;
  auto& duplicate_record = duplicate_record_iter->second;
  if (!graph.hasNode(existing_record.node_id) ||
      !graph.hasNode(duplicate_record.node_id)) {
    return false;
  }

  const auto duplicate_node_id = duplicate_record.node_id;
  const auto transferred_track_ids = getAssociatedTrackIds(duplicate_record);

  auto& existing_attrs =
      graph.getNode(existing_record.node_id).attributes<ObjectNodeAttributes>();
  auto& duplicate_attrs =
      graph.getNode(duplicate_node_id).attributes<ObjectNodeAttributes>();

  const auto prev_support_size = existing_attrs.mesh_connections.size();
  mergeUniqueIndices(existing_attrs.mesh_connections, duplicate_attrs.mesh_connections);
  if (existing_attrs.mesh_connections.size() != prev_support_size) {
    bumpSupportGeneration(existing_record);
  }
  if (isValidSemanticFeature(duplicate_attrs.semantic_feature)) {
    if (!isValidSemanticFeature(existing_attrs.semantic_feature) ||
        !duplicate_attrs.semantic_feature.isApprox(existing_attrs.semantic_feature)) {
      bumpPrototypeGeneration(existing_record);
    }
    existing_attrs.semantic_feature = duplicate_attrs.semantic_feature;
  }
  mergeOpenVocabFeatures(existing_attrs,
                         existing_record,
                         duplicate_attrs,
                         duplicate_record,
                         config.open_vocab.verbosity);

  if (existing_record.source_track_ids.empty() &&
      existing_record.source_track_id != 0u) {
    existing_record.source_track_ids.push_back(existing_record.source_track_id);
  }
  appendUniqueTrackIds(existing_record.source_track_ids, transferred_track_ids);
  mergeTrackColors(existing_record.source_track_colors,
                   duplicate_record.source_track_colors);

  existing_record.first_observed_ns =
      std::min(existing_record.first_observed_ns, duplicate_record.first_observed_ns);
  existing_record.last_observed_ns = std::max(
      std::max(existing_record.last_observed_ns, duplicate_record.last_observed_ns),
      timestamp_ns);
  existing_record.num_track_updates += duplicate_record.num_track_updates;
  existing_record.frames_observed += duplicate_record.frames_observed;
  existing_record.keyframes_observed += duplicate_record.keyframes_observed;
  existing_record.is_tracked =
      existing_record.is_tracked || duplicate_record.is_tracked;
  existing_record.untracked_updates =
      existing_record.is_tracked ? 0u
                                 : std::min(existing_record.untracked_updates,
                                            duplicate_record.untracked_updates);
  existing_record.state_generation =
      std::max(existing_record.state_generation, duplicate_record.state_generation);
  existing_record.support_generation =
      std::max(existing_record.support_generation, duplicate_record.support_generation);
  existing_record.prototype_generation = std::max(
      existing_record.prototype_generation, duplicate_record.prototype_generation);
  bumpStateGeneration(existing_record);
  if (enable_historical_candidate_override) {
    existing_record.historical_revived_candidate_override = true;
  }

  existing_record.has_historical_association =
      existing_record.has_historical_association ||
      duplicate_record.has_historical_association;
  refreshTrackedSupportState(offsets, existing_attrs, &existing_record);
  existing_attrs.last_update_time_ns = timestamp_ns;
  if (!existing_attrs.mesh_connections.empty()) {
    ScopedTimer geometry_timer(kTrackedObjectGeometryTimer, timestamp_ns);
    updateTrackedObjectGeometry(*graph.mesh(),
                                existing_attrs,
                                getAssociatedTrackIds(existing_record),
                                config.bounding_box_type);
  }

  for (const auto track_id : transferred_track_ids) {
    track_to_object_uid_[track_id] = existing_object_uid;
  }
  if (existing_attrs.is_active) {
    active_frontier_.insert(existing_object_uid);
  } else {
    active_frontier_.erase(existing_object_uid);
  }

  syncTrackedNodeIdentity(existing_attrs, existing_record);
  markTrackedMetadata(existing_attrs, existing_record);
  backend_sync_node_ids_.insert(existing_record.node_id);

  graph.removeNode(duplicate_node_id);
  active_frontier_.erase(duplicate_object_uid);
  active_objects_.erase(duplicate_object_uid);
  return true;
}

}  // namespace hydra
