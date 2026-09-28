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
#include <glog/logging.h>

#include <algorithm>
#include <sstream>

#include "hydra/frontend/trackgraph_segment_updater.h"

namespace hydra {

TrackGraphSegmentUpdater::~TrackGraphSegmentUpdater() {
  if (!config.log_memory_stats || last_memory_stats_payload_.empty()) {
    return;
  }

  LOG(INFO) << "[tracked-object-memory] snapshot: " << last_memory_stats_payload_
            << " final=true";
}

TrackGraphSegmentUpdater::EvidenceMemoryStats
TrackGraphSegmentUpdater::collectEvidenceMemoryStats(
    const std::unordered_map<uint32_t, TrackObservation>& observations,
    const TrackVertexEvidenceMap& evidence_by_track,
    const WinnerVertexMap& owned_winner_vertices_by_track,
    const std::unordered_map<uint32_t, RootSupportMap>& track_candidate_roots,
    const RootSupportMap& root_support,
    const RootPairEvidenceMap& pair_evidence,
    const RootVertexSetMap& root_support_vertices,
    const RootPairVertexSetMap& pair_evidence_vertices) const {
  EvidenceMemoryStats stats;
  stats.observation_records = observations.size();
  stats.observation_buckets = observations.bucket_count();
  for (const auto& [track_id, observation] : observations) {
    (void)track_id;
    stats.observation_feature_floats +=
        static_cast<size_t>(observation.prototype.size());
  }

  stats.evidence_records = evidence_by_track.size();
  stats.evidence_buckets = evidence_by_track.bucket_count();
  auto add_index_vector = [&](const auto& indices) {
    stats.evidence_index_entries += indices.size();
    stats.evidence_index_capacity += indices.capacity();
  };
  auto add_candidate_map = [&](const auto& candidates) {
    stats.evidence_nested_map_entries += candidates.size();
    stats.evidence_nested_map_buckets += candidates.bucket_count();
    for (const auto& [root_uid, indices] : candidates) {
      (void)root_uid;
      add_index_vector(indices);
    }
  };
  for (const auto& [track_id, evidence] : evidence_by_track) {
    (void)track_id;
    add_index_vector(evidence.global_vertex_indices);
    add_index_vector(evidence.confirmed_vertex_indices);
    add_index_vector(evidence.association_vertex_indices);
    add_index_vector(evidence.confirmed_association_vertex_indices);
    add_index_vector(evidence.raw_winner_association_vertex_indices);
    add_candidate_map(evidence.candidate_root_vertex_indices);
    add_candidate_map(evidence.confirmed_candidate_root_vertex_indices);
    add_candidate_map(evidence.raw_winner_overlap_candidate_root_vertex_indices);
  }

  stats.winner_track_entries = owned_winner_vertices_by_track.size();
  stats.winner_map_buckets = owned_winner_vertices_by_track.bucket_count();
  for (const auto& [track_id, indices] : owned_winner_vertices_by_track) {
    (void)track_id;
    stats.winner_index_entries += indices.size();
    stats.winner_index_capacity += indices.capacity();
  }

  stats.candidate_root_track_entries = track_candidate_roots.size();
  stats.candidate_root_outer_buckets = track_candidate_roots.bucket_count();
  for (const auto& [track_id, candidates] : track_candidate_roots) {
    (void)track_id;
    stats.candidate_root_entries += candidates.size();
    stats.candidate_root_inner_buckets += candidates.bucket_count();
  }

  stats.root_support_entries = root_support.size();
  stats.root_support_buckets = root_support.bucket_count();
  stats.pair_evidence_entries = pair_evidence.size();
  stats.pair_evidence_buckets = pair_evidence.bucket_count();

  stats.root_vertex_set_entries = root_support_vertices.size();
  stats.root_vertex_set_outer_buckets = root_support_vertices.bucket_count();
  for (const auto& [root_uid, vertices] : root_support_vertices) {
    (void)root_uid;
    stats.root_vertex_entries += vertices.size();
    stats.root_vertex_inner_buckets += vertices.bucket_count();
  }

  stats.pair_vertex_set_entries = pair_evidence_vertices.size();
  stats.pair_vertex_set_outer_buckets = pair_evidence_vertices.bucket_count();
  for (const auto& [root_pair, vertices] : pair_evidence_vertices) {
    (void)root_pair;
    stats.pair_vertex_entries += vertices.size();
    stats.pair_vertex_inner_buckets += vertices.bucket_count();
  }

  return stats;
}

void TrackGraphSegmentUpdater::recordMemoryStats(
    uint64_t timestamp_ns,
    const DynamicSceneGraph& graph,
    const EvidenceMemoryStats& evidence_stats) {
  if (!config.log_memory_stats) {
    return;
  }

  size_t object_records = 0;
  size_t graph_nodes_present = 0;
  size_t graph_nodes_missing = 0;
  size_t mesh_connection_entries = 0;
  size_t max_mesh_connections = 0;
  size_t semantic_feature_floats = 0;
  size_t open_vocab_feature_floats = 0;
  size_t source_track_entries = 0;
  size_t source_track_capacity = 0;
  size_t source_color_entries = 0;
  size_t open_vocab_source_entries = 0;
  size_t open_vocab_source_capacity = 0;

  auto add_record = [&](const auto& record) {
    ++object_records;
    source_track_entries += record.source_track_ids.size();
    source_track_capacity += record.source_track_ids.capacity();
    source_color_entries += record.source_track_colors.size();
    open_vocab_source_entries += record.open_vocab_feature_source_track_ids.size();
    open_vocab_source_capacity += record.open_vocab_feature_source_track_ids.capacity();

    if (!graph.hasNode(record.node_id)) {
      ++graph_nodes_missing;
      return;
    }

    ++graph_nodes_present;
    const auto& attrs =
        graph.getNode(record.node_id).template attributes<ObjectNodeAttributes>();
    mesh_connection_entries += attrs.mesh_connections.size();
    max_mesh_connections =
        std::max(max_mesh_connections, attrs.mesh_connections.size());
    semantic_feature_floats += static_cast<size_t>(attrs.semantic_feature.size());
    open_vocab_feature_floats += static_cast<size_t>(attrs.open_vocab_features.size());
  };
  for (const auto& [object_uid, record] : active_objects_) {
    (void)object_uid;
    add_record(record);
  }
  for (const auto& [object_uid, record] : frozen_candidates_) {
    (void)object_uid;
    add_record(record);
  }

  size_t provisional_feature_floats = 0;

  for (const auto& [track_id, record] : provisional_tracks_) {
    (void)track_id;
    provisional_feature_floats += static_cast<size_t>(record.semantic_feature.size());
  }

  ++memory_stats_update_count_;
  std::ostringstream message;
  message
      << "ts=" << timestamp_ns << " update_index=" << memory_stats_update_count_
      << " active_objects=" << active_objects_.size()
      << " active_object_buckets=" << active_objects_.bucket_count()
      << " frozen_objects=" << frozen_candidates_.size()
      << " frozen_object_buckets=" << frozen_candidates_.bucket_count()
      << " object_records=" << object_records
      << " graph_nodes_present=" << graph_nodes_present
      << " graph_nodes_missing=" << graph_nodes_missing
      << " mesh_connection_entries=" << mesh_connection_entries
      << " max_mesh_connections=" << max_mesh_connections
      << " semantic_feature_floats=" << semantic_feature_floats
      << " open_vocab_feature_floats=" << open_vocab_feature_floats
      << " source_track_entries=" << source_track_entries
      << " source_track_capacity=" << source_track_capacity
      << " source_color_entries=" << source_color_entries
      << " open_vocab_source_entries=" << open_vocab_source_entries
      << " open_vocab_source_capacity=" << open_vocab_source_capacity

      << " track_to_object_entries=" << track_to_object_uid_.size()
      << " track_to_object_buckets=" << track_to_object_uid_.bucket_count()
      << " provisional_tracks=" << provisional_tracks_.size()
      << " provisional_track_buckets=" << provisional_tracks_.bucket_count()
      << " provisional_feature_floats=" << provisional_feature_floats

      << " active_frontier_entries=" << active_frontier_.size()
      << " active_frontier_buckets=" << active_frontier_.bucket_count()
      << " ignore_decision_entries=" << source_track_ignore_decisions_.size()
      << " ignore_decision_buckets=" << source_track_ignore_decisions_.bucket_count()
      << " backend_sync_entries=" << backend_sync_node_ids_.size()
      << " backend_sync_buckets=" << backend_sync_node_ids_.bucket_count()
      << " open_vocab_cache_entries="
      << (open_vocab_feature_cache_ ? open_vocab_feature_cache_->size() : 0u)
      << " observation_records=" << evidence_stats.observation_records
      << " observation_buckets=" << evidence_stats.observation_buckets
      << " observation_feature_floats=" << evidence_stats.observation_feature_floats
      << " evidence_records=" << evidence_stats.evidence_records
      << " evidence_buckets=" << evidence_stats.evidence_buckets
      << " evidence_index_entries=" << evidence_stats.evidence_index_entries
      << " evidence_index_capacity=" << evidence_stats.evidence_index_capacity
      << " evidence_nested_map_entries=" << evidence_stats.evidence_nested_map_entries
      << " evidence_nested_map_buckets=" << evidence_stats.evidence_nested_map_buckets
      << " winner_track_entries=" << evidence_stats.winner_track_entries
      << " winner_map_buckets=" << evidence_stats.winner_map_buckets
      << " winner_index_entries=" << evidence_stats.winner_index_entries
      << " winner_index_capacity=" << evidence_stats.winner_index_capacity
      << " candidate_root_track_entries=" << evidence_stats.candidate_root_track_entries
      << " candidate_root_outer_buckets=" << evidence_stats.candidate_root_outer_buckets
      << " candidate_root_entries=" << evidence_stats.candidate_root_entries
      << " candidate_root_inner_buckets=" << evidence_stats.candidate_root_inner_buckets
      << " root_support_entries=" << evidence_stats.root_support_entries
      << " root_support_buckets=" << evidence_stats.root_support_buckets
      << " pair_evidence_entries=" << evidence_stats.pair_evidence_entries
      << " pair_evidence_buckets=" << evidence_stats.pair_evidence_buckets
      << " root_vertex_set_entries=" << evidence_stats.root_vertex_set_entries
      << " root_vertex_set_outer_buckets="
      << evidence_stats.root_vertex_set_outer_buckets
      << " root_vertex_entries=" << evidence_stats.root_vertex_entries
      << " root_vertex_inner_buckets=" << evidence_stats.root_vertex_inner_buckets
      << " pair_vertex_set_entries=" << evidence_stats.pair_vertex_set_entries
      << " pair_vertex_set_outer_buckets="
      << evidence_stats.pair_vertex_set_outer_buckets
      << " pair_vertex_entries=" << evidence_stats.pair_vertex_entries
      << " pair_vertex_inner_buckets=" << evidence_stats.pair_vertex_inner_buckets;
  last_memory_stats_payload_ = message.str();

  if (memory_stats_update_count_ == 1u ||
      memory_stats_update_count_ % config.memory_stats_log_interval == 0u) {
    LOG(INFO) << "[tracked-object-memory] snapshot: " << last_memory_stats_payload_
              << " final=false";
  }
}

}  // namespace hydra
