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
#include <limits>

#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

TrackGraphSegmentUpdater::AdoptionDecision TrackGraphSegmentUpdater::decideAdoption(
    uint32_t track_id,
    const TrackObservation& obs,
    const TrackVertexEvidence& evidence,
    const std::unordered_map<uint32_t, RootSupportMap>& track_candidate_roots,
    const RootSupportMap& root_support,
    const DynamicSceneGraph& graph) const {
  AdoptionDecision decision;
  const size_t association_support = evidence.association_vertex_indices.size();
  const auto candidate_iter = track_candidate_roots.find(track_id);
  if (candidate_iter == track_candidate_roots.end() || candidate_iter->second.empty()) {
    return decision;
  }

  struct Candidate {
    uint64_t root_uid;
    size_t shared;
    float similarity;
    bool direct_alias;
  };
  std::vector<Candidate> accepted;
  accepted.reserve(candidate_iter->second.size());
  for (const auto& [root_uid, shared] : candidate_iter->second) {
    if (shared < config.reidentifier.min_shared_vertices) {
      continue;
    }
    const auto root = active_objects_.find(root_uid);
    if (root == active_objects_.end() || !graph.hasNode(root->second.node_id)) {
      continue;
    }
    const auto& attrs =
        graph.getNode(root->second.node_id).attributes<ObjectNodeAttributes>();
    const float similarity =
        computeFeatureSimilarity(obs.prototype, attrs.semantic_feature);
    if (similarity < 0.0f) {
      continue;
    }
    size_t existing_support =
        root_support.count(root_uid) ? root_support.at(root_uid) : 0u;
    if (existing_support == 0u) {
      existing_support = shared;
    }
    if (association_support == 0u || existing_support == 0u) {
      continue;
    }

    const double fraction_new_inside_existing =
        static_cast<double>(shared) / static_cast<double>(association_support);
    const double fraction_existing_inside_new =
        static_cast<double>(shared) / static_cast<double>(existing_support);
    const bool symmetric =
        std::min(fraction_new_inside_existing, fraction_existing_inside_new) >=
        config.reidentifier.symmetric_overlap_threshold;
    const bool track_inside_candidate =
        fraction_new_inside_existing >=
            config.reidentifier.contained_overlap_threshold &&
        fraction_existing_inside_new <=
            config.reidentifier.contained_reverse_overlap_max;
    const bool candidate_inside_track =
        fraction_existing_inside_new >=
            config.reidentifier.contained_overlap_threshold &&
        fraction_new_inside_existing <=
            config.reidentifier.contained_reverse_overlap_max;
    if (symmetric &&
        similarity >= config.reidentifier.feature_similarity_reid_threshold) {
      accepted.push_back({root_uid, shared, similarity, true});
    } else if ((track_inside_candidate || candidate_inside_track) &&
               similarity >= config.reidentifier.feature_similarity_partof_low) {
      accepted.push_back({root_uid, shared, similarity, false});
    }
  }

  // Prefer direct aliases, then shared support, appearance, and finally stable UID.
  std::sort(
      accepted.begin(), accepted.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.direct_alias != rhs.direct_alias)
          return lhs.direct_alias > rhs.direct_alias;
        if (lhs.shared != rhs.shared) return lhs.shared > rhs.shared;
        if (lhs.similarity != rhs.similarity) return lhs.similarity > rhs.similarity;
        return lhs.root_uid < rhs.root_uid;
      });
  if (accepted.empty()) {
    return decision;
  }
  if (accepted.size() > 1u && accepted[0].shared == accepted[1].shared &&
      std::abs(accepted[0].similarity - accepted[1].similarity) < 0.05f) {
    decision.ambiguous = true;
    return decision;
  }
  decision.target_root_uid = accepted.front().root_uid;
  return decision;
}

void TrackGraphSegmentUpdater::updateOrCreateObjects(
    uint64_t timestamp_ns,
    const InputData& input,
    const std::unordered_map<uint32_t, TrackObservation>& observations,
    UpdateEvidence& extracted,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  const auto& vertex_evidence_by_track = extracted.by_track;
  const auto& track_candidate_roots = extracted.candidate_roots;
  auto& root_support = extracted.root_support;
  auto& pair_evidence = extracted.pair_evidence;
  auto& root_support_vertices = extracted.root_vertices;
  auto& pair_evidence_vertices = extracted.pair_vertices;
  ScopedTimer phase_timer(phaseTimerName(config, "update_or_create_objects"),
                          timestamp_ns);

  auto sync_record_metadata = [&](TrackedObjectRecord& record) {
    if (!graph.hasNode(record.node_id)) {
      return;
    }

    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    markTrackedMetadata(attrs, record);
    backend_sync_node_ids_.insert(record.node_id);
  };

  auto touch_provisional =
      [&](uint32_t track_id,
          const TrackObservation& obs,
          const TrackVertexEvidence& evidence) -> ProvisionalTrackedObjectRecord& {
    auto& provisional = provisional_tracks_[track_id];
    if (provisional.track_id == 0u) {
      provisional.track_id = track_id;
      provisional.first_observed_ns = timestamp_ns;
    }

    provisional.last_observed_ns = timestamp_ns;
    ++provisional.frames_observed;
    if (input.tracking_is_keyframe) {
      ++provisional.keyframes_observed;
    }
    if (provisional.last_confirmed_support_size !=
            evidence.confirmed_vertex_indices.size() &&
        provisional.support_generation != std::numeric_limits<uint64_t>::max()) {
      ++provisional.support_generation;
    }
    provisional.last_confirmed_support_size = evidence.confirmed_vertex_indices.size();
    if (isValidSemanticFeature(obs.prototype) &&
        (!isValidSemanticFeature(provisional.semantic_feature) ||
         !provisional.semantic_feature.isApprox(obs.prototype))) {
      if (provisional.prototype_generation != std::numeric_limits<uint64_t>::max()) {
        ++provisional.prototype_generation;
      }
      provisional.semantic_feature = obs.prototype;
    }
    return provisional;
  };

  auto add_owned_association_evidence = [&](uint64_t root_uid,
                                            const TrackVertexEvidence& evidence) {
    if (root_uid == 0u) {
      return;
    }

    for (const auto global_idx : evidence.confirmed_association_vertex_indices) {
      addRootSupportVote(root_uid, global_idx, root_support, root_support_vertices);
    }

    for (const auto& [candidate_root_uid, vertices] :
         evidence.confirmed_candidate_root_vertex_indices) {
      if (candidate_root_uid == root_uid) {
        continue;
      }
      for (const auto global_idx : vertices) {
        addRootPairEvidenceVote(root_uid,
                                candidate_root_uid,
                                global_idx,
                                pair_evidence,
                                pair_evidence_vertices);
      }
    }
  };

  for (const auto& [track_id, obs] : observations) {
    const auto evidence_iter = vertex_evidence_by_track.find(track_id);
    const TrackVertexEvidence empty_evidence;
    const auto& evidence = evidence_iter != vertex_evidence_by_track.end()
                               ? evidence_iter->second
                               : empty_evidence;
    if (track_to_object_uid_.count(track_id)) {
      updateExistingObject(timestamp_ns,
                           track_id,
                           input.tracking_is_keyframe,
                           obs,
                           evidence,
                           offsets,
                           graph);
      continue;
    }

    auto& provisional = touch_provisional(track_id, obs, evidence);
    const auto adoption =
        config.enable_active_window_adoption
            ? decideAdoption(
                  track_id, obs, evidence, track_candidate_roots, root_support, graph)
            : AdoptionDecision{};
    if (adoption.target_root_uid != 0u &&
        adoptTrackIntoExistingRoot(timestamp_ns,
                                   track_id,
                                   input.tracking_is_keyframe,
                                   obs,
                                   evidence,
                                   adoption.target_root_uid,
                                   offsets,
                                   graph)) {
      const auto record = active_objects_.find(adoption.target_root_uid);
      if (record != active_objects_.end()) {
        add_owned_association_evidence(record->second.object_uid, evidence);
        sync_record_metadata(record->second);
      }
      provisional_tracks_.erase(track_id);
      continue;
    }

    if (evidence.global_vertex_indices.empty()) {
      continue;
    }
    const bool has_confirmed_support =
        evidence.confirmed_vertex_indices.size() >= config.min_points_create;
    const bool has_raw_support =
        config.raw_winner_admission.enabled &&
        evidence.global_vertex_indices.size() >= config.min_points_create;
    if (adoption.ambiguous || (!has_confirmed_support && !has_raw_support)) {
      continue;
    }

    createObjectForTrack(timestamp_ns,
                         track_id,
                         input.tracking_is_keyframe,
                         obs,
                         evidence,
                         offsets,
                         graph,
                         !has_confirmed_support && has_raw_support,
                         &provisional);
    const auto owner = track_to_object_uid_.find(track_id);
    if (owner != track_to_object_uid_.end()) {
      const auto record = active_objects_.find(owner->second);
      if (record != active_objects_.end()) {
        add_owned_association_evidence(record->second.object_uid, evidence);
        sync_record_metadata(record->second);
      }
    }
    provisional_tracks_.erase(track_id);
  }
}

}  // namespace hydra
