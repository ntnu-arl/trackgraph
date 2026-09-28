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
#include "hydra/frontend/active_segment_reid.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace hydra {
namespace {

size_t getAssociatedTrackCount(const TrackedObjectRecord& record) {
  if (!record.source_track_ids.empty()) {
    return record.source_track_ids.size();
  }

  return record.source_track_id == 0u ? 0u : 1u;
}

bool isProtectedRoot(const TrackedObjectRecord& record) {
  return getAssociatedTrackCount(record) > 1u || record.has_historical_association;
}

}  // namespace

void declare_config(ActiveSegmentReid::Config& config) {
  using namespace config;
  name("ActiveSegmentReid::Config");
  field(config.enabled, "enabled");
  field(config.symmetric_overlap_threshold, "symmetric_overlap_threshold");
  field(config.contained_overlap_threshold, "contained_overlap_threshold");
  field(config.contained_reverse_overlap_max, "contained_reverse_overlap_max");
  field(config.min_shared_vertices, "min_shared_vertices");
  field(config.feature_similarity_reid_threshold, "feature_similarity_reid_threshold");
  field(config.feature_similarity_partof_low, "feature_similarity_partof_low");
  field(config.min_observation_frames, "min_observation_frames");
  field(config.candidate_age_window, "candidate_age_window");
  field(config.evaluate_on_keyframes_only, "evaluate_on_keyframes_only");
  field(config.create_partof_edges, "create_partof_edges");
  field(config.partof_edge_weight_overlap_factor, "partof_edge_weight_overlap_factor");
  field(config.debug_decisions, "debug_decisions");
  field(config.max_debug_decisions_per_object, "max_debug_decisions_per_object");
  field(config.log_decisions, "log_decisions");
  field(config.log_summary, "log_summary");
  field(config.timer_namespace, "timer_namespace");
}

ActiveSegmentReid::ActiveSegmentReid(const Config& config)
    : config(config::checkValid(config)) {}

float ActiveSegmentReid::computeFeatureSimilarity(
    const ObjectNodeAttributes& attrs_a, const ObjectNodeAttributes& attrs_b) const {
  if (attrs_a.semantic_feature.size() == 0 || attrs_b.semantic_feature.size() == 0) {
    return -1.0f;
  }

  if (attrs_a.semantic_feature.size() != attrs_b.semantic_feature.size()) {
    return -1.0f;
  }

  const Eigen::Map<const Eigen::VectorXf> vec_a(attrs_a.semantic_feature.data(),
                                                attrs_a.semantic_feature.size());
  const Eigen::Map<const Eigen::VectorXf> vec_b(attrs_b.semantic_feature.data(),
                                                attrs_b.semantic_feature.size());

  const float norm_a = vec_a.norm();
  const float norm_b = vec_b.norm();
  if (norm_a < 1e-8f || norm_b < 1e-8f) {
    return -1.0f;
  }

  return vec_a.dot(vec_b) / (norm_a * norm_b);
}

ActiveSegmentReid::ReidentificationResult ActiveSegmentReid::evaluate(
    uint64_t /*timestamp_ns*/,
    const std::unordered_map<uint64_t, TrackedObjectRecord>& objects,
    const RootSupportMap& root_support,
    const RootPairEvidenceMap& pair_evidence,
    const DynamicSceneGraph& graph) const {
  ReidentificationResult result;
  result.evaluation_summary.total_objects = objects.size();
  if (!config.enabled) {
    return result;
  }

  if (pair_evidence.empty()) {
    return result;
  }

  struct CandidateObjectPair {
    uint64_t uid_new;
    uint64_t uid_existing;
    size_t shared_live = 0u;
    size_t shared_history = 0u;
    double accumulated_weight = 0.0;
    size_t order_index = 0u;
  };
  std::vector<CandidateObjectPair> ordered_candidates;

  for (const auto& [pair, evidence] : pair_evidence) {
    const size_t total_shared = evidence.shared_live + evidence.shared_history;
    if (total_shared < config.min_shared_vertices) {
      continue;
    }

    auto rec_a = objects.find(pair.root_a);
    auto rec_b = objects.find(pair.root_b);
    if (rec_a == objects.end() || rec_b == objects.end()) {
      continue;
    }

    // At least one must be a recent candidate (eligible for re-id).
    auto is_candidate = [&](const TrackedObjectRecord& r) {
      return r.is_tracked && r.frames_observed >= config.min_observation_frames &&
             r.frames_observed <
                 config.min_observation_frames + config.candidate_age_window;
    };

    // Determine which is "new" and which is "existing" based on age.
    // The newer object is always the active re-id candidate.
    const auto& rec_a_ref = rec_a->second;
    const auto& rec_b_ref = rec_b->second;
    const bool a_is_newer =
        rec_a_ref.first_observed_ns > rec_b_ref.first_observed_ns ||
        (rec_a_ref.first_observed_ns == rec_b_ref.first_observed_ns &&
         rec_a_ref.object_uid > rec_b_ref.object_uid);

    const auto& newer = a_is_newer ? rec_a_ref : rec_b_ref;
    const auto& older = a_is_newer ? rec_b_ref : rec_a_ref;

    if (!is_candidate(newer)) {
      continue;
    }

    ordered_candidates.push_back(CandidateObjectPair{newer.object_uid,
                                                     older.object_uid,
                                                     evidence.shared_live,
                                                     evidence.shared_history,
                                                     evidence.accumulated_weight});
  }
  result.evaluation_summary.proposed_pairs = ordered_candidates.size();
  std::sort(ordered_candidates.begin(),
            ordered_candidates.end(),
            [](const CandidateObjectPair& lhs, const CandidateObjectPair& rhs) {
              const size_t lhs_total = lhs.shared_live + lhs.shared_history;
              const size_t rhs_total = rhs.shared_live + rhs.shared_history;
              if (lhs_total != rhs_total) {
                return lhs_total > rhs_total;
              }
              if (lhs.accumulated_weight != rhs.accumulated_weight) {
                return lhs.accumulated_weight > rhs.accumulated_weight;
              }
              if (lhs.uid_new != rhs.uid_new) {
                return lhs.uid_new < rhs.uid_new;
              }
              return lhs.uid_existing < rhs.uid_existing;
            });
  for (size_t i = 0; i < ordered_candidates.size(); ++i) {
    ordered_candidates[i].order_index = i;
  }

  std::unordered_map<uint64_t, std::vector<CandidateObjectPair>> candidates_by_new_uid;
  std::vector<uint64_t> new_uid_order;
  for (const auto& candidate : ordered_candidates) {
    auto [iter, inserted] = candidates_by_new_uid.emplace(
        candidate.uid_new, std::vector<CandidateObjectPair>());
    if (inserted) {
      new_uid_order.push_back(candidate.uid_new);
    }
    iter->second.push_back(candidate);
  }

  std::unordered_map<uint64_t, size_t> per_object_evaluation_counts;
  std::unordered_set<uint64_t> evaluated_new_uids;

  for (const auto new_uid : new_uid_order) {
    const auto& grouped_candidates = candidates_by_new_uid.at(new_uid);
    const auto& new_record = objects.at(new_uid);

    for (const auto& candidate : grouped_candidates) {
      const auto existing_uid = candidate.uid_existing;
      const auto& existing_record = objects.at(existing_uid);

      if (!graph.hasNode(new_record.node_id) ||
          !graph.hasNode(existing_record.node_id)) {
        continue;
      }

      const auto& new_attrs =
          graph.getNode(new_record.node_id).attributes<ObjectNodeAttributes>();
      const auto& existing_attrs =
          graph.getNode(existing_record.node_id).attributes<ObjectNodeAttributes>();
      ++result.evaluation_summary.evaluated_pairs;
      per_object_evaluation_counts[new_uid]++;
      per_object_evaluation_counts[existing_uid]++;
      evaluated_new_uids.insert(new_uid);
      const size_t new_support = root_support.count(new_uid)
                                     ? root_support.at(new_uid)
                                     : new_attrs.mesh_connections.size();
      const size_t existing_support = root_support.count(existing_uid)
                                          ? root_support.at(existing_uid)
                                          : existing_attrs.mesh_connections.size();

      if (new_support == 0u || existing_support == 0u) {
        continue;
      }

      const float similarity = computeFeatureSimilarity(new_attrs, existing_attrs);

      if (similarity < 0.0f) {
        continue;
      }

      const size_t total_shared = candidate.shared_live + candidate.shared_history;

      if (total_shared < config.min_shared_vertices) {
        continue;
      }

      const double fraction_new_inside_existing =
          static_cast<double>(total_shared) / static_cast<double>(new_support);
      const double fraction_existing_inside_new =
          static_cast<double>(total_shared) / static_cast<double>(existing_support);

      const bool symmetric =
          std::min(fraction_new_inside_existing, fraction_existing_inside_new) >=
          config.symmetric_overlap_threshold;
      const bool newer_inside_existing =
          fraction_new_inside_existing >= config.contained_overlap_threshold &&
          fraction_existing_inside_new <= config.contained_reverse_overlap_max;
      const bool existing_inside_newer =
          fraction_existing_inside_new >= config.contained_overlap_threshold &&
          fraction_new_inside_existing <= config.contained_reverse_overlap_max;

      if (symmetric && similarity >= config.feature_similarity_reid_threshold) {
        result.merges[new_uid] = existing_uid;

        break;
      }

      if ((newer_inside_existing || existing_inside_newer) &&
          similarity >= config.feature_similarity_partof_low) {
        const bool keep_existing_root =
            existing_inside_newer && isProtectedRoot(existing_record);

        uint64_t parent_uid = 0u;
        uint64_t child_uid = 0u;
        if (newer_inside_existing || keep_existing_root) {
          parent_uid = existing_uid;
          child_uid = new_uid;
        } else if (existing_inside_newer) {
          parent_uid = new_uid;
          child_uid = existing_uid;
        }

        if (parent_uid == 0u || child_uid == 0u) {
          continue;
        }

        result.merges[child_uid] = parent_uid;
        if (child_uid == new_uid) {
          break;
        }
        continue;
      }
    }
  }

  result.evaluation_summary.evaluated_new_objects = evaluated_new_uids.size();
  result.evaluation_summary.involved_objects = per_object_evaluation_counts.size();
  size_t total_evaluations = 0;
  for (const auto& [_, count] : per_object_evaluation_counts) {
    total_evaluations += count;
    result.evaluation_summary.max_evaluations_for_object =
        std::max(result.evaluation_summary.max_evaluations_for_object, count);
  }

  if (result.evaluation_summary.total_objects > 0) {
    result.evaluation_summary.avg_evaluations_per_object =
        static_cast<double>(total_evaluations) /
        static_cast<double>(result.evaluation_summary.total_objects);
  }

  if (result.evaluation_summary.involved_objects > 0) {
    result.evaluation_summary.avg_evaluations_per_involved_object =
        static_cast<double>(total_evaluations) /
        static_cast<double>(result.evaluation_summary.involved_objects);
  }

  if (result.evaluation_summary.evaluated_new_objects > 0) {
    result.evaluation_summary.avg_existing_candidates_per_new_object =
        static_cast<double>(result.evaluation_summary.evaluated_pairs) /
        static_cast<double>(result.evaluation_summary.evaluated_new_objects);
  }

  return result;
}

}  // namespace hydra
