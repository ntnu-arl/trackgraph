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
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hydra/common/dsg_types.h"
#include "hydra/frontend/tracked_object_types.h"

namespace hydra {

class ActiveSegmentReid {
 public:
  struct Config {
    bool enabled = true;

    // Overlap thresholds on extended re-identification support.
    double symmetric_overlap_threshold = 0.6;
    double contained_overlap_threshold = 0.7;
    double contained_reverse_overlap_max = 0.4;
    size_t min_shared_vertices = 10;

    // Feature confirmation thresholds.
    double feature_similarity_reid_threshold = 0.85;
    double feature_similarity_partof_low = 0.3;

    // Evidence accumulation.
    size_t min_observation_frames = 3;
    size_t candidate_age_window = 10;
    bool evaluate_on_keyframes_only = false;
    bool create_partof_edges = false;
    double partof_edge_weight_overlap_factor = 0.5;
    // Legacy diagnostic keys are accepted for archived YAML compatibility.
    bool debug_decisions = false;
    size_t max_debug_decisions_per_object = 20;
    bool log_decisions = false;
    bool log_summary = false;

    std::string timer_namespace = "frontend/object_reidentification";
  } const config;

  struct ReidentificationResult {
    struct EvaluationSummary {
      size_t total_objects = 0;
      size_t proposed_pairs = 0;
      size_t evaluated_pairs = 0;
      size_t evaluated_new_objects = 0;
      size_t involved_objects = 0;
      size_t max_evaluations_for_object = 0;
      double avg_evaluations_per_object = 0.0;
      double avg_evaluations_per_involved_object = 0.0;
      double avg_existing_candidates_per_new_object = 0.0;
    };

    // duplicate object uid -> surviving object uid.
    std::unordered_map<uint64_t, uint64_t> merges;
    // Per-pass candidate/evaluation summary for logging.
    EvaluationSummary evaluation_summary;
  };

  explicit ActiveSegmentReid(const Config& config);

  ReidentificationResult evaluate(
      uint64_t timestamp_ns,
      const std::unordered_map<uint64_t, TrackedObjectRecord>& objects,
      const RootSupportMap& root_support,
      const RootPairEvidenceMap& pair_evidence,
      const DynamicSceneGraph& graph) const;

 private:
  float computeFeatureSimilarity(const ObjectNodeAttributes& attrs_a,
                                 const ObjectNodeAttributes& attrs_b) const;
};

void declare_config(ActiveSegmentReid::Config& config);

}  // namespace hydra
