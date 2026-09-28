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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "hydra/common/dsg_types.h"
#include "hydra/frontend/tracked_object_types.h"
#include "hydra/openset/openset_types.h"

namespace hydra {

class TrackGraphLongtermReid {
 public:
  enum class OverlapMode {
    NN_QUERY_TO_CANDIDATE,
    BBOX_QUERY_IN_CANDIDATE,
  };

  struct Config {
    Config()
        : enabled(false),
          num_candidates(5),
          max_candidate_radius_m(2.0),
          min_query_keyframes(2),
          inactive_retry_window(32),
          overlap_mode(OverlapMode::NN_QUERY_TO_CANDIDATE),
          nn_search_radius_m(0.10),
          overlap_weight(0.5),
          feature_weight(0.5),
          min_overlap_fraction(0.3),
          min_feature_similarity(0.7),
          min_joint_score(0.75),
          debug_decisions(false),
          max_debug_decisions_per_object(20),
          log_jobs(false),
          log_proposals(false),
          log_candidate_evaluations(false),
          timer_namespace("frontend/historical_object_associator") {}

    bool enabled;
    size_t num_candidates;
    double max_candidate_radius_m;
    uint32_t min_query_keyframes;
    uint32_t inactive_retry_window;
    OverlapMode overlap_mode;
    double nn_search_radius_m;
    double overlap_weight;
    double feature_weight;
    double min_overlap_fraction;
    double min_feature_similarity;
    double min_joint_score;
    // Legacy detailed diagnostics are accepted but no longer collected.
    bool debug_decisions;
    size_t max_debug_decisions_per_object;
    bool log_jobs;  // Enables the per-pass summary.
    bool log_proposals;
    bool log_candidate_evaluations;
    std::string timer_namespace;
  } const config;

  struct QuerySnapshot {
    uint64_t object_uid = 0;
    NodeId node_id = 0;
    uint64_t state_generation = 0;
    uint64_t first_observed_ns = 0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    FeatureVector semantic_feature;
    std::vector<Eigen::Vector3f> support_points;
  };

  struct QueryHeader {
    uint64_t object_uid = 0;
    NodeId node_id = 0;
    uint64_t state_generation = 0;
    uint64_t first_observed_ns = 0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    FeatureVector semantic_feature;
  };

  struct CandidateSnapshot {
    uint64_t root_object_uid = 0;
    NodeId root_node_id = 0;
    uint64_t state_generation = 0;
    uint64_t first_observed_ns = 0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    spark_dsg::BoundingBox aggregate_bounding_box;
    FeatureVector semantic_feature;
    std::vector<Eigen::Vector3f> support_points;
  };

  struct CandidateHeader {
    uint64_t root_object_uid = 0;
    NodeId root_node_id = 0;
    uint64_t state_generation = 0;
    uint64_t first_observed_ns = 0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    spark_dsg::BoundingBox aggregate_bounding_box;
    FeatureVector semantic_feature;
  };

  struct ShortlistEntry {
    size_t candidate_index = 0;
    size_t candidate_rank = 0;
    double centroid_distance_m = 0.0;
  };

  struct ShortlistResult {
    std::vector<std::vector<ShortlistEntry>> candidates_by_query;
  };

  struct MergeProposal {
    uint64_t query_object_uid = 0;
    uint64_t candidate_root_uid = 0;
    uint64_t query_generation = 0;
    uint64_t candidate_generation = 0;
    double overlap_fraction = 0.0;
    float feature_similarity = 0.0f;
    double joint_score = 0.0;
    uint64_t decision_timestamp_ns = 0;
  };

  struct PairEvaluation {
    std::optional<MergeProposal> proposal;
    std::vector<std::string> rejection_reasons;
    std::optional<double> centroid_distance_m;
    std::optional<double> overlap_fraction;
    std::optional<float> feature_similarity;
    std::optional<double> joint_score;
  };

  struct Result {
    std::vector<MergeProposal> proposals;
  };

  explicit TrackGraphLongtermReid(const Config& config);
  ~TrackGraphLongtermReid();

  ShortlistResult shortlistCandidates(
      uint64_t timestamp_ns,
      const std::vector<QueryHeader>& queries,
      const std::vector<CandidateHeader>& candidates) const;

  Result evaluate(uint64_t timestamp_ns,
                  std::vector<QuerySnapshot> queries,
                  std::vector<CandidateSnapshot> candidates) const;

  PairEvaluation evaluatePair(uint64_t timestamp_ns,
                              const QuerySnapshot& query,
                              const CandidateSnapshot& candidate) const;

 private:
  Result evaluateImpl(uint64_t timestamp_ns,
                      const std::vector<QuerySnapshot>& queries,
                      const std::vector<CandidateSnapshot>& candidates) const;
};

void declare_config(TrackGraphLongtermReid::Config& config);

}  // namespace hydra
