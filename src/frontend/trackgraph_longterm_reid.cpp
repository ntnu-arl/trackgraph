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
#include "hydra/frontend/trackgraph_longterm_reid.h"

#include <config_utilities/config.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <nanoflann.hpp>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "hydra/openset/embedding_distances.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {
namespace {

using timing::ScopedTimer;

template <typename CandidateT>
struct CentroidAdaptor {
  explicit CentroidAdaptor(const std::vector<CandidateT>& candidates)
      : candidates(candidates) {}

  size_t kdtree_get_point_count() const { return candidates.size(); }

  double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return candidates[idx].centroid(dim);
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }

  const std::vector<CandidateT>& candidates;
};

struct SupportPointAdaptor {
  explicit SupportPointAdaptor(const std::vector<Eigen::Vector3f>& points)
      : points(points) {}

  size_t kdtree_get_point_count() const { return points.size(); }

  double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return static_cast<double>(points[idx](dim));
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }

  const std::vector<Eigen::Vector3f>& points;
};

using SupportDist = nanoflann::L2_Simple_Adaptor<double, SupportPointAdaptor>;
using SupportKDTree =
    nanoflann::KDTreeSingleIndexAdaptor<SupportDist, SupportPointAdaptor, 3, size_t>;

struct CandidateSupportLookup {
  explicit CandidateSupportLookup(const std::vector<Eigen::Vector3f>& points)
      : adaptor(points), tree(3, adaptor) {
    tree.buildIndex();
  }

  SupportPointAdaptor adaptor;
  SupportKDTree tree;
};

bool isValidFeature(const FeatureVector& feature) {
  return feature.size() > 0 && feature.allFinite();
}

template <typename CandidateT, typename QueryT>
bool isCandidateOlder(const CandidateT& candidate, const QueryT& query) {
  if (candidate.first_observed_ns != query.first_observed_ns) {
    return candidate.first_observed_ns < query.first_observed_ns;
  }

  return candidate.root_object_uid < query.object_uid;
}

std::optional<timing::ElapsedTimeRecorder::Entry> getLastTimingEntry(
    const std::string& timer_name) {
  return timing::ElapsedTimeRecorder::instance().getLastEntry(timer_name);
}

void appendTimingFields(
    std::ostringstream& message,
    const std::string& prefix,
    const std::optional<timing::ElapsedTimeRecorder::Entry>& entry) {
  if (!entry) {
    message << " " << prefix << "_wall_ms=n/a";
    return;
  }

  message << " " << prefix << "_wall_ms=" << entry->elapsed_milliseconds();
  if (entry->cpu_timing_enabled) {
    message << " " << prefix << "_process_cpu_ms=" << entry->process_cpu_milliseconds()
            << " " << prefix << "_thread_cpu_ms=" << entry->thread_cpu_milliseconds();
  }
}

std::string getOverlapTimerName(const TrackGraphLongtermReid::Config& config) {
  return config.timer_namespace +
         (config.overlap_mode ==
                  TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE
              ? "/bbox_overlap"
              : "/nn_overlap");
}

double computeBoundingBoxOverlapFraction(
    const TrackGraphLongtermReid::QuerySnapshot& query,
    const TrackGraphLongtermReid::CandidateSnapshot& candidate) {
  size_t inside_count = 0;
  for (const auto& point : query.support_points) {
    if (candidate.aggregate_bounding_box.contains(point)) {
      ++inside_count;
    }
  }

  return static_cast<double>(inside_count) / query.support_points.size();
}

double computeNnOverlapFraction(
    const TrackGraphLongtermReid::QuerySnapshot& query,
    const TrackGraphLongtermReid::CandidateSnapshot& candidate,
    double nn_search_radius_m,
    const CandidateSupportLookup* lookup = nullptr) {
  CandidateSupportLookup local_lookup(candidate.support_points);
  const auto& tree = lookup ? lookup->tree : local_lookup.tree;
  const double radius_sq = nn_search_radius_m * nn_search_radius_m;

  size_t matched = 0;
  for (const auto& point : query.support_points) {
    std::array<double, 3> query_point{point.x(), point.y(), point.z()};
    size_t nn_index = 0;
    double nn_distance_sq = std::numeric_limits<double>::infinity();
    const auto found =
        tree.knnSearch(query_point.data(), 1, &nn_index, &nn_distance_sq);
    if (found > 0 && nn_distance_sq <= radius_sq) {
      ++matched;
    }
  }

  return static_cast<double>(matched) / query.support_points.size();
}

TrackGraphLongtermReid::PairEvaluation evaluatePairImpl(
    const TrackGraphLongtermReid::Config& config,
    uint64_t timestamp_ns,
    const TrackGraphLongtermReid::QuerySnapshot& query,
    const TrackGraphLongtermReid::CandidateSnapshot& candidate,
    const CandidateSupportLookup* lookup = nullptr) {
  ScopedTimer pair_timer(config.timer_namespace + "/pair_evaluation", timestamp_ns);
  TrackGraphLongtermReid::PairEvaluation result;
  const auto centroid_distance_m = (query.centroid - candidate.centroid).norm();
  result.centroid_distance_m = centroid_distance_m;

  if (centroid_distance_m > config.max_candidate_radius_m) {
    result.rejection_reasons = {"outside_radius"};
    return result;
  }
  if (candidate.root_object_uid == query.object_uid) {
    result.rejection_reasons = {"self_candidate"};
    return result;
  }
  if (!isCandidateOlder(candidate, query)) {
    result.rejection_reasons = {"candidate_not_older"};
    return result;
  }

  const bool invalid_geometry =
      (config.overlap_mode ==
       TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE)
          ? !candidate.aggregate_bounding_box.isValid()
          : candidate.support_points.empty();
  if (invalid_geometry || !isValidFeature(candidate.semantic_feature) ||
      candidate.semantic_feature.size() != query.semantic_feature.size()) {
    result.rejection_reasons = {"invalid_candidate_bbox_or_feature"};
    return result;
  }

  const auto overlap_timer_name = getOverlapTimerName(config);
  double overlap_fraction = 0.0;
  {
    ScopedTimer overlap_timer(overlap_timer_name, timestamp_ns);
    overlap_fraction =
        config.overlap_mode ==
                TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE
            ? computeBoundingBoxOverlapFraction(query, candidate)
            : computeNnOverlapFraction(
                  query, candidate, config.nn_search_radius_m, lookup);
  }
  result.overlap_fraction = overlap_fraction;
  if (overlap_fraction < config.min_overlap_fraction) {
    result.rejection_reasons = {"overlap_below_threshold"};
    return result;
  }

  const CosineDistance cosine;
  const float feature_similarity =
      cosine.score(query.semantic_feature, candidate.semantic_feature);
  result.feature_similarity = feature_similarity;
  if (!std::isfinite(feature_similarity) ||
      feature_similarity < config.min_feature_similarity) {
    result.rejection_reasons = {"feature_similarity_below_threshold"};
    return result;
  }

  const double joint_score = config.overlap_weight * overlap_fraction +
                             config.feature_weight * feature_similarity;
  result.joint_score = joint_score;
  if (joint_score < config.min_joint_score) {
    result.rejection_reasons = {"joint_score_below_threshold"};
    return result;
  }

  TrackGraphLongtermReid::MergeProposal proposal;
  proposal.query_object_uid = query.object_uid;
  proposal.candidate_root_uid = candidate.root_object_uid;
  proposal.query_generation = query.state_generation;
  proposal.candidate_generation = candidate.state_generation;
  proposal.overlap_fraction = overlap_fraction;
  proposal.feature_similarity = feature_similarity;
  proposal.joint_score = joint_score;
  proposal.decision_timestamp_ns = timestamp_ns;
  result.proposal = proposal;
  return result;
}

}  // namespace

void declare_config(TrackGraphLongtermReid::Config& config) {
  using namespace config;
  name("TrackGraphLongtermReid::Config");
  field(config.enabled, "enabled");
  field(config.num_candidates, "num_candidates");
  field(config.max_candidate_radius_m, "max_candidate_radius_m");
  field(config.min_query_keyframes, "min_query_keyframes");
  field(config.inactive_retry_window, "inactive_retry_window");
  enum_field(config.overlap_mode,
             "overlap_mode",
             {{TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE,
               "nn_query_to_candidate"},
              {TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE,
               "bbox_query_in_candidate"}});
  field(config.nn_search_radius_m, "nn_search_radius_m");
  field(config.overlap_weight, "overlap_weight");
  field(config.feature_weight, "feature_weight");
  field(config.min_overlap_fraction, "min_overlap_fraction");
  field(config.min_feature_similarity, "min_feature_similarity");
  field(config.min_joint_score, "min_joint_score");
  field(config.debug_decisions, "debug_decisions");
  field(config.max_debug_decisions_per_object, "max_debug_decisions_per_object");
  field(config.log_jobs, "log_jobs");
  field(config.log_proposals, "log_proposals");
  field(config.log_candidate_evaluations, "log_candidate_evaluations");
  field(config.timer_namespace, "timer_namespace");

  check(config.num_candidates, GT, 0u, "num_candidates");
  check(config.max_candidate_radius_m, GT, 0.0, "max_candidate_radius_m");
  check(config.nn_search_radius_m, GT, 0.0, "nn_search_radius_m");
  check(config.overlap_weight, GE, 0.0, "overlap_weight");
  check(config.feature_weight, GE, 0.0, "feature_weight");
  check(config.min_overlap_fraction, GE, 0.0, "min_overlap_fraction");
  check(config.min_overlap_fraction, LE, 1.0, "min_overlap_fraction");
  check(config.min_feature_similarity, GE, -1.0, "min_feature_similarity");
  check(config.min_feature_similarity, LE, 1.0, "min_feature_similarity");
}

TrackGraphLongtermReid::TrackGraphLongtermReid(const Config& config)
    : config(config::checkValid(config)) {}

TrackGraphLongtermReid::~TrackGraphLongtermReid() = default;

TrackGraphLongtermReid::ShortlistResult TrackGraphLongtermReid::shortlistCandidates(
    uint64_t timestamp_ns,
    const std::vector<QueryHeader>& queries,
    const std::vector<CandidateHeader>& candidates) const {
  const auto shortlist_timer_name = config.timer_namespace + "/shortlist";
  const auto build_kdtree_timer_name = config.timer_namespace + "/build_kdtree";
  const auto query_timer_name = config.timer_namespace + "/query_evaluation";
  const auto knn_timer_name = config.timer_namespace + "/knn_search";
  ScopedTimer shortlist_timer(shortlist_timer_name, timestamp_ns);

  ShortlistResult result;
  result.candidates_by_query.resize(queries.size());
  if (queries.empty() || candidates.empty()) {
    return result;
  }

  using Dist = nanoflann::L2_Simple_Adaptor<double, CentroidAdaptor<CandidateHeader>>;
  using KDTree = nanoflann::
      KDTreeSingleIndexAdaptor<Dist, CentroidAdaptor<CandidateHeader>, 3, size_t>;

  const CentroidAdaptor<CandidateHeader> adaptor(candidates);
  KDTree tree(3, adaptor);
  {
    ScopedTimer build_timer(build_kdtree_timer_name, timestamp_ns);
    tree.buildIndex();
  }

  for (size_t query_index = 0; query_index < queries.size(); ++query_index) {
    ScopedTimer query_timer(query_timer_name, timestamp_ns);
    const auto& query = queries[query_index];
    auto& shortlisted = result.candidates_by_query[query_index];

    if (!isValidFeature(query.semantic_feature)) {
      continue;
    }

    const size_t knn_limit = std::min(config.num_candidates, candidates.size());
    std::vector<size_t> candidate_indices(knn_limit);
    std::vector<double> candidate_distances(knn_limit);
    size_t num_found = 0;
    {
      ScopedTimer knn_timer(knn_timer_name, timestamp_ns);
      num_found = tree.knnSearch(query.centroid.data(),
                                 knn_limit,
                                 candidate_indices.data(),
                                 candidate_distances.data());
      knn_timer.stop();
    }
    if (num_found == 0) {
      continue;
    }

    for (size_t i = 0; i < num_found; ++i) {
      const auto candidate_index = candidate_indices[i];
      const auto& candidate = candidates[candidate_index];
      const auto centroid_distance_m = std::sqrt(candidate_distances[i]);
      if (centroid_distance_m > config.max_candidate_radius_m) {
        continue;
      }
      if (candidate.root_object_uid == query.object_uid) {
        continue;
      }
      if (!isCandidateOlder(candidate, query)) {
        continue;
      }
      if (!candidate.aggregate_bounding_box.isValid() ||
          !isValidFeature(candidate.semantic_feature) ||
          candidate.semantic_feature.size() != query.semantic_feature.size()) {
        continue;
      }

      shortlisted.push_back({candidate_index, i, centroid_distance_m});
    }
  }

  return result;
}

TrackGraphLongtermReid::Result TrackGraphLongtermReid::evaluate(
    uint64_t timestamp_ns,
    std::vector<QuerySnapshot> queries,
    std::vector<CandidateSnapshot> candidates) const {
  if (!config.enabled) {
    return {};
  }

  LOG_IF(INFO, config.log_jobs)
      << "[historical-reid] evaluate: ts=" << timestamp_ns
      << " queries=" << queries.size() << " candidates=" << candidates.size();
  return evaluateImpl(timestamp_ns, queries, candidates);
}

TrackGraphLongtermReid::Result TrackGraphLongtermReid::evaluateImpl(
    uint64_t timestamp_ns,
    const std::vector<QuerySnapshot>& queries,
    const std::vector<CandidateSnapshot>& candidates) const {
  const auto evaluate_timer_name = config.timer_namespace + "/evaluate";
  const auto build_kdtree_timer_name = config.timer_namespace + "/build_kdtree";
  const auto query_timer_name = config.timer_namespace + "/query_evaluation";
  const auto knn_timer_name = config.timer_namespace + "/knn_search";
  ScopedTimer evaluate_timer(evaluate_timer_name, timestamp_ns);

  Result result;
  auto& results = result.proposals;
  auto log_job_state = [&](const std::string& event_name) {
    evaluate_timer.stop();
    if (!config.log_jobs) {
      return;
    }

    std::ostringstream msg;
    msg << "[historical-reid] " << event_name << ": ts=" << timestamp_ns
        << " queries=" << queries.size() << " candidates=" << candidates.size()
        << " proposals=" << results.size();
    appendTimingFields(msg, "evaluate", getLastTimingEntry(evaluate_timer_name));
    LOG(INFO) << msg.str();
  };

  if (queries.empty() || candidates.empty()) {
    log_job_state("job_skipped");
    return result;
  }

  using Dist = nanoflann::L2_Simple_Adaptor<double, CentroidAdaptor<CandidateSnapshot>>;
  using KDTree = nanoflann::
      KDTreeSingleIndexAdaptor<Dist, CentroidAdaptor<CandidateSnapshot>, 3, size_t>;

  const CentroidAdaptor<CandidateSnapshot> adaptor(candidates);
  KDTree tree(3, adaptor);
  {
    ScopedTimer build_timer(build_kdtree_timer_name, timestamp_ns);
    tree.buildIndex();
  }

  for (const auto& query : queries) {
    ScopedTimer query_timer(query_timer_name, timestamp_ns);
    if (query.support_points.empty() || !isValidFeature(query.semantic_feature)) {
      continue;
    }

    const size_t knn_limit = std::min(config.num_candidates, candidates.size());
    std::vector<size_t> candidate_indices(knn_limit);
    std::vector<double> candidate_distances(knn_limit);
    size_t num_found = 0;
    {
      ScopedTimer knn_timer(knn_timer_name, timestamp_ns);
      num_found = tree.knnSearch(query.centroid.data(),
                                 knn_limit,
                                 candidate_indices.data(),
                                 candidate_distances.data());
      knn_timer.stop();
    }
    if (num_found == 0) {
      continue;
    }

    std::unordered_map<size_t, std::unique_ptr<CandidateSupportLookup>>
        support_lookup_cache;
    std::optional<MergeProposal> best;
    for (size_t i = 0; i < num_found; ++i) {
      const auto candidate_index = candidate_indices[i];
      const auto& candidate = candidates[candidate_index];

      const CandidateSupportLookup* lookup = nullptr;
      if (config.overlap_mode == OverlapMode::NN_QUERY_TO_CANDIDATE &&
          !candidate.support_points.empty()) {
        auto& cached_lookup = support_lookup_cache[candidate_index];
        if (!cached_lookup) {
          cached_lookup =
              std::make_unique<CandidateSupportLookup>(candidate.support_points);
        }
        lookup = cached_lookup.get();
      }

      const auto evaluation =
          evaluatePairImpl(config, timestamp_ns, query, candidate, lookup);
      if (!evaluation.proposal) {
        continue;
      }

      const auto& proposal = *evaluation.proposal;
      if (!best || proposal.joint_score > best->joint_score ||
          (proposal.joint_score == best->joint_score &&
           proposal.overlap_fraction > best->overlap_fraction)) {
        best = proposal;
      }
    }

    if (best) {
      results.push_back(*best);
    }
  }

  log_job_state("job_finished");
  return result;
}

TrackGraphLongtermReid::PairEvaluation TrackGraphLongtermReid::evaluatePair(
    uint64_t timestamp_ns,
    const QuerySnapshot& query,
    const CandidateSnapshot& candidate) const {
  return evaluatePairImpl(config, timestamp_ns, query, candidate);
}

}  // namespace hydra
