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

#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

namespace {
bool historicalProposalPreferred(const TrackGraphLongtermReid::MergeProposal& lhs,
                                 const TrackGraphLongtermReid::MergeProposal& rhs) {
  if (lhs.joint_score != rhs.joint_score) {
    return lhs.joint_score > rhs.joint_score;
  }
  if (lhs.overlap_fraction != rhs.overlap_fraction) {
    return lhs.overlap_fraction > rhs.overlap_fraction;
  }
  return lhs.query_object_uid < rhs.query_object_uid;
}

std::vector<Eigen::Vector3f> extractHistoricalSupportPoints(
    const spark_dsg::Mesh& mesh, const ObjectNodeAttributes& attrs) {
  std::vector<Eigen::Vector3f> support_points;
  support_points.reserve(attrs.mesh_connections.size());
  for (const auto idx : attrs.mesh_connections) {
    if (idx < mesh.numVertices()) {
      support_points.push_back(mesh.pos(idx));
    }
  }
  return support_points;
}

TrackGraphLongtermReid::QueryHeader makeHistoricalQueryHeader(
    const TrackedObjectRecord& record, const ObjectNodeAttributes& attrs) {
  TrackGraphLongtermReid::QueryHeader query;
  query.object_uid = record.object_uid;
  query.node_id = record.node_id;
  query.state_generation = record.state_generation;
  query.first_observed_ns = record.first_observed_ns;
  query.centroid = attrs.bounding_box.world_P_center.cast<double>();
  query.semantic_feature = attrs.semantic_feature;
  return query;
}

TrackGraphLongtermReid::CandidateHeader makeHistoricalCandidateHeader(
    uint64_t object_uid,
    NodeId node_id,
    uint64_t state_generation,
    uint64_t first_observed_ns,
    const ObjectNodeAttributes& attrs) {
  TrackGraphLongtermReid::CandidateHeader candidate;
  candidate.root_object_uid = object_uid;
  candidate.root_node_id = node_id;
  candidate.state_generation = state_generation;
  candidate.first_observed_ns = first_observed_ns;
  candidate.centroid = attrs.bounding_box.world_P_center.cast<double>();
  candidate.aggregate_bounding_box = attrs.bounding_box;
  candidate.semantic_feature = attrs.semantic_feature;
  return candidate;
}

TrackGraphLongtermReid::QuerySnapshot makeHistoricalQuerySnapshot(
    const TrackGraphLongtermReid::QueryHeader& header,
    std::vector<Eigen::Vector3f> support_points) {
  TrackGraphLongtermReid::QuerySnapshot query;
  query.object_uid = header.object_uid;
  query.node_id = header.node_id;
  query.state_generation = header.state_generation;
  query.first_observed_ns = header.first_observed_ns;
  query.centroid = header.centroid;
  query.semantic_feature = header.semantic_feature;
  query.support_points = std::move(support_points);
  return query;
}

TrackGraphLongtermReid::CandidateSnapshot makeHistoricalCandidateSnapshot(
    const TrackGraphLongtermReid::CandidateHeader& header,
    std::vector<Eigen::Vector3f> support_points = {}) {
  TrackGraphLongtermReid::CandidateSnapshot candidate;
  candidate.root_object_uid = header.root_object_uid;
  candidate.root_node_id = header.root_node_id;
  candidate.state_generation = header.state_generation;
  candidate.first_observed_ns = header.first_observed_ns;
  candidate.centroid = header.centroid;
  candidate.aggregate_bounding_box = header.aggregate_bounding_box;
  candidate.semantic_feature = header.semantic_feature;
  candidate.support_points = std::move(support_points);
  return candidate;
}

struct HistoricalApplyPairState {
  TrackGraphLongtermReid::QuerySnapshot query;
  TrackGraphLongtermReid::CandidateSnapshot candidate;
};

std::optional<std::string> buildHistoricalApplyPairState(
    const DynamicSceneGraph& graph,
    const spark_dsg::Mesh& mesh,
    const std::unordered_map<uint64_t, TrackedObjectRecord>& active_objects,
    const std::unordered_map<uint64_t, HistoricalTrackedObjectRecord>&
        frozen_candidates,
    const TrackGraphSegmentUpdater::Config& config,
    const TrackGraphLongtermReid::MergeProposal& proposal,
    bool require_historical_candidate,
    HistoricalApplyPairState& result) {
  const auto query_iter = active_objects.find(proposal.query_object_uid);
  if (query_iter == active_objects.end()) {
    return "query_missing";
  }

  const auto active_candidate_iter = active_objects.find(proposal.candidate_root_uid);
  const auto frozen_candidate_iter =
      frozen_candidates.find(proposal.candidate_root_uid);
  const bool candidate_is_active = active_candidate_iter != active_objects.end();
  const bool candidate_is_frozen = frozen_candidate_iter != frozen_candidates.end();
  if (!candidate_is_active && !candidate_is_frozen) {
    return "candidate_missing";
  }

  const auto& query_record = query_iter->second;
  const auto candidate_node_id = candidate_is_active
                                     ? active_candidate_iter->second.node_id
                                     : frozen_candidate_iter->second.node_id;
  if (!graph.hasNode(query_record.node_id) || !graph.hasNode(candidate_node_id)) {
    return "query_or_candidate_node_missing";
  }

  const auto& query_attrs =
      graph.getNode(query_record.node_id).attributes<ObjectNodeAttributes>();
  const auto& candidate_attrs =
      graph.getNode(candidate_node_id).attributes<ObjectNodeAttributes>();
  if (!query_record.is_tracked) {
    return "query_not_tracked";
  }
  if (query_attrs.mesh_connections.empty() || !query_attrs.bounding_box.isValid() ||
      !isValidSemanticFeature(query_attrs.semantic_feature) ||
      !candidate_attrs.bounding_box.isValid() ||
      !isValidSemanticFeature(candidate_attrs.semantic_feature)) {
    return "query_or_candidate_invalid";
  }

  if (candidate_is_active) {
    const bool candidate_historical_override =
        active_candidate_iter->second.historical_revived_candidate_override;
    if (require_historical_candidate && !candidate_historical_override) {
      return "candidate_is_active";
    }
  }

  result.query =
      makeHistoricalQuerySnapshot(makeHistoricalQueryHeader(query_record, query_attrs),
                                  extractHistoricalSupportPoints(mesh, query_attrs));
  if (result.query.support_points.empty()) {
    return "query_support_empty";
  }

  result.candidate = makeHistoricalCandidateSnapshot(makeHistoricalCandidateHeader(
      proposal.candidate_root_uid,
      candidate_node_id,
      candidate_is_active ? active_candidate_iter->second.state_generation
                          : frozen_candidate_iter->second.state_generation,
      candidate_is_active ? active_candidate_iter->second.first_observed_ns
                          : frozen_candidate_iter->second.first_observed_ns,
      candidate_attrs));
  if (config.historical_associator.overlap_mode ==
      TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE) {
    result.candidate.support_points =
        extractHistoricalSupportPoints(mesh, candidate_attrs);
    if (result.candidate.support_points.empty()) {
      return "candidate_support_empty";
    }
  }

  return std::nullopt;
}

struct HistoricalProposalBucket {
  uint64_t candidate_root_uid = 0;
  double best_initial_joint_score = 0.0;
  std::vector<TrackGraphLongtermReid::MergeProposal> proposals;
};

}  // namespace

void TrackGraphSegmentUpdater::applyLongtermReidResults(
    const std::vector<TrackGraphLongtermReid::MergeProposal>& proposals,
    uint64_t timestamp_ns,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  if (!historical_associator_ || proposals.empty() || !graph.hasMesh()) {
    return;
  }

  std::unordered_set<uint64_t> merged_query_uids;
  std::unordered_map<uint64_t, size_t> bucket_indices;
  std::vector<HistoricalProposalBucket> buckets;
  for (const auto& proposal : proposals) {
    const auto [iter, inserted] =
        bucket_indices.emplace(proposal.candidate_root_uid, buckets.size());
    if (inserted) {
      buckets.push_back(HistoricalProposalBucket{
          proposal.candidate_root_uid, proposal.joint_score, {}});
    }

    auto& bucket = buckets[iter->second];
    bucket.best_initial_joint_score =
        std::max(bucket.best_initial_joint_score, proposal.joint_score);
    bucket.proposals.push_back(proposal);
  }

  std::sort(
      buckets.begin(),
      buckets.end(),
      [](const HistoricalProposalBucket& lhs, const HistoricalProposalBucket& rhs) {
        if (lhs.best_initial_joint_score != rhs.best_initial_joint_score) {
          return lhs.best_initial_joint_score > rhs.best_initial_joint_score;
        }
        return lhs.candidate_root_uid < rhs.candidate_root_uid;
      });

  for (auto& bucket : buckets) {
    std::sort(
        bucket.proposals.begin(), bucket.proposals.end(), historicalProposalPreferred);
    bool bucket_has_accepted_merge = false;

    while (!bucket.proposals.empty()) {
      const auto proposal = bucket.proposals.front();
      bucket.proposals.erase(bucket.proposals.begin());
      if (merged_query_uids.count(proposal.query_object_uid)) {
        continue;
      }

      HistoricalApplyPairState pair_state;
      const auto precheck_reason =
          buildHistoricalApplyPairState(graph,
                                        *graph.mesh(),
                                        active_objects_,
                                        frozen_candidates_,
                                        config,
                                        proposal,
                                        !bucket_has_accepted_merge,
                                        pair_state);
      if (precheck_reason) {
        continue;
      }

      const auto refreshed = historical_associator_->evaluatePair(
          proposal.decision_timestamp_ns, pair_state.query, pair_state.candidate);
      if (!refreshed.proposal) {
        continue;
      }

      const auto& refreshed_proposal = *refreshed.proposal;
      const auto query_iter = active_objects_.find(refreshed_proposal.query_object_uid);
      if (query_iter == active_objects_.end()) {
        continue;
      }

      if (!mergeTrackedObjectIntoExisting(refreshed_proposal.candidate_root_uid,
                                          refreshed_proposal.query_object_uid,
                                          timestamp_ns,
                                          offsets,
                                          graph,
                                          true)) {
        continue;
      }

      merged_query_uids.insert(refreshed_proposal.query_object_uid);
      bucket_has_accepted_merge = true;
      auto existing_iter = active_objects_.find(refreshed_proposal.candidate_root_uid);
      if (existing_iter != active_objects_.end() &&
          graph.hasNode(existing_iter->second.node_id)) {
        existing_iter->second.has_historical_association = true;
        auto& attrs = graph.getNode(existing_iter->second.node_id)
                          .attributes<ObjectNodeAttributes>();
        markTrackedMetadata(attrs, existing_iter->second);
      }

      std::vector<TrackGraphLongtermReid::MergeProposal> remaining;
      for (const auto& remaining_proposal : bucket.proposals) {
        if (merged_query_uids.count(remaining_proposal.query_object_uid)) {
          continue;
        }

        HistoricalApplyPairState remaining_pair_state;
        const auto remaining_precheck_reason =
            buildHistoricalApplyPairState(graph,
                                          *graph.mesh(),
                                          active_objects_,
                                          frozen_candidates_,
                                          config,
                                          remaining_proposal,
                                          false,
                                          remaining_pair_state);
        if (remaining_precheck_reason) {
          continue;
        }

        const auto reevaluated = historical_associator_->evaluatePair(
            remaining_proposal.decision_timestamp_ns,
            remaining_pair_state.query,
            remaining_pair_state.candidate);
        if (!reevaluated.proposal) {
          continue;
        }

        remaining.push_back(*reevaluated.proposal);
      }

      bucket.proposals = std::move(remaining);
      std::sort(bucket.proposals.begin(),
                bucket.proposals.end(),
                historicalProposalPreferred);
    }
  }
}

void TrackGraphSegmentUpdater::associateLongtermSegments(
    uint64_t timestamp_ns,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    DynamicSceneGraph& graph) {
  if (!historical_associator_ || !graph.hasMesh()) {
    return;
  }

  const auto header_timer_name =
      config.historical_associator.timer_namespace + "/header_build";
  const auto query_support_timer_name =
      config.historical_associator.timer_namespace + "/query_support_materialization";
  const auto candidate_support_timer_name =
      config.historical_associator.timer_namespace +
      "/candidate_support_materialization";

  std::vector<TrackGraphLongtermReid::QueryHeader> query_headers;
  std::vector<TrackGraphLongtermReid::CandidateHeader> candidate_headers;
  {
    ScopedTimer header_timer(header_timer_name, timestamp_ns);
    auto append_candidate_header = [&](uint64_t object_uid,
                                       NodeId node_id,
                                       uint64_t state_generation,
                                       uint64_t first_observed_ns) {
      if (!graph.hasNode(node_id)) {
        return;
      }

      const auto& attrs = graph.getNode(node_id).attributes<ObjectNodeAttributes>();
      if (!attrs.bounding_box.isValid() ||
          !isValidSemanticFeature(attrs.semantic_feature)) {
        return;
      }

      candidate_headers.push_back(makeHistoricalCandidateHeader(
          object_uid, node_id, state_generation, first_observed_ns, attrs));
    };
    for (const auto& [object_uid, record] : active_objects_) {
      (void)object_uid;
      if (!graph.hasNode(record.node_id)) {
        continue;
      }

      const auto& attrs =
          graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
      const auto associated_track_ids = getAssociatedTrackIds(record);
      const bool eligible_active_candidate =
          record.historical_revived_candidate_override ||
          associated_track_ids.size() > 1;
      const bool unresolved = associated_track_ids.size() <= 1;
      const bool eligible_tracked_query =
          unresolved && record.is_tracked &&
          record.keyframes_observed >=
              config.historical_associator.min_query_keyframes &&
          !attrs.mesh_connections.empty() &&
          isValidSemanticFeature(attrs.semantic_feature);
      if (eligible_tracked_query) {
        query_headers.push_back(makeHistoricalQueryHeader(record, attrs));
      }

      if (eligible_active_candidate) {
        append_candidate_header(record.object_uid,
                                record.node_id,
                                record.state_generation,
                                record.first_observed_ns);
      }
    }

    for (const auto& [object_uid, record] : frozen_candidates_) {
      (void)object_uid;
      append_candidate_header(record.object_uid,
                              record.node_id,
                              record.state_generation,
                              record.first_observed_ns);
    }
  }

  auto shortlist_result = historical_associator_->shortlistCandidates(
      timestamp_ns, query_headers, candidate_headers);
  std::vector<TrackGraphLongtermReid::MergeProposal> proposals;
  proposals.reserve(query_headers.size());

  size_t shortlist_pairs = 0;
  std::unordered_map<size_t, TrackGraphLongtermReid::CandidateSnapshot>
      candidate_snapshot_cache;

  auto materialize_query =
      [&](size_t query_index) -> std::optional<TrackGraphLongtermReid::QuerySnapshot> {
    const auto& query_header = query_headers.at(query_index);
    if (!graph.hasNode(query_header.node_id)) {
      return std::nullopt;
    }

    const auto& attrs =
        graph.getNode(query_header.node_id).attributes<ObjectNodeAttributes>();
    std::vector<Eigen::Vector3f> support_points;
    {
      ScopedTimer support_timer(query_support_timer_name, timestamp_ns);
      support_points = extractHistoricalSupportPoints(*graph.mesh(), attrs);
    }
    if (support_points.empty()) {
      return std::nullopt;
    }

    return makeHistoricalQuerySnapshot(query_header, std::move(support_points));
  };
  auto get_candidate_snapshot =
      [&](size_t candidate_index) -> const TrackGraphLongtermReid::CandidateSnapshot* {
    const auto& candidate_header = candidate_headers.at(candidate_index);
    if (config.historical_associator.overlap_mode !=
        TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE) {
      return nullptr;
    }

    auto iter = candidate_snapshot_cache.find(candidate_index);
    if (iter != candidate_snapshot_cache.end()) {
      return &iter->second;
    }

    if (!graph.hasNode(candidate_header.root_node_id)) {
      return nullptr;
    }

    const auto& attrs =
        graph.getNode(candidate_header.root_node_id).attributes<ObjectNodeAttributes>();
    std::vector<Eigen::Vector3f> support_points;
    {
      ScopedTimer support_timer(candidate_support_timer_name, timestamp_ns);
      support_points = extractHistoricalSupportPoints(*graph.mesh(), attrs);
    }
    auto [inserted_iter, inserted] = candidate_snapshot_cache.emplace(
        candidate_index,
        makeHistoricalCandidateSnapshot(candidate_header, std::move(support_points)));
    (void)inserted;
    return &inserted_iter->second;
  };

  for (size_t query_index = 0; query_index < query_headers.size(); ++query_index) {
    const auto& shortlisted = shortlist_result.candidates_by_query.at(query_index);
    shortlist_pairs += shortlisted.size();
    if (shortlisted.empty()) {
      continue;
    }

    auto query = materialize_query(query_index);
    if (!query) {
      continue;
    }

    std::optional<TrackGraphLongtermReid::MergeProposal> best;
    for (const auto& shortlist_entry : shortlisted) {
      const auto& candidate_header =
          candidate_headers.at(shortlist_entry.candidate_index);
      TrackGraphLongtermReid::CandidateSnapshot candidate =
          makeHistoricalCandidateSnapshot(candidate_header);
      if (config.historical_associator.overlap_mode ==
          TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE) {
        const auto* cached_candidate =
            get_candidate_snapshot(shortlist_entry.candidate_index);
        if (!cached_candidate) {
          continue;
        }
        candidate = *cached_candidate;
      }

      const auto evaluation =
          historical_associator_->evaluatePair(timestamp_ns, *query, candidate);
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
      proposals.push_back(*best);
    }
  }

  LOG_IF(INFO, config.historical_associator.log_jobs)
      << "[historical-reid] evaluating_in_pass: ts=" << timestamp_ns
      << " queries=" << query_headers.size()
      << " candidates=" << candidate_headers.size()
      << " shortlist_pairs=" << shortlist_pairs << " proposals=" << proposals.size();
  applyLongtermReidResults(proposals, timestamp_ns, offsets, graph);
}

}  // namespace hydra
