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
#include "segment_support.h"

#include <kimera_pgmo/mesh_delta.h>

#include <cmath>
#include <limits>

namespace hydra {
using namespace trackgraph;

namespace trackgraph {
bool addRootSupportVote(
    uint64_t root_uid,
    size_t global_vertex_idx,
    RootSupportMap& root_support,
    TrackGraphSegmentUpdater::RootVertexSetMap& root_support_vertices) {
  if (root_uid == 0u) {
    return false;
  }

  if (!root_support_vertices[root_uid].insert(global_vertex_idx).second) {
    return false;
  }

  ++root_support[root_uid];
  return true;
}

bool addRootPairEvidenceVote(
    uint64_t root_a,
    uint64_t root_b,
    size_t global_vertex_idx,
    RootPairEvidenceMap& pair_evidence,
    TrackGraphSegmentUpdater::RootPairVertexSetMap& pair_evidence_vertices) {
  if (root_a == 0u || root_b == 0u || root_a == root_b) {
    return false;
  }

  const auto pair = RootPair::make(root_a, root_b);
  if (!pair_evidence_vertices[pair].insert(global_vertex_idx).second) {
    return false;
  }

  auto& evidence = pair_evidence[pair];
  ++evidence.shared_history;
  evidence.accumulated_weight += 1.0;
  return true;
}

std::vector<uint32_t> getAssociatedTrackIds(const TrackedObjectRecord& record) {
  if (!record.source_track_ids.empty()) {
    return record.source_track_ids;
  }

  if (record.source_track_id == 0u) {
    return {};
  }

  return {record.source_track_id};
}

HistoricalTrackedObjectRecord makeHistoricalRecord(const TrackedObjectRecord& record) {
  HistoricalTrackedObjectRecord historical;
  historical.object_uid = record.object_uid;
  historical.node_id = record.node_id;
  historical.source_track_ids = getAssociatedTrackIds(record);
  historical.source_track_colors = record.source_track_colors;
  historical.first_observed_ns = record.first_observed_ns;
  historical.last_observed_ns = record.last_observed_ns;
  historical.num_track_updates = record.num_track_updates;
  historical.frames_observed = record.frames_observed;
  historical.keyframes_observed = record.keyframes_observed;
  historical.untracked_updates = record.untracked_updates;
  historical.state_generation = record.state_generation;
  historical.support_generation = record.support_generation;
  historical.prototype_generation = record.prototype_generation;
  historical.is_tracked = record.is_tracked;
  historical.historical_revived_candidate_override =
      record.historical_revived_candidate_override;
  historical.open_vocab_feature_source_track_ids =
      record.open_vocab_feature_source_track_ids;
  historical.open_vocab_encoder_id = record.open_vocab_encoder_id;
  historical.open_vocab_ignore_base_state = record.open_vocab_ignore_base_state;
  historical.open_vocab_ignore_effective = record.open_vocab_ignore_effective;
  historical.open_vocab_ignore_votes = record.open_vocab_ignore_votes;
  historical.open_vocab_ignore_best_prompt = record.open_vocab_ignore_best_prompt;
  historical.open_vocab_ignore_best_score = record.open_vocab_ignore_best_score;
  historical.has_historical_association = record.has_historical_association;
  return historical;
}

TrackedObjectRecord makeActiveRecord(const HistoricalTrackedObjectRecord& record) {
  TrackedObjectRecord active;
  active.object_uid = record.object_uid;
  active.node_id = record.node_id;
  active.source_track_ids = record.source_track_ids;
  active.source_track_id =
      record.source_track_ids.empty() ? 0u : record.source_track_ids.front();
  active.source_track_colors = record.source_track_colors;
  active.first_observed_ns = record.first_observed_ns;
  active.last_observed_ns = record.last_observed_ns;
  active.num_track_updates = record.num_track_updates;
  active.frames_observed = record.frames_observed;
  active.keyframes_observed = record.keyframes_observed;
  active.untracked_updates = record.untracked_updates;
  active.state_generation = record.state_generation;
  active.support_generation = record.support_generation;
  active.prototype_generation = record.prototype_generation;
  active.is_tracked = record.is_tracked;
  active.historical_revived_candidate_override =
      record.historical_revived_candidate_override;
  active.open_vocab_feature_source_track_ids =
      record.open_vocab_feature_source_track_ids;
  active.open_vocab_encoder_id = record.open_vocab_encoder_id;
  active.open_vocab_ignore_base_state = record.open_vocab_ignore_base_state;
  active.open_vocab_ignore_effective = record.open_vocab_ignore_effective;
  active.open_vocab_ignore_votes = record.open_vocab_ignore_votes;
  active.open_vocab_ignore_best_prompt = record.open_vocab_ignore_best_prompt;
  active.open_vocab_ignore_best_score = record.open_vocab_ignore_best_score;
  active.has_historical_association = record.has_historical_association;
  return active;
}

void appendUniqueTrackIds(std::vector<uint32_t>& dst,
                          const std::vector<uint32_t>& src) {
  std::unordered_set<uint32_t> seen(dst.begin(), dst.end());
  for (const auto track_id : src) {
    if (!seen.insert(track_id).second) {
      continue;
    }
    dst.push_back(track_id);
  }
}

void mergeTrackColors(std::unordered_map<uint32_t, spark_dsg::Color>& dst,
                      const std::unordered_map<uint32_t, spark_dsg::Color>& src) {
  for (const auto& [track_id, color] : src) {
    dst.try_emplace(track_id, color);
  }
}

spark_dsg::Color makeCanonicalTrackColor(uint32_t source_track_id) {
  // Deterministic color keyed to the track that first created the object node.
  constexpr float kGoldenRatioConjugate = 0.61803398875f;
  const float hue = std::fmod(
      (static_cast<float>(source_track_id % 9973) * kGoldenRatioConjugate), 1.0f);
  return spark_dsg::Color::fromHSV(hue, 0.70f, 0.95f);
}

spark_dsg::Color getObjectCreationColor(const TrackObservation& observation) {
  if (observation.has_display_color) {
    return spark_dsg::Color(observation.display_color_r,
                            observation.display_color_g,
                            observation.display_color_b);
  }

  return makeCanonicalTrackColor(observation.track_id);
}

bool rememberSourceTrackColor(TrackedObjectRecord& record,
                              const TrackObservation& observation) {
  if (observation.track_id == 0u) {
    return false;
  }

  const auto color = getObjectCreationColor(observation);
  const auto [iter, inserted] =
      record.source_track_colors.emplace(observation.track_id, color);
  if (inserted) {
    return true;
  }

  if (!observation.has_display_color || iter->second == color) {
    return false;
  }

  iter->second = color;
  return true;
}

bool hasActiveWindowSupport(const std::list<size_t>& support_indices,
                            size_t archived_vertices) {
  return std::any_of(support_indices.begin(), support_indices.end(), [&](size_t idx) {
    return idx >= archived_vertices;
  });
}

void refreshTrackedSupportState(const kimera_pgmo::MeshOffsetInfo& offsets,
                                spark_dsg::ObjectNodeAttributes& attrs,
                                TrackedObjectRecord* record) {
  (void)record;
  attrs.is_active =
      !attrs.mesh_connections.empty() &&
      hasActiveWindowSupport(attrs.mesh_connections, offsets.archived_vertices);
}

void setTrackedPrototypeIfAvailable(const TrackObservation& observation,
                                    spark_dsg::ObjectNodeAttributes& attrs) {
  if (observation.prototype.size() == 0) {
    return;
  }

  if (!observation.prototype.allFinite()) {
    return;
  }

  attrs.semantic_feature = observation.prototype;
}

bool isValidSemanticFeature(const FeatureVector& feature) {
  return feature.size() > 0 && feature.allFinite();
}

float computeFeatureSimilarity(const FeatureVector& lhs, const FeatureVector& rhs) {
  if (!isValidSemanticFeature(lhs) || !isValidSemanticFeature(rhs) ||
      lhs.size() != rhs.size()) {
    return -1.0f;
  }

  const float lhs_norm = lhs.norm();
  const float rhs_norm = rhs.norm();
  if (lhs_norm < 1e-8f || rhs_norm < 1e-8f) {
    return -1.0f;
  }

  return lhs.dot(rhs) / (lhs_norm * rhs_norm);
}

void bumpStateGeneration(TrackedObjectRecord& record) {
  if (record.state_generation != std::numeric_limits<uint64_t>::max()) {
    ++record.state_generation;
  }
}

void bumpSupportGeneration(TrackedObjectRecord& record) {
  if (record.support_generation != std::numeric_limits<uint64_t>::max()) {
    ++record.support_generation;
  }
}

void bumpPrototypeGeneration(TrackedObjectRecord& record) {
  if (record.prototype_generation != std::numeric_limits<uint64_t>::max()) {
    ++record.prototype_generation;
  }
}

}  // namespace trackgraph

}  // namespace hydra
