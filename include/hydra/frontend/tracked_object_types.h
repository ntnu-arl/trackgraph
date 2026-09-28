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

#include <spark_dsg/color.h>
#include <spark_dsg/mesh.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hydra/common/dsg_types.h"
#include "hydra/openset/openset_types.h"

namespace hydra {

enum class OpenVocabIgnoreState { Unknown, Keep, Ignore };

struct OpenVocabIgnoreVotes {
  size_t unknown = 0;
  size_t keep = 0;
  size_t ignore = 0;
};

struct TrackedObjectRecord {
  uint64_t object_uid = 0;
  NodeId node_id = 0;
  uint32_t source_track_id = 0;
  std::vector<uint32_t> source_track_ids;
  std::unordered_map<uint32_t, spark_dsg::Color> source_track_colors;
  uint64_t first_observed_ns = 0;
  uint64_t last_observed_ns = 0;
  uint32_t num_track_updates = 0;
  uint32_t frames_observed = 0;
  uint32_t keyframes_observed = 0;
  uint32_t untracked_updates = 0;
  uint64_t state_generation = 1;
  uint64_t support_generation = 1;
  uint64_t prototype_generation = 1;
  bool is_tracked = false;
  bool historical_revived_candidate_override = false;
  // A historical match protects this identity during inverse containment.
  bool has_historical_association = false;
  std::vector<uint32_t> open_vocab_feature_source_track_ids;
  std::string open_vocab_encoder_id;
  OpenVocabIgnoreState open_vocab_ignore_base_state = OpenVocabIgnoreState::Unknown;
  bool open_vocab_ignore_effective = false;
  OpenVocabIgnoreVotes open_vocab_ignore_votes;
  std::string open_vocab_ignore_best_prompt;
  std::optional<float> open_vocab_ignore_best_score;
};

struct HistoricalTrackedObjectRecord {
  uint64_t object_uid = 0;
  NodeId node_id = 0;
  std::vector<uint32_t> source_track_ids;
  std::unordered_map<uint32_t, spark_dsg::Color> source_track_colors;
  uint64_t first_observed_ns = 0;
  uint64_t last_observed_ns = 0;
  uint32_t num_track_updates = 0;
  uint32_t frames_observed = 0;
  uint32_t keyframes_observed = 0;
  uint32_t untracked_updates = 0;
  uint64_t state_generation = 1;
  uint64_t support_generation = 1;
  uint64_t prototype_generation = 1;
  bool is_tracked = false;
  bool historical_revived_candidate_override = false;
  // A historical match protects this identity during inverse containment.
  bool has_historical_association = false;
  std::vector<uint32_t> open_vocab_feature_source_track_ids;
  std::string open_vocab_encoder_id;
  OpenVocabIgnoreState open_vocab_ignore_base_state = OpenVocabIgnoreState::Unknown;
  bool open_vocab_ignore_effective = false;
  OpenVocabIgnoreVotes open_vocab_ignore_votes;
  std::string open_vocab_ignore_best_prompt;
  std::optional<float> open_vocab_ignore_best_score;
};

struct ProvisionalTrackedObjectRecord {
  uint32_t track_id = 0;
  uint64_t first_observed_ns = 0;
  uint64_t last_observed_ns = 0;
  uint32_t frames_observed = 0;
  uint32_t keyframes_observed = 0;
  uint64_t support_generation = 1;
  uint64_t prototype_generation = 1;
  size_t last_confirmed_support_size = 0;
  FeatureVector semantic_feature;
};

struct TrackVertexEvidence {
  std::vector<size_t> global_vertex_indices;
  std::vector<size_t> confirmed_vertex_indices;
  std::vector<size_t> association_vertex_indices;
  std::vector<size_t> confirmed_association_vertex_indices;
  std::vector<size_t> raw_winner_association_vertex_indices;
  std::unordered_map<uint64_t, std::vector<size_t>> candidate_root_vertex_indices;
  std::unordered_map<uint64_t, std::vector<size_t>>
      confirmed_candidate_root_vertex_indices;
  std::unordered_map<uint64_t, std::vector<size_t>>
      raw_winner_overlap_candidate_root_vertex_indices;
};

struct RootPair {
  uint64_t root_a = 0;
  uint64_t root_b = 0;

  static RootPair make(uint64_t a, uint64_t b) {
    return (a <= b) ? RootPair{a, b} : RootPair{b, a};
  }

  bool operator==(const RootPair& other) const {
    return root_a == other.root_a && root_b == other.root_b;
  }
};

struct RootPairHash {
  size_t operator()(const RootPair& p) const {
    return std::hash<uint64_t>{}(p.root_a) ^ (std::hash<uint64_t>{}(p.root_b) << 1);
  }
};

struct RootPairEvidence {
  size_t shared_live = 0;
  size_t shared_history = 0;
  double accumulated_weight = 0.0;
};

using RootSupportMap = std::unordered_map<uint64_t, size_t>;
using RootPairEvidenceMap =
    std::unordered_map<RootPair, RootPairEvidence, RootPairHash>;

/**
 * Ordered pair of track ids for co-occurrence keying.
 * Canonical form: lower track id first.
 */
struct TrackPair {
  uint32_t track_a;
  uint32_t track_b;

  static TrackPair make(uint32_t a, uint32_t b) {
    return (a <= b) ? TrackPair{a, b} : TrackPair{b, a};
  }

  bool operator==(const TrackPair& other) const {
    return track_a == other.track_a && track_b == other.track_b;
  }
};

struct TrackPairHash {
  size_t operator()(const TrackPair& p) const {
    // Combine two 32-bit values into one 64-bit hash.
    return std::hash<uint64_t>{}((static_cast<uint64_t>(p.track_a) << 32) | p.track_b);
  }
};

/**
 * Per-pair co-occurrence evidence accumulated during the vertex pass.
 * Built from per-vertex runner-up track hypotheses on the persistent mesh.
 */
struct TrackCooccurrence {
  size_t shared_vertex_count = 0;
  double accumulated_likelihood = 0.0;
};

using TrackCooccurrenceMap =
    std::unordered_map<TrackPair, TrackCooccurrence, TrackPairHash>;

}  // namespace hydra
