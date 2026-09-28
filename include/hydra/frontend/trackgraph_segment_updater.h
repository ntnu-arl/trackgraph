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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hydra/common/dsg_types.h"
#include "hydra/common/open_vocab_feature_cache.h"
#include "hydra/common/open_vocab_prompt_bank.h"
#include "hydra/frontend/active_segment_reid.h"
#include "hydra/frontend/tracked_object_types.h"
#include "hydra/frontend/trackgraph_longterm_reid.h"
#include "hydra/input/input_data.h"

namespace kimera_pgmo {
class MeshDelta;
struct MeshOffsetInfo;
}  // namespace kimera_pgmo

namespace hydra {

struct TrackedObjectOpenVocabIgnoreFilterConfig {
  bool enabled = false;
  std::string source = "auto";
  std::vector<std::string> prompts;
  float ignore_score_threshold = 0.70f;
  std::string service_name = "tracking/open_vocab/encode_text";
  std::string prompt_bank_path;
  size_t verbosity = 0;
};

void declare_config(TrackedObjectOpenVocabIgnoreFilterConfig& config);

struct TrackedObjectOpenVocabConfig {
  bool enabled = false;
  std::string expected_encoder_id =
      "openclip:ViT-L-14:laion2b_s32b_b82k:precision=fp16";
  size_t max_cache_entries = 1024;
  size_t verbosity = 0;
  TrackedObjectOpenVocabIgnoreFilterConfig ignore_filter;
};

void declare_config(TrackedObjectOpenVocabConfig& config);

struct RawWinnerAdmissionConfig {
  bool enabled = false;
  double min_likelihood_ratio = 0.3;
};

void declare_config(RawWinnerAdmissionConfig& config);

class TrackGraphSegmentUpdater {
 public:
  using ObjectRecord = TrackedObjectRecord;
  using HistoricalRecord = HistoricalTrackedObjectRecord;
  using TrackVertexEvidenceMap = std::unordered_map<uint32_t, TrackVertexEvidence>;
  using WinnerVertexMap = std::unordered_map<uint32_t, std::vector<size_t>>;
  using RootVertexSetMap = std::unordered_map<uint64_t, std::unordered_set<size_t>>;
  using RootPairVertexSetMap =
      std::unordered_map<RootPair, std::unordered_set<size_t>, RootPairHash>;

  enum class OwnedWinnerSupportMode {
    CONFIRMED_ONLY,
    WINNER_APPEND,
    INVALID,
  };

  struct ExtractedEvidenceSummary {
    bool used_active_delta_confirmed = false;
    size_t winner_labeled_vertices = 0;
    size_t winner_labeled_tracks = 0;
    size_t confirmed_vertices = 0;
    size_t confirmed_tracks = 0;
    size_t association_vertices = 0;
    size_t association_tracks = 0;
    size_t raw_winner_association_vertices = 0;
    size_t raw_winner_association_tracks = 0;
    size_t raw_winner_overlap_votes = 0;
    size_t vertices_with_confirmed_entries = 0;
    size_t candidate_root_votes = 0;
    size_t root_support_votes = 0;
    size_t root_pair_votes = 0;
  };

  struct Config {
    std::string layer_id = DsgLayers::OBJECTS;
    size_t min_points_create = 20;
    size_t min_points_update = 5;
    double depth_consistency_tol_m = 0.05;
    spark_dsg::BoundingBox::Type bounding_box_type = spark_dsg::BoundingBox::Type::AABB;
    bool mark_missing_tracks_inactive = true;
    OwnedWinnerSupportMode enable_owned_winner_support =
        OwnedWinnerSupportMode::CONFIRMED_ONLY;
    bool log_config = false;
    bool log_timing = false;
    bool log_runtime = false;
    // Emit updater container cardinalities on the first update, every N updates,
    // and once more at destruction. Disabled by default to avoid scan overhead.
    bool log_memory_stats = false;
    size_t memory_stats_log_interval = 50;
    std::string timer_namespace = "frontend/tracked_objects";
    RawWinnerAdmissionConfig raw_winner_admission;
    bool enable_active_window_adoption = true;
    // Fast active-window re-identifier that relies on shared current-window support.
    ActiveSegmentReid::Config reidentifier;
    TrackGraphLongtermReid::Config historical_associator;
    TrackedObjectOpenVocabConfig open_vocab;
    bool merge_on_partof_match = false;
  } const config;

  explicit TrackGraphSegmentUpdater(
      const Config& config,
      OpenVocabFeatureCache::Ptr cache = nullptr,
      OpenVocabPromptBank::Ptr ignore_prompt_bank = nullptr);
  ~TrackGraphSegmentUpdater();

  void update(uint64_t timestamp_ns,
              const InputData& input,
              const kimera_pgmo::MeshDelta& delta,
              const kimera_pgmo::MeshOffsetInfo& offsets,
              DynamicSceneGraph& graph);

  std::vector<NodeId> takeBackendSyncNodeIds();

 private:
  // Evidence belongs to one update pass. Root votes are extended as tracks are
  // adopted/created before active re-identification evaluates them.
  struct UpdateEvidence {
    TrackVertexEvidenceMap by_track;
    WinnerVertexMap owned_winners;
    std::unordered_map<uint32_t, RootSupportMap> candidate_roots;
    RootSupportMap root_support;
    RootPairEvidenceMap pair_evidence;
    RootVertexSetMap root_vertices;
    RootPairVertexSetMap pair_vertices;
    ExtractedEvidenceSummary summary;
  };

  struct AdoptionDecision {
    uint64_t target_root_uid = 0u;
    bool ambiguous = false;
  };

  UpdateEvidence collectEvidence(
      uint64_t timestamp_ns,
      const std::unordered_map<uint32_t, TrackObservation>& observations,
      const kimera_pgmo::MeshDelta& delta,
      const kimera_pgmo::MeshOffsetInfo& offsets,
      const DynamicSceneGraph& graph);

  AdoptionDecision decideAdoption(
      uint32_t track_id,
      const TrackObservation& obs,
      const TrackVertexEvidence& evidence,
      const std::unordered_map<uint32_t, RootSupportMap>& track_candidate_roots,
      const RootSupportMap& root_support,
      const DynamicSceneGraph& graph) const;

  void updateOrCreateObjects(
      uint64_t timestamp_ns,
      const InputData& input,
      const std::unordered_map<uint32_t, TrackObservation>& observations,
      UpdateEvidence& extracted,
      const kimera_pgmo::MeshOffsetInfo& offsets,
      DynamicSceneGraph& graph);

  struct EvidenceMemoryStats {
    size_t observation_records = 0;
    size_t observation_buckets = 0;
    size_t observation_feature_floats = 0;
    size_t evidence_records = 0;
    size_t evidence_buckets = 0;
    size_t evidence_index_entries = 0;
    size_t evidence_index_capacity = 0;
    size_t evidence_nested_map_entries = 0;
    size_t evidence_nested_map_buckets = 0;
    size_t winner_track_entries = 0;
    size_t winner_map_buckets = 0;
    size_t winner_index_entries = 0;
    size_t winner_index_capacity = 0;
    size_t candidate_root_track_entries = 0;
    size_t candidate_root_outer_buckets = 0;
    size_t candidate_root_entries = 0;
    size_t candidate_root_inner_buckets = 0;
    size_t root_support_entries = 0;
    size_t root_support_buckets = 0;
    size_t pair_evidence_entries = 0;
    size_t pair_evidence_buckets = 0;
    size_t root_vertex_set_entries = 0;
    size_t root_vertex_set_outer_buckets = 0;
    size_t root_vertex_entries = 0;
    size_t root_vertex_inner_buckets = 0;
    size_t pair_vertex_set_entries = 0;
    size_t pair_vertex_set_outer_buckets = 0;
    size_t pair_vertex_entries = 0;
    size_t pair_vertex_inner_buckets = 0;
  };

  TrackVertexEvidenceMap extractTrackVertexEvidence(
      const std::unordered_map<uint32_t, TrackObservation>& observations,
      const kimera_pgmo::MeshDelta& delta,
      const kimera_pgmo::MeshOffsetInfo& offsets,
      const DynamicSceneGraph& graph,
      WinnerVertexMap& owned_winner_vertices_by_track,
      std::unordered_map<uint32_t, RootSupportMap>& track_candidate_roots,
      RootSupportMap& root_support,
      RootPairEvidenceMap& pair_evidence,
      RootVertexSetMap& root_support_vertices,
      RootPairVertexSetMap& pair_evidence_vertices,
      ExtractedEvidenceSummary* summary = nullptr);

  void applyOwnedWinnerSupport(uint64_t timestamp_ns,
                               const WinnerVertexMap& owned_winner_vertices_by_track,
                               const kimera_pgmo::MeshOffsetInfo& offsets,
                               DynamicSceneGraph& graph);

  void remapExistingObjectConnections(const kimera_pgmo::MeshOffsetInfo& offsets,
                                      DynamicSceneGraph& graph);

  void markMissingTracksInactive(
      uint64_t timestamp_ns,
      const std::unordered_map<uint32_t, TrackObservation>& observations,
      DynamicSceneGraph& graph);

  void updateExistingObject(uint64_t timestamp_ns,
                            uint32_t track_id,
                            bool tracking_is_keyframe,
                            const TrackObservation& observation,
                            const TrackVertexEvidence& evidence,
                            const kimera_pgmo::MeshOffsetInfo& offsets,
                            DynamicSceneGraph& graph);

  void createObjectForTrack(
      uint64_t timestamp_ns,
      uint32_t track_id,
      bool tracking_is_keyframe,
      const TrackObservation& observation,
      const TrackVertexEvidence& evidence,
      const kimera_pgmo::MeshOffsetInfo& offsets,
      DynamicSceneGraph& graph,
      bool use_raw_winner_creation_support,
      const ProvisionalTrackedObjectRecord* provisional = nullptr);

  bool adoptTrackIntoExistingRoot(uint64_t timestamp_ns,
                                  uint32_t track_id,
                                  bool tracking_is_keyframe,
                                  const TrackObservation& observation,
                                  const TrackVertexEvidence& evidence,
                                  uint64_t target_root_uid,
                                  const kimera_pgmo::MeshOffsetInfo& offsets,
                                  DynamicSceneGraph& graph);

  void applyActiveReidResults(const ActiveSegmentReid::ReidentificationResult& result,
                              uint64_t timestamp_ns,
                              const kimera_pgmo::MeshOffsetInfo& offsets,
                              DynamicSceneGraph& graph);

  void applyLongtermReidResults(
      const std::vector<TrackGraphLongtermReid::MergeProposal>& proposals,
      uint64_t timestamp_ns,
      const kimera_pgmo::MeshOffsetInfo& offsets,
      DynamicSceneGraph& graph);

  bool mergeTrackedObjectIntoExisting(
      uint64_t existing_object_uid,
      uint64_t duplicate_object_uid,
      uint64_t timestamp_ns,
      const kimera_pgmo::MeshOffsetInfo& offsets,
      DynamicSceneGraph& graph,
      bool enable_historical_candidate_override = false);

  void associateLongtermSegments(uint64_t timestamp_ns,
                                 const kimera_pgmo::MeshOffsetInfo& offsets,
                                 DynamicSceneGraph& graph);

  void freezeInactiveArchivedObjects(DynamicSceneGraph& graph);

  void syncOpenVocabFeatures(DynamicSceneGraph& graph);

  void refreshOpenVocabIgnoreStates(DynamicSceneGraph& graph);

  struct SourceTrackIgnoreDecision {
    OpenVocabIgnoreState state = OpenVocabIgnoreState::Unknown;
    std::string encoder_id;
    std::string best_prompt;
    std::optional<float> best_score;
  };

  SourceTrackIgnoreDecision classifySourceTrackIgnoreDecision(
      const OpenVocabFeatureEntry& entry) const;

  EvidenceMemoryStats collectEvidenceMemoryStats(
      const std::unordered_map<uint32_t, TrackObservation>& observations,
      const TrackVertexEvidenceMap& evidence_by_track,
      const WinnerVertexMap& owned_winner_vertices_by_track,
      const std::unordered_map<uint32_t, RootSupportMap>& track_candidate_roots,
      const RootSupportMap& root_support,
      const RootPairEvidenceMap& pair_evidence,
      const RootVertexSetMap& root_support_vertices,
      const RootPairVertexSetMap& pair_evidence_vertices) const;

  void recordMemoryStats(uint64_t timestamp_ns,
                         const DynamicSceneGraph& graph,
                         const EvidenceMemoryStats& evidence_stats);

  NodeSymbol next_node_id_;
  uint64_t next_object_uid_ = 1;
  std::unordered_map<uint32_t, uint64_t> track_to_object_uid_;
  std::unordered_map<uint64_t, ObjectRecord> active_objects_;
  std::unordered_map<uint64_t, HistoricalRecord> frozen_candidates_;
  std::unordered_map<uint32_t, ProvisionalTrackedObjectRecord> provisional_tracks_;
  std::unordered_set<uint64_t> active_frontier_;
  std::unique_ptr<ActiveSegmentReid> reidentifier_;
  std::unique_ptr<TrackGraphLongtermReid> historical_associator_;
  OpenVocabFeatureCache::Ptr open_vocab_feature_cache_;
  OpenVocabPromptBank::Ptr open_vocab_ignore_prompt_bank_;
  std::unordered_map<uint32_t, SourceTrackIgnoreDecision>
      source_track_ignore_decisions_;
  std::unordered_set<NodeId> backend_sync_node_ids_;
  size_t memory_stats_update_count_ = 0;
  std::string last_memory_stats_payload_;
};

void declare_config(TrackGraphSegmentUpdater::Config& config);

}  // namespace hydra
