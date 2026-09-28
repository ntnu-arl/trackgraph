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
// Implementation helpers shared by the segment updater's translation units.
// These are intentionally not installed as public Hydra API.
#include <algorithm>
#include <list>
#include <sstream>

#include "hydra/frontend/trackgraph_segment_updater.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra::trackgraph {
using timing::ScopedTimer;
constexpr const char* kTrackedObjectGeometryTimer =
    "frontend/tracked_objects/geometry_update";
constexpr const char* kTrackedObjectMetadataTimer =
    "frontend/tracked_objects/metadata_write";
using OwnedWinnerSupportMode = TrackGraphSegmentUpdater::OwnedWinnerSupportMode;
inline bool usesWinnerAppend(OwnedWinnerSupportMode mode) {
  return mode == OwnedWinnerSupportMode::WINNER_APPEND;
}

bool addRootSupportVote(
    uint64_t root_uid,
    size_t global_vertex_idx,
    RootSupportMap& root_support,
    TrackGraphSegmentUpdater::RootVertexSetMap& root_support_vertices);

bool addRootPairEvidenceVote(
    uint64_t root_a,
    uint64_t root_b,
    size_t global_vertex_idx,
    RootPairEvidenceMap& pair_evidence,
    TrackGraphSegmentUpdater::RootPairVertexSetMap& pair_evidence_vertices);

std::vector<uint32_t> getAssociatedTrackIds(const TrackedObjectRecord& record);

HistoricalTrackedObjectRecord makeHistoricalRecord(const TrackedObjectRecord& record);

TrackedObjectRecord makeActiveRecord(const HistoricalTrackedObjectRecord& record);

void appendUniqueTrackIds(std::vector<uint32_t>& dst, const std::vector<uint32_t>& src);

void mergeTrackColors(std::unordered_map<uint32_t, spark_dsg::Color>& dst,
                      const std::unordered_map<uint32_t, spark_dsg::Color>& src);

spark_dsg::Color makeCanonicalTrackColor(uint32_t source_track_id);

spark_dsg::Color getObjectCreationColor(const TrackObservation& observation);

bool rememberSourceTrackColor(TrackedObjectRecord& record,
                              const TrackObservation& observation);

bool hasActiveWindowSupport(const std::list<size_t>& support_indices,
                            size_t archived_vertices);

void refreshTrackedSupportState(const kimera_pgmo::MeshOffsetInfo& offsets,
                                spark_dsg::ObjectNodeAttributes& attrs,
                                TrackedObjectRecord* record = nullptr);

void setTrackedPrototypeIfAvailable(const TrackObservation& observation,
                                    spark_dsg::ObjectNodeAttributes& attrs);

bool isValidSemanticFeature(const FeatureVector& feature);

float computeFeatureSimilarity(const FeatureVector& lhs, const FeatureVector& rhs);

void bumpStateGeneration(TrackedObjectRecord& record);

void bumpSupportGeneration(TrackedObjectRecord& record);

void bumpPrototypeGeneration(TrackedObjectRecord& record);

void syncTrackedNodeIdentity(spark_dsg::ObjectNodeAttributes& attrs,
                             const TrackedObjectRecord& record);

void markTrackedMetadata(spark_dsg::ObjectNodeAttributes& attrs,
                         const TrackedObjectRecord& record);

bool upsertOpenVocabFeatureColumn(spark_dsg::ObjectNodeAttributes& attrs,
                                  std::vector<uint32_t>& source_track_ids,
                                  std::string& encoder_id,
                                  uint32_t source_track_id,
                                  const FeatureVector& feature,
                                  const std::string& incoming_encoder_id,
                                  size_t verbosity);

void mergeOpenVocabFeatures(spark_dsg::ObjectNodeAttributes& dst_attrs,
                            TrackedObjectRecord& dst_record,
                            const spark_dsg::ObjectNodeAttributes& src_attrs,
                            const TrackedObjectRecord& src_record,
                            size_t verbosity);

std::optional<timing::ElapsedTimeRecorder::Entry> getLastTimingEntry(
    const std::string& timer_name);

std::string phaseTimerName(const TrackGraphSegmentUpdater::Config& config,
                           const std::string& phase);

void logTrackedObjectPassTiming(
    uint64_t timestamp_ns,
    const std::string& update_timer_name,
    const std::optional<timing::ElapsedTimeRecorder::Entry>& update_timing,
    const std::string& reid_timer_name,
    const std::optional<timing::ElapsedTimeRecorder::Entry>& reid_timing,
    bool reidentifier_enabled,
    bool enabled);

void logReidentificationEvaluationSummary(
    uint64_t timestamp_ns,
    const ActiveSegmentReid::ReidentificationResult::EvaluationSummary& summary,
    bool enabled);

void logExtractedEvidenceSummary(
    uint64_t timestamp_ns,
    const TrackGraphSegmentUpdater::ExtractedEvidenceSummary& summary,
    bool enabled);

void logRootAssociationEvidenceSummary(uint64_t timestamp_ns,
                                       const RootSupportMap& root_support,
                                       const RootPairEvidenceMap& pair_evidence,
                                       bool enabled);

template <typename Container>
void mergeUniqueIndices(std::list<size_t>& dst, const Container& src) {
  std::unordered_set<size_t> seen(dst.begin(), dst.end());
  for (const auto idx : src) {
    if (!seen.insert(idx).second) {
      continue;
    }
    dst.push_back(idx);
  }
}

template <typename Callback>
void forEachAssociatedTrackId(const TrackedObjectRecord& record,
                              const Callback& callback) {
  if (!record.source_track_ids.empty()) {
    for (const auto track_id : record.source_track_ids) {
      callback(track_id);
    }
    return;
  }

  if (record.source_track_id != 0u) {
    callback(record.source_track_id);
  }
}

}  // namespace hydra::trackgraph
