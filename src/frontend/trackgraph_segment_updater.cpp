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
#include "hydra/frontend/trackgraph_segment_updater.h"

#include <glog/logging.h>

#include <algorithm>

#include "trackgraph/segment_support.h"

namespace hydra {
using namespace trackgraph;

void TrackGraphSegmentUpdater::update(uint64_t timestamp_ns,
                                      const InputData& input,
                                      const kimera_pgmo::MeshDelta& delta,
                                      const kimera_pgmo::MeshOffsetInfo& offsets,
                                      DynamicSceneGraph& graph) {
  const auto update_timer_name = config.timer_namespace + "/update";
  const auto active_window_reid_timer_name = config.reidentifier.timer_namespace;
  std::optional<timing::ElapsedTimeRecorder::Entry> reid_timing;
  ActiveSegmentReid::ReidentificationResult reid_result;

  ScopedTimer timer(update_timer_name, timestamp_ns);

  if (!graph.hasMesh()) {
    timer.stop();
    if (config.log_memory_stats) {
      EvidenceMemoryStats memory_stats;
      memory_stats.observation_records = input.track_observations.size();
      for (const auto& observation : input.track_observations) {
        memory_stats.observation_feature_floats +=
            static_cast<size_t>(observation.prototype.size());
      }
      recordMemoryStats(timestamp_ns, graph, memory_stats);
    }
    logTrackedObjectPassTiming(timestamp_ns,
                               update_timer_name,
                               getLastTimingEntry(update_timer_name),
                               active_window_reid_timer_name,
                               reid_timing,
                               reidentifier_ != nullptr,
                               config.log_timing);
    LOG(ERROR) << "Cannot update tracked objects without graph mesh";
    return;
  }

  auto sync_open_vocab_features = [&]() {
    ScopedTimer phase_timer(phaseTimerName(config, "sync_open_vocab_features"),
                            timestamp_ns);
    syncOpenVocabFeatures(graph);
  };
  auto refresh_open_vocab_ignore_states = [&]() {
    ScopedTimer phase_timer(phaseTimerName(config, "refresh_open_vocab_ignore_states"),
                            timestamp_ns);
    refreshOpenVocabIgnoreStates(graph);
  };
  auto launch_historical_associator = [&]() {
    ScopedTimer phase_timer(phaseTimerName(config, "launch_historical_associator"),
                            timestamp_ns);
    associateLongtermSegments(timestamp_ns, offsets, graph);
  };

  {
    ScopedTimer phase_timer(phaseTimerName(config, "remap_existing_connections"),
                            timestamp_ns);
    remapExistingObjectConnections(offsets, graph);
  }
  std::unordered_map<uint32_t, TrackObservation> observations;
  {
    ScopedTimer phase_timer(phaseTimerName(config, "build_observation_index"),
                            timestamp_ns);
    observations.reserve(input.track_observations.size());
    for (const auto& obs : input.track_observations) {
      observations[obs.track_id] = obs;
    }
  }
  for (auto iter = provisional_tracks_.begin(); iter != provisional_tracks_.end();) {
    if (!observations.count(iter->first)) {
      iter = provisional_tracks_.erase(iter);
    } else {
      ++iter;
    }
  }

  {
    ScopedTimer phase_timer(phaseTimerName(config, "mark_missing_tracks_inactive"),
                            timestamp_ns);
    markMissingTracksInactive(timestamp_ns, observations, graph);
  }

  if (observations.empty()) {
    sync_open_vocab_features();
    refresh_open_vocab_ignore_states();
    launch_historical_associator();
    {
      ScopedTimer phase_timer(
          phaseTimerName(config, "freeze_inactive_archived_objects"), timestamp_ns);
      freezeInactiveArchivedObjects(graph);
    }
    timer.stop();
    if (config.log_memory_stats) {
      EvidenceMemoryStats memory_stats;
      memory_stats.observation_records = observations.size();
      memory_stats.observation_buckets = observations.bucket_count();
      recordMemoryStats(timestamp_ns, graph, memory_stats);
    }
    logTrackedObjectPassTiming(timestamp_ns,
                               update_timer_name,
                               getLastTimingEntry(update_timer_name),
                               active_window_reid_timer_name,
                               reid_timing,
                               reidentifier_ != nullptr,
                               config.log_timing);
    return;
  }

  auto evidence = collectEvidence(timestamp_ns, observations, delta, offsets, graph);
  logExtractedEvidenceSummary(timestamp_ns,
                              evidence.summary,
                              config.log_runtime || config.reidentifier.log_summary);
  updateOrCreateObjects(timestamp_ns, input, observations, evidence, offsets, graph);

  {
    ScopedTimer phase_timer(
        phaseTimerName(config, "evaluate_root_association_evidence"), timestamp_ns);

    logRootAssociationEvidenceSummary(
        timestamp_ns,
        evidence.root_support,
        evidence.pair_evidence,
        config.log_runtime || config.reidentifier.log_summary);

    if (reidentifier_ && (!config.reidentifier.evaluate_on_keyframes_only ||
                          input.tracking_is_keyframe)) {
      ScopedTimer reid_timer(active_window_reid_timer_name, timestamp_ns);
      reid_result = reidentifier_->evaluate(timestamp_ns,
                                            active_objects_,
                                            evidence.root_support,
                                            evidence.pair_evidence,
                                            graph);
      reid_timer.stop();
      reid_timing = getLastTimingEntry(active_window_reid_timer_name);
    }
  }

  if (reidentifier_ && !reid_result.merges.empty()) {
    {
      ScopedTimer phase_timer(phaseTimerName(config, "apply_reidentification_results"),
                              timestamp_ns);
      applyActiveReidResults(reid_result, timestamp_ns, offsets, graph);
    }
  }
  if (reidentifier_) {
    {
      ScopedTimer phase_timer(phaseTimerName(config, "log_reidentification_results"),
                              timestamp_ns);
      logReidentificationEvaluationSummary(timestamp_ns,
                                           reid_result.evaluation_summary,
                                           config.reidentifier.log_summary);
    }
  }

  if (usesWinnerAppend(config.enable_owned_winner_support)) {
    ScopedTimer phase_timer(phaseTimerName(config, "apply_owned_winner_support"),
                            timestamp_ns);
    applyOwnedWinnerSupport(timestamp_ns, evidence.owned_winners, offsets, graph);
  }

  {
    ScopedTimer phase_timer(phaseTimerName(config, "freeze_inactive_archived_objects"),
                            timestamp_ns);
    freezeInactiveArchivedObjects(graph);
  }

  sync_open_vocab_features();
  refresh_open_vocab_ignore_states();
  launch_historical_associator();

  timer.stop();
  if (config.log_memory_stats) {
    const auto memory_stats = collectEvidenceMemoryStats(observations,
                                                         evidence.by_track,
                                                         evidence.owned_winners,
                                                         evidence.candidate_roots,
                                                         evidence.root_support,
                                                         evidence.pair_evidence,
                                                         evidence.root_vertices,
                                                         evidence.pair_vertices);
    recordMemoryStats(timestamp_ns, graph, memory_stats);
  }
  logTrackedObjectPassTiming(timestamp_ns,
                             update_timer_name,
                             getLastTimingEntry(update_timer_name),
                             active_window_reid_timer_name,
                             reid_timing,
                             reidentifier_ != nullptr,
                             config.log_timing);
}

std::vector<NodeId> TrackGraphSegmentUpdater::takeBackendSyncNodeIds() {
  std::vector<NodeId> node_ids(backend_sync_node_ids_.begin(),
                               backend_sync_node_ids_.end());
  std::sort(node_ids.begin(), node_ids.end());
  backend_sync_node_ids_.clear();
  return node_ids;
}

}  // namespace hydra
