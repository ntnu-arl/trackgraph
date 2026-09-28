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

namespace hydra::trackgraph {

std::optional<timing::ElapsedTimeRecorder::Entry> getLastTimingEntry(
    const std::string& timer_name) {
  return timing::ElapsedTimeRecorder::instance().getLastEntry(timer_name);
}

void appendTimingFields(
    std::ostringstream& message,
    const std::string& prefix,
    const std::optional<timing::ElapsedTimeRecorder::Entry>& entry) {
  if (!entry) {
    message << " " << prefix << "_ms=n/a";
    return;
  }

  message << " " << prefix << "_ms=" << entry->elapsed_milliseconds();
  if (entry->cpu_timing_enabled) {
    message << " " << prefix << "_process_cpu_ms=" << entry->process_cpu_milliseconds()
            << " " << prefix << "_thread_cpu_ms=" << entry->thread_cpu_milliseconds();
  }
}

std::string phaseTimerName(const TrackGraphSegmentUpdater::Config& config,
                           const std::string& phase) {
  return config.timer_namespace + "/" + phase;
}

void logTrackedObjectPassTiming(
    uint64_t timestamp_ns,
    const std::string& update_timer_name,
    const std::optional<timing::ElapsedTimeRecorder::Entry>& update_timing,
    const std::string& reid_timer_name,
    const std::optional<timing::ElapsedTimeRecorder::Entry>& reid_timing,
    bool reidentifier_enabled,
    bool enabled) {
  if (!enabled) {
    return;
  }

  std::ostringstream message;
  message << "TrackGraphSegmentUpdater timing: ts=" << timestamp_ns
          << " update_timer=" << update_timer_name;
  appendTimingFields(message, "update", update_timing);

  if (reidentifier_enabled) {
    message << " reid_timer=" << reid_timer_name;
    appendTimingFields(message, "reid", reid_timing);
  } else {
    message << " reid_ms=disabled";
  }

  LOG(INFO) << message.str();
}

void logReidentificationEvaluationSummary(
    uint64_t timestamp_ns,
    const ActiveSegmentReid::ReidentificationResult::EvaluationSummary& summary,
    bool enabled) {
  if (!enabled) {
    return;
  }

  LOG(INFO) << "[active-window-reid] summary: ts=" << timestamp_ns
            << " total_objects=" << summary.total_objects
            << " proposed_pairs=" << summary.proposed_pairs
            << " evaluated_pairs=" << summary.evaluated_pairs
            << " evaluated_new_objects=" << summary.evaluated_new_objects
            << " involved_objects=" << summary.involved_objects
            << " avg_evals_per_object=" << summary.avg_evaluations_per_object
            << " avg_evals_per_involved_object="
            << summary.avg_evaluations_per_involved_object
            << " avg_existing_candidates_per_new_object="
            << summary.avg_existing_candidates_per_new_object
            << " max_evals_for_object=" << summary.max_evaluations_for_object;
}

void logExtractedEvidenceSummary(
    uint64_t timestamp_ns,
    const TrackGraphSegmentUpdater::ExtractedEvidenceSummary& summary,
    bool enabled) {
  if (!enabled) {
    return;
  }

  LOG(INFO) << "[active-window-support-history] extract: ts=" << timestamp_ns
            << " used_active_delta_confirmed="
            << (summary.used_active_delta_confirmed ? "true" : "false")
            << " winner_labeled_vertices=" << summary.winner_labeled_vertices
            << " winner_labeled_tracks=" << summary.winner_labeled_tracks
            << " confirmed_vertices=" << summary.confirmed_vertices
            << " confirmed_tracks=" << summary.confirmed_tracks
            << " association_vertices=" << summary.association_vertices
            << " association_tracks=" << summary.association_tracks
            << " raw_winner_association_vertices="
            << summary.raw_winner_association_vertices
            << " raw_winner_association_tracks="
            << summary.raw_winner_association_tracks
            << " raw_winner_overlap_votes=" << summary.raw_winner_overlap_votes
            << " vertices_with_confirmed_entries="
            << summary.vertices_with_confirmed_entries
            << " candidate_root_votes=" << summary.candidate_root_votes
            << " root_support_votes=" << summary.root_support_votes
            << " root_pair_votes=" << summary.root_pair_votes;
}

void logRootAssociationEvidenceSummary(uint64_t timestamp_ns,
                                       const RootSupportMap& root_support,
                                       const RootPairEvidenceMap& pair_evidence,
                                       bool enabled) {
  if (!enabled) {
    return;
  }

  size_t shared_live = 0u;
  size_t shared_history = 0u;
  double accumulated_weight = 0.0;
  for (const auto& [pair, evidence] : pair_evidence) {
    (void)pair;
    shared_live += evidence.shared_live;
    shared_history += evidence.shared_history;
    accumulated_weight += evidence.accumulated_weight;
  }

  LOG(INFO) << "[active-window-reid] evidence: ts=" << timestamp_ns
            << " root_support_count=" << root_support.size()
            << " pair_evidence_count=" << pair_evidence.size()
            << " shared_live=" << shared_live << " shared_history=" << shared_history
            << " shared_total=" << (shared_live + shared_history)
            << " accumulated_weight=" << accumulated_weight;
}

}  // namespace hydra::trackgraph
