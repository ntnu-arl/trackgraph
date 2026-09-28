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
#include <config_utilities/config.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <cctype>
#include <stdexcept>

#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

namespace {
std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

const char* ownedWinnerSupportModeName(OwnedWinnerSupportMode mode) {
  switch (mode) {
    case OwnedWinnerSupportMode::CONFIRMED_ONLY:
      return "confirmed_only";
    case OwnedWinnerSupportMode::WINNER_APPEND:
      return "winner_append";
    case OwnedWinnerSupportMode::INVALID:
      return "invalid";
  }

  return "invalid";
}

struct OwnedWinnerSupportModeConversion {
  static std::string toIntermediate(OwnedWinnerSupportMode value, std::string& error) {
    if (value == OwnedWinnerSupportMode::INVALID) {
      error = "Invalid owned-winner support mode";
    }

    return ownedWinnerSupportModeName(value);
  }

  static void fromIntermediate(const std::string& intermediate,
                               OwnedWinnerSupportMode& value,
                               std::string& error) {
    const auto mode = lowercase(intermediate);
    if (mode == "false" || mode == "0" || mode == "confirmed_only") {
      value = OwnedWinnerSupportMode::CONFIRMED_ONLY;
      return;
    }

    if (mode == "true" || mode == "1" || mode == "winner_append") {
      value = OwnedWinnerSupportMode::WINNER_APPEND;
      return;
    }

    value = OwnedWinnerSupportMode::INVALID;
    error = "Expected one of false, true, confirmed_only, winner_append";
  }
};

TrackGraphSegmentUpdater::Config validateTrackGraphSegmentUpdaterConfig(
    TrackGraphSegmentUpdater::Config config) {
  config.raw_winner_admission = config::checkValid(config.raw_winner_admission);
  config.reidentifier = config::checkValid(config.reidentifier);
  config.historical_associator = config::checkValid(config.historical_associator);
  config.open_vocab = config::checkValid(config.open_vocab);
  if (config.raw_winner_admission.enabled &&
      !usesWinnerAppend(config.enable_owned_winner_support)) {
    throw std::runtime_error(
        "raw_winner_admission.enabled requires enable_owned_winner_support to be "
        "winner_append");
  }
  return config::checkValid(config);
}

}  // namespace

void declare_config(TrackedObjectOpenVocabIgnoreFilterConfig& config) {
  using namespace config;
  name("TrackedObjectOpenVocabIgnoreFilterConfig");
  field(config.enabled, "enabled");
  field(config.source, "source");
  field(config.prompts, "prompts");
  field(config.ignore_score_threshold, "ignore_score_threshold");
  field(config.service_name, "service_name");
  field(config.prompt_bank_path, "prompt_bank_path");
  field(config.verbosity, "verbosity");

  if (config.source != "auto" && config.source != "prompt_bank" &&
      config.source != "service") {
    throw std::runtime_error(
        "TrackedObjectOpenVocabIgnoreFilterConfig.source must be one of: "
        "auto, prompt_bank, service");
  }
  check(config.ignore_score_threshold, GE, -1.0f, "ignore_score_threshold");
  check(config.ignore_score_threshold, LE, 1.0f, "ignore_score_threshold");
}

void declare_config(TrackedObjectOpenVocabConfig& config) {
  using namespace config;
  name("TrackedObjectOpenVocabConfig");
  field(config.enabled, "enabled");
  field(config.expected_encoder_id, "expected_encoder_id");
  field(config.max_cache_entries, "max_cache_entries");
  field(config.verbosity, "verbosity");
  field(config.ignore_filter, "ignore_filter");
}

void declare_config(RawWinnerAdmissionConfig& config) {
  using namespace config;
  name("RawWinnerAdmissionConfig");
  field(config.enabled, "enabled");
  field(config.min_likelihood_ratio, "min_likelihood_ratio");
  check(config.min_likelihood_ratio, GE, 0.0, "min_likelihood_ratio");
  check(config.min_likelihood_ratio, LE, 1.0, "min_likelihood_ratio");
}

void declare_config(TrackGraphSegmentUpdater::Config& config) {
  using namespace config;
  name("TrackGraphSegmentUpdater::Config");
  field(config.layer_id, "layer_id");
  field(config.min_points_create, "min_points_create");
  field(config.min_points_update, "min_points_update");
  field(config.depth_consistency_tol_m, "depth_consistency_tol_m");
  enum_field(config.bounding_box_type,
             "bounding_box_type",
             {{spark_dsg::BoundingBox::Type::INVALID, "INVALID"},
              {spark_dsg::BoundingBox::Type::AABB, "AABB"},
              {spark_dsg::BoundingBox::Type::OBB, "OBB"},
              {spark_dsg::BoundingBox::Type::RAABB, "RAABB"}});
  field(config.mark_missing_tracks_inactive, "mark_missing_tracks_inactive");
  field<OwnedWinnerSupportModeConversion>(config.enable_owned_winner_support,
                                          "enable_owned_winner_support");
  checkCondition(config.enable_owned_winner_support != OwnedWinnerSupportMode::INVALID,
                 "enable_owned_winner_support must be one of false, true, "
                 "confirmed_only, winner_append");
  field(config.log_config, "log_config");
  field(config.log_timing, "log_timing");
  field(config.log_runtime, "log_runtime");
  field(config.log_memory_stats, "log_memory_stats");
  field(config.memory_stats_log_interval, "memory_stats_log_interval");
  checkCondition(config.memory_stats_log_interval > 0,
                 "memory_stats_log_interval must be greater than zero");
  field(config.timer_namespace, "timer_namespace");
  field(config.raw_winner_admission, "raw_winner_admission");
  checkCondition(!config.raw_winner_admission.enabled ||
                     usesWinnerAppend(config.enable_owned_winner_support),
                 "raw_winner_admission.enabled requires enable_owned_winner_support "
                 "to be winner_append");
  field(config.enable_active_window_adoption, "enable_active_window_adoption");
  field(config.reidentifier, "reidentifier");
  field(config.historical_associator, "historical_associator");
  field(config.open_vocab, "open_vocab");
  field(config.merge_on_partof_match, "merge_on_partof_match");
}

TrackGraphSegmentUpdater::TrackGraphSegmentUpdater(
    const Config& config,
    OpenVocabFeatureCache::Ptr cache,
    OpenVocabPromptBank::Ptr ignore_prompt_bank)
    : config(validateTrackGraphSegmentUpdaterConfig(config)),
      next_node_id_('O', 0),
      open_vocab_feature_cache_(std::move(cache)),
      open_vocab_ignore_prompt_bank_(std::move(ignore_prompt_bank)) {
  if (this->config.reidentifier.enabled) {
    reidentifier_ = std::make_unique<ActiveSegmentReid>(this->config.reidentifier);
  }
  if (this->config.historical_associator.enabled) {
    historical_associator_ =
        std::make_unique<TrackGraphLongtermReid>(this->config.historical_associator);
  }

  if (open_vocab_feature_cache_) {
    open_vocab_feature_cache_->setMaxEntries(this->config.open_vocab.max_cache_entries);
  }
  if (this->config.merge_on_partof_match ||
      this->config.reidentifier.create_partof_edges) {
    LOG(WARNING) << "TrackGraphSegmentUpdater config uses deprecated part_of knobs; "
                 << "active-window fragment matches now use merge_from_containment "
                    "semantics only";
  }
  LOG_IF(INFO, this->config.log_config)
      << "TrackGraphSegmentUpdater config: active_window_reidentifier.enabled="
      << this->config.reidentifier.enabled
      << " merge_on_partof_match=" << this->config.merge_on_partof_match
      << " enable_owned_winner_support="
      << ownedWinnerSupportModeName(this->config.enable_owned_winner_support)
      << " active_window_adoption.enabled="
      << this->config.enable_active_window_adoption
      << " raw_winner_admission.enabled=" << this->config.raw_winner_admission.enabled
      << " raw_winner_admission.min_likelihood_ratio="
      << this->config.raw_winner_admission.min_likelihood_ratio
      << " log_memory_stats=" << this->config.log_memory_stats
      << " memory_stats_log_interval=" << this->config.memory_stats_log_interval
      << " historical_associator.enabled=" << this->config.historical_associator.enabled
      << " open_vocab.enabled=" << this->config.open_vocab.enabled;

  if (this->config.open_vocab.ignore_filter.enabled &&
      (!open_vocab_ignore_prompt_bank_ || open_vocab_ignore_prompt_bank_->empty())) {
    LOG(WARNING) << "Tracked-object ignore filter enabled but no prompt bank is "
                    "available; all objects will remain visible with "
                    "open_vocab_ignore_base_state='unknown'";
  }
}

}  // namespace hydra
