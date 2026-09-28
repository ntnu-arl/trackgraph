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
#include "hydra/reconstruction/tracked_projective_integrator.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace hydra {
namespace {

static const auto registration =
    config::RegistrationWithConfig<ProjectiveIntegrator,
                                   TrackedProjectiveIntegrator,
                                   TrackedProjectiveIntegrator::Config>(
        "TrackedProjectiveIntegrator");

inline bool measurementOutsideTruncation(const VolumetricMap::Config& map_config,
                                         const ProjectiveIntegrator::Config& config,
                                         const ProjectiveIntegrator::VoxelMeasurement& measurement) {
  const auto is_outside = std::abs(measurement.sdf) >= map_config.truncation_distance;
  return is_outside && (config.skip_extra_colors_and_labels ||
                        !measurement.within_extra_integration_distance);
}

template <typename IdArray, typename LikelihoodArray>
void updateSparseTrackSupport(uint32_t track_id,
                              float evidence,
                              IdArray& track_ids,
                              LikelihoodArray& track_likelihoods) {
  size_t slot = InstanceVoxel::kMaxTracks;
  for (size_t i = 0; i < InstanceVoxel::kMaxTracks; ++i) {
    if (track_ids[i] == track_id) {
      slot = i;
      break;
    }
  }

  if (slot == InstanceVoxel::kMaxTracks) {
    for (size_t i = 0; i < InstanceVoxel::kMaxTracks; ++i) {
      if (track_ids[i] == InstanceVoxel::NO_TRACK) {
        slot = i;
        break;
      }
    }
  }

  if (slot == InstanceVoxel::kMaxTracks) {
    slot = static_cast<size_t>(std::distance(
        track_likelihoods.begin(),
        std::min_element(track_likelihoods.begin(), track_likelihoods.end())));
    track_ids[slot] = track_id;
    track_likelihoods[slot] = 0.0f;
  }

  if (track_ids[slot] == InstanceVoxel::NO_TRACK) {
    track_ids[slot] = track_id;
    track_likelihoods[slot] = 0.0f;
  }

  track_likelihoods[slot] += evidence;
}

template <typename IdArray, typename LikelihoodArray>
uint32_t selectWinningTrack(const IdArray& track_ids,
                            const LikelihoodArray& track_likelihoods) {
  size_t winner = InstanceVoxel::kMaxTracks;
  float winner_likelihood = std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < InstanceVoxel::kMaxTracks; ++i) {
    if (track_ids[i] == InstanceVoxel::NO_TRACK) {
      continue;
    }

    if (track_likelihoods[i] > winner_likelihood) {
      winner = i;
      winner_likelihood = track_likelihoods[i];
    }
  }

  return winner == InstanceVoxel::kMaxTracks ? InstanceVoxel::NO_TRACK
                                             : track_ids[winner];
}

}  // namespace

TrackedProjectiveIntegrator::TrackedProjectiveIntegrator(const Config& config)
    : ProjectiveIntegrator(config), config(config::checkValid(config)) {
  if (config.semantic_integrator) {
    LOG(FATAL) << "TrackedProjectiveIntegrator does not support semantic integrators";
  }
}

void declare_config(TrackedProjectiveIntegrator::Config& config) {
  using namespace config;
  name("TrackedProjectiveIntegrator");
  base<ProjectiveIntegrator::Config>(config);
  field(config.propagation_source_scale, "propagation_source_scale");
  field(config.staleness_slope, "staleness_slope");
  field(config.propagation_confidence_threshold, "propagation_confidence_threshold");
  field(config.use_confidence_weighting, "use_confidence_weighting");
  field(config.log_keyframe_track_diagnostics, "log_keyframe_track_diagnostics");

  check(config.propagation_source_scale, GE, 0.0f, "propagation_source_scale");
  check(config.staleness_slope, GE, 0.0f, "staleness_slope");
  check(config.propagation_confidence_threshold,
        GE,
        0.0f,
        "propagation_confidence_threshold");
}

bool TrackedProjectiveIntegrator::computeLabel(const VolumetricMap::Config& map_config,
                                               const InputData& data,
                                               const cv::Mat& /*integration_mask*/,
                                               VoxelMeasurement& measurement) const {
  measurement.track_id = InstanceVoxel::NO_TRACK;
  measurement.label_weight_scale = 0.0f;

  if (measurementOutsideTruncation(map_config, config, measurement)) {
    if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe &&
        !data.instance_image.empty() && !data.track_observations.empty()) {
      const auto instance_id =
          static_cast<uint32_t>(std::max(0, interpolator_->interpolateID(
                                                data.instance_image,
                                                measurement.interpolation_weights)));
      const auto observation = getTrackObservation(instance_id, data);
      if (observation) {
        recordOutsideTruncationSample(observation->track_id);
      }
    }
    return true;
  }

  if (data.instance_image.empty() || data.track_observations.empty()) {
    return true;
  }

  const auto instance_id =
      static_cast<uint32_t>(std::max(0, interpolator_->interpolateID(
                                            data.instance_image,
                                            measurement.interpolation_weights)));
  if (!instance_id) {
    return true;
  }

  const auto observation = getTrackObservation(instance_id, data);
  if (!observation) {
    return true;
  }

  measurement.track_id = observation->track_id;
  if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe) {
    recordNearSurfaceSample(measurement.track_id);
  }

  if (data.tracking_is_keyframe) {
    measurement.label_weight_scale = 1.0f;
  } else {
    const auto staleness_scale =
        1.0f / (1.0f + config.staleness_slope * observation->frames_since_detection);
    measurement.label_weight_scale = config.propagation_source_scale * staleness_scale;
  }

  if (!data.tracking_is_keyframe && !data.tracking_confidence_image.empty()) {
    const auto pixel_confidence = std::clamp(
        interpolator_->interpolateRange(data.tracking_confidence_image,
                                        measurement.interpolation_weights),
        0.0f,
        1.0f);
    if (pixel_confidence < config.propagation_confidence_threshold) {
      measurement.track_id = InstanceVoxel::NO_TRACK;
      measurement.label_weight_scale = 0.0f;
      return true;
    }

    if (config.use_confidence_weighting) {
      measurement.label_weight_scale *= pixel_confidence;
    }
  }

  return true;
}

void TrackedProjectiveIntegrator::updateLabelVoxel(const InputData& data,
                                                   const VoxelMeasurement& measurement,
                                                   VoxelTuple& voxels) const {
  if (!voxels.instance || measurement.track_id == InstanceVoxel::NO_TRACK) {
    if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe &&
        measurement.track_id != InstanceVoxel::NO_TRACK) {
      recordMissingInstanceLayerUpdate(measurement.track_id);
    }
    return;
  }

  const auto evidence = measurement.weight * measurement.label_weight_scale;
  if (evidence <= 0.0f) {
    if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe) {
      recordZeroEvidenceUpdate(measurement.track_id);
    }
    return;
  }

  auto& voxel = *voxels.instance;
  if (voxel.empty) {
    voxel = InstanceVoxel();
    voxel.empty = false;
  }

  updateSparseTrackSupport(
      measurement.track_id, evidence, voxel.track_ids, voxel.track_likelihoods);
  if (data.tracking_is_keyframe) {
    updateSparseTrackSupport(measurement.track_id,
                             evidence,
                             voxel.confirmed_track_ids,
                             voxel.confirmed_track_likelihoods);
  }
  if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe) {
    recordLabelUpdate(measurement.track_id, data.tracking_is_keyframe, evidence);
  }

  voxel.winning_track_id = selectWinningTrack(voxel.track_ids, voxel.track_likelihoods);
}

bool TrackedProjectiveIntegrator::requiresSemanticLayer() const { return false; }

bool TrackedProjectiveIntegrator::requiresInstanceLayer() const { return true; }

void TrackedProjectiveIntegrator::startFrame(const InputData& data) const {
  auto frame = std::make_shared<FrameState>();
  frame->timestamp_ns = data.timestamp_ns;
  frame->tracking_is_keyframe = data.tracking_is_keyframe;
  frame->observations.reserve(data.track_observations.size());
  if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe) {
    frame->diagnostics.reserve(data.track_observations.size());
  }
  for (const auto& observation : data.track_observations) {
    frame->observations[observation.instance_id] = {
        observation.track_id, observation.frames_since_detection};
    if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe) {
      frame->diagnostics[observation.track_id];
    }
  }

  if (config.log_keyframe_track_diagnostics && data.tracking_is_keyframe &&
      !data.instance_image.empty()) {
    std::unordered_map<uint32_t, uint32_t> track_by_instance;
    track_by_instance.reserve(data.track_observations.size());
    for (const auto& observation : data.track_observations) {
      track_by_instance[observation.instance_id] = observation.track_id;
    }

    const auto increment_pixel_count = [&](uint32_t instance_id) {
      if (instance_id == 0u) {
        return;
      }
      const auto iter = track_by_instance.find(instance_id);
      if (iter == track_by_instance.end()) {
        return;
      }
      ++frame->diagnostics[iter->second].frame_instance_pixels;
    };

    if (data.instance_image.type() == CV_16UC1) {
      for (int row = 0; row < data.instance_image.rows; ++row) {
        const auto* row_data = data.instance_image.ptr<uint16_t>(row);
        for (int col = 0; col < data.instance_image.cols; ++col) {
          increment_pixel_count(row_data[col]);
        }
      }
    } else if (data.instance_image.type() == CV_32SC1) {
      for (int row = 0; row < data.instance_image.rows; ++row) {
        const auto* row_data = data.instance_image.ptr<int32_t>(row);
        for (int col = 0; col < data.instance_image.cols; ++col) {
          increment_pixel_count(static_cast<uint32_t>(std::max(0, row_data[col])));
        }
      }
    } else if (data.instance_image.type() == CV_8UC1) {
      for (int row = 0; row < data.instance_image.rows; ++row) {
        const auto* row_data = data.instance_image.ptr<uint8_t>(row);
        for (int col = 0; col < data.instance_image.cols; ++col) {
          increment_pixel_count(row_data[col]);
        }
      }
    }
  }

  active_frame_ = frame;
}

void TrackedProjectiveIntegrator::finishFrame() const {
  const auto frame = active_frame_;
  if (frame && config.log_keyframe_track_diagnostics && frame->tracking_is_keyframe) {
    std::vector<std::pair<uint32_t, TrackFrameDiagnostics>> diagnostics;
    diagnostics.reserve(frame->diagnostics.size());
    {
      std::lock_guard<std::mutex> lock(frame->diagnostics_mutex);
      for (const auto& [track_id, diag] : frame->diagnostics) {
        diagnostics.emplace_back(track_id, diag);
      }
    }

    std::sort(diagnostics.begin(), diagnostics.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });

    size_t tracks_with_pixels = 0;
    size_t tracks_with_near_surface_samples = 0;
    size_t tracks_with_confirmed_updates = 0;
    for (const auto& [track_id, diag] : diagnostics) {
      if (diag.frame_instance_pixels > 0u) {
        ++tracks_with_pixels;
      }
      if (diag.near_surface_samples > 0u) {
        ++tracks_with_near_surface_samples;
      }
      if (diag.confirmed_updates > 0u) {
        ++tracks_with_confirmed_updates;
      }
    }

    LOG(INFO) << "[tracked-projective-integrator] keyframe_summary: ts="
              << frame->timestamp_ns << " observations=" << frame->observations.size()
              << " tracks_with_pixels=" << tracks_with_pixels
              << " tracks_with_near_surface_samples="
              << tracks_with_near_surface_samples
              << " tracks_with_confirmed_updates=" << tracks_with_confirmed_updates;

    for (const auto& [track_id, diag] : diagnostics) {
      if (diag.frame_instance_pixels == 0u && diag.outside_truncation_samples == 0u &&
          diag.near_surface_samples == 0u && diag.label_updates == 0u &&
          diag.confirmed_updates == 0u) {
        continue;
      }

      std::ostringstream message;
      message << "[tracked-projective-integrator] keyframe_track: ts="
              << frame->timestamp_ns << " track_id=" << track_id
              << " frame_instance_pixels=" << diag.frame_instance_pixels
              << " outside_truncation_samples=" << diag.outside_truncation_samples
              << " near_surface_samples=" << diag.near_surface_samples
              << " missing_instance_layer_updates="
              << diag.missing_instance_layer_updates
              << " zero_evidence_updates=" << diag.zero_evidence_updates
              << " label_updates=" << diag.label_updates
              << " confirmed_updates=" << diag.confirmed_updates
              << " confirmed_evidence=" << diag.confirmed_evidence;
      LOG(INFO) << message.str();
    }
  }

  active_frame_.reset();
}

std::optional<TrackedProjectiveIntegrator::TrackObservationInfo>
TrackedProjectiveIntegrator::getTrackObservation(uint32_t instance_id,
                                                 const InputData& data) const {
  if (const auto frame = active_frame_) {
    const auto iter = frame->observations.find(instance_id);
    if (iter != frame->observations.end()) {
      return iter->second;
    }
  }

  for (const auto& observation : data.track_observations) {
    if (observation.instance_id == instance_id) {
      return TrackObservationInfo{observation.track_id,
                                  observation.frames_since_detection};
    }
  }

  return std::nullopt;
}

void TrackedProjectiveIntegrator::recordOutsideTruncationSample(
    uint32_t track_id) const {
  const auto frame = active_frame_;
  if (!frame || !frame->tracking_is_keyframe) {
    return;
  }

  std::lock_guard<std::mutex> lock(frame->diagnostics_mutex);
  ++frame->diagnostics[track_id].outside_truncation_samples;
}

void TrackedProjectiveIntegrator::recordNearSurfaceSample(uint32_t track_id) const {
  const auto frame = active_frame_;
  if (!frame || !frame->tracking_is_keyframe) {
    return;
  }

  std::lock_guard<std::mutex> lock(frame->diagnostics_mutex);
  ++frame->diagnostics[track_id].near_surface_samples;
}

void TrackedProjectiveIntegrator::recordMissingInstanceLayerUpdate(
    uint32_t track_id) const {
  const auto frame = active_frame_;
  if (!frame || !frame->tracking_is_keyframe) {
    return;
  }

  std::lock_guard<std::mutex> lock(frame->diagnostics_mutex);
  ++frame->diagnostics[track_id].missing_instance_layer_updates;
}

void TrackedProjectiveIntegrator::recordZeroEvidenceUpdate(uint32_t track_id) const {
  const auto frame = active_frame_;
  if (!frame || !frame->tracking_is_keyframe) {
    return;
  }

  std::lock_guard<std::mutex> lock(frame->diagnostics_mutex);
  ++frame->diagnostics[track_id].zero_evidence_updates;
}

void TrackedProjectiveIntegrator::recordLabelUpdate(uint32_t track_id,
                                                    bool confirmed,
                                                    float evidence) const {
  const auto frame = active_frame_;
  if (!frame || !frame->tracking_is_keyframe) {
    return;
  }

  std::lock_guard<std::mutex> lock(frame->diagnostics_mutex);
  auto& diag = frame->diagnostics[track_id];
  ++diag.label_updates;
  if (confirmed) {
    ++diag.confirmed_updates;
    diag.confirmed_evidence += evidence;
  }
}

}  // namespace hydra
