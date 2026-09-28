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

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "hydra/reconstruction/projective_integrator.h"

namespace hydra {

class TrackedProjectiveIntegrator : public ProjectiveIntegrator {
 public:
  struct Config : ProjectiveIntegrator::Config {
    float propagation_source_scale = 0.0f; //0.25f;
    float staleness_slope = 0.1f;
    float propagation_confidence_threshold = 0.6f;
    bool use_confidence_weighting = true;
    bool log_keyframe_track_diagnostics = false;
  } const config;

  explicit TrackedProjectiveIntegrator(const Config& config);

  bool computeLabel(const VolumetricMap::Config& map_config,
                    const InputData& data,
                    const cv::Mat& integration_mask,
                    VoxelMeasurement& measurement) const override;

  void updateLabelVoxel(const InputData& data,
                        const VoxelMeasurement& measurement,
                        VoxelTuple& voxels) const override;

  bool requiresSemanticLayer() const override;

  bool requiresInstanceLayer() const override;

 protected:
  void startFrame(const InputData& data) const override;

  void finishFrame() const override;

 private:
  struct TrackObservationInfo {
    uint32_t track_id = InstanceVoxel::NO_TRACK;
    uint32_t frames_since_detection = 0;
  };

  struct TrackFrameDiagnostics {
    size_t frame_instance_pixels = 0;
    size_t outside_truncation_samples = 0;
    size_t near_surface_samples = 0;
    size_t missing_instance_layer_updates = 0;
    size_t zero_evidence_updates = 0;
    size_t label_updates = 0;
    size_t confirmed_updates = 0;
    float confirmed_evidence = 0.0f;
  };

  struct FrameState {
    uint64_t timestamp_ns = 0;
    bool tracking_is_keyframe = false;
    std::unordered_map<uint32_t, TrackObservationInfo> observations;
    std::unordered_map<uint32_t, TrackFrameDiagnostics> diagnostics;
    mutable std::mutex diagnostics_mutex;
  };

  std::optional<TrackObservationInfo> getTrackObservation(uint32_t instance_id,
                                                          const InputData& data) const;

  void recordOutsideTruncationSample(uint32_t track_id) const;

  void recordNearSurfaceSample(uint32_t track_id) const;

  void recordMissingInstanceLayerUpdate(uint32_t track_id) const;

  void recordZeroEvidenceUpdate(uint32_t track_id) const;

  void recordLabelUpdate(uint32_t track_id, bool confirmed, float evidence) const;

  mutable std::shared_ptr<FrameState> active_frame_;
};

void declare_config(TrackedProjectiveIntegrator::Config& config);

}  // namespace hydra
