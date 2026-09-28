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
#include <nlohmann/json.hpp>

#include "hydra/utils/mesh_utilities.h"
#include "segment_support.h"

namespace hydra::trackgraph {
nlohmann::json makeTrackColorsJson(
    const std::unordered_map<uint32_t, spark_dsg::Color>& source_track_colors) {
  nlohmann::json colors = nlohmann::json::object();
  for (const auto& [track_id, color] : source_track_colors) {
    colors[std::to_string(track_id)] = {static_cast<int>(color.r),
                                        static_cast<int>(color.g),
                                        static_cast<int>(color.b)};
  }
  return colors;
}

void appendOptionalField(nlohmann::json& target,
                         const char* key,
                         const std::optional<float>& value) {
  if (value) {
    target[key] = *value;
  }
}

const char* toString(OpenVocabIgnoreState state) {
  switch (state) {
    case OpenVocabIgnoreState::Unknown:
      return "unknown";
    case OpenVocabIgnoreState::Keep:
      return "keep";
    case OpenVocabIgnoreState::Ignore:
      return "ignore";
  }

  return "unknown";
}

void syncTrackedNodeIdentity(spark_dsg::ObjectNodeAttributes& attrs,
                             const TrackedObjectRecord& record) {
  auto* tracked = hydra::getTrackedObjectNodeAttributes(attrs);
  if (!tracked) {
    return;
  }

  tracked->object_uid = record.object_uid;
  tracked->source_track_ids = getAssociatedTrackIds(record);
  tracked->open_vocab_ignore_effective = record.open_vocab_ignore_effective;
}

void markTrackedMetadata(spark_dsg::ObjectNodeAttributes& attrs,
                         const TrackedObjectRecord& record) {
  ScopedTimer timer(kTrackedObjectMetadataTimer, record.last_observed_ns);
  nlohmann::json tracked{
      {"object_uid", record.object_uid},
      {"source_track_ids", getAssociatedTrackIds(record)},
      {"is_tracked", record.is_tracked},
      {"open_vocab_ignore_base_state", toString(record.open_vocab_ignore_base_state)},
      {"open_vocab_ignore_votes",
       {{"unknown", record.open_vocab_ignore_votes.unknown},
        {"keep", record.open_vocab_ignore_votes.keep},
        {"ignore", record.open_vocab_ignore_votes.ignore}}}};
  if (!record.source_track_colors.empty()) {
    tracked["source_track_colors"] = makeTrackColorsJson(record.source_track_colors);
  }
  if (!record.open_vocab_feature_source_track_ids.empty()) {
    tracked["open_vocab_feature_source_track_ids"] =
        record.open_vocab_feature_source_track_ids;
  }
  if (!record.open_vocab_encoder_id.empty()) {
    tracked["open_vocab_encoder_id"] = record.open_vocab_encoder_id;
  }
  if (!record.open_vocab_ignore_best_prompt.empty()) {
    tracked["open_vocab_ignore_best_prompt"] = record.open_vocab_ignore_best_prompt;
  }
  appendOptionalField(
      tracked, "open_vocab_ignore_best_score", record.open_vocab_ignore_best_score);
  auto metadata = attrs.metadata.get();
  metadata["hydra_tracked_object"] = std::move(tracked);
  attrs.metadata.set(metadata);
}

}  // namespace hydra::trackgraph
