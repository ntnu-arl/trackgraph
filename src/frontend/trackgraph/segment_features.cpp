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

#include "hydra/openset/embedding_distances.h"
#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

namespace trackgraph {
std::optional<size_t> getTrackIdIndex(const std::vector<uint32_t>& source_track_ids,
                                      uint32_t source_track_id) {
  const auto iter =
      std::find(source_track_ids.begin(), source_track_ids.end(), source_track_id);
  if (iter == source_track_ids.end()) {
    return std::nullopt;
  }

  return static_cast<size_t>(std::distance(source_track_ids.begin(), iter));
}

bool upsertOpenVocabFeatureColumn(spark_dsg::ObjectNodeAttributes& attrs,
                                  std::vector<uint32_t>& source_track_ids,
                                  std::string& encoder_id,
                                  uint32_t source_track_id,
                                  const FeatureVector& feature,
                                  const std::string& incoming_encoder_id,
                                  size_t verbosity) {
  if (feature.size() == 0 || !feature.allFinite()) {
    return false;
  }

  if (attrs.open_vocab_features.size() != 0 &&
      static_cast<size_t>(attrs.open_vocab_features.cols()) !=
          source_track_ids.size()) {
    LOG_IF(WARNING, verbosity > 0)
        << "Skipping open-vocabulary feature for source_track_id=" << source_track_id
        << " due to metadata/column count mismatch: cols="
        << attrs.open_vocab_features.cols()
        << " source_track_ids=" << source_track_ids.size();
    return false;
  }

  if (!encoder_id.empty() && !incoming_encoder_id.empty() &&
      encoder_id != incoming_encoder_id) {
    LOG_IF(WARNING, verbosity > 0)
        << "Skipping open-vocabulary feature for source_track_id=" << source_track_id
        << " due to encoder mismatch: existing='" << encoder_id << "' incoming='"
        << incoming_encoder_id << "'";
    return false;
  }

  if (attrs.open_vocab_features.size() != 0 &&
      attrs.open_vocab_features.rows() != feature.size()) {
    LOG_IF(WARNING, verbosity > 0)
        << "Skipping open-vocabulary feature for source_track_id=" << source_track_id
        << " due to feature size mismatch: existing_rows="
        << attrs.open_vocab_features.rows() << " incoming_rows=" << feature.size();
    return false;
  }

  const auto existing_index = getTrackIdIndex(source_track_ids, source_track_id);
  if (existing_index) {
    if (attrs.open_vocab_features.cols() <=
        static_cast<Eigen::Index>(*existing_index)) {
      LOG_IF(WARNING, verbosity > 0)
          << "Skipping open-vocabulary feature replacement for source_track_id="
          << source_track_id << " because column index " << *existing_index
          << " is out of bounds for " << attrs.open_vocab_features.cols() << " columns";
      return false;
    }

    attrs.open_vocab_features.col(static_cast<Eigen::Index>(*existing_index)) = feature;
    if (encoder_id.empty()) {
      encoder_id = incoming_encoder_id;
    }
    return true;
  }

  const Eigen::Index next_col = attrs.open_vocab_features.cols();
  Eigen::MatrixXf updated(feature.size(), next_col + 1);
  if (next_col > 0) {
    updated.leftCols(next_col) = attrs.open_vocab_features;
  }
  updated.col(next_col) = feature;
  attrs.open_vocab_features = updated;
  source_track_ids.push_back(source_track_id);
  if (encoder_id.empty()) {
    encoder_id = incoming_encoder_id;
  }
  return true;
}

void mergeOpenVocabFeatures(spark_dsg::ObjectNodeAttributes& dst_attrs,
                            TrackedObjectRecord& dst_record,
                            const spark_dsg::ObjectNodeAttributes& src_attrs,
                            const TrackedObjectRecord& src_record,
                            size_t verbosity) {
  const auto& src_track_ids = src_record.open_vocab_feature_source_track_ids;
  const auto cols = src_attrs.open_vocab_features.cols();
  if (cols == 0 || src_track_ids.empty()) {
    return;
  }

  if (static_cast<size_t>(cols) != src_track_ids.size()) {
    LOG_IF(WARNING, verbosity > 0)
        << "Skipping open-vocabulary gallery merge because metadata/column count "
           "mismatch: cols="
        << cols << " source_track_ids=" << src_track_ids.size();
    return;
  }

  for (Eigen::Index i = 0; i < cols; ++i) {
    FeatureVector feature = src_attrs.open_vocab_features.col(i);
    upsertOpenVocabFeatureColumn(dst_attrs,
                                 dst_record.open_vocab_feature_source_track_ids,
                                 dst_record.open_vocab_encoder_id,
                                 src_track_ids.at(static_cast<size_t>(i)),
                                 feature,
                                 src_record.open_vocab_encoder_id,
                                 verbosity);
  }
}

bool ignoreVotesEqual(const OpenVocabIgnoreVotes& lhs,
                      const OpenVocabIgnoreVotes& rhs) {
  return lhs.unknown == rhs.unknown && lhs.keep == rhs.keep && lhs.ignore == rhs.ignore;
}

}  // namespace trackgraph

TrackGraphSegmentUpdater::SourceTrackIgnoreDecision
TrackGraphSegmentUpdater::classifySourceTrackIgnoreDecision(
    const OpenVocabFeatureEntry& entry) const {
  SourceTrackIgnoreDecision decision;
  decision.encoder_id = entry.encoder_id;

  if (!config.open_vocab.ignore_filter.enabled) {
    return decision;
  }

  if (entry.feature.size() == 0 || !entry.feature.allFinite()) {
    return decision;
  }

  if (!config.open_vocab.expected_encoder_id.empty() &&
      entry.encoder_id != config.open_vocab.expected_encoder_id) {
    return decision;
  }

  if (!open_vocab_ignore_prompt_bank_ || open_vocab_ignore_prompt_bank_->empty()) {
    return decision;
  }

  if (!open_vocab_ignore_prompt_bank_->encoder_id.empty() &&
      open_vocab_ignore_prompt_bank_->encoder_id != entry.encoder_id) {
    return decision;
  }

  for (const auto& prompt_embedding : open_vocab_ignore_prompt_bank_->embeddings) {
    if (prompt_embedding.size() != entry.feature.size() ||
        !prompt_embedding.allFinite()) {
      LOG_IF(WARNING, config.open_vocab.ignore_filter.verbosity > 0)
          << "Skipping ignore classification due to prompt-bank / feature size "
             "mismatch "
             "or invalid prompt embedding: prompt_dim="
          << prompt_embedding.size() << " feature_dim=" << entry.feature.size();
      return decision;
    }
  }

  const CosineDistance cosine;
  const auto best = open_vocab_ignore_prompt_bank_->getBestScore(cosine, entry.feature);
  decision.best_score = best.score;
  if (best.index < open_vocab_ignore_prompt_bank_->names.size()) {
    decision.best_prompt = open_vocab_ignore_prompt_bank_->names.at(best.index);
  }
  decision.state = best.score >= config.open_vocab.ignore_filter.ignore_score_threshold
                       ? OpenVocabIgnoreState::Ignore
                       : OpenVocabIgnoreState::Keep;
  return decision;
}

void TrackGraphSegmentUpdater::syncOpenVocabFeatures(DynamicSceneGraph& graph) {
  if (!config.open_vocab.enabled || !open_vocab_feature_cache_) {
    return;
  }

  for (auto& [object_uid, record] : active_objects_) {
    if (!graph.hasNode(record.node_id)) {
      continue;
    }

    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    bool updated = false;
    const auto source_track_ids = getAssociatedTrackIds(record);
    for (const auto source_track_id : source_track_ids) {
      auto entry = open_vocab_feature_cache_->consume(source_track_id);
      if (!entry) {
        continue;
      }

      source_track_ignore_decisions_[source_track_id] =
          classifySourceTrackIgnoreDecision(*entry);

      if (!config.open_vocab.expected_encoder_id.empty() &&
          entry->encoder_id != config.open_vocab.expected_encoder_id) {
        LOG_IF(WARNING, config.open_vocab.verbosity > 0)
            << "Dropping open-vocabulary feature for source_track_id="
            << source_track_id << " because encoder_id='" << entry->encoder_id
            << "' does not match expected_encoder_id='"
            << config.open_vocab.expected_encoder_id << "'";
        continue;
      }

      updated |=
          upsertOpenVocabFeatureColumn(attrs,
                                       record.open_vocab_feature_source_track_ids,
                                       record.open_vocab_encoder_id,
                                       source_track_id,
                                       entry->feature,
                                       entry->encoder_id,
                                       config.open_vocab.verbosity);
    }

    if (updated) {
      markTrackedMetadata(attrs, record);
    }
  }
}

void TrackGraphSegmentUpdater::refreshOpenVocabIgnoreStates(DynamicSceneGraph& graph) {
  for (auto& [object_uid, record] : active_objects_) {
    (void)object_uid;
    const auto previous_base_state = record.open_vocab_ignore_base_state;
    const auto previous_effective = record.open_vocab_ignore_effective;
    const auto previous_votes = record.open_vocab_ignore_votes;
    const auto previous_best_prompt = record.open_vocab_ignore_best_prompt;
    const auto previous_best_score = record.open_vocab_ignore_best_score;
    const auto source_track_ids = getAssociatedTrackIds(record);
    OpenVocabIgnoreVotes votes;
    std::optional<float> best_score;
    std::string best_prompt;
    for (const auto source_track_id : source_track_ids) {
      const auto decision_iter = source_track_ignore_decisions_.find(source_track_id);
      if (decision_iter == source_track_ignore_decisions_.end() ||
          decision_iter->second.state == OpenVocabIgnoreState::Unknown) {
        ++votes.unknown;
        continue;
      }

      if (decision_iter->second.state == OpenVocabIgnoreState::Ignore) {
        ++votes.ignore;
      } else {
        ++votes.keep;
      }

      if (decision_iter->second.best_score &&
          (!best_score || *decision_iter->second.best_score > *best_score)) {
        best_score = decision_iter->second.best_score;
        best_prompt = decision_iter->second.best_prompt;
      }
    }

    if (votes.ignore == 0 && votes.keep == 0) {
      record.open_vocab_ignore_base_state = OpenVocabIgnoreState::Unknown;
    } else if (votes.ignore > votes.keep) {
      record.open_vocab_ignore_base_state = OpenVocabIgnoreState::Ignore;
    } else {
      record.open_vocab_ignore_base_state = OpenVocabIgnoreState::Keep;
    }
    record.open_vocab_ignore_effective =
        record.open_vocab_ignore_base_state == OpenVocabIgnoreState::Ignore;
    record.open_vocab_ignore_votes = votes;
    record.open_vocab_ignore_best_prompt = std::move(best_prompt);
    record.open_vocab_ignore_best_score = best_score;

    if (!graph.hasNode(record.node_id)) {
      continue;
    }

    auto& attrs = graph.getNode(record.node_id).attributes<ObjectNodeAttributes>();
    if (previous_effective != record.open_vocab_ignore_effective) {
      syncTrackedNodeIdentity(attrs, record);
    }

    const bool metadata_changed =
        previous_base_state != record.open_vocab_ignore_base_state ||
        !ignoreVotesEqual(previous_votes, record.open_vocab_ignore_votes) ||
        previous_best_prompt != record.open_vocab_ignore_best_prompt ||
        previous_best_score != record.open_vocab_ignore_best_score;
    if (metadata_changed) {
      markTrackedMetadata(attrs, record);
    }
  }
}

}  // namespace hydra
