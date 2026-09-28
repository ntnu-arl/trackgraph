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
#include <kimera_pgmo/mesh_delta.h>
#include <kimera_pgmo/mesh_traits.h>

#include <array>

#include "hydra/utils/mesh_utilities.h"
#include "segment_support.h"

namespace hydra {
using namespace trackgraph;

namespace {
template <typename TrackIdArrayT, typename TrackLikelihoodArrayT, typename CallbackT>
void forEachPositiveConfirmedTrack(const TrackIdArrayT& track_ids,
                                   const TrackLikelihoodArrayT& track_likelihoods,
                                   const CallbackT& callback) {
  for (size_t slot = 0; slot < spark_dsg::Mesh::kTrackDataSize; ++slot) {
    const auto track_id = track_ids[slot];
    const auto likelihood = track_likelihoods[slot];
    if (track_id == spark_dsg::Mesh::noTrack() || likelihood <= 0.0f) {
      continue;
    }

    callback(track_id, likelihood);
  }
}

template <typename TrackIdArrayT, typename TrackLikelihoodArrayT>
std::optional<float> findPositiveTrackLikelihood(
    const TrackIdArrayT& track_ids,
    const TrackLikelihoodArrayT& track_likelihoods,
    uint32_t query_track_id) {
  std::optional<float> best_likelihood;
  for (size_t slot = 0; slot < spark_dsg::Mesh::kTrackDataSize; ++slot) {
    const auto track_id = track_ids[slot];
    const auto likelihood = track_likelihoods[slot];
    if (track_id != query_track_id || likelihood <= 0.0f) {
      continue;
    }

    if (!best_likelihood || likelihood > *best_likelihood) {
      best_likelihood = likelihood;
    }
  }

  return best_likelihood;
}

bool isValidRawTrackId(uint32_t track_id) {
  return track_id != InstanceVoxel::NO_TRACK && track_id != spark_dsg::Mesh::noTrack();
}

template <typename ContainerT>
bool appendUniqueIndex(ContainerT& values, size_t index) {
  if (std::find(values.begin(), values.end(), index) != values.end()) {
    return false;
  }

  values.push_back(index);
  return true;
}

template <typename T, size_t N>
bool containsFirstN(const std::array<T, N>& values, size_t count, const T& value) {
  for (size_t i = 0; i < count; ++i) {
    if (values[i] == value) {
      return true;
    }
  }

  return false;
}

}  // namespace

TrackGraphSegmentUpdater::UpdateEvidence TrackGraphSegmentUpdater::collectEvidence(
    uint64_t timestamp_ns,
    const std::unordered_map<uint32_t, TrackObservation>& observations,
    const kimera_pgmo::MeshDelta& delta,
    const kimera_pgmo::MeshOffsetInfo& offsets,
    const DynamicSceneGraph& graph) {
  UpdateEvidence evidence;
  evidence.candidate_roots.reserve(observations.size());
  evidence.root_support.reserve(active_objects_.size());
  evidence.root_vertices.reserve(active_objects_.size());
  if (usesWinnerAppend(config.enable_owned_winner_support)) {
    evidence.owned_winners.reserve(track_to_object_uid_.size());
  }
  evidence.by_track = [&]() {
    ScopedTimer phase_timer(phaseTimerName(config, "extract_track_vertex_evidence"),
                            timestamp_ns);
    return extractTrackVertexEvidence(observations,
                                      delta,
                                      offsets,
                                      graph,
                                      evidence.owned_winners,
                                      evidence.candidate_roots,
                                      evidence.root_support,
                                      evidence.pair_evidence,
                                      evidence.root_vertices,
                                      evidence.pair_vertices,
                                      &evidence.summary);
  }();
  return evidence;
}

TrackGraphSegmentUpdater::TrackVertexEvidenceMap
TrackGraphSegmentUpdater::extractTrackVertexEvidence(
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
    ExtractedEvidenceSummary* summary) {
  TrackVertexEvidenceMap evidence_by_track;
  if (observations.empty()) {
    return evidence_by_track;
  }
  evidence_by_track.reserve(observations.size());

  if (!kimera_pgmo::traits::get_vertex_properties(delta).has_label) {
    LOG_IF(INFO, config.log_runtime)
        << "Skipping tracked object update: mesh delta has no labels";
    return evidence_by_track;
  }

  if (summary) {
    summary->used_active_delta_confirmed =
        kimera_pgmo::traits::get_vertex_properties(delta).has_confirmed_track_data;
  }

  // Raw support is sourced from the current mesh winners. Association support is
  // sourced from active-delta confirmed entries, even when the confirmed track is
  // not the winner.
  auto tracked_object_owner = [&](uint32_t track_id, bool require_tracked) -> uint64_t {
    const auto owner_iter = track_to_object_uid_.find(track_id);
    if (owner_iter == track_to_object_uid_.end() || owner_iter->second == 0u) {
      return 0u;
    }

    const auto record_iter = active_objects_.find(owner_iter->second);
    if (record_iter == active_objects_.end() ||
        !graph.hasNode(record_iter->second.node_id)) {
      return 0u;
    }
    if (require_tracked && !record_iter->second.is_tracked) {
      return 0u;
    }

    const auto& attrs =
        graph.getNode(record_iter->second.node_id).attributes<ObjectNodeAttributes>();
    if (getTrackedObjectNodeAttributes(attrs) == nullptr) {
      return 0u;
    }
    if (require_tracked && !attrs.is_active) {
      return 0u;
    }

    return owner_iter->second;
  };
  auto has_valid_tracked_object_owner = [&](uint32_t track_id) {
    return tracked_object_owner(track_id, false) != 0u;
  };
  const bool raw_winner_admission_enabled = config.raw_winner_admission.enabled;
  std::unordered_set<size_t> claimed_global_vertices;
  claimed_global_vertices.reserve(delta.getNumVertices() -
                                  delta.getNumArchivedVertices());
  for (size_t local_idx = delta.getNumArchivedVertices();
       local_idx < delta.getNumVertices();
       ++local_idx) {
    const auto& vertex = delta.getVertex(local_idx);
    const auto winner_track = vertex.traits.label;
    const bool winner_is_active = winner_track != InstanceVoxel::NO_TRACK &&
                                  observations.find(winner_track) != observations.end();

    const auto global_idx = offsets.toGlobalVertex(local_idx);
    if (!claimed_global_vertices.insert(global_idx).second) {
      continue;
    }

    if (usesWinnerAppend(config.enable_owned_winner_support) &&
        winner_track != InstanceVoxel::NO_TRACK &&
        has_valid_tracked_object_owner(winner_track)) {
      owned_winner_vertices_by_track[winner_track].push_back(global_idx);
    }

    TrackVertexEvidence* winner_evidence = nullptr;
    if (winner_is_active) {
      if (summary) {
        ++summary->winner_labeled_vertices;
      }
      auto& evidence = evidence_by_track[winner_track];
      winner_evidence = &evidence;
      if (summary && evidence.global_vertex_indices.empty()) {
        ++summary->winner_labeled_tracks;
      }
      evidence.global_vertex_indices.push_back(global_idx);
      if (raw_winner_admission_enabled) {
        if (summary && evidence.raw_winner_association_vertex_indices.empty()) {
          ++summary->raw_winner_association_tracks;
        }
        if (appendUniqueIndex(evidence.raw_winner_association_vertex_indices,
                              global_idx)) {
          appendUniqueIndex(evidence.association_vertex_indices, global_idx);
          if (summary) {
            ++summary->raw_winner_association_vertices;
          }
        }
      }
    }

    bool has_confirmed_entries = false;
    bool winner_confirmed = false;
    std::array<uint32_t, spark_dsg::Mesh::kTrackDataSize> local_active_confirmed_tracks;
    size_t local_active_confirmed_track_count = 0u;
    std::array<uint64_t, spark_dsg::Mesh::kTrackDataSize> local_confirmed_roots;
    size_t local_confirmed_root_count = 0u;
    forEachPositiveConfirmedTrack(
        vertex.traits.confirmed_track_ids,
        vertex.traits.confirmed_track_likelihoods,
        [&](const auto track_id, const auto likelihood) {
          (void)likelihood;
          if (track_id == 0u) {
            return;
          }

          has_confirmed_entries = true;

          if (track_id == winner_track) {
            winner_confirmed = true;
          }

          if (observations.find(track_id) != observations.end() &&
              !containsFirstN(local_active_confirmed_tracks,
                              local_active_confirmed_track_count,
                              track_id)) {
            local_active_confirmed_tracks[local_active_confirmed_track_count++] =
                track_id;
          }

          const auto root_iter = track_to_object_uid_.find(track_id);
          if (root_iter != track_to_object_uid_.end() && root_iter->second != 0u &&
              !containsFirstN(local_confirmed_roots,
                              local_confirmed_root_count,
                              root_iter->second)) {
            local_confirmed_roots[local_confirmed_root_count++] = root_iter->second;
            if (addRootSupportVote(root_iter->second,
                                   global_idx,
                                   root_support,
                                   root_support_vertices) &&
                summary) {
              ++summary->root_support_votes;
            }
          }
        });

    if (winner_evidence && raw_winner_admission_enabled) {
      const auto winner_likelihood = findPositiveTrackLikelihood(
          vertex.traits.track_ids, vertex.traits.track_likelihoods, winner_track);
      if (winner_likelihood && *winner_likelihood > 0.0f) {
        for (size_t slot = 0; slot < spark_dsg::Mesh::kTrackDataSize; ++slot) {
          const auto candidate_track = vertex.traits.track_ids[slot];
          const auto candidate_likelihood = vertex.traits.track_likelihoods[slot];
          if (!isValidRawTrackId(candidate_track) || candidate_track == winner_track ||
              candidate_likelihood <= 0.0f) {
            continue;
          }

          const double likelihood_ratio = static_cast<double>(candidate_likelihood) /
                                          static_cast<double>(*winner_likelihood);
          if (likelihood_ratio < config.raw_winner_admission.min_likelihood_ratio) {
            continue;
          }

          const auto root_uid = tracked_object_owner(candidate_track, true);
          if (root_uid == 0u) {
            continue;
          }

          if (appendUniqueIndex(
                  winner_evidence->candidate_root_vertex_indices[root_uid],
                  global_idx)) {
            ++track_candidate_roots[winner_track][root_uid];
            if (summary) {
              ++summary->candidate_root_votes;
            }
          }
          if (appendUniqueIndex(
                  winner_evidence
                      ->raw_winner_overlap_candidate_root_vertex_indices[root_uid],
                  global_idx) &&
              summary) {
            ++summary->raw_winner_overlap_votes;
          }
        }
      }
    }

    if (!has_confirmed_entries) {
      continue;
    }

    if (summary) {
      summary->used_active_delta_confirmed = true;
    }
    if (summary) {
      ++summary->vertices_with_confirmed_entries;
    }

    if (winner_evidence && winner_confirmed) {
      if (summary && winner_evidence->confirmed_vertex_indices.empty()) {
        ++summary->confirmed_tracks;
      }
      winner_evidence->confirmed_vertex_indices.push_back(global_idx);
      if (summary) {
        ++summary->confirmed_vertices;
      }
    }

    for (size_t track_index = 0; track_index < local_active_confirmed_track_count;
         ++track_index) {
      const auto track_id = local_active_confirmed_tracks[track_index];
      auto& evidence = evidence_by_track[track_id];
      if (summary && evidence.confirmed_association_vertex_indices.empty()) {
        ++summary->association_tracks;
      }
      const bool added_confirmed_association =
          appendUniqueIndex(evidence.confirmed_association_vertex_indices, global_idx);
      appendUniqueIndex(evidence.association_vertex_indices, global_idx);
      if (summary && added_confirmed_association) {
        ++summary->association_vertices;
      }

      const auto current_root_iter = track_to_object_uid_.find(track_id);
      const uint64_t current_root_uid = current_root_iter == track_to_object_uid_.end()
                                            ? 0u
                                            : current_root_iter->second;
      for (size_t root_index = 0; root_index < local_confirmed_root_count;
           ++root_index) {
        const auto root_uid = local_confirmed_roots[root_index];
        if (root_uid == current_root_uid) {
          continue;
        }

        const bool added_candidate_root = appendUniqueIndex(
            evidence.candidate_root_vertex_indices[root_uid], global_idx);
        appendUniqueIndex(evidence.confirmed_candidate_root_vertex_indices[root_uid],
                          global_idx);
        if (added_candidate_root) {
          ++track_candidate_roots[track_id][root_uid];
        }
        if (summary && added_candidate_root) {
          ++summary->candidate_root_votes;
        }

        if (current_root_uid != 0u &&
            addRootPairEvidenceVote(current_root_uid,
                                    root_uid,
                                    global_idx,
                                    pair_evidence,
                                    pair_evidence_vertices) &&
            summary) {
          ++summary->root_pair_votes;
        }
      }
    }
  }

  return evidence_by_track;
}

}  // namespace hydra
