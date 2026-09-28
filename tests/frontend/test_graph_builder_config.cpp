#include <config_utilities/parsing/yaml.h>
#include <config_utilities/validation.h>
#include <gtest/gtest.h>

#include "hydra/frontend/graph_builder.h"

namespace hydra {
namespace {

TEST(GraphBuilderConfig, ReadsNestedObjectConfigs) {
  const auto config = config::fromYaml<GraphBuilder::Config>(YAML::Load(R"(
object_mode: tracked_objects
objects:
  min_cluster_size: 123
  cluster_tolerance: 0.42
tracked_objects:
  min_points_create: 11
  min_points_update: 7
  log_config: true
  log_timing: true
  log_runtime: true
  log_memory_stats: true
  memory_stats_log_interval: 17
  merge_on_partof_match: true
  enable_owned_winner_support: winner_append
  enable_active_window_adoption: false
  raw_winner_admission:
    enabled: true
    min_likelihood_ratio: 0.25
  open_vocab:
    ignore_filter:
      source: auto
  reidentifier:
    enabled: true
    min_shared_vertices: 5
    min_observation_frames: 3
    candidate_age_window: 64
    debug_decisions: true
    max_debug_decisions_per_object: 12
    log_decisions: true
    log_summary: true
  historical_associator:
    enabled: true
    num_candidates: 7
    max_candidate_radius_m: 1.5
    min_query_keyframes: 4
    inactive_retry_window: 11
    overlap_mode: bbox_query_in_candidate
    nn_search_radius_m: 0.2
    overlap_weight: 0.6
    feature_weight: 0.4
    min_overlap_fraction: 0.25
    min_feature_similarity: 0.8
    min_joint_score: 0.9
    debug_decisions: true
    max_debug_decisions_per_object: 15
    log_jobs: true
    log_proposals: true
    log_candidate_evaluations: true
)"));

  EXPECT_EQ(config.object_mode, "tracked_objects");
  EXPECT_EQ(config.object_config.clustering.min_cluster_size, 123u);
  EXPECT_DOUBLE_EQ(config.object_config.clustering.cluster_tolerance, 0.42);

  EXPECT_EQ(config.trackgraph_segment_updater_config.min_points_create, 11u);
  EXPECT_EQ(config.trackgraph_segment_updater_config.min_points_update, 7u);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.log_config);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.log_timing);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.log_runtime);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.log_memory_stats);
  EXPECT_EQ(config.trackgraph_segment_updater_config.memory_stats_log_interval, 17u);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.merge_on_partof_match);
  EXPECT_EQ(config.trackgraph_segment_updater_config.enable_owned_winner_support,
            TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND);
  EXPECT_FALSE(config.trackgraph_segment_updater_config.enable_active_window_adoption);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.raw_winner_admission.enabled);
  EXPECT_DOUBLE_EQ(
      config.trackgraph_segment_updater_config.raw_winner_admission.min_likelihood_ratio, 0.25);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.reidentifier.enabled);
  EXPECT_EQ(config.trackgraph_segment_updater_config.reidentifier.min_shared_vertices, 5u);
  EXPECT_EQ(config.trackgraph_segment_updater_config.reidentifier.min_observation_frames, 3u);
  EXPECT_EQ(config.trackgraph_segment_updater_config.reidentifier.candidate_age_window, 64u);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.reidentifier.debug_decisions);
  EXPECT_EQ(config.trackgraph_segment_updater_config.reidentifier.max_debug_decisions_per_object,
            12u);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.reidentifier.log_decisions);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.reidentifier.log_summary);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.historical_associator.enabled);
  EXPECT_EQ(config.trackgraph_segment_updater_config.historical_associator.num_candidates, 7u);
  EXPECT_DOUBLE_EQ(config.trackgraph_segment_updater_config.historical_associator.max_candidate_radius_m,
                   1.5);
  EXPECT_EQ(config.trackgraph_segment_updater_config.historical_associator.min_query_keyframes,
            4u);
  EXPECT_EQ(config.trackgraph_segment_updater_config.historical_associator.inactive_retry_window,
            11u);
  EXPECT_EQ(config.trackgraph_segment_updater_config.historical_associator.overlap_mode,
            TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE);
  EXPECT_DOUBLE_EQ(config.trackgraph_segment_updater_config.historical_associator.nn_search_radius_m,
                   0.2);
  EXPECT_DOUBLE_EQ(config.trackgraph_segment_updater_config.historical_associator.overlap_weight,
                   0.6);
  EXPECT_DOUBLE_EQ(config.trackgraph_segment_updater_config.historical_associator.feature_weight,
                   0.4);
  EXPECT_DOUBLE_EQ(
      config.trackgraph_segment_updater_config.historical_associator.min_overlap_fraction, 0.25);
  EXPECT_DOUBLE_EQ(
      config.trackgraph_segment_updater_config.historical_associator.min_feature_similarity, 0.8);
  EXPECT_DOUBLE_EQ(config.trackgraph_segment_updater_config.historical_associator.min_joint_score,
                   0.9);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.historical_associator.debug_decisions);
  EXPECT_EQ(
      config.trackgraph_segment_updater_config.historical_associator.max_debug_decisions_per_object,
      15u);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.historical_associator.log_jobs);
  EXPECT_TRUE(config.trackgraph_segment_updater_config.historical_associator.log_proposals);
  EXPECT_TRUE(
      config.trackgraph_segment_updater_config.historical_associator.log_candidate_evaluations);
}

TEST(GraphBuilderConfig, HistoricalAssociatorDefaultsToNnQueryToCandidateOverlap) {
  const auto config = config::fromYaml<GraphBuilder::Config>(YAML::Load(R"(
object_mode: tracked_objects
tracked_objects:
  open_vocab:
    ignore_filter:
      source: auto
  historical_associator:
    enabled: true
)"));

  EXPECT_EQ(config.trackgraph_segment_updater_config.historical_associator.overlap_mode,
            TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE);
  EXPECT_DOUBLE_EQ(config.trackgraph_segment_updater_config.historical_associator.nn_search_radius_m,
                   0.10);
}

TEST(GraphBuilderConfig, ParsesTrackedObjectOwnedWinnerSupportModes) {
  using Mode = TrackGraphSegmentUpdater::OwnedWinnerSupportMode;

  {
    const auto parsed_config = config::fromYaml<TrackGraphSegmentUpdater::Config>(
        YAML::Load("enable_owned_winner_support: false"));
    EXPECT_EQ(parsed_config.enable_owned_winner_support, Mode::CONFIRMED_ONLY);
    EXPECT_TRUE(config::isValid(parsed_config));
  }

  {
    const auto parsed_config = config::fromYaml<TrackGraphSegmentUpdater::Config>(
        YAML::Load("enable_owned_winner_support: true"));
    EXPECT_EQ(parsed_config.enable_owned_winner_support, Mode::WINNER_APPEND);
    EXPECT_TRUE(config::isValid(parsed_config));
  }

  {
    const auto parsed_config = config::fromYaml<TrackGraphSegmentUpdater::Config>(
        YAML::Load("enable_owned_winner_support: confirmed_only"));
    EXPECT_EQ(parsed_config.enable_owned_winner_support, Mode::CONFIRMED_ONLY);
    EXPECT_TRUE(config::isValid(parsed_config));
  }

  {
    const auto parsed_config = config::fromYaml<TrackGraphSegmentUpdater::Config>(
        YAML::Load("enable_owned_winner_support: winner_append"));
    EXPECT_EQ(parsed_config.enable_owned_winner_support, Mode::WINNER_APPEND);
    EXPECT_TRUE(config::isValid(parsed_config));
  }

  {
    const auto parsed_config = config::fromYaml<TrackGraphSegmentUpdater::Config>(
        YAML::Load("enable_owned_winner_support: winner_active_refresh"));
    EXPECT_EQ(parsed_config.enable_owned_winner_support, Mode::INVALID);
    EXPECT_FALSE(config::isValid(parsed_config));
  }

  {
    const auto parsed_config = config::fromYaml<TrackGraphSegmentUpdater::Config>(
        YAML::Load("enable_owned_winner_support: invalid_mode"));
    EXPECT_EQ(parsed_config.enable_owned_winner_support, Mode::INVALID);
    EXPECT_FALSE(config::isValid(parsed_config));
  }
}

}  // namespace
}  // namespace hydra
