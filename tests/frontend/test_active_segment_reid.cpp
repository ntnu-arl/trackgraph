/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <hydra/frontend/active_segment_reid.h>

#include <Eigen/Core>
#include <algorithm>
#include <tuple>

namespace hydra {
namespace {

ActiveSegmentReid::Config makeConfig() {
  ActiveSegmentReid::Config config;
  config.min_shared_vertices = 1;
  config.symmetric_overlap_threshold = 0.6;
  config.contained_overlap_threshold = 0.7;
  config.contained_reverse_overlap_max = 0.4;
  config.feature_similarity_reid_threshold = 0.85;
  config.feature_similarity_partof_low = 0.3;
  config.min_observation_frames = 3;
  config.candidate_age_window = 10;
  return config;
}

TrackedObjectRecord makeRecord(uint64_t object_uid,
                               NodeId node_id,
                               uint32_t track_id,
                               uint32_t frames_observed,
                               uint64_t first_observed_ns = 0,
                               bool is_tracked = true) {
  TrackedObjectRecord record;
  record.object_uid = object_uid;
  record.node_id = node_id;
  record.source_track_id = track_id;
  record.source_track_ids = {track_id};
  record.first_observed_ns = first_observed_ns;
  record.last_observed_ns = first_observed_ns + frames_observed;
  record.num_track_updates = frames_observed;
  record.frames_observed = frames_observed;
  record.is_tracked = is_tracked;
  return record;
}

Eigen::VectorXf makeFeature(float x, float y) {
  Eigen::VectorXf feature(2);
  feature << x, y;
  return feature;
}

void addObject(DynamicSceneGraph& graph,
               NodeId node_id,
               const std::vector<size_t>& mesh_connections,
               const Eigen::VectorXf* feature = nullptr) {
  auto attrs = std::make_unique<ObjectNodeAttributes>();
  attrs->mesh_connections.insert(
      attrs->mesh_connections.end(), mesh_connections.begin(), mesh_connections.end());
  if (feature) {
    attrs->semantic_feature = *feature;
  }
  graph.emplaceNode(DsgLayers::OBJECTS, node_id, std::move(attrs));
}

RootSupportMap makeSupport(
    std::initializer_list<std::pair<const uint64_t, size_t>> entries) {
  return RootSupportMap(entries.begin(), entries.end());
}

RootPairEvidenceMap makeEvidence(
    std::initializer_list<std::tuple<uint64_t, uint64_t, size_t, size_t, double>>
        entries) {
  RootPairEvidenceMap evidence;
  for (const auto& [root_a, root_b, shared_live, shared_history, weight] : entries) {
    evidence.emplace(RootPair::make(root_a, root_b),
                     RootPairEvidence{shared_live, shared_history, weight});
  }
  return evidence;
}

}  // namespace

TEST(ActiveSegmentReid, EmptyEvidenceNoAction) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1, 2, 3}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, false)},
  };

  ActiveSegmentReid reidentifier(makeConfig());
  const auto result = reidentifier.evaluate(0u, objects, {}, {}, graph);

  EXPECT_TRUE(result.merges.empty());
}

TEST(ActiveSegmentReid, SymmetricOverlapHighFeatureMergesNewIntoExisting) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1, 2, 3}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, false)},
  };

  ActiveSegmentReid reidentifier(makeConfig());
  const auto result = reidentifier.evaluate(0u,
                                            objects,
                                            makeSupport({{1u, 4u}, {2u, 4u}}),
                                            makeEvidence({{1u, 2u, 4u, 0u, 2.0}}),
                                            graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(1u), 2u);
}

TEST(ActiveSegmentReid, LegacyDebugOptionsDoNotChangeDirectAliasMerge) {
  auto config = makeConfig();
  config.debug_decisions = true;

  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1, 2, 3}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, false)},
  };

  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(42u,
                                            objects,
                                            makeSupport({{1u, 4u}, {2u, 4u}}),
                                            makeEvidence({{1u, 2u, 2u, 2u, 2.0}}),
                                            graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(1u), 2u);
  config.debug_decisions = false;
  const auto without_debug =
      ActiveSegmentReid(config).evaluate(42u,
                                         objects,
                                         makeSupport({{1u, 4u}, {2u, 4u}}),
                                         makeEvidence({{1u, 2u, 2u, 2u, 2.0}}),
                                         graph);
  EXPECT_EQ(result.merges, without_debug.merges);
}

TEST(ActiveSegmentReid, ContainmentMergesFragmentIntoWhole) {
  auto config = makeConfig();
  config.debug_decisions = true;

  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3, 4}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, true)},
  };

  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(7u,
                                            objects,
                                            makeSupport({{1u, 2u}, {2u, 5u}}),
                                            makeEvidence({{1u, 2u, 0u, 2u, 1.0}}),
                                            graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(1u), 2u);
}

TEST(ActiveSegmentReid, InverseContainmentAllowsNewerWholeToSurviveWhenUnprotected) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3, 4}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 8u, 1u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 3u, 10u, true)},
  };

  ActiveSegmentReid reidentifier(makeConfig());
  const auto result = reidentifier.evaluate(0u,
                                            objects,
                                            makeSupport({{1u, 2u}, {2u, 5u}}),
                                            makeEvidence({{1u, 2u, 2u, 0u, 1.0}}),
                                            graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(1u), 2u);
}

TEST(ActiveSegmentReid, InverseContainmentKeepsProtectedExistingRoot) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3, 4}, &feature);

  auto protected_record = makeRecord(1u, "O0"_id, 11u, 8u, 1u, true);
  protected_record.source_track_ids = {11u, 44u};
  protected_record.has_historical_association = true;

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, protected_record},
      {2u, makeRecord(2u, "O1"_id, 22u, 3u, 10u, true)},
  };

  auto config = makeConfig();
  config.debug_decisions = true;
  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(0u,
                                            objects,
                                            makeSupport({{1u, 2u}, {2u, 5u}}),
                                            makeEvidence({{1u, 2u, 2u, 0u, 1.0}}),
                                            graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(2u), 1u);
}

TEST(ActiveSegmentReid, HistoricalProtectionWorksWithASingleSourceTrack) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3, 4}, &feature);

  auto protected_record = makeRecord(1u, "O0"_id, 11u, 8u, 1u, true);
  protected_record.source_track_ids = {11u};
  protected_record.has_historical_association = true;

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, protected_record},
      {2u, makeRecord(2u, "O1"_id, 22u, 3u, 10u, true)},
  };

  auto config = makeConfig();
  config.debug_decisions = true;
  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(0u,
                                            objects,
                                            makeSupport({{1u, 2u}, {2u, 5u}}),
                                            makeEvidence({{1u, 2u, 2u, 0u, 1.0}}),
                                            graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(2u), 1u);
}

TEST(ActiveSegmentReid, WeakAsymmetricOverlapDoesNotMerge) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1, 2, 3}, &feature);
  addObject(graph, "O1"_id, {3, 4, 5, 6}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, true)},
  };

  auto config = makeConfig();
  config.debug_decisions = true;
  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(0u,
                                            objects,
                                            makeSupport({{1u, 4u}, {2u, 4u}}),
                                            makeEvidence({{1u, 2u, 1u, 0u, 0.5}}),
                                            graph);

  EXPECT_TRUE(result.merges.empty());
}

TEST(ActiveSegmentReid, NewTrackTooYoungProducesNoDecision) {
  auto config = makeConfig();
  config.debug_decisions = true;

  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1, 2, 3}, &feature);
  addObject(graph, "O1"_id, {0, 1, 2, 3}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 2u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, false)},
  };

  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(7u,
                                            objects,
                                            makeSupport({{1u, 4u}, {2u, 4u}}),
                                            makeEvidence({{1u, 2u, 4u, 0u, 2.0}}),
                                            graph);

  EXPECT_TRUE(result.merges.empty());
}

TEST(ActiveSegmentReid, InvalidFeatureSimilarityDefersMerge) {
  auto config = makeConfig();
  config.debug_decisions = true;

  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1, 2, 3});
  addObject(graph, "O1"_id, {0, 1, 2, 3}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, false)},
  };

  ActiveSegmentReid reidentifier(config);
  const auto result = reidentifier.evaluate(0u,
                                            objects,
                                            makeSupport({{1u, 4u}, {2u, 4u}}),
                                            makeEvidence({{1u, 2u, 2u, 2u, 2.0}}),
                                            graph);

  EXPECT_TRUE(result.merges.empty());
}

TEST(ActiveSegmentReid, PrefersStrongerCandidateForSameNewRoot) {
  DynamicSceneGraph graph;
  const auto feature = makeFeature(1.0f, 0.0f);
  addObject(graph, "O0"_id, {0, 1}, &feature);
  addObject(graph, "O1"_id, {0, 1}, &feature);
  addObject(graph, "O2"_id, {0, 1}, &feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, true)},
      {3u, makeRecord(3u, "O2"_id, 33u, 8u, 2u, true)},
  };

  ActiveSegmentReid reidentifier(makeConfig());
  const auto result = reidentifier.evaluate(
      0u,
      objects,
      makeSupport({{1u, 2u}, {2u, 2u}, {3u, 2u}}),
      makeEvidence({{1u, 2u, 1u, 0u, 1.0}, {1u, 3u, 2u, 0u, 4.0}}),
      graph);

  ASSERT_EQ(result.merges.size(), 1u);
  EXPECT_EQ(result.merges.at(1u), 3u);
}

TEST(ActiveSegmentReid, EvaluationSummaryTracksPerPassFanout) {
  DynamicSceneGraph graph;
  const auto new_feature = makeFeature(1.0f, 0.0f);
  const auto other_feature = makeFeature(0.0f, 1.0f);
  addObject(graph, "O0"_id, {0, 1}, &new_feature);
  addObject(graph, "O1"_id, {0, 1}, &other_feature);
  addObject(graph, "O2"_id, {10, 11}, &other_feature);

  const std::unordered_map<uint64_t, TrackedObjectRecord> objects{
      {1u, makeRecord(1u, "O0"_id, 11u, 3u, 10u, true)},
      {2u, makeRecord(2u, "O1"_id, 22u, 8u, 1u, true)},
      {3u, makeRecord(3u, "O2"_id, 33u, 8u, 2u, true)},
  };

  ActiveSegmentReid reidentifier(makeConfig());
  const auto result = reidentifier.evaluate(
      0u,
      objects,
      makeSupport({{1u, 2u}, {2u, 2u}, {3u, 2u}}),
      makeEvidence({{1u, 2u, 2u, 0u, 1.0}, {1u, 3u, 1u, 0u, 0.5}}),
      graph);

  EXPECT_TRUE(result.merges.empty());
  EXPECT_EQ(result.evaluation_summary.total_objects, 3u);
  EXPECT_EQ(result.evaluation_summary.proposed_pairs, 2u);
  EXPECT_EQ(result.evaluation_summary.evaluated_pairs, 2u);
  EXPECT_EQ(result.evaluation_summary.evaluated_new_objects, 1u);
  EXPECT_EQ(result.evaluation_summary.involved_objects, 3u);
  EXPECT_EQ(result.evaluation_summary.max_evaluations_for_object, 2u);
  EXPECT_DOUBLE_EQ(result.evaluation_summary.avg_existing_candidates_per_new_object,
                   2.0);
  EXPECT_DOUBLE_EQ(result.evaluation_summary.avg_evaluations_per_object, 4.0 / 3.0);
  EXPECT_DOUBLE_EQ(result.evaluation_summary.avg_evaluations_per_involved_object,
                   4.0 / 3.0);
}

}  // namespace hydra
