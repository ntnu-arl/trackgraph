/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <hydra/frontend/trackgraph_longterm_reid.h>
#include <hydra/frontend/trackgraph_segment_updater.h>
#include <hydra/input/camera.h>
#include <hydra/utils/pgmo_mesh_traits.h>
#include <hydra/utils/timing_utilities.h>
#include <kimera_pgmo/mesh_delta.h>
#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/serialization/graph_json_serialization.h>

#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace hydra {
namespace {

using kimera_pgmo::MeshDelta;

std::shared_ptr<Camera> createCamera() {
  Camera::Config config;
  config.min_range = 0.1f;
  config.max_range = 10.0f;
  config.width = 4;
  config.height = 4;
  config.cx = 2.0f;
  config.cy = 2.0f;
  config.fx = 2.0f;
  config.fy = 2.0f;
  config.extrinsics = ParamSensorExtrinsics::Config();
  return std::make_shared<Camera>(config, "test_camera");
}

std::shared_ptr<spark_dsg::Mesh> createMesh(bool with_track_data = true) {
  return std::make_shared<spark_dsg::Mesh>(
      true, true, true, false, with_track_data, with_track_data);
}

InputData makeInput(const std::vector<TrackObservation>& observations,
                    bool tracking_is_keyframe = false) {
  InputData input(createCamera());
  input.timestamp_ns = 1;
  input.tracking_is_keyframe = tracking_is_keyframe;
  input.track_observations = observations;
  return input;
}

TrackObservation makeObservation(uint32_t track_id, uint32_t instance_id) {
  TrackObservation observation;
  observation.track_id = track_id;
  observation.instance_id = instance_id;
  return observation;
}

TrackObservation makeObservation(uint32_t track_id,
                                 uint32_t instance_id,
                                 const Eigen::VectorXf& prototype) {
  auto observation = makeObservation(track_id, instance_id);
  observation.prototype = prototype;
  return observation;
}

spark_dsg::Mesh::TrackIdArray makeTrackIds(std::initializer_list<uint32_t> ids) {
  auto track_ids = spark_dsg::Mesh::makeEmptyTrackIds();
  size_t idx = 0;
  for (const auto id : ids) {
    if (idx >= track_ids.size()) {
      break;
    }
    track_ids[idx++] = id;
  }
  return track_ids;
}

spark_dsg::Mesh::TrackLikelihoodArray makeTrackLikelihoods(
    std::initializer_list<float> likelihoods) {
  auto track_likelihoods = spark_dsg::Mesh::makeEmptyTrackLikelihoods();
  size_t idx = 0;
  for (const auto likelihood : likelihoods) {
    if (idx >= track_likelihoods.size()) {
      break;
    }
    track_likelihoods[idx++] = likelihood;
  }
  return track_likelihoods;
}

void addPoints(
    MeshDelta& delta,
    uint32_t label,
    const Eigen::Vector3f& offset,
    const Eigen::Vector3f& scale,
    const spark_dsg::Mesh::TrackIdArray* track_ids = nullptr,
    const spark_dsg::Mesh::TrackLikelihoodArray* track_likelihoods = nullptr,
    const spark_dsg::Mesh::TrackIdArray* confirmed_track_ids = nullptr,
    const spark_dsg::Mesh::TrackLikelihoodArray* confirmed_track_likelihoods = nullptr,
    bool archive = false) {
  const auto corners = BoundingBox(scale, offset).corners();
  for (const auto& corner : corners) {
    kimera_pgmo::traits::VertexTraits traits;
    traits.properties.has_label = true;
    traits.label = label;
    if (track_ids && track_likelihoods) {
      traits.properties.has_track_data = true;
      traits.track_ids = *track_ids;
      traits.track_likelihoods = *track_likelihoods;
    }
    if (confirmed_track_ids && confirmed_track_likelihoods) {
      traits.properties.has_confirmed_track_data = true;
      traits.confirmed_track_ids = *confirmed_track_ids;
      traits.confirmed_track_likelihoods = *confirmed_track_likelihoods;
    } else if (track_ids && track_likelihoods) {
      traits.properties.has_confirmed_track_data = true;
      traits.confirmed_track_ids = *track_ids;
      traits.confirmed_track_likelihoods = *track_likelihoods;
    } else if (label != InstanceVoxel::NO_TRACK) {
      traits.properties.has_confirmed_track_data = true;
      traits.confirmed_track_ids = spark_dsg::Mesh::makeEmptyTrackIds();
      traits.confirmed_track_likelihoods = spark_dsg::Mesh::makeEmptyTrackLikelihoods();
      traits.confirmed_track_ids[0] = label;
      traits.confirmed_track_likelihoods[0] = 1.0f;
    }
    delta.addVertex(corner, traits, archive);
  }
}

void addRawLabeledPointsWithoutConfirmed(MeshDelta& delta,
                                         uint32_t label,
                                         const Eigen::Vector3f& offset,
                                         const Eigen::Vector3f& scale,
                                         bool archive = false) {
  const auto corners = BoundingBox(scale, offset).corners();
  for (const auto& corner : corners) {
    kimera_pgmo::traits::VertexTraits traits;
    traits.properties.has_label = true;
    traits.label = label;
    delta.addVertex(corner, traits, archive);
  }
}

void addRawWinnerPointsWithoutConfirmed(
    MeshDelta& delta,
    uint32_t label,
    const Eigen::Vector3f& offset,
    const Eigen::Vector3f& scale,
    const spark_dsg::Mesh::TrackIdArray& track_ids,
    const spark_dsg::Mesh::TrackLikelihoodArray& track_likelihoods,
    bool archive = false) {
  const auto no_confirmed_ids = makeTrackIds({});
  const auto no_confirmed_likelihoods = makeTrackLikelihoods({});
  addPoints(delta,
            label,
            offset,
            scale,
            &track_ids,
            &track_likelihoods,
            &no_confirmed_ids,
            &no_confirmed_likelihoods,
            archive);
}

MeshDelta makeDelta(
    const MeshDelta::TrackingInfo& tracking,
    uint32_t label,
    const Eigen::Vector3f& offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f),
    const Eigen::Vector3f& scale = Eigen::Vector3f::Constant(0.1f),
    const spark_dsg::Mesh::TrackIdArray* track_ids = nullptr,
    const spark_dsg::Mesh::TrackLikelihoodArray* track_likelihoods = nullptr,
    const spark_dsg::Mesh::TrackIdArray* confirmed_track_ids = nullptr,
    const spark_dsg::Mesh::TrackLikelihoodArray* confirmed_track_likelihoods =
        nullptr) {
  MeshDelta delta(tracking);
  addPoints(delta,
            label,
            offset,
            scale,
            track_ids,
            track_likelihoods,
            confirmed_track_ids,
            confirmed_track_likelihoods);
  return delta;
}

TrackGraphLongtermReid::QuerySnapshot makeHistoricalQuerySnapshot(
    uint64_t object_uid,
    uint64_t first_observed_ns,
    const std::vector<Eigen::Vector3f>& support_points,
    const FeatureVector& feature) {
  TrackGraphLongtermReid::QuerySnapshot query;
  query.object_uid = object_uid;
  query.first_observed_ns = first_observed_ns;
  const BoundingBox bbox(support_points, BoundingBox::Type::AABB);
  query.centroid = bbox.world_P_center.cast<double>();
  query.semantic_feature = feature;
  query.support_points = support_points;
  return query;
}

template <size_t N>
TrackGraphLongtermReid::QuerySnapshot makeHistoricalQuerySnapshot(
    uint64_t object_uid,
    uint64_t first_observed_ns,
    const std::array<Eigen::Vector3f, N>& support_points,
    const FeatureVector& feature) {
  return makeHistoricalQuerySnapshot(
      object_uid,
      first_observed_ns,
      std::vector<Eigen::Vector3f>(support_points.begin(), support_points.end()),
      feature);
}

TrackGraphLongtermReid::QueryHeader makeHistoricalQueryHeader(
    uint64_t object_uid,
    uint64_t first_observed_ns,
    const Eigen::Vector3d& centroid,
    const FeatureVector& feature) {
  TrackGraphLongtermReid::QueryHeader query;
  query.object_uid = object_uid;
  query.first_observed_ns = first_observed_ns;
  query.centroid = centroid;
  query.semantic_feature = feature;
  return query;
}

TrackGraphLongtermReid::CandidateSnapshot makeHistoricalCandidateSnapshot(
    uint64_t object_uid,
    uint64_t first_observed_ns,
    const std::vector<Eigen::Vector3f>& support_points,
    const FeatureVector& feature) {
  TrackGraphLongtermReid::CandidateSnapshot candidate;
  candidate.root_object_uid = object_uid;
  candidate.first_observed_ns = first_observed_ns;
  candidate.aggregate_bounding_box =
      BoundingBox(support_points, BoundingBox::Type::AABB);
  candidate.centroid = candidate.aggregate_bounding_box.world_P_center.cast<double>();
  candidate.semantic_feature = feature;
  candidate.support_points = support_points;
  return candidate;
}

template <size_t N>
TrackGraphLongtermReid::CandidateSnapshot makeHistoricalCandidateSnapshot(
    uint64_t object_uid,
    uint64_t first_observed_ns,
    const std::array<Eigen::Vector3f, N>& support_points,
    const FeatureVector& feature) {
  return makeHistoricalCandidateSnapshot(
      object_uid,
      first_observed_ns,
      std::vector<Eigen::Vector3f>(support_points.begin(), support_points.end()),
      feature);
}

TrackGraphLongtermReid::CandidateHeader makeHistoricalCandidateHeader(
    uint64_t object_uid,
    uint64_t first_observed_ns,
    const std::vector<Eigen::Vector3f>& support_points,
    const FeatureVector& feature) {
  TrackGraphLongtermReid::CandidateHeader candidate;
  candidate.root_object_uid = object_uid;
  candidate.first_observed_ns = first_observed_ns;
  candidate.aggregate_bounding_box =
      BoundingBox(support_points, BoundingBox::Type::AABB);
  candidate.centroid = candidate.aggregate_bounding_box.world_P_center.cast<double>();
  candidate.semantic_feature = feature;
  return candidate;
}

MeshDelta makeArchivedDelta(
    const MeshDelta::TrackingInfo& tracking,
    uint32_t label,
    const Eigen::Vector3f& offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f),
    const Eigen::Vector3f& scale = Eigen::Vector3f::Constant(0.1f)) {
  MeshDelta delta(tracking);
  addPoints(delta, label, offset, scale, nullptr, nullptr, nullptr, nullptr, true);
  return delta;
}

OpenVocabFeatureEntry makeOpenVocabEntry(const Eigen::VectorXf& feature,
                                         uint64_t timestamp_ns = 0,
                                         uint32_t frame_index = 0,
                                         uint32_t source_instance_id = 0) {
  OpenVocabFeatureEntry entry;
  entry.feature = feature;
  entry.encoder_id = "openclip:ViT-L-14:laion2b_s32b_b82k:precision=fp16";
  entry.timestamp_ns = timestamp_ns;
  entry.frame_index = frame_index;
  entry.source_instance_id = source_instance_id;
  return entry;
}

OpenVocabPromptBank::Ptr makeIgnorePromptBank(
    const std::vector<std::pair<std::string, Eigen::VectorXf>>& prompts) {
  auto bank = std::make_shared<OpenVocabPromptBank>();
  bank->encoder_id = "openclip:ViT-L-14:laion2b_s32b_b82k:precision=fp16";
  for (const auto& [name, embedding] : prompts) {
    bank->names.push_back(name);
    bank->embeddings.push_back(embedding.normalized());
  }
  return bank;
}

kimera_pgmo::MeshDelta::TrackingInfo makeIdentityRemapTracking(size_t num_vertices) {
  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < num_vertices; ++i) {
    remap[i] = i;
  }
  return MeshDelta::TrackingInfo::with_remap(0, num_vertices, 0, remap);
}

void seedFrozenTrackedObject(TrackGraphSegmentUpdater& updater,
                             DynamicSceneGraph& graph,
                             kimera_pgmo::MeshOffsetInfo& offsets,
                             uint64_t initial_timestamp_ns,
                             uint32_t track_id,
                             uint32_t source_instance_id,
                             const FeatureVector& feature,
                             const Eigen::Vector3f& offset,
                             const Eigen::Vector3f& scale) {
  auto initial = makeDelta({0, 0, 0}, track_id, offset, scale);
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(
      initial_timestamp_ns,
      makeInput({makeObservation(track_id, source_instance_id, feature)}, true),
      initial,
      offsets,
      graph);

  auto steady = makeDelta(
      makeIdentityRemapTracking(graph.mesh()->numVertices()), track_id, offset, scale);
  steady.updateMesh(*graph.mesh(), offsets);
  updater.update(
      initial_timestamp_ns + 1,
      makeInput({makeObservation(track_id, source_instance_id, feature)}, true),
      steady,
      offsets,
      graph);

  MeshDelta archive(makeIdentityRemapTracking(graph.mesh()->numVertices()));
  addPoints(archive, track_id, offset, scale, nullptr, nullptr, nullptr, nullptr, true);
  archive.updateMesh(*graph.mesh(), offsets);
  updater.update(initial_timestamp_ns + 2, makeInput({}), archive, offsets, graph);
}

template <typename TickFunc, typename Predicate>
bool waitForCondition(const TickFunc& tick,
                      const Predicate& predicate,
                      size_t max_passes = 20) {
  for (size_t i = 0; i < max_passes; ++i) {
    if (predicate()) {
      return true;
    }
    tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  return predicate();
}

NodeId findNodeByName(const DynamicSceneGraph& graph, const std::string& name) {
  for (const auto& [node_id, node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
    if (node->attributes<ObjectNodeAttributes>().name == name) {
      return node_id;
    }
  }

  return NodeId(0);
}

const TrackedObjectNodeAttributes& trackedNodeAttrs(const DynamicSceneGraph& graph,
                                                    NodeId node_id) {
  return graph.getNode(node_id).attributes<TrackedObjectNodeAttributes>();
}

std::vector<uint32_t> trackedSourceTrackIds(const DynamicSceneGraph& graph,
                                            NodeId node_id) {
  return trackedNodeAttrs(graph, node_id).source_track_ids;
}

bool trackedIgnoreEffective(const DynamicSceneGraph& graph, NodeId node_id) {
  return trackedNodeAttrs(graph, node_id).open_vocab_ignore_effective;
}

uint64_t findObjectUidByName(const DynamicSceneGraph& graph, const std::string& name) {
  const auto node_id = findNodeByName(graph, name);
  if (node_id == NodeId(0)) {
    return 0u;
  }

  return trackedNodeAttrs(graph, node_id).object_uid;
}

}  // namespace

// Exercise the persistent graph contract with both supported ownership policies.
// Runtime diagnostics must not affect geometry, identities, or metadata histories.
TEST(TrackGraphSegmentUpdater, RuntimeLoggingDoesNotChangeSerializedGraph) {
  using Mode = TrackGraphSegmentUpdater::OwnedWinnerSupportMode;
  for (const auto mode : {Mode::CONFIRMED_ONLY, Mode::WINNER_APPEND}) {
    auto replay = [mode](bool log_runtime) {
      TrackGraphSegmentUpdater::Config config;
      config.min_points_create = 4;
      config.enable_owned_winner_support = mode;
      config.log_runtime = log_runtime;
      config.reidentifier.log_decisions = log_runtime;
      config.reidentifier.debug_decisions = true;
      TrackGraphSegmentUpdater updater(config);
      DynamicSceneGraph graph;
      graph.setMesh(createMesh(false));
      kimera_pgmo::MeshOffsetInfo offsets;
      const auto prototype = Eigen::Vector2f(1.0f, 0.0f);
      const auto observation = makeObservation(11u, 1u, prototype);
      std::vector<nlohmann::json> snapshots;
      auto capture = [&]() {
        const auto metadata =
            graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
        const auto& tracked = metadata.at("hydra_tracked_object");
        for (const auto* field : {"association_history",
                                  "association_evaluation_history",
                                  "merge_history",
                                  "reid_debug_history",
                                  "historical_association_history",
                                  "historical_debug_history"}) {
          EXPECT_FALSE(tracked.contains(field)) << field;
        }
        snapshots.push_back(
            nlohmann::json::parse(spark_dsg::io::json::writeGraph(graph, true)));
      };

      MeshDelta initial({0, 0, 0});
      addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
      initial.updateMesh(*graph.mesh(), offsets);
      updater.update(1, makeInput({observation}, true), initial, offsets, graph);
      capture();

      std::map<size_t, size_t> remap;
      for (size_t i = 0; i < 8; ++i) {
        remap[i] = i;
      }
      MeshDelta raw(MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap));
      addRawLabeledPointsWithoutConfirmed(
          raw, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
      addRawLabeledPointsWithoutConfirmed(
          raw, 11u, {3, 2, 3}, Eigen::Vector3f::Constant(0.1f));
      raw.updateMesh(*graph.mesh(), offsets);
      updater.update(2, makeInput({observation}), raw, offsets, graph);
      capture();

      updater.update(3, makeInput({}), raw, offsets, graph);
      capture();
      updater.update(4, makeInput({observation}, true), raw, offsets, graph);
      capture();
      return snapshots;
    };
    EXPECT_EQ(replay(false), replay(true));
  }
}

TEST(TrackGraphSegmentUpdater, RawWinnerAdmissionDisabledAllowsAllOwnedSupportModes) {
  for (const auto mode :
       {TrackGraphSegmentUpdater::OwnedWinnerSupportMode::CONFIRMED_ONLY,
        TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND}) {
    TrackGraphSegmentUpdater::Config config;
    config.enable_owned_winner_support = mode;
    config.raw_winner_admission.enabled = false;
    EXPECT_NO_THROW({ TrackGraphSegmentUpdater updater(config); });
  }
}

TEST(TrackGraphSegmentUpdater, RawWinnerAdmissionRequiresAppendPathSupport) {
  {
    TrackGraphSegmentUpdater::Config config;
    config.raw_winner_admission.enabled = true;
    config.enable_owned_winner_support =
        TrackGraphSegmentUpdater::OwnedWinnerSupportMode::CONFIRMED_ONLY;
    EXPECT_THROW({ TrackGraphSegmentUpdater updater(config); }, std::runtime_error);
  }

  for (const auto mode :
       {TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND}) {
    TrackGraphSegmentUpdater::Config config;
    config.raw_winner_admission.enabled = true;
    config.enable_owned_winner_support = mode;
    EXPECT_NO_THROW({ TrackGraphSegmentUpdater updater(config); });
  }
}

TEST(TrackGraphSegmentUpdater, MemoryStatsAreOptInAndIncludeFinalSnapshot) {
  EXPECT_FALSE(TrackGraphSegmentUpdater::Config().log_memory_stats);

  TrackGraphSegmentUpdater::Config config;
  config.log_memory_stats = true;
  config.memory_stats_log_interval = 2;

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  MeshDelta delta({0, 0, 0});

  testing::internal::CaptureStderr();
  {
    TrackGraphSegmentUpdater updater(config);
    updater.update(42, makeInput({}), delta, offsets, graph);
  }
  const auto log_output = testing::internal::GetCapturedStderr();

  EXPECT_NE(log_output.find("[tracked-object-memory] snapshot: ts=42 update_index=1"),
            std::string::npos);
  EXPECT_NE(log_output.find("observation_records=0"), std::string::npos);
  EXPECT_NE(log_output.find("final=false"), std::string::npos);
  EXPECT_NE(log_output.find("final=true"), std::string::npos);
}

TEST(TrackGraphSegmentUpdater, VertexEvidenceComesFromMeshLabelsOnly) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta delta({0, 0, 0});
  addPoints(delta, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  addPoints(delta, 42u, {4, 5, 6}, Eigen::Vector3f::Constant(0.1f));
  delta.updateMesh(*graph.mesh(), offsets);

  auto input = makeInput({makeObservation(11u, 1u)});
  updater.update(0, input, delta, offsets, graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(attrs.is_active);
  EXPECT_EQ(attrs.name, "track_11");
  EXPECT_EQ(attrs.semantic_label, SemanticNodeAttributes::NO_SEMANTIC_LABEL);
  const auto metadata = attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  const auto& tracked_metadata = metadata.at("hydra_tracked_object");
  EXPECT_EQ(tracked_metadata.at("source_track_ids").get<std::vector<uint32_t>>(),
            (std::vector<uint32_t>{11u}));

  EXPECT_TRUE(tracked_metadata.at("is_tracked").get<bool>());
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));

  EXPECT_EQ(
      std::vector<size_t>(attrs.mesh_connections.begin(), attrs.mesh_connections.end()),
      (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7}));

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 8; ++i) {
    remap[i] = i;
  }

  MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap));
  unchanged.updateMesh(*graph.mesh(), offsets);
  updater.update(1, makeInput({}), unchanged, offsets, graph);
  const auto& untracked_attrs =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_FALSE(untracked_attrs.is_active);
  const auto untracked_metadata = untracked_attrs.metadata.get();
  ASSERT_TRUE(untracked_metadata.contains("hydra_tracked_object"));
  const auto& untracked_tracked_metadata =
      untracked_metadata.at("hydra_tracked_object");
  EXPECT_FALSE(untracked_tracked_metadata.at("is_tracked").get<bool>());

  auto archived =
      makeArchivedDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                        11u,
                        Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                        Eigen::Vector3f::Constant(0.1f));
  archived.updateMesh(*graph.mesh(), offsets);
  updater.update(2, makeInput({}), archived, offsets, graph);
  EXPECT_FALSE(graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().is_active);
}

TEST(TrackGraphSegmentUpdater, UntrackedObjectRemainsActiveWhileSupportStaysInWindow) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);
  }

  {
    auto active_support =
        makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                  11u,
                  Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                  Eigen::Vector3f::Constant(0.1f));
    active_support.updateMesh(*graph.mesh(), offsets);
    updater.update(1, makeInput({}), active_support, offsets, graph);
  }

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(attrs.is_active);

  auto archived =
      makeArchivedDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                        11u,
                        Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                        Eigen::Vector3f::Constant(0.1f));
  archived.updateMesh(*graph.mesh(), offsets);
  updater.update(2, makeInput({}), archived, offsets, graph);
  EXPECT_FALSE(graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().is_active);
}

TEST(TrackGraphSegmentUpdater,
     MissingTrackMetadataUpdatesAndRestoresWithoutDebugState) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);
  }

  const auto metadata_before =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();

  {
    auto active_support =
        makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                  11u,
                  Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                  Eigen::Vector3f::Constant(0.1f));
    active_support.updateMesh(*graph.mesh(), offsets);
    updater.update(1, makeInput({}), active_support, offsets, graph);
  }

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(attrs.is_active);
  const auto missing_metadata = attrs.metadata.get();
  EXPECT_NE(missing_metadata, metadata_before);
  ASSERT_TRUE(missing_metadata.contains("hydra_tracked_object"));
  const auto& missing_tracked_metadata = missing_metadata.at("hydra_tracked_object");
  EXPECT_FALSE(missing_tracked_metadata.at("is_tracked").get<bool>());

  {
    auto active_support =
        makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                  11u,
                  Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                  Eigen::Vector3f::Constant(0.1f));
    active_support.updateMesh(*graph.mesh(), offsets);
    updater.update(
        2, makeInput({makeObservation(11u, 1u)}), active_support, offsets, graph);
  }

  const auto restored_metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(restored_metadata.contains("hydra_tracked_object"));
  const auto& restored_tracked_metadata = restored_metadata.at("hydra_tracked_object");
  EXPECT_TRUE(restored_tracked_metadata.at("is_tracked").get<bool>());
}

TEST(TrackGraphSegmentUpdater, RefreshesCurrentSupportForActiveTrack) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    addPoints(initial, 11u, {4, 5, 6}, Eigen::Vector3f::Constant(0.1f));
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);
  }

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 16; ++i) {
    remap[i] = i;
  }

  const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 16, 0, remap);
  MeshDelta relabeled(tracking);
  addPoints(relabeled, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  addPoints(relabeled, 22u, {4, 5, 6}, Eigen::Vector3f::Constant(0.1f));
  relabeled.updateMesh(*graph.mesh(), offsets);
  updater.update(1,
                 makeInput({makeObservation(11u, 1u), makeObservation(22u, 2u)}),
                 relabeled,
                 offsets,
                 graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  ASSERT_TRUE(graph.hasNode("O1"_id));

  const auto& old_attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  const auto& new_attrs = graph.getNode("O1"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(old_attrs.is_active);
  EXPECT_EQ(
      std::vector<size_t>(old_attrs.mesh_connections.begin(),
                          old_attrs.mesh_connections.end()),
      (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(std::vector<size_t>(new_attrs.mesh_connections.begin(),
                                new_attrs.mesh_connections.end()),
            (std::vector<size_t>{8, 9, 10, 11, 12, 13, 14, 15}));
}

TEST(TrackGraphSegmentUpdater, GeometryUsesConfirmedSupportWhenAvailable) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto no_confirmed_ids = makeTrackIds({});
  const auto no_confirmed_likelihoods = makeTrackLikelihoods({});
  const auto supported_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto outlier_offset = Eigen::Vector3f(3.0f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  MeshDelta initial({0, 0, 0});
  addPoints(initial,
            11u,
            supported_offset,
            scale,
            &track_11_ids,
            &track_11_likelihoods,
            &track_11_ids,
            &track_11_likelihoods);
  addPoints(initial,
            11u,
            outlier_offset,
            scale,
            &track_11_ids,
            &track_11_likelihoods,
            &no_confirmed_ids,
            &no_confirmed_likelihoods);
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(
      0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_NEAR(attrs.position.x(), supported_offset.x(), 1.0e-5);
  EXPECT_NEAR(attrs.bounding_box.world_P_center.x(), supported_offset.x(), 1.0e-5);
  EXPECT_LT(attrs.bounding_box.dimensions.x(), 0.5f);
  EXPECT_EQ(attrs.mesh_connections.size(), 8u);
}

TEST(TrackGraphSegmentUpdater, GeometryDoesNotFallbackToUnconfirmedSupport) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto no_confirmed_ids = makeTrackIds({});
  const auto no_confirmed_likelihoods = makeTrackLikelihoods({});
  const auto supported_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto outlier_offset = Eigen::Vector3f(3.0f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  MeshDelta initial({0, 0, 0});
  addPoints(initial,
            11u,
            supported_offset,
            scale,
            &track_11_ids,
            &track_11_likelihoods,
            &track_11_ids,
            &track_11_likelihoods);
  addPoints(initial,
            11u,
            outlier_offset,
            scale,
            &track_11_ids,
            &track_11_likelihoods,
            &no_confirmed_ids,
            &no_confirmed_likelihoods);
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(
      0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_NEAR(attrs.position.x(), supported_offset.x(), 1.0e-5);
  EXPECT_NEAR(attrs.bounding_box.world_P_center.x(), supported_offset.x(), 1.0e-5);
  EXPECT_LT(attrs.bounding_box.dimensions.x(), 0.5f);
  EXPECT_EQ(attrs.mesh_connections.size(), 8u);
}

TEST(TrackGraphSegmentUpdater,
     ActiveDeltaConfirmedSupportCreatesObjectWithoutPersistentMeshTrackData) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto supported_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  MeshDelta initial({0, 0, 0});
  addPoints(initial,
            11u,
            supported_offset,
            scale,
            nullptr,
            nullptr,
            &track_11_ids,
            &track_11_likelihoods);
  initial.updateMesh(*graph.mesh(), offsets);

  updater.update(0,
                 makeInput({makeObservation(11u, 1u, prototype)}, true),
                 initial,
                 offsets,
                 graph);
  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.mesh()->has_confirmed_track_data);
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_NEAR(attrs.position.x(), supported_offset.x(), 1.0e-5);
  EXPECT_NEAR(attrs.bounding_box.world_P_center.x(), supported_offset.x(), 1.0e-5);
  EXPECT_LT(attrs.bounding_box.dimensions.x(), 0.5f);
  EXPECT_EQ(attrs.mesh_connections.size(), 8u);
}

TEST(TrackGraphSegmentUpdater,
     RawWinnerSupportWithoutActiveDeltaConfirmedDoesNotCreateObject) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();

  {
    MeshDelta propagation_only({0, 0, 0});
    addRawLabeledPointsWithoutConfirmed(
        propagation_only, 22u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    propagation_only.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(22u, 2u, prototype)}, false),
                   propagation_only,
                   offsets,
                   graph);
  }

  EXPECT_EQ(graph.getLayer(DsgLayers::OBJECTS).numNodes(), 0u);

  {
    const auto track_22_ids = makeTrackIds({22u});
    const auto track_22_likelihoods = makeTrackLikelihoods({4.0f});
    MeshDelta confirmed(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(confirmed,
              22u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_22_ids,
              &track_22_likelihoods);
    confirmed.updateMesh(*graph.mesh(), offsets);

    updater.update(1,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   confirmed,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{22u}));
}

TEST(TrackGraphSegmentUpdater,
     ExistingOwnedTrackAddsRawWinnerSupportWithoutConfirmedEvidence) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.enable_owned_winner_support =
      TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();

  {
    const auto track_11_ids = makeTrackIds({11u});
    const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
    MeshDelta confirmed({0, 0, 0});
    addPoints(confirmed,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    confirmed.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   confirmed,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto initial_metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();

  EXPECT_EQ(
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().mesh_connections.size(),
      8u);

  {
    std::map<size_t, size_t> remap;
    for (size_t i = 0; i < 8; ++i) {
      remap[i] = i;
    }

    const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap);
    MeshDelta raw_only(tracking);
    addRawLabeledPointsWithoutConfirmed(
        raw_only, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    addRawLabeledPointsWithoutConfirmed(
        raw_only, 11u, {3, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    raw_only.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype)}, false),
                   raw_only,
                   offsets,
                   graph);
  }

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_EQ(attrs.mesh_connections.size(), 16u);
  EXPECT_NEAR(attrs.position.x(), 2.0f, 1.0e-5);
  const auto metadata = attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater, DefaultKeepsOwnedRawWinnerSupportOutOfMeshConnections) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();

  {
    const auto track_11_ids = makeTrackIds({11u});
    const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
    MeshDelta confirmed({0, 0, 0});
    addPoints(confirmed,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    confirmed.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   confirmed,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto initial_metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();

  {
    std::map<size_t, size_t> remap;
    for (size_t i = 0; i < 8; ++i) {
      remap[i] = i;
    }

    const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap);
    MeshDelta raw_only(tracking);
    addRawLabeledPointsWithoutConfirmed(
        raw_only, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    addRawLabeledPointsWithoutConfirmed(
        raw_only, 11u, {3, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    raw_only.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype)}, false),
                   raw_only,
                   offsets,
                   graph);
  }

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_EQ(attrs.mesh_connections.size(), 8u);
  EXPECT_NEAR(attrs.position.x(), 1.0f, 1.0e-5);
  const auto metadata = attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater, RawWinnerAdmissionCreatesFromRawWinnerSupport) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.enable_owned_winner_support =
      TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND;
  config.raw_winner_admission.enabled = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_ids = makeTrackIds({5317u});
  const auto track_likelihoods = makeTrackLikelihoods({4.0f});

  MeshDelta raw_only({0, 0, 0});
  addRawWinnerPointsWithoutConfirmed(raw_only,
                                     5317u,
                                     {1, 2, 3},
                                     Eigen::Vector3f::Constant(0.1f),
                                     track_ids,
                                     track_likelihoods);
  raw_only.updateMesh(*graph.mesh(), offsets);

  updater.update(0,
                 makeInput({makeObservation(5317u, 1u, prototype)}, true),
                 raw_only,
                 offsets,
                 graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{5317u}));
  EXPECT_EQ(
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().mesh_connections.size(),
      8u);
}

TEST(TrackGraphSegmentUpdater,
     RawWinnerAdmissionKeepsRawWinnerSupportBelowThresholdProvisional) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 12;
  config.enable_owned_winner_support =
      TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND;
  config.raw_winner_admission.enabled = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_ids = makeTrackIds({5317u});
  const auto track_likelihoods = makeTrackLikelihoods({4.0f});

  MeshDelta raw_only({0, 0, 0});
  addRawWinnerPointsWithoutConfirmed(raw_only,
                                     5317u,
                                     {1, 2, 3},
                                     Eigen::Vector3f::Constant(0.1f),
                                     track_ids,
                                     track_likelihoods);
  raw_only.updateMesh(*graph.mesh(), offsets);

  updater.update(0,
                 makeInput({makeObservation(5317u, 1u, prototype)}, true),
                 raw_only,
                 offsets,
                 graph);

  EXPECT_EQ(graph.getLayer(DsgLayers::OBJECTS).numNodes(), 0u);
}

TEST(TrackGraphSegmentUpdater, RawWinnerAdmissionStillPrefersConfirmedCreationSupport) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.enable_owned_winner_support =
      TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND;
  config.raw_winner_admission.enabled = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_ids = makeTrackIds({44u});
  const auto track_likelihoods = makeTrackLikelihoods({4.0f});

  MeshDelta mixed({0, 0, 0});
  addRawWinnerPointsWithoutConfirmed(mixed,
                                     44u,
                                     {1, 2, 3},
                                     Eigen::Vector3f::Constant(0.1f),
                                     track_ids,
                                     track_likelihoods);
  addPoints(mixed,
            44u,
            {3, 2, 3},
            Eigen::Vector3f::Constant(0.1f),
            nullptr,
            nullptr,
            &track_ids,
            &track_likelihoods);
  mixed.updateMesh(*graph.mesh(), offsets);

  updater.update(
      0, makeInput({makeObservation(44u, 4u, prototype)}, true), mixed, offsets, graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_EQ(attrs.mesh_connections.size(), 8u);
  EXPECT_NEAR(attrs.position.x(), 3.0f, 1.0e-5);
}

TEST(TrackGraphSegmentUpdater, ActiveDeltaConfirmedSupportCreatesObjectOnNonKeyframe) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_33_ids = makeTrackIds({33u});
  const auto track_33_likelihoods = makeTrackLikelihoods({4.0f});

  MeshDelta confirmed_mesh({0, 0, 0});
  addPoints(confirmed_mesh,
            33u,
            {1, 2, 3},
            Eigen::Vector3f::Constant(0.1f),
            nullptr,
            nullptr,
            &track_33_ids,
            &track_33_likelihoods);
  confirmed_mesh.updateMesh(*graph.mesh(), offsets);

  updater.update(0,
                 makeInput({makeObservation(33u, 3u, prototype)}, false),
                 confirmed_mesh,
                 offsets,
                 graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_EQ(attrs.mesh_connections.size(), 8u);
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{33u}));
}

TEST(TrackGraphSegmentUpdater,
     ActiveDeltaAdoptsContainedFragmentBeforeCreatingSeparateFragmentNode) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 1;
  config.reidentifier.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});
  const auto offset_a = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto offset_b = Eigen::Vector3f(1.4f, 2.0f, 3.0f);
  const auto offset_c = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              offset_a,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    addPoints(initial,
              11u,
              offset_b,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    addPoints(initial,
              11u,
              offset_c,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);

    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));

  {
    MeshDelta fragment(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(fragment,
              11u,
              offset_a,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    addPoints(fragment,
              22u,
              offset_b,
              scale,
              nullptr,
              nullptr,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    addPoints(fragment,
              11u,
              offset_c,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    fragment.updateMesh(*graph.mesh(), offsets);

    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)},
                             true),
                   fragment,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater,
     DisabledActiveWindowAdoptionCreatesSeparateFragmentNode) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.enable_active_window_adoption = false;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = false;
  config.reidentifier.min_shared_vertices = 1;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});
  const auto offset_a = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto offset_b = Eigen::Vector3f(1.4f, 2.0f, 3.0f);
  const auto offset_c = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              offset_a,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    addPoints(initial,
              11u,
              offset_b,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    addPoints(initial,
              11u,
              offset_c,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);

    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));

  {
    MeshDelta fragment(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(fragment,
              11u,
              offset_a,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    addPoints(fragment,
              22u,
              offset_b,
              scale,
              nullptr,
              nullptr,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    addPoints(fragment,
              11u,
              offset_c,
              scale,
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    fragment.updateMesh(*graph.mesh(), offsets);

    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)},
                             true),
                   fragment,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  ASSERT_TRUE(graph.hasNode("O1"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O1"_id), (std::vector<uint32_t>{22u}));

  const auto root_metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(root_metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater, ActiveDeltaAdoptsConfirmedNonWinnerWithoutGeometry) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 4;
  config.reidentifier.min_observation_frames = 1;
  config.reidentifier.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);

    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));

  {
    MeshDelta relabeled(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(relabeled,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);

    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)},
                             true),
                   relabeled,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater, RawWinnerOverlapAdoptionHonorsLikelihoodRatioThreshold) {
  auto run_candidate = [](float candidate_likelihood) {
    TrackGraphSegmentUpdater::Config config;
    config.min_points_create = 4;
    config.enable_owned_winner_support =
        TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND;
    config.raw_winner_admission.enabled = true;
    config.raw_winner_admission.min_likelihood_ratio = 0.3;
    config.reidentifier.enabled = true;
    config.reidentifier.min_shared_vertices = 1;
    config.reidentifier.min_observation_frames = 1;
    config.reidentifier.debug_decisions = true;
    TrackGraphSegmentUpdater updater(config);

    DynamicSceneGraph graph;
    graph.setMesh(createMesh(true));
    kimera_pgmo::MeshOffsetInfo offsets;
    const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
    const auto track_11_ids = makeTrackIds({11u});
    const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
    const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
    const auto track_22_with_11_likelihoods =
        makeTrackLikelihoods({100.0f, candidate_likelihood});

    {
      MeshDelta initial({0, 0, 0});
      addPoints(initial,
                11u,
                {1, 2, 3},
                Eigen::Vector3f::Constant(0.1f),
                nullptr,
                nullptr,
                &track_11_ids,
                &track_11_likelihoods);
      initial.updateMesh(*graph.mesh(), offsets);

      updater.update(0,
                     makeInput({makeObservation(11u, 1u, prototype)}, true),
                     initial,
                     offsets,
                     graph);
    }

    MeshDelta relabeled(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addRawWinnerPointsWithoutConfirmed(relabeled,
                                       22u,
                                       {1, 2, 3},
                                       Eigen::Vector3f::Constant(0.1f),
                                       track_22_with_11_ids,
                                       track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);

    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)},
                             true),
                   relabeled,
                   offsets,
                   graph);

    return std::make_pair(
        graph.hasNode("O1"_id),
        graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get());
  };

  const auto below = run_candidate(29.0f);
  EXPECT_TRUE(below.first);

  const auto at_threshold = run_candidate(30.0f);
  EXPECT_FALSE(at_threshold.first);
  const auto& tracked = at_threshold.second.at("hydra_tracked_object");
  EXPECT_EQ(tracked.at("source_track_ids").get<std::vector<uint32_t>>(),
            (std::vector<uint32_t>{11u, 22u}));
}

TEST(TrackGraphSegmentUpdater, RawWinnerOverlapIgnoresLowLikelihoodNeighborLeak) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.enable_owned_winner_support =
      TrackGraphSegmentUpdater::OwnedWinnerSupportMode::WINNER_APPEND;
  config.raw_winner_admission.enabled = true;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 1;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto leaked_track_22_ids = makeTrackIds({22u, 11u});
  const auto leaked_track_22_likelihoods = makeTrackLikelihoods({100.0f, 5.0f});

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }

  MeshDelta relabeled(makeIdentityRemapTracking(graph.mesh()->numVertices()));
  addRawWinnerPointsWithoutConfirmed(relabeled,
                                     22u,
                                     {1.4f, 2, 3},
                                     Eigen::Vector3f::Constant(0.1f),
                                     leaked_track_22_ids,
                                     leaked_track_22_likelihoods);
  relabeled.updateMesh(*graph.mesh(), offsets);
  updater.update(1,
                 makeInput({makeObservation(11u, 1u, prototype),
                            makeObservation(22u, 2u, prototype)},
                           true),
                 relabeled,
                 offsets,
                 graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  ASSERT_TRUE(graph.hasNode("O1"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O1"_id), (std::vector<uint32_t>{22u}));
}

TEST(TrackGraphSegmentUpdater, ActiveDeltaRejectsAdoptionBelowMinSharedVertices) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 9;
  config.reidentifier.min_observation_frames = 1;
  config.reidentifier.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({4.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);

    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));

  {
    MeshDelta candidate(
        MeshDelta::TrackingInfo::with_remap(0, graph.mesh()->numVertices(), 0, {}));
    addPoints(candidate,
              22u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    candidate.updateMesh(*graph.mesh(), offsets);

    updater.update(1,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   candidate,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  ASSERT_TRUE(graph.hasNode("O1"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O1"_id), (std::vector<uint32_t>{22u}));

  const auto metadata =
      graph.getNode("O1"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater, ActiveDeltaUsesConfirmedSupportAcrossEntireActiveMesh) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 12;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  MeshDelta delta({0, 0, 0});
  addPoints(delta, 44u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  addPoints(delta, 44u, {1.4f, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  delta.updateMesh(*graph.mesh(), offsets);

  updater.update(0,
                 makeInput({makeObservation(44u, 4u, prototype)}, false),
                 delta,
                 offsets,
                 graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_EQ(attrs.mesh_connections.size(), 16u);
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{44u}));
}

TEST(TrackGraphSegmentUpdater,
     ActiveDeltaKeepsRawOnlyTrackProvisionalWithoutConfirmedEvidence) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();

  MeshDelta raw_only({0, 0, 0});
  addRawLabeledPointsWithoutConfirmed(
      raw_only, 5317u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  raw_only.updateMesh(*graph.mesh(), offsets);

  updater.update(0,
                 makeInput({makeObservation(5317u, 1u, prototype)}, true),
                 raw_only,
                 offsets,
                 graph);

  EXPECT_EQ(graph.getLayer(DsgLayers::OBJECTS).numNodes(), 0u);
}

TEST(TrackGraphSegmentUpdater,
     ProvisionalKeyframesSurviveNoMeshEvidenceBeforeCreation) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(false));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_77_ids = makeTrackIds({77u});
  const auto track_77_likelihoods = makeTrackLikelihoods({4.0f});

  {
    MeshDelta empty_delta({0, 0, 0});
    updater.update(0,
                   makeInput({makeObservation(77u, 7u, prototype)}, true),
                   empty_delta,
                   offsets,
                   graph);
  }
  EXPECT_EQ(graph.getLayer(DsgLayers::OBJECTS).numNodes(), 0u);

  {
    MeshDelta confirmed({0, 0, 0});
    addPoints(confirmed,
              77u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              nullptr,
              nullptr,
              &track_77_ids,
              &track_77_likelihoods);
    confirmed.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(77u, 7u, prototype)}, true),
                   confirmed,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
}

TEST(TrackGraphSegmentUpdater, PropagationFrameCreationSeedsOneKeyframeObservation) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta delta({0, 0, 0});
  addPoints(delta, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  delta.updateMesh(*graph.mesh(), offsets);

  updater.update(
      0, makeInput({makeObservation(11u, 1u)}, false), delta, offsets, graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));
}

TEST(TrackGraphSegmentUpdater, PropagationOnlyRunnerUpDoesNotCreatePartOfEdge) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 1;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({3.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});
  const auto track_22_confirmed_ids = makeTrackIds({22u});
  const auto track_22_confirmed_likelihoods = makeTrackLikelihoods({4.0f});
  const auto offset_a = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto offset_b = Eigen::Vector3f(1.4f, 2.0f, 3.0f);
  const auto offset_c = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(initial, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(
        0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);
  }

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 16; ++i) {
    remap[i] = i;
  }
  const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 16, 0, remap);

  MeshDelta relabeled(tracking);
  addPoints(relabeled, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
  addPoints(relabeled,
            22u,
            offset_b,
            scale,
            &track_22_with_11_ids,
            &track_22_with_11_likelihoods,
            &track_22_confirmed_ids,
            &track_22_confirmed_likelihoods);
  addPoints(relabeled, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
  relabeled.updateMesh(*graph.mesh(), offsets);
  updater.update(1,
                 makeInput({makeObservation(11u, 1u, prototype),
                            makeObservation(22u, 2u, prototype)}),
                 relabeled,
                 offsets,
                 graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  if (!graph.hasNode("O1"_id)) {
    EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
  }
  EXPECT_FALSE(graph.hasEdge("O0"_id, "O1"_id));
  EXPECT_FALSE(graph.hasEdge("O1"_id, "O0"_id));
}

TEST(TrackGraphSegmentUpdater, ReidentifiesReturningTrackIntoInactiveObject) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 3;
  config.reidentifier.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({3.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(
        0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);
  }

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 8; ++i) {
    remap[i] = i;
  }
  const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap);
  {
    auto steady = makeDelta(tracking,
                            11u,
                            Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                            Eigen::Vector3f::Constant(0.1f),
                            &track_11_ids,
                            &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(
        1, makeInput({makeObservation(11u, 1u, prototype)}), steady, offsets, graph);
  }
  {
    auto steady = makeDelta(tracking,
                            11u,
                            Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                            Eigen::Vector3f::Constant(0.1f),
                            &track_11_ids,
                            &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(
        2, makeInput({makeObservation(11u, 1u, prototype)}), steady, offsets, graph);
  }
  {
    auto steady = makeDelta(tracking,
                            11u,
                            Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                            Eigen::Vector3f::Constant(0.1f),
                            &track_11_ids,
                            &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(3, makeInput({}), steady, offsets, graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_TRUE(graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().is_active);

  {
    auto relabeled = makeDelta(tracking,
                               22u,
                               Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                               Eigen::Vector3f::Constant(0.1f),
                               &track_22_with_11_ids,
                               &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(
        4, makeInput({makeObservation(22u, 2u, prototype)}), relabeled, offsets, graph);
  }
  {
    auto relabeled = makeDelta(tracking,
                               22u,
                               Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                               Eigen::Vector3f::Constant(0.1f),
                               &track_22_with_11_ids,
                               &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(
        5, makeInput({makeObservation(22u, 2u, prototype)}), relabeled, offsets, graph);
  }
  {
    auto relabeled = makeDelta(tracking,
                               22u,
                               Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                               Eigen::Vector3f::Constant(0.1f),
                               &track_22_with_11_ids,
                               &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(
        6, makeInput({makeObservation(22u, 2u, prototype)}), relabeled, offsets, graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(attrs.is_active);
  const auto metadata = attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));

  const auto track_22_ids = makeTrackIds({22u});
  const auto track_22_likelihoods = makeTrackLikelihoods({4.0f});
  const auto second_offset = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  auto revisited = makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                             22u,
                             second_offset,
                             Eigen::Vector3f::Constant(0.1f),
                             &track_22_ids,
                             &track_22_likelihoods);
  revisited.updateMesh(*graph.mesh(), offsets);
  updater.update(
      7, makeInput({makeObservation(22u, 2u, prototype)}), revisited, offsets, graph);

  const auto& updated_attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_GT(updated_attrs.position.x(), 1.1);
}

TEST(TrackGraphSegmentUpdater, ContainedFragmentAlwaysMergesAndRecordsPartOfReason) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 3;
  config.reidentifier.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({3.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});
  const auto offset_a = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto offset_b = Eigen::Vector3f(1.4f, 2.0f, 3.0f);
  const auto offset_c = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(initial, 11u, offset_b, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(initial, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(
        0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);
  }

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 24; ++i) {
    remap[i] = i;
  }
  const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 24, 0, remap);
  {
    MeshDelta steady(tracking);
    addPoints(steady, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(steady, 11u, offset_b, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(steady, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(
        1, makeInput({makeObservation(11u, 1u, prototype)}), steady, offsets, graph);
  }
  {
    MeshDelta steady(tracking);
    addPoints(steady, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(steady, 11u, offset_b, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(steady, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(
        2, makeInput({makeObservation(11u, 1u, prototype)}), steady, offsets, graph);
  }

  {
    MeshDelta relabeled(tracking);
    addPoints(relabeled, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(relabeled,
              22u,
              offset_b,
              scale,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    addPoints(relabeled, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(3,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)}),
                   relabeled,
                   offsets,
                   graph);
  }

  {
    MeshDelta relabeled(tracking);
    addPoints(relabeled, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(relabeled,
              22u,
              offset_b,
              scale,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    addPoints(relabeled, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(4,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)}),
                   relabeled,
                   offsets,
                   graph);
  }
  {
    MeshDelta relabeled(tracking);
    addPoints(relabeled, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(relabeled,
              22u,
              offset_b,
              scale,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    addPoints(relabeled, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(5,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)}),
                   relabeled,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));
  EXPECT_FALSE(graph.hasEdge("O0"_id, "O1"_id));

  const auto& parent_attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(parent_attrs.is_active);

  const auto parent_metadata = parent_attrs.metadata.get();
  ASSERT_TRUE(parent_metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
}

TEST(TrackGraphSegmentUpdater, MergesContainedNewFragmentWhenConfigured) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 3;
  config.merge_on_partof_match = true;
  config.reidentifier.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({3.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});
  const auto offset_a = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto offset_b = Eigen::Vector3f(1.4f, 2.0f, 3.0f);
  const auto offset_c = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(initial, 11u, offset_b, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(initial, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(
        0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);
  }

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 24; ++i) {
    remap[i] = i;
  }
  const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 24, 0, remap);
  for (uint64_t timestamp = 1; timestamp <= 2; ++timestamp) {
    MeshDelta steady(tracking);
    addPoints(steady, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(steady, 11u, offset_b, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(steady, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp,
                   makeInput({makeObservation(11u, 1u, prototype)}),
                   steady,
                   offsets,
                   graph);
  }

  for (uint64_t timestamp = 3; timestamp <= 5; ++timestamp) {
    MeshDelta relabeled(tracking);
    addPoints(relabeled, 11u, offset_a, scale, &track_11_ids, &track_11_likelihoods);
    addPoints(relabeled,
              22u,
              offset_b,
              scale,
              &track_22_with_11_ids,
              &track_22_with_11_likelihoods);
    addPoints(relabeled, 11u, offset_c, scale, &track_11_ids, &track_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(22u, 2u, prototype)}),
                   relabeled,
                   offsets,
                   graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  const auto metadata = attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
}

TEST(TrackGraphSegmentUpdater, LateOpenVocabFeatureAttachesWithoutObservations) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.open_vocab.enabled = true;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  TrackGraphSegmentUpdater updater(config, cache);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta initial({0, 0, 0});
  addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);

  ASSERT_TRUE(graph.hasNode("O0"_id));
  auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_EQ(attrs.open_vocab_features.cols(), 0);
  EXPECT_FALSE(attrs.metadata.get()
                   .at("hydra_tracked_object")
                   .contains("open_vocab_encoder_id"));

  const auto feature = (Eigen::Vector3f() << 1.0f, 2.0f, 3.0f).finished();
  cache->store(11u, makeOpenVocabEntry(feature, 1, 1, 1));

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 8; ++i) {
    remap[i] = i;
  }
  MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap));
  unchanged.updateMesh(*graph.mesh(), offsets);
  updater.update(1, makeInput({}), unchanged, offsets, graph);

  const auto& updated_attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  ASSERT_EQ(updated_attrs.open_vocab_features.rows(), 3);
  ASSERT_EQ(updated_attrs.open_vocab_features.cols(), 1);
  EXPECT_TRUE(updated_attrs.open_vocab_features.col(0).isApprox(feature));
  const auto metadata = updated_attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  ASSERT_TRUE(metadata.at("hydra_tracked_object")
                  .contains("open_vocab_feature_source_track_ids"));
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_feature_source_track_ids"),
      nlohmann::json::array({11u}));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_encoder_id"),
            "openclip:ViT-L-14:laion2b_s32b_b82k:precision=fp16");
}

TEST(TrackGraphSegmentUpdater, LateOpenVocabFeatureRefreshReplacesExistingColumn) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.open_vocab.enabled = true;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  TrackGraphSegmentUpdater updater(config, cache);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta initial({0, 0, 0});
  addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);

  auto makeUnchangedDelta = [&]() {
    const size_t num_vertices = graph.mesh()->numVertices();
    std::map<size_t, size_t> remap;
    for (size_t i = 0; i < num_vertices; ++i) {
      remap[i] = i;
    }
    MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, num_vertices, 0, remap));
    unchanged.updateMesh(*graph.mesh(), offsets);
    return unchanged;
  };

  const auto first_feature = (Eigen::Vector3f() << 0.0f, 1.0f, 0.0f).finished();
  cache->store(11u, makeOpenVocabEntry(first_feature, 1, 1, 1));
  ASSERT_NO_THROW({
    auto unchanged = makeUnchangedDelta();
    updater.update(1, makeInput({}), unchanged, offsets, graph);
  });

  {
    const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
    ASSERT_EQ(attrs.open_vocab_features.rows(), 3);
    ASSERT_EQ(attrs.open_vocab_features.cols(), 1);
    EXPECT_TRUE(attrs.open_vocab_features.col(0).isApprox(first_feature));
    const auto metadata = attrs.metadata.get();
    ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
    EXPECT_EQ(
        metadata.at("hydra_tracked_object").at("open_vocab_feature_source_track_ids"),
        nlohmann::json::array({11u}));
  }

  const auto refreshed_feature = (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished();
  cache->store(11u, makeOpenVocabEntry(refreshed_feature, 2, 2, 1));
  EXPECT_EQ(cache->size(), 1u);
  ASSERT_NO_THROW({
    auto unchanged = makeUnchangedDelta();
    updater.update(2, makeInput({}), unchanged, offsets, graph);
  });
  EXPECT_EQ(cache->size(), 0u);

  const auto& refreshed_attrs =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  ASSERT_EQ(refreshed_attrs.open_vocab_features.rows(), 3);
  ASSERT_EQ(refreshed_attrs.open_vocab_features.cols(), 1);
  EXPECT_TRUE(refreshed_attrs.open_vocab_features.col(0).isApprox(refreshed_feature));
  const auto refreshed_metadata = refreshed_attrs.metadata.get();
  ASSERT_TRUE(refreshed_metadata.contains("hydra_tracked_object"));
  const auto& tracked = refreshed_metadata.at("hydra_tracked_object");
  EXPECT_EQ(tracked.at("open_vocab_feature_source_track_ids"),
            nlohmann::json::array({11u}));
}

TEST(TrackGraphSegmentUpdater, LateOpenVocabFeatureRefreshRecomputesIgnoreState) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.open_vocab.enabled = true;
  config.open_vocab.ignore_filter.enabled = true;
  config.open_vocab.ignore_filter.ignore_score_threshold = 0.70f;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  auto bank = makeIgnorePromptBank(
      {{"wall", (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished()}});
  TrackGraphSegmentUpdater updater(config, cache, bank);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta initial({0, 0, 0});
  addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);

  auto makeUnchangedDelta = [&]() {
    const size_t num_vertices = graph.mesh()->numVertices();
    std::map<size_t, size_t> remap;
    for (size_t i = 0; i < num_vertices; ++i) {
      remap[i] = i;
    }
    MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, num_vertices, 0, remap));
    unchanged.updateMesh(*graph.mesh(), offsets);
    return unchanged;
  };

  cache->store(
      11u,
      makeOpenVocabEntry((Eigen::Vector3f() << 0.0f, 1.0f, 0.0f).finished(), 1, 1, 1));
  ASSERT_NO_THROW({
    auto unchanged = makeUnchangedDelta();
    updater.update(1, makeInput({}), unchanged, offsets, graph);
  });

  {
    const auto metadata =
        graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
    EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_base_state"),
              "keep");
    EXPECT_FALSE(trackedIgnoreEffective(graph, "O0"_id));
  }

  const auto ignore_feature = (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished();
  cache->store(11u, makeOpenVocabEntry(ignore_feature, 2, 2, 1));
  ASSERT_NO_THROW({
    auto unchanged = makeUnchangedDelta();
    updater.update(2, makeInput({}), unchanged, offsets, graph);
  });

  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  ASSERT_EQ(attrs.open_vocab_features.rows(), 3);
  ASSERT_EQ(attrs.open_vocab_features.cols(), 1);
  EXPECT_TRUE(attrs.open_vocab_features.col(0).isApprox(ignore_feature));

  const auto metadata = attrs.metadata.get();
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_feature_source_track_ids"),
      nlohmann::json::array({11u}));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_base_state"),
            "ignore");
  EXPECT_TRUE(trackedIgnoreEffective(graph, "O0"_id));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_best_prompt"),
            "wall");
  EXPECT_FLOAT_EQ(metadata.at("hydra_tracked_object")
                      .at("open_vocab_ignore_best_score")
                      .get<float>(),
                  1.0f);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("ignore"),
      1u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("keep"), 0u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("unknown"),
      0u);
}

TEST(TrackGraphSegmentUpdater,
     IgnoreFilterMarksObjectIgnoredWhenLateFeatureCrossesThreshold) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.open_vocab.enabled = true;
  config.open_vocab.ignore_filter.enabled = true;
  config.open_vocab.ignore_filter.ignore_score_threshold = 0.70f;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  auto bank = makeIgnorePromptBank(
      {{"wall", (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished()}});
  TrackGraphSegmentUpdater updater(config, cache, bank);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta initial({0, 0, 0});
  addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);

  cache->store(
      11u,
      makeOpenVocabEntry((Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished(), 1, 1, 1));

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 8; ++i) {
    remap[i] = i;
  }
  MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap));
  unchanged.updateMesh(*graph.mesh(), offsets);
  updater.update(1, makeInput({}), unchanged, offsets, graph);

  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_base_state"),
            "ignore");
  EXPECT_TRUE(trackedIgnoreEffective(graph, "O0"_id));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_best_prompt"),
            "wall");
  EXPECT_FLOAT_EQ(metadata.at("hydra_tracked_object")
                      .at("open_vocab_ignore_best_score")
                      .get<float>(),
                  1.0f);
  ASSERT_TRUE(metadata.at("hydra_tracked_object").contains("open_vocab_ignore_votes"));
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("ignore"),
      1u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("keep"), 0u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("unknown"),
      0u);
}

TEST(TrackGraphSegmentUpdater, IgnoreFilterLeavesEncoderMismatchUnknown) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.open_vocab.enabled = true;
  config.open_vocab.ignore_filter.enabled = true;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  auto bank = makeIgnorePromptBank(
      {{"wall", (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished()}});
  TrackGraphSegmentUpdater updater(config, cache, bank);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  MeshDelta initial({0, 0, 0});
  addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
  initial.updateMesh(*graph.mesh(), offsets);
  updater.update(0, makeInput({makeObservation(11u, 1u)}), initial, offsets, graph);

  auto entry =
      makeOpenVocabEntry((Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished(), 1, 1, 1);
  entry.encoder_id = "other";
  cache->store(11u, entry);

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 8; ++i) {
    remap[i] = i;
  }
  MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap));
  unchanged.updateMesh(*graph.mesh(), offsets);
  updater.update(1, makeInput({}), unchanged, offsets, graph);

  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_base_state"),
            "unknown");
  EXPECT_FALSE(trackedIgnoreEffective(graph, "O0"_id));
  EXPECT_FALSE(
      metadata.at("hydra_tracked_object").contains("open_vocab_ignore_best_prompt"));
  EXPECT_FALSE(
      metadata.at("hydra_tracked_object").contains("open_vocab_ignore_best_score"));
  ASSERT_TRUE(metadata.at("hydra_tracked_object").contains("open_vocab_ignore_votes"));
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("ignore"),
      0u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("keep"), 0u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("unknown"),
      1u);
}

TEST(TrackGraphSegmentUpdater, IgnoreFilterDoesNotPropagateWithoutRuntimePartOfEdges) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.open_vocab.enabled = true;
  config.open_vocab.ignore_filter.enabled = true;
  config.open_vocab.ignore_filter.ignore_score_threshold = 0.70f;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  auto bank = makeIgnorePromptBank(
      {{"wall", (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished()}});
  TrackGraphSegmentUpdater updater(config, cache, bank);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, {1, 2, 3}, Eigen::Vector3f::Constant(0.1f));
    addPoints(initial, 22u, {4, 5, 6}, Eigen::Vector3f::Constant(0.1f));
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u), makeObservation(22u, 2u)}),
                   initial,
                   offsets,
                   graph);
  }

  NodeId parent_node = 0;
  NodeId child_node = 0;
  for (const auto& [node_id, node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto& attrs = node->attributes<ObjectNodeAttributes>();
    if (attrs.name == "track_11") {
      parent_node = node_id;
    } else if (attrs.name == "track_22") {
      child_node = node_id;
    }
  }
  ASSERT_NE(parent_node, NodeId(0));
  ASSERT_NE(child_node, NodeId(0));

  cache->store(
      11u,
      makeOpenVocabEntry((Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished(), 1, 1, 1));
  cache->store(
      22u,
      makeOpenVocabEntry((Eigen::Vector3f() << 0.0f, 1.0f, 0.0f).finished(), 1, 1, 2));

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 16; ++i) {
    remap[i] = i;
  }
  MeshDelta unchanged(MeshDelta::TrackingInfo::with_remap(0, 16, 0, remap));
  unchanged.updateMesh(*graph.mesh(), offsets);
  updater.update(1, makeInput({}), unchanged, offsets, graph);

  const auto parent_metadata =
      graph.getNode(parent_node).attributes<ObjectNodeAttributes>().metadata.get();
  const auto child_metadata =
      graph.getNode(child_node).attributes<ObjectNodeAttributes>().metadata.get();
  const auto& parent_tracked = parent_metadata.at("hydra_tracked_object");
  const auto& child_tracked = child_metadata.at("hydra_tracked_object");
  EXPECT_EQ(parent_tracked.at("open_vocab_ignore_base_state"), "ignore");
  EXPECT_TRUE(trackedIgnoreEffective(graph, parent_node));
  EXPECT_EQ(child_tracked.at("open_vocab_ignore_base_state"), "keep");
  EXPECT_FALSE(trackedIgnoreEffective(graph, child_node));
}

TEST(TrackGraphSegmentUpdater, MergedObjectPreservesOpenVocabFeaturesBySourceTrackId) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 3;
  config.open_vocab.enabled = true;
  config.open_vocab.ignore_filter.enabled = true;
  config.open_vocab.ignore_filter.ignore_score_threshold = 0.70f;
  auto cache = std::make_shared<OpenVocabFeatureCache>();
  auto bank = makeIgnorePromptBank(
      {{"wall", (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished()}});
  TrackGraphSegmentUpdater updater(config, cache, bank);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto feature_11 = (Eigen::Vector3f() << 1.0f, 0.0f, 0.0f).finished();
  const auto feature_22 = (Eigen::Vector3f() << 0.0f, 1.0f, 0.0f).finished();
  const auto track_11_ids = makeTrackIds({11u});
  const auto track_11_likelihoods = makeTrackLikelihoods({3.0f});
  const auto track_22_with_11_ids = makeTrackIds({22u, 11u});
  const auto track_22_with_11_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial,
              11u,
              {1, 2, 3},
              Eigen::Vector3f::Constant(0.1f),
              &track_11_ids,
              &track_11_likelihoods);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(
        0, makeInput({makeObservation(11u, 1u, prototype)}), initial, offsets, graph);
  }

  cache->store(11u, makeOpenVocabEntry(feature_11, 1, 1, 1));

  std::map<size_t, size_t> remap;
  for (size_t i = 0; i < 8; ++i) {
    remap[i] = i;
  }
  const auto tracking = MeshDelta::TrackingInfo::with_remap(0, 8, 0, remap);
  {
    auto steady = makeDelta(tracking,
                            11u,
                            Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                            Eigen::Vector3f::Constant(0.1f),
                            &track_11_ids,
                            &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(
        1, makeInput({makeObservation(11u, 1u, prototype)}), steady, offsets, graph);
  }
  {
    auto steady = makeDelta(tracking,
                            11u,
                            Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                            Eigen::Vector3f::Constant(0.1f),
                            &track_11_ids,
                            &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(
        2, makeInput({makeObservation(11u, 1u, prototype)}), steady, offsets, graph);
  }
  {
    auto steady = makeDelta(tracking,
                            11u,
                            Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                            Eigen::Vector3f::Constant(0.1f),
                            &track_11_ids,
                            &track_11_likelihoods);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(3, makeInput({}), steady, offsets, graph);
  }

  cache->store(22u, makeOpenVocabEntry(feature_22, 4, 4, 2));

  {
    auto relabeled = makeDelta(tracking,
                               22u,
                               Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                               Eigen::Vector3f::Constant(0.1f),
                               &track_22_with_11_ids,
                               &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(
        4, makeInput({makeObservation(22u, 2u, prototype)}), relabeled, offsets, graph);
  }
  {
    auto relabeled = makeDelta(tracking,
                               22u,
                               Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                               Eigen::Vector3f::Constant(0.1f),
                               &track_22_with_11_ids,
                               &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(
        5, makeInput({makeObservation(22u, 2u, prototype)}), relabeled, offsets, graph);
  }
  {
    auto relabeled = makeDelta(tracking,
                               22u,
                               Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                               Eigen::Vector3f::Constant(0.1f),
                               &track_22_with_11_ids,
                               &track_22_with_11_likelihoods);
    relabeled.updateMesh(*graph.mesh(), offsets);
    updater.update(
        6, makeInput({makeObservation(22u, 2u, prototype)}), relabeled, offsets, graph);
  }

  ASSERT_TRUE(graph.hasNode("O0"_id));
  EXPECT_FALSE(graph.hasNode("O1"_id));
  const auto& attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  ASSERT_EQ(attrs.open_vocab_features.rows(), 3);
  ASSERT_EQ(attrs.open_vocab_features.cols(), 2);
  EXPECT_TRUE(attrs.open_vocab_features.col(0).isApprox(feature_11));
  EXPECT_TRUE(attrs.open_vocab_features.col(1).isApprox(feature_22));
  const auto metadata = attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_feature_source_track_ids"),
      nlohmann::json::array({11u, 22u}));
  EXPECT_EQ(metadata.at("hydra_tracked_object").at("open_vocab_ignore_base_state"),
            "keep");
  EXPECT_FALSE(trackedIgnoreEffective(graph, "O0"_id));
  ASSERT_TRUE(metadata.at("hydra_tracked_object").contains("open_vocab_ignore_votes"));
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("ignore"),
      1u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("keep"), 1u);
  EXPECT_EQ(
      metadata.at("hydra_tracked_object").at("open_vocab_ignore_votes").at("unknown"),
      0u);
}

TEST(TrackGraphSegmentUpdater, HistoricalAssociatorMergesDisjointRemeshedRevisit) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 2;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  config.historical_associator.log_proposals = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }
  {
    auto steady =
        makeDelta(makeIdentityRemapTracking(8), 11u, candidate_offset, candidate_scale);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   steady,
                   offsets,
                   graph);
  }
  {
    auto inactive_but_supported =
        makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                  11u,
                  candidate_offset,
                  candidate_scale);
    inactive_but_supported.updateMesh(*graph.mesh(), offsets);
    updater.update(2, makeInput({}), inactive_but_supported, offsets, graph);
  }
  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& still_active_attrs =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  EXPECT_TRUE(still_active_attrs.is_active);

  auto run_query_pass = [&](uint64_t timestamp_ns, bool archive_candidate_support) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    if (archive_candidate_support) {
      addPoints(query,
                11u,
                candidate_offset,
                candidate_scale,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                true);
    } else {
      addPoints(query, 11u, candidate_offset, candidate_scale);
    }
    addPoints(query, 22u, query_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   query,
                   offsets,
                   graph);
  };

  run_query_pass(3, false);
  ASSERT_TRUE(graph.hasNode("O1"_id));
  run_query_pass(4, false);
  EXPECT_TRUE(graph.hasNode("O0"_id));
  EXPECT_TRUE(graph.hasNode("O1"_id));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));

  run_query_pass(5, true);

  uint64_t next_timestamp = 6;
  const bool merged =
      waitForCondition([&]() { run_query_pass(next_timestamp++, true); },
                       [&]() { return !graph.hasNode("O1"_id); });

  ASSERT_TRUE(merged);
  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto& merged_attrs = graph.getNode("O0"_id).attributes<ObjectNodeAttributes>();
  const auto metadata = merged_attrs.metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
  EXPECT_TRUE(merged_attrs.is_active);

  EXPECT_GT(merged_attrs.position.x(), candidate_offset.x());

  const auto backend_sync_node_ids = updater.takeBackendSyncNodeIds();
  EXPECT_NE(
      std::find(backend_sync_node_ids.begin(), backend_sync_node_ids.end(), "O0"_id),
      backend_sync_node_ids.end());
  EXPECT_TRUE(updater.takeBackendSyncNodeIds().empty());
}

TEST(TrackGraphLongtermReid, NnQueryToCandidateOverlapAcceptsNearbySupport) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphLongtermReid::Config config;
  config.overlap_mode = TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE;
  config.nn_search_radius_m = 0.10;
  config.min_overlap_fraction = 0.5;
  config.min_feature_similarity = 0.8;
  config.min_joint_score = 0.75;
  TrackGraphLongtermReid associator(config);

  FeatureVector feature(2);
  feature << 1.0f, 0.0f;
  const std::vector<Eigen::Vector3f> candidate_points = {
      {1.0f, 2.0f, 3.0f},
      {1.1f, 2.0f, 3.0f},
      {1.0f, 2.1f, 3.0f},
      {1.1f, 2.1f, 3.0f},
  };
  const std::vector<Eigen::Vector3f> query_points = {
      {1.03f, 2.0f, 3.0f},
      {1.13f, 2.0f, 3.0f},
      {1.03f, 2.1f, 3.0f},
      {1.13f, 2.1f, 3.0f},
  };

  TrackGraphLongtermReid::CandidateSnapshot candidate;
  candidate = makeHistoricalCandidateSnapshot(1u, 10u, candidate_points, feature);
  TrackGraphLongtermReid::QuerySnapshot query;
  query = makeHistoricalQuerySnapshot(2u, 20u, query_points, feature);

  const auto evaluation = associator.evaluatePair(30u, query, candidate);
  ASSERT_TRUE(evaluation.proposal);
  ASSERT_TRUE(evaluation.overlap_fraction);
  EXPECT_DOUBLE_EQ(*evaluation.overlap_fraction, 1.0);
  const auto overlap_timing = timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.timer_namespace + "/nn_overlap");
  ASSERT_TRUE(overlap_timing);
  EXPECT_GT(overlap_timing->elapsed.count(), 0);
  EXPECT_FALSE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.timer_namespace + "/bbox_overlap"));
}

TEST(TrackGraphLongtermReid, NnQueryToCandidateRejectsBboxContainedButSupportFar) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphLongtermReid::Config config;
  config.overlap_mode = TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE;
  config.nn_search_radius_m = 0.10;
  config.min_overlap_fraction = 0.5;
  config.min_feature_similarity = 0.8;
  config.min_joint_score = 0.75;
  TrackGraphLongtermReid associator(config);

  FeatureVector feature(2);
  feature << 1.0f, 0.0f;
  const auto candidate_box =
      BoundingBox(Eigen::Vector3f::Constant(1.0f), Eigen::Vector3f(1.0f, 2.0f, 3.0f));
  const auto query_box =
      BoundingBox(Eigen::Vector3f::Constant(0.05f), Eigen::Vector3f(1.0f, 2.0f, 3.0f));
  const auto candidate_corners = candidate_box.corners();
  const auto query_corners = query_box.corners();
  const std::vector<Eigen::Vector3f> candidate_points(candidate_corners.begin(),
                                                      candidate_corners.end());
  const std::vector<Eigen::Vector3f> query_points(query_corners.begin(),
                                                  query_corners.end());
  const auto candidate =
      makeHistoricalCandidateSnapshot(1u, 10u, candidate_points, feature);
  const auto query = makeHistoricalQuerySnapshot(2u, 20u, query_points, feature);

  const auto evaluation = associator.evaluatePair(30u, query, candidate);
  EXPECT_FALSE(evaluation.proposal);
  EXPECT_EQ(evaluation.rejection_reasons,
            std::vector<std::string>{"overlap_below_threshold"});
}

TEST(TrackGraphLongtermReid, BboxOverlapFallbackAcceptsContainedQuery) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphLongtermReid::Config config;
  config.overlap_mode = TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE;
  config.nn_search_radius_m = 0.10;
  config.min_overlap_fraction = 0.5;
  config.min_feature_similarity = 0.8;
  config.min_joint_score = 0.75;
  TrackGraphLongtermReid associator(config);

  FeatureVector feature(2);
  feature << 1.0f, 0.0f;
  const auto candidate_box =
      BoundingBox(Eigen::Vector3f::Constant(1.0f), Eigen::Vector3f(1.0f, 2.0f, 3.0f));
  const auto query_box =
      BoundingBox(Eigen::Vector3f::Constant(0.05f), Eigen::Vector3f(1.0f, 2.0f, 3.0f));
  const auto candidate_corners = candidate_box.corners();
  const auto query_corners = query_box.corners();
  const std::vector<Eigen::Vector3f> candidate_points(candidate_corners.begin(),
                                                      candidate_corners.end());
  const std::vector<Eigen::Vector3f> query_points(query_corners.begin(),
                                                  query_corners.end());
  auto candidate = makeHistoricalCandidateSnapshot(1u, 10u, candidate_points, feature);
  candidate.support_points.clear();
  const auto query = makeHistoricalQuerySnapshot(2u, 20u, query_points, feature);

  const auto evaluation = associator.evaluatePair(30u, query, candidate);
  ASSERT_TRUE(evaluation.proposal);
  ASSERT_TRUE(evaluation.overlap_fraction);
  EXPECT_DOUBLE_EQ(*evaluation.overlap_fraction, 1.0);
  const auto overlap_timing = timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.timer_namespace + "/bbox_overlap");
  ASSERT_TRUE(overlap_timing);
  EXPECT_GT(overlap_timing->elapsed.count(), 0);
  EXPECT_FALSE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.timer_namespace + "/nn_overlap"));
}

TEST(TrackGraphLongtermReid, ShortlistCandidatesUsesHeaderOnlyPrechecks) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphLongtermReid::Config config;
  config.debug_decisions = true;
  config.num_candidates = 4;
  config.max_candidate_radius_m = 1.0;
  TrackGraphLongtermReid associator(config);

  FeatureVector feature(2);
  feature << 1.0f, 0.0f;
  FeatureVector incompatible_feature(3);
  incompatible_feature << 1.0f, 0.0f, 0.0f;
  const auto near_box =
      BoundingBox(Eigen::Vector3f::Constant(0.1f), Eigen::Vector3f(1.05f, 2.05f, 3.0f));
  const auto far_box =
      BoundingBox(Eigen::Vector3f::Constant(0.1f), Eigen::Vector3f(4.1f, 4.0f, 4.0f));
  const auto near_corners = near_box.corners();
  const auto far_corners = far_box.corners();
  const std::vector<Eigen::Vector3f> near_points(near_corners.begin(),
                                                 near_corners.end());
  const std::vector<Eigen::Vector3f> far_points(far_corners.begin(), far_corners.end());

  const auto query =
      makeHistoricalQueryHeader(10u, 20u, Eigen::Vector3d(1.05, 2.05, 3.0), feature);
  const std::vector<TrackGraphLongtermReid::CandidateHeader> candidates = {
      makeHistoricalCandidateHeader(1u, 10u, near_points, feature),
      makeHistoricalCandidateHeader(2u, 30u, near_points, feature),
      makeHistoricalCandidateHeader(3u, 10u, near_points, incompatible_feature),
      makeHistoricalCandidateHeader(4u, 10u, far_points, feature),
  };

  const auto shortlist = associator.shortlistCandidates(40u, {query}, candidates);
  ASSERT_EQ(shortlist.candidates_by_query.size(), 1u);
  ASSERT_EQ(shortlist.candidates_by_query.front().size(), 1u);
  EXPECT_EQ(shortlist.candidates_by_query.front().front().candidate_index, 0u);

  EXPECT_FALSE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.timer_namespace + "/pair_evaluation"));
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorDoesNotMaterializeQuerySupportWhenShortlistIsEmpty) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.overlap_mode =
      TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 0.2;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto feature = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto far_query_offset = Eigen::Vector3f(4.0f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  seedFrozenTrackedObject(
      updater, graph, offsets, 0, 11u, 1u, feature, candidate_offset, candidate_scale);
  timing::ElapsedTimeRecorder::instance().reset();

  auto query = makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                         22u,
                         far_query_offset,
                         query_scale);
  query.updateMesh(*graph.mesh(), offsets);
  updater.update(
      10, makeInput({makeObservation(22u, 2u, feature)}, true), query, offsets, graph);

  EXPECT_FALSE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.historical_associator.timer_namespace + "/query_support_materialization"));
  EXPECT_FALSE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.historical_associator.timer_namespace +
      "/candidate_support_materialization"));
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorOnlyMaterializesShortlistedCandidateSupportForNnOverlap) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.overlap_mode =
      TrackGraphLongtermReid::OverlapMode::NN_QUERY_TO_CANDIDATE;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.num_candidates = 4;
  config.historical_associator.max_candidate_radius_m = 0.5;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto feature = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto near_candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto far_candidate_offset = Eigen::Vector3f(3.5f, 2.0f, 3.0f);
  const auto query_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, near_candidate_offset, candidate_scale);
    addPoints(initial, 55u, far_candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, feature),
                              makeObservation(55u, 5u, feature)},
                             true),
                   initial,
                   offsets,
                   graph);
  }
  {
    MeshDelta steady(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(steady, 11u, near_candidate_offset, candidate_scale);
    addPoints(steady, 55u, far_candidate_offset, candidate_scale);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(11u, 1u, feature),
                              makeObservation(55u, 5u, feature)},
                             true),
                   steady,
                   offsets,
                   graph);
  }
  {
    MeshDelta archive(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(archive,
              11u,
              near_candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(archive,
              55u,
              far_candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    archive.updateMesh(*graph.mesh(), offsets);
    updater.update(2, makeInput({}), archive, offsets, graph);
  }
  timing::ElapsedTimeRecorder::instance().reset();

  auto query = makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                         22u,
                         query_offset,
                         query_scale);
  query.updateMesh(*graph.mesh(), offsets);
  updater.update(
      20, makeInput({makeObservation(22u, 2u, feature)}, true), query, offsets, graph);

  const auto query_support_timer = timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.historical_associator.timer_namespace + "/query_support_materialization");
  ASSERT_TRUE(query_support_timer);
  EXPECT_GT(query_support_timer->elapsed.count(), 0);

  const auto candidate_support_stats = timing::ElapsedTimeRecorder::instance().getStats(
      config.historical_associator.timer_namespace +
      "/candidate_support_materialization");
  EXPECT_EQ(candidate_support_stats.num_measurements, 1u);
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorBboxOverlapDoesNotMaterializeCandidateSupport) {
  timing::ElapsedTimeRecorder::instance().reset();
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.overlap_mode =
      TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 0.5;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto feature = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  seedFrozenTrackedObject(
      updater, graph, offsets, 0, 11u, 1u, feature, candidate_offset, candidate_scale);
  timing::ElapsedTimeRecorder::instance().reset();

  auto query = makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                         22u,
                         query_offset,
                         query_scale);
  query.updateMesh(*graph.mesh(), offsets);
  updater.update(
      10, makeInput({makeObservation(22u, 2u, feature)}, true), query, offsets, graph);

  EXPECT_TRUE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.historical_associator.timer_namespace + "/query_support_materialization"));
  EXPECT_FALSE(timing::ElapsedTimeRecorder::instance().getLastEntry(
      config.historical_associator.timer_namespace +
      "/candidate_support_materialization"));
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorCanReuseHistoricallyRevivedActiveCandidate) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto first_query_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto second_query_offset = Eigen::Vector3f(0.95f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }

  auto run_first_query_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query,
              11u,
              candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query, 22u, first_query_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   query,
                   offsets,
                   graph);
  };

  run_first_query_pass(1);
  uint64_t next_timestamp = 2;
  const bool first_merged =
      waitForCondition([&]() { run_first_query_pass(next_timestamp++); },
                       [&]() { return !graph.hasNode("O1"_id); });

  ASSERT_TRUE(first_merged);
  ASSERT_TRUE(graph.hasNode("O0"_id));
  {
    const auto metadata =
        graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
    EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));
  }

  auto run_second_query_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query,
              11u,
              candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query, 22u, first_query_offset, query_scale);
    addPoints(query, 33u, second_query_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype),
                              makeObservation(33u, 3u, prototype)},
                             true),
                   query,
                   offsets,
                   graph);
  };

  run_second_query_pass(next_timestamp++);
  const auto query_33_uid = findObjectUidByName(graph, "track_33");

  const bool second_merged =
      query_33_uid == 0u
          ? true
          : waitForCondition(
                [&]() { run_second_query_pass(next_timestamp++); },
                [&]() { return findNodeByName(graph, "track_33") == NodeId(0); });

  ASSERT_TRUE(second_merged);
  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id),
            (std::vector<uint32_t>{11u, 22u, 33u}));

  if (query_33_uid != 0u) {
  }
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorBatchesMultipleQueriesIntoSameCandidate) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  config.historical_associator.log_jobs = true;
  config.historical_associator.log_proposals = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto candidate_feature = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto strong_feature = candidate_feature;
  const auto medium_feature = (Eigen::Vector2f() << 0.9f, 0.4358899f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_22_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto query_33_offset = Eigen::Vector3f(0.95f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, candidate_feature)}, true),
                   initial,
                   offsets,
                   graph);
  }

  auto run_batch_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query,
              11u,
              candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query, 22u, query_22_offset, query_scale);
    addPoints(query, 33u, query_33_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, strong_feature),
                              makeObservation(33u, 3u, medium_feature)},
                             true),
                   query,
                   offsets,
                   graph);
  };

  run_batch_pass(1);

  uint64_t next_timestamp = 2;
  const bool merged =
      waitForCondition([&]() { run_batch_pass(next_timestamp++); },
                       [&]() {
                         return findNodeByName(graph, "track_22") == NodeId(0) &&
                                findNodeByName(graph, "track_33") == NodeId(0);
                       });

  ASSERT_TRUE(merged);
  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id),
            (std::vector<uint32_t>{11u, 22u, 33u}));
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorBatchReevaluationRejectsWithConcreteReason) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.overlap_mode =
      TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.7;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  config.historical_associator.log_proposals = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto candidate_feature = Eigen::Vector2f(1.0f, 1.0f).normalized();
  const auto strong_feature = (Eigen::Vector2f() << 0.9238795f, 0.3826834f).finished();
  const auto weak_feature = (Eigen::Vector2f() << 0.0f, 1.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_22_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto query_33_offset = Eigen::Vector3f(0.95f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, candidate_feature)}, true),
                   initial,
                   offsets,
                   graph);
  }

  auto run_batch_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query,
              11u,
              candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query, 22u, query_22_offset, query_scale);
    addPoints(query, 33u, query_33_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, strong_feature),
                              makeObservation(33u, 3u, weak_feature)},
                             true),
                   query,
                   offsets,
                   graph);
  };

  run_batch_pass(1);
  uint64_t next_timestamp = 2;
  const bool settled =
      waitForCondition([&]() { run_batch_pass(next_timestamp++); },
                       [&]() {
                         return findNodeByName(graph, "track_22") == NodeId(0) &&
                                findNodeByName(graph, "track_33") != NodeId(0);
                       });

  ASSERT_TRUE(settled);

  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));

  const auto surviving_query = findNodeByName(graph, "track_33");
  ASSERT_NE(surviving_query, NodeId(0));
  const auto query_metadata =
      graph.getNode(surviving_query).attributes<ObjectNodeAttributes>().metadata.get();
}

TEST(TrackGraphSegmentUpdater, HistoricalAssociatorBatchOrderingIsDeterministic) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.overlap_mode =
      TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto candidate_feature = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto weakest_feature = Eigen::Vector2f(0.98f, 0.2f).normalized();
  const auto middle_feature = Eigen::Vector2f(0.995f, 0.1f).normalized();
  const auto strongest_feature = candidate_feature;
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_22_offset = Eigen::Vector3f(1.0f, 1.97f, 3.0f);
  const auto query_33_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_44_offset = Eigen::Vector3f(1.0f, 2.03f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, candidate_feature)}, true),
                   initial,
                   offsets,
                   graph);
  }

  auto run_batch_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query,
              11u,
              candidate_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query, 22u, query_22_offset, query_scale);
    addPoints(query, 33u, query_33_offset, query_scale);
    addPoints(query, 44u, query_44_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, weakest_feature),
                              makeObservation(33u, 3u, middle_feature),
                              makeObservation(44u, 4u, strongest_feature)},
                             true),
                   query,
                   offsets,
                   graph);
  };

  run_batch_pass(1);

  uint64_t next_timestamp = 2;
  const bool merged =
      waitForCondition([&]() { run_batch_pass(next_timestamp++); },
                       [&]() {
                         return findNodeByName(graph, "track_22") == NodeId(0) &&
                                findNodeByName(graph, "track_33") == NodeId(0) &&
                                findNodeByName(graph, "track_44") == NodeId(0);
                       });

  ASSERT_TRUE(merged);
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id),
            (std::vector<uint32_t>{11u, 44u, 33u, 22u}));
}

TEST(TrackGraphSegmentUpdater, HistoricalAssociatorBatchingStaysLocalAcrossCandidates) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto feature_a = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto feature_b = (Eigen::Vector2f() << 0.0f, 1.0f).finished();
  const auto candidate_a_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto candidate_b_offset = Eigen::Vector3f(3.0f, 2.0f, 3.0f);
  const auto query_a_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto query_b_offset = Eigen::Vector3f(3.05f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, candidate_a_offset, candidate_scale);
    addPoints(initial, 55u, candidate_b_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, feature_a),
                              makeObservation(55u, 5u, feature_b)},
                             true),
                   initial,
                   offsets,
                   graph);
  }

  auto run_batch_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query,
              11u,
              candidate_a_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query,
              55u,
              candidate_b_offset,
              candidate_scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(query, 22u, query_a_offset, query_scale);
    addPoints(query, 66u, query_b_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, feature_a),
                              makeObservation(66u, 6u, feature_b)},
                             true),
                   query,
                   offsets,
                   graph);
  };

  run_batch_pass(1);
  uint64_t next_timestamp = 2;
  const bool merged =
      waitForCondition([&]() { run_batch_pass(next_timestamp++); },
                       [&]() {
                         return findNodeByName(graph, "track_22") == NodeId(0) &&
                                findNodeByName(graph, "track_66") == NodeId(0);
                       });

  ASSERT_TRUE(merged);
  const auto root_a = findNodeByName(graph, "track_11");
  const auto root_b = findNodeByName(graph, "track_55");
  ASSERT_NE(root_a, NodeId(0));
  ASSERT_NE(root_b, NodeId(0));
  const auto metadata_a =
      graph.getNode(root_a).attributes<ObjectNodeAttributes>().metadata.get();
  const auto metadata_b =
      graph.getNode(root_b).attributes<ObjectNodeAttributes>().metadata.get();
  EXPECT_EQ(trackedSourceTrackIds(graph, root_a), (std::vector<uint32_t>{11u, 22u}));
  EXPECT_EQ(trackedSourceTrackIds(graph, root_b), (std::vector<uint32_t>{55u, 66u}));
}

TEST(TrackGraphSegmentUpdater,
     ActiveWindowReidentifierMergesPartOfClassifiedFragmentIntoHistoricallyMergedRoot) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = true;
  config.reidentifier.min_shared_vertices = 1;
  config.reidentifier.min_observation_frames = 3;
  config.reidentifier.debug_decisions = true;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 2;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh(true));
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_offset = Eigen::Vector3f(1.05f, 2.0f, 3.0f);
  const auto extra_offset_a = Eigen::Vector3f(1.35f, 2.0f, 3.0f);
  const auto extra_offset_b = Eigen::Vector3f(1.65f, 2.0f, 3.0f);
  const auto extra_offset_c = Eigen::Vector3f(1.95f, 2.0f, 3.0f);
  const auto candidate_scale = Eigen::Vector3f::Constant(0.2f);
  const auto query_scale = Eigen::Vector3f::Constant(0.1f);
  const auto track_22_ids = makeTrackIds({22u});
  const auto track_22_likelihoods = makeTrackLikelihoods({4.0f});
  const auto track_22_with_33_ids = makeTrackIds({22u, 33u});
  const auto track_22_with_33_likelihoods = makeTrackLikelihoods({4.0f, 1.0f});
  const auto track_33_ids = makeTrackIds({33u});
  const auto track_33_likelihoods = makeTrackLikelihoods({4.0f});

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, candidate_scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }
  {
    auto steady =
        makeDelta(makeIdentityRemapTracking(8), 11u, candidate_offset, candidate_scale);
    steady.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   steady,
                   offsets,
                   graph);
  }
  {
    auto inactive_but_supported =
        makeDelta(makeIdentityRemapTracking(graph.mesh()->numVertices()),
                  11u,
                  candidate_offset,
                  candidate_scale);
    inactive_but_supported.updateMesh(*graph.mesh(), offsets);
    updater.update(2, makeInput({}), inactive_but_supported, offsets, graph);
  }

  auto run_historical_query_pass = [&](uint64_t timestamp_ns,
                                       bool archive_candidate_support) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    if (archive_candidate_support) {
      addPoints(query,
                11u,
                candidate_offset,
                candidate_scale,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                true);
    } else {
      addPoints(query, 11u, candidate_offset, candidate_scale);
    }
    addPoints(query, 22u, query_offset, query_scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   query,
                   offsets,
                   graph);
  };

  run_historical_query_pass(3, false);
  run_historical_query_pass(4, false);
  run_historical_query_pass(5, true);

  uint64_t next_timestamp = 6;
  const bool merged =
      waitForCondition([&]() { run_historical_query_pass(next_timestamp++, true); },
                       [&]() { return !graph.hasNode("O1"_id); });

  ASSERT_TRUE(merged);
  ASSERT_TRUE(graph.hasNode("O0"_id));
  const auto root_metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(root_metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u, 22u}));

  EXPECT_FALSE(updater.takeBackendSyncNodeIds().empty());
  EXPECT_TRUE(updater.takeBackendSyncNodeIds().empty());

  auto run_active_window_pass = [&](uint64_t timestamp_ns,
                                    bool include_nonwinner_overlap) {
    const auto* track_22_pass_ids =
        include_nonwinner_overlap ? &track_22_with_33_ids : &track_22_ids;
    const auto* track_22_pass_likelihoods = include_nonwinner_overlap
                                                ? &track_22_with_33_likelihoods
                                                : &track_22_likelihoods;
    MeshDelta revisit(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(revisit,
              22u,
              candidate_offset,
              candidate_scale,
              track_22_pass_ids,
              track_22_pass_likelihoods);
    addPoints(revisit,
              22u,
              query_offset,
              query_scale,
              track_22_pass_ids,
              track_22_pass_likelihoods);
    addPoints(revisit,
              33u,
              extra_offset_a,
              query_scale,
              &track_33_ids,
              &track_33_likelihoods);
    addPoints(revisit,
              33u,
              extra_offset_b,
              query_scale,
              &track_33_ids,
              &track_33_likelihoods);
    addPoints(revisit,
              33u,
              extra_offset_c,
              query_scale,
              &track_33_ids,
              &track_33_likelihoods);
    revisit.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype),
                              makeObservation(33u, 3u, prototype)},
                             false),
                   revisit,
                   offsets,
                   graph);
  };

  run_active_window_pass(next_timestamp++, false);
  run_active_window_pass(next_timestamp++, true);
  run_active_window_pass(next_timestamp++, true);

  const auto root_node = findNodeByName(graph, "track_11");
  ASSERT_NE(root_node, NodeId(0));
  EXPECT_EQ(findNodeByName(graph, "track_33"), NodeId(0));

  const auto& parent_metadata =
      graph.getNode(root_node).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(parent_metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, root_node),
            (std::vector<uint32_t>{11u, 22u, 33u}));

  {}
}

TEST(TrackGraphSegmentUpdater, HistoricalAssociatorRejectsLowFeatureSimilarity) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto candidate_feature = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto query_feature = (Eigen::Vector2f() << 0.0f, 1.0f).finished();
  const auto offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, offset, scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, candidate_feature)}, true),
                   initial,
                   offsets,
                   graph);
  }
  {
    MeshDelta archived(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(archived, 11u, offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(archived, 22u, offset, scale);
    archived.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(22u, 2u, query_feature)}, true),
                   archived,
                   offsets,
                   graph);
  }

  auto run_query_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(query, 11u, offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(query, 22u, offset, scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, query_feature)}, true),
                   query,
                   offsets,
                   graph);
  };

  run_query_pass(2);

  EXPECT_TRUE(graph.hasNode("O0"_id));
  EXPECT_TRUE(graph.hasNode("O1"_id));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));

  const auto query_metadata =
      graph.getNode("O1"_id).attributes<ObjectNodeAttributes>().metadata.get();
}

TEST(TrackGraphSegmentUpdater, HistoricalAssociatorRejectsLowPseudoOverlap) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.overlap_mode =
      TrackGraphLongtermReid::OverlapMode::BBOX_QUERY_IN_CANDIDATE;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 1.0;
  config.historical_associator.min_overlap_fraction = 0.3;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  config.historical_associator.debug_decisions = true;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto candidate_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto query_offset = Eigen::Vector3f(1.25f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    auto initial = makeDelta({0, 0, 0}, 11u, candidate_offset, scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype)}, true),
                   initial,
                   offsets,
                   graph);
  }
  {
    MeshDelta archived(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(archived,
              11u,
              candidate_offset,
              scale,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              true);
    addPoints(archived, 22u, query_offset, scale);
    archived.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   archived,
                   offsets,
                   graph);
  }

  auto run_query_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(
        query, 11u, candidate_offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(query, 22u, query_offset, scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   query,
                   offsets,
                   graph);
  };

  run_query_pass(2);

  EXPECT_TRUE(graph.hasNode("O0"_id));
  EXPECT_TRUE(graph.hasNode("O1"_id));
  const auto metadata =
      graph.getNode("O0"_id).attributes<ObjectNodeAttributes>().metadata.get();
  EXPECT_EQ(trackedSourceTrackIds(graph, "O0"_id), (std::vector<uint32_t>{11u}));

  const auto query_metadata =
      graph.getNode("O1"_id).attributes<ObjectNodeAttributes>().metadata.get();
}

TEST(TrackGraphSegmentUpdater,
     HistoricalAssociatorIgnoresDebugPartOfEdgesWhenMatching) {
  TrackGraphSegmentUpdater::Config config;
  config.min_points_create = 4;
  config.reidentifier.enabled = false;
  config.historical_associator.enabled = true;
  config.historical_associator.min_query_keyframes = 1;
  config.historical_associator.max_candidate_radius_m = 5.0;
  config.historical_associator.min_overlap_fraction = 0.5;
  config.historical_associator.min_feature_similarity = 0.8;
  config.historical_associator.min_joint_score = 0.75;
  TrackGraphSegmentUpdater updater(config);

  DynamicSceneGraph graph;
  graph.setMesh(createMesh());
  kimera_pgmo::MeshOffsetInfo offsets;
  const auto prototype = (Eigen::Vector2f() << 1.0f, 0.0f).finished();
  const auto parent_offset = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
  const auto child_offset = Eigen::Vector3f(1.8f, 2.0f, 3.0f);
  const auto scale = Eigen::Vector3f::Constant(0.1f);

  {
    MeshDelta initial({0, 0, 0});
    addPoints(initial, 11u, parent_offset, scale);
    addPoints(initial, 33u, child_offset, scale);
    initial.updateMesh(*graph.mesh(), offsets);
    updater.update(0,
                   makeInput({makeObservation(11u, 1u, prototype),
                              makeObservation(33u, 3u, prototype)},
                             true),
                   initial,
                   offsets,
                   graph);
  }

  const auto parent_node = findNodeByName(graph, "track_11");
  const auto child_node = findNodeByName(graph, "track_33");
  ASSERT_NE(parent_node, NodeId(0));
  ASSERT_NE(child_node, NodeId(0));
  auto edge_attrs = std::make_unique<EdgeAttributes>(1.0);
  edge_attrs->metadata.add(nlohmann::json{{"tracked_object_relationship", "part_of"}});
  ASSERT_TRUE(graph.insertEdge(parent_node, child_node, std::move(edge_attrs)));

  {
    MeshDelta archived(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(
        archived, 11u, parent_offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(
        archived, 33u, child_offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(archived, 22u, child_offset, scale);
    archived.updateMesh(*graph.mesh(), offsets);
    updater.update(1,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   archived,
                   offsets,
                   graph);
  }

  auto run_query_pass = [&](uint64_t timestamp_ns) {
    MeshDelta query(makeIdentityRemapTracking(graph.mesh()->numVertices()));
    addPoints(
        query, 11u, parent_offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(
        query, 33u, child_offset, scale, nullptr, nullptr, nullptr, nullptr, true);
    addPoints(query, 22u, child_offset, scale);
    query.updateMesh(*graph.mesh(), offsets);
    updater.update(timestamp_ns,
                   makeInput({makeObservation(22u, 2u, prototype)}, true),
                   query,
                   offsets,
                   graph);
  };

  uint64_t next_timestamp = 2;
  const bool merged = waitForCondition(
      [&]() { run_query_pass(next_timestamp++); },
      [&]() { return findNodeByName(graph, "track_22") == NodeId(0); });

  ASSERT_TRUE(merged);
  EXPECT_TRUE(graph.hasNode(parent_node));
  EXPECT_TRUE(graph.hasNode(child_node));
  EXPECT_TRUE(graph.hasEdge(parent_node, child_node));

  const auto parent_metadata =
      graph.getNode(parent_node).attributes<ObjectNodeAttributes>().metadata.get();
  const auto child_metadata =
      graph.getNode(child_node).attributes<ObjectNodeAttributes>().metadata.get();
  ASSERT_TRUE(parent_metadata.contains("hydra_tracked_object"));
  ASSERT_TRUE(child_metadata.contains("hydra_tracked_object"));
  EXPECT_EQ(trackedSourceTrackIds(graph, parent_node), (std::vector<uint32_t>{11u}));
  EXPECT_EQ(trackedSourceTrackIds(graph, child_node),
            (std::vector<uint32_t>{33u, 22u}));
}

}  // namespace hydra
