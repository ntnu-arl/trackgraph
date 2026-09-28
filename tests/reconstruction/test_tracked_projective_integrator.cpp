/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 * -------------------------------------------------------------------------- */
#include <config_utilities/parsing/yaml.h>
#include <gtest/gtest.h>
#include <hydra/active_window/reconstruction_module.h>
#include <hydra/frontend/trackgraph_segment_updater.h>
#include <hydra/input/camera.h>
#include <hydra/reconstruction/mesh_integrator.h>
#include <hydra/reconstruction/tracked_projective_integrator.h>
#include <hydra/utils/pgmo_mesh_traits.h>
#include <kimera_pgmo/compression/delta_compression.h>
#include <kimera_pgmo/mesh_offset_info.h>
#include <spark_dsg/dynamic_scene_graph.h>

#include <filesystem>

#include "hydra_test_config.h"

namespace hydra {
namespace {

TrackedProjectiveIntegrator::Config makeConfig() {
  TrackedProjectiveIntegrator::Config config;
  config.num_threads = 1;
  return config;
}

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

InputData makeInput() {
  InputData input(createCamera());
  input.timestamp_ns = 1;
  return input;
}

TrackObservation makeObservation(uint32_t track_id,
                                 uint32_t instance_id,
                                 float confidence = 0.0f,
                                 uint32_t age = 0,
                                 uint32_t frames_since_detection = 0) {
  TrackObservation observation;
  observation.track_id = track_id;
  observation.instance_id = instance_id;
  observation.confidence = confidence;
  observation.age = age;
  observation.frames_since_detection = frames_since_detection;
  return observation;
}

ProjectiveIntegrator::VoxelMeasurement makeMeasurement(uint32_t track_id,
                                                       float weight,
                                                       float scale) {
  ProjectiveIntegrator::VoxelMeasurement measurement;
  measurement.sdf = 0.1f;
  measurement.weight = weight;
  measurement.track_id = track_id;
  measurement.label_weight_scale = scale;
  return measurement;
}

}  // namespace

TEST(TrackedProjectiveIntegrator, GeometryUpdatesWithoutTrackEvidence) {
  TrackedProjectiveIntegrator integrator(makeConfig());
  auto input = makeInput();

  VolumetricMap::Config map_config;
  map_config.with_instances = true;

  TsdfVoxel tsdf_without_track;
  InstanceVoxel instance_without_track;
  VoxelTuple voxels_without_track;
  voxels_without_track.tsdf = &tsdf_without_track;
  voxels_without_track.instance = &instance_without_track;
  integrator.updateVoxel(
      map_config,
      input,
      makeMeasurement(InstanceVoxel::NO_TRACK, 2.0f, 0.0f),
      voxels_without_track);

  TsdfVoxel tsdf_with_track;
  InstanceVoxel instance_with_track;
  VoxelTuple voxels_with_track;
  voxels_with_track.tsdf = &tsdf_with_track;
  voxels_with_track.instance = &instance_with_track;
  integrator.updateVoxel(
      map_config, input, makeMeasurement(11u, 2.0f, 1.0f), voxels_with_track);

  EXPECT_FLOAT_EQ(tsdf_without_track.weight, tsdf_with_track.weight);
  EXPECT_FLOAT_EQ(tsdf_without_track.distance, tsdf_with_track.distance);
  EXPECT_TRUE(instance_without_track.empty);
  EXPECT_FALSE(instance_with_track.empty);
  EXPECT_EQ(instance_with_track.winning_track_id, 11u);
}

TEST(TrackedProjectiveIntegrator, KeyframeAndPropagationWeightingCorrect) {
  auto config = makeConfig();
  config.propagation_source_scale = 0.25f;
  TrackedProjectiveIntegrator integrator(config);
  VolumetricMap::Config map_config;

  auto keyframe_input = makeInput();
  keyframe_input.instance_image = cv::Mat(1, 1, CV_32SC1, cv::Scalar(7));
  keyframe_input.tracking_is_keyframe = true;
  keyframe_input.track_observations.push_back(makeObservation(11u, 7u, 0.1f, 0u, 10u));

  ProjectiveIntegrator::VoxelMeasurement keyframe_measurement;
  keyframe_measurement.interpolation_weights = InterpolationWeights(0, 0);
  keyframe_measurement.interpolation_weights.valid = true;
  ASSERT_TRUE(
      integrator.computeLabel(map_config, keyframe_input, cv::Mat(), keyframe_measurement));
  EXPECT_EQ(keyframe_measurement.track_id, 11u);
  EXPECT_FLOAT_EQ(keyframe_measurement.label_weight_scale, 1.0f);

  auto propagated_input_low_conf = makeInput();
  propagated_input_low_conf.instance_image = cv::Mat(1, 1, CV_32SC1, cv::Scalar(7));
  propagated_input_low_conf.tracking_is_keyframe = false;
  propagated_input_low_conf.track_observations.push_back(
      makeObservation(11u, 7u, 0.1f, 0u, 10u));

  ProjectiveIntegrator::VoxelMeasurement propagated_measurement_low_conf;
  propagated_measurement_low_conf.interpolation_weights = InterpolationWeights(0, 0);
  propagated_measurement_low_conf.interpolation_weights.valid = true;
  ASSERT_TRUE(integrator.computeLabel(map_config,
                                      propagated_input_low_conf,
                                      cv::Mat(),
                                      propagated_measurement_low_conf));
  EXPECT_EQ(propagated_measurement_low_conf.track_id, 11u);
  EXPECT_FLOAT_EQ(propagated_measurement_low_conf.label_weight_scale, 0.125f);

  auto propagated_input_high_conf = propagated_input_low_conf;
  propagated_input_high_conf.track_observations.front().confidence = 0.95f;

  ProjectiveIntegrator::VoxelMeasurement propagated_measurement_high_conf;
  propagated_measurement_high_conf.interpolation_weights = InterpolationWeights(0, 0);
  propagated_measurement_high_conf.interpolation_weights.valid = true;
  ASSERT_TRUE(integrator.computeLabel(map_config,
                                      propagated_input_high_conf,
                                      cv::Mat(),
                                      propagated_measurement_high_conf));
  EXPECT_FLOAT_EQ(propagated_measurement_low_conf.label_weight_scale,
                  propagated_measurement_high_conf.label_weight_scale);
}

TEST(TrackedProjectiveIntegrator, KeyframeConfidenceImageDoesNotSuppressConfirmation) {
  auto config = makeConfig();
  config.use_confidence_weighting = true;
  TrackedProjectiveIntegrator integrator(config);
  VolumetricMap::Config map_config;

  auto keyframe_input = makeInput();
  keyframe_input.instance_image = cv::Mat(1, 1, CV_32SC1, cv::Scalar(7));
  keyframe_input.tracking_confidence_image = cv::Mat(1, 1, CV_32FC1, cv::Scalar(0.0f));
  keyframe_input.tracking_is_keyframe = true;
  keyframe_input.track_observations.push_back(makeObservation(11u, 7u));

  ProjectiveIntegrator::VoxelMeasurement measurement;
  measurement.interpolation_weights = InterpolationWeights(0, 0);
  measurement.interpolation_weights.valid = true;
  ASSERT_TRUE(integrator.computeLabel(map_config, keyframe_input, cv::Mat(), measurement));

  EXPECT_EQ(measurement.track_id, 11u);
  EXPECT_FLOAT_EQ(measurement.label_weight_scale, 1.0f);
}

TEST(TrackedProjectiveIntegrator, PropagationWeightingConfigurable) {
  auto config = makeConfig();
  config.propagation_source_scale = 0.5f;
  config.staleness_slope = 0.2f;
  TrackedProjectiveIntegrator integrator(config);
  VolumetricMap::Config map_config;

  auto input = makeInput();
  input.instance_image = cv::Mat(1, 1, CV_32SC1, cv::Scalar(7));
  input.tracking_is_keyframe = false;
  input.track_observations.push_back(makeObservation(11u, 7u, 0.1f, 0u, 10u));

  ProjectiveIntegrator::VoxelMeasurement measurement;
  measurement.interpolation_weights = InterpolationWeights(0, 0);
  measurement.interpolation_weights.valid = true;
  ASSERT_TRUE(integrator.computeLabel(map_config, input, cv::Mat(), measurement));
  EXPECT_EQ(measurement.track_id, 11u);
  EXPECT_FLOAT_EQ(measurement.label_weight_scale, 1.0f / 6.0f);
}

TEST(TrackedProjectiveIntegrator, PropagationConfidenceThresholdSkipsLowConfidencePixels) {
  auto config = makeConfig();
  config.propagation_source_scale = 0.5f;
  config.propagation_confidence_threshold = 0.6f;
  TrackedProjectiveIntegrator integrator(config);
  VolumetricMap::Config map_config;

  auto low_conf_input = makeInput();
  low_conf_input.instance_image = cv::Mat(1, 1, CV_32SC1, cv::Scalar(7));
  low_conf_input.tracking_confidence_image = cv::Mat(1, 1, CV_32FC1, cv::Scalar(0.59f));
  low_conf_input.tracking_is_keyframe = false;
  low_conf_input.track_observations.push_back(makeObservation(11u, 7u, 0.1f, 0u, 0u));

  ProjectiveIntegrator::VoxelMeasurement low_conf_measurement;
  low_conf_measurement.interpolation_weights = InterpolationWeights(0, 0);
  low_conf_measurement.interpolation_weights.valid = true;
  ASSERT_TRUE(
      integrator.computeLabel(map_config, low_conf_input, cv::Mat(), low_conf_measurement));
  EXPECT_EQ(low_conf_measurement.track_id, InstanceVoxel::NO_TRACK);
  EXPECT_FLOAT_EQ(low_conf_measurement.label_weight_scale, 0.0f);

  auto threshold_conf_input = low_conf_input;
  threshold_conf_input.tracking_confidence_image =
      cv::Mat(1, 1, CV_32FC1, cv::Scalar(0.6f));

  ProjectiveIntegrator::VoxelMeasurement threshold_conf_measurement;
  threshold_conf_measurement.interpolation_weights = InterpolationWeights(0, 0);
  threshold_conf_measurement.interpolation_weights.valid = true;
  ASSERT_TRUE(integrator.computeLabel(map_config,
                                      threshold_conf_input,
                                      cv::Mat(),
                                      threshold_conf_measurement));
  EXPECT_EQ(threshold_conf_measurement.track_id, 11u);
  EXPECT_FLOAT_EQ(threshold_conf_measurement.label_weight_scale, 0.3f);
}

TEST(TrackedProjectiveIntegrator, IntegrationMaskIgnored) {
  TrackedProjectiveIntegrator integrator(makeConfig());
  VolumetricMap::Config map_config;

  auto input = makeInput();
  input.instance_image = cv::Mat(1, 1, CV_32SC1, cv::Scalar(7));
  input.tracking_is_keyframe = true;
  input.track_observations.push_back(makeObservation(11u, 7u, 0.1f, 0u, 0u));

  cv::Mat integration_mask(1, 1, CV_32SC1, cv::Scalar(1));

  ProjectiveIntegrator::VoxelMeasurement measurement;
  measurement.interpolation_weights = InterpolationWeights(0, 0);
  measurement.interpolation_weights.valid = true;
  ASSERT_TRUE(integrator.computeLabel(map_config, input, integration_mask, measurement));
  EXPECT_EQ(measurement.track_id, 11u);
  EXPECT_FLOAT_EQ(measurement.label_weight_scale, 1.0f);
}

TEST(TrackedProjectiveIntegrator, SparseHypothesisUpdateCorrect) {
  TrackedProjectiveIntegrator integrator(makeConfig());
  auto input = makeInput();
  input.tracking_is_keyframe = true;
  VolumetricMap::Config map_config;
  map_config.with_instances = true;

  TsdfVoxel tsdf;
  InstanceVoxel instance;
  VoxelTuple voxels;
  voxels.tsdf = &tsdf;
  voxels.instance = &instance;

  integrator.updateVoxel(map_config, input, makeMeasurement(10u, 1.0f, 1.0f), voxels);
  integrator.updateVoxel(map_config, input, makeMeasurement(20u, 2.0f, 1.0f), voxels);
  integrator.updateVoxel(map_config, input, makeMeasurement(30u, 3.0f, 1.0f), voxels);
  integrator.updateVoxel(map_config, input, makeMeasurement(40u, 4.0f, 1.0f), voxels);
  EXPECT_EQ(instance.winning_track_id, 40u);

  integrator.updateVoxel(map_config, input, makeMeasurement(50u, 1.5f, 1.0f), voxels);
  EXPECT_EQ(instance.winning_track_id, 40u);
  EXPECT_EQ(std::count(instance.track_ids.begin(), instance.track_ids.end(), 10u), 0);
  EXPECT_EQ(std::count(instance.track_ids.begin(), instance.track_ids.end(), 50u), 1);
  EXPECT_EQ(std::count(instance.confirmed_track_ids.begin(),
                       instance.confirmed_track_ids.end(),
                       10u),
            0);
  EXPECT_EQ(std::count(instance.confirmed_track_ids.begin(),
                       instance.confirmed_track_ids.end(),
                       50u),
            1);

  integrator.updateVoxel(map_config, input, makeMeasurement(20u, 5.0f, 1.0f), voxels);
  EXPECT_EQ(instance.winning_track_id, 20u);
  EXPECT_EQ(std::count(instance.confirmed_track_ids.begin(),
                       instance.confirmed_track_ids.end(),
                       20u),
            1);
}

TEST(TrackedProjectiveIntegrator, PropagationDoesNotModifyConfirmedSupport) {
  TrackedProjectiveIntegrator integrator(makeConfig());
  VolumetricMap::Config map_config;
  map_config.with_instances = true;

  TsdfVoxel tsdf;
  InstanceVoxel instance;
  VoxelTuple voxels;
  voxels.tsdf = &tsdf;
  voxels.instance = &instance;

  auto keyframe_input = makeInput();
  keyframe_input.tracking_is_keyframe = true;
  integrator.updateVoxel(map_config,
                         keyframe_input,
                         makeMeasurement(11u, 2.0f, 1.0f),
                         voxels);

  const auto confirmed_ids_after_keyframe = instance.confirmed_track_ids;
  const auto confirmed_likelihoods_after_keyframe = instance.confirmed_track_likelihoods;

  auto propagation_input = makeInput();
  propagation_input.tracking_is_keyframe = false;
  integrator.updateVoxel(map_config,
                         propagation_input,
                         makeMeasurement(22u, 3.0f, 1.0f),
                         voxels);

  EXPECT_EQ(instance.winning_track_id, 22u);
  EXPECT_EQ(instance.confirmed_track_ids, confirmed_ids_after_keyframe);
  EXPECT_EQ(instance.confirmed_track_likelihoods, confirmed_likelihoods_after_keyframe);
}

TEST(TrackedProjectiveIntegrator, MeshUsesInstanceLabels) {
  VolumetricMap::Config map_config;
  map_config.voxel_size = 1.0f;
  map_config.voxels_per_side = 2;
  map_config.truncation_distance = 1.0f;
  map_config.with_instances = true;

  VolumetricMap map(map_config);
  const BlockIndex block_index = BlockIndex::Zero();
  ASSERT_TRUE(map.allocateBlock(block_index));

  auto& tsdf = map.getTsdfLayer().getBlock(block_index);
  const std::array<float, 8> sdf_values = {-1.0f, 1.0f, 10.0f, 2.0f,
                                           3.0f,  10.0f, 10.0f, 10.0f};
  auto& instances = map.getInstanceLayer()->getBlock(block_index);
  for (size_t i = 0; i < tsdf.numVoxels(); ++i) {
    auto& voxel = tsdf.getVoxel(i);
    voxel.weight = 1.0f;
    voxel.distance = sdf_values[i];

    auto& instance = instances.getVoxel(i);
    instance.empty = false;
    instance.winning_track_id = 23u;
  }

  MeshIntegratorConfig mesh_config;
  mesh_config.integrator_threads = 1;
  MeshIntegrator mesh_integrator(mesh_config);
  mesh_integrator.generateMesh(map, false, false);

  const auto& mesh = map.getMeshLayer().getBlock(block_index);
  ASSERT_TRUE(mesh.has_labels);
  ASSERT_FALSE(mesh.labels.empty());
  for (const auto label : mesh.labels) {
    EXPECT_EQ(label, 23u);
  }
}

TEST(TrackedProjectiveIntegrator, KeyframeConfirmedSupportCreatesObjectAfterMeshing) {
  constexpr uint32_t kTrackId = 23u;
  VolumetricMap::Config map_config;
  map_config.voxel_size = 1.0f;
  map_config.voxels_per_side = 2;
  map_config.truncation_distance = 1.0f;
  map_config.with_instances = true;

  VolumetricMap map(map_config);
  const BlockIndex block_index = BlockIndex::Zero();
  ASSERT_TRUE(map.allocateBlock(block_index));

  TrackedProjectiveIntegrator integrator(makeConfig());
  auto input = makeInput();
  input.tracking_is_keyframe = true;

  auto& tsdf = map.getTsdfLayer().getBlock(block_index);
  auto& instances = map.getInstanceLayer()->getBlock(block_index);
  const std::array<float, 8> sdf_values = {-0.5f, 0.5f, 0.5f, 0.5f,
                                           0.5f,  0.5f, 0.5f, 0.5f};
  for (size_t i = 0; i < tsdf.numVoxels(); ++i) {
    VoxelTuple voxels;
    voxels.tsdf = &tsdf.getVoxel(i);
    voxels.instance = &instances.getVoxel(i);
    ProjectiveIntegrator::VoxelMeasurement measurement;
    measurement.sdf = sdf_values[i];
    measurement.weight = 1.0f;
    measurement.track_id = kTrackId;
    measurement.label_weight_scale = 1.0f;
    integrator.updateVoxel(map_config, input, measurement, voxels);
  }

  MeshIntegratorConfig mesh_config;
  mesh_config.integrator_threads = 1;
  MeshIntegrator mesh_integrator(mesh_config);
  mesh_integrator.generateMesh(map, false, false);

  const auto& mesh_block = map.getMeshLayer().getBlock(block_index);
  ASSERT_GT(mesh_block.numVertices(), 0u);
  ASSERT_TRUE(mesh_block.has_labels);
  ASSERT_TRUE(mesh_block.has_confirmed_track_data);
  for (size_t i = 0; i < mesh_block.numVertices(); ++i) {
    EXPECT_EQ(mesh_block.labels.at(i), kTrackId);
    EXPECT_EQ(mesh_block.confirmed_track_ids.at(i).at(0), kTrackId);
    EXPECT_GT(mesh_block.confirmed_track_likelihoods.at(i).at(0), 0.0f);
  }

  kimera_pgmo::DeltaCompression compression(0.001);
  auto delta = compression.update(BlockMeshIter(map.getMeshLayer()), input.timestamp_ns);
  ASSERT_TRUE(delta);
  ASSERT_GT(delta->getNumVertices(), 0u);
  for (size_t i = 0; i < delta->getNumVertices(); ++i) {
    const auto& traits = delta->getVertex(i).traits;
    EXPECT_TRUE(traits.properties.has_label);
    EXPECT_TRUE(traits.properties.has_confirmed_track_data);
    EXPECT_EQ(traits.label, kTrackId);
    EXPECT_EQ(traits.confirmed_track_ids.at(0), kTrackId);
    EXPECT_GT(traits.confirmed_track_likelihoods.at(0), 0.0f);
  }

  TrackGraphSegmentUpdater::Config updater_config;
  updater_config.min_points_create = 1;
  updater_config.min_points_update = 1;
  updater_config.reidentifier.enabled = false;
  updater_config.historical_associator.enabled = false;
  updater_config.open_vocab.enabled = false;
  TrackGraphSegmentUpdater updater(updater_config);

  DynamicSceneGraph graph;
  graph.setMesh(std::make_shared<spark_dsg::Mesh>(true, true, true, false, true, true));
  kimera_pgmo::MeshOffsetInfo offsets;
  delta->updateMesh(*graph.mesh(), offsets);

  InputData object_input(createCamera());
  object_input.timestamp_ns = input.timestamp_ns;
  object_input.tracking_is_keyframe = true;
  object_input.track_observations.push_back(makeObservation(kTrackId, 1u));
  updater.update(input.timestamp_ns,
                 object_input,
                 *delta,
                 offsets,
                 graph);

  const auto object_nodes = graph.getLayer(DsgLayers::OBJECTS).numNodes();
  EXPECT_EQ(object_nodes, 1u);
}

TEST(TrackedProjectiveIntegrator, DatasetConfigSelectsTrackedIntegrator) {
  const auto config_path =
      (std::filesystem::path(HYDRA_PACKAGE_PATH) /
       "config/trackgraph_config/trackgraph_common_config.yaml")
          .string();
  const auto config =
      config::fromYamlFile<ReconstructionModule::Config>(config_path, "active_window");

  EXPECT_FALSE(config.volumetric_map.with_semantics);
  EXPECT_TRUE(config.volumetric_map.with_instances);

  auto integrator = config.tsdf.create();
  auto* tracked_integrator = dynamic_cast<TrackedProjectiveIntegrator*>(integrator.get());
  ASSERT_NE(tracked_integrator, nullptr);
  EXPECT_FLOAT_EQ(tracked_integrator->config.propagation_confidence_threshold, 0.6f);
}

}  // namespace hydra
