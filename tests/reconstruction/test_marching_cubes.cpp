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
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>
#include <hydra/reconstruction/marching_cubes.h>
#include <hydra/reconstruction/mesh_integrator.h>

#include <set>

namespace hydra {

static constexpr float TEST_TOLERANCE = 1.0e-6f;

using PointMatrix = Eigen::Matrix<float, 3, 8>;
using SdfMatrix = std::array<float, 8>;

void fillPointsFromMatrices(const PointMatrix& pos,
                            const SdfMatrix& sdf,
                            MarchingCubes::SdfPoints& points) {
  for (size_t i = 0; i < 8; ++i) {
    points[i].pos = pos.col(i);
    points[i].distance = sdf[i];
    points[i].weight = 1.0;
  }
}

TEST(MarchingCubes, EdgeInterpolation) {
  // add zero-crossings at just the bottom right? corner
  SdfMatrix sdf_values{-1.0, 1.0, 10.0, 2.0, 3.0, 10.0, 10.0, 10.0};

  // make vertices for everything that should be used for interpolation
  PointMatrix vertex_coordinates = PointMatrix::Zero();
  vertex_coordinates.col(0) << -1.0, -1.0, -1.0;
  vertex_coordinates.col(1) << 1.0, 1.0, 1.0;
  vertex_coordinates.col(3) << 1.0, 1.0, 1.0;
  vertex_coordinates.col(4) << 1.0, 1.0, 1.0;

  MarchingCubes::SdfPoints sdf_points;
  fillPointsFromMatrices(vertex_coordinates, sdf_values, sdf_points);
  MarchingCubes::EdgePoints edge_coords;
  for (size_t i = 0; i < edge_coords.size(); ++i) {
    edge_coords[i].pos.setZero();
  }

  MarchingCubes::EdgeStatus edge_status;
  MarchingCubes::interpolateEdges(sdf_points, edge_coords, edge_status);

  std::set<int> valid_edges{0, 3, 8};
  for (size_t i = 0; i < edge_coords.size(); ++i) {
    if (valid_edges.count(i)) {
      continue;
    }

    EXPECT_EQ(0u, edge_status[i]);
    EXPECT_EQ(0.0f, edge_coords[i].pos.norm())
        << "i: " << edge_coords[i].pos.transpose();
  }

  // see kEdgeIndexPairs for edge index to voxel index mapping
  EXPECT_EQ(3u, edge_status[0]);  // both voxels are valid
  EXPECT_EQ(2u, edge_status[3]);  // the second voxel is valid
  EXPECT_EQ(1u, edge_status[8]);  // the first voxel is valid

  Eigen::Vector3f expected_edge0;
  expected_edge0 << 0.0f, 0.0f, 0.0f;
  const auto& result0 = edge_coords[0].pos;
  EXPECT_NEAR(0.0f, (expected_edge0 - result0).norm(), TEST_TOLERANCE)
      << "0: " << result0.transpose();

  Eigen::Vector3f expected_edge3;
  expected_edge3 << -1.0f / 3.0f, -1.0f / 3.0f, -1.0f / 3.0f;
  const auto& result3 = edge_coords[3].pos;
  EXPECT_NEAR(0.0f, (expected_edge3 - result3).norm(), TEST_TOLERANCE)
      << "3: " << result3.transpose();

  Eigen::Vector3f expected_edge8;
  expected_edge8 << -1.0f / 2.0f, -1.0f / 2.0f, -1.0f / 2.0f;
  const auto& result8 = edge_coords[8].pos;
  EXPECT_NEAR(0.0f, (expected_edge8 - result8).norm(), TEST_TOLERANCE)
      << "8: " << result8.transpose();
}

TEST(MarchingCubes, CubeMeshingNearestVertexIndexCorrect) {
  // add zero-crossings at just the bottom right? corner
  SdfMatrix sdf_values{-1.0, 1.0, 10.0, 2.0, 3.0, 10.0, 10.0, 10.0};

  // make vertices for everything that should be used for interpolation
  PointMatrix vertex_coordinates = PointMatrix::Zero();
  vertex_coordinates.col(0) << -1.0, -1.0, -1.0;
  vertex_coordinates.col(1) << 1.0, 1.0, 1.0;
  vertex_coordinates.col(3) << 1.0, 1.0, 1.0;
  vertex_coordinates.col(4) << 1.0, 1.0, 1.0;

  MarchingCubes::SdfPoints sdf_points;
  fillPointsFromMatrices(vertex_coordinates, sdf_values, sdf_points);
  OccupancyVoxel actual_voxels[8];
  for (size_t i = 0; i < 8; ++i) {
    sdf_points[i].vertex_voxel = &(actual_voxels[i]);
  }

  Mesh mesh;
  BlockIndex block = BlockIndex::Zero();
  MarchingCubes::meshCube(block, sdf_points, mesh);
  EXPECT_EQ(3u, mesh.numVertices());

  EXPECT_TRUE(actual_voxels[0].on_surface);
  EXPECT_TRUE(actual_voxels[1].on_surface);
  // the last vertex overwrites the index for 0, and is the only valid vertex for 1
  EXPECT_EQ(2u, actual_voxels[0].block_vertex_index);
  EXPECT_EQ(2u, actual_voxels[1].block_vertex_index);
}

TEST(MarchingCubes, CubeMeshingCopiesTrackDataFromWinningEndpoint) {
  SdfMatrix sdf_values{-1.0, 1.0, 10.0, 2.0, 3.0, 10.0, 10.0, 10.0};

  PointMatrix vertex_coordinates = PointMatrix::Zero();
  vertex_coordinates.col(0) << -1.0, -1.0, -1.0;
  vertex_coordinates.col(1) << 1.0, 1.0, 1.0;
  vertex_coordinates.col(3) << 1.0, 1.0, 1.0;
  vertex_coordinates.col(4) << 1.0, 1.0, 1.0;

  MarchingCubes::SdfPoints sdf_points;
  fillPointsFromMatrices(vertex_coordinates, sdf_values, sdf_points);
  for (auto& point : sdf_points) {
    point.track_ids = spark_dsg::Mesh::makeEmptyTrackIds();
    point.track_likelihoods = spark_dsg::Mesh::makeEmptyTrackLikelihoods();
    point.confirmed_track_ids = spark_dsg::Mesh::makeEmptyConfirmedTrackIds();
    point.confirmed_track_likelihoods = spark_dsg::Mesh::makeEmptyConfirmedTrackLikelihoods();
  }

  sdf_points[0].label = 10u;
  sdf_points[0].has_track_data = true;
  sdf_points[0].track_ids = {{10u, 11u, 12u, InstanceVoxel::NO_TRACK}};
  sdf_points[0].track_likelihoods = {{3.0f, 2.0f, 1.0f, 0.0f}};
  sdf_points[0].has_confirmed_track_data = true;
  sdf_points[0].confirmed_track_ids = {{10u, 11u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[0].confirmed_track_likelihoods = {{2.5f, 1.0f, 0.0f, 0.0f}};
  sdf_points[0].weight = 1.0f;

  sdf_points[1].label = 20u;
  sdf_points[1].has_track_data = true;
  sdf_points[1].track_ids = {{20u, 21u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[1].track_likelihoods = {{4.0f, 1.5f, 0.0f, 0.0f}};
  sdf_points[1].has_confirmed_track_data = true;
  sdf_points[1].confirmed_track_ids = {{20u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[1].confirmed_track_likelihoods = {{3.5f, 0.0f, 0.0f, 0.0f}};
  sdf_points[1].weight = 2.0f;

  sdf_points[3].label = 30u;
  sdf_points[3].has_track_data = true;
  sdf_points[3].track_ids = {{30u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[3].track_likelihoods = {{9.0f, 0.0f, 0.0f, 0.0f}};
  sdf_points[3].has_confirmed_track_data = true;
  sdf_points[3].confirmed_track_ids = {{30u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[3].confirmed_track_likelihoods = {{8.0f, 0.0f, 0.0f, 0.0f}};

  sdf_points[4].label = 40u;
  sdf_points[4].has_track_data = true;
  sdf_points[4].track_ids = {{40u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[4].track_likelihoods = {{8.0f, 0.0f, 0.0f, 0.0f}};
  sdf_points[4].has_confirmed_track_data = true;
  sdf_points[4].confirmed_track_ids = {{40u, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK, InstanceVoxel::NO_TRACK}};
  sdf_points[4].confirmed_track_likelihoods = {{7.0f, 0.0f, 0.0f, 0.0f}};

  Mesh mesh(true, false, true, false, true, true);
  BlockIndex block = BlockIndex::Zero();
  MarchingCubes::meshCube(block, sdf_points, mesh);

  ASSERT_EQ(3u, mesh.numVertices());
  ASSERT_EQ(3u, mesh.labels.size());
  ASSERT_EQ(3u, mesh.track_ids.size());
  ASSERT_EQ(3u, mesh.track_likelihoods.size());
  ASSERT_EQ(3u, mesh.confirmed_track_ids.size());
  ASSERT_EQ(3u, mesh.confirmed_track_likelihoods.size());

  EXPECT_EQ(10u, mesh.labels[0]);
  EXPECT_EQ(10u, mesh.labels[1]);
  EXPECT_EQ(20u, mesh.labels[2]);

  EXPECT_EQ(sdf_points[0].track_ids, mesh.track_ids[0]);
  EXPECT_EQ(sdf_points[0].track_ids, mesh.track_ids[1]);
  EXPECT_EQ(sdf_points[1].track_ids, mesh.track_ids[2]);

  EXPECT_EQ(sdf_points[0].track_likelihoods, mesh.track_likelihoods[0]);
  EXPECT_EQ(sdf_points[0].track_likelihoods, mesh.track_likelihoods[1]);
  EXPECT_EQ(sdf_points[1].track_likelihoods, mesh.track_likelihoods[2]);

  EXPECT_EQ(sdf_points[0].confirmed_track_ids, mesh.confirmed_track_ids[0]);
  EXPECT_EQ(sdf_points[0].confirmed_track_ids, mesh.confirmed_track_ids[1]);
  EXPECT_EQ(sdf_points[1].confirmed_track_ids, mesh.confirmed_track_ids[2]);

  EXPECT_EQ(sdf_points[0].confirmed_track_likelihoods, mesh.confirmed_track_likelihoods[0]);
  EXPECT_EQ(sdf_points[0].confirmed_track_likelihoods, mesh.confirmed_track_likelihoods[1]);
  EXPECT_EQ(sdf_points[1].confirmed_track_likelihoods, mesh.confirmed_track_likelihoods[2]);
}

}  // namespace hydra
