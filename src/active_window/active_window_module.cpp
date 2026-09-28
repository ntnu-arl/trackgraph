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
#include "hydra/active_window/active_window_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>

#include <malloc.h>

#include <type_traits>

#include "hydra/common/global_info.h"

namespace hydra {

void declare_config(ActiveWindowModule::Config& config) {
  using namespace config;
  name("ActiveWindowModule::Config");
  field(config.max_input_queue_size, "max_input_queue_size");
  field(config.volumetric_map, "volumetric_map");
  config.map_window.setOptional();
  field(config.map_window, "map_window");
  field(config.sinks, "sinks");
  field(config.validate_mesh_fields, "validate_mesh_fields");
  field(config.log_memory_stats, "log_memory_stats");
  field(config.memory_stats_interval, "memory_stats_interval");
  check(config.memory_stats_interval, GT, 0, "memory_stats_interval");
}

ActiveWindowModule::Config::Config(bool with_semantics,
                                   bool with_tracking,
                                   bool with_instances)
    : max_input_queue_size(0),
      volumetric_map({0.1, 16, 0.3, with_semantics, with_tracking, with_instances}) {}

ActiveWindowModule::ActiveWindowModule(const Config& config,
                                       const OutputQueue::Ptr& queue)
    : config(config::checkValid(config)),
      input_queue_(new InputQueue(config.max_input_queue_size, config.log_memory_stats)),
      output_queue_(queue),
      sinks_(Sink::instantiate(config.sinks)),
      map_(config.volumetric_map),
      map_window_(config.map_window ? config.map_window.create()
                                    : GlobalInfo::instance().createVolumetricWindow()) {
  const auto mesh_config = GlobalInfo::instance().getConfig().mesh;
  if (config.validate_mesh_fields) {
    // double-check that the frontend and backend will actually receive labels and
    // first-seen stamps
    if ((config.volumetric_map.with_semantics || config.volumetric_map.with_instances) &&
        !mesh_config.with_labels) {
      LOG(FATAL) << "Volumetric map contains label-bearing layers, but mesh does not "
                    "have labels enabled!";
    }

    if (config.volumetric_map.with_tracking && !mesh_config.with_first_seen_stamps) {
      LOG(FATAL) << "Volumetric map contains tracking layer, but mesh does not have "
                    "first-seen stamps enabled!";
    }
  }
}

void ActiveWindowModule::start() {
  spin_thread_.reset(new std::thread(&ActiveWindowModule::spin, this));
  LOG(INFO) << "[Active Window] started!";
}

void ActiveWindowModule::stop() { stopImpl(); }

void ActiveWindowModule::stopImpl() {
  should_shutdown_ = true;

  if (spin_thread_) {
    VLOG(2) << "[Active Window] stopping!";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "[Active Window] stopped!";
  }

  if (config.log_memory_stats) {
    logMemoryStats(0, true);
  }

  VLOG(2) << "[Active Window] input queue: " << input_queue_->size();
  if (output_queue_) {
    VLOG(2) << "[Active Window] output queue: " << output_queue_->size();
  } else {
    VLOG(2) << "[Active Window] output queue: n/a";
  }
}

std::string ActiveWindowModule::printInfo() const {
  return config::toString(config) + "\n" + Sink::printSinks(sinks_);
}

void ActiveWindowModule::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    bool has_data = input_queue_->poll();
    if (hydra::GlobalInfo::instance().force_shutdown() || !has_data) {
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    const auto msg = input_queue_->pop();
    auto output = spinOnce(*msg);
    ++processed_inputs_;
    if (config.log_memory_stats &&
        processed_inputs_ % config.memory_stats_interval == 0) {
      logMemoryStats(msg->timestamp_ns);
    }
    if (!output) {
      continue;
    }

    Sink::callAll(sinks_, msg->timestamp_ns, map_, *output);
    if (output_queue_) {
      output_queue_->push(output);
    }
  }
}

bool ActiveWindowModule::step(const InputPacket::Ptr& msg) {
  if (!msg) {
    return false;
  }

  auto output = spinOnce(*msg);
  ++processed_inputs_;
  if (config.log_memory_stats &&
      processed_inputs_ % config.memory_stats_interval == 0) {
    logMemoryStats(msg->timestamp_ns);
  }
  if (!output) {
    return false;
  }

  Sink::callAll(sinks_, msg->timestamp_ns, map_, *output);
  if (output_queue_) {
    output_queue_->push(output);
  }

  return true;
}

void ActiveWindowModule::addSink(const Sink::Ptr& sink) {
  if (sink) {
    sinks_.push_back(sink);
  }
}

void ActiveWindowModule::logMemoryStats(uint64_t timestamp_ns, bool final) const {
  const auto& tsdf = map_.getTsdfLayer();
  const auto* instances = map_.getInstanceLayer();
  const auto& mesh = map_.getMeshLayer();

  const size_t voxels_per_block = config.volumetric_map.voxels_per_side *
                                  config.volumetric_map.voxels_per_side *
                                  config.volumetric_map.voxels_per_side;
  const size_t tsdf_payload_bytes =
      tsdf.numBlocks() * voxels_per_block * sizeof(TsdfVoxel);
  const size_t instance_payload_bytes =
      (instances ? instances->numBlocks() : 0) * voxels_per_block *
      sizeof(InstanceVoxel);

  size_t observed_tsdf_voxels = 0;
  for (const auto& block : tsdf) {
    for (const auto& voxel : block) {
      observed_tsdf_voxels += voxel.weight > 0.0f;
    }
  }

  size_t initialized_instance_voxels = 0;
  size_t raw_track_slots = 0;
  size_t confirmed_track_slots = 0;
  if (instances) {
    for (const auto& block : *instances) {
      for (const auto& voxel : block) {
        initialized_instance_voxels += !voxel.empty;
        for (const auto track_id : voxel.track_ids) {
          raw_track_slots += track_id != InstanceVoxel::NO_TRACK;
        }
        for (const auto track_id : voxel.confirmed_track_ids) {
          confirmed_track_slots += track_id != InstanceVoxel::NO_TRACK;
        }
      }
    }
  }

  size_t mesh_vertices = 0;
  size_t mesh_faces = 0;
  size_t mesh_size_bytes = 0;
  size_t mesh_capacity_bytes = 0;
  for (const auto& block : mesh) {
    mesh_vertices += block.numVertices();
    mesh_faces += block.numFaces();
    mesh_size_bytes += block.totalBytes();
    const auto add_capacity = [&mesh_capacity_bytes](const auto& values) {
      using Container = std::decay_t<decltype(values)>;
      mesh_capacity_bytes +=
          values.capacity() * sizeof(typename Container::value_type);
    };
    add_capacity(block.points);
    add_capacity(block.colors);
    add_capacity(block.stamps);
    add_capacity(block.first_seen_stamps);
    add_capacity(block.labels);
    add_capacity(block.track_ids);
    add_capacity(block.track_likelihoods);
    add_capacity(block.confirmed_track_ids);
    add_capacity(block.confirmed_track_likelihoods);
    add_capacity(block.faces);
  }

  const auto allocator = mallinfo2();
  const auto input_queue_stats = input_queue_->statistics();
  const auto output_queue_stats =
      output_queue_ ? output_queue_->statistics() : OutputQueue::Statistics{};
  LOG(INFO) << "[MemoryStats][ActiveWindow] final=" << final
            << " processed_inputs=" << processed_inputs_
            << " timestamp_ns=" << timestamp_ns
            << " input_queue_size=" << input_queue_->size()
            << " input_queue_peak=" << input_queue_stats.peak_depth
            << " input_queue_pushed=" << input_queue_stats.total_pushed
            << " input_queue_popped=" << input_queue_stats.total_popped
            << " output_queue_size="
            << (output_queue_ ? output_queue_->size() : 0)
            << " output_queue_peak=" << output_queue_stats.peak_depth
            << " tsdf_blocks=" << tsdf.numBlocks()
            << " tsdf_payload_bytes=" << tsdf_payload_bytes
            << " observed_tsdf_voxels=" << observed_tsdf_voxels
            << " instance_blocks=" << (instances ? instances->numBlocks() : 0)
            << " instance_payload_bytes=" << instance_payload_bytes
            << " initialized_instance_voxels=" << initialized_instance_voxels
            << " raw_track_slots=" << raw_track_slots
            << " confirmed_track_slots=" << confirmed_track_slots
            << " mesh_blocks=" << mesh.numBlocks()
            << " mesh_vertices=" << mesh_vertices
            << " mesh_faces=" << mesh_faces
            << " mesh_size_bytes=" << mesh_size_bytes
            << " mesh_capacity_bytes=" << mesh_capacity_bytes
            << " malloc_arena_bytes=" << allocator.arena
            << " malloc_in_use_bytes=" << allocator.uordblks
            << " malloc_free_bytes=" << allocator.fordblks
            << " malloc_mmap_bytes=" << allocator.hblkhd
            << " malloc_releasable_bytes=" << allocator.keepcost;
}

}  // namespace hydra
