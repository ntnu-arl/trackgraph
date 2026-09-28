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
#pragma once
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace hydra::timing {

struct TimeStatistics {
  double last_s = 0.0;
  double mean_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
  double stddev_s = 0.0;
  size_t num_measurements = 0;
};

struct ElapsedStatistics {
  double last_s = 0.0;
  double mean_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
  double stddev_s = 0.0;
  size_t num_measurements = 0;
  TimeStatistics process_cpu;
  TimeStatistics thread_cpu;
};

struct CpuTimingSnapshot {
  std::chrono::nanoseconds process_cpu = std::chrono::nanoseconds(0);
  std::chrono::nanoseconds thread_cpu = std::chrono::nanoseconds(0);
  bool enabled = false;
};

std::ostream& operator<<(std::ostream& out, const TimeStatistics& stats);
std::ostream& operator<<(std::ostream& out, const ElapsedStatistics& stats);

class ElapsedTimeRecorder {
 public:
  struct Entry {
    uint64_t timestamp;
    std::chrono::nanoseconds elapsed;
    std::chrono::nanoseconds process_cpu_elapsed = std::chrono::nanoseconds(0);
    std::chrono::nanoseconds thread_cpu_elapsed = std::chrono::nanoseconds(0);
    bool cpu_timing_enabled = false;
    double elapsed_seconds() const;
    double elapsed_milliseconds() const;
    double process_cpu_seconds() const;
    double process_cpu_milliseconds() const;
    double thread_cpu_seconds() const;
    double thread_cpu_milliseconds() const;
  };

  static ElapsedTimeRecorder& instance();

  void reset();

  void start(const std::string& timer_name, const uint64_t timestamp);

  void stop(const std::string& timer_name);

  void record(const std::string& timer_name,
              const uint64_t timestamp,
              const std::chrono::nanoseconds elapsed,
              const std::chrono::nanoseconds process_cpu_elapsed =
                  std::chrono::nanoseconds(0),
              const std::chrono::nanoseconds thread_cpu_elapsed =
                  std::chrono::nanoseconds(0),
              bool cpu_timing_enabled = false);

  std::vector<std::string> timerNames() const;

  std::optional<double> getLastElapsed(const std::string& timer_name) const;

  std::optional<Entry> getLastEntry(const std::string& timer_name) const;

  ElapsedStatistics getStats(const std::string& timer_name) const;

  std::string printAllStats() const;

  void logTimers(const std::filesystem::path& output,
                 const std::string& name_prefix = "",
                 const std::string& name_suffix = "_timing_raw") const;

  void logStats(const std::filesystem::path& stat_filepath) const;

  //! Whether or not timing is enabled
  bool timing_disabled;

  //! Whether or not process/thread CPU timing is enabled
  bool cpu_timing_enabled;

  //! Whether or not timers log to console
  bool disable_output;

 private:
  struct TimePoint {
    uint64_t timestamp;
    std::chrono::time_point<std::chrono::high_resolution_clock> now;
    CpuTimingSnapshot cpu;
    std::thread::id thread_id;
  };

  ElapsedTimeRecorder();

  void add(const std::string& name,
           const uint64_t timestamp,
           const std::chrono::nanoseconds elapsed,
           const std::chrono::nanoseconds process_cpu_elapsed,
           const std::chrono::nanoseconds thread_cpu_elapsed,
           bool cpu_timing_enabled);

  mutable std::mutex mutex_;
  static std::unique_ptr<ElapsedTimeRecorder> instance_;

  std::map<std::string, TimePoint> starts_;
  std::map<std::string, std::vector<Entry>> elapsed_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& name, uint64_t timestamp);

  ScopedTimer(const std::string& name,
              uint64_t timestamp,
              bool verbose,
              int verbosity,
              bool elapsed_only = true,
              bool verbosity_disables = false);

  ~ScopedTimer();

  void start();
  void stop();
  void reset(const std::string& name);
  void reset(const std::string& name, uint64_t timestamp);

 private:
  std::string name_;
  uint64_t timestamp_;
  bool verbose_;
  int verbosity_;
  bool elapsed_only_;
  bool verbosity_disables_;
  bool is_running_ = false;
};

}  // namespace hydra::timing
