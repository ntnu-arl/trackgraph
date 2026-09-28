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
#include "hydra/utils/timing_utilities.h"

#include <glog/logging.h>
#include <glog/stl_logging.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

namespace hydra::timing {

bool operator<(const ElapsedTimeRecorder::Entry& lhs,
               const ElapsedTimeRecorder::Entry& rhs) {
  return lhs.elapsed < rhs.elapsed;
}

namespace {

using Entries = std::vector<ElapsedTimeRecorder::Entry>;

double toSeconds(const std::chrono::nanoseconds elapsed) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
}

std::chrono::nanoseconds clampSubtract(const std::chrono::nanoseconds lhs,
                                       const std::chrono::nanoseconds rhs) {
  return lhs > rhs ? lhs - rhs : std::chrono::nanoseconds(0);
}

CpuTimingSnapshot sampleCpuTiming(bool enabled) {
  CpuTimingSnapshot result;
  result.enabled = enabled;
  if (!enabled) {
    return result;
  }

  timespec process_ts{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_ts) == 0) {
    result.process_cpu = std::chrono::seconds(process_ts.tv_sec) +
                         std::chrono::nanoseconds(process_ts.tv_nsec);
  }

  timespec thread_ts{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread_ts) == 0) {
    result.thread_cpu = std::chrono::seconds(thread_ts.tv_sec) +
                        std::chrono::nanoseconds(thread_ts.tv_nsec);
  }

  return result;
}

template <typename Getter, typename Include>
TimeStatistics computeTimeStats(const Entries& entries, Getter getter, Include include) {
  std::vector<double> values;
  values.reserve(entries.size());
  for (const auto& entry : entries) {
    if (include(entry)) {
      values.push_back(toSeconds(getter(entry)));
    }
  }

  if (values.empty()) {
    return {};
  }

  const auto last_elapsed = values.back();
  if (values.size() == 1) {
    return {last_elapsed, last_elapsed, last_elapsed, last_elapsed, 0.0, 1};
  }

  const size_t N = values.size();
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(N);

  const double variance = std::accumulate(
      values.begin(), values.end(), 0.0, [&](double total, const auto& elapsed_s) {
        const double mean_diff = elapsed_s - mean;
        return total + (mean_diff * mean_diff / N);
      });

  const auto min_entry = std::min_element(values.begin(), values.end());
  const auto max_entry = std::max_element(values.begin(), values.end());
  return {last_elapsed, mean, *min_entry, *max_entry, std::sqrt(variance), N};
}

ElapsedStatistics computeStats(const Entries& entries) {
  ElapsedStatistics result;
  const auto wall = computeTimeStats(
      entries, [](const auto& entry) { return entry.elapsed; }, [](const auto&) {
        return true;
      });
  result.last_s = wall.last_s;
  result.mean_s = wall.mean_s;
  result.min_s = wall.min_s;
  result.max_s = wall.max_s;
  result.stddev_s = wall.stddev_s;
  result.num_measurements = wall.num_measurements;
  result.process_cpu =
      computeTimeStats(entries,
                       [](const auto& entry) { return entry.process_cpu_elapsed; },
                       [](const auto& entry) { return entry.cpu_timing_enabled; });
  result.thread_cpu =
      computeTimeStats(entries,
                       [](const auto& entry) { return entry.thread_cpu_elapsed; },
                       [](const auto& entry) { return entry.cpu_timing_enabled; });
  return result;
}

void writeEntries(const std::filesystem::path& output_csv, const Entries& entries) {
  std::ofstream output_file;
  output_file.open(output_csv);
  if (!output_file.is_open()) {
    LOG(ERROR) << "Could not open file " << output_csv << " for writing";
    return;
  }

  std::stringstream ss;
  ss << "timestamp(ns),elapsed(s),process_cpu_elapsed(s),thread_cpu_elapsed(s),"
        "cpu_timing_enabled\n";
  for (const auto& entry : entries) {
    ss << entry.timestamp << "," << entry.elapsed_seconds() << ","
       << entry.process_cpu_seconds() << "," << entry.thread_cpu_seconds() << ","
       << (entry.cpu_timing_enabled ? 1 : 0) << "\n";
  }

  output_file << ss.str();
  output_file.close();
}

std::filesystem::path getTimerPath(const std::filesystem::path& output,
                                   std::string name,
                                   const std::string& prefix,
                                   const std::string& suffix) {
  // replace all '/' with '_' to avoid creating a directory.
  std::transform(name.cbegin(), name.cend(), name.begin(), [](const char c) {
    return c == '/' ? '_' : c;
  });
  return (output / std::filesystem::path(prefix + name + suffix + ".csv"))
      .lexically_normal();
}

}  // namespace

decltype(ElapsedTimeRecorder::instance_) ElapsedTimeRecorder::instance_;

std::ostream& operator<<(std::ostream& out, const TimeStatistics& stats) {
  if (!stats.num_measurements) {
    out << "N/A [s] over 0 call(s)";
    return out;
  }

  out << stats.mean_s;
  if (stats.num_measurements > 1) {
    out << " +/- " << stats.stddev_s;
    out << " [" << stats.min_s << ", " << stats.max_s << "]";
  }

  out << " [s] over " << stats.num_measurements << " call(s) (last: " << stats.last_s
      << " [s])";
  return out;
}

std::ostream& operator<<(std::ostream& out, const ElapsedStatistics& stats) {
  TimeStatistics wall;
  wall.last_s = stats.last_s;
  wall.mean_s = stats.mean_s;
  wall.min_s = stats.min_s;
  wall.max_s = stats.max_s;
  wall.stddev_s = stats.stddev_s;
  wall.num_measurements = stats.num_measurements;
  out << "wall=" << wall;
  if (stats.process_cpu.num_measurements) {
    out << ", process_cpu=" << stats.process_cpu
        << ", thread_cpu=" << stats.thread_cpu;
  }
  return out;
}

double ElapsedTimeRecorder::Entry::elapsed_seconds() const {
  return toSeconds(elapsed);
}

double ElapsedTimeRecorder::Entry::elapsed_milliseconds() const {
  return 1000.0 * elapsed_seconds();
}

double ElapsedTimeRecorder::Entry::process_cpu_seconds() const {
  return toSeconds(process_cpu_elapsed);
}

double ElapsedTimeRecorder::Entry::process_cpu_milliseconds() const {
  return 1000.0 * process_cpu_seconds();
}

double ElapsedTimeRecorder::Entry::thread_cpu_seconds() const {
  return toSeconds(thread_cpu_elapsed);
}

double ElapsedTimeRecorder::Entry::thread_cpu_milliseconds() const {
  return 1000.0 * thread_cpu_seconds();
}

ElapsedTimeRecorder::ElapsedTimeRecorder()
    : timing_disabled(false), cpu_timing_enabled(false), disable_output(true) {}

ElapsedTimeRecorder& ElapsedTimeRecorder::instance() {
  if (!instance_) {
    instance_.reset(new ElapsedTimeRecorder());
  }

  return *instance_;
}

void ElapsedTimeRecorder::reset() { instance_.reset(new ElapsedTimeRecorder()); }

void ElapsedTimeRecorder::start(const std::string& name, const uint64_t timestamp) {
  const auto start_cpu = sampleCpuTiming(cpu_timing_enabled);
  {  // start critical section
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = starts_.find(name);
    if (iter == starts_.end()) {
      starts_.emplace(name,
                      TimePoint{timestamp,
                                std::chrono::high_resolution_clock::now(),
                                start_cpu,
                                std::this_thread::get_id()});
      return;  // break to avoid error statement
    }
  }  // end critical section

  LOG(ERROR) << "Timer '" << name << "' was already started. Discarding time point!";
}

void ElapsedTimeRecorder::stop(const std::string& name) {
  // we grab the time point first (to not mess up timing with later processing)
  const auto stop_point = std::chrono::high_resolution_clock::now();
  const auto stop_cpu = sampleCpuTiming(cpu_timing_enabled);
  {  // start critical section
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = starts_.find(name);
    if (iter != starts_.end()) {
      const bool cpu_enabled = iter->second.cpu.enabled && stop_cpu.enabled;
      const auto process_cpu_elapsed =
          cpu_enabled ? clampSubtract(stop_cpu.process_cpu, iter->second.cpu.process_cpu)
                      : std::chrono::nanoseconds(0);
      const auto thread_cpu_elapsed =
          cpu_enabled && iter->second.thread_id == std::this_thread::get_id()
              ? clampSubtract(stop_cpu.thread_cpu, iter->second.cpu.thread_cpu)
              : std::chrono::nanoseconds(0);
      add(name,
          iter->second.timestamp,
          stop_point - iter->second.now,
          process_cpu_elapsed,
          thread_cpu_elapsed,
          cpu_enabled);
      starts_.erase(iter);
      return;  // break to avoid error statement
    }
  }  // end critical section

  LOG(ERROR) << "Timer '" << name << "' was not started. Discarding time point!";
}

void ElapsedTimeRecorder::record(const std::string& name,
                                 const uint64_t timestamp,
                                 const std::chrono::nanoseconds elapsed,
                                 const std::chrono::nanoseconds process_cpu_elapsed,
                                 const std::chrono::nanoseconds thread_cpu_elapsed,
                                 bool cpu_timing_enabled) {
  std::unique_lock<std::mutex> lock(mutex_);
  add(name,
      timestamp,
      elapsed,
      process_cpu_elapsed,
      thread_cpu_elapsed,
      cpu_timing_enabled);
}

std::vector<std::string> ElapsedTimeRecorder::timerNames() const {
  std::vector<std::string> names;
  {  // start critical section
    std::unique_lock<std::mutex> lock(mutex_);
    std::transform(elapsed_.begin(),
                   elapsed_.end(),
                   std::back_inserter(names),
                   [](const auto& entry) { return entry.first; });
  }

  return names;
}

std::optional<double> ElapsedTimeRecorder::getLastElapsed(const std::string& n) const {
  {  // start critical section
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = elapsed_.find(n);
    if (iter != elapsed_.end()) {
      // NOTE(nathan) invariant that every timer has at least one sample
      return iter->second.back().elapsed_seconds();
    }
  }  // end critical section

  return std::nullopt;
}

std::optional<ElapsedTimeRecorder::Entry> ElapsedTimeRecorder::getLastEntry(
    const std::string& n) const {
  {  // start critical section
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = elapsed_.find(n);
    if (iter != elapsed_.end()) {
      return iter->second.back();
    }
  }  // end critical section

  return std::nullopt;
}

ElapsedStatistics ElapsedTimeRecorder::getStats(const std::string& name) const {
  std::vector<Entry> durations;
  {  // start critical section
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = elapsed_.find(name);
    if (iter != elapsed_.end()) {
      durations = iter->second;
    }
  }  // end critical section

  return computeStats(durations);
}

std::string ElapsedTimeRecorder::printAllStats() const {
  std::stringstream ss;
  const auto timers = timerNames();
  for (const auto& name : timers) {
    const auto stats = getStats(name);
    ss << name << ": " << stats << std::endl;
  }

  return ss.str();
};

//! Suffix for individual timing files
std::string timing_suffix = "_timing_raw.csv";
/**
 * If true log all timers into a single directory, replacing '/' with '_' in the
 * names. If false create separate directories for separators '/' (default).
 */
bool log_raw_timers_to_single_dir = false;

void ElapsedTimeRecorder::logTimers(const std::filesystem::path& output,
                                    const std::string& name_prefix,
                                    const std::string& name_suffix) const {
  const auto all_timers = timerNames();
  VLOG(5) << "Saving timers: [" << all_timers << "]";
  for (const auto& name : all_timers) {
    VLOG(5) << "Saving timer '" << name << "'";

    std::vector<Entry> entries;
    {  // start critical section
      std::unique_lock<std::mutex> lock(mutex_);
      auto iter = elapsed_.find(name);
      if (iter != elapsed_.end()) {
        entries = iter->second;
      }
    }  // end critical section

    if (entries.empty()) {
      LOG(ERROR) << "Invalid timer encountered while saving '" << name << "'";
      continue;
    }

    const auto output_csv = getTimerPath(output, name, name_prefix, name_suffix);
    VLOG(2) << "Writing " << entries.size() << " measurements for timer '" << name
            << "' to '" << output_csv << "'";
    writeEntries(output_csv, entries);
    VLOG(5) << "Saved timer '" << name << "'";
  }
}

void ElapsedTimeRecorder::logStats(const std::filesystem::path& filename) const {
  std::ofstream output_file;
  output_file.open(filename);

  // file format
  auto append_ms_stats = [](std::stringstream& ss, const TimeStatistics& stats) {
    if (!stats.num_measurements) {
      ss << "nan,nan,nan,nan,nan,0";
      return;
    }

    ss << 1000.0 * stats.mean_s << "," << 1000.0 * stats.min_s << ","
       << 1000.0 * stats.max_s << "," << 1000.0 * stats.stddev_s << ","
       << 1000.0 * stats.last_s << "," << stats.num_measurements;
  };

  std::stringstream ss;
  ss << "name,mean[s],min[s],max[s],std-dev[s],last[s],mean[ms],min[ms],"
        "max[ms],std-dev[ms],last[ms],num-measurements,"
        "process-cpu-mean[ms],process-cpu-min[ms],process-cpu-max[ms],"
        "process-cpu-std-dev[ms],process-cpu-last[ms],"
        "process-cpu-num-measurements,thread-cpu-mean[ms],thread-cpu-min[ms],"
        "thread-cpu-max[ms],thread-cpu-std-dev[ms],thread-cpu-last[ms],"
        "thread-cpu-num-measurements\n";
  for (const auto& str_timer_pair : elapsed_) {
    const ElapsedStatistics& stats = getStats(str_timer_pair.first);
    ss << str_timer_pair.first << "," << stats.mean_s << "," << stats.min_s << ","
       << stats.max_s << "," << stats.stddev_s << "," << stats.last_s << ","
       << 1000.0 * stats.mean_s << "," << 1000.0 * stats.min_s << ","
       << 1000.0 * stats.max_s << "," << 1000.0 * stats.stddev_s << ","
       << 1000.0 * stats.last_s << "," << stats.num_measurements << ",";
    append_ms_stats(ss, stats.process_cpu);
    ss << ",";
    append_ms_stats(ss, stats.thread_cpu);
    ss << "\n";
  }
  output_file << ss.str();
  output_file.close();
}

// NOTE(nathan) this is intentionally NOT threadsafe, the callee is responsible
// for locking the mutex
void ElapsedTimeRecorder::add(const std::string& name,
                              const uint64_t timestamp,
                              const std::chrono::nanoseconds elapsed,
                              const std::chrono::nanoseconds process_cpu_elapsed,
                              const std::chrono::nanoseconds thread_cpu_elapsed,
                              bool cpu_timing_enabled) {
  auto iter = elapsed_.find(name);
  if (iter == elapsed_.end()) {
    iter = elapsed_.emplace(name, std::vector<Entry>()).first;
  }

  iter->second.emplace_back(Entry{timestamp,
                                  elapsed,
                                  process_cpu_elapsed,
                                  thread_cpu_elapsed,
                                  cpu_timing_enabled});
}

ScopedTimer::ScopedTimer(const std::string& name,
                         uint64_t timestamp,
                         bool verbose,
                         int verbosity,
                         bool elapsed_only,
                         bool verbosity_disables)
    : name_(name),
      timestamp_(timestamp),
      verbose_(verbose),
      verbosity_(verbosity),
      elapsed_only_(elapsed_only),
      verbosity_disables_(verbosity_disables) {
  start();
}

ScopedTimer::ScopedTimer(const std::string& name, uint64_t timestamp)
    : ScopedTimer(name, timestamp, false, 1, true, false) {}

ScopedTimer::~ScopedTimer() { stop(); }

void ScopedTimer::start() {
  if (is_running_) {
    return;
  }

  if (ElapsedTimeRecorder::instance().timing_disabled) {
    return;
  }

  if (verbosity_disables_ and !VLOG_IS_ON(verbosity_)) {
    return;
  }

  ElapsedTimeRecorder::instance().start(name_, timestamp_);
  is_running_ = true;
}

void ScopedTimer::stop() {
  if (!is_running_) {
    return;
  }

  if (ElapsedTimeRecorder::instance().timing_disabled) {
    return;
  }

  if (verbosity_disables_ and !VLOG_IS_ON(verbosity_)) {
    return;
  }

  is_running_ = false;
  ElapsedTimeRecorder::instance().stop(name_);
  if (!verbose_) {
    return;
  }

  if (ElapsedTimeRecorder::instance().disable_output) {
    return;
  }

  if (!VLOG_IS_ON(verbosity_)) {
    return;
  }

  if (elapsed_only_) {
    VLOG(verbosity_) << "{Timer " << name_
                     << "}: " << *ElapsedTimeRecorder::instance().getLastElapsed(name_)
                     << " [s] elapsed";
  } else {
    VLOG(verbosity_) << "{Timer " << name_
                     << "}: " << ElapsedTimeRecorder::instance().getStats(name_);
  }
}

void ScopedTimer::reset(const std::string& name) {
  stop();
  name_ = name;
  start();
}

void ScopedTimer::reset(const std::string& name, uint64_t timestamp) {
  stop();
  name_ = name;
  timestamp_ = timestamp;
  start();
}

}  // namespace hydra::timing
