#pragma once

#include <hydra/openset/openset_types.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace hydra {

struct OpenVocabFeatureEntry {
  FeatureVector feature;
  std::string encoder_id;
  uint64_t timestamp_ns = 0;
  uint32_t frame_index = 0;
  uint32_t source_instance_id = 0;
};

class OpenVocabFeatureCache {
 public:
  using Ptr = std::shared_ptr<OpenVocabFeatureCache>;

  void setMaxEntries(size_t max_entries) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_entries_ = max_entries;
    pruneLocked();
  }

  void store(uint32_t source_track_id, OpenVocabFeatureEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_[source_track_id] = std::move(entry);
    pruneLocked();
  }

  std::optional<OpenVocabFeatureEntry> consume(uint32_t source_track_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = entries_.find(source_track_id);
    if (iter == entries_.end()) {
      return std::nullopt;
    }

    auto result = std::move(iter->second);
    entries_.erase(iter);
    return result;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
  }

 private:
  void pruneLocked() {
    if (max_entries_ == 0 || entries_.size() <= max_entries_) {
      return;
    }

    while (entries_.size() > max_entries_) {
      auto oldest_iter = entries_.begin();
      for (auto iter = entries_.begin(); iter != entries_.end(); ++iter) {
        if (iter->second.timestamp_ns < oldest_iter->second.timestamp_ns) {
          oldest_iter = iter;
        }
      }
      entries_.erase(oldest_iter);
    }
  }

  mutable std::mutex mutex_;
  size_t max_entries_ = 0;
  std::unordered_map<uint32_t, OpenVocabFeatureEntry> entries_;
};

}  // namespace hydra
