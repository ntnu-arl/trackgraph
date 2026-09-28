#include <gtest/gtest.h>

#include "hydra/common/message_queue.h"

namespace hydra {
namespace {

TEST(MessageQueue, StatisticsAreOptIn) {
  MessageQueue<int> queue;
  ASSERT_TRUE(queue.push(1));

  const auto stats = queue.statistics();
  EXPECT_FALSE(stats.enabled);
  EXPECT_EQ(stats.current_depth, 1u);
  EXPECT_EQ(stats.peak_depth, 0u);
  EXPECT_EQ(stats.total_pushed, 0u);
  EXPECT_EQ(stats.total_popped, 0u);
}

TEST(MessageQueue, TracksDepthAndOperationsWhenEnabled) {
  MessageQueue<int> queue(2, true);
  EXPECT_TRUE(queue.push(1));
  EXPECT_TRUE(queue.push(2));
  EXPECT_FALSE(queue.push(3, false));

  auto stats = queue.statistics();
  EXPECT_TRUE(stats.enabled);
  EXPECT_EQ(stats.current_depth, 2u);
  EXPECT_EQ(stats.peak_depth, 2u);
  EXPECT_EQ(stats.total_pushed, 2u);
  EXPECT_EQ(stats.total_popped, 0u);
  EXPECT_EQ(stats.total_rejected, 1u);
  EXPECT_EQ(stats.total_cleared, 0u);

  EXPECT_EQ(queue.pop(), 1);
  EXPECT_TRUE(queue.push(4));
  queue.clear();

  stats = queue.statistics();
  EXPECT_EQ(stats.current_depth, 0u);
  EXPECT_EQ(stats.peak_depth, 2u);
  EXPECT_EQ(stats.total_pushed, 3u);
  EXPECT_EQ(stats.total_popped, 1u);
  EXPECT_EQ(stats.total_rejected, 1u);
  EXPECT_EQ(stats.total_cleared, 2u);
}

}  // namespace
}  // namespace hydra
