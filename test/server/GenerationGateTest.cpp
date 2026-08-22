#include "GenerationGate.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace {

using namespace std::chrono_literals;

TEST(ServerGenerationGate, DeadlineIncludesTimeWaitingForActiveRequest) {
    sandy::server::GenerationGate gate;
    auto active = gate.acquire(nullptr);
    ASSERT_TRUE(active) << active.error();

    sandy::server::RequestControl control;
    control.submittedAt = sandy::server::RequestControl::Clock::now();
    control.deadline = control.submittedAt + 60ms;
    auto pending = std::async(std::launch::async, [&]() {
        return gate.acquire(&control);
    });

    auto result = pending.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "request deadline exceeded");
}

TEST(ServerGenerationGate, CancelledPendingRequestNeverAcquiresGate) {
    sandy::server::GenerationGate gate;
    auto active = gate.acquire(nullptr);
    ASSERT_TRUE(active) << active.error();

    std::atomic_bool cancelled = false;
    sandy::server::RequestControl control;
    control.clientCancelled = [&]() { return cancelled.load(); };
    auto pending = std::async(std::launch::async, [&]() {
        return gate.acquire(&control);
    });

    std::this_thread::sleep_for(30ms);
    cancelled.store(true);
    auto result = pending.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "request cancelled");
}

TEST(ServerGenerationGate, ReleasingLeaseAllowsNextRequest) {
    sandy::server::GenerationGate gate;
    auto active = gate.acquire(nullptr);
    ASSERT_TRUE(active) << active.error();

    auto pending = std::async(std::launch::async, [&]() {
        return gate.acquire(nullptr);
    });
    EXPECT_EQ(pending.wait_for(30ms), std::future_status::timeout);

    active.take().reset();
    EXPECT_EQ(pending.wait_for(200ms), std::future_status::ready);
    auto result = pending.get();
    ASSERT_TRUE(result) << result.error();
}

} // namespace
