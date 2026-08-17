#pragma once

#include "CpuDevice.h"
#include "Engine.h"
#include "EngineTypes.h"
#include "Result.h"
#include "Sampler.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace sandy::server {

struct GenerateResult {
    std::vector<int64_t> outputIds;
    std::string finishReason;
    int32_t promptTokens = 0;
    int32_t completionTokens = 0;
};

class GemmaSession {
public:
    GemmaSession(
        device::CpuDevice& device,
        engine::Engine& engine,
        const engine::CompiledKernelGraph& compiled,
        const engine::TensorMap& weights);
    GemmaSession(const GemmaSession&) = delete;
    GemmaSession& operator=(const GemmaSession&) = delete;
    ~GemmaSession();

    Result<void> initialize();
    Result<GenerateResult> generate(
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds);

private:
    static constexpr int kLocalCacheCount = 12;
    static constexpr int kGlobalCacheCount = 3;

    struct Caches {
        device::DevicePagedPoolId localPool = 0;
        device::DevicePagedPoolId globalPool = 0;
        std::array<device::DevicePagedTensorId, kLocalCacheCount> localK{};
        std::array<device::DevicePagedTensorId, kLocalCacheCount> localV{};
        std::array<device::DevicePagedTensorId, kGlobalCacheCount> globalK{};
        std::array<device::DevicePagedTensorId, kGlobalCacheCount> globalV{};
    };

    Result<device::DevicePagedTensorView> pagedView(device::DevicePagedTensorId tensor);
    Result<engine::RunTensorTuple> makeLocalTuple(
        const std::array<device::DevicePagedTensorId, kLocalCacheCount>& ids);
    Result<engine::RunTensorTuple> makeGlobalTuple(
        const std::array<device::DevicePagedTensorId, kGlobalCacheCount>& ids);
    Result<std::vector<engine::RunInput>> makeInputs(int64_t token, int64_t position);
    Result<engine::TensorBufferPtr> requireLogits(std::vector<engine::RunOutput>& outputs);
    Result<std::pair<int64_t, float>> evalToken(int64_t token, int64_t position);
    bool shouldStop(int64_t token, const std::unordered_set<int64_t>& stopTokens) const;
    void destroyCaches();

    device::CpuDevice& device_;
    engine::Engine& engine_;
    const engine::CompiledKernelGraph& compiled_;
    const engine::TensorMap& weights_;
    Sampler sampler_;
    Caches caches_;
    bool initialized_ = false;
};

} // namespace sandy::server
