#pragma once

#include "CpuDevice.h"
#include "Engine.h"
#include "EngineTypes.h"
#include "Result.h"
#include "Sampler.h"
#include "Tensor.h"

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

struct CacheGroupConfig {
    int32_t count = 0;
    core::Shape tensorShape;
    int64_t growDim = 2;
    int64_t pageSize = 16;
};

struct SessionConfig {
    std::vector<CacheGroupConfig> cacheGroups;
};

class Session {
public:
    Session(
        device::CpuDevice& device,
        engine::Engine& engine,
        const engine::CompiledKernelGraph& compiled,
        const engine::TensorMap& weights,
        SessionConfig config);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session();

    Result<void> initialize();
    Result<GenerateResult> generate(
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds);

private:
    struct CacheGroup {
        device::DevicePagedPoolId pool = 0;
        std::vector<device::DevicePagedTensorId> tensors;
    };

    Result<device::DevicePagedTensorView> pagedView(device::DevicePagedTensorId tensor);
    Result<engine::RunTensorTuple> makeCacheTuple(const CacheGroup& group);
    Result<std::vector<engine::RunInput>> makeInputs(int64_t token, int64_t position);
    Result<engine::TensorBufferPtr> requireLogits(std::vector<engine::RunOutput>& outputs);
    Result<std::pair<int64_t, float>> evalToken(int64_t token, int64_t position);
    bool shouldStop(int64_t token, const std::unordered_set<int64_t>& stopTokens) const;
    void destroyCaches();

    device::CpuDevice& device_;
    engine::Engine& engine_;
    const engine::CompiledKernelGraph& compiled_;
    const engine::TensorMap& weights_;
    SessionConfig config_;
    Sampler sampler_;
    std::vector<CacheGroup> caches_;
    bool initialized_ = false;
};

} // namespace sandy::server
