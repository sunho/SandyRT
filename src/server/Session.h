#pragma once

#include "Device.h"
#include "Engine.h"
#include "EngineTypes.h"
#include "Logger.h"
#include "Result.h"
#include "Sampler.h"
#include "Tensor.h"

#include <cstdint>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace sandy::server {

inline constexpr int32_t kDefaultPrefillChunkTokens = 2048;

struct GenerateResult {
    std::vector<int64_t> outputIds;
    std::string finishReason;
    int32_t promptTokens = 0;
    int32_t completionTokens = 0;
    double prefillMilliseconds = 0.0;
    double decodeMilliseconds = 0.0;
    double prefillTokensPerSecond = 0.0;
    double decodeTokensPerSecond = 0.0;
};

struct CacheGroupConfig {
    int32_t count = 0;
    core::Shape tensorShape;
    int64_t growDim = 2;
    int64_t pageSize = 32;
};

struct SessionConfig {
    std::vector<CacheGroupConfig> cacheGroups;
    int32_t prefillChunkTokens = kDefaultPrefillChunkTokens;
};

class Session {
public:
    Session(
        device::Device& device,
        engine::Engine& engine,
        const engine::CompiledKernelGraph& compiled,
        const engine::CompiledKernelGraph* prefillCompiled,
        const engine::DeviceWeightMap& weights,
        SessionConfig config,
        RequestLogger* logger = nullptr);
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
    Result<std::vector<engine::RunInput>> makeTokenInputs(int64_t token, int64_t position);
    Result<std::vector<engine::RunInput>> makePrefillInputs(
        const std::vector<int64_t>& inputIds,
        size_t begin,
        size_t end,
        int64_t position);
    Result<std::pair<int64_t, float>> sampleOutputs(
        std::vector<engine::RunOutput>& outputs);
    Result<std::vector<engine::RunOutput>> runValuesProfiled(
        const engine::CompiledKernelGraph& graph,
        std::span<const engine::RunInput> inputs,
        const std::string& phase);
    Result<std::pair<int64_t, float>> evalToken(
        int64_t token,
        int64_t position,
        const std::string& phase);
    Result<std::pair<int64_t, float>> prefillChunk(
        const std::vector<int64_t>& inputIds,
        size_t begin,
        size_t end,
        int64_t position,
        const std::string& phase);
    bool shouldStop(int64_t token, const std::unordered_set<int64_t>& stopTokens) const;
    void destroyCaches();

    device::Device& device_;
    engine::Engine& engine_;
    const engine::CompiledKernelGraph& compiled_;
    const engine::CompiledKernelGraph* prefillCompiled_ = nullptr;
    const engine::DeviceWeightMap& weights_;
    SessionConfig config_;
    RequestLogger* logger_ = nullptr;
    Sampler sampler_;
    std::vector<CacheGroup> caches_;
    bool initialized_ = false;
};

} // namespace sandy::server
