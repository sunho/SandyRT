#pragma once

#include "CpuDevice.h"
#include "Engine.h"
#include "EngineTypes.h"
#include "GemmaSession.h"
#include "Result.h"
#include "SafeTensorWeights.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sandy::server {

struct GemmaModelConfig {
    std::string modelId = "gemma4e2b";
    std::string modelPath;
    std::string weightsPath;
    int32_t eosTokenId = 1;
    int32_t maxContextTokens = 0;
};

class GemmaModel {
public:
    static Result<std::unique_ptr<GemmaModel>> load(GemmaModelConfig config);

    Result<GenerateResult> generate(
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds);

    const GemmaModelConfig& config() const { return config_; }

private:
    explicit GemmaModel(GemmaModelConfig config);

    Result<void> initialize();

    GemmaModelConfig config_;
    std::unique_ptr<weight::EagerSafeTensorWeights> weights_;
    engine::TensorMap weightMap_;
    std::unique_ptr<engine::CompiledKernelGraph> compiled_;
    std::unique_ptr<engine::Engine> engine_;
    device::CpuDevice* cpuDevice_ = nullptr;
    std::mutex generateMutex_;
};

} // namespace sandy::server
