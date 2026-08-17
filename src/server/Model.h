#pragma once

#include "Device.h"
#include "Engine.h"
#include "EngineTypes.h"
#include "Result.h"
#include "SafeTensorWeights.h"
#include "Session.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sandy::server {

struct ModelConfig {
    std::string modelId;
    std::string architecture = "gemma4e2b";
    std::string modelPath;
    std::string weightsPath;
    int32_t eosTokenId = -1;
    int32_t maxContextTokens = 0;
    SessionConfig session;
};

Result<ModelConfig> applyModelPreset(ModelConfig config);

class Model {
public:
    static Result<std::unique_ptr<Model>> load(ModelConfig config);

    Result<GenerateResult> generate(
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds);

    const ModelConfig& config() const { return config_; }
    const std::string& backend() const { return backend_; }

private:
    explicit Model(ModelConfig config);

    Result<void> initialize();

    ModelConfig config_;
    std::unique_ptr<weight::EagerSafeTensorWeights> weights_;
    engine::TensorMap weightMap_;
    std::unique_ptr<engine::CompiledKernelGraph> compiled_;
    std::unique_ptr<engine::DeviceWeightMap> deviceWeights_;
    std::unique_ptr<engine::Engine> engine_;
    device::Device* device_ = nullptr;
    std::string backend_ = "cpu";
    std::mutex generateMutex_;
};

} // namespace sandy::server
