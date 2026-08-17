#include "Model.h"

#include "Compiler.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace sandy::server {

namespace {

void add_tensors_to_map(const weight::Weights& tensors, engine::TensorMap& map) {
    for (const auto& desc : tensors.descriptors()) {
        auto tensor = tensors.get_tensor(desc.name);
        if (tensor)
            map[desc.name] = tensor;
    }
}

CacheGroupConfig cache_group(int32_t count, std::vector<int64_t> shape) {
    CacheGroupConfig config;
    config.count = count;
    config.tensorShape = core::Shape(std::move(shape));
    config.growDim = 2;
    config.pageSize = 16;
    return config;
}

bool is_gemma_architecture(const std::string& architecture) {
    return architecture == "gemma4e2b" || architecture == "gemma4e4b" ||
           architecture == "gemma";
}

} // namespace

Result<ModelConfig> applyModelPreset(ModelConfig config) {
    if (is_gemma_architecture(config.architecture)) {
        if (config.modelId.empty())
            config.modelId = "gemma4e2b";
        if (config.eosTokenId < 0)
            config.eosTokenId = 1;
        if (config.session.cacheGroups.empty()) {
            config.session.cacheGroups.push_back(cache_group(12, {1, 1, -1, 256}));
            config.session.cacheGroups.push_back(cache_group(12, {1, 1, -1, 256}));
            config.session.cacheGroups.push_back(cache_group(3, {1, 1, -1, 512}));
            config.session.cacheGroups.push_back(cache_group(3, {1, 1, -1, 512}));
        }
        return config;
    }

    if (config.architecture == "tinyllama") {
        if (config.modelId.empty())
            config.modelId = "tinyllama";
        if (config.eosTokenId < 0)
            config.eosTokenId = 2;
        if (config.maxContextTokens <= 0)
            config.maxContextTokens = 2048;
        if (config.session.cacheGroups.empty()) {
            config.session.cacheGroups.push_back(cache_group(22, {1, 4, -1, 64}));
            config.session.cacheGroups.push_back(cache_group(22, {1, 4, -1, 64}));
        }
        return config;
    }

    return make_error("unknown model architecture: " + config.architecture);
}

Model::Model(ModelConfig config)
    : config_(std::move(config)) {}

Result<std::unique_ptr<Model>> Model::load(ModelConfig config) {
    auto preset = applyModelPreset(std::move(config));
    if (!preset)
        return make_error(preset.error());
    auto model = std::unique_ptr<Model>(new Model(preset.take()));
    auto init = model->initialize();
    if (!init)
        return make_error(init.error());
    return model;
}

Result<void> Model::initialize() {
    if (config_.modelPath.empty())
        return make_error("--model is required");
    if (config_.weightsPath.empty())
        return make_error("--weights is required");

    Compiler compiler;
    auto highGraph = compiler.load_sandygo(config_.modelPath);

    weights_ = weight::EagerSafeTensorWeights::load(config_.weightsPath);
    if (!weights_)
        return make_error("failed to load weights: " + config_.weightsPath);
    add_tensors_to_map(*weights_, weightMap_);

    ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["input_id"] =
        core::TensorDesc("input_id", core::Shape({1, 1}), core::DType::I64);
    options.input_tensor_descs["position_id"] =
        core::TensorDesc("position_id", core::Shape({1}), core::DType::I64);

    auto midResult = compiler.materialize_mid_ir(highGraph, *weights_, options);
    if (!midResult)
        return make_error(midResult.error());

    auto cpu = std::make_unique<device::CpuDevice>();
    cpuDevice_ = cpu.get();
    std::vector<std::unique_ptr<device::Device>> devices;
    devices.push_back(std::move(cpu));
    engine_ = std::make_unique<engine::Engine>(std::move(devices));

    auto compiled = engine_->compile(**midResult);
    if (!compiled)
        return make_error(compiled.error());
    compiled_ = compiled.take();
    return {};
}

Result<GenerateResult> Model::generate(
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds) {
    std::lock_guard<std::mutex> lock(generateMutex_);
    if (!engine_ || !compiled_ || !cpuDevice_)
        return make_error("model is not initialized");

    std::vector<int64_t> effectiveStopTokens = stopTokenIds;
    if (config_.eosTokenId >= 0) {
        bool found = false;
        for (auto token : effectiveStopTokens) {
            if (token == config_.eosTokenId) {
                found = true;
                break;
            }
        }
        if (!found)
            effectiveStopTokens.push_back(config_.eosTokenId);
    }

    Session session(*cpuDevice_, *engine_, *compiled_, weightMap_, config_.session);
    return session.generate(inputIds, maxTokens, effectiveStopTokens);
}

} // namespace sandy::server
