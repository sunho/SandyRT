#include "GemmaModel.h"

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

} // namespace

GemmaModel::GemmaModel(GemmaModelConfig config)
    : config_(std::move(config)) {}

Result<std::unique_ptr<GemmaModel>> GemmaModel::load(GemmaModelConfig config) {
    auto model = std::unique_ptr<GemmaModel>(new GemmaModel(std::move(config)));
    auto init = model->initialize();
    if (!init)
        return make_error(init.error());
    return model;
}

Result<void> GemmaModel::initialize() {
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

Result<GenerateResult> GemmaModel::generate(
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

    GemmaSession session(*cpuDevice_, *engine_, *compiled_, weightMap_);
    return session.generate(inputIds, maxTokens, effectiveStopTokens);
}

} // namespace sandy::server
