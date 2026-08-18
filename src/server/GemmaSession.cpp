#include "GemmaSession.h"

#include "HostTensorBuffer.h"
#include "Tensor.h"

#include <cstring>
#include <memory>
#include <utility>

namespace sandy::server {

namespace {

std::shared_ptr<HostTensorBuffer> make_i64_buffer(
        std::string name,
        core::Shape shape,
        int64_t value) {
    std::vector<uint8_t> bytes(sizeof(int64_t));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return std::make_shared<HostTensorBuffer>(
        core::TensorDesc(std::move(name), std::move(shape), core::DType::I64),
        std::move(bytes));
}

} // namespace

GemmaSession::GemmaSession(
        device::CpuDevice& device,
        engine::Engine& engine,
        const engine::CompiledKernelGraph& compiled,
        const engine::TensorMap& weights)
    : device_(device),
      engine_(engine),
      compiled_(compiled),
      weights_(weights) {}

GemmaSession::~GemmaSession() {
    destroyCaches();
}

Result<void> GemmaSession::initialize() {
    if (initialized_)
        return {};

    device::DevicePagedPoolDesc localPoolDesc;
    localPoolDesc.templateDesc = core::TensorDesc(
        core::Shape({1, 1, core::Shape::kDynamic, 256}),
        core::DType::BF16);
    localPoolDesc.growDim = 2;
    localPoolDesc.pageSize = 32;

    device::DevicePagedPoolDesc globalPoolDesc;
    globalPoolDesc.templateDesc = core::TensorDesc(
        core::Shape({1, 1, core::Shape::kDynamic, 512}),
        core::DType::BF16);
    globalPoolDesc.growDim = 2;
    globalPoolDesc.pageSize = 32;

    auto localPool = device_.createPagedPool(localPoolDesc);
    if (!localPool)
        return make_error(localPool.error());
    caches_.localPool = *localPool;

    auto globalPool = device_.createPagedPool(globalPoolDesc);
    if (!globalPool)
        return make_error(globalPool.error());
    caches_.globalPool = *globalPool;

    for (auto& id : caches_.localK) {
        auto tensor = device_.allocPaged(caches_.localPool, core::Shape({1, 1, 0, 256}));
        if (!tensor)
            return make_error(tensor.error());
        id = *tensor;
    }
    for (auto& id : caches_.localV) {
        auto tensor = device_.allocPaged(caches_.localPool, core::Shape({1, 1, 0, 256}));
        if (!tensor)
            return make_error(tensor.error());
        id = *tensor;
    }
    for (auto& id : caches_.globalK) {
        auto tensor = device_.allocPaged(caches_.globalPool, core::Shape({1, 1, 0, 512}));
        if (!tensor)
            return make_error(tensor.error());
        id = *tensor;
    }
    for (auto& id : caches_.globalV) {
        auto tensor = device_.allocPaged(caches_.globalPool, core::Shape({1, 1, 0, 512}));
        if (!tensor)
            return make_error(tensor.error());
        id = *tensor;
    }

    initialized_ = true;
    return {};
}

Result<device::DevicePagedTensorView> GemmaSession::pagedView(
        device::DevicePagedTensorId tensor) {
    auto meta = device_.pagedMeta(tensor);
    if (!meta)
        return make_error(meta.error());
    return device::DevicePagedTensorView{tensor, meta.take()};
}

Result<engine::RunTensorTuple> GemmaSession::makeLocalTuple(
        const std::array<device::DevicePagedTensorId, kLocalCacheCount>& ids) {
    engine::RunTensorTuple tuple;
    tuple.elements.reserve(ids.size());
    for (auto id : ids) {
        auto view = pagedView(id);
        if (!view)
            return make_error(view.error());
        tuple.elements.push_back(view.take());
    }
    return tuple;
}

Result<engine::RunTensorTuple> GemmaSession::makeGlobalTuple(
        const std::array<device::DevicePagedTensorId, kGlobalCacheCount>& ids) {
    engine::RunTensorTuple tuple;
    tuple.elements.reserve(ids.size());
    for (auto id : ids) {
        auto view = pagedView(id);
        if (!view)
            return make_error(view.error());
        tuple.elements.push_back(view.take());
    }
    return tuple;
}

Result<std::vector<engine::RunInput>> GemmaSession::makeInputs(
        int64_t token,
        int64_t position) {
    auto localK = makeLocalTuple(caches_.localK);
    if (!localK)
        return make_error(localK.error());
    auto localV = makeLocalTuple(caches_.localV);
    if (!localV)
        return make_error(localV.error());
    auto globalK = makeGlobalTuple(caches_.globalK);
    if (!globalK)
        return make_error(globalK.error());
    auto globalV = makeGlobalTuple(caches_.globalV);
    if (!globalV)
        return make_error(globalV.error());

    std::vector<engine::RunInput> inputs;
    inputs.reserve(6);
    inputs.push_back(make_i64_buffer("input_id", core::Shape({1, 1}), token));
    inputs.push_back(make_i64_buffer("position_id", core::Shape({1}), position));
    inputs.push_back(localK.take());
    inputs.push_back(localV.take());
    inputs.push_back(globalK.take());
    inputs.push_back(globalV.take());
    return inputs;
}

Result<engine::TensorBufferPtr> GemmaSession::requireLogits(
        std::vector<engine::RunOutput>& outputs) {
    if (outputs.empty())
        return make_error("eval-token graph produced no outputs");
    auto* tensor = std::get_if<engine::TensorBufferPtr>(&outputs[0]);
    if (!tensor || !*tensor)
        return make_error("eval-token output 0 must be logits tensor");
    return *tensor;
}

Result<std::pair<int64_t, float>> GemmaSession::evalToken(
        int64_t token,
        int64_t position) {
    auto inputs = makeInputs(token, position);
    if (!inputs)
        return make_error(inputs.error());
    auto outputs = engine_.runValues(compiled_, *inputs, weights_);
    if (!outputs)
        return make_error(outputs.error());
    auto logits = requireLogits(*outputs);
    if (!logits)
        return make_error(logits.error());
    return sampler_.argmaxLast(*logits);
}

bool GemmaSession::shouldStop(
        int64_t token,
        const std::unordered_set<int64_t>& stopTokens) const {
    return stopTokens.find(token) != stopTokens.end();
}

Result<GenerateResult> GemmaSession::generate(
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds) {
    if (!initialized_) {
        auto init = initialize();
        if (!init)
            return make_error(init.error());
    }
    if (inputIds.empty())
        return make_error("GenerateRequest.input_ids must not be empty");
    if (maxTokens < 0)
        return make_error("GenerateRequest.max_tokens must be >= 0");

    GenerateResult result;
    result.promptTokens = static_cast<int32_t>(inputIds.size());
    if (maxTokens == 0) {
        result.finishReason = "length";
        return result;
    }

    std::unordered_set<int64_t> stopTokens(stopTokenIds.begin(), stopTokenIds.end());

    int64_t nextToken = 0;
    for (size_t i = 0; i < inputIds.size(); i++) {
        auto next = evalToken(inputIds[i], static_cast<int64_t>(i));
        if (!next)
            return make_error(next.error());
        if (i + 1 == inputIds.size())
            nextToken = next->first;
    }

    for (int32_t step = 0; step < maxTokens; step++) {
        result.outputIds.push_back(nextToken);
        result.completionTokens = static_cast<int32_t>(result.outputIds.size());

        if (shouldStop(nextToken, stopTokens)) {
            result.finishReason = "stop";
            return result;
        }
        if (step + 1 >= maxTokens) {
            result.finishReason = "length";
            return result;
        }

        auto next = evalToken(
            nextToken,
            static_cast<int64_t>(inputIds.size() + result.outputIds.size() - 1));
        if (!next)
            return make_error(next.error());
        nextToken = next->first;
    }

    result.finishReason = "length";
    return result;
}

void GemmaSession::destroyCaches() {
    for (auto id : caches_.localK) if (id != 0) (void)device_.deallocPaged(id);
    for (auto id : caches_.localV) if (id != 0) (void)device_.deallocPaged(id);
    for (auto id : caches_.globalK) if (id != 0) (void)device_.deallocPaged(id);
    for (auto id : caches_.globalV) if (id != 0) (void)device_.deallocPaged(id);
    if (caches_.localPool != 0) (void)device_.destroyPagedPool(caches_.localPool);
    if (caches_.globalPool != 0) (void)device_.destroyPagedPool(caches_.globalPool);
    caches_ = Caches{};
    initialized_ = false;
}

} // namespace sandy::server
