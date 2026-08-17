#include "Session.h"

#include "HostTensorBuffer.h"

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

core::Shape zero_grow_shape(core::Shape shape, int64_t growDim) {
    auto dims = shape.dims();
    dims[static_cast<size_t>(growDim)] = 0;
    return core::Shape(std::move(dims));
}

} // namespace

Session::Session(
        device::Device& device,
        engine::Engine& engine,
        const engine::CompiledKernelGraph& compiled,
        const engine::TensorMap& weights,
        SessionConfig config)
    : device_(device),
      engine_(engine),
      compiled_(compiled),
      weights_(weights),
      config_(std::move(config)) {}

Session::~Session() {
    destroyCaches();
}

Result<void> Session::initialize() {
    if (initialized_)
        return {};

    caches_.clear();
    caches_.reserve(config_.cacheGroups.size());
    for (const auto& groupConfig : config_.cacheGroups) {
        if (groupConfig.count <= 0)
            return make_error("cache group count must be positive");
        if (groupConfig.growDim < 0 ||
            groupConfig.growDim >= groupConfig.tensorShape.rank()) {
            return make_error("cache group growDim is out of range");
        }
        if (groupConfig.pageSize <= 0)
            return make_error("cache group pageSize must be positive");

        device::DevicePagedPoolDesc poolDesc;
        poolDesc.templateDesc = core::TensorDesc(
            groupConfig.tensorShape,
            core::DType::BF16);
        poolDesc.growDim = groupConfig.growDim;
        poolDesc.pageSize = groupConfig.pageSize;

        auto pool = device_.createPagedPool(poolDesc);
        if (!pool)
            return make_error(pool.error());

        CacheGroup group;
        group.pool = *pool;
        group.tensors.reserve(static_cast<size_t>(groupConfig.count));
        auto initialShape = zero_grow_shape(groupConfig.tensorShape, groupConfig.growDim);
        for (int32_t i = 0; i < groupConfig.count; i++) {
            auto tensor = device_.allocPaged(group.pool, initialShape);
            if (!tensor)
                return make_error(tensor.error());
            group.tensors.push_back(*tensor);
        }
        caches_.push_back(std::move(group));
    }

    initialized_ = true;
    return {};
}

Result<device::DevicePagedTensorView> Session::pagedView(
        device::DevicePagedTensorId tensor) {
    auto meta = device_.pagedMeta(tensor);
    if (!meta)
        return make_error(meta.error());
    return device::DevicePagedTensorView{tensor, meta.take()};
}

Result<engine::RunTensorTuple> Session::makeCacheTuple(const CacheGroup& group) {
    engine::RunTensorTuple tuple;
    tuple.elements.reserve(group.tensors.size());
    for (auto id : group.tensors) {
        auto view = pagedView(id);
        if (!view)
            return make_error(view.error());
        tuple.elements.push_back(view.take());
    }
    return tuple;
}

Result<std::vector<engine::RunInput>> Session::makeInputs(
        int64_t token,
        int64_t position) {
    std::vector<engine::RunInput> inputs;
    inputs.reserve(2 + caches_.size());
    inputs.push_back(make_i64_buffer("input_id", core::Shape({1, 1}), token));
    inputs.push_back(make_i64_buffer("position_id", core::Shape({1}), position));
    for (const auto& group : caches_) {
        auto tuple = makeCacheTuple(group);
        if (!tuple)
            return make_error(tuple.error());
        inputs.push_back(tuple.take());
    }
    return inputs;
}

Result<engine::TensorBufferPtr> Session::requireLogits(
        std::vector<engine::RunOutput>& outputs) {
    if (outputs.empty())
        return make_error("eval-token graph produced no outputs");
    auto* tensor = std::get_if<engine::TensorBufferPtr>(&outputs[0]);
    if (!tensor || !*tensor)
        return make_error("eval-token output 0 must be logits tensor");
    return *tensor;
}

Result<std::pair<int64_t, float>> Session::evalToken(
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

bool Session::shouldStop(
        int64_t token,
        const std::unordered_set<int64_t>& stopTokens) const {
    return stopTokens.find(token) != stopTokens.end();
}

Result<GenerateResult> Session::generate(
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

void Session::destroyCaches() {
    for (auto& group : caches_) {
        for (auto id : group.tensors) {
            if (id != 0)
                (void)device_.deallocPaged(id);
        }
        if (group.pool != 0)
            (void)device_.destroyPagedPool(group.pool);
    }
    caches_.clear();
    initialized_ = false;
}

} // namespace sandy::server
