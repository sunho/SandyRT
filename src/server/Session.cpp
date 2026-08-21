#include "Session.h"

#include "HostTensorBuffer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace sandy::server {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double tokens_per_second(int64_t tokens, double milliseconds) {
    return milliseconds > 0.0
        ? static_cast<double>(tokens) * 1000.0 / milliseconds
        : 0.0;
}

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

std::shared_ptr<HostTensorBuffer> make_i64_vector_buffer(
        std::string name,
        core::Shape shape,
        std::span<const int64_t> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    if (!values.empty())
        std::memcpy(bytes.data(), values.data(), bytes.size());
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
        const engine::CompiledKernelGraph* prefillCompiled,
        const engine::DeviceWeightMap& weights,
        SessionConfig config,
        RequestLogger* logger)
    : device_(device),
      engine_(engine),
      compiled_(compiled),
      prefillCompiled_(prefillCompiled),
      weights_(weights),
      config_(std::move(config)),
      logger_(logger) {}

Session::~Session() {
    destroyCaches();
}

Result<void> Session::initialize() {
    ServerStageScope timer(logger_, "session", "server.session.initialize");
    if (initialized_)
        return {};

    if (logger_) {
        logger_->logf(
            "server.session.initialize.start cache_groups=%zu",
            config_.cacheGroups.size());
    }
    caches_.clear();
    caches_.reserve(config_.cacheGroups.size());
    for (size_t groupIndex = 0; groupIndex < config_.cacheGroups.size(); groupIndex++) {
        const auto& groupConfig = config_.cacheGroups[groupIndex];
        ServerStageScope groupTimer(
            logger_,
            "session",
            "server.session.cache_group_" + std::to_string(groupIndex));
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
        if (logger_) {
            logger_->logf(
                "server.session.cache_group index=%zu count=%d grow_dim=%lld "
                "page_size=%lld",
                groupIndex,
                groupConfig.count,
                static_cast<long long>(groupConfig.growDim),
                static_cast<long long>(groupConfig.pageSize));
        }
    }

    initialized_ = true;
    if (logger_)
        logger_->log("server.session.initialize.done");
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

Result<std::vector<engine::RunInput>> Session::makeTokenInputs(
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

Result<std::vector<engine::RunInput>> Session::makePrefillInputs(
        const std::vector<int64_t>& inputIds,
        size_t begin,
        size_t end,
        int64_t position) {
    if (begin >= end || end > inputIds.size())
        return make_error("invalid prefill chunk range");
    auto tokenCount = static_cast<int64_t>(end - begin);

    std::vector<engine::RunInput> inputs;
    inputs.reserve(2 + caches_.size());
    inputs.push_back(make_i64_vector_buffer(
        "input_id",
        core::Shape({1, tokenCount}),
        std::span<const int64_t>(inputIds.data() + begin, end - begin)));
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

Result<std::vector<engine::RunOutput>> Session::runValuesProfiled(
        const engine::CompiledKernelGraph& graph,
        std::span<const engine::RunInput> inputs,
        const std::string& phase) {
    ServerStageScope timer(logger_, phase, "server.engine.run_values");
    engine::EngineRunOptions options;
    const engine::EngineRunOptions* optionsPtr = nullptr;
    if (logger_ && logger_->profileEnabled()) {
        options.profileKernel = [this, phase](const engine::EngineProfileEvent& event) {
            logger_->logProfileKernel(phase, event);
        };
        options.profileStage = [this, phase](const engine::EngineProfileStageEvent& event) {
            logger_->logProfileStage(phase, event);
        };
        options.profileDeviceRunBoundary =
            [this, phase](const engine::EngineDeviceRunBoundaryEvent& event) {
                logger_->logDeviceBoundary(phase, event);
            };
        optionsPtr = &options;
    }

    if (logger_) {
        logger_->logf(
            "server.engine.run_values.start phase=%s inputs=%zu profile=%d",
            phase.c_str(),
            inputs.size(),
            logger_->profileEnabled() ? 1 : 0);
    }
    auto outputs = engine_.runValues(graph, inputs, weights_, optionsPtr);
    if (!outputs) {
        if (logger_) {
            logger_->logf(
                "server.engine.run_values.error phase=%s message=%s",
                phase.c_str(),
                outputs.error().c_str());
        }
        return make_error(outputs.error());
    }
    if (logger_) {
        logger_->logf(
            "server.engine.run_values.done phase=%s outputs=%zu",
            phase.c_str(),
            outputs->size());
    }
    return outputs.take();
}

Result<std::pair<int64_t, float>> Session::evalToken(
        int64_t token,
        int64_t position,
        const std::string& phase) {
    ServerStageScope timer(logger_, phase, "server.eval_token.total");
    if (logger_) {
        logger_->logf(
            "server.eval_token.start phase=%s token=%lld position=%lld",
            phase.c_str(),
            static_cast<long long>(token),
            static_cast<long long>(position));
    }
    auto inputs = [&]() {
        ServerStageScope inputTimer(logger_, phase, "server.eval_token.make_inputs");
        return makeTokenInputs(token, position);
    }();
    if (!inputs)
        return make_error(inputs.error());
    auto outputs = runValuesProfiled(compiled_, *inputs, phase);
    if (!outputs)
        return make_error(outputs.error());
    auto logits = [&]() {
        ServerStageScope logitsTimer(logger_, phase, "server.eval_token.require_logits");
        return requireLogits(*outputs);
    }();
    if (!logits)
        return make_error(logits.error());
    auto sampled = [&]() {
        ServerStageScope sampleTimer(logger_, phase, "server.sampler.argmax");
        return sampler_.argmaxLast(*logits);
    }();
    if (!sampled)
        return make_error(sampled.error());
    if (logger_) {
        logger_->logf(
            "server.eval_token.done phase=%s next_token=%lld score=%.6f",
            phase.c_str(),
            static_cast<long long>(sampled->first),
            sampled->second);
    }
    return sampled.take();
}

Result<std::pair<int64_t, float>> Session::prefillChunk(
        const std::vector<int64_t>& inputIds,
        size_t begin,
        size_t end,
        int64_t position,
        const std::string& phase) {
    ServerStageScope timer(logger_, phase, "server.prefill_chunk.total");
    if (!prefillCompiled_)
        return make_error("prefill graph is not loaded");
    if (logger_) {
        logger_->logf(
            "server.prefill_chunk.start phase=%s begin=%zu end=%zu tokens=%zu "
            "position=%lld",
            phase.c_str(),
            begin,
            end,
            end - begin,
            static_cast<long long>(position));
    }
    auto inputs = [&]() {
        ServerStageScope inputTimer(logger_, phase, "server.prefill_chunk.make_inputs");
        return makePrefillInputs(inputIds, begin, end, position);
    }();
    if (!inputs)
        return make_error(inputs.error());
    auto outputs = runValuesProfiled(*prefillCompiled_, *inputs, phase);
    if (!outputs)
        return make_error(outputs.error());
    auto logits = [&]() {
        ServerStageScope logitsTimer(logger_, phase, "server.prefill_chunk.require_logits");
        return requireLogits(*outputs);
    }();
    if (!logits)
        return make_error(logits.error());
    auto sampled = [&]() {
        ServerStageScope sampleTimer(logger_, phase, "server.sampler.argmax");
        return sampler_.argmaxLast(*logits);
    }();
    if (!sampled)
        return make_error(sampled.error());
    if (logger_) {
        logger_->logf(
            "server.prefill_chunk.done phase=%s next_token=%lld score=%.6f",
            phase.c_str(),
            static_cast<long long>(sampled->first),
            sampled->second);
    }
    return sampled.take();
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
    ServerStageScope totalTimer(logger_, "request", "server.generate.total");
    if (logger_) {
        logger_->logf(
            "server.generate.start prompt_tokens=%zu max_tokens=%d stop_tokens=%zu "
            "prefill=%d prefill_chunk_tokens=%d",
            inputIds.size(),
            maxTokens,
            stopTokenIds.size(),
            prefillCompiled_ ? 1 : 0,
            config_.prefillChunkTokens);
    }
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
        if (logger_) {
            logger_->logf(
                "server.generate.finish reason=%s prompt_tokens=%d completion_tokens=%d",
                result.finishReason.c_str(),
                result.promptTokens,
                result.completionTokens);
        }
        return result;
    }

    std::unordered_set<int64_t> stopTokens(stopTokenIds.begin(), stopTokenIds.end());

    int64_t nextToken = 0;
    auto prefillStart = Clock::now();
    if (prefillCompiled_ && config_.prefillChunkTokens > 0) {
        ServerStageScope promptTimer(logger_, "prompt", "server.prompt.prefill");
        size_t begin = 0;
        auto chunkTokens = static_cast<size_t>(config_.prefillChunkTokens);
        while (begin < inputIds.size()) {
            auto end = std::min(inputIds.size(), begin + chunkTokens);
            auto phase = "prefill_" + std::to_string(begin) + "_" + std::to_string(end);
            auto next = prefillChunk(
                inputIds,
                begin,
                end,
                static_cast<int64_t>(begin),
                phase);
            if (!next)
                return make_error(next.error());
            if (end == inputIds.size())
                nextToken = next->first;
            begin = end;
        }
    } else {
        ServerStageScope promptTimer(logger_, "prompt", "server.prompt.eval_tokens");
        for (size_t i = 0; i < inputIds.size(); i++) {
            auto phase = "prompt_token_" + std::to_string(i);
            auto next = evalToken(inputIds[i], static_cast<int64_t>(i), phase);
            if (!next)
                return make_error(next.error());
            if (i + 1 == inputIds.size())
                nextToken = next->first;
        }
    }
    result.prefillMilliseconds = elapsed_milliseconds(prefillStart, Clock::now());
    result.prefillTokensPerSecond = tokens_per_second(
        result.promptTokens,
        result.prefillMilliseconds);

    ServerStageScope decodeTimer(logger_, "decode", "server.decode.total");
    auto decodeStart = Clock::now();
    for (int32_t step = 0; step < maxTokens; step++) {
        result.outputIds.push_back(nextToken);
        result.completionTokens = static_cast<int32_t>(result.outputIds.size());
        if (logger_) {
            logger_->logf(
                "server.decode.output step=%d token=%lld completion_tokens=%d",
                step,
                static_cast<long long>(nextToken),
                result.completionTokens);
        }

        if (shouldStop(nextToken, stopTokens)) {
            result.finishReason = "stop";
            break;
        }
        if (step + 1 >= maxTokens) {
            result.finishReason = "length";
            break;
        }

        auto phase = "decode_step_" + std::to_string(step + 1);
        auto next = evalToken(
            nextToken,
            static_cast<int64_t>(inputIds.size() + result.outputIds.size() - 1),
            phase);
        if (!next)
            return make_error(next.error());
        nextToken = next->first;
    }

    result.decodeMilliseconds = elapsed_milliseconds(decodeStart, Clock::now());
    result.decodeTokensPerSecond = tokens_per_second(
        result.completionTokens,
        result.decodeMilliseconds);
    if (result.finishReason.empty())
        result.finishReason = "length";
    if (logger_) {
        logger_->logf(
            "server.throughput prefill_tokens=%d prefill_ms=%.3f "
            "prefill_toks_per_s=%.3f decode_tokens=%d decode_ms=%.3f "
            "decode_toks_per_s=%.3f",
            result.promptTokens,
            result.prefillMilliseconds,
            result.prefillTokensPerSecond,
            result.completionTokens,
            result.decodeMilliseconds,
            result.decodeTokensPerSecond);
        logger_->logf(
            "server.generate.finish reason=%s prompt_tokens=%d completion_tokens=%d",
            result.finishReason.c_str(),
            result.promptTokens,
            result.completionTokens);
    }
    std::fprintf(
        stderr,
        "sandy throughput: prefill_tokens=%d prefill_ms=%.3f prefill_toks_per_s=%.3f "
        "decode_tokens=%d decode_ms=%.3f decode_toks_per_s=%.3f\n",
        result.promptTokens,
        result.prefillMilliseconds,
        result.prefillTokensPerSecond,
        result.completionTokens,
        result.decodeMilliseconds,
        result.decodeTokensPerSecond);
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
