#include "Model.h"

#include "Compiler.h"
#include "CpuDevice.h"
#ifdef SANDY_SERVER_ENABLE_CUDA
#include "CudaDevice.h"
#endif

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace sandy::server {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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
    config.pageSize = 32;
    return config;
}

Result<std::unique_ptr<engine::CompiledKernelGraph>> compile_model_graph(
        Compiler& compiler,
        engine::Engine& engine,
        const std::string& path,
        const weight::Weights& weights,
        const SandyGoCompileOptions& sandyGoOptions,
        const ir::mid_ir::MaterializeOptions& options,
        const engine::EngineCompileOptions& compileOptions) {
    auto highGraph = compiler.load_sandygo(path, sandyGoOptions);
    auto midResult = compiler.materialize_mid_ir(highGraph, weights, options);
    if (!midResult)
        return make_error(midResult.error());
    auto compiled = engine.compile(**midResult, &compileOptions);
    if (!compiled)
        return make_error(compiled.error());
    return compiled.take();
}

Result<void> load_model_config(ModelConfig& config) {
    auto path = std::filesystem::path(config.modelPath).parent_path() / "model_config.json";
    std::error_code filesystemError;
    bool exists = std::filesystem::exists(path, filesystemError);
    if (filesystemError) {
        return make_error(
            "failed to inspect model config " + path.string() + ": " +
            filesystemError.message());
    }
    if (!exists)
        return {};

    try {
        std::ifstream input(path);
        if (!input)
            return make_error("failed to open model config: " + path.string());
        auto json = nlohmann::json::parse(input);
        if (json.contains("top_k"))
            config.sampling.topK = json.at("top_k").get<int32_t>();
        if (json.contains("top_p"))
            config.sampling.topP = json.at("top_p").get<float>();
        if (json.contains("temperature"))
            config.sampling.temperature = json.at("temperature").get<float>();
    } catch (const std::exception& error) {
        return make_error(
            "failed to parse model config " + path.string() + ": " + error.what());
    }

    auto validated = resolveSamplingConfig(config.sampling);
    if (!validated)
        return make_error("invalid model config " + path.string() + ": " + validated.error());
    config.sampling = validated.take();
    return {};
}

} // namespace

Result<ModelConfig> applyModelPreset(ModelConfig config) {
    if (config.architecture == "gemma4e2b" || config.architecture == "gemma") {
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

    if (config.architecture == "gemma4e4b") {
        if (config.modelId.empty())
            config.modelId = "gemma4e4b";
        if (config.eosTokenId < 0)
            config.eosTokenId = 1;
        if (config.session.cacheGroups.empty()) {
            config.session.cacheGroups.push_back(cache_group(20, {1, 2, -1, 256}));
            config.session.cacheGroups.push_back(cache_group(20, {1, 2, -1, 256}));
            config.session.cacheGroups.push_back(cache_group(4, {1, 2, -1, 512}));
            config.session.cacheGroups.push_back(cache_group(4, {1, 2, -1, 512}));
        }
        return config;
    }

    if (config.architecture == "gemma4a4b26b" ||
        config.architecture == "gemma4a4b" ||
        config.architecture == "gemma4moe") {
        if (config.modelId.empty())
            config.modelId = "gemma4a4b26b";
        if (config.eosTokenId < 0)
            config.eosTokenId = 1;
        if (config.session.cacheGroups.empty()) {
            config.session.cacheGroups.push_back(cache_group(25, {1, 8, -1, 256}));
            config.session.cacheGroups.push_back(cache_group(25, {1, 8, -1, 256}));
            config.session.cacheGroups.push_back(cache_group(5, {1, 2, -1, 512}));
            config.session.cacheGroups.push_back(cache_group(5, {1, 2, -1, 512}));
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

    auto loadedConfig = load_model_config(config_);
    if (!loadedConfig)
        return make_error(loadedConfig.error());

    weights_ = weight::EagerSafeTensorWeights::load(config_.weightsPath);
    if (!weights_)
        return make_error("failed to load weights: " + config_.weightsPath);
    add_tensors_to_map(*weights_, weightMap_);

    std::unique_ptr<device::Device> device;
#ifdef SANDY_SERVER_ENABLE_CUDA
    device = std::make_unique<device::CudaDevice>();
    backend_ = "cuda";
#else
    device = std::make_unique<device::CpuDevice>();
    backend_ = "cpu";
#endif
    device_ = device.get();
    std::vector<std::unique_ptr<device::Device>> devices;
    devices.push_back(std::move(device));
    engine_ = std::make_unique<engine::Engine>(std::move(devices));

    engine::EngineCompileOptions compileOptions;
#ifdef SANDY_SERVER_ENABLE_CUDA
    compileOptions.fusor.attention = true;
#endif

    Compiler compiler;
    SandyGoCompileOptions sandyGoOptions;
    if (config_.architecture == "gemma4a4b26b" ||
        config_.architecture == "gemma4a4b" ||
        config_.architecture == "gemma4moe") {
        sandyGoOptions.configConstants["TOP_K"] = config_.sampling.topK;
    }
    ir::mid_ir::MaterializeOptions evalOptions;
    evalOptions.input_tensor_descs["input_id"] =
        core::TensorDesc("input_id", core::Shape({1, 1}), core::DType::I64);
    evalOptions.input_tensor_descs["position_id"] =
        core::TensorDesc("position_id", core::Shape({1}), core::DType::I64);

    auto compiled = compile_model_graph(
        compiler,
        *engine_,
        config_.modelPath,
        *weights_,
        sandyGoOptions,
        evalOptions,
        compileOptions);
    if (!compiled)
        return make_error(compiled.error());
    compiled_ = compiled.take();

    if (!config_.prefillModelPath.empty()) {
        ir::mid_ir::MaterializeOptions prefillOptions;
        prefillOptions.input_tensor_descs["input_id"] =
            core::TensorDesc(
                "input_id",
                core::Shape({1, core::Shape::kDynamic}),
                core::DType::I64);
        prefillOptions.input_tensor_descs["position_id"] =
            core::TensorDesc("position_id", core::Shape({1}), core::DType::I64);

        auto prefillCompiled = compile_model_graph(
            compiler,
            *engine_,
            config_.prefillModelPath,
            *weights_,
            sandyGoOptions,
            prefillOptions,
            compileOptions);
        if (!prefillCompiled)
            return make_error(prefillCompiled.error());
        prefillCompiled_ = prefillCompiled.take();
    }

    auto deviceWeights = engine_->loadWeights(*compiled_, weightMap_);
    if (!deviceWeights)
        return make_error(deviceWeights.error());
    deviceWeights_ = deviceWeights.take();
    return {};
}

Result<GenerateResult> Model::generate(
        const std::string& requestId,
        const std::vector<int64_t>& inputIds,
        int32_t maxTokens,
        const std::vector<int64_t>& stopTokenIds,
        RequestLogger* logger,
        const SamplingOverrides& samplingOverrides) {
    std::unique_ptr<RequestLogger> ownedLogger;
    if (!logger) {
        ownedLogger = RequestLogger::create(config_.logging, requestId);
        logger = ownedLogger.get();
    }
    if (logger) {
        logger->logf(
            "server.model.generate.start request_id=%s model_id=%s architecture=%s "
            "backend=%s prompt_tokens=%zu max_tokens=%d stop_tokens=%zu",
            requestId.c_str(),
            config_.modelId.c_str(),
            config_.architecture.c_str(),
            backend_.c_str(),
            inputIds.size(),
            maxTokens,
            stopTokenIds.size());
    }

    auto lockStart = Clock::now();
    std::unique_lock<std::mutex> lock(generateMutex_);
    if (logger)
        logger->logServerStage("request", "server.model.generate.queue_wait", elapsed_ms(lockStart));
    ServerStageScope totalTimer(logger, "request", "server.model.generate.total");

    if (!engine_ || !compiled_ || !device_ || !deviceWeights_)
        return make_error("model is not initialized");

    auto sampling = resolveSamplingConfig(config_.sampling, samplingOverrides);
    if (!sampling)
        return make_error(sampling.error());

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
    if (logger) {
        logger->logf(
            "server.model.generate.effective_stop_tokens count=%zu eos_token_id=%d",
            effectiveStopTokens.size(),
            config_.eosTokenId);
    }

    auto sessionConfig = config_.session;
    sessionConfig.sampling = sampling.take();
    Session session(
        *device_,
        *engine_,
        *compiled_,
        prefillCompiled_.get(),
        *deviceWeights_,
        std::move(sessionConfig),
        logger);
    auto generated = session.generate(inputIds, maxTokens, effectiveStopTokens);
    if (!generated) {
        if (logger) {
            logger->logf(
                "server.model.generate.error request_id=%s message=%s",
                requestId.c_str(),
                generated.error().c_str());
        }
        return make_error(generated.error());
    }
    auto result = generated.take();
    if (logger) {
        logger->logf(
            "server.model.generate.done request_id=%s finish_reason=%s "
            "prompt_tokens=%d completion_tokens=%d",
            requestId.c_str(),
            result.finishReason.c_str(),
            result.promptTokens,
            result.completionTokens);
    }
    return result;
}

} // namespace sandy::server
