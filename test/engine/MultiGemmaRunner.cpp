#include "Compiler.h"
#include "CpuDevice.h"
#ifdef SANDY_RUNNER_ENABLE_CUDA
#include "CudaDevice.h"
#endif
#include "Engine.h"
#include "SafeTensorWeights.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr int kGemma4E2BLocalCacheCount = 12;
constexpr int kGemma4E2BGlobalCacheCount = 3;
constexpr int kGemma4E4BLocalCacheCount = 20;
constexpr int kGemma4E4BGlobalCacheCount = 4;
constexpr int kGemma4A4BLocalCacheCount = 25;
constexpr int kGemma4A4BGlobalCacheCount = 5;
constexpr int kTinyLlamaKVLayerCount = 22;

class HostTensorBuffer final : public sandy::core::TensorBuffer {
public:
    HostTensorBuffer(sandy::core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

bool parse_int64(const char* text, int64_t& out) {
    std::string_view view(text);
    auto result = std::from_chars(view.data(), view.data() + view.size(), out);
    return result.ec == std::errc{} && result.ptr == view.data() + view.size();
}

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

float read_bf16(std::span<const uint8_t> data, size_t index) {
    sandy::core::BFloat16 value = sandy::core::bfloat16_from_bits(0);
    std::memcpy(&value, data.data() + index * sizeof(sandy::core::BFloat16), sizeof(value));
    return sandy::core::bfloat16_to_float(value);
}

float read_float_value(std::span<const uint8_t> data, sandy::core::DType dtype, size_t index) {
    switch (dtype) {
        case sandy::core::DType::F32: return read_f32(data, index);
        case sandy::core::DType::BF16: return read_bf16(data, index);
        default: return -std::numeric_limits<float>::infinity();
    }
}

void add_tensors_to_map(const sandy::weight::Weights& tensors, sandy::engine::TensorMap& map) {
    for (const auto& desc : tensors.descriptors()) {
        auto tensor = tensors.get_tensor(desc.name);
        if (!tensor) {
            fprintf(stderr, "tensor listed in descriptors but missing: %s\n", desc.name.c_str());
            std::abort();
        }
        map[desc.name] = tensor;
    }
}

std::shared_ptr<HostTensorBuffer> make_input_buffer(
        const std::vector<int64_t>& tokens,
        int64_t& tokenIndex) {
    tokenIndex = static_cast<int64_t>(tokens.size()) - 1;
    std::vector<uint8_t> bytes(tokens.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), tokens.data(), bytes.size());
    return std::make_shared<HostTensorBuffer>(
        sandy::core::TensorDesc(
            "input_ids",
            sandy::core::Shape({1, static_cast<int64_t>(tokens.size())}),
            sandy::core::DType::I64),
        std::move(bytes));
}

std::shared_ptr<HostTensorBuffer> make_i64_buffer(
        std::string name,
        sandy::core::Shape shape,
        int64_t value) {
    std::vector<uint8_t> bytes(sizeof(int64_t));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return std::make_shared<HostTensorBuffer>(
        sandy::core::TensorDesc(
            std::move(name),
            std::move(shape),
        sandy::core::DType::I64),
        std::move(bytes));
}

std::shared_ptr<HostTensorBuffer> make_i64_vector_buffer(
        std::string name,
        sandy::core::Shape shape,
        std::span<const int64_t> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    if (!values.empty())
        std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_shared<HostTensorBuffer>(
        sandy::core::TensorDesc(
            std::move(name),
            std::move(shape),
            sandy::core::DType::I64),
        std::move(bytes));
}

Result<std::pair<int64_t, float>> argmax_at(
        sandy::core::TensorBuffer& output,
        int64_t tokenIndex) {
    auto access = output.access();
    if (!access)
        return make_error(access.error());

    const auto& desc = access->desc();
    if (desc.dtype != sandy::core::DType::F32 && desc.dtype != sandy::core::DType::BF16)
        return make_error("output logits must be F32 or BF16");
    if (desc.shape.rank() < 2)
        return make_error("output logits must have rank >= 2");

    int64_t vocab = desc.shape.dim(desc.shape.rank() - 1);
    int64_t numel = desc.shape.numel();
    if (vocab <= 0 || numel <= 0)
        return make_error("output logits must have static shape");
    int64_t rows = numel / vocab;
    if (tokenIndex < 0 || tokenIndex >= rows)
        return make_error("token index out of output bounds");

    size_t base = static_cast<size_t>(tokenIndex * vocab);
    int64_t bestId = 0;
    float bestValue = -std::numeric_limits<float>::infinity();
    for (int64_t id = 0; id < vocab; id++) {
        float value = read_float_value(access->data(), desc.dtype, base + static_cast<size_t>(id));
        if (value > bestValue) {
            bestValue = value;
            bestId = id;
        }
    }
    return std::pair<int64_t, float>{bestId, bestValue};
}

struct EvalTokenCaches {
    struct Group {
        sandy::device::DevicePagedPoolId pool = 0;
        std::vector<sandy::device::DevicePagedTensorId> tensors;
    };
    std::vector<Group> groups;
};

struct ProfileStat {
    int64_t count = 0;
    double totalMs = 0.0;
    double maxMs = 0.0;
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_runner_stage(
        bool profile,
        std::string_view stage,
        Clock::time_point start,
        Clock::time_point end) {
    if (!profile)
        return;
    printf("[profile] runner_stage=%.*s wall_ms=%.3f\n",
           static_cast<int>(stage.size()),
           stage.data(),
           elapsed_ms(start, end));
    fflush(stdout);
}

Result<sandy::device::DevicePagedTensorView> paged_view(
        sandy::device::Device& device,
        sandy::device::DevicePagedTensorId tensor) {
    auto meta = device.pagedMeta(tensor);
    if (!meta)
        return make_error(meta.error());
    return sandy::device::DevicePagedTensorView{tensor, meta.take()};
}

Result<sandy::engine::RunTensorTuple> make_paged_tuple(
        sandy::device::Device& device,
        const std::vector<sandy::device::DevicePagedTensorId>& ids) {
    sandy::engine::RunTensorTuple tuple;
    tuple.elements.reserve(ids.size());
    for (auto id : ids) {
        auto view = paged_view(device, id);
        if (!view)
            return make_error(view.error());
        tuple.elements.push_back(view.take());
    }
    return tuple;
}

Result<void> add_cache_group(
        sandy::device::Device& device,
        EvalTokenCaches& caches,
        int count,
        sandy::core::Shape shape,
        int64_t growDim,
        int64_t pageSize) {
    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(shape, sandy::core::DType::BF16);
    poolDesc.growDim = growDim;
    poolDesc.pageSize = pageSize;

    auto pool = device.createPagedPool(poolDesc);
    if (!pool)
        return make_error(pool.error());

    EvalTokenCaches::Group group;
    group.pool = *pool;
    group.tensors.reserve(static_cast<size_t>(count));

    auto initialDims = shape.dims();
    initialDims[static_cast<size_t>(growDim)] = 0;
    sandy::core::Shape initialShape(std::move(initialDims));
    for (int i = 0; i < count; i++) {
        auto tensor = device.allocPaged(group.pool, initialShape);
        if (!tensor)
            return make_error(tensor.error());
        group.tensors.push_back(*tensor);
    }

    caches.groups.push_back(std::move(group));
    return {};
}

Result<EvalTokenCaches> create_eval_token_caches(
        sandy::device::Device& device,
        std::string_view architecture) {
    EvalTokenCaches caches;
    if (architecture == "tinyllama") {
        auto k = add_cache_group(
            device,
            caches,
            kTinyLlamaKVLayerCount,
            sandy::core::Shape({1, 4, sandy::core::Shape::kDynamic, 64}),
            2,
            32);
        if (!k)
            return make_error(k.error());
        auto v = add_cache_group(
            device,
            caches,
            kTinyLlamaKVLayerCount,
            sandy::core::Shape({1, 4, sandy::core::Shape::kDynamic, 64}),
            2,
            32);
        if (!v)
            return make_error(v.error());
        return caches;
    }

    int localCount = kGemma4E2BLocalCacheCount;
    int globalCount = kGemma4E2BGlobalCacheCount;
    int localKvHeads = 1;
    int globalKvHeads = 1;
    if (architecture == "gemma4e4b") {
        localCount = kGemma4E4BLocalCacheCount;
        globalCount = kGemma4E4BGlobalCacheCount;
        localKvHeads = 2;
        globalKvHeads = 2;
    } else if (
            architecture == "gemma4a4b26b" ||
            architecture == "gemma4a4b" ||
            architecture == "gemma4moe") {
        localCount = kGemma4A4BLocalCacheCount;
        globalCount = kGemma4A4BGlobalCacheCount;
        localKvHeads = 8;
        globalKvHeads = 2;
    }

    auto localK = add_cache_group(
        device,
        caches,
        localCount,
        sandy::core::Shape({1, localKvHeads, sandy::core::Shape::kDynamic, 256}),
        2,
        32);
    if (!localK)
        return make_error(localK.error());
    auto localV = add_cache_group(
        device,
        caches,
        localCount,
        sandy::core::Shape({1, localKvHeads, sandy::core::Shape::kDynamic, 256}),
        2,
        32);
    if (!localV)
        return make_error(localV.error());
    auto globalK = add_cache_group(
        device,
        caches,
        globalCount,
        sandy::core::Shape({1, globalKvHeads, sandy::core::Shape::kDynamic, 512}),
        2,
        32);
    if (!globalK)
        return make_error(globalK.error());
    auto globalV = add_cache_group(
        device,
        caches,
        globalCount,
        sandy::core::Shape({1, globalKvHeads, sandy::core::Shape::kDynamic, 512}),
        2,
        32);
    if (!globalV)
        return make_error(globalV.error());
    return caches;
}

Result<std::vector<sandy::engine::RunInput>> make_eval_inputs(
        sandy::device::Device& device,
        const EvalTokenCaches& caches,
        int64_t token,
        int64_t position) {
    std::vector<sandy::engine::RunInput> inputs;
    inputs.reserve(2 + caches.groups.size());
    inputs.push_back(make_i64_buffer("input_id", sandy::core::Shape({1, 1}), token));
    inputs.push_back(make_i64_buffer("position_id", sandy::core::Shape({1}), position));
    for (const auto& group : caches.groups) {
        auto tuple = make_paged_tuple(device, group.tensors);
        if (!tuple)
            return make_error(tuple.error());
        inputs.push_back(tuple.take());
    }
    return inputs;
}

Result<std::vector<sandy::engine::RunInput>> make_prefill_inputs(
        sandy::device::Device& device,
        const EvalTokenCaches& caches,
        const std::vector<int64_t>& tokens,
        int64_t position) {
    if (tokens.empty())
        return make_error("prefill tokens must not be empty");

    std::vector<sandy::engine::RunInput> inputs;
    inputs.reserve(2 + caches.groups.size());
    inputs.push_back(make_i64_vector_buffer(
        "input_id",
        sandy::core::Shape({1, static_cast<int64_t>(tokens.size())}),
        tokens));
    inputs.push_back(make_i64_buffer("position_id", sandy::core::Shape({1}), position));
    for (const auto& group : caches.groups) {
        auto tuple = make_paged_tuple(device, group.tensors);
        if (!tuple)
            return make_error(tuple.error());
        inputs.push_back(tuple.take());
    }
    return inputs;
}

Result<sandy::engine::RunTensorTuple*> require_tuple_output(
        std::vector<sandy::engine::RunOutput>& outputs,
        size_t index) {
    if (index >= outputs.size())
        return make_error("runner output tuple index out of range");
    auto* tuple = std::get_if<sandy::engine::RunTensorTuple>(&outputs[index]);
    if (!tuple)
        return make_error("runner expected tuple output");
    return tuple;
}

Result<sandy::engine::TensorBufferPtr> require_tensor_tuple_element(
        sandy::engine::RunTensorTuple& tuple,
        size_t index) {
    if (index >= tuple.elements.size())
        return make_error("runner tuple element index out of range");
    auto* tensor = std::get_if<sandy::engine::TensorBufferPtr>(&tuple.elements[index]);
    if (!tensor || !*tensor)
        return make_error("runner expected dense tensor tuple element");
    return *tensor;
}

Result<sandy::engine::TensorBufferPtr> require_tensor_output(
        std::vector<sandy::engine::RunOutput>& outputs,
        size_t index) {
    if (index >= outputs.size())
        return make_error("runner output tensor index out of range");
    auto* tensor = std::get_if<sandy::engine::TensorBufferPtr>(&outputs[index]);
    if (!tensor || !*tensor)
        return make_error("runner expected dense tensor output");
    return *tensor;
}

void usage() {
    fprintf(stderr,
            "usage: multi_gemma4_runner "
            "[--eval-token|--prefill] "
            "[--architecture gemma4e2b|gemma4e4b|gemma4a4b26b|tinyllama] [--profile] "
            "<program.sandy.go> <weights.safetensors> "
            "<emit_count> <token_id>...\n");
}

} // namespace

int main(int argc, char* argv[]) {
    int arg = 1;
    bool evalTokenMode = false;
    bool prefillMode = false;
    bool profile = false;
    bool dumpKernelIR = false;
    std::string architecture = "gemma4e2b";
    while (arg < argc && std::string_view(argv[arg]).starts_with("--")) {
        std::string_view option(argv[arg]);
        if (option == "--eval-token") {
            evalTokenMode = true;
        } else if (option == "--prefill") {
            prefillMode = true;
        } else if (option == "--architecture") {
            if (arg + 1 >= argc) {
                fprintf(stderr, "--architecture requires a value\n");
                usage();
                return 1;
            }
            architecture = argv[++arg];
        } else if (option == "--profile") {
            profile = true;
        } else if (option == "--dump-kernel-ir") {
            dumpKernelIR = true;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[arg]);
            usage();
            return 1;
        }
        arg++;
    }

    if (argc - arg < 4) {
        usage();
        return 1;
    }
    if (evalTokenMode && prefillMode) {
        fprintf(stderr, "--eval-token and --prefill are mutually exclusive\n");
        return 1;
    }

    const char* programPath = argv[arg];
    const char* weightsPath = argv[arg + 1];
    int64_t emitCount = 0;
    if (!parse_int64(argv[arg + 2], emitCount) || emitCount < 0) {
        fprintf(stderr, "emit_count must be a non-negative integer\n");
        return 1;
    }

    std::vector<int64_t> tokens;
    for (int i = arg + 3; i < argc; i++) {
        int64_t token = 0;
        if (!parse_int64(argv[i], token) || token < 0) {
            fprintf(stderr, "invalid token id: %s\n", argv[i]);
            return 1;
        }
        tokens.push_back(token);
    }
    if (tokens.empty()) {
        fprintf(stderr, "at least one token id is required\n");
        return 1;
    }

    auto totalStart = Clock::now();

    auto stageStart = Clock::now();
    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo(programPath);
    print_runner_stage(profile, "load_program", stageStart, Clock::now());

    stageStart = Clock::now();
    auto weights = sandy::weight::EagerSafeTensorWeights::load(weightsPath);
    print_runner_stage(profile, "load_host_weights", stageStart, Clock::now());

    stageStart = Clock::now();
    sandy::engine::TensorMap weightMap;
    add_tensors_to_map(*weights, weightMap);
    print_runner_stage(profile, "map_host_weights", stageStart, Clock::now());

    if (evalTokenMode || prefillMode) {
        sandy::ir::mid_ir::MaterializeOptions options;
        options.input_tensor_descs["input_id"] =
            sandy::core::TensorDesc(
                "input_id",
                prefillMode
                    ? sandy::core::Shape({1, sandy::core::Shape::kDynamic})
                    : sandy::core::Shape({1, 1}),
                sandy::core::DType::I64);
        options.input_tensor_descs["position_id"] =
            sandy::core::TensorDesc(
                "position_id",
                sandy::core::Shape({1}),
                sandy::core::DType::I64);

        stageStart = Clock::now();
        auto midResult = compiler.materialize_mid_ir(highGraph, *weights, options);
        print_runner_stage(profile, "materialize_mid_ir", stageStart, Clock::now());
        if (!midResult) {
            fprintf(stderr, "materialize error: %s\n", midResult.error().c_str());
            return 1;
        }

        stageStart = Clock::now();
        std::unique_ptr<sandy::device::Device> device;
#ifdef SANDY_RUNNER_ENABLE_CUDA
        device = std::make_unique<sandy::device::CudaDevice>();
#else
        device = std::make_unique<sandy::device::CpuDevice>();
#endif
        auto* runtimeDevice = device.get();
        std::vector<std::unique_ptr<sandy::device::Device>> devices;
        devices.push_back(std::move(device));
        sandy::engine::Engine engine(std::move(devices));
        print_runner_stage(profile, "create_engine", stageStart, Clock::now());

        sandy::engine::EngineCompileOptions compileOptions;
#ifdef SANDY_RUNNER_ENABLE_CUDA
        compileOptions.fusor.attention = true;
#endif
        stageStart = Clock::now();
        auto planResult = engine.compile(**midResult, &compileOptions);
        print_runner_stage(profile, "compile", stageStart, Clock::now());
        if (!planResult) {
            fprintf(stderr, "plan error: %s\n", planResult.error().c_str());
            return 1;
        }
        if (dumpKernelIR) {
            (*planResult)->graph->dump();
            return 0;
        }

        stageStart = Clock::now();
        auto deviceWeights = engine.loadWeights(**planResult, weightMap);
        print_runner_stage(profile, "load_device_weights", stageStart, Clock::now());
        if (!deviceWeights) {
            fprintf(stderr, "device weight load error: %s\n", deviceWeights.error().c_str());
            return 1;
        }

        stageStart = Clock::now();
        auto caches = create_eval_token_caches(*runtimeDevice, architecture);
        print_runner_stage(profile, "create_caches", stageStart, Clock::now());
        if (!caches) {
            fprintf(stderr, "cache creation error: %s\n", caches.error().c_str());
            return 1;
        }

        sandy::engine::EngineRunOptions runOptions;
        std::unordered_map<int, ProfileStat> profileStats;
        std::unordered_map<std::string, ProfileStat> profileStageStats;
        int64_t profileKernelCount = 0;
        int64_t profileStageCount = 0;
        double profileTotalMs = 0.0;
        double profileStageTotalMs = 0.0;
        if (profile) {
            runOptions.profileKernel = [&](const sandy::engine::EngineProfileEvent& event) {
                profileKernelCount++;
                profileTotalMs += event.elapsedMs;
                auto key = static_cast<int>(event.opKind);
                auto& stat = profileStats[key];
                stat.count++;
                stat.totalMs += event.elapsedMs;
                stat.maxMs = std::max(stat.maxMs, event.elapsedMs);
            };
            runOptions.profileStage = [&](const sandy::engine::EngineProfileStageEvent& event) {
                profileStageCount++;
                profileStageTotalMs += event.elapsedMs;
                auto& stat = profileStageStats[event.stage];
                stat.count++;
                stat.totalMs += event.elapsedMs;
                stat.maxMs = std::max(stat.maxMs, event.elapsedMs);
            };
        }

        auto runOnce = [&](std::vector<sandy::engine::RunInput>& inputs) {
            auto start = Clock::now();
            auto result = engine.runValues(
                **planResult,
                inputs,
                **deviceWeights,
                profile ? &runOptions : nullptr);
            auto end = Clock::now();
            return std::pair<
                Result<std::vector<sandy::engine::RunOutput>>,
                double>{std::move(result), elapsed_ms(start, end)};
        };

        if (prefillMode) {
            auto inputs = make_prefill_inputs(*runtimeDevice, *caches, tokens, 0);
            if (!inputs) {
                fprintf(stderr, "prefill input error: %s\n", inputs.error().c_str());
                return 1;
            }

            auto prefillStart = Clock::now();
            auto [run, wallMs] = runOnce(*inputs);
            if (!run) {
                fprintf(stderr, "prefill run error: %s\n", run.error().c_str());
                return 1;
            }
            print_runner_stage(profile, "prefill_eval", prefillStart, Clock::now());
            if (profile) {
                double tokensPerSecond = wallMs <= 0.0
                    ? 0.0
                    : static_cast<double>(tokens.size()) * 1000.0 / wallMs;
                printf("[profile] prefill tokens=%zu wall_ms=%.3f toks_per_s=%.3f\n",
                       tokens.size(),
                       wallMs,
                       tokensPerSecond);
            }

            if (emitCount > 0) {
                auto logits = require_tensor_output(*run, 0);
                if (!logits) {
                    fprintf(stderr, "logits output error: %s\n", logits.error().c_str());
                    return 1;
                }
                auto next = argmax_at(**logits, static_cast<int64_t>(tokens.size()) - 1);
                if (!next) {
                    fprintf(stderr, "argmax error at prefill tail: %s\n", next.error().c_str());
                    return 1;
                }
                auto pair = next.take();
                printf("[prefill_next] token=%lld score=%.6g\n",
                       static_cast<long long>(pair.first),
                       pair.second);
            }

            printf("[all_tokens]");
            for (int64_t token : tokens)
                printf(" %lld", static_cast<long long>(token));
            printf("\n");
            if (profile) {
                printf("[profile] total kernels=%lld time_ms=%.3f\n",
                       static_cast<long long>(profileKernelCount),
                       profileTotalMs);
                printf("[profile] by op:\n");
                std::vector<std::pair<int, ProfileStat>> stats(
                    profileStats.begin(),
                    profileStats.end());
                std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
                    return a.second.totalMs > b.second.totalMs;
                });
                for (const auto& [kind, stat] : stats) {
                    double avgMs = stat.count == 0 ? 0.0 : stat.totalMs / static_cast<double>(stat.count);
                    printf("  %s count=%lld total_ms=%.3f avg_ms=%.3f max_ms=%.3f\n",
                           sandy::ir::kernel_ir::op_kind_name(
                               static_cast<sandy::ir::kernel_ir::OpKind>(kind)),
                           static_cast<long long>(stat.count),
                           stat.totalMs,
                           avgMs,
                           stat.maxMs);
                }
                printf("[profile] engine stages=%lld summed_time_ms=%.3f\n",
                       static_cast<long long>(profileStageCount),
                       profileStageTotalMs);
                printf("[profile] by engine stage:\n");
                std::vector<std::pair<std::string, ProfileStat>> stageStats(
                    profileStageStats.begin(),
                    profileStageStats.end());
                std::sort(stageStats.begin(), stageStats.end(), [](const auto& a, const auto& b) {
                    return a.second.totalMs > b.second.totalMs;
                });
                for (const auto& [stage, stat] : stageStats) {
                    double avgMs = stat.count == 0 ? 0.0 : stat.totalMs / static_cast<double>(stat.count);
                    printf("  %s count=%lld total_ms=%.3f avg_ms=%.3f max_ms=%.3f\n",
                           stage.c_str(),
                           static_cast<long long>(stat.count),
                           stat.totalMs,
                           avgMs,
                           stat.maxMs);
                }
                print_runner_stage(profile, "total_runner", totalStart, Clock::now());
            }
            return 0;
        }

        auto evalOnce = [&](int64_t token, int64_t position) {
            auto inputs = make_eval_inputs(*runtimeDevice, *caches, token, position);
            if (!inputs)
                return Result<std::vector<sandy::engine::RunOutput>>(make_error(inputs.error()));
            auto [result, wallMs] = runOnce(*inputs);
            if (profile) {
                printf("[profile] eval_token position=%lld wall_ms=%.3f\n",
                       static_cast<long long>(position),
                       wallMs);
            }
            return result;
        };

        int64_t nextToken = 0;
        float nextScore = 0.0f;
        auto promptStart = Clock::now();
        for (size_t i = 0; i < tokens.size(); i++) {
            auto run = evalOnce(tokens[i], static_cast<int64_t>(i));
            if (!run) {
                fprintf(stderr, "run error at prompt pos %zu: %s\n", i, run.error().c_str());
                return 1;
            }
            if (emitCount > 0 && i + 1 == tokens.size()) {
                auto logits = require_tensor_output(*run, 0);
                if (!logits) {
                    fprintf(stderr, "logits output error: %s\n", logits.error().c_str());
                    return 1;
                }
                auto next = argmax_at(**logits, 0);
                if (!next) {
                    fprintf(stderr, "argmax error at prompt pos %zu: %s\n", i, next.error().c_str());
                    return 1;
                }
                auto pair = next.take();
                nextToken = pair.first;
                nextScore = pair.second;
            }
        }
        print_runner_stage(profile, "prompt_eval", promptStart, Clock::now());

        std::vector<int64_t> generated;
        auto generateStart = Clock::now();
        for (int64_t step = 0; step < emitCount; step++) {
            auto emitted = nextToken;
            auto score = nextScore;
            generated.push_back(emitted);
            tokens.push_back(emitted);
            printf("[step %lld] pos=%lld token=%lld score=%.6g\n",
                   static_cast<long long>(step),
                   static_cast<long long>(tokens.size() - 2),
                   static_cast<long long>(emitted),
                   score);
            fflush(stdout);

            if (step + 1 >= emitCount)
                break;

            auto position = static_cast<int64_t>(tokens.size() - 1);
            auto run = evalOnce(emitted, position);
            if (!run) {
                fprintf(stderr, "run error at generated pos %lld: %s\n",
                        static_cast<long long>(position),
                        run.error().c_str());
                return 1;
            }
            auto logits = require_tensor_output(*run, 0);
            if (!logits) {
                fprintf(stderr, "logits output error: %s\n", logits.error().c_str());
                return 1;
            }
            auto next = argmax_at(**logits, 0);
            if (!next) {
                fprintf(stderr, "argmax error at generated pos %lld: %s\n",
                        static_cast<long long>(position),
                        next.error().c_str());
                return 1;
            }
            auto pair = next.take();
            nextToken = pair.first;
            nextScore = pair.second;
        }
        print_runner_stage(profile, "generate_eval", generateStart, Clock::now());

        printf("[generated]");
        for (int64_t token : generated)
            printf(" %lld", static_cast<long long>(token));
        printf("\n[all_tokens]");
        for (int64_t token : tokens)
            printf(" %lld", static_cast<long long>(token));
        printf("\n");
        if (profile) {
            printf("[profile] total kernels=%lld time_ms=%.3f\n",
                   static_cast<long long>(profileKernelCount),
                   profileTotalMs);
            printf("[profile] by op:\n");
            std::vector<std::pair<int, ProfileStat>> stats(
                profileStats.begin(),
                profileStats.end());
            std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
                return a.second.totalMs > b.second.totalMs;
            });
            for (const auto& [kind, stat] : stats) {
                double avgMs = stat.count == 0 ? 0.0 : stat.totalMs / static_cast<double>(stat.count);
                printf("  %s count=%lld total_ms=%.3f avg_ms=%.3f max_ms=%.3f\n",
                       sandy::ir::kernel_ir::op_kind_name(
                           static_cast<sandy::ir::kernel_ir::OpKind>(kind)),
                       static_cast<long long>(stat.count),
                       stat.totalMs,
                       avgMs,
                       stat.maxMs);
            }
            printf("[profile] engine stages=%lld summed_time_ms=%.3f\n",
                   static_cast<long long>(profileStageCount),
                   profileStageTotalMs);
            printf("[profile] by engine stage:\n");
            std::vector<std::pair<std::string, ProfileStat>> stageStats(
                profileStageStats.begin(),
                profileStageStats.end());
            std::sort(stageStats.begin(), stageStats.end(), [](const auto& a, const auto& b) {
                return a.second.totalMs > b.second.totalMs;
            });
            for (const auto& [stage, stat] : stageStats) {
                double avgMs = stat.count == 0 ? 0.0 : stat.totalMs / static_cast<double>(stat.count);
                printf("  %s count=%lld total_ms=%.3f avg_ms=%.3f max_ms=%.3f\n",
                       stage.c_str(),
                       static_cast<long long>(stat.count),
                       stat.totalMs,
                       avgMs,
                       stat.maxMs);
            }
            print_runner_stage(profile, "total_runner", totalStart, Clock::now());
        }
        return 0;
    }

    std::vector<int64_t> generated;
    for (int64_t step = 0; step < emitCount; step++) {
        int64_t tokenIndex = 0;
        auto input = make_input_buffer(tokens, tokenIndex);
        std::vector<sandy::engine::TensorBufferPtr> inputs{input};

        sandy::ir::mid_ir::MaterializeOptions options;
        options.input_tensor_descs["input_ids"] =
            sandy::core::TensorDesc(
                "input_ids",
                sandy::core::Shape({1, static_cast<int64_t>(tokens.size())}),
                sandy::core::DType::I64);

        auto midResult = compiler.materialize_mid_ir(highGraph, *weights, options);
        if (!midResult) {
            fprintf(stderr, "materialize error at step %lld: %s\n",
                    static_cast<long long>(step),
                    midResult.error().c_str());
            return 1;
        }

        std::vector<std::unique_ptr<sandy::device::Device>> devices;
#ifdef SANDY_RUNNER_ENABLE_CUDA
        devices.push_back(std::make_unique<sandy::device::CudaDevice>());
#else
        devices.push_back(std::make_unique<sandy::device::CpuDevice>());
#endif
        sandy::engine::Engine engine(std::move(devices));
        sandy::engine::EngineCompileOptions compileOptions;
#ifdef SANDY_RUNNER_ENABLE_CUDA
        compileOptions.fusor.attention = true;
#endif
        auto planResult = engine.compile(**midResult, &compileOptions);
        if (!planResult) {
            fprintf(stderr, "plan error at step %lld: %s\n",
                    static_cast<long long>(step),
                    planResult.error().c_str());
            return 1;
        }
        if (dumpKernelIR) {
            (*planResult)->graph->dump();
            return 0;
        }

        auto runResult = engine.run(**planResult, inputs, weightMap);
        if (!runResult) {
            fprintf(stderr, "run error at step %lld: %s\n",
                    static_cast<long long>(step),
                    runResult.error().c_str());
            return 1;
        }
        auto outputs = runResult.take();
        if (outputs.empty() || !outputs[0]) {
            fprintf(stderr, "runner produced no logits output\n");
            return 1;
        }

        auto next = argmax_at(*outputs[0], tokenIndex);
        if (!next) {
            fprintf(stderr, "argmax error at step %lld: %s\n",
                    static_cast<long long>(step),
                    next.error().c_str());
            return 1;
        }
        auto [token, score] = next.take();
        tokens.push_back(token);
        generated.push_back(token);
        printf("[step %lld] pos=%lld token=%lld score=%.6g\n",
               static_cast<long long>(step),
               static_cast<long long>(tokenIndex),
               static_cast<long long>(token),
               score);
        fflush(stdout);
    }

    printf("[generated]");
    for (int64_t token : generated)
        printf(" %lld", static_cast<long long>(token));
    printf("\n[all_tokens]");
    for (int64_t token : tokens)
        printf(" %lld", static_cast<long long>(token));
    printf("\n");
    return 0;
}
