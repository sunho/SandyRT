#include "Compiler.h"
#include "CpuDevice.h"
#include "Engine.h"
#include "SafeTensorWeights.h"

#include <algorithm>
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
#include <vector>

namespace {

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

void usage() {
    fprintf(stderr,
            "usage: multi_gemma4_runner "
            "<program.sandy.go> <weights.safetensors> <emit_count> <token_id>...\n");
}

} // namespace

int main(int argc, char* argv[]) {
    int arg = 1;
    if (argc - arg < 4) {
        usage();
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

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo(programPath);

    auto weights = sandy::weight::EagerSafeTensorWeights::load(weightsPath);
    sandy::engine::TensorMap weightMap;
    add_tensors_to_map(*weights, weightMap);

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
        devices.push_back(std::make_unique<sandy::device::CpuDevice>());
        sandy::engine::Engine engine(std::move(devices));
        auto planResult = engine.compile(**midResult);
        if (!planResult) {
            fprintf(stderr, "plan error at step %lld: %s\n",
                    static_cast<long long>(step),
                    planResult.error().c_str());
            return 1;
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
