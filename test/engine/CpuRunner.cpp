#include "Compiler.h"
#include "CpuInterpreterBackend.h"
#include "Engine.h"
#include "SafeTensorWeights.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace {

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

void add_tensors_to_map(const sandy::weight::Weights& tensors,
                        sandy::engine::TensorMap& map) {
    for (const auto& desc : tensors.descriptors()) {
        auto tensor = tensors.get_tensor(desc.name);
        if (!tensor) {
            fprintf(stderr, "tensor listed in descriptors but missing: %s\n", desc.name.c_str());
            abort();
        }
        map[desc.name] = tensor;
    }
}

void print_tensor(const std::string& name,
                  const sandy::engine::backend::BackendBuffer& buffer) {
    const auto& desc = buffer.desc();
    printf("%s: %s%s\n",
           name.c_str(),
           sandy::core::dtype_name(desc.dtype),
           desc.shape.str().c_str());

    if (desc.dtype != sandy::core::DType::F32) {
        printf("  raw bytes: %zu\n", buffer.data().size());
        return;
    }

    auto numel = desc.shape.numel();
    if (numel < 0) {
        printf("  dynamic output shape\n");
        return;
    }

    printf("  [");
    for (int64_t i = 0; i < numel; i++) {
        if (i > 0) printf(", ");
        printf("%.6g", read_f32(buffer.data(), static_cast<size_t>(i)));
    }
    printf("]\n");

    if (numel > 0) {
        int64_t best = 0;
        float bestValue = read_f32(buffer.data(), 0);
        for (int64_t i = 1; i < numel; i++) {
            float value = read_f32(buffer.data(), static_cast<size_t>(i));
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        printf("  argmax: %lld (%.6g)\n", static_cast<long long>(best), bestValue);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: cpu_runner <program.sandy.go> <weights.safetensors> <inputs.safetensors>\n");
        return 1;
    }

    printf("[1/6] sandy go parsing: %s\n", argv[1]);
    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo(argv[1]);
    highGraph.dump();

    printf("[2/6] loading weights: %s\n", argv[2]);
    auto weights = sandy::weight::EagerSafeTensorWeights::load(argv[2]);

    printf("[3/6] loading inputs: %s\n", argv[3]);
    auto inputs = sandy::weight::EagerSafeTensorWeights::load(argv[3]);

    sandy::ir::mid_ir::MaterializeOptions options;
    for (const auto& desc : inputs->descriptors())
        options.input_tensor_descs[desc.name] = desc;

    printf("[4/6] materializing mid ir\n");
    auto midResult = compiler.materialize_mid_ir(highGraph, *weights, options);
    if (!midResult) {
        fprintf(stderr, "materialize error: %s\n", midResult.error().c_str());
        return 1;
    }
    auto midGraph = midResult.take();
    midGraph->dump();

    printf("[5/6] creating cpu backend plan\n");
    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());
    auto planResult = engine.create_plan(*midGraph);
    if (!planResult) {
        fprintf(stderr, "plan error: %s\n", planResult.error().c_str());
        return 1;
    }
    auto plan = planResult.take();

    sandy::engine::TensorMap inputMap;
    sandy::engine::TensorMap weightMap;
    add_tensors_to_map(*inputs, inputMap);
    add_tensors_to_map(*weights, weightMap);

    printf("[6/6] backend mid ir runs\n");
    auto runResult = engine.run(*plan, inputMap, weightMap);
    if (!runResult) {
        fprintf(stderr, "run error: %s\n", runResult.error().c_str());
        return 1;
    }

    auto outputs = runResult.take();
    for (const auto& [name, buffer] : outputs) {
        if (!buffer) {
            fprintf(stderr, "null output buffer: %s\n", name.c_str());
            return 1;
        }
        print_tensor(name, *buffer);
    }

    return 0;
}
