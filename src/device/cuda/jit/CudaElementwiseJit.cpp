#include "CudaElementwiseJit.h"

#include "CacheKey.h"
#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

#include <bit>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace sandy::device {

namespace {

using ir::kernel_ir::ScalarNode;
using ir::kernel_ir::ScalarOp;

int expected_arity(ScalarOp op) {
    switch (op) {
        case ScalarOp::Load:
        case ScalarOp::Constant:
            return 0;
        case ScalarOp::Neg:
        case ScalarOp::Sqrt:
        case ScalarOp::Rsqrt:
        case ScalarOp::Exp:
        case ScalarOp::Log:
        case ScalarOp::Tanh:
        case ScalarOp::ReLU:
        case ScalarOp::Cast:
            return 1;
        case ScalarOp::Add:
        case ScalarOp::Sub:
        case ScalarOp::Mul:
        case ScalarOp::Div:
        case ScalarOp::Max:
        case ScalarOp::Min:
            return 2;
    }
    return -1;
}

std::string scalar_name(ir::kernel_ir::ScalarId id) {
    return "s" + std::to_string(id);
}

Result<std::string> jit_float_dtype(core::DType dtype) {
    switch (dtype) {
        case core::DType::F32:
            return std::string("SANDY_JIT_F32");
        case core::DType::BF16:
            return std::string("SANDY_JIT_BF16");
        default:
            return make_error("CUDA elementwise JIT unsupported compute dtype");
    }
}

Result<std::string> scalar_expression(const ScalarNode& node) {
    auto operand = [&](size_t index) { return scalar_name(node.operands[index]); };
    switch (node.op) {
        case ScalarOp::Load: {
            auto dtype = jit_float_dtype(node.dtype);
            if (!dtype)
                return make_error(dtype.error());
            auto index = std::to_string(node.inputIndex);
            return "loader.template load<" + index + ", " + dtype.take() +
                ", SandyElementwiseInputAccess<" + index + ">>()";
        }
        case ScalarOp::Constant: {
            float value = static_cast<float>(node.constant);
            uint32_t bits = std::bit_cast<uint32_t>(value);
            std::ostringstream out;
            out << "__uint_as_float(0x" << std::hex << std::setw(8)
                << std::setfill('0') << bits << "u)";
            return out.str();
        }
        case ScalarOp::Add: return operand(0) + " + " + operand(1);
        case ScalarOp::Sub: return operand(0) + " - " + operand(1);
        case ScalarOp::Mul: return operand(0) + " * " + operand(1);
        case ScalarOp::Div: return operand(0) + " / " + operand(1);
        case ScalarOp::Max: return "fmaxf(" + operand(0) + ", " + operand(1) + ")";
        case ScalarOp::Min: return "fminf(" + operand(0) + ", " + operand(1) + ")";
        case ScalarOp::Neg: return "-" + operand(0);
        case ScalarOp::Sqrt: return "sqrtf(" + operand(0) + ")";
        case ScalarOp::Rsqrt: return "rsqrtf(" + operand(0) + ")";
        case ScalarOp::Exp: return "expf(" + operand(0) + ")";
        case ScalarOp::Log: return "logf(" + operand(0) + ")";
        case ScalarOp::Tanh: return "tanhf(" + operand(0) + ")";
        case ScalarOp::ReLU: return "fmaxf(" + operand(0) + ", 0.0f)";
        case ScalarOp::Cast: return operand(0);
    }
    return make_error("unsupported CUDA JIT scalar operation");
}

core::CacheKey scalar_dag_key(const CudaElementwiseProgram& program) {
    core::CacheKeyBuilder key("cuda-elementwise-scalar-dag-v1");
    key.addU32(program.result).addU64(program.scalars.size());
    for (const auto& node : program.scalars) {
        key.addU32(node.id)
           .addU32(static_cast<uint32_t>(node.op))
           .addDType(node.dtype)
           .addU32(node.inputIndex)
           .addU64(std::bit_cast<uint64_t>(node.constant))
           .addU64(node.operands.size());
        for (auto operand : node.operands)
            key.addU32(operand);
    }
    return std::move(key).finish();
}

} // namespace

Result<CudaElementwiseJitSource> emitCudaElementwiseJitSource(
        const CudaElementwiseProgram& program) {
    const ScalarNode* resultNode = nullptr;
    for (const auto& node : program.scalars) {
        if (node.id == program.result) {
            resultNode = &node;
            break;
        }
    }
    if (!resultNode)
        return make_error("CUDA elementwise JIT result references missing scalar");
    auto outputDtype = jit_float_dtype(resultNode->dtype);
    if (!outputDtype)
        return make_error(outputDtype.error());

    std::unordered_set<ir::kernel_ir::ScalarId> seen;
    std::ostringstream source;
    source << "#pragma once\n\n"
           << "struct GeneratedElementwiseEvaluator {\n"
           << "    static constexpr int kOutputDType = " << *outputDtype << ";\n\n"
           << "    template <typename Loader>\n"
           << "    __device__ __forceinline__ static float eval(const Loader& loader) {\n";

    for (const auto& node : program.scalars) {
        if (!seen.insert(node.id).second)
            return make_error("CUDA elementwise JIT found duplicate scalar id");
        int arity = expected_arity(node.op);
        if (arity < 0 || node.operands.size() != static_cast<size_t>(arity))
            return make_error("CUDA elementwise JIT scalar operation has invalid arity");
        if (node.op == ScalarOp::Load && node.inputIndex >= program.elementwiseInputs.size())
            return make_error("CUDA elementwise JIT load references invalid input");
        for (auto operand : node.operands) {
            if (!seen.contains(operand))
                return make_error("CUDA elementwise JIT operands must be in dependency order");
        }
        auto expression = scalar_expression(node);
        if (!expression)
            return make_error(expression.error());
        source << "        const float " << scalar_name(node.id)
               << " = " << *expression << ";\n";
    }
    source << "        return " << scalar_name(program.result) << ";\n"
           << "    }\n"
           << "};\n";

    auto key = scalar_dag_key(program);
    std::ostringstream entry;
    entry << "sandy_jit_elementwise_" << std::hex << key.hash();
    return CudaElementwiseJitSource{source.str(), entry.str()};
}

Result<CudaJitCache::KernelPtr> compileCudaElementwiseJit(
        int cudaDevice,
        CudaJitCache& cache,
        const CudaElementwiseProgram& program,
        std::span<const int> inputAccesses,
        int outputAccess) {
    auto generated = emitCudaElementwiseJitSource(program);
    if (!generated)
        return make_error(generated.error());
    if (inputAccesses.size() != program.elementwiseInputs.size())
        return make_error("CUDA elementwise JIT access policy count mismatch");

    std::ostringstream config;
    config << "#pragma once\n\n"
           << "template <int InputIndex>\n"
           << "inline constexpr int SandyElementwiseInputAccess = "
           << "SANDY_JIT_CONTIGUOUS;\n";
    for (size_t i = 0; i < inputAccesses.size(); ++i) {
        auto access = cudaJitAccessConstant(inputAccesses[i]);
        if (!access)
            return make_error(access.error());
        config << "template <>\ninline constexpr int "
               << "SandyElementwiseInputAccess<" << i << "> = "
               << *access << ";\n";
    }
    auto outputPolicy = cudaJitAccessConstant(outputAccess);
    if (!outputPolicy)
        return make_error(outputPolicy.error());
    config << "inline constexpr int SandyElementwiseOutputAccess = "
           << *outputPolicy << ";\n";

    CudaJitRequest request;
    request.sourceName = "CudaJitElementwiseKernel.cu";
    request.source = embeddedElementwiseKernelSource();
    request.headers = embeddedElementwiseHeaders();
    for (auto& header : request.headers) {
        if (header.name == "generated/ElementwiseEvaluator.cuh")
            header.source = generated->evaluatorSource;
        else if (header.name == "generated/ElementwiseConfig.cuh")
            header.source = config.str();
    }
    request.entryName = generated->entryName;
    request.options = {
        "-DSANDY_JIT_ENTRY_NAME=" + generated->entryName,
        "-lineinfo",
    };
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
