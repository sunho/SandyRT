#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "jit/CudaJitLaunchUtils.cuh"

#include <cmath>

namespace sandy::device {

namespace {

constexpr int kMaxScalars = 32;

struct DeviceScalarNode {
    int id = 0;
    ir::kernel_ir::ScalarOp op = ir::kernel_ir::ScalarOp::Constant;
    core::DType dtype = core::DType::F32;
    uint32_t inputIndex = 0;
    double constant = 0.0;
    int operands[2]{};
    int operandCount = 0;
};

struct DeviceElementwiseProgram {
    cuda_kernel::TensorInputs<> inputs;
    ir::kernel_ir::BroadcastMode inputBroadcasts[cuda_kernel::kMaxInputs]{};
    cuda_kernel::TensorArg output;
    DeviceScalarNode scalars[kMaxScalars]{};
    int scalarCount = 0;
    int result = 0;
    int64_t numel = 0;
};

Result<void> validate_elementwise_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda elementwise ") + name + " unsupported dtype");
    return {};
}

Result<DeviceElementwiseProgram> pack_elementwise_program(
        const CudaLaunchContext& context,
        const CudaElementwiseProgram& program) {
    if (context.inputs.size() > cuda_kernel::kMaxInputs)
        return make_error("cuda elementwise input count exceeds kernel max inputs");
    if (program.scalars.size() > kMaxScalars)
        return make_error("cuda elementwise scalar count exceeds kernel max scalars");
    if (program.result >= kMaxScalars)
        return make_error("cuda elementwise result scalar id exceeds kernel max scalars");

    auto outputDtype = validate_elementwise_dtype(context.outputs[0], "output");
    if (!outputDtype)
        return make_error(outputDtype.error());

    DeviceElementwiseProgram packed;
    auto output = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!output)
        return make_error(output.error());
    packed.output = output.take();
    packed.numel = packed.output.numel;
    packed.result = static_cast<int>(program.result);

    packed.inputs.count = static_cast<int>(context.inputs.size());
    for (int i = 0; i < packed.inputs.count; ++i) {
        auto dtype = validate_elementwise_dtype(context.inputs[static_cast<size_t>(i)], "input");
        if (!dtype)
            return make_error(dtype.error());
        auto input = cuda_kernel::pack_tensor_arg(context.inputs[static_cast<size_t>(i)]);
        if (!input)
            return make_error(input.error());
        packed.inputs.items[i] = input.take();
        packed.inputBroadcasts[i] =
            program.elementwiseInputs[static_cast<size_t>(i)].broadcast;
    }

    bool seen[kMaxScalars]{};
    packed.scalarCount = static_cast<int>(program.scalars.size());
    for (int i = 0; i < packed.scalarCount; ++i) {
        const auto& src = program.scalars[static_cast<size_t>(i)];
        if (src.id >= kMaxScalars)
            return make_error("cuda elementwise scalar id exceeds kernel max scalars");
        if (seen[src.id])
            return make_error("cuda elementwise duplicate scalar id");
        if (src.operands.size() > 2)
            return make_error("cuda elementwise scalar op has too many operands");
        if (src.op == ir::kernel_ir::ScalarOp::Load &&
            src.inputIndex >= context.inputs.size()) {
            return make_error("cuda elementwise scalar load references invalid input");
        }

        auto& dst = packed.scalars[i];
        dst.id = static_cast<int>(src.id);
        dst.op = src.op;
        dst.dtype = src.dtype;
        dst.inputIndex = src.inputIndex;
        dst.constant = src.constant;
        dst.operandCount = static_cast<int>(src.operands.size());

        for (int j = 0; j < dst.operandCount; ++j) {
            auto operand = src.operands[static_cast<size_t>(j)];
            if (operand >= kMaxScalars)
                return make_error("cuda elementwise operand id exceeds kernel max scalars");
            if (!seen[operand])
                return make_error("cuda elementwise scalar operands must be in dependency order");
            dst.operands[j] = static_cast<int>(operand);
        }

        seen[src.id] = true;
    }

    if (!seen[program.result])
        return make_error("cuda elementwise result references missing scalar");

    return packed;
}

Result<SandyElementwiseParams> pack_jit_elementwise_params(
        const DeviceElementwiseProgram& source) {
    SandyElementwiseParams target{};
    for (int i = 0; i < source.inputs.count; ++i) {
        auto input = pack_jit_tensor_arg(source.inputs.items[i]);
        if (!input)
            return make_error(input.error());
        target.inputs[i] = input.take();
        switch (source.inputBroadcasts[i]) {
            case ir::kernel_ir::BroadcastMode::None:
                target.broadcasts[i] = SANDY_JIT_BROADCAST_NONE;
                break;
            case ir::kernel_ir::BroadcastMode::RightAligned:
                target.broadcasts[i] = SANDY_JIT_BROADCAST_RIGHT_ALIGNED;
                break;
        }
    }
    auto output = pack_jit_tensor_arg(source.output);
    if (!output)
        return make_error(output.error());
    target.output = output.take();
    return target;
}

__device__ float apply_scalar(
        const DeviceScalarNode& node,
        const float* values) {
    float a = node.operandCount > 0 ? values[node.operands[0]] : 0.0f;
    float b = node.operandCount > 1 ? values[node.operands[1]] : 0.0f;

    switch (node.op) {
        case ir::kernel_ir::ScalarOp::Constant:
            return static_cast<float>(node.constant);
        case ir::kernel_ir::ScalarOp::Add:
            return a + b;
        case ir::kernel_ir::ScalarOp::Sub:
            return a - b;
        case ir::kernel_ir::ScalarOp::Mul:
            return a * b;
        case ir::kernel_ir::ScalarOp::Div:
            return a / b;
        case ir::kernel_ir::ScalarOp::Max:
            return fmaxf(a, b);
        case ir::kernel_ir::ScalarOp::Min:
            return fminf(a, b);
        case ir::kernel_ir::ScalarOp::Neg:
            return -a;
        case ir::kernel_ir::ScalarOp::Sqrt:
            return sqrtf(a);
        case ir::kernel_ir::ScalarOp::Rsqrt:
            return rsqrtf(a);
        case ir::kernel_ir::ScalarOp::Exp:
            return expf(a);
        case ir::kernel_ir::ScalarOp::Log:
            return logf(a);
        case ir::kernel_ir::ScalarOp::Tanh:
            return tanhf(a);
        case ir::kernel_ir::ScalarOp::ReLU:
            return fmaxf(a, 0.0f);
        case ir::kernel_ir::ScalarOp::Cast:
            return a;
        case ir::kernel_ir::ScalarOp::Load:
            return 0.0f;
    }

    return 0.0f;
}

__global__ void elementwise_kernel(DeviceElementwiseProgram program) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.numel)
        return;

    float values[kMaxScalars]{};
    for (int i = 0; i < program.scalarCount; ++i) {
        const auto& node = program.scalars[i];
        if (node.op == ir::kernel_ir::ScalarOp::Load) {
            const auto& input = program.inputs.items[node.inputIndex];
            int64_t inputLinear = linear;
            if (program.inputBroadcasts[node.inputIndex] ==
                ir::kernel_ir::BroadcastMode::RightAligned) {
                inputLinear = input.numel == 0 ? 0 : linear % input.numel;
            }
            values[node.id] = cuda_kernel::load_float(
                input,
                inputLinear);
        } else {
            values[node.id] = apply_scalar(node, values);
        }
    }

    cuda_kernel::store_float(program.output, linear, values[program.result]);
}

} // namespace

Result<void> launch_cuda_elementwise(
        const CudaLaunchContext& context,
        const CudaElementwiseProgram& program) {
    auto valid = validate_context(
        context,
        program.elementwiseInputs.size(),
        1,
        "elementwise");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_elementwise_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.numel == 0)
        return {};

    int blocks = static_cast<int>(
        (launchProgram.numel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    if (program.jitKernel) {
        auto params = pack_jit_elementwise_params(launchProgram);
        if (!params)
            return make_error(params.error());
        auto launchParams = params.take();
        void* arguments[] = {&launchParams};
        return program.jitKernel->launch(
            dim3(static_cast<unsigned>(blocks)),
            dim3(cuda_kernel::kBlockSize),
            0,
            context.stream,
            arguments);
    }
    elementwise_kernel<<<blocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
        launchProgram);
    return cuda_check(cudaGetLastError(), "cuda elementwise launch");
}

} // namespace sandy::device
