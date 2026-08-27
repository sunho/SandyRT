#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "jit/CudaElementwiseJit.h"
#include "jit/CudaJitLaunchUtils.cuh"

#include <array>

namespace sandy::device {

namespace {

struct PackedElementwiseJitLaunch {
    SandyElementwiseParams params{};
    std::array<int, cuda_kernel::kMaxInputs + 1> accesses{};
    size_t inputCount = 0;
    int64_t numel = 0;
};

Result<void> validate_elementwise_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda elementwise ") + name + " unsupported dtype");
    return {};
}

Result<PackedElementwiseJitLaunch> pack_elementwise_jit_launch(
        const CudaLaunchContext& context,
        const CudaElementwiseProgram& program) {
    if (context.inputs.size() > cuda_kernel::kMaxInputs)
        return make_error("cuda elementwise input count exceeds kernel max inputs");

    auto outputDtype = validate_elementwise_dtype(context.outputs[0], "output");
    if (!outputDtype)
        return make_error(outputDtype.error());

    PackedElementwiseJitLaunch packed;
    auto output = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!output)
        return make_error("cuda elementwise output: " + output.error());
    auto outputArg = output.take();
    packed.numel = outputArg.numel;
    packed.accesses[context.inputs.size()] = jit_access_kind(outputArg.access);
    auto jitOutput = pack_jit_tensor_arg(outputArg);
    if (!jitOutput)
        return make_error(jitOutput.error());
    packed.params.output = jitOutput.take();

    packed.inputCount = context.inputs.size();
    for (size_t i = 0; i < packed.inputCount; ++i) {
        auto dtype = validate_elementwise_dtype(context.inputs[i], "input");
        if (!dtype)
            return make_error(dtype.error());
        auto input = cuda_kernel::pack_tensor_arg(context.inputs[i]);
        if (!input)
            return make_error(
                "cuda elementwise input " + std::to_string(i) + ": " + input.error());
        auto inputArg = input.take();
        packed.accesses[i] = jit_access_kind(inputArg.access);
        auto jitInput = pack_jit_tensor_arg(inputArg);
        if (!jitInput)
            return make_error(jitInput.error());
        packed.params.inputs[i] = jitInput.take();
        switch (program.elementwiseInputs[i].broadcast) {
            case ir::kernel_ir::BroadcastMode::None:
                packed.params.broadcasts[i] = SANDY_JIT_BROADCAST_NONE;
                break;
            case ir::kernel_ir::BroadcastMode::RightAligned:
                packed.params.broadcasts[i] = SANDY_JIT_BROADCAST_RIGHT_ALIGNED;
                break;
        }
    }
    return packed;
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

    if (!program.jitVariants)
        return make_error("cuda elementwise JIT kernel is missing");
    if (!context.jitCache)
        return make_error("cuda elementwise JIT cache is null");

    auto packed = pack_elementwise_jit_launch(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.numel == 0)
        return {};

    int blocks = static_cast<int>(
        (launchProgram.numel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    std::span<const int> inputAccesses(
        launchProgram.accesses.data(), launchProgram.inputCount);
    int outputAccess = launchProgram.accesses[launchProgram.inputCount];
    std::span<const int> accesses(
        launchProgram.accesses.data(), launchProgram.inputCount + 1);
    auto jit = program.jitVariants->getOrCompile(
        cudaJitAccessKey(accesses),
        [&] {
            return compileCudaElementwiseJit(
                context.cudaDevice,
                *context.jitCache,
                program,
                inputAccesses,
                outputAccess);
        });
    if (!jit)
        return make_error(jit.error());

    void* arguments[] = {&launchProgram.params};
    return (*jit)->launch(
        dim3(static_cast<unsigned>(blocks)),
        dim3(cuda_kernel::kBlockSize),
        0,
        context.stream,
        arguments);
}

} // namespace sandy::device
