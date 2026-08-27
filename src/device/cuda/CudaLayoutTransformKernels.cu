#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "jit/CudaJitLaunchUtils.cuh"
#include "jit/templates/CudaJitLayoutTransformAbi.cuh"

namespace sandy::device {

namespace {

__device__ int dtype_size_bytes(core::DType dtype) {
    switch (dtype) {
        case core::DType::F32:
        case core::DType::I32:
            return 4;
        case core::DType::F16:
        case core::DType::BF16:
            return 2;
        case core::DType::I64:
            return 8;
        case core::DType::U8:
            return 1;
    }
    return 0;
}

__global__ void layout_transform_kernel(
        cuda_kernel::TensorArg input,
        cuda_kernel::TensorArg output) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= output.numel)
        return;

    int elementSize = dtype_size_bytes(output.dtype);
    if (elementSize == 0)
        return;

    int64_t inputIndex = cuda_kernel::storage_index(input, linear);
    int64_t outputIndex = cuda_kernel::storage_index(output, linear);
    const uint8_t* inputBytes = static_cast<const uint8_t*>(input.data);
    uint8_t* outputBytes = static_cast<uint8_t*>(output.data);
    inputBytes += inputIndex * elementSize;
    outputBytes += outputIndex * elementSize;

    switch (elementSize) {
        case 1:
            *outputBytes = *inputBytes;
            return;
        case 2:
            *reinterpret_cast<uint16_t*>(outputBytes) =
                *reinterpret_cast<const uint16_t*>(inputBytes);
            return;
        case 4:
            *reinterpret_cast<uint32_t*>(outputBytes) =
                *reinterpret_cast<const uint32_t*>(inputBytes);
            return;
        case 8:
            *reinterpret_cast<uint64_t*>(outputBytes) =
                *reinterpret_cast<const uint64_t*>(inputBytes);
            return;
    }
}

Result<void> validate_layout_transform(
        const CudaDeviceBufferView& input,
        const CudaDeviceBufferView& output) {
    if (input.paged || output.paged)
        return make_error("cuda layout_transform does not support paged tensors");
    if (input.view.desc.dtype != output.view.desc.dtype)
        return make_error("cuda layout_transform dtype mismatch");
    auto inputNumel = input.view.desc.shape.numel();
    auto outputNumel = output.view.desc.shape.numel();
    if (inputNumel < 0 || outputNumel < 0)
        return make_error("cuda layout_transform requires static shapes");
    if (inputNumel != outputNumel)
        return make_error("cuda layout_transform element count mismatch");
    return {};
}

} // namespace

Result<void> launch_cuda_layout_transform(
        const CudaLaunchContext& context,
        const CudaLayoutTransformProgram& program) {
    if (program.transform == ir::kernel_ir::LayoutTransformKind::Slice)
        return make_error("slice layout aliases must not launch on CUDA");
    auto valid = validate_context(context, 1, 1, "layout_transform");
    if (!valid)
        return make_error(valid.error());

    auto transformValid = validate_layout_transform(context.inputs[0], context.outputs[0]);
    if (!transformValid)
        return make_error(transformValid.error());

    auto input = cuda_kernel::pack_tensor_arg(context.inputs[0]);
    if (!input)
        return make_error(input.error());
    auto output = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!output)
        return make_error(output.error());

    auto inputArg = input.take();
    auto outputArg = output.take();
    if (outputArg.numel == 0)
        return {};

    int blocks = static_cast<int>(
        (outputArg.numel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    if (program.jitKernel) {
        auto jitInput = pack_jit_tensor_arg(inputArg);
        if (!jitInput)
            return make_error(jitInput.error());
        auto jitOutput = pack_jit_tensor_arg(outputArg);
        if (!jitOutput)
            return make_error(jitOutput.error());
        SandyLayoutTransformParams params{jitInput.take(), jitOutput.take()};
        void* arguments[] = {&params};
        return program.jitKernel->launch(
            dim3(static_cast<unsigned>(blocks)),
            dim3(cuda_kernel::kBlockSize),
            0,
            context.stream,
            arguments);
    }
    layout_transform_kernel<<<blocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
        inputArg,
        outputArg);
    return cuda_check(cudaGetLastError(), "cuda layout_transform launch");
}

} // namespace sandy::device
