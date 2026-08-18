#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <limits>
#include <string>

namespace sandy::device {

namespace {

struct DeviceReductionProgram {
    cuda_kernel::TensorArg input;
    cuda_kernel::TensorArg output;
    int64_t axis = 0;
    int64_t reduceDim = 0;
};

Result<void> validate_reduction_tensor_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda reduction ") + name + " unsupported dtype");
    return {};
}

Result<DeviceReductionProgram> pack_reduction_program(
        const CudaLaunchContext& context,
        const CudaReductionProgram& program) {
    if (program.reduce != ir::kernel_ir::ReduceOp::Sum)
        return make_error("cuda reduction only supports sum");
    if (program.axes.size() != 1)
        return make_error("cuda reduction only supports one axis");
    if (!program.keepDims)
        return make_error("cuda reduction only supports keepDims=true");

    const auto& inputView = context.inputs[0];
    const auto& outputView = context.outputs[0];

    auto inputDtype = validate_reduction_tensor_dtype(inputView, "input");
    if (!inputDtype)
        return make_error(inputDtype.error());
    auto outputDtype = validate_reduction_tensor_dtype(outputView, "output");
    if (!outputDtype)
        return make_error(outputDtype.error());
    if (inputView.view.desc.dtype != outputView.view.desc.dtype)
        return make_error("cuda reduction output dtype mismatch");

    int rank = inputView.view.desc.shape.rank();
    if (rank < 1)
        return make_error("cuda reduction input must have rank >= 1");
    if (outputView.view.desc.shape.rank() != rank)
        return make_error("cuda reduction keepDims output rank mismatch");

    int64_t axis = program.axes[0];
    if (axis < -rank || axis >= rank)
        return make_error("cuda reduction axis out of range");
    axis = axis < 0 ? axis + rank : axis;

    for (int dim = 0; dim < rank; ++dim) {
        int64_t inputDim = inputView.view.desc.shape.dim(dim);
        int64_t outputDim = outputView.view.desc.shape.dim(dim);
        if (inputDim < 0 || outputDim < 0)
            return make_error("cuda reduction requires static shapes");
        if (dim == axis) {
            if (outputDim != 1)
                return make_error("cuda reduction keepDims output reduced dimension must be 1");
        } else if (outputDim != inputDim) {
            return make_error("cuda reduction output shape mismatch");
        }
    }

    auto input = cuda_kernel::pack_tensor_arg(inputView);
    if (!input)
        return make_error(input.error());
    auto output = cuda_kernel::pack_tensor_arg(outputView);
    if (!output)
        return make_error(output.error());

    DeviceReductionProgram packed;
    packed.input = input.take();
    packed.output = output.take();
    packed.axis = axis;
    packed.reduceDim = inputView.view.desc.shape.dim(static_cast<int>(axis));
    return packed;
}

__global__ void reduction_sum_keepdims_kernel(DeviceReductionProgram program) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.output.numel)
        return;

    int64_t remaining = linear;
    int64_t inputBase = program.input.storageOffset;
    for (int dim = program.output.rank - 1; dim >= 0; --dim) {
        int64_t coord = remaining % program.output.dims[dim];
        remaining /= program.output.dims[dim];
        if (dim != program.axis)
            inputBase += coord * program.input.strides[dim];
    }

    float sum = 0.0f;
    for (int64_t reduceIndex = 0; reduceIndex < program.reduceDim; ++reduceIndex) {
        int64_t inputStorage =
            inputBase + reduceIndex * program.input.strides[program.axis];
        sum += cuda_kernel::load_float_at_storage(program.input, inputStorage);
    }

    cuda_kernel::store_float(program.output, linear, sum);
}

} // namespace

Result<void> launch_cuda_sliding_query_key_score(
        const CudaLaunchContext& context,
        const CudaSlidingQueryKeyScoreProgram&) {
    auto valid = validate_context(context, 2, 1, "sliding_query_key_score");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("sliding_query_key_score");
}

Result<void> launch_cuda_reduction(
        const CudaLaunchContext& context,
        const CudaReductionProgram& program) {
    auto valid = validate_context(context, 1, 1, "reduction");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_reduction_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.output.numel == 0)
        return {};

    int64_t blockCount =
        (launchProgram.output.numel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize;
    if (blockCount > std::numeric_limits<int>::max())
        return make_error("cuda reduction block count exceeds grid limit");

    reduction_sum_keepdims_kernel<<<
        static_cast<int>(blockCount),
        cuda_kernel::kBlockSize,
        0,
        context.stream>>>(launchProgram);
    return cuda_check(cudaGetLastError(), "cuda reduction launch");
}

} // namespace sandy::device
