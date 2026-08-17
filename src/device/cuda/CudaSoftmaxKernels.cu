#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <cmath>
#include <limits>
#include <string>

namespace sandy::device {

namespace {

struct DeviceSoftmaxProgram {
    cuda_kernel::TensorArg x;
    cuda_kernel::TensorArg output;
    int64_t rows = 0;
    int64_t hidden = 0;
};

Result<void> validate_softmax_tensor_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda softmax ") + name + " unsupported dtype");
    return {};
}

Result<DeviceSoftmaxProgram> pack_softmax_program(
        const CudaLaunchContext& context,
        const CudaSoftmaxProgram& program) {
    const auto& xView = context.inputs[0];
    const auto& outputView = context.outputs[0];

    auto xDtype = validate_softmax_tensor_dtype(xView, "input");
    if (!xDtype)
        return make_error(xDtype.error());
    auto outputDtype = validate_softmax_tensor_dtype(outputView, "output");
    if (!outputDtype)
        return make_error(outputDtype.error());
    if (xView.view.desc.dtype != outputView.view.desc.dtype)
        return make_error("cuda softmax output dtype mismatch");
    if (xView.view.desc.shape != outputView.view.desc.shape)
        return make_error("cuda softmax output shape mismatch");

    int rank = xView.view.desc.shape.rank();
    if (rank < 1)
        return make_error("cuda softmax input must have rank >= 1");
    if (program.axis < -rank || program.axis >= rank)
        return make_error("cuda softmax axis out of range");
    int64_t axis = program.axis < 0 ? program.axis + rank : program.axis;
    if (axis != rank - 1)
        return make_error("cuda softmax only supports last dimension");

    int64_t total = xView.view.desc.shape.numel();
    if (total < 0)
        return make_error("cuda softmax input must have static shape");

    int64_t hidden = xView.view.desc.shape.dim(rank - 1);
    if (hidden < 0)
        return make_error("cuda softmax hidden dimension must be static");

    DeviceSoftmaxProgram packed;
    packed.hidden = hidden;
    packed.rows = hidden == 0 ? 0 : total / hidden;

    auto x = cuda_kernel::pack_tensor_arg(xView);
    if (!x)
        return make_error(x.error());
    auto output = cuda_kernel::pack_tensor_arg(outputView);
    if (!output)
        return make_error(output.error());
    packed.x = x.take();
    packed.output = output.take();

    return packed;
}

__device__ float block_reduce_sum(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    return shared[0];
}

__device__ float block_reduce_max(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }

    return shared[0];
}

__global__ void softmax_last_dim_kernel(DeviceSoftmaxProgram program) {
    extern __shared__ float shared[];

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= program.rows)
        return;

    float localMax = -INFINITY;
    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        localMax = fmaxf(localMax, cuda_kernel::load_float(program.x, linear));
    }

    float maxValue = block_reduce_max(localMax, shared);
    if (isinf(maxValue) && maxValue < 0.0f) {
        for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
            int64_t linear = row * program.hidden + col;
            cuda_kernel::store_float(program.output, linear, 0.0f);
        }
        return;
    }

    float localSum = 0.0f;
    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        localSum += expf(cuda_kernel::load_float(program.x, linear) - maxValue);
    }

    float sum = block_reduce_sum(localSum, shared);
    float invSum = 1.0f / sum;
    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        float value = expf(cuda_kernel::load_float(program.x, linear) - maxValue) * invSum;
        cuda_kernel::store_float(program.output, linear, value);
    }
}

} // namespace

Result<void> launch_cuda_softmax(
        const CudaLaunchContext& context,
        const CudaSoftmaxProgram& program) {
    auto valid = validate_context(context, 1, 1, "softmax");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_softmax_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.rows == 0)
        return {};
    if (launchProgram.rows > std::numeric_limits<int>::max())
        return make_error("cuda softmax row count exceeds grid limit");

    int blocks = static_cast<int>(launchProgram.rows);
    int threads = cuda_kernel::kBlockSize;
    size_t sharedBytes = static_cast<size_t>(threads) * sizeof(float);
    softmax_last_dim_kernel<<<blocks, threads, sharedBytes, context.stream>>>(launchProgram);
    return cuda_check(cudaGetLastError(), "cuda softmax launch");
}

} // namespace sandy::device
