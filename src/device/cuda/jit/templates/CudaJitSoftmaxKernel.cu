#include "CudaJitSoftmaxAbi.cuh"
#include "CudaJitSoftmaxConfig.cuh"
#include "CudaJitTensorAccess.cuh"

__device__ __forceinline__ float sandy_softmax_reduce_sum(
        float value,
        float* shared) {
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

__device__ __forceinline__ float sandy_softmax_reduce_max(
        float value,
        float* shared) {
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

extern "C" __global__ void sandy_jit_softmax(SandySoftmaxParams params) {
    extern __shared__ float shared[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= params.rows)
        return;

    const float negativeInfinity = -__int_as_float(0x7f800000);
    float localMax = negativeInfinity;
    for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
        int64_t linear = row * params.hidden + col;
        localMax = fmaxf(
            localMax,
            sandy_runtime_load<SandySoftmaxDType, SandySoftmaxInputAccess>(
                params.input, linear));
    }
    float maxValue = sandy_softmax_reduce_max(localMax, shared);
    if (maxValue == negativeInfinity) {
        for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
            int64_t linear = row * params.hidden + col;
            sandy_runtime_store<SandySoftmaxDType, SandySoftmaxOutputAccess>(
                params.output, linear, 0.0f);
        }
        return;
    }

    float localSum = 0.0f;
    for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
        int64_t linear = row * params.hidden + col;
        localSum += expf(
            sandy_runtime_load<SandySoftmaxDType, SandySoftmaxInputAccess>(
                params.input, linear) - maxValue);
    }
    float invSum = 1.0f / sandy_softmax_reduce_sum(localSum, shared);
    for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
        int64_t linear = row * params.hidden + col;
        float value = expf(
            sandy_runtime_load<SandySoftmaxDType, SandySoftmaxInputAccess>(
                params.input, linear) - maxValue) * invSum;
        sandy_runtime_store<SandySoftmaxDType, SandySoftmaxOutputAccess>(
            params.output, linear, value);
    }
}
