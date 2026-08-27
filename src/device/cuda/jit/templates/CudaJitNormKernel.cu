#include "CudaJitNormAbi.cuh"
#include "CudaJitNormConfig.cuh"
#include "CudaJitTensorAccess.cuh"

__device__ __forceinline__ float sandy_norm_reduce_sum(
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

extern "C" __global__ void sandy_jit_norm(SandyNormParams params) {
    extern __shared__ float shared[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= params.rows)
        return;

    float mean = 0.0f;
    if constexpr (SandyNormIsLayer) {
        float localSum = 0.0f;
        for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
            int64_t linear = row * params.hidden + col;
            localSum += sandy_runtime_load<SandyNormDType, SandyNormInputAccess>(
                params.input, linear);
        }
        mean = sandy_norm_reduce_sum(localSum, shared) /
            static_cast<float>(params.hidden);
    }

    float localSquareSum = 0.0f;
    for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
        int64_t linear = row * params.hidden + col;
        float value = sandy_runtime_load<SandyNormDType, SandyNormInputAccess>(
            params.input, linear);
        if constexpr (SandyNormIsLayer)
            value -= mean;
        localSquareSum += value * value;
    }
    float squareSum = sandy_norm_reduce_sum(localSquareSum, shared);
    float inverse = rsqrtf(
        squareSum / static_cast<float>(params.hidden) + params.epsilon);

    for (int64_t col = tid; col < params.hidden; col += blockDim.x) {
        int64_t linear = row * params.hidden + col;
        float value = sandy_runtime_load<SandyNormDType, SandyNormInputAccess>(
            params.input, linear);
        if constexpr (SandyNormIsLayer)
            value -= mean;
        float scale = 1.0f;
        if constexpr (SandyNormHasWeight)
            scale = sandy_runtime_load<SandyNormDType, SandyNormWeightAccess>(
                params.weight, col);
        float result = value * inverse * scale;
        if constexpr (SandyNormIsLayer)
            result += sandy_runtime_load<SandyNormDType, SandyNormBiasAccess>(
                params.bias, col);
        sandy_runtime_store<SandyNormDType, SandyNormOutputAccess>(
            params.output, linear, result);
    }
}
