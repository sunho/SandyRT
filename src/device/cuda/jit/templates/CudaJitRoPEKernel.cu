#include "CudaJitRoPEAbi.cuh"
#include "CudaJitRoPEConfig.cuh"
#include "CudaJitTensorAccess.cuh"

__device__ __forceinline__ int64_t sandy_rope_position(
        const SandyRoPEParams& p, int64_t vector) {
    int64_t seqPosition = vector % p.seq;
    if constexpr (!SandyRoPEHasPositions)
        return seqPosition;
    int64_t index = p.positionCount == 1 ? 0 :
        (p.positionCount == p.seq ? seqPosition : vector);
    int64_t position = sandy_runtime_load_integer<
        SandyRoPEPositionType, SandyRoPEPositionAccess>(p.positions, index);
    if (p.positionCount == 1)
        position += seqPosition;
    if (position < 0 && p.validationFailure)
        atomicCAS(p.validationFailure, 0xffffffffu, p.op);
    return position;
}

__device__ __forceinline__ void sandy_rope_store_pair(
        const SandyRoPEParams& p, int64_t vector,
        int64_t firstCol, int64_t secondCol, float c, float s) {
    int64_t firstLinear = vector * p.dim + firstCol;
    int64_t secondLinear = vector * p.dim + secondCol;
    float first = sandy_runtime_load<SandyRoPEDType, SandyRoPEInputAccess>(
        p.input, firstLinear);
    float second = sandy_runtime_load<SandyRoPEDType, SandyRoPEInputAccess>(
        p.input, secondLinear);
    sandy_runtime_store<SandyRoPEDType, SandyRoPEOutputAccess>(
        p.output, firstLinear, first * c - second * s);
    sandy_runtime_store<SandyRoPEDType, SandyRoPEOutputAccess>(
        p.output, secondLinear, first * s + second * c);
}

extern "C" __global__ void sandy_jit_rope(SandyRoPEParams p) {
    int64_t rotatedPairs = p.rotaryDim / 2;
    int64_t skippedPairs = (p.dim - p.rotaryDim) / 2;
    int64_t workPerVector = rotatedPairs + skippedPairs;
    int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (work >= p.vectors * workPerVector)
        return;
    int64_t vector = work / workPerVector;
    int64_t offset = work % workPerVector;
    if (offset >= rotatedPairs) {
        int64_t skipped = offset - rotatedPairs;
        int64_t first = p.rotaryDim + 2 * skipped;
        int64_t second = first + 1;
        if constexpr (SandyRoPESplitHalf) {
            first = rotatedPairs + skipped;
            second = p.dim / 2 + rotatedPairs + skipped;
        }
        int64_t firstLinear = vector * p.dim + first;
        int64_t secondLinear = vector * p.dim + second;
        sandy_runtime_store<SandyRoPEDType, SandyRoPEOutputAccess>(
            p.output, firstLinear,
            sandy_runtime_load<SandyRoPEDType, SandyRoPEInputAccess>(p.input, firstLinear));
        sandy_runtime_store<SandyRoPEDType, SandyRoPEOutputAccess>(
            p.output, secondLinear,
            sandy_runtime_load<SandyRoPEDType, SandyRoPEInputAccess>(p.input, secondLinear));
        return;
    }
    int64_t position = sandy_rope_position(p, vector);
    if (position < 0)
        return;
    int64_t first = 2 * offset;
    int64_t second = first + 1;
    float denominator = static_cast<float>(p.rotaryDim);
    if constexpr (SandyRoPESplitHalf) {
        first = offset;
        second = p.dim / 2 + offset;
        denominator = static_cast<float>(p.dim);
    }
    float exponent = static_cast<float>(2 * offset) / denominator;
    float angle = static_cast<float>(position) / powf(p.theta, exponent);
    sandy_rope_store_pair(p, vector, first, second, cosf(angle), sinf(angle));
}
