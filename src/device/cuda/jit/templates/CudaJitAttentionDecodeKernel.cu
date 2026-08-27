#include "CudaJitAttentionAbi.cuh"
#include "CudaJitAttentionDecodeConfig.cuh"
#include "CudaJitTensorAccess.cuh"

__device__ __forceinline__ float sandy_attention_negative_infinity() {
    return -__int_as_float(0x7f800000);
}

__device__ __forceinline__ int64_t sandy_attention_max_i64(
        int64_t lhs,
        int64_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int64_t sandy_attention_min_i64(
        int64_t lhs,
        int64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ float sandy_attention_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ int64_t sandy_attention_position(
        const SandyAttentionDecodeParams& params,
        int64_t batch) {
    if constexpr (!SandyAttentionHasPositionOffsets)
        return 0;
    int64_t linear = SandyAttentionRank == 4 ? batch : 0;
    if constexpr (SandyAttentionPositionDType == SANDY_JIT_I32) {
        return sandy_runtime_load_integer<
            int32_t,
            SandyAttentionPositionAccess>(params.positionOffsets, linear);
    } else {
        return sandy_runtime_load_integer<
            int64_t,
            SandyAttentionPositionAccess>(params.positionOffsets, linear);
    }
}

__device__ __forceinline__ int64_t sandy_attention_q_linear(
        const SandyAttentionDecodeParams& params,
        int64_t batch,
        int64_t qHead,
        int dim) {
    if constexpr (SandyAttentionRank == 4)
        return (batch * params.qHeads + qHead) * SandyAttentionHeadDim + dim;
    return qHead * SandyAttentionHeadDim + dim;
}

__device__ __forceinline__ void sandy_attention_decode_partial(
        SandyAttentionDecodeParams params) {
    constexpr int kLaneValues = (SandyAttentionHeadDim + 31) / 32;
    int lane = threadIdx.x & 31;
    int split = blockIdx.x;
    int64_t qHead = blockIdx.y;
    int64_t batch = blockIdx.z;
    int64_t kvHead = qHead / SandyAttentionQueryHeadsPerKv;
    int64_t prefix = SandyAttentionRank == 4
        ? batch * params.kvHeads + kvHead
        : kvHead;

    int64_t queryPosition = sandy_attention_position(params, batch);
    int64_t minKey = SandyAttentionWindow > 0
        ? sandy_attention_max_i64(
              0, queryPosition + 1 - SandyAttentionWindow)
        : 0;
    int64_t maxKey = queryPosition;
    int64_t splitStart = static_cast<int64_t>(split) * params.splitSize;
    int64_t splitEnd = sandy_attention_min_i64(
        params.tk, splitStart + params.splitSize);
    int64_t firstKey = sandy_attention_max_i64(splitStart, minKey);
    int64_t lastKey = sandy_attention_min_i64(splitEnd, maxKey + 1);

    float runningMax = sandy_attention_negative_infinity();
    float runningSum = 0.0f;
    float output[kLaneValues];
    float qValues[kLaneValues];
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        output[frag] = 0.0f;
        qValues[frag] = dim < SandyAttentionHeadDim
            ? sandy_runtime_load<
                  SandyAttentionDType,
                  SandyAttentionQAccess>(
                      params.q,
                      sandy_attention_q_linear(params, batch, qHead, dim))
            : 0.0f;
    }

    if (firstKey < lastKey) {
        int64_t firstPage = firstKey >> params.k.pageShift;
        int64_t lastPage = (lastKey - 1) >> params.k.pageShift;
        for (int64_t pageOrdinal = firstPage;
             pageOrdinal <= lastPage;
             ++pageOrdinal) {
            const auto* kPage = sandy_paged_page<SandyAttentionDType>(
                params.k, pageOrdinal);
            const auto* vPage = sandy_paged_page<SandyAttentionDType>(
                params.v, pageOrdinal);
            int64_t pageStart = pageOrdinal * params.k.pageSize;
            int64_t pageFirstKey = sandy_attention_max_i64(firstKey, pageStart);
            int64_t pageLastKey = sandy_attention_min_i64(
                lastKey, pageStart + params.k.pageSize);
            int64_t slot = pageFirstKey - pageStart;
            const auto* kRow = sandy_paged_row<SandyAttentionDType>(
                params.k, kPage, prefix, slot);
            const auto* vRow = sandy_paged_row<SandyAttentionDType>(
                params.v, vPage, prefix, slot);

            for (int64_t key = pageFirstKey;
                 key < pageLastKey;
                 ++key) {
                float partialDot = 0.0f;
                #pragma unroll
                for (int frag = 0; frag < kLaneValues; ++frag) {
                    int dim = lane + frag * 32;
                    if (dim < SandyAttentionHeadDim) {
                        partialDot += qValues[frag] *
                            sandy_typed_load<SandyAttentionDType>(kRow, dim);
                    }
                }
                float score = sandy_attention_warp_sum(partialDot) *
                    SandyAttentionScale;
                float nextMax = fmaxf(runningMax, score);
                float oldScale = isinf(runningMax) && runningMax < 0.0f
                    ? 0.0f
                    : expf(runningMax - nextMax);
                float newScale = expf(score - nextMax);
                #pragma unroll
                for (int frag = 0; frag < kLaneValues; ++frag) {
                    int dim = lane + frag * 32;
                    if (dim < SandyAttentionHeadDim) {
                        float vValue = sandy_typed_load<
                            SandyAttentionDType>(vRow, dim);
                        output[frag] = oldScale * output[frag] +
                            newScale * vValue;
                    }
                }
                runningSum = oldScale * runningSum + newScale;
                runningMax = nextMax;
                kRow += params.k.pagedInnerElements;
                vRow += params.v.pagedInnerElements;
            }
        }
    }

    int64_t stateIndex =
        ((batch * params.qHeads + qHead) * params.numSplits + split) *
        (SandyAttentionHeadDim + 2);
    if (lane == 0) {
        params.partial[stateIndex] = runningMax;
        params.partial[stateIndex + 1] = runningSum;
    }
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        if (dim < SandyAttentionHeadDim)
            params.partial[stateIndex + 2 + dim] = output[frag];
    }
}

__device__ __forceinline__ void sandy_attention_decode_reduce(
        SandyAttentionDecodeParams params) {
    constexpr int kLaneValues = (SandyAttentionHeadDim + 31) / 32;
    int lane = threadIdx.x & 31;
    int64_t qHead = blockIdx.x;
    int64_t batch = blockIdx.y;
    float runningMax = sandy_attention_negative_infinity();
    float runningSum = 0.0f;
    float output[kLaneValues];
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag)
        output[frag] = 0.0f;

    for (int split = 0; split < params.numSplits; ++split) {
        int64_t stateIndex =
            ((batch * params.qHeads + qHead) * params.numSplits + split) *
            (SandyAttentionHeadDim + 2);
        float splitMax = params.partial[stateIndex];
        float splitSum = params.partial[stateIndex + 1];
        if (splitSum == 0.0f || (isinf(splitMax) && splitMax < 0.0f))
            continue;
        float nextMax = fmaxf(runningMax, splitMax);
        float oldScale = isinf(runningMax) && runningMax < 0.0f
            ? 0.0f
            : expf(runningMax - nextMax);
        float splitScale = expf(splitMax - nextMax);
        #pragma unroll
        for (int frag = 0; frag < kLaneValues; ++frag) {
            int dim = lane + frag * 32;
            if (dim < SandyAttentionHeadDim) {
                float splitOutput = params.partial[
                    stateIndex + 2 + dim];
                output[frag] = oldScale * output[frag] +
                    splitScale * splitOutput;
            }
        }
        runningSum = oldScale * runningSum + splitScale * splitSum;
        runningMax = nextMax;
    }

    float inverse = runningSum == 0.0f ? 0.0f : 1.0f / runningSum;
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        if (dim < SandyAttentionHeadDim) {
            sandy_runtime_store<
                SandyAttentionDType,
                SandyAttentionOutputAccess>(
                    params.output,
                    sandy_attention_q_linear(params, batch, qHead, dim),
                    output[frag] * inverse);
        }
    }
}

extern "C" __global__ void sandy_jit_attention_decode(
        SandyAttentionDecodeParams params) {
    if (params.reduce)
        sandy_attention_decode_reduce(params);
    else
        sandy_attention_decode_partial(params);
}
