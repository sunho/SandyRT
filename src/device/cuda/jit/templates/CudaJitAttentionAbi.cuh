#pragma once

#include "CudaJitAbi.cuh"

struct SandyAttentionDecodeParams {
    SandyJitTensorArg q;
    SandyJitTensorArg k;
    SandyJitTensorArg v;
    SandyJitTensorArg output;
    SandyJitTensorArg positionOffsets;
    float* partial;
    int64_t batch;
    int64_t qHeads;
    int64_t kvHeads;
    int64_t tk;
    int64_t window;
    float scale;
    int32_t splitSize;
    int32_t numSplits;
};
