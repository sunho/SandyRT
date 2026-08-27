#pragma once

#include "CudaJitAbi.cuh"

struct SandyReductionParams {
    SandyJitTensorArg input;
    SandyJitTensorArg output;
    int64_t axis;
    int64_t reduceDim;
};
