#pragma once

#include "CudaJitAbi.cuh"

struct SandyNormParams {
    SandyJitTensorArg input;
    SandyJitTensorArg weight;
    SandyJitTensorArg bias;
    SandyJitTensorArg output;
    int64_t rows;
    int64_t hidden;
    float epsilon;
};
