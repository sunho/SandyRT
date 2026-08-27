#pragma once

#include "CudaJitAbi.cuh"

struct SandySoftmaxParams {
    SandyJitTensorArg input;
    SandyJitTensorArg output;
    int64_t rows;
    int64_t hidden;
};
