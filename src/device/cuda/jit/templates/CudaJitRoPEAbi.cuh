#pragma once
#include "CudaJitAbi.cuh"
struct SandyRoPEParams {
    SandyJitTensorArg input;
    SandyJitTensorArg positions;
    SandyJitTensorArg output;
    int64_t seq, dim, rotaryDim, vectors, positionCount;
    float theta;
    uint32_t* validationFailure;
    uint32_t op;
};
