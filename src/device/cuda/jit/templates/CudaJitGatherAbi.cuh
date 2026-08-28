#pragma once

#include "CudaJitAbi.cuh"

struct SandyGatherParams {
    SandyJitTensorArg ids;
    SandyJitTensorArg table;
    SandyJitTensorArg output;
    int64_t vocab;
    int64_t hidden;
    uint32_t* validationFailure;
    uint32_t op;
};
