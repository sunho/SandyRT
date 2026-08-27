#pragma once

#include "CudaJitAbi.cuh"

struct SandyLayoutTransformParams {
    SandyJitTensorArg input;
    SandyJitTensorArg output;
};
