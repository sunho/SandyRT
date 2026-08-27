#include "CudaJitElementwiseTemplate.cuh"

#ifndef SANDY_JIT_ENTRY_NAME
#define SANDY_JIT_ENTRY_NAME sandy_jit_elementwise
#endif

extern "C" __global__ void SANDY_JIT_ENTRY_NAME(SandyElementwiseParams params) {
    int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= params.output.numel)
        return;
    sandy_run_elementwise<
        SandyRuntimeLoader,
        GeneratedElementwiseEvaluator,
        SandyRuntimeStorer>(params, linear);
}

