#include "CudaJitReductionAbi.cuh"
#include "CudaJitReductionConfig.cuh"
#include "CudaJitTensorAccess.cuh"

extern "C" __global__ void sandy_jit_reduction_sum_keepdims(
        SandyReductionParams params) {
    int64_t outputLinear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (outputLinear >= params.output.numel)
        return;

    int64_t remaining = outputLinear;
    int64_t inputLinear = 0;
    int64_t logicalStride = 1;
    int64_t axisStride = 1;
    for (int dim = params.output.rank - 1; dim >= 0; --dim) {
        int64_t coordinate = remaining % params.output.dims[dim];
        remaining /= params.output.dims[dim];
        if (dim == params.axis)
            axisStride = logicalStride;
        else
            inputLinear += coordinate * logicalStride;
        logicalStride *= params.input.dims[dim];
    }

    float sum = 0.0f;
    for (int64_t index = 0; index < params.reduceDim; ++index)
        sum += sandy_runtime_load<
            SandyReductionDType,
            SandyReductionInputAccess>(
            params.input, inputLinear + index * axisStride);
    sandy_runtime_store<
        SandyReductionDType,
        SandyReductionOutputAccess>(params.output, outputLinear, sum);
}
