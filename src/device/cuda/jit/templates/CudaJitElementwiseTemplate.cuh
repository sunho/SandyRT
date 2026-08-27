#pragma once
#include "CudaJitAbi.cuh"
#include "CudaJitTensorAccess.cuh"
#include "generated/ElementwiseEvaluator.cuh"

template <typename Loader, typename Evaluator, typename Storer>
__device__ __forceinline__ void sandy_run_elementwise(
        const SandyElementwiseParams& params,
        int64_t linear) {
    Loader loader{params, linear};
    float value = Evaluator::eval(loader);
    Storer::template store<Evaluator::kOutputDType>(
        params.output, linear, value);
}
