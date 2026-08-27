#pragma once

// Editor-visible implementation of the generated interface. NVRTC receives a
// virtual header with the same name for each real scalar DAG.
struct GeneratedElementwiseEvaluator {
    static constexpr int kOutputDType = SANDY_JIT_F32;

    template <typename Loader>
    __device__ __forceinline__ static float eval(const Loader& loader) {
        return loader.template load<SANDY_JIT_F32>(0);
    }
};
