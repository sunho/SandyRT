#include "CudaReductionJit.h"

#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

#include <utility>

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaReductionJit(
        int cudaDevice,
        CudaJitCache& cache,
        core::DType dtype,
        int inputAccess,
        int outputAccess) {
    std::string config = "#pragma once\nconstexpr int SandyReductionDType = ";
    switch (dtype) {
        case core::DType::F32:
            config += "SANDY_JIT_F32;\n";
            break;
        case core::DType::BF16:
            config += "SANDY_JIT_BF16;\n";
            break;
        default:
            return make_error("CUDA reduction JIT unsupported dtype");
    }
    auto inputPolicy = cudaJitAccessConstant(inputAccess);
    if (!inputPolicy)
        return make_error(inputPolicy.error());
    auto outputPolicy = cudaJitAccessConstant(outputAccess);
    if (!outputPolicy)
        return make_error(outputPolicy.error());
    config += "constexpr int SandyReductionInputAccess = " +
        std::string(*inputPolicy) + ";\n";
    config += "constexpr int SandyReductionOutputAccess = " +
        std::string(*outputPolicy) + ";\n";
    CudaJitRequest request;
    request.sourceName = "CudaJitReductionKernel.cu";
    request.source = embeddedReductionKernelSource();
    request.headers = embeddedReductionHeaders();
    request.headers.push_back({"CudaJitReductionConfig.cuh", std::move(config)});
    request.entryName = "sandy_jit_reduction_sum_keepdims";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
