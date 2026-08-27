#include "CudaSoftmaxJit.h"

#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

#include <utility>

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaSoftmaxJit(
        int cudaDevice,
        CudaJitCache& cache,
        core::DType dtype,
        int inputAccess,
        int outputAccess) {
    std::string config = "#pragma once\ninline constexpr int SandySoftmaxDType = ";
    switch (dtype) {
        case core::DType::F32:
            config += "SANDY_JIT_F32;\n";
            break;
        case core::DType::BF16:
            config += "SANDY_JIT_BF16;\n";
            break;
        default:
            return make_error("CUDA softmax JIT unsupported dtype");
    }
    auto inputPolicy = cudaJitAccessConstant(inputAccess);
    if (!inputPolicy)
        return make_error(inputPolicy.error());
    auto outputPolicy = cudaJitAccessConstant(outputAccess);
    if (!outputPolicy)
        return make_error(outputPolicy.error());
    config += "inline constexpr int SandySoftmaxInputAccess = " +
        std::string(*inputPolicy) + ";\n";
    config += "inline constexpr int SandySoftmaxOutputAccess = " +
        std::string(*outputPolicy) + ";\n";

    CudaJitRequest request;
    request.sourceName = "CudaJitSoftmaxKernel.cu";
    request.source = embeddedSoftmaxKernelSource();
    request.headers = embeddedSoftmaxHeaders();
    request.headers.push_back({"CudaJitSoftmaxConfig.cuh", std::move(config)});
    request.entryName = "sandy_jit_softmax";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
