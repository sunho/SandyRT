#include "CudaNormJit.h"

#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

#include <sstream>
#include <utility>

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaNormJit(
        int cudaDevice,
        CudaJitCache& cache,
        ir::kernel_ir::NormKind norm,
        bool hasWeight,
        core::DType dtype,
        std::span<const int> inputAccesses,
        int outputAccess) {
    size_t expectedInputs = norm == ir::kernel_ir::NormKind::LayerNorm
        ? 3u
        : (hasWeight ? 2u : 1u);
    if (inputAccesses.size() != expectedInputs)
        return make_error("CUDA norm JIT access policy count mismatch");

    std::ostringstream config;
    config << "#pragma once\ninline constexpr int SandyNormDType = ";
    switch (dtype) {
        case core::DType::F32:
            config << "SANDY_JIT_F32;\n";
            break;
        case core::DType::BF16:
            config << "SANDY_JIT_BF16;\n";
            break;
        default:
            return make_error("CUDA norm JIT unsupported dtype");
    }
    config << "inline constexpr bool SandyNormIsLayer = "
           << (norm == ir::kernel_ir::NormKind::LayerNorm ? "true" : "false")
           << ";\ninline constexpr bool SandyNormHasWeight = "
           << (hasWeight ? "true" : "false") << ";\n";
    const char* names[] = {
        "SandyNormInputAccess",
        "SandyNormWeightAccess",
        "SandyNormBiasAccess",
    };
    for (size_t i = 0; i < 3; ++i) {
        int raw = i < inputAccesses.size()
            ? inputAccesses[i]
            : SANDY_JIT_CONTIGUOUS;
        auto access = cudaJitAccessConstant(raw);
        if (!access)
            return make_error(access.error());
        config << "inline constexpr int " << names[i] << " = "
               << *access << ";\n";
    }
    auto outputPolicy = cudaJitAccessConstant(outputAccess);
    if (!outputPolicy)
        return make_error(outputPolicy.error());
    config << "inline constexpr int SandyNormOutputAccess = "
           << *outputPolicy << ";\n";

    CudaJitRequest request;
    request.sourceName = "CudaJitNormKernel.cu";
    request.source = embeddedNormKernelSource();
    request.headers = embeddedNormHeaders();
    request.headers.push_back({"CudaJitNormConfig.cuh", config.str()});
    request.entryName = "sandy_jit_norm";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
