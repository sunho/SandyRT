#include "CudaLayoutTransformJit.h"

#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

namespace sandy::device {

namespace {

Result<std::string> layout_element_config(core::DType dtype) {
    const char* type = nullptr;
    switch (dtype) {
        case core::DType::U8:
            type = "uint8_t";
            break;
        case core::DType::F16:
        case core::DType::BF16:
            type = "uint16_t";
            break;
        case core::DType::F32:
        case core::DType::I32:
            type = "uint32_t";
            break;
        case core::DType::I64:
            type = "uint64_t";
            break;
    }
    if (!type)
        return make_error("CUDA layout transform JIT unsupported dtype");
    return std::string("#pragma once\nusing SandyLayoutElement = ") + type + ";\n";
}

} // namespace

Result<CudaJitCache::KernelPtr> compileCudaLayoutTransformJit(
        int cudaDevice,
        CudaJitCache& cache,
        core::DType dtype,
        int inputAccess,
        int outputAccess) {
    auto config = layout_element_config(dtype);
    if (!config)
        return make_error(config.error());
    auto inputPolicy = cudaJitAccessConstant(inputAccess);
    if (!inputPolicy)
        return make_error(inputPolicy.error());
    auto outputPolicy = cudaJitAccessConstant(outputAccess);
    if (!outputPolicy)
        return make_error(outputPolicy.error());
    *config += "inline constexpr int SandyLayoutInputAccess = " +
        std::string(*inputPolicy) + ";\n";
    *config += "inline constexpr int SandyLayoutOutputAccess = " +
        std::string(*outputPolicy) + ";\n";
    CudaJitRequest request;
    request.sourceName = "CudaJitLayoutTransformKernel.cu";
    request.source = embeddedLayoutTransformKernelSource();
    request.headers = embeddedLayoutTransformHeaders();
    request.headers.push_back({"CudaJitLayoutTransformConfig.cuh", config.take()});
    request.entryName = "sandy_jit_layout_transform";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
