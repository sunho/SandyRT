#include "CudaAttentionJit.h"

#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

#include <iomanip>
#include <limits>
#include <sstream>

namespace sandy::device {

namespace {

Result<std::string_view> dtype_constant(core::DType dtype) {
    switch (dtype) {
        case core::DType::F32:
            return std::string_view("SANDY_JIT_F32");
        case core::DType::BF16:
            return std::string_view("SANDY_JIT_BF16");
        default:
            return make_error("CUDA attention decode JIT unsupported dtype");
    }
}

Result<std::string_view> position_dtype_constant(core::DType dtype) {
    switch (dtype) {
        case core::DType::I32:
            return std::string_view("SANDY_JIT_I32");
        case core::DType::I64:
            return std::string_view("SANDY_JIT_I64");
        default:
            return make_error("CUDA attention decode JIT unsupported position dtype");
    }
}

} // namespace

Result<CudaJitCache::KernelPtr> compileCudaAttentionDecodeJit(
        int cudaDevice,
        CudaJitCache& cache,
        core::DType dtype,
        int qAccess,
        int outputAccess,
        int rank,
        int64_t headDim,
        int64_t queryHeadsPerKv,
        bool hasPositionOffsets,
        core::DType positionDtype,
        int positionAccess,
        int64_t window,
        float scale) {
    auto dtypeName = dtype_constant(dtype);
    if (!dtypeName)
        return make_error(dtypeName.error());
    auto positionDtypeName = position_dtype_constant(positionDtype);
    if (!positionDtypeName)
        return make_error(positionDtypeName.error());
    auto qPolicy = cudaJitAccessConstant(qAccess);
    if (!qPolicy)
        return make_error(qPolicy.error());
    auto outputPolicy = cudaJitAccessConstant(outputAccess);
    if (!outputPolicy)
        return make_error(outputPolicy.error());
    auto positionPolicy = cudaJitAccessConstant(positionAccess);
    if (!positionPolicy)
        return make_error(positionPolicy.error());
    if (rank != 3 && rank != 4)
        return make_error("CUDA attention decode JIT rank must be 3 or 4");
    if (headDim <= 0 || queryHeadsPerKv <= 0)
        return make_error("CUDA attention decode JIT invalid static dimensions");

    std::ostringstream config;
    config << "#pragma once\n"
           << "inline constexpr int SandyAttentionDType = "
           << *dtypeName << ";\n"
           << "inline constexpr int SandyAttentionQAccess = "
           << *qPolicy << ";\n"
           << "inline constexpr int SandyAttentionOutputAccess = "
           << *outputPolicy << ";\n"
           << "inline constexpr int SandyAttentionPositionDType = "
           << *positionDtypeName << ";\n"
           << "inline constexpr int SandyAttentionPositionAccess = "
           << *positionPolicy << ";\n"
           << "inline constexpr int SandyAttentionHeadDim = "
           << headDim << ";\n"
           << "inline constexpr int SandyAttentionRank = " << rank << ";\n"
           << "inline constexpr int SandyAttentionQueryHeadsPerKv = "
           << queryHeadsPerKv << ";\n"
           << "inline constexpr bool SandyAttentionHasPositionOffsets = "
           << (hasPositionOffsets ? "true" : "false") << ";\n"
           << "inline constexpr int64_t SandyAttentionWindow = "
           << window << ";\n"
           << std::scientific
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "inline constexpr float SandyAttentionScale = "
           << scale << "f;\n";

    CudaJitRequest request;
    request.sourceName = "CudaJitAttentionDecodeKernel.cu";
    request.source = embeddedAttentionDecodeKernelSource();
    request.headers = embeddedAttentionDecodeHeaders();
    for (auto& header : request.headers) {
        if (header.name == "CudaJitAttentionDecodeConfig.cuh")
            header.source = config.str();
    }
    request.entryName = "sandy_jit_attention_decode";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
