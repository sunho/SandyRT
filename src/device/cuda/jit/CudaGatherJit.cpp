#include "CudaGatherJit.h"

#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"

#include <utility>

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaGatherJit(
        int cudaDevice,
        CudaJitCache& cache,
        core::DType idsDtype,
        core::DType valueDtype,
        int tableRank,
        int idsAccess,
        int tableAccess,
        int outputAccess) {
    std::string config = "#pragma once\nusing SandyGatherIndex = ";
    switch (idsDtype) {
        case core::DType::I32:
            config += "int32_t;\n";
            break;
        case core::DType::I64:
            config += "int64_t;\n";
            break;
        default:
            return make_error("CUDA gather JIT ids must be i32 or i64");
    }
    config += "constexpr int SandyGatherValueDType = ";
    switch (valueDtype) {
        case core::DType::F32:
            config += "SANDY_JIT_F32;\n";
            break;
        case core::DType::BF16:
            config += "SANDY_JIT_BF16;\n";
            break;
        default:
            return make_error("CUDA gather JIT unsupported value dtype");
    }
    if (tableRank != 1 && tableRank != 2)
        return make_error("CUDA gather JIT table rank must be one or two");
    config += "constexpr int SandyGatherTableRank = " +
        std::to_string(tableRank) + ";\n";
    const int accesses[] = {idsAccess, tableAccess, outputAccess};
    const char* names[] = {
        "SandyGatherIdsAccess",
        "SandyGatherTableAccess",
        "SandyGatherOutputAccess",
    };
    for (size_t i = 0; i < 3; ++i) {
        auto policy = cudaJitAccessConstant(accesses[i]);
        if (!policy)
            return make_error(policy.error());
        config += std::string("inline constexpr int ") + names[i] + " = " +
            std::string(*policy) + ";\n";
    }

    CudaJitRequest request;
    request.sourceName = "CudaJitGatherKernel.cu";
    request.source = embeddedGatherKernelSource();
    request.headers = embeddedGatherHeaders();
    request.headers.push_back({"CudaJitGatherConfig.cuh", std::move(config)});
    request.entryName = "sandy_jit_gather";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
