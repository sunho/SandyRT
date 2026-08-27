#pragma once

#include "CudaJit.h"
#include "CudaKernels.h"

#include <string>
#include <span>

namespace sandy::device {

struct CudaElementwiseJitSource {
    std::string evaluatorSource;
    std::string entryName;
};

Result<CudaElementwiseJitSource> emitCudaElementwiseJitSource(
    const CudaElementwiseProgram& program);

Result<CudaJitCache::KernelPtr> compileCudaElementwiseJit(
    int cudaDevice,
    CudaJitCache& cache,
    const CudaElementwiseProgram& program,
    std::span<const int> inputAccesses,
    int outputAccess);

} // namespace sandy::device
