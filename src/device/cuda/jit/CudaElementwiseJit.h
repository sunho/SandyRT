#pragma once

#include "CudaJit.h"
#include "CudaKernels.h"

#include <string>

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
    const CudaElementwiseProgram& program);

} // namespace sandy::device
