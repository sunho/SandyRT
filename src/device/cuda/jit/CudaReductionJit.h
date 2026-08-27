#pragma once

#include "CudaJit.h"

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaReductionJit(
    int cudaDevice,
    CudaJitCache& cache);

} // namespace sandy::device
