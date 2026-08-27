#pragma once

#include "CudaJit.h"
#include "Tensor.h"

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaReductionJit(
    int cudaDevice,
    CudaJitCache& cache,
    core::DType dtype,
    int inputAccess,
    int outputAccess);

} // namespace sandy::device
