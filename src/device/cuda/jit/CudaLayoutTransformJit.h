#pragma once

#include "CudaJit.h"
#include "Tensor.h"

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaLayoutTransformJit(
    int cudaDevice,
    CudaJitCache& cache,
    core::DType dtype);

} // namespace sandy::device
