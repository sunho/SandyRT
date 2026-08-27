#pragma once

#include "CudaJit.h"
#include "Tensor.h"

namespace sandy::device {

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
    float scale);

} // namespace sandy::device
