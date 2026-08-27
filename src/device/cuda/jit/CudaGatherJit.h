#pragma once

#include "CudaJit.h"
#include "Tensor.h"

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaGatherJit(
    int cudaDevice,
    CudaJitCache& cache,
    core::DType idsDtype,
    core::DType valueDtype,
    int tableRank,
    int idsAccess,
    int tableAccess,
    int outputAccess);

} // namespace sandy::device
