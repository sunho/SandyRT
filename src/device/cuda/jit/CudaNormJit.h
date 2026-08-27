#pragma once

#include "CudaJit.h"
#include "KernelIR.h"
#include "Tensor.h"

#include <span>

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaNormJit(
    int cudaDevice,
    CudaJitCache& cache,
    ir::kernel_ir::NormKind norm,
    bool hasWeight,
    core::DType dtype,
    std::span<const int> inputAccesses,
    int outputAccess);

} // namespace sandy::device
