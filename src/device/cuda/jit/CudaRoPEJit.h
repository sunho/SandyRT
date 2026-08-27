#pragma once
#include "CudaJit.h"
#include "Tensor.h"
namespace sandy::device {
Result<CudaJitCache::KernelPtr> compileCudaRoPEJit(
    int, CudaJitCache&, core::DType, bool, bool, core::DType,
    int, int, int);
}
