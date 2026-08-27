#include "CudaReductionJit.h"

#include "CudaJitEmbeddedSources.h"

namespace sandy::device {

Result<CudaJitCache::KernelPtr> compileCudaReductionJit(
        int cudaDevice,
        CudaJitCache& cache) {
    CudaJitRequest request;
    request.sourceName = "CudaJitReductionKernel.cu";
    request.source = embeddedReductionKernelSource();
    request.headers = embeddedReductionHeaders();
    request.entryName = "sandy_jit_reduction_sum_keepdims";
    request.options = {"-lineinfo"};
    return cache.getOrCompile(cudaDevice, request);
}

} // namespace sandy::device
