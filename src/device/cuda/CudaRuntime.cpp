#include "CudaRuntime.h"

#include <cstdint>
#include <limits>

namespace sandy::device {

Result<void> cuda_check(cudaError_t status, const std::string& context) {
    if (status == cudaSuccess)
        return {};
    return make_error(context + ": " + cudaGetErrorString(status));
}

Result<void> cuda_configure_default_memory_pool(int cudaDevice) {
#if CUDART_VERSION >= 11020
    cudaMemPool_t pool = nullptr;
    auto got = cuda_check(
        cudaDeviceGetDefaultMemPool(&pool, cudaDevice),
        "cudaDeviceGetDefaultMemPool");
    if (!got)
        return make_error(got.error());

    uint64_t threshold = std::numeric_limits<uint64_t>::max();
    return cuda_check(
        cudaMemPoolSetAttribute(
            pool,
            cudaMemPoolAttrReleaseThreshold,
            &threshold),
        "cudaMemPoolSetAttribute release threshold");
#else
    (void)cudaDevice;
    return {};
#endif
}

Result<void> cuda_malloc_stream_ordered(
        void** ptr,
        size_t bytes,
        cudaStream_t stream,
        const std::string& context) {
    *ptr = nullptr;
    if (bytes == 0)
        return {};
#if CUDART_VERSION >= 11020
    if (stream)
        return cuda_check(cudaMallocAsync(ptr, bytes, stream), context);
#endif
    return cuda_check(cudaMalloc(ptr, bytes), context);
}

Result<void> cuda_free_stream_ordered(
        void* ptr,
        cudaStream_t stream,
        const std::string& context) {
    if (!ptr)
        return {};
#if CUDART_VERSION >= 11020
    if (stream)
        return cuda_check(cudaFreeAsync(ptr, stream), context);
#endif
    return cuda_check(cudaFree(ptr), context);
}

} // namespace sandy::device
