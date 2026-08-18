#pragma once

#include "Result.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace sandy::device {

Result<void> cuda_check(cudaError_t status, const std::string& context);
Result<void> cuda_configure_default_memory_pool(int cudaDevice);
Result<void> cuda_malloc_stream_ordered(
    void** ptr,
    size_t bytes,
    cudaStream_t stream,
    const std::string& context);

template <typename T>
Result<void> cuda_malloc_stream_ordered(
        T** ptr,
        size_t bytes,
        cudaStream_t stream,
        const std::string& context) {
    void* raw = nullptr;
    auto allocated = cuda_malloc_stream_ordered(&raw, bytes, stream, context);
    if (!allocated) {
        *ptr = nullptr;
        return make_error(allocated.error());
    }
    *ptr = static_cast<T*>(raw);
    return {};
}

Result<void> cuda_free_stream_ordered(
    void* ptr,
    cudaStream_t stream,
    const std::string& context);

} // namespace sandy::device
