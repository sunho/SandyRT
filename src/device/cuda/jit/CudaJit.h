#pragma once

#include "CacheKey.h"
#include "Result.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::device {

struct CudaJitHeader {
    std::string name;
    std::string source;
};

struct CudaJitRequest {
    std::string sourceName;
    std::string source;
    std::vector<CudaJitHeader> headers;
    std::string entryName;
    std::vector<std::string> options;
    uint32_t abiVersion = 1;
};

class CudaJitModule {
public:
    explicit CudaJitModule(CUmodule module) : module_(module) {}
    ~CudaJitModule();
    CudaJitModule(const CudaJitModule&) = delete;
    CudaJitModule& operator=(const CudaJitModule&) = delete;

private:
    CUmodule module_ = nullptr;
};

class CudaJitKernel {
public:
    CudaJitKernel(std::shared_ptr<CudaJitModule> module, CUfunction function)
        : module_(std::move(module)), function_(function) {}

    Result<void> launch(
        dim3 grid,
        dim3 block,
        size_t sharedBytes,
        cudaStream_t stream,
        std::span<void*> arguments) const;

private:
    std::shared_ptr<CudaJitModule> module_;
    CUfunction function_ = nullptr;
};

struct CudaJitCacheStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t entries = 0;
    double compileMilliseconds = 0.0;
};

Result<core::CacheKey> buildCudaJitCacheKey(int cudaDevice, const CudaJitRequest& request);

class CudaJitCache {
public:
    using KernelPtr = std::shared_ptr<const CudaJitKernel>;

    Result<KernelPtr> getOrCompile(int cudaDevice, const CudaJitRequest& request);
    CudaJitCacheStats stats() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<core::CacheKey, KernelPtr, core::CacheKeyHash> kernels_;
    size_t hits_ = 0;
    size_t misses_ = 0;
    double compileMilliseconds_ = 0.0;
};

} // namespace sandy::device
