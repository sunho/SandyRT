#pragma once

#include "CudaJit.h"
#include "Result.h"
#include "templates/CudaJitAbi.cuh"

#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sandy::device {

inline Result<std::string_view> cudaJitAccessConstant(int access) {
    switch (access) {
        case SANDY_JIT_CONTIGUOUS:
            return std::string_view("SANDY_JIT_CONTIGUOUS");
        case SANDY_JIT_STRIDED:
            return std::string_view("SANDY_JIT_STRIDED");
        case SANDY_JIT_PAGED:
            return std::string_view("SANDY_JIT_PAGED");
        default:
            return make_error("invalid CUDA JIT tensor access policy");
    }
}

inline std::string cudaJitAccessKey(std::span<const int> accesses) {
    std::string key;
    key.reserve(accesses.size());
    for (int access : accesses)
        key.push_back(static_cast<char>('0' + access));
    return key;
}

class CudaJitVariants {
public:
    using Compiler = std::function<Result<CudaJitCache::KernelPtr>()>;

    Result<CudaJitCache::KernelPtr> getOrCompile(
            std::string key,
            const Compiler& compiler) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = kernels_.find(key);
        if (found != kernels_.end())
            return found->second;
        auto compiled = compiler();
        if (!compiled)
            return make_error(compiled.error());
        auto kernel = compiled.take();
        kernels_.emplace(std::move(key), kernel);
        return kernel;
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, CudaJitCache::KernelPtr> kernels_;
};

} // namespace sandy::device
