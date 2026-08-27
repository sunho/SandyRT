#include "CudaJit.h"

#include "CudaRuntime.h"

#include <nvrtc.h>

#include <string_view>
#include <utility>

namespace sandy::device {

namespace {

std::string driver_error(CUresult status, std::string_view context) {
    const char* name = nullptr;
    const char* message = nullptr;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &message);
    return std::string(context) + ": " + (name ? name : "CUDA_ERROR_UNKNOWN") +
           (message ? std::string(" (") + message + ")" : "");
}

Result<CudaJitCache::KernelPtr> compile_kernel(
        int cudaDevice,
        const CudaJitRequest& request) {
    auto selected = cuda_check(cudaSetDevice(cudaDevice), "cudaSetDevice CUDA JIT");
    if (!selected)
        return make_error(selected.error());
    auto initialized = cuda_check(cudaFree(nullptr), "initialize CUDA context for JIT");
    if (!initialized)
        return make_error(initialized.error());

    cudaDeviceProp props{};
    auto properties = cuda_check(
        cudaGetDeviceProperties(&props, cudaDevice),
        "cudaGetDeviceProperties CUDA JIT");
    if (!properties)
        return make_error(properties.error());

    std::vector<std::string> optionStorage = {
        "--std=c++17",
        "--gpu-architecture=compute_" +
            std::to_string(props.major) + std::to_string(props.minor),
    };
    optionStorage.insert(optionStorage.end(), request.options.begin(), request.options.end());
    std::vector<const char*> options;
    for (const auto& option : optionStorage)
        options.push_back(option.c_str());

    std::vector<const char*> headerSources;
    std::vector<const char*> headerNames;
    for (const auto& header : request.headers) {
        headerSources.push_back(header.source.c_str());
        headerNames.push_back(header.name.c_str());
    }

    nvrtcProgram program = nullptr;
    auto created = nvrtcCreateProgram(
        &program,
        request.source.c_str(),
        request.sourceName.c_str(),
        static_cast<int>(headerSources.size()),
        headerSources.data(),
        headerNames.data());
    if (created != NVRTC_SUCCESS)
        return make_error(std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(created));

    auto compiled = nvrtcCompileProgram(
        program,
        static_cast<int>(options.size()),
        options.data());
    size_t logBytes = 0;
    (void)nvrtcGetProgramLogSize(program, &logBytes);
    std::string log(logBytes, '\0');
    if (logBytes)
        (void)nvrtcGetProgramLog(program, log.data());
    if (compiled != NVRTC_SUCCESS) {
        std::string error = std::string("NVRTC compile failed: ") +
            nvrtcGetErrorString(compiled) + "\n" + log;
        (void)nvrtcDestroyProgram(&program);
        return make_error(std::move(error));
    }

    size_t ptxBytes = 0;
    auto sized = nvrtcGetPTXSize(program, &ptxBytes);
    if (sized != NVRTC_SUCCESS) {
        (void)nvrtcDestroyProgram(&program);
        return make_error(std::string("nvrtcGetPTXSize: ") + nvrtcGetErrorString(sized));
    }
    std::vector<char> ptx(ptxBytes);
    auto got = nvrtcGetPTX(program, ptx.data());
    (void)nvrtcDestroyProgram(&program);
    if (got != NVRTC_SUCCESS)
        return make_error(std::string("nvrtcGetPTX: ") + nvrtcGetErrorString(got));

    CUmodule rawModule = nullptr;
    auto loaded = cuModuleLoadData(&rawModule, ptx.data());
    if (loaded != CUDA_SUCCESS)
        return make_error(driver_error(loaded, "cuModuleLoadData"));
    auto module = std::make_shared<CudaJitModule>(rawModule);
    CUfunction function = nullptr;
    auto found = cuModuleGetFunction(&function, rawModule, request.entryName.c_str());
    if (found != CUDA_SUCCESS)
        return make_error(driver_error(found, "cuModuleGetFunction " + request.entryName));
    CudaJitCache::KernelPtr kernel = std::make_shared<CudaJitKernel>(
        std::move(module), function);
    return kernel;
}

} // namespace

CudaJitModule::~CudaJitModule() {
    if (module_)
        (void)cuModuleUnload(module_);
}

Result<void> CudaJitKernel::launch(
        dim3 grid,
        dim3 block,
        size_t sharedBytes,
        cudaStream_t stream,
        std::span<void*> arguments) const {
    auto launched = cuLaunchKernel(
        function_,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        static_cast<unsigned>(sharedBytes),
        reinterpret_cast<CUstream>(stream),
        arguments.data(),
        nullptr);
    if (launched != CUDA_SUCCESS)
        return make_error(driver_error(launched, "cuLaunchKernel"));
    return {};
}

Result<core::CacheKey> buildCudaJitCacheKey(
        int cudaDevice,
        const CudaJitRequest& request) {
    cudaDeviceProp props{};
    auto properties = cuda_check(
        cudaGetDeviceProperties(&props, cudaDevice),
        "cudaGetDeviceProperties CUDA JIT key");
    if (!properties)
        return make_error(properties.error());
    int nvrtcMajor = 0;
    int nvrtcMinor = 0;
    auto versioned = nvrtcVersion(&nvrtcMajor, &nvrtcMinor);
    if (versioned != NVRTC_SUCCESS)
        return make_error(std::string("nvrtcVersion: ") + nvrtcGetErrorString(versioned));

    core::CacheKeyBuilder key("cuda-jit-v1");
    key.addU32(request.abiVersion)
       .addU32(static_cast<uint32_t>(props.major))
       .addU32(static_cast<uint32_t>(props.minor))
       .addU32(static_cast<uint32_t>(nvrtcMajor))
       .addU32(static_cast<uint32_t>(nvrtcMinor))
       .addString(request.sourceName)
       .addString(request.source)
       .addString(request.entryName)
       .addU64(request.headers.size());
    for (const auto& header : request.headers)
        key.addString(header.name).addString(header.source);
    key.addU64(request.options.size());
    for (const auto& option : request.options)
        key.addString(option);
    return std::move(key).finish();
}

Result<CudaJitCache::KernelPtr> CudaJitCache::getOrCompile(
        int cudaDevice,
        const CudaJitRequest& request) {
    auto key = buildCudaJitCacheKey(cudaDevice, request);
    if (!key)
        return make_error(key.error());
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = kernels_.find(*key);
    if (found != kernels_.end()) {
        hits_++;
        return found->second;
    }
    auto kernel = compile_kernel(cudaDevice, request);
    if (!kernel)
        return make_error(kernel.error());
    kernels_.emplace(key.take(), *kernel);
    misses_++;
    return kernel.take();
}

CudaJitCacheStats CudaJitCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return CudaJitCacheStats{hits_, misses_, kernels_.size()};
}

} // namespace sandy::device

