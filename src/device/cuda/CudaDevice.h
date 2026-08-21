#pragma once

#include "CudaKernels.h"
#include "CudaPagedTensor.h"
#include "Device.h"
#include "KernelIR.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sandy::device {

class CudaDevice final : public Device {
public:
    explicit CudaDevice(int cudaDevice = 0);
    ~CudaDevice() override;

    Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) override;

    Result<DeviceBufferId> alloc(core::TensorDesc desc) override;
    Result<void> dealloc(DeviceBufferId buffer) override;

    std::unique_ptr<DeviceScratchAllocator> createScratchAllocator() override;

    Result<DeviceBufferId> load(core::TensorBuffer& src) override;

    Result<DevicePagedPoolId> createPagedPool(DevicePagedPoolDesc desc) override;
    Result<void> destroyPagedPool(DevicePagedPoolId pool) override;
    Result<DevicePagedTensorId> allocPaged(
        DevicePagedPoolId pool,
        core::Shape logicalShape) override;
    Result<void> deallocPaged(DevicePagedTensorId tensor) override;
    Result<void> reservePaged(DevicePagedTensorId tensor, int64_t pageCount) override;
    Result<void> appendPaged(DevicePagedTensorId dst, core::TensorBuffer& denseChunk) override;
    Result<void> appendPaged(DevicePagedTensorId dst, DeviceTensorView denseChunk) override;
    Result<DevicePagedTensorMeta> pagedMeta(DevicePagedTensorId tensor) const override;

    Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceRunValue> inputs,
        std::span<const DeviceRunValue> outputs) override;

    Result<TensorBufferPtr> read(DeviceTensorView src) override;
    Result<TensorBufferPtr> read(DeviceBufferId src);

private:
    using KernelProgram = std::variant<
        std::monostate,
        CudaElementwiseProgram,
        CudaLayoutTransformProgram,
        CudaMatMulProgram,
        CudaMoeGatherProgram,
        CudaMoeMatMulProgram,
        CudaReductionProgram,
        CudaSoftmaxProgram,
        CudaTopKProgram,
        CudaNormProgram,
        CudaRoPEProgram,
        CudaSlidingQueryKeyScoreProgram,
        CudaAttentionProgram>;

    struct CudaDeviceBuffer {
        core::TensorDesc desc;
        void* data = nullptr;
        size_t bytes = 0;
    };

    struct CudaDeviceKernel {
        ir::kernel_ir::OpKind kind = ir::kernel_ir::OpKind::Input;
        size_t inputCount = 0;
        size_t outputCount = 0;
        KernelProgram program;
    };

    struct CudaDeviceGraph {
        std::unordered_map<ir::kernel_ir::OpId, CudaDeviceKernel> kernels;
    };

    Result<void> ensure_stream();
    Result<void> ensure_cublas_handle();
    Result<void> ensure_async_memory_pool();
    Result<void> set_device() const;
    Result<CudaDeviceBufferView> buffer_view(DeviceBufferId buffer, bool writable);
    Result<CudaDevicePagedTensorView> paged_tensor_view(DevicePagedTensorId tensor) const;
    Result<void> sync_paged_tensor_table(DevicePagedTensorId tensor);

    int cudaDevice_ = 0;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublasHandle_ = nullptr;
    cudaDeviceProp deviceProps_{};
    std::string initializationError_;
    bool asyncMemoryPoolConfigured_ = false;
    DeviceBufferId nextBufferId_ = 1;
    DeviceCompiledGraphId nextGraphId_ = 1;
    DevicePagedPoolId nextPagedPoolId_ = 1;
    DevicePagedTensorId nextPagedTensorId_ = 1;
    std::unordered_map<DeviceBufferId, CudaDeviceBuffer> buffers_;
    std::unordered_map<DeviceCompiledGraphId, CudaDeviceGraph> graphs_;
    std::unordered_map<DevicePagedPoolId, CudaPagedTensorPool> pagedPools_;
    std::unordered_map<DevicePagedTensorId, CudaPagedTensor> pagedTensors_;
};

} // namespace sandy::device
