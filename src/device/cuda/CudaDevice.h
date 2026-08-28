#pragma once

#include "CudaKernels.h"
#include "CudaPagedTensor.h"
#include "Device.h"
#include "KernelIR.h"

#include <absl/container/flat_hash_map.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace sandy::device {

struct CudaGraphCacheStats {
    size_t captures = 0;
    size_t replays = 0;
    size_t eagerRegions = 0;
    size_t captureFailures = 0;
};

class CudaDevice final : public Device {
public:
    explicit CudaDevice(int cudaDevice = 0);
    ~CudaDevice() override;

    Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) override;

    Result<DeviceBufferId> alloc(core::TensorDesc desc) override;
    Result<void> dealloc(DeviceBufferId buffer) override;

    std::unique_ptr<DeviceScratchAllocator> createScratchAllocator() override;

    Result<DeviceBufferId> load(core::TensorBuffer& src) override;

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

    CudaJitCacheStats jitCacheStats() const { return jitCache_.stats(); }
    CudaGraphCacheStats graphCacheStats() const { return graphCacheStats_; }

private:
    Result<void> beginExecutableRun() override;
    Result<void> endExecutableRun() override;
    void abortExecutableRun() override;
    Result<void> executeCommands(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands,
        const std::function<void(ir::kernel_ir::OpId, double)>& profileKernel) override;
    Result<DeviceCompiledGraphId> compileExecutableGraph(
        const ir::kernel_ir::Graph& graph,
        std::span<const ir::kernel_ir::OpId> ops) override;
    Result<void> destroyCompiledGraph(DeviceCompiledGraphId graph) override;
    Result<DevicePagedPoolId> createPagedPoolImpl(DevicePagedPoolDesc desc) override;

    using KernelProgram = std::variant<
        std::monostate,
        CudaElementwiseProgram,
        CudaLayoutTransformProgram,
        CudaGatherProgram,
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

    struct CudaGraphTensorBinding {
        DeviceBufferId buffer = 0;
        core::DType dtype = core::DType::F32;
        std::vector<int64_t> shape;
        std::vector<int64_t> strides;
        int64_t storageOffset = 0;

        bool operator==(const CudaGraphTensorBinding&) const = default;
    };

    struct CudaGraphRegionKey {
        std::vector<ir::kernel_ir::OpId> ops;
        std::vector<CudaGraphTensorBinding> bindings;

        bool operator==(const CudaGraphRegionKey&) const = default;
    };

    struct CudaCapturedRegion {
        CudaGraphRegionKey key;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        bool captureDisabled = false;
    };

    struct CudaDeviceGraph {
        absl::flat_hash_map<ir::kernel_ir::OpId, CudaDeviceKernel> kernels;
        std::vector<CudaCapturedRegion> capturedRegions;
    };

    Result<void> ensure_stream();
    Result<void> ensure_cublas_handle();
    Result<void> ensure_async_memory_pool();
    Result<void> begin_validation();
    Result<void> end_validation();
    Result<void> set_device() const;
    Result<void> run_current(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceRunValue> inputs,
        std::span<const DeviceRunValue> outputs);
    bool is_capture_eligible(
        const CudaDeviceKernel& kernel,
        const DeviceRunCommand& command) const;
    Result<CudaGraphRegionKey> graph_region_key(
        std::span<const DeviceRunCommand> commands) const;
    Result<void> execute_eager_commands(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands);
    Result<void> execute_recordable_region(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands);
    Result<void> capture_region(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands,
        CudaCapturedRegion& region);
    void destroy_captured_regions(CudaDeviceGraph& graph);
    Result<CudaDeviceBufferView> buffer_view(DeviceBufferId buffer, bool writable);
    Result<CudaDevicePagedTensorView> paged_tensor_view(DevicePagedTensorId tensor) const;
    Result<void> sync_paged_tensor_table(DevicePagedTensorId tensor);

    int cudaDevice_ = 0;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublasHandle_ = nullptr;
    ir::kernel_ir::OpId* validationFailure_ = nullptr;
    bool executableRunActive_ = false;
    cudaDeviceProp deviceProps_{};
    std::string initializationError_;
    bool asyncMemoryPoolConfigured_ = false;
    DeviceBufferId nextBufferId_ = 1;
    DeviceCompiledGraphId nextGraphId_ = 1;
    DevicePagedPoolId nextPagedPoolId_ = 1;
    DevicePagedTensorId nextPagedTensorId_ = 1;
    absl::flat_hash_map<DeviceBufferId, CudaDeviceBuffer> buffers_;
    absl::flat_hash_map<DeviceCompiledGraphId, CudaDeviceGraph> graphs_;
    absl::flat_hash_map<DevicePagedPoolId, CudaPagedTensorPool> pagedPools_;
    absl::flat_hash_map<DevicePagedTensorId, CudaPagedTensor> pagedTensors_;
    CudaJitCache jitCache_;
    CudaGraphCacheStats graphCacheStats_;
};

} // namespace sandy::device
