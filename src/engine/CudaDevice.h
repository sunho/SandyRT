#pragma once

#include "CudaKernels.h"
#include "Device.h"
#include "KernelIR.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sandy::engine {

class CudaDevice final : public Device {
public:
    explicit CudaDevice(int cudaDevice = 0);
    ~CudaDevice() override;

    Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) override;

    Result<DeviceBufferId> alloc(core::TensorDesc desc) override;
    Result<void> dealloc(DeviceBufferId buffer) override;

    Result<DeviceBufferId> load(core::TensorBuffer& src) override;

    Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceTensorView> inputs,
        std::span<const DeviceTensorView> outputs) override;

    Result<TensorBufferPtr> read(DeviceTensorView src) override;
    Result<TensorBufferPtr> read(DeviceBufferId src);

private:
    using KernelProgram = std::variant<
        std::monostate,
        CudaElementwiseProgram,
        CudaLayoutTransformProgram,
        CudaMatMulProgram,
        CudaReductionProgram,
        CudaSoftmaxProgram,
        CudaNormProgram,
        CudaRoPEProgram,
        CudaSlidingQueryKeyScoreProgram,
        CudaCustomProgram>;

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
    Result<void> set_device() const;
    Result<CudaDeviceBufferView> buffer_view(DeviceBufferId buffer, bool writable);

    int cudaDevice_ = 0;
    cudaStream_t stream_ = nullptr;
    DeviceBufferId nextBufferId_ = 1;
    DeviceCompiledGraphId nextGraphId_ = 1;
    std::unordered_map<DeviceBufferId, CudaDeviceBuffer> buffers_;
    std::unordered_map<DeviceCompiledGraphId, CudaDeviceGraph> graphs_;
};

} // namespace sandy::engine
