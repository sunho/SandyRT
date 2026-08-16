#pragma once

#include "Allocator.h"
#include "Device.h"
#include "KernelIR.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::device {

class CpuDevice final : public Device {
public:
    Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) override;

    Result<DeviceBufferId> alloc(core::TensorDesc desc) override;
    Result<void> dealloc(DeviceBufferId buffer) override;

    Result<DeviceBufferId> load(core::TensorBuffer& src) override;

    Result<DevicePagedPoolId> createPagedPool(DevicePagedPoolDesc desc) override;
    Result<void> destroyPagedPool(DevicePagedPoolId pool) override;
    Result<DevicePagedTensorId> allocPaged(
        DevicePagedPoolId pool,
        core::Shape logicalShape) override;
    Result<void> deallocPaged(DevicePagedTensorId tensor) override;
    Result<void> reservePaged(DevicePagedTensorId tensor, int64_t pageCount) override;
    Result<void> appendPaged(DevicePagedTensorId dst, core::TensorBuffer& denseChunk) override;
    Result<DevicePagedTensorMeta> pagedMeta(DevicePagedTensorId tensor) const override;

    Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceTensorView> inputs,
        std::span<const DeviceTensorView> outputs) override;

    Result<TensorBufferPtr> read(DeviceTensorView src) override;
    Result<TensorBufferPtr> read(DeviceBufferId src);

private:
    struct CpuDeviceBuffer {
        core::TensorDesc desc;
        std::vector<uint8_t> data;
        std::optional<core::TensorBuffer::Access> borrowed;
    };

    struct CpuDeviceKernel {
        ir::kernel_ir::OpKind kind = ir::kernel_ir::OpKind::Input;
        size_t inputCount = 0;
        size_t outputCount = 0;

        ir::kernel_ir::LayoutTransformKind layoutTransform =
            ir::kernel_ir::LayoutTransformKind::Contiguous;
        std::vector<int64_t> dims;

        ir::kernel_ir::ScalarOp scalarOp = ir::kernel_ir::ScalarOp::Constant;
        double constant = 0.0;

        bool transposeLhs = false;
        bool transposeRhs = false;
        int64_t axis = -1;
        int64_t window = 0;
        double scale = -1.0;
        double theta = 10000.0;
        int64_t rotaryDim = -1;
        bool splitHalf = false;
        ir::kernel_ir::NormKind norm = ir::kernel_ir::NormKind::RMSNorm;
        double epsilon = 0.0;
        std::string customName;
    };

    struct CpuDeviceGraph {
        std::unordered_map<ir::kernel_ir::OpId, CpuDeviceKernel> kernels;
    };

    struct CpuPagedPool {
        DevicePagedPoolDesc desc;
        int64_t pageElementCount = 0;
        size_t pageBytes = 0;
        core::FixedPagePool pages;
    };

    struct CpuPagedTensor {
        DevicePagedPoolId pool = 0;
        core::Shape logicalShape;
        int64_t growLength = 0;
        std::vector<uint32_t> pageIndices;
    };

    DeviceBufferId nextBufferId_ = 1;
    DeviceCompiledGraphId nextGraphId_ = 1;
    DevicePagedPoolId nextPagedPoolId_ = 1;
    DevicePagedTensorId nextPagedTensorId_ = 1;
    std::unordered_map<DeviceBufferId, CpuDeviceBuffer> buffers_;
    std::unordered_map<DeviceCompiledGraphId, CpuDeviceGraph> graphs_;
    std::unordered_map<DevicePagedPoolId, CpuPagedPool> pagedPools_;
    std::unordered_map<DevicePagedTensorId, CpuPagedTensor> pagedTensors_;
};

} // namespace sandy::device
