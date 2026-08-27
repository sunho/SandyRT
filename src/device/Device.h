#pragma once

#include "DeviceTypes.h"
#include "KernelIR.h"
#include "Result.h"
#include "Tensor.h"
#include "TensorBuffer.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sandy::device {

class DeviceScratchAllocator {
public:
    virtual ~DeviceScratchAllocator() = default;

    virtual Result<void> alloc(
        ir::kernel_ir::ValueId value,
        core::TensorDesc desc) = 0;
    virtual Result<void> free(ir::kernel_ir::ValueId value) = 0;
    virtual Result<DeviceScratchLayout> finalizeLayout() = 0;
};

class Device {
public:
    virtual ~Device() = default;

    virtual Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) = 0;

    virtual Result<DeviceBufferId> alloc(core::TensorDesc desc) = 0;
    virtual Result<void> dealloc(DeviceBufferId buffer) = 0;

    virtual std::unique_ptr<DeviceScratchAllocator> createScratchAllocator();

    virtual Result<DeviceBufferId> load(core::TensorBuffer& src) = 0;

    Result<DevicePagedPoolId> createPagedPool(DevicePagedPoolDesc desc);
    virtual Result<void> destroyPagedPool(DevicePagedPoolId pool);
    virtual Result<DevicePagedTensorId> allocPaged(
        DevicePagedPoolId pool,
        core::Shape logicalShape);
    virtual Result<void> deallocPaged(DevicePagedTensorId tensor);
    virtual Result<void> reservePaged(DevicePagedTensorId tensor, int64_t pageCount);
    virtual Result<void> appendPaged(DevicePagedTensorId dst, core::TensorBuffer& denseChunk);
    virtual Result<void> appendPaged(DevicePagedTensorId dst, DeviceTensorView denseChunk);
    virtual Result<DevicePagedTensorMeta> pagedMeta(DevicePagedTensorId tensor) const;

    virtual Result<std::vector<int64_t>> defaultStrides(const core::Shape& shape) const;
    virtual Result<TensorViewDesc> defaultView(core::TensorDesc desc) const;
    virtual Result<bool> isDefaultView(const TensorViewDesc& view) const;

    virtual Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceRunValue> inputs,
        std::span<const DeviceRunValue> outputs) = 0;

    virtual Result<TensorBufferPtr> read(DeviceTensorView src) = 0;
    virtual Result<TensorBufferPtr> read(DevicePagedTensorView src);

protected:
    virtual Result<DevicePagedPoolId> createPagedPoolImpl(DevicePagedPoolDesc desc);
};

} // namespace sandy::device
