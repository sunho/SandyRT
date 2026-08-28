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

class Device;

class DeviceScratchAllocator {
public:
    virtual ~DeviceScratchAllocator() = default;

    virtual Result<void> alloc(
        ir::kernel_ir::ValueId value,
        core::TensorDesc desc) = 0;
    virtual Result<void> free(ir::kernel_ir::ValueId value) = 0;
    virtual Result<DeviceScratchLayout> finalizeLayout() = 0;
};

class DeviceExecutable {
public:
    ~DeviceExecutable();

    DeviceExecutable(const DeviceExecutable&) = delete;
    DeviceExecutable& operator=(const DeviceExecutable&) = delete;

    DeviceCompiledGraphId compiledGraph() const { return compiledGraph_; }
    const DeviceExecutableDesc& desc() const { return desc_; }
    const ir::kernel_ir::Graph& graph() const { return *graph_; }
    bool hasFixedScratch(ir::kernel_ir::ValueId value) const {
        return fixedViews_.contains(value);
    }
    size_t fixedScratchValueCount() const { return fixedViews_.size(); }

private:
    friend class Device;

    DeviceExecutable(
        Device& owner,
        const ir::kernel_ir::Graph& graph,
        DeviceCompiledGraphId compiledGraph,
        DeviceExecutableDesc desc);

    Device* owner_ = nullptr;
    const ir::kernel_ir::Graph* graph_ = nullptr;
    DeviceCompiledGraphId compiledGraph_ = 0;
    DeviceExecutableDesc desc_;
    DeviceBufferId fixedScratchBuffer_ = 0;
    std::unordered_map<ir::kernel_ir::ValueId, DeviceTensorView> fixedViews_;
};

using DeviceExecutablePtr = std::shared_ptr<DeviceExecutable>;

class Device {
public:
    friend class DeviceExecutable;

    virtual ~Device() = default;

    virtual Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) = 0;

    Result<DeviceExecutablePtr> compileExecutable(
        const ir::kernel_ir::Graph& graph,
        DeviceExecutableDesc desc);
    Result<void> runExecutable(
        const DeviceExecutable& executable,
        DeviceExecutableRunState& state);

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
    virtual Result<DeviceCompiledGraphId> compileExecutableGraph(
        const ir::kernel_ir::Graph& graph,
        std::span<const ir::kernel_ir::OpId> ops);
    virtual Result<void> destroyCompiledGraph(DeviceCompiledGraphId graph);
    virtual Result<DevicePagedPoolId> createPagedPoolImpl(DevicePagedPoolDesc desc);
};

} // namespace sandy::device
