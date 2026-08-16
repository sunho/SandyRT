#pragma once

#include "DeviceTypes.h"
#include "KernelIR.h"
#include "Result.h"
#include "Tensor.h"
#include "TensorBuffer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sandy::device {

class Device {
public:
    virtual ~Device() = default;

    virtual Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) = 0;

    virtual Result<DeviceBufferId> alloc(core::TensorDesc desc) = 0;
    virtual Result<void> dealloc(DeviceBufferId buffer) = 0;

    virtual Result<DeviceBufferId> load(core::TensorBuffer& src) = 0;

    virtual Result<std::vector<int64_t>> defaultStrides(const core::Shape& shape) const;
    virtual Result<TensorViewDesc> defaultView(core::TensorDesc desc) const;
    virtual Result<bool> isDefaultView(const TensorViewDesc& view) const;

    virtual Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceTensorView> inputs,
        std::span<const DeviceTensorView> outputs) = 0;

    virtual Result<TensorBufferPtr> read(DeviceTensorView src) = 0;
};

} // namespace sandy::device
