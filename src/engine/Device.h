#pragma once

#include "EngineTypes.h"
#include "KernelIR.h"
#include "Result.h"
#include "Tensor.h"
#include "TensorBuffer.h"

#include <cstdint>
#include <span>

namespace sandy::engine {

class Device {
public:
    virtual ~Device() = default;

    virtual Result<DeviceCompiledGraphId> compile(const ir::kernel_ir::Graph& graph) = 0;

    virtual Result<DeviceBufferId> alloc(core::TensorDesc desc) = 0;
    virtual Result<void> dealloc(DeviceBufferId buffer) = 0;

    virtual Result<DeviceBufferId> load(core::TensorBuffer& src) = 0;

    virtual Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) = 0;

    virtual Result<TensorBufferPtr> read(DeviceBufferId src) = 0;
};

} // namespace sandy::engine
