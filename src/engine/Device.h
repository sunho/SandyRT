#pragma once

#include "EngineTypes.h"
#include "Result.h"
#include "Tensor.h"
#include "TensorBuffer.h"

#include <cstdint>
#include <span>

namespace sandy::ir::mid_ir {
struct Op;
} // namespace sandy::ir::mid_ir

namespace sandy::engine {

using DeviceBufferId = uint32_t;
using DeviceProgramId = uint32_t;

class Device {
public:
    virtual ~Device() = default;

    virtual Result<DeviceProgramId> compile(const ir::mid_ir::Op& op) = 0;

    virtual Result<DeviceBufferId> alloc(core::TensorDesc desc) = 0;
    virtual Result<void> dealloc(DeviceBufferId buffer) = 0;

    virtual Result<DeviceBufferId> load(core::TensorBuffer& src) = 0;

    virtual Result<void> run(
        DeviceProgramId program,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) = 0;

    virtual Result<TensorBufferPtr> read(DeviceBufferId src) = 0;
};

} // namespace sandy::engine
