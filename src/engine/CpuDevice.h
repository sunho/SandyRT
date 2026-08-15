#pragma once

#include "Device.h"
#include "MidIR.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sandy::engine {

class CpuDevice final : public Device {
public:
    Result<DeviceProgramId> compile(const ir::mid_ir::Op& op) override;

    Result<DeviceBufferId> alloc(core::TensorDesc desc) override;
    Result<void> dealloc(DeviceBufferId buffer) override;

    Result<DeviceBufferId> load(core::TensorBuffer& src) override;

    Result<void> run(
        DeviceProgramId program,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) override;

    Result<TensorBufferPtr> read(DeviceBufferId src) override;

private:
    struct CpuDeviceBuffer {
        core::TensorDesc desc;
        std::vector<uint8_t> data;
    };

    struct CpuDeviceProgram {
        ir::mid_ir::OpKind kind = ir::mid_ir::OpKind::NUM_KINDS;
        ir::mid_ir::AttrMap attrs;
        std::vector<core::TensorDesc> inputDescs;
        std::vector<core::TensorDesc> outputDescs;
    };

    DeviceBufferId nextBufferId_ = 1;
    DeviceProgramId nextProgramId_ = 1;
    std::unordered_map<DeviceBufferId, CpuDeviceBuffer> buffers_;
    std::unordered_map<DeviceProgramId, CpuDeviceProgram> programs_;
};

} // namespace sandy::engine
