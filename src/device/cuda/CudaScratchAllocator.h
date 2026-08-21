#pragma once

#include "Device.h"

#include <cstddef>
#include <unordered_map>

namespace sandy::device {

class CudaDevice;

class CudaScratchAllocator final : public DeviceScratchAllocator {
public:
    explicit CudaScratchAllocator(CudaDevice& device) : device_(device) {}

    Result<void> alloc(
        ir::kernel_ir::ValueId value,
        core::TensorDesc desc) override;
    Result<void> free(ir::kernel_ir::ValueId value) override;
    Result<DeviceScratchAllocation> finalize() override;

    size_t requiredBytes() const { return cursor_; }

private:
    struct Placement {
        core::TensorDesc desc;
        size_t byteOffset = 0;
    };

    CudaDevice& device_;
    size_t cursor_ = 0;
    bool finalized_ = false;
    std::unordered_map<ir::kernel_ir::ValueId, Placement> placements_;
    std::unordered_map<ir::kernel_ir::ValueId, bool> live_;
};

} // namespace sandy::device

