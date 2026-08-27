#pragma once

#include "Device.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace sandy::device {

class CudaDevice;

class CudaScratchAllocator final : public DeviceScratchAllocator {
public:
    CudaScratchAllocator() = default;

    Result<void> alloc(
        ir::kernel_ir::ValueId value,
        core::TensorDesc desc) override;
    Result<void> free(ir::kernel_ir::ValueId value) override;
    Result<DeviceScratchLayout> finalizeLayout() override;

    size_t requiredBytes() const { return cursor_; }

private:
    struct Placement {
        core::TensorDesc desc;
        size_t byteOffset = 0;
        size_t reservedBytes = 0;
    };

    struct FreeRange {
        size_t byteOffset = 0;
        size_t bytes = 0;
    };

    size_t cursor_ = 0;
    bool finalized_ = false;
    std::unordered_map<ir::kernel_ir::ValueId, Placement> placements_;
    std::unordered_map<ir::kernel_ir::ValueId, bool> live_;
    std::vector<FreeRange> freeRanges_;
};

} // namespace sandy::device
