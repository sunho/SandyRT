#pragma once

#include "Device.h"
#include "EngineTypes.h"
#include "RuntimeTensorDesc.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace sandy::engine {

struct RuntimeScratchPlan {
    std::unordered_map<ir::kernel_ir::ValueId, device::DeviceTensorView> views;
    std::unordered_map<ir::kernel_ir::ValueId, ir::kernel_ir::DeviceId> devices;
    std::unordered_map<ir::kernel_ir::DeviceId, device::DeviceBufferId> buffers;
};

// Simulates the graph in execution order. Value use counts are consumed here,
// before execution, and translated into allocator alloc/free events.
Result<RuntimeScratchPlan> planRuntimeScratch(
    const CompiledKernelGraph& compiled,
    const RuntimeTensorDescs& tensorDescs,
    std::vector<std::unique_ptr<device::Device>>& devices);

} // namespace sandy::engine
