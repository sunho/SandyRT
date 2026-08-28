#pragma once

#include "DeviceTypes.h"
#include "KernelIR.h"
#include "Result.h"

#include <cstdint>
#include <vector>

namespace sandy::engine {

struct DeviceExecutableNodePlan {
    ir::kernel_ir::DeviceId device = 0;
    device::DeviceExecutableDesc executable;
};

struct KernelExecutionPlan {
    std::vector<DeviceExecutableNodePlan> nodes;
    std::vector<int32_t> nodeForOp;
};

Result<KernelExecutionPlan> partitionKernelGraph(
    const ir::kernel_ir::Graph& graph);

} // namespace sandy::engine
