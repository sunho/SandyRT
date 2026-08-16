#pragma once

#include "DeviceTypes.h"
#include "KernelIR.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace sandy::engine {

struct RunOptions {};

using device::DeviceBufferId;
using device::DeviceCompiledGraphId;
using device::DeviceProgramId;
using device::DeviceTensorView;
using device::TensorBufferPtr;
using device::TensorMap;
using device::TensorViewDesc;

struct CompiledKernelGraph {
    std::unique_ptr<ir::kernel_ir::Graph> graph;

    ir::kernel_ir::DeviceId defaultDevice = 0;
    std::unordered_map<ir::kernel_ir::DeviceId, DeviceCompiledGraphId> deviceGraphs;

    // Legacy single-device fields kept for tests and manually constructed graphs.
    uint32_t device = 0;
    DeviceCompiledGraphId deviceGraph = 0;
};

} // namespace sandy::engine
