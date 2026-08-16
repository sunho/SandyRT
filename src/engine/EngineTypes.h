#pragma once

#include "KernelIR.h"
#include "TensorBuffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace sandy::engine {

struct RunOptions {};

using DeviceBufferId = uint32_t;
using DeviceCompiledGraphId = uint32_t;
using DeviceProgramId = DeviceCompiledGraphId;

using TensorBufferPtr = std::shared_ptr<core::TensorBuffer>;
using TensorMap = std::unordered_map<std::string, TensorBufferPtr>;

struct CompiledKernelGraph {
    std::unique_ptr<ir::kernel_ir::Graph> graph;

    ir::kernel_ir::DeviceId defaultDevice = 0;
    std::unordered_map<ir::kernel_ir::DeviceId, DeviceCompiledGraphId> deviceGraphs;

    // Legacy single-device fields kept for tests and manually constructed graphs.
    uint32_t device = 0;
    DeviceCompiledGraphId deviceGraph = 0;
};

} // namespace sandy::engine
