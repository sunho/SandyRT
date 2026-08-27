#pragma once

#include "DeviceTypes.h"
#include "KernelIR.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <string>
#include <variant>
#include <vector>

namespace sandy::engine {

using CompiledProgramId = uint64_t;

struct RunOptions {};

using device::DeviceBufferId;
using device::DeviceCompiledGraphId;
using device::DevicePagedTensorView;
using device::DeviceProgramId;
using device::DeviceTensorView;
using device::TensorBufferPtr;
using device::TensorMap;
using device::TensorViewDesc;

using RunTensorLike = std::variant<TensorBufferPtr, DevicePagedTensorView>;

struct RunTensorTuple {
    std::vector<RunTensorLike> elements;
};

using RunInput = std::variant<TensorBufferPtr, DevicePagedTensorView, RunTensorTuple>;
using RunOutput = std::variant<TensorBufferPtr, DevicePagedTensorView, RunTensorTuple>;

struct CompiledKernelGraph {
    CompiledProgramId programId = 0;
    std::unique_ptr<ir::kernel_ir::Graph> graph;

    ir::kernel_ir::DeviceId defaultDevice = 0;
    std::unordered_map<ir::kernel_ir::DeviceId, DeviceCompiledGraphId> deviceGraphs;

    // Legacy single-device fields kept for tests and manually constructed graphs.
    uint32_t device = 0;
    DeviceCompiledGraphId deviceGraph = 0;
};

struct DeviceWeightMap {
    DeviceWeightMap() = default;
    DeviceWeightMap(const DeviceWeightMap&) = delete;
    DeviceWeightMap& operator=(const DeviceWeightMap&) = delete;
    DeviceWeightMap(DeviceWeightMap&&) = default;
    DeviceWeightMap& operator=(DeviceWeightMap&&) = default;

    struct DeviceWeights {
        std::unordered_map<std::string, DeviceTensorView> tensors;
    };

    std::unordered_map<ir::kernel_ir::DeviceId, DeviceWeights> weightsByDevice;
};

} // namespace sandy::engine
