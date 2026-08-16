#pragma once

#include "Device.h"
#include "DeviceWiseCopier.h"
#include "EngineTypes.h"
#include "KernelIR.h"
#include "MidIR.h"
#include "Result.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace sandy::engine {

struct EngineProfileEvent {
    size_t opIndex = 0;
    ir::kernel_ir::OpId op = 0;
    uint32_t device = 0;
    DeviceCompiledGraphId deviceGraph = 0;
    ir::kernel_ir::OpKind opKind = ir::kernel_ir::OpKind::Input;
    size_t inputCount = 0;
    size_t outputCount = 0;
    double elapsedMs = 0.0;
};

struct EngineRunOptions {
    std::function<void(const EngineProfileEvent&)> profileKernel;
};

class Engine {
public:
    explicit Engine(
        std::vector<std::unique_ptr<device::Device>> devices,
        std::unique_ptr<device::DeviceWiseCopier> copier = nullptr);

    Result<std::unique_ptr<CompiledKernelGraph>> compile(const ir::mid_ir::Graph& graph);

    Result<std::vector<TensorBufferPtr>> run(
        const CompiledKernelGraph& compiled,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options = nullptr);

private:
    std::vector<std::unique_ptr<device::Device>> devices_;
    std::unique_ptr<device::DeviceWiseCopier> copier_;
};

} // namespace sandy::engine
