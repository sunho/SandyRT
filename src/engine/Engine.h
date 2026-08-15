#pragma once

#include "Device.h"
#include "EngineTypes.h"
#include "InvocPlan.h"
#include "MidIR.h"
#include "Result.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace sandy::engine {

struct InvocProfileEvent {
    size_t instructionIndex = 0;
    InvocProgramId program = 0;
    InvocDeviceId device = 0;
    DeviceProgramId deviceProgram = 0;
    ir::mid_ir::OpKind opKind = ir::mid_ir::OpKind::NUM_KINDS;
    size_t inputCount = 0;
    size_t outputCount = 0;
    double elapsedMs = 0.0;
};

struct EngineRunOptions {
    std::function<void(const InvocProfileEvent&)> profileKernel;
};

class Engine {
public:
    explicit Engine(std::vector<std::unique_ptr<Device>> devices);

    Result<std::unique_ptr<InvocPlan>> compile(const ir::mid_ir::Graph& graph);

    Result<std::vector<TensorBufferPtr>> run(
        const InvocPlan& plan,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options = nullptr);

private:
    std::vector<std::unique_ptr<Device>> devices_;
};

} // namespace sandy::engine
