#pragma once

#include "Device.h"
#include "EngineTypes.h"
#include "InvocPlan.h"
#include "MidIR.h"
#include "Result.h"

#include <memory>
#include <span>
#include <vector>

namespace sandy::engine {

class Engine {
public:
    explicit Engine(std::vector<std::unique_ptr<Device>> devices);

    Result<std::unique_ptr<InvocPlan>> compile(const ir::mid_ir::Graph& graph);

    Result<std::vector<TensorBufferPtr>> run(
        const InvocPlan& plan,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights);

private:
    std::vector<std::unique_ptr<Device>> devices_;
};

} // namespace sandy::engine
