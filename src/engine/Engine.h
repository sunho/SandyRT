#pragma once

#include "Backend.h"
#include "Device.h"
#include "EngineTypes.h"
#include "InvocPlan.h"
#include "Result.h"

#include <memory>
#include <vector>

namespace sandy::engine {

class Plan {
public:
    virtual ~Plan() = default;

private:
    friend class Engine;
    virtual const backend::Program& backend_program() const = 0;
};

class Engine {
public:
    explicit Engine(std::unique_ptr<backend::Backend> backend);
    explicit Engine(std::vector<std::unique_ptr<Device>> devices);

    Result<std::unique_ptr<InvocPlan>> compile(const ir::mid_ir::Graph& graph);

    Result<std::unique_ptr<Plan>> create_plan(const ir::mid_ir::Graph& graph);

    Result<backend::BackendRunResult> run(
        const Plan& plan,
        const TensorMap& inputs,
        const TensorMap& weights,
        const RunOptions& options = {});

private:
    std::unique_ptr<backend::Backend> backend_;
    std::vector<std::unique_ptr<Device>> devices_;
};

} // namespace sandy::engine
