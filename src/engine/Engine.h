#pragma once

#include "Backend.h"
#include "EngineTypes.h"
#include "Result.h"

#include <memory>

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

    Result<std::unique_ptr<Plan>> create_plan(const ir::mid_ir::Graph& graph);

    Result<void> run(
        const Plan& plan,
        const TensorMap& inputs,
        const TensorMap& weights,
        const RunOptions& options = {});

private:
    std::unique_ptr<backend::Backend> backend_;
};

} // namespace sandy::engine
