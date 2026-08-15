#include "Engine.h"

#include "InvocPlanner.h"

#include <memory>
#include <utility>

namespace sandy::engine {

namespace {

class SimplePlan final : public Plan {
public:
    explicit SimplePlan(std::unique_ptr<backend::Program> program)
        : program_(std::move(program)) {}

private:
    const backend::Program& backend_program() const override { return *program_; }

    std::unique_ptr<backend::Program> program_;
};

Result<backend::BackendBufferMap> create_backend_buffers(
        backend::Backend& backend,
        const TensorMap& tensors) {
    backend::BackendBufferMap buffers;
    for (const auto& [name, tensor] : tensors) {
        if (!tensor)
            return make_error("null tensor buffer for '" + name + "'");

        auto result = backend.create_buffer(*tensor);
        if (!result)
            return make_error(result.error());

        buffers[name] = result.take();
    }
    return buffers;
}

} // namespace

Engine::Engine(std::unique_ptr<backend::Backend> backend)
    : backend_(std::move(backend)) {}

Engine::Engine(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)) {}

Result<std::unique_ptr<InvocPlan>> Engine::compile(const ir::mid_ir::Graph& graph) {
    if (devices_.empty())
        return make_error("engine has no devices");

    InvocPlanner planner(0);
    auto draftResult = planner.plan(graph);
    if (!draftResult)
        return make_error(draftResult.error());
    auto draft = draftResult.take();

    auto plan = std::make_unique<InvocPlan>();
    plan->instructions = std::move(draft.instructions);
    plan->outputs = std::move(draft.outputs);

    for (const auto& source : draft.programSources) {
        if (source.device >= devices_.size())
            return make_error("invocation program references invalid device");
        if (!source.op)
            return make_error("invocation program has no MidIR op");

        auto compiled = devices_[source.device]->compile(*source.op);
        if (!compiled)
            return make_error(compiled.error());
        plan->programs.push_back({
            source.id,
            source.device,
            compiled.take(),
        });
    }

    return plan;
}

Result<std::unique_ptr<Plan>> Engine::create_plan(const ir::mid_ir::Graph& graph) {
    if (!backend_)
        return make_error("engine has no backend");

    auto program = backend_->compile(graph);
    if (!program)
        return make_error(program.error());

    std::unique_ptr<Plan> plan = std::make_unique<SimplePlan>(program.take());
    return plan;
}

Result<backend::BackendRunResult> Engine::run(
        const Plan& plan,
        const TensorMap& inputs,
        const TensorMap& weights,
        const RunOptions& options) {
    if (!backend_)
        return make_error("engine has no backend");

    auto backendInputs = create_backend_buffers(*backend_, inputs);
    if (!backendInputs)
        return make_error(backendInputs.error());

    auto backendWeights = create_backend_buffers(*backend_, weights);
    if (!backendWeights)
        return make_error(backendWeights.error());

    return backend_->run(
        plan.backend_program(),
        backendInputs.take(),
        backendWeights.take(),
        options);
}

} // namespace sandy::engine
