#include "CpuInterpreterBackend.h"

#include "MidIR.h"

#include <memory>
#include <utility>

namespace sandy::engine::backend {

namespace {

class CpuProgram final : public Program {
public:
    explicit CpuProgram(const ir::mid_ir::Graph& graph)
        : graph_(&graph) {}

    const ir::mid_ir::Graph& graph() const { return *graph_; }

private:
    const ir::mid_ir::Graph* graph_;
};

class CpuBackendBuffer final : public BackendBuffer {
public:
    explicit CpuBackendBuffer(core::TensorBuffer::Access access)
        : access_(std::move(access)) {
        (void)access_.data();
    }

    const core::TensorDesc& desc() const override { return access_.desc(); }

private:
    core::TensorBuffer::Access access_;
};

Result<void> validate_buffers(const BackendBufferMap& buffers) {
    for (const auto& [name, buffer] : buffers) {
        if (!buffer)
            return make_error("null backend buffer for '" + name + "'");

        (void)buffer->desc();
    }
    return {};
}

} // namespace

Result<BackendBufferPtr> CpuInterpreterBackend::create_buffer(core::TensorBuffer& buffer) {
    auto access = buffer.access();
    if (!access)
        return make_error(access.error());

    BackendBufferPtr backendBuffer = std::make_shared<CpuBackendBuffer>(access.take());
    return backendBuffer;
}

Result<std::unique_ptr<Program>> CpuInterpreterBackend::compile(
        const ir::mid_ir::Graph& graph) {
    std::unique_ptr<Program> program = std::make_unique<CpuProgram>(graph);
    return program;
}

Result<void> CpuInterpreterBackend::run(
        const Program& program,
        const BackendBufferMap& inputs,
        const BackendBufferMap& weights,
        const RunOptions&) {
    (void)program;

    auto inputResult = validate_buffers(inputs);
    if (!inputResult)
        return inputResult;

    return validate_buffers(weights);
}

} // namespace sandy::engine::backend
