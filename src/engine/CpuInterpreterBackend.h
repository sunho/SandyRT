#pragma once

#include "Backend.h"

namespace sandy::engine::backend {

class CpuInterpreterBackend final : public Backend {
public:
    Result<BackendBufferPtr> create_buffer(core::TensorBuffer& buffer) override;

    Result<std::unique_ptr<Program>> compile(
        const ir::mid_ir::Graph& graph) override;

    Result<void> run(
        const Program& program,
        const BackendBufferMap& inputs,
        const BackendBufferMap& weights,
        const RunOptions& options) override;
};

} // namespace sandy::engine::backend
