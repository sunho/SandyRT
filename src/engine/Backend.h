#pragma once

#include "EngineTypes.h"
#include "Result.h"
#include "TensorBuffer.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace sandy::ir::mid_ir {
class Graph;
} // namespace sandy::ir::mid_ir

namespace sandy::engine::backend {

class Program {
public:
    virtual ~Program() = default;
};

class BackendBuffer {
public:
    virtual ~BackendBuffer() = default;
    virtual const core::TensorDesc& desc() const = 0;
};

using BackendBufferPtr = std::shared_ptr<BackendBuffer>;
using BackendBufferMap = std::unordered_map<std::string, BackendBufferPtr>;

class Backend {
public:
    virtual ~Backend() = default;

    virtual Result<BackendBufferPtr> create_buffer(core::TensorBuffer& buffer) = 0;

    virtual Result<std::unique_ptr<Program>> compile(
        const ir::mid_ir::Graph& graph) = 0;

    virtual Result<void> run(
        const Program& program,
        const BackendBufferMap& inputs,
        const BackendBufferMap& weights,
        const RunOptions& options) = 0;
};

} // namespace sandy::engine::backend
