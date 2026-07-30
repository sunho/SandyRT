#pragma once

#include "EngineTypes.h"
#include "Result.h"
#include "TensorBuffer.h"

#include <memory>
#include <span>
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
    virtual std::span<const uint8_t> data() const = 0;
};

using BackendBufferPtr = std::unique_ptr<BackendBuffer>;
using BackendBufferMap = std::unordered_map<std::string, BackendBufferPtr>;
using BackendRunResult = BackendBufferMap;

class Backend {
public:
    virtual ~Backend() = default;

    virtual Result<BackendBufferPtr> create_buffer(core::TensorBuffer& buffer) = 0;

    virtual Result<std::unique_ptr<Program>> compile(
        const ir::mid_ir::Graph& graph) = 0;

    virtual Result<BackendRunResult> run(
        const Program& program,
        BackendBufferMap inputs,
        BackendBufferMap weights,
        const RunOptions& options) = 0;
};

} // namespace sandy::engine::backend
