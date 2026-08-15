#pragma once

#include "MidIR.h"
#include "Result.h"

#include <memory>
#include <vector>

namespace sandy::ir::mid_ir {

struct PassResult {
    bool changed = false;
};

class Pass {
public:
    virtual ~Pass() = default;
    virtual const char* name() const = 0;
    virtual Result<PassResult> run(Graph& graph) = 0;
};

class PassManager {
public:
    void add(std::unique_ptr<Pass> pass);
    Result<PassResult> run(Graph& graph);

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

std::unique_ptr<Pass> createFuseTransposeIntoMatMulPass();
std::unique_ptr<Pass> createDeadCodeEliminationPass();

} // namespace sandy::ir::mid_ir
