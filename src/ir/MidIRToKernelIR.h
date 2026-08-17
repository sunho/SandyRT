#pragma once

#include "KernelIR.h"
#include "MidIR.h"
#include "Result.h"

#include <memory>

namespace sandy::ir::kernel_ir {

struct FusorOptions {
    bool attention = false;
};

struct LoweringOptions {
    FusorOptions fusor;
};

class MidIRToKernelIRLowering {
public:
    explicit MidIRToKernelIRLowering(LoweringOptions options = {});
    Result<std::unique_ptr<Graph>> lower(const mid_ir::Graph& graph);

private:
    LoweringOptions options_;
};

Result<std::unique_ptr<Graph>> lowerMidIRToKernelIR(
    const mid_ir::Graph& graph,
    const LoweringOptions& options = {});

} // namespace sandy::ir::kernel_ir
