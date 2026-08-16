#pragma once

#include "KernelIR.h"
#include "MidIR.h"
#include "Result.h"

#include <memory>

namespace sandy::ir::kernel_ir {

class MidIRToKernelIRLowering {
public:
    Result<std::unique_ptr<Graph>> lower(const mid_ir::Graph& graph);
};

Result<std::unique_ptr<Graph>> lowerMidIRToKernelIR(const mid_ir::Graph& graph);

} // namespace sandy::ir::kernel_ir
