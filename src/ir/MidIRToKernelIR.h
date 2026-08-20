#pragma once

#include "KernelIR.h"
#include "MidIR.h"
#include "Result.h"

#include <cstdint>
#include <memory>

namespace sandy::ir::kernel_ir {

struct FusorOptions {
    bool attention = false;
    bool elementwise = false;
    uint32_t maxElementwiseInputs = 8;
    uint32_t maxElementwiseScalars = 32;
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
