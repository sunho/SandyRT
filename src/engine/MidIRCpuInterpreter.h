#pragma once

#include "MidIR.h"
#include "Result.h"
#include "TensorCalc.h"

#include <span>

namespace sandy::engine::debug {

Result<void> runMidIROpOnCpu(
    const ir::mid_ir::Op& op,
    std::span<const core::TensorRef> inputs,
    std::span<const core::MutableTensorRef> outputs);

} // namespace sandy::engine::debug
