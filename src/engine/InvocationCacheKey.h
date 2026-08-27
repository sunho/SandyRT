#pragma once

#include "CacheKey.h"
#include "EngineTypes.h"
#include "RuntimeTensorDesc.h"

#include <string_view>

namespace sandy::engine {

// Builds a canonical key from InputOp descriptors in graph order. Tensor tuple
// elements are already represented by their lowered InputOps and tupleElement
// selectors, which are included in the encoding.
Result<core::CacheKey> buildInvocationCacheKey(
    std::string_view domain,
    CompiledProgramId program,
    const ir::kernel_ir::Graph& graph,
    const RuntimeTensorDescs& inputDescs);

} // namespace sandy::engine

