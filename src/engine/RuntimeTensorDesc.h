#pragma once

#include "KernelIR.h"
#include "Result.h"
#include "Tensor.h"

#include <optional>
#include <string>
#include <vector>

namespace sandy::engine {

class RuntimeTensorDescs {
public:
    explicit RuntimeTensorDescs(size_t valueCount = 0) : descs_(valueCount) {}

    bool has(ir::kernel_ir::ValueId value) const;
    const core::TensorDesc& get(ir::kernel_ir::ValueId value) const;
    Result<const core::TensorDesc*> lookup(ir::kernel_ir::ValueId value) const;
    Result<void> set(ir::kernel_ir::ValueId value, core::TensorDesc desc);

private:
    std::vector<std::optional<core::TensorDesc>> descs_;
};

Result<void> verifyRuntimeTensorDesc(
    const core::TensorDesc& desc,
    const ir::kernel_ir::ValueType& type,
    const std::string& valueName);

// Resolves every tensor descriptor for one invocation before graph execution.
// inputDescs must contain concrete descriptors for all InputOp outputs. The
// walk also models shape-changing side effects such as PagedAppendOp.
Result<RuntimeTensorDescs> inferRuntimeTensorDescs(
    const ir::kernel_ir::Graph& graph,
    RuntimeTensorDescs inputDescs);

} // namespace sandy::engine

