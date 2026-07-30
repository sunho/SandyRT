#pragma once

#include "BuiltinLowering.h"
#include "HighIR.h"
#include "MidIR.h"
#include "Result.h"
#include "Weight.h"

#include <unordered_map>

namespace sandy::ir::mid_ir {

struct MaterializeOptions {
    std::unordered_map<std::string, ir::TensorDesc> input_tensor_descs;
};

class MidIRMaterializer {
public:
    MidIRMaterializer();
    Result<Graph> materialize(const high_ir::Graph& graph,
                              const weight::Weights& weights,
                              const MaterializeOptions& options = {});

private:
    BuiltinLowering lowering_;
};

} // namespace sandy::ir::mid_ir
