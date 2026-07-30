#pragma once

#include "HighIR.h"
#include "MidIRMaterializer.h"
#include "Result.h"
#include "Weight.h"

#include <memory>
#include <string>

namespace sandy {

class Compiler {
public:
    ir::high_ir::Graph load_sandygo(const std::string& path);
    Result<std::unique_ptr<ir::mid_ir::Graph>> materialize_mid_ir(
        const ir::high_ir::Graph& graph,
        const weight::Weights& weights,
        const ir::mid_ir::MaterializeOptions& options = {});
};

} // namespace sandy
