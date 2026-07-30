#pragma once

#include "HighIR.h"
#include "MidIR.h"
#include "Weight.h"

#include <string>

namespace sandy {

class Compiler {
public:
    high_ir::Graph load_sandygo(const std::string& path);
    mid_ir::Graph materialize_mid_ir(const high_ir::Graph& graph,
                                     const weight::Weights& weights);
};

} // namespace sandy
