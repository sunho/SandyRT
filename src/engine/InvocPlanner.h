#pragma once

#include "InvocPlan.h"
#include "Result.h"

#include <vector>

namespace sandy::ir::mid_ir {
class Graph;
struct Op;
} // namespace sandy::ir::mid_ir

namespace sandy::engine {

struct InvocProgramSource {
    InvocProgramId id = 0;
    InvocDeviceId device = 0;
    const ir::mid_ir::Op* op = nullptr;
};

struct InvocPlanDraft {
    std::vector<InvocProgramSource> programSources;
    std::vector<InvocInstruction> instructions;
    std::vector<InvocValueId> outputs;
};

class InvocPlanner {
public:
    explicit InvocPlanner(InvocDeviceId defaultDevice = 0);

    Result<InvocPlanDraft> plan(const ir::mid_ir::Graph& graph);

private:
    InvocDeviceId defaultDevice_;
};

} // namespace sandy::engine
