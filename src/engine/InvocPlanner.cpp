#include "InvocPlanner.h"

namespace sandy::engine {

InvocPlanner::InvocPlanner(InvocDeviceId defaultDevice)
    : defaultDevice_(defaultDevice) {}

Result<InvocPlanDraft> InvocPlanner::plan(const ir::mid_ir::Graph&) {
    (void)defaultDevice_;
    return make_error("InvocPlanner is not implemented yet");
}

} // namespace sandy::engine
