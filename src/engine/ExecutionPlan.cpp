#include "ExecutionPlan.h"

#include <set>

namespace sandy::engine {

namespace {

using ir::kernel_ir::LayoutTransformOp;
using ir::kernel_ir::Op;
using ir::kernel_ir::OpKind;
using ir::kernel_ir::ValueId;

bool belongs_to_device_executable(const Op& op) {
    return op.kind() != OpKind::Input &&
        op.kind() != OpKind::TensorTupleCreate &&
        op.kind() != OpKind::DeviceTransfer;
}

bool is_stable_import(const ir::kernel_ir::Graph& graph, ValueId value) {
    const auto& def = graph.value(value).def;
    if (def.op == ir::kernel_ir::kInvalidOpId)
        return false;
    const auto& producer = graph.op(def.op);
    if (producer.kind() != OpKind::Input)
        return false;
    const auto& input = static_cast<const ir::kernel_ir::InputOp&>(producer);
    return input.source().kind == ir::kernel_ir::InputSourceKind::Weight &&
        !graph.value(value).type.shape.has_dynamic();
}

} // namespace

Result<KernelExecutionPlan> partitionKernelGraph(
        const ir::kernel_ir::Graph& graph) {
    KernelExecutionPlan plan;
    plan.nodeForOp.assign(graph.ops().size(), -1);

    int32_t current = -1;
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (!belongs_to_device_executable(op)) {
            current = -1;
            continue;
        }
        if (current < 0 || plan.nodes[static_cast<size_t>(current)].device != op.device()) {
            current = static_cast<int32_t>(plan.nodes.size());
            plan.nodes.push_back(DeviceExecutableNodePlan{op.device(), {}});
        }
        plan.nodes[static_cast<size_t>(current)].executable.ops.push_back(op.id());
        plan.nodeForOp[op.id()] = current;
    }

    std::vector<std::set<ValueId>> imports(plan.nodes.size());
    std::vector<std::set<ValueId>> mutableImports(plan.nodes.size());
    std::vector<std::set<ValueId>> exports(plan.nodes.size());

    auto producerNode = [&](ValueId value) -> int32_t {
        const auto& def = graph.value(value).def;
        if (def.op == ir::kernel_ir::kInvalidOpId || def.op >= plan.nodeForOp.size())
            return -1;
        return plan.nodeForOp[def.op];
    };

    for (size_t nodeIndex = 0; nodeIndex < plan.nodes.size(); ++nodeIndex) {
        for (auto opId : plan.nodes[nodeIndex].executable.ops) {
            const auto& op = graph.op(opId);
            for (auto input : op.inputs()) {
                if (producerNode(input) != static_cast<int32_t>(nodeIndex))
                    imports[nodeIndex].insert(input);
            }
            if (op.kind() == OpKind::PagedAppend) {
                auto cache = static_cast<const ir::kernel_ir::PagedAppendOp&>(op).cache();
                mutableImports[nodeIndex].insert(cache);
                imports[nodeIndex].insert(cache);
            }
        }
    }

    for (const auto& value : graph.values()) {
        auto producer = producerNode(value.id);
        if (producer < 0)
            continue;
        for (const auto& use : value.uses) {
            auto consumer = use.op < plan.nodeForOp.size() ? plan.nodeForOp[use.op] : -1;
            if (consumer != producer)
                exports[static_cast<size_t>(producer)].insert(value.id);
        }
    }
    for (auto output : graph.outputs()) {
        auto producer = producerNode(output);
        if (producer >= 0)
            exports[static_cast<size_t>(producer)].insert(output);
    }

    // An exported alias cannot retain backing storage from private scratch.
    // Export its source recursively so the engine supplies standalone storage
    // for the backing value as well.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t nodeIndex = 0; nodeIndex < plan.nodes.size(); ++nodeIndex) {
            std::vector<ValueId> currentExports(exports[nodeIndex].begin(), exports[nodeIndex].end());
            for (auto value : currentExports) {
                const auto& def = graph.value(value).def;
                if (def.op == ir::kernel_ir::kInvalidOpId)
                    continue;
                const auto& op = graph.op(def.op);
                if (op.kind() != OpKind::LayoutTransform ||
                    !static_cast<const LayoutTransformOp&>(op).aliasesInput())
                    continue;
                auto source = op.inputs()[0];
                if (producerNode(source) == static_cast<int32_t>(nodeIndex)) {
                    changed |= exports[nodeIndex].insert(source).second;
                } else {
                    imports[nodeIndex].insert(source);
                }
            }
        }
    }

    for (size_t index = 0; index < plan.nodes.size(); ++index) {
        auto& desc = plan.nodes[index].executable;
        desc.imports.assign(imports[index].begin(), imports[index].end());
        for (auto value : desc.imports) {
            if (is_stable_import(graph, value))
                desc.stableImports.push_back(value);
        }
        desc.mutableImports.assign(mutableImports[index].begin(), mutableImports[index].end());
        desc.exports.assign(exports[index].begin(), exports[index].end());
    }
    return plan;
}

} // namespace sandy::engine
