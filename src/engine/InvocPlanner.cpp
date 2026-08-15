#include "InvocPlanner.h"

#include "MidIR.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace sandy::engine {

namespace {

Result<int64_t> get_int_attr(const ir::mid_ir::Op& op, const std::string& name) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end() || it->second.kind != ir::mid_ir::AttrValue::Int)
        return make_error("missing int attr '" + name + "'");
    return it->second.intVal;
}

Result<std::string> get_string_attr(const ir::mid_ir::Op& op, const std::string& name) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end() || it->second.kind != ir::mid_ir::AttrValue::String)
        return make_error("missing string attr '" + name + "'");
    return it->second.strVal;
}

Result<InvocValueId> lookup_value(
        const std::unordered_map<const ir::mid_ir::Value*, InvocValueId>& valueIds,
        const ir::mid_ir::Value* value) {
    auto it = valueIds.find(value);
    if (it == valueIds.end())
        return make_error("value is used before it is planned");
    return it->second;
}

void remember_materialized(
        InvocValueId value,
        std::unordered_set<InvocValueId>& materializedValues,
        std::vector<InvocValueId>& materializedOrder) {
    if (materializedValues.insert(value).second)
        materializedOrder.push_back(value);
}

} // namespace

InvocPlanner::InvocPlanner(InvocDeviceId defaultDevice)
    : defaultDevice_(defaultDevice) {}

Result<InvocPlanDraft> InvocPlanner::plan(const ir::mid_ir::Graph& graph) {
    InvocPlanDraft draft;
    std::unordered_map<const ir::mid_ir::Value*, InvocValueId> valueIds;
    std::unordered_set<InvocValueId> materializedValues;
    std::vector<InvocValueId> materializedOrder;
    InvocValueId nextValueId = 0;
    InvocProgramId nextProgramId = 0;

    for (const auto* op : graph.entry()->ops) {
        switch (op->kind) {
            case ir::mid_ir::OpKind::Input: {
                if (op->results.size() != 1)
                    return make_error("input op must have exactly one result");
                auto index = get_int_attr(*op, "index");
                if (!index)
                    return make_error(index.error());

                auto value = nextValueId++;
                valueIds[op->results[0]] = value;
                remember_materialized(value, materializedValues, materializedOrder);
                draft.instructions.push_back(InvocInstruction::load_input({
                    defaultDevice_,
                    index.take(),
                    value,
                }));
                break;
            }
            case ir::mid_ir::OpKind::Weight: {
                if (op->results.size() != 1)
                    return make_error("weight op must have exactly one result");
                auto name = get_string_attr(*op, "name");
                if (!name)
                    return make_error(name.error());

                auto value = nextValueId++;
                valueIds[op->results[0]] = value;
                remember_materialized(value, materializedValues, materializedOrder);
                draft.instructions.push_back(InvocInstruction::load_weight({
                    defaultDevice_,
                    name.take(),
                    value,
                }));
                break;
            }
            case ir::mid_ir::OpKind::Reshape: {
                if (op->operands.size() != 1 || op->results.size() != 1)
                    return make_error("reshape op must have one operand and one result");
                auto operandValue = lookup_value(valueIds, op->operands[0]);
                if (!operandValue)
                    return make_error(operandValue.error());
                valueIds[op->results[0]] = operandValue.take();
                break;
            }
            case ir::mid_ir::OpKind::NUM_KINDS:
                return make_error("invalid MidIR op kind");
            default: {
                std::vector<InvocValueId> inputs;
                inputs.reserve(op->operands.size());
                for (auto* operand : op->operands) {
                    auto value = lookup_value(valueIds, operand);
                    if (!value)
                        return make_error(value.error());
                    inputs.push_back(value.take());
                }

                std::vector<InvocValueId> outputs;
                outputs.reserve(op->results.size());
                for (auto* result : op->results) {
                    auto value = nextValueId++;
                    valueIds[result] = value;
                    remember_materialized(value, materializedValues, materializedOrder);
                    outputs.push_back(value);
                    draft.instructions.push_back(InvocInstruction::alloc({
                        defaultDevice_,
                        value,
                        core::TensorDesc(result->shape, result->dtype),
                    }));
                }

                auto program = nextProgramId++;
                draft.programSources.push_back({program, defaultDevice_, op});
                draft.instructions.push_back(InvocInstruction::run_kernel({
                    defaultDevice_,
                    program,
                    std::move(inputs),
                    std::move(outputs),
                }));
                break;
            }
        }
    }

    std::unordered_set<InvocValueId> outputValues;
    for (auto* output : graph.outputs()) {
        auto value = lookup_value(valueIds, output);
        if (!value)
            return make_error(value.error());
        auto outputValue = value.take();
        draft.outputs.push_back(outputValue);
        outputValues.insert(outputValue);
    }

    for (auto value : materializedOrder) {
        if (outputValues.find(value) != outputValues.end())
            continue;
        draft.instructions.push_back(InvocInstruction::dealloc({
            defaultDevice_,
            value,
        }));
    }

    if (!draft.outputs.empty()) {
        draft.instructions.push_back(InvocInstruction::store_outputs({
            defaultDevice_,
            draft.outputs,
        }));
    }

    return draft;
}

} // namespace sandy::engine
