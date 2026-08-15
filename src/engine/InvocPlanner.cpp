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
        std::unordered_set<InvocValueId>& materializedValues) {
    materializedValues.insert(value);
}

} // namespace

InvocPlanner::InvocPlanner(InvocDeviceId defaultDevice)
    : defaultDevice_(defaultDevice) {}

Result<InvocPlanDraft> InvocPlanner::plan(const ir::mid_ir::Graph& graph) {
    InvocPlanDraft draft;
    std::unordered_map<const ir::mid_ir::Value*, InvocValueId> valueIds;
    std::unordered_set<InvocValueId> materializedValues;
    std::unordered_map<InvocValueId, int64_t> remainingUses;
    InvocValueId nextValueId = 0;

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
                remember_materialized(value, materializedValues);
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
                remember_materialized(value, materializedValues);
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
                for (auto* operand : op->operands) {
                    auto value = lookup_value(valueIds, operand);
                    if (!value)
                        return make_error(value.error());
                    remainingUses[value.take()]++;
                }

                for (auto* result : op->results) {
                    auto value = nextValueId++;
                    valueIds[result] = value;
                    remember_materialized(value, materializedValues);
                }
                break;
            }
        }
    }

    std::unordered_set<InvocValueId> outputValues;
    std::vector<core::TensorDesc> outputDescs;
    draft.outputs.clear();
    for (auto* output : graph.outputs()) {
        auto value = lookup_value(valueIds, output);
        if (!value)
            return make_error(value.error());
        auto outputValue = value.take();
        draft.outputs.push_back(outputValue);
        outputValues.insert(outputValue);
        outputDescs.emplace_back(output->shape, output->dtype);
    }

    std::unordered_set<InvocValueId> deallocatedValues;
    auto maybe_dealloc = [&](InvocValueId value) {
        if (materializedValues.find(value) == materializedValues.end())
            return;
        if (outputValues.find(value) != outputValues.end())
            return;
        if (deallocatedValues.find(value) != deallocatedValues.end())
            return;
        if (remainingUses[value] != 0)
            return;
        deallocatedValues.insert(value);
        draft.instructions.push_back(InvocInstruction::dealloc({
            defaultDevice_,
            value,
        }));
    };

    InvocProgramId nextProgramId = 0;
    for (const auto* op : graph.entry()->ops) {
        switch (op->kind) {
            case ir::mid_ir::OpKind::Input: {
                auto value = lookup_value(valueIds, op->results[0]);
                if (!value)
                    return make_error(value.error());
                auto index = get_int_attr(*op, "index");
                if (!index)
                    return make_error(index.error());

                auto inputValue = value.take();
                draft.instructions.push_back(InvocInstruction::load_input({
                    defaultDevice_,
                    index.take(),
                    inputValue,
                }));
                maybe_dealloc(inputValue);
                break;
            }
            case ir::mid_ir::OpKind::Weight: {
                auto value = lookup_value(valueIds, op->results[0]);
                if (!value)
                    return make_error(value.error());
                auto name = get_string_attr(*op, "name");
                if (!name)
                    return make_error(name.error());

                auto weightValue = value.take();
                draft.instructions.push_back(InvocInstruction::load_weight({
                    defaultDevice_,
                    name.take(),
                    weightValue,
                }));
                maybe_dealloc(weightValue);
                break;
            }
            case ir::mid_ir::OpKind::Reshape:
                break;
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
                    auto value = lookup_value(valueIds, result);
                    if (!value)
                        return make_error(value.error());
                    auto outputValue = value.take();
                    outputs.push_back(outputValue);
                    draft.instructions.push_back(InvocInstruction::alloc({
                        defaultDevice_,
                        outputValue,
                        core::TensorDesc(result->shape, result->dtype),
                    }));
                }

                auto program = nextProgramId++;
                draft.programSources.push_back({program, defaultDevice_, op});
                draft.instructions.push_back(InvocInstruction::run_kernel({
                    defaultDevice_,
                    program,
                    inputs,
                    outputs,
                }));

                for (auto input : inputs) {
                    remainingUses[input]--;
                    maybe_dealloc(input);
                }
                for (auto output : outputs)
                    maybe_dealloc(output);
                break;
            }
        }
    }

    if (!draft.outputs.empty()) {
        draft.instructions.push_back(InvocInstruction::store_outputs({
            defaultDevice_,
            draft.outputs,
            std::move(outputDescs),
        }));
    }

    return draft;
}

} // namespace sandy::engine
