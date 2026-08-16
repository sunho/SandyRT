#include "MidIRMaterializer.h"

#include <memory>

namespace sandy::ir::mid_ir {

namespace {

Result<int64_t> infer_paged_tensor_grow_dim(const std::vector<int64_t>& dims) {
    int64_t growDim = -1;
    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] != core::Shape::kDynamic)
            continue;
        if (growDim >= 0)
            return make_error("PagedTensor input shape must have exactly one dynamic grow dimension");
        growDim = static_cast<int64_t>(i);
    }
    if (growDim < 0)
        return make_error("PagedTensor input shape must have one dynamic grow dimension");
    return growDim;
}

} // namespace

MidIRMaterializer::MidIRMaterializer()
    : lowering_(BuiltinLowering::createDefault()) {
    register_all_ops();
}

Result<std::unique_ptr<Graph>> MidIRMaterializer::materialize(
        const high_ir::Graph& graph,
        const weight::Weights& weights,
        const MaterializeOptions& options) {
    auto mid_graph = std::make_unique<Graph>();
    Builder builder(*mid_graph);

    std::unordered_map<int, Value*> value_map;
    int64_t nextInputIndex = 0;

    for (auto& op : graph.ops()) {
        switch (op.kind) {
            case high_ir::Op::Input: {
                auto it = options.input_tensor_descs.find(op.inputName);
                if (it == options.input_tensor_descs.end())
                    return make_error("no shape provided for input '" + op.inputName + "'");
                Value* v = nullptr;
                if (op.inputKind == high_ir::InputKind::PagedTensor) {
                    auto growDim = infer_paged_tensor_grow_dim(op.inputPagedTensorDims);
                    if (!growDim)
                        return make_error("input '" + op.inputName + "': " + growDim.error());
                    v = builder.createPagedTensorInput(
                        nextInputIndex++,
                        core::Shape(op.inputPagedTensorDims),
                        it->second.dtype,
                        growDim.take(),
                        op.inputPagedTensorPageSize);
                } else {
                    v = builder.createInput(nextInputIndex++, it->second.shape, it->second.dtype);
                }
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::Weight: {
                auto tensor = weights.get_tensor(op.weightName);
                if (!tensor)
                    return make_error("weight not found: " + op.weightName);
                const auto& desc = tensor->desc();
                auto* v = builder.createWeight(op.weightName, desc.shape, desc.dtype);
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::Builtin: {
                std::vector<Value*> operands;
                for (auto* hv : op.operands)
                    operands.push_back(value_map.at(hv->id));

                AttrMap attrs;
                for (auto& a : op.attrs) {
                    switch (a.type) {
                        case high_ir::Type::Int:
                            attrs[a.name] = AttrValue::make_int(a.intVal);
                            break;
                        case high_ir::Type::Float:
                            attrs[a.name] = AttrValue::make_float(a.floatVal);
                            break;
                        case high_ir::Type::String:
                            attrs[a.name] = AttrValue::make_string(a.strVal);
                            break;
                        case high_ir::Type::IntList:
                            attrs[a.name] = AttrValue::make_int_list(a.intListVal);
                            break;
                        default:
                            break;
                    }
                }

                auto* fn = lowering_.lookup(op.name);
                if (!fn)
                    return make_error("no lowering for builtin '" + op.name + "'");

                auto results = (*fn)(builder, operands, attrs, static_cast<int>(op.results.size()));
                if (!results)
                    return make_error(results.error());
                auto midResults = results.take();
                if (midResults.size() != op.results.size()) {
                    return make_error("lowering for builtin '" + op.name +
                                      "' returned wrong number of results");
                }
                for (size_t i = 0; i < midResults.size(); i++)
                    value_map[op.results[i]->id] = midResults[i];
                break;
            }
            case high_ir::Op::IntConst: {
                auto* v = builder.createConstantF32(static_cast<float>(op.intVal));
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::FloatConst: {
                auto* v = builder.createConstantF32(static_cast<float>(op.floatVal));
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::StringConst:
                return make_error("unexpected const op in materialization");
        }
    }

    std::vector<Value*> outputs;
    for (auto* hv : graph.outputs())
        outputs.push_back(value_map.at(hv->id));
    builder.setOutputs(outputs);

    return mid_graph;
}

} // namespace sandy::ir::mid_ir
