#include "MidIRMaterializer.h"

#include <memory>

namespace sandy::ir::mid_ir {

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

    for (auto& op : graph.ops()) {
        switch (op.kind) {
            case high_ir::Op::Input: {
                auto it = options.input_tensor_descs.find(op.inputName);
                if (it == options.input_tensor_descs.end())
                    return make_error("no shape provided for input '" + op.inputName + "'");
                auto* v = builder.createInput(op.inputName, it->second.shape, it->second.dtype);
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
                        default:
                            break;
                    }
                }

                auto* fn = lowering_.lookup(op.name);
                if (!fn)
                    return make_error("no lowering for builtin '" + op.name + "'");

                auto results = (*fn)(builder, operands, attrs);
                for (size_t i = 0; i < results.size(); i++)
                    value_map[op.results[i]->id] = results[i];
                break;
            }
            case high_ir::Op::IntConst:
            case high_ir::Op::FloatConst:
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
