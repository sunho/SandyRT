#include "MidIRToKernelIR.h"

#include <unordered_map>

namespace sandy::ir::kernel_ir {

namespace {

using ValueMap = std::unordered_map<const mid_ir::Value*, ValueId>;

ValueType value_type_from_mid(const mid_ir::Value& value) {
    return ValueType{
        ValueKind::Tensor,
        value.dtype,
        value.shape,
    };
}

Result<const mid_ir::AttrValue*> find_attr(
    const mid_ir::Op& op,
    const char* name,
    mid_ir::AttrValue::Kind kind)
{
    auto it = op.attrs.find(name);
    if (it == op.attrs.end()) {
        return make_error(std::string(mid_ir::op_kind_name(op.kind)) +
                          " missing attr '" + name + "'");
    }
    if (it->second.kind != kind) {
        return make_error(std::string(mid_ir::op_kind_name(op.kind)) +
                          " attr '" + name + "' has unexpected kind");
    }
    return &it->second;
}

Result<ValueId> mapped_value(
    const ValueMap& valueMap,
    const mid_ir::Value* value)
{
    auto it = valueMap.find(value);
    if (it == valueMap.end()) {
        return make_error("MidIR value %" + std::to_string(value->id) +
                          " has not been lowered");
    }
    return it->second;
}

Result<ValueId> add_single_result_value(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.results.size() != 1) {
        return make_error(std::string(mid_ir::op_kind_name(op.kind)) +
                          " lowering expects one result");
    }

    auto* result = op.results[0];
    auto id = graph.addValue(value_type_from_mid(*result));
    valueMap[result] = id;
    return id;
}

BroadcastMode broadcast_mode(
    const mid_ir::Value& input,
    const mid_ir::Value& output)
{
    return input.shape == output.shape ? BroadcastMode::None
                                       : BroadcastMode::RightAligned;
}

ScalarOp unary_scalar_op(mid_ir::OpKind kind) {
    switch (kind) {
        case mid_ir::OpKind::ReLU: return ScalarOp::ReLU;
        case mid_ir::OpKind::Sqrt: return ScalarOp::Sqrt;
        case mid_ir::OpKind::Tanh: return ScalarOp::Tanh;
        default: return ScalarOp::Constant;
    }
}

ScalarOp binary_scalar_op(mid_ir::OpKind kind) {
    switch (kind) {
        case mid_ir::OpKind::Add: return ScalarOp::Add;
        case mid_ir::OpKind::Mul: return ScalarOp::Mul;
        default: return ScalarOp::Constant;
    }
}

Result<void> lower_input(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto attr = find_attr(op, "index", mid_ir::AttrValue::Int);
    if (!attr)
        return make_error(attr.error());

    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    InputSource source;
    source.kind = InputSourceKind::Argument;
    source.index = (*attr)->intVal;
    graph.addOp<InputOp>(std::move(source), output.take());
    return {};
}

Result<void> lower_weight(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto attr = find_attr(op, "name", mid_ir::AttrValue::String);
    if (!attr)
        return make_error(attr.error());

    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    InputSource source;
    source.kind = InputSourceKind::Weight;
    source.name = (*attr)->strVal;
    graph.addOp<InputOp>(std::move(source), output.take());
    return {};
}

Result<void> lower_constant(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto attr = find_attr(op, "value", mid_ir::AttrValue::Float);
    if (!attr)
        return make_error(attr.error());

    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    auto outputId = output.take();
    std::vector<ScalarNode> scalars = {
        ScalarNode{
            0,
            ScalarOp::Constant,
            op.results[0]->dtype,
            0,
            (*attr)->floatVal,
            {},
        },
    };
    std::vector<ElementwiseStore> stores = {
        ElementwiseStore{outputId, 0},
    };
    graph.addOp<ElementwiseKernelOp>(
        std::vector<ElementwiseInput>{},
        std::vector<ValueId>{outputId},
        outputId,
        std::move(scalars),
        std::move(stores));
    return {};
}

Result<void> lower_unary_elementwise(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 1) {
        return make_error(std::string(mid_ir::op_kind_name(op.kind)) +
                          " lowering expects one operand");
    }

    auto input = mapped_value(valueMap, op.operands[0]);
    if (!input)
        return make_error(input.error());

    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    auto inputId = input.take();
    auto outputId = output.take();
    std::vector<ElementwiseInput> inputs = {
        ElementwiseInput{inputId, BroadcastMode::None},
    };
    std::vector<ScalarNode> scalars = {
        ScalarNode{0, ScalarOp::Load, op.operands[0]->dtype, 0, 0.0, {}},
        ScalarNode{1, unary_scalar_op(op.kind), op.results[0]->dtype, 0, 0.0, {0}},
    };
    std::vector<ElementwiseStore> stores = {
        ElementwiseStore{outputId, 1},
    };
    graph.addOp<ElementwiseKernelOp>(
        std::move(inputs),
        std::vector<ValueId>{outputId},
        outputId,
        std::move(scalars),
        std::move(stores));
    return {};
}

Result<void> lower_binary_elementwise(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 2) {
        return make_error(std::string(mid_ir::op_kind_name(op.kind)) +
                          " lowering expects two operands");
    }

    auto lhs = mapped_value(valueMap, op.operands[0]);
    if (!lhs)
        return make_error(lhs.error());
    auto rhs = mapped_value(valueMap, op.operands[1]);
    if (!rhs)
        return make_error(rhs.error());

    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    auto outputId = output.take();
    std::vector<ElementwiseInput> inputs = {
        ElementwiseInput{
            lhs.take(),
            broadcast_mode(*op.operands[0], *op.results[0]),
        },
        ElementwiseInput{
            rhs.take(),
            broadcast_mode(*op.operands[1], *op.results[0]),
        },
    };
    std::vector<ScalarNode> scalars = {
        ScalarNode{0, ScalarOp::Load, op.operands[0]->dtype, 0, 0.0, {}},
        ScalarNode{1, ScalarOp::Load, op.operands[1]->dtype, 1, 0.0, {}},
        ScalarNode{2, binary_scalar_op(op.kind), op.results[0]->dtype, 0, 0.0, {0, 1}},
    };
    std::vector<ElementwiseStore> stores = {
        ElementwiseStore{outputId, 2},
    };
    graph.addOp<ElementwiseKernelOp>(
        std::move(inputs),
        std::vector<ValueId>{outputId},
        outputId,
        std::move(scalars),
        std::move(stores));
    return {};
}

Result<void> lower_op(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    switch (op.kind) {
        case mid_ir::OpKind::Input:
            return lower_input(graph, op, valueMap);
        case mid_ir::OpKind::Weight:
            return lower_weight(graph, op, valueMap);
        case mid_ir::OpKind::Constant:
            return lower_constant(graph, op, valueMap);
        case mid_ir::OpKind::Add:
        case mid_ir::OpKind::Mul:
            return lower_binary_elementwise(graph, op, valueMap);
        case mid_ir::OpKind::ReLU:
        case mid_ir::OpKind::Sqrt:
        case mid_ir::OpKind::Tanh:
            return lower_unary_elementwise(graph, op, valueMap);
        default:
            return make_error(std::string("unsupported MidIR op for KernelIR lowering: ") +
                              mid_ir::op_kind_name(op.kind));
    }
}

} // namespace

Result<std::unique_ptr<Graph>> MidIRToKernelIRLowering::lower(
    const mid_ir::Graph& midGraph)
{
    auto graph = std::make_unique<Graph>();
    ValueMap valueMap;

    for (const auto* op : midGraph.entry()->ops) {
        auto result = lower_op(*graph, *op, valueMap);
        if (!result)
            return make_error(result.error());
    }

    std::vector<ValueId> outputs;
    outputs.reserve(midGraph.outputs().size());
    for (const auto* output : midGraph.outputs()) {
        auto mapped = mapped_value(valueMap, output);
        if (!mapped)
            return make_error(mapped.error());
        outputs.push_back(mapped.take());
    }
    graph->setOutputs(std::move(outputs));

    auto verifyResult = graph->verify();
    if (!verifyResult)
        return make_error(verifyResult.error());

    return graph;
}

Result<std::unique_ptr<Graph>> lowerMidIRToKernelIR(const mid_ir::Graph& graph) {
    MidIRToKernelIRLowering lowering;
    return lowering.lower(graph);
}

} // namespace sandy::ir::kernel_ir
