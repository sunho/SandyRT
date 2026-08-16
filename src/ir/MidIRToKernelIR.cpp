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

Result<int64_t> attr_int_or(
    const mid_ir::AttrMap& attrs,
    const std::string& name,
    int64_t fallback)
{
    auto it = attrs.find(name);
    if (it == attrs.end())
        return fallback;
    if (it->second.kind != mid_ir::AttrValue::Int)
        return make_error("attr '" + name + "' must be int");
    return it->second.intVal;
}

Result<double> attr_float_or(
    const mid_ir::AttrMap& attrs,
    const std::string& name,
    double fallback)
{
    auto it = attrs.find(name);
    if (it == attrs.end())
        return fallback;
    if (it->second.kind != mid_ir::AttrValue::Float)
        return make_error("attr '" + name + "' must be float");
    return it->second.floatVal;
}

ScalarOp binary_scalar_op(mid_ir::OpKind kind) {
    switch (kind) {
        case mid_ir::OpKind::Add: return ScalarOp::Add;
        case mid_ir::OpKind::Mul: return ScalarOp::Mul;
        default: return ScalarOp::Constant;
    }
}

Result<std::vector<ValueId>> mapped_operands(
    const ValueMap& valueMap,
    const mid_ir::Op& op)
{
    std::vector<ValueId> operands;
    operands.reserve(op.operands.size());
    for (auto* operand : op.operands) {
        auto mapped = mapped_value(valueMap, operand);
        if (!mapped)
            return make_error(mapped.error());
        operands.push_back(mapped.take());
    }
    return operands;
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

Result<void> lower_paged_tensor_input(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto index = find_attr(op, "index", mid_ir::AttrValue::Int);
    if (!index)
        return make_error(index.error());
    auto growDim = find_attr(op, "grow_dim", mid_ir::AttrValue::Int);
    if (!growDim)
        return make_error(growDim.error());
    auto pageSize = find_attr(op, "page_size", mid_ir::AttrValue::Int);
    if (!pageSize)
        return make_error(pageSize.error());

    if (op.results.size() != 1) {
        return make_error("paged tensor input lowering expects one result");
    }

    auto* result = op.results[0];
    ValueType type{
        ValueKind::PagedTensor,
        result->dtype,
        result->shape,
        PagedTensorMeta{(*growDim)->intVal, (*pageSize)->intVal},
    };
    auto output = graph.addValue(std::move(type));
    valueMap[result] = output;

    InputSource source;
    source.kind = InputSourceKind::Argument;
    source.index = (*index)->intVal;
    graph.addOp<InputOp>(std::move(source), output);
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
    graph.addOp<ElementwiseKernelOp>(
        std::vector<ElementwiseInput>{},
        outputId,
        0,
        std::move(scalars));
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
    graph.addOp<ElementwiseKernelOp>(
        std::move(inputs),
        outputId,
        1,
        std::move(scalars));
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
    graph.addOp<ElementwiseKernelOp>(
        std::move(inputs),
        outputId,
        2,
        std::move(scalars));
    return {};
}

Result<void> lower_layout_transform(
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

    LayoutTransformKind kind = LayoutTransformKind::Contiguous;
    std::vector<int64_t> dims;
    switch (op.kind) {
        case mid_ir::OpKind::Reshape: {
            auto attr = find_attr(op, "shape", mid_ir::AttrValue::IntList);
            if (!attr)
                return make_error(attr.error());
            kind = LayoutTransformKind::Reshape;
            dims = (*attr)->intListVal;
            break;
        }
        case mid_ir::OpKind::Transpose:
            kind = LayoutTransformKind::Transpose;
            break;
        case mid_ir::OpKind::Permute: {
            auto attr = find_attr(op, "dims", mid_ir::AttrValue::IntList);
            if (!attr)
                return make_error(attr.error());
            kind = LayoutTransformKind::Permute;
            dims = (*attr)->intListVal;
            break;
        }
        default:
            return make_error("unsupported layout transform");
    }

    graph.addOp<LayoutTransformOp>(
        kind,
        input.take(),
        output.take(),
        std::move(dims));
    return {};
}

Result<void> lower_linear(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 3) {
        return make_error("linear lowering expects three operands");
    }
    auto inputs = mapped_operands(valueMap, op);
    if (!inputs)
        return make_error(inputs.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    graph.addOp<CustomKernelOp>(
        "linear",
        inputs.take(),
        std::vector<ValueId>{output.take()},
        op.attrs);
    return {};
}

Result<void> lower_matmul(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 2) {
        return make_error("matmul lowering expects two operands");
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

    auto transposeLhs = attr_int_or(op.attrs, "transpose_lhs", 0);
    if (!transposeLhs)
        return make_error(transposeLhs.error());
    auto transposeRhs = attr_int_or(op.attrs, "transpose_rhs", 0);
    if (!transposeRhs)
        return make_error(transposeRhs.error());

    graph.addOp<MatMulKernelOp>(
        lhs.take(),
        rhs.take(),
        output.take(),
        transposeLhs.take() != 0,
        transposeRhs.take() != 0);
    return {};
}

Result<void> lower_embedding(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 2) {
        return make_error("embedding lowering expects two operands");
    }
    auto ids = mapped_value(valueMap, op.operands[0]);
    if (!ids)
        return make_error(ids.error());
    auto table = mapped_value(valueMap, op.operands[1]);
    if (!table)
        return make_error(table.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());

    graph.addOp<GatherKernelOp>(ids.take(), table.take(), output.take());
    return {};
}

Result<void> lower_softmax(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 1) {
        return make_error("softmax lowering expects one operand");
    }
    auto input = mapped_value(valueMap, op.operands[0]);
    if (!input)
        return make_error(input.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    auto axis = attr_int_or(op.attrs, "dim", -1);
    if (!axis)
        return make_error(axis.error());

    graph.addOp<SoftmaxKernelOp>(input.take(), output.take(), axis.take());
    return {};
}

Result<void> lower_rope(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 1) {
        return make_error("rope lowering expects one operand");
    }
    auto input = mapped_value(valueMap, op.operands[0]);
    if (!input)
        return make_error(input.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    auto theta = attr_float_or(op.attrs, "rope_theta", 10000.0);
    if (!theta)
        return make_error(theta.error());
    auto rotaryDim = attr_int_or(op.attrs, "rotary_dim", -1);
    if (!rotaryDim)
        return make_error(rotaryDim.error());
    auto splitHalf = attr_int_or(op.attrs, "split_half", 0);
    if (!splitHalf)
        return make_error(splitHalf.error());

    graph.addOp<RoPEKernelOp>(
        input.take(),
        output.take(),
        theta.take(),
        rotaryDim.take(),
        splitHalf.take() != 0);
    return {};
}

Result<void> lower_norm(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto inputs = mapped_operands(valueMap, op);
    if (!inputs)
        return make_error(inputs.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    auto epsilon = attr_float_or(
        op.attrs,
        "epsilon",
        op.kind == mid_ir::OpKind::LayerNorm ? 1.0e-5 : 1.0e-6);
    if (!epsilon)
        return make_error(epsilon.error());

    graph.addOp<NormKernelOp>(
        op.kind == mid_ir::OpKind::LayerNorm ? NormKind::LayerNorm : NormKind::RMSNorm,
        inputs.take(),
        output.take(),
        epsilon.take());
    return {};
}

Result<void> lower_sliding_query_key_score(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 2) {
        return make_error("sliding_query_key_score lowering expects two operands");
    }
    auto query = mapped_value(valueMap, op.operands[0]);
    if (!query)
        return make_error(query.error());
    auto key = mapped_value(valueMap, op.operands[1]);
    if (!key)
        return make_error(key.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    auto window = attr_int_or(op.attrs, "window", 0);
    if (!window)
        return make_error(window.error());
    auto scale = attr_float_or(op.attrs, "scale", -1.0);
    if (!scale)
        return make_error(scale.error());

    graph.addOp<SlidingQueryKeyScoreKernelOp>(
        query.take(),
        key.take(),
        output.take(),
        window.take(),
        scale.take());
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
        case mid_ir::OpKind::PagedTensorInput:
            return lower_paged_tensor_input(graph, op, valueMap);
        case mid_ir::OpKind::Weight:
            return lower_weight(graph, op, valueMap);
        case mid_ir::OpKind::Constant:
            return lower_constant(graph, op, valueMap);
        case mid_ir::OpKind::Linear:
            return lower_linear(graph, op, valueMap);
        case mid_ir::OpKind::Add:
        case mid_ir::OpKind::Mul:
            return lower_binary_elementwise(graph, op, valueMap);
        case mid_ir::OpKind::ReLU:
        case mid_ir::OpKind::Sqrt:
        case mid_ir::OpKind::Tanh:
            return lower_unary_elementwise(graph, op, valueMap);
        case mid_ir::OpKind::MatMul:
            return lower_matmul(graph, op, valueMap);
        case mid_ir::OpKind::Transpose:
        case mid_ir::OpKind::Reshape:
        case mid_ir::OpKind::Permute:
            return lower_layout_transform(graph, op, valueMap);
        case mid_ir::OpKind::SlidingQueryKeyScore:
            return lower_sliding_query_key_score(graph, op, valueMap);
        case mid_ir::OpKind::Softmax:
            return lower_softmax(graph, op, valueMap);
        case mid_ir::OpKind::Embedding:
            return lower_embedding(graph, op, valueMap);
        case mid_ir::OpKind::RoPE:
            return lower_rope(graph, op, valueMap);
        case mid_ir::OpKind::RMSNorm:
        case mid_ir::OpKind::LayerNorm:
            return lower_norm(graph, op, valueMap);
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
