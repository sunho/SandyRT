#include "MidIRToKernelIR.h"
#include "ShapeUtil.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sandy::ir::kernel_ir {

namespace {

using ValueMap = std::unordered_map<const mid_ir::Value*, ValueId>;

enum class FusionKind {
    Elementwise,
    Attention,
};

using FusionId = uint32_t;

struct FusionGroup {
    FusionId id = 0;
    FusionKind kind = FusionKind::Elementwise;
    const mid_ir::Op* root = nullptr;
    std::vector<const mid_ir::Op*> ops;
    std::vector<const mid_ir::Value*> inputs;
    std::vector<const mid_ir::Value*> outputs;
};

struct FusionPlan {
    std::vector<FusionGroup> groups;
    std::unordered_map<const mid_ir::Op*, FusionId> owner;
    std::unordered_map<const mid_ir::Op*, FusionId> rootOwner;

    bool isClaimed(const mid_ir::Op* op) const {
        return owner.find(op) != owner.end();
    }

    const FusionGroup* groupForRoot(const mid_ir::Op* op) const {
        auto it = rootOwner.find(op);
        if (it == rootOwner.end())
            return nullptr;
        return &groups[static_cast<size_t>(it->second)];
    }

    void add(FusionGroup group) {
        auto id = static_cast<FusionId>(groups.size());
        group.id = id;
        for (auto* op : group.ops)
            owner[op] = id;
        if (group.root)
            rootOwner[group.root] = id;
        groups.push_back(std::move(group));
    }
};

ValueType value_type_from_mid(const mid_ir::Value& value) {
    ValueType type;
    switch (value.kind) {
        case mid_ir::ValueKind::Tensor:
            type.kind = ValueKind::Tensor;
            break;
        case mid_ir::ValueKind::PagedTensor:
            type.kind = ValueKind::PagedTensor;
            type.paged = PagedTensorMeta{value.growDim, value.pageSize};
            break;
        case mid_ir::ValueKind::TensorTuple:
            type.kind = ValueKind::TensorTuple;
            break;
    }
    type.dtype = value.dtype;
    type.shape = value.shape;
    type.elements.reserve(value.elements.size());
    for (const auto& element : value.elements) {
        ValueType e;
        switch (element.kind) {
            case mid_ir::ValueKind::Tensor:
                e.kind = ValueKind::Tensor;
                break;
            case mid_ir::ValueKind::PagedTensor:
                e.kind = ValueKind::PagedTensor;
                e.paged = PagedTensorMeta{element.growDim, element.pageSize};
                break;
            case mid_ir::ValueKind::TensorTuple:
                e.kind = ValueKind::TensorTuple;
                break;
        }
        e.dtype = element.dtype;
        e.shape = element.shape;
        type.elements.push_back(std::move(e));
    }
    return type;
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
        case mid_ir::OpKind::Div: return ScalarOp::Div;
        default: return ScalarOp::Constant;
    }
}

bool is_graph_output(const mid_ir::Graph& graph, const mid_ir::Value* value) {
    const auto& outputs = graph.outputs();
    return std::find(outputs.begin(), outputs.end(), value) != outputs.end();
}

bool is_fusible_elementwise(const mid_ir::Op& op) {
    switch (op.kind) {
        case mid_ir::OpKind::Constant:
        case mid_ir::OpKind::Add:
        case mid_ir::OpKind::Mul:
        case mid_ir::OpKind::Div:
        case mid_ir::OpKind::ReLU:
        case mid_ir::OpKind::Sqrt:
        case mid_ir::OpKind::Tanh:
            return true;
        default:
            return false;
    }
}

bool is_unary_elementwise(mid_ir::OpKind kind) {
    switch (kind) {
        case mid_ir::OpKind::ReLU:
        case mid_ir::OpKind::Sqrt:
        case mid_ir::OpKind::Tanh:
            return true;
        default:
            return false;
    }
}

bool is_binary_elementwise(mid_ir::OpKind kind) {
    switch (kind) {
        case mid_ir::OpKind::Add:
        case mid_ir::OpKind::Mul:
        case mid_ir::OpKind::Div:
            return true;
        default:
            return false;
    }
}

bool elementwise_op_shape_is_supported(const mid_ir::Op& op) {
    if (op.results.size() != 1)
        return false;
    if (op.kind == mid_ir::OpKind::Constant)
        return op.operands.empty();
    if (is_unary_elementwise(op.kind))
        return op.operands.size() == 1;
    if (is_binary_elementwise(op.kind))
        return op.operands.size() == 2;
    return false;
}

bool has_single_use(const mid_ir::Value& value) {
    return value.uses.size() == 1;
}

bool can_broadcast_to(const mid_ir::Value& input, const mid_ir::Value& output) {
    if (input.kind != mid_ir::ValueKind::Tensor ||
        output.kind != mid_ir::ValueKind::Tensor) {
        return false;
    }
    auto shape = core::broadcast_shape(input.shape, output.shape);
    return shape && *shape == output.shape;
}

class FusionPattern {
public:
    virtual ~FusionPattern() = default;
    virtual int priority() const = 0;
    virtual bool enabled(const LoweringOptions& options) const = 0;
    virtual std::optional<FusionGroup> match(
        const mid_ir::Graph& graph,
        const mid_ir::Op& root,
        const FusionPlan& plan,
        const LoweringOptions& options) const = 0;
};

class AttentionFusionPattern final : public FusionPattern {
public:
    int priority() const override { return 100; }

    bool enabled(const LoweringOptions& options) const override {
        return options.fusor.attention;
    }

    std::optional<FusionGroup> match(
        const mid_ir::Graph&,
        const mid_ir::Op& root,
        const FusionPlan& plan,
        const LoweringOptions&) const override
    {
        if (plan.isClaimed(&root) || root.kind != mid_ir::OpKind::Attention)
            return std::nullopt;
        if (root.results.size() != 1 ||
            (root.operands.size() != 3 && root.operands.size() != 4)) {
            return std::nullopt;
        }

        FusionGroup group;
        group.kind = FusionKind::Attention;
        group.root = &root;
        group.ops = {&root};
        group.inputs.assign(root.operands.begin(), root.operands.end());
        group.outputs.assign(root.results.begin(), root.results.end());
        return group;
    }
};

class ElementwiseFusionPattern final : public FusionPattern {
public:
    int priority() const override { return 10; }

    bool enabled(const LoweringOptions& options) const override {
        return options.fusor.elementwise;
    }

    std::optional<FusionGroup> match(
        const mid_ir::Graph& graph,
        const mid_ir::Op& root,
        const FusionPlan& plan,
        const LoweringOptions& options) const override
    {
        if (plan.isClaimed(&root) || !is_fusible_elementwise(root) ||
            !elementwise_op_shape_is_supported(root)) {
            return std::nullopt;
        }

        ElementwiseCollector collector(graph, plan, root);
        collector.collect(root);

        auto group = collector.take();
        if (group.inputs.size() > options.fusor.maxElementwiseInputs)
            return std::nullopt;
        auto scalarCount = group.inputs.size() + group.ops.size();
        if (scalarCount > options.fusor.maxElementwiseScalars)
            return std::nullopt;
        return group;
    }

private:
    class ElementwiseCollector {
    public:
        ElementwiseCollector(
            const mid_ir::Graph& graph,
            const FusionPlan& plan,
            const mid_ir::Op& root)
            : graph_(graph), plan_(plan)
        {
            group_.kind = FusionKind::Elementwise;
            group_.root = &root;
            group_.outputs.assign(root.results.begin(), root.results.end());
        }

        void collect(const mid_ir::Op& op) {
            if (!visitedOps_.insert(&op).second)
                return;

            for (auto* operand : op.operands) {
                auto* producer = operand ? operand->def : nullptr;
                if (canInlineProducer(operand, producer)) {
                    collect(*producer);
                } else {
                    addInput(operand);
                }
            }

            group_.ops.push_back(&op);
        }

        FusionGroup take() { return std::move(group_); }

    private:
        bool canInlineProducer(
            const mid_ir::Value* value,
            const mid_ir::Op* producer) const
        {
            if (!value || !producer)
                return false;
            if (plan_.isClaimed(producer))
                return false;
            if (!is_fusible_elementwise(*producer) ||
                !elementwise_op_shape_is_supported(*producer)) {
                return false;
            }
            if (producer->def && producer->def->has_side_effects())
                return false;
            if (!has_single_use(*value) || is_graph_output(graph_, value))
                return false;
            return can_broadcast_to(*value, *group_.outputs[0]);
        }

        void addInput(const mid_ir::Value* value) {
            if (!value || !seenInputs_.insert(value).second)
                return;
            group_.inputs.push_back(value);
        }

        const mid_ir::Graph& graph_;
        const FusionPlan& plan_;
        FusionGroup group_;
        std::unordered_set<const mid_ir::Op*> visitedOps_;
        std::unordered_set<const mid_ir::Value*> seenInputs_;
    };
};

FusionPlan build_fusion_plan(
    const mid_ir::Graph& graph,
    const LoweringOptions& options)
{
    FusionPlan plan;
    std::vector<std::unique_ptr<FusionPattern>> patterns;
    patterns.push_back(std::make_unique<AttentionFusionPattern>());
    patterns.push_back(std::make_unique<ElementwiseFusionPattern>());
    std::sort(
        patterns.begin(),
        patterns.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs->priority() > rhs->priority();
        });

    const auto& ops = graph.entry()->ops;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const auto* op = *it;
        if (!op || plan.isClaimed(op))
            continue;

        for (const auto& pattern : patterns) {
            if (!pattern->enabled(options))
                continue;
            auto group = pattern->match(graph, *op, plan, options);
            if (!group)
                continue;
            plan.add(std::move(*group));
            break;
        }
    }

    return plan;
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
    auto tupleElement = op.attrs.find("tuple_element");
    if (tupleElement != op.attrs.end()) {
        if (tupleElement->second.kind != mid_ir::AttrValue::Int)
            return make_error("input tuple_element attr must be int");
        source.tupleElement = tupleElement->second.intVal;
    }
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
    auto tupleElement = op.attrs.find("tuple_element");
    if (tupleElement != op.attrs.end()) {
        if (tupleElement->second.kind != mid_ir::AttrValue::Int)
            return make_error("paged tensor input tuple_element attr must be int");
        source.tupleElement = tupleElement->second.intVal;
    }
    graph.addOp<InputOp>(std::move(source), output);
    return {};
}

Result<void> lower_tensor_tuple_create(
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
    graph.addOp<TensorTupleCreateOp>(inputs.take(), output.take());
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
    std::vector<int64_t> indices;
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
        case mid_ir::OpKind::Slice: {
            auto kindsAttr = find_attr(op, "kinds", mid_ir::AttrValue::IntList);
            if (!kindsAttr)
                return make_error(kindsAttr.error());
            auto indicesAttr = find_attr(op, "indices", mid_ir::AttrValue::IntList);
            if (!indicesAttr)
                return make_error(indicesAttr.error());
            kind = LayoutTransformKind::Slice;
            dims = (*kindsAttr)->intListVal;
            indices = (*indicesAttr)->intListVal;
            break;
        }
        default:
            return make_error("unsupported layout transform");
    }

    graph.addOp<LayoutTransformOp>(
        kind,
        input.take(),
        output.take(),
        std::move(dims),
        std::move(indices));
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

    auto inputValues = inputs.take();
    graph.addOp<LinearKernelOp>(
        inputValues[0],
        inputValues[1],
        inputValues[2],
        output.take());
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

Result<void> lower_topk(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 1 || op.results.size() != 2)
        return make_error("topk lowering expects one operand and two results");
    auto input = mapped_value(valueMap, op.operands[0]);
    if (!input)
        return make_error(input.error());

    auto values = graph.addValue(value_type_from_mid(*op.results[0]));
    auto indices = graph.addValue(value_type_from_mid(*op.results[1]));
    valueMap[op.results[0]] = values;
    valueMap[op.results[1]] = indices;

    auto k = attr_int_or(op.attrs, "k", 0);
    if (!k)
        return make_error(k.error());
    auto axis = attr_int_or(op.attrs, "dim", -1);
    if (!axis)
        return make_error(axis.error());
    graph.addOp<TopKKernelOp>(input.take(), values, indices, k.take(), axis.take());
    return {};
}

Result<void> lower_sum(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 1)
        return make_error("sum lowering expects one operand");
    auto input = mapped_value(valueMap, op.operands[0]);
    if (!input)
        return make_error(input.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    auto axis = attr_int_or(op.attrs, "dim", -1);
    if (!axis)
        return make_error(axis.error());
    bool keepDims = false;
    auto keepDim = op.attrs.find("keepdim");
    if (keepDim == op.attrs.end())
        keepDim = op.attrs.find("keepdims");
    if (keepDim != op.attrs.end()) {
        if (keepDim->second.kind != mid_ir::AttrValue::Int)
            return make_error("sum keepdim attr must be int");
        keepDims = keepDim->second.intVal != 0;
    }

    graph.addOp<ReductionKernelOp>(
        ReduceOp::Sum,
        input.take(),
        output.take(),
        std::vector<int64_t>{axis.take()},
        keepDims);
    return {};
}

Result<void> lower_moe_gather(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto inputs = mapped_operands(valueMap, op);
    if (!inputs)
        return make_error(inputs.error());
    auto inputValues = inputs.take();
    if (inputValues.size() != 3 || op.results.size() != 4)
        return make_error("moe_gather lowering expects three operands and four results");
    std::vector<ValueId> outputs;
    outputs.reserve(op.results.size());
    for (auto* result : op.results) {
        auto id = graph.addValue(value_type_from_mid(*result));
        valueMap[result] = id;
        outputs.push_back(id);
    }
    auto numExperts = attr_int_or(op.attrs, "num_experts", 0);
    if (!numExperts)
        return make_error(numExperts.error());
    auto topK = attr_int_or(op.attrs, "top_k", 0);
    if (!topK)
        return make_error(topK.error());
    graph.addOp<MoeGatherKernelOp>(
        inputValues[0],
        inputValues[1],
        inputValues[2],
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        numExperts.take(),
        topK.take());
    return {};
}

Result<void> lower_moe_matmul(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto inputs = mapped_operands(valueMap, op);
    if (!inputs)
        return make_error(inputs.error());
    auto inputValues = inputs.take();
    if (inputValues.size() != 3)
        return make_error("moe_matmul lowering expects three operands");
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    bool transposeRhs = false;
    auto transpose = op.attrs.find("transpose_rhs");
    if (transpose != op.attrs.end()) {
        if (transpose->second.kind != mid_ir::AttrValue::Int)
            return make_error("moe_matmul transpose_rhs attr must be int");
        transposeRhs = transpose->second.intVal != 0;
    }
    graph.addOp<MoeMatMulKernelOp>(
        inputValues[0],
        inputValues[1],
        inputValues[2],
        output.take(),
        transposeRhs);
    return {};
}

Result<void> lower_moe_scatter_sum(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    auto inputs = mapped_operands(valueMap, op);
    if (!inputs)
        return make_error(inputs.error());
    auto inputValues = inputs.take();
    if (inputValues.size() != 4)
        return make_error("moe_scatter_sum lowering expects four operands");
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    graph.addOp<MoeScatterSumKernelOp>(
        inputValues[0],
        inputValues[1],
        inputValues[2],
        inputValues[3],
        output.take());
    return {};
}

Result<void> lower_paged_append(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 2) {
        return make_error("paged_append lowering expects two operands");
    }
    auto cache = mapped_value(valueMap, op.operands[0]);
    if (!cache)
        return make_error(cache.error());
    auto chunk = mapped_value(valueMap, op.operands[1]);
    if (!chunk)
        return make_error(chunk.error());

    graph.addOp<PagedAppendOp>(cache.take(), chunk.take());
    return {};
}

Result<void> lower_rope(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap)
{
    if (op.operands.size() != 1 && op.operands.size() != 2) {
        return make_error("rope lowering expects one or two operands");
    }
    auto inputs = mapped_operands(valueMap, op);
    if (!inputs)
        return make_error(inputs.error());
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
        inputs.take(),
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
    if (op.operands.size() != 2 && op.operands.size() != 3) {
        return make_error("sliding_query_key_score lowering expects two or three operands");
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

    if (op.operands.size() == 3) {
        auto positionIds = mapped_value(valueMap, op.operands[2]);
        if (!positionIds)
            return make_error(positionIds.error());
        graph.addOp<SlidingQueryKeyScoreKernelOp>(
            query.take(),
            key.take(),
            positionIds.take(),
            output.take(),
            window.take(),
            scale.take());
    } else {
        graph.addOp<SlidingQueryKeyScoreKernelOp>(
            query.take(),
            key.take(),
            output.take(),
            window.take(),
            scale.take());
    }
    return {};
}

Result<void> lower_attention(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap,
    bool fuseAttention)
{
    if (op.operands.size() != 3 && op.operands.size() != 4) {
        return make_error("attention lowering expects three or four operands");
    }
    auto query = mapped_value(valueMap, op.operands[0]);
    if (!query)
        return make_error(query.error());
    auto key = mapped_value(valueMap, op.operands[1]);
    if (!key)
        return make_error(key.error());
    auto value = mapped_value(valueMap, op.operands[2]);
    if (!value)
        return make_error(value.error());
    auto output = add_single_result_value(graph, op, valueMap);
    if (!output)
        return make_error(output.error());
    auto window = attr_int_or(op.attrs, "window", 0);
    if (!window)
        return make_error(window.error());
    auto scale = attr_float_or(op.attrs, "scale", -1.0);
    if (!scale)
        return make_error(scale.error());

    if (fuseAttention) {
        if (op.operands.size() == 4) {
            auto positionOffsets = mapped_value(valueMap, op.operands[3]);
            if (!positionOffsets)
                return make_error(positionOffsets.error());
            graph.addOp<AttentionKernelOp>(
                query.take(),
                key.take(),
                value.take(),
                positionOffsets.take(),
                output.take(),
                window.take(),
                scale.take());
        } else {
            graph.addOp<AttentionKernelOp>(
                query.take(),
                key.take(),
                value.take(),
                output.take(),
                window.take(),
                scale.take());
        }
        return {};
    }

    const auto& qShape = op.operands[0]->shape;
    const auto& kShape = op.operands[1]->shape;
    int rank = qShape.rank();
    std::vector<int64_t> scoreDims;
    if (rank == 4)
        scoreDims.push_back(qShape.dim(0));
    scoreDims.push_back(qShape.dim(rank - 3));
    scoreDims.push_back(qShape.dim(rank - 2));
    scoreDims.push_back(kShape.dim(rank - 2));

    auto scores = graph.addValue(ValueType{
        ValueKind::Tensor,
        op.operands[0]->dtype,
        core::Shape(scoreDims),
    });
    auto probs = graph.addValue(ValueType{
        ValueKind::Tensor,
        op.operands[0]->dtype,
        core::Shape(std::move(scoreDims)),
    });

    if (op.operands.size() == 4) {
        auto positionOffsets = mapped_value(valueMap, op.operands[3]);
        if (!positionOffsets)
            return make_error(positionOffsets.error());
        graph.addOp<SlidingQueryKeyScoreKernelOp>(
            query.take(),
            key.take(),
            positionOffsets.take(),
            scores,
            window.take(),
            scale.take());
    } else {
        graph.addOp<SlidingQueryKeyScoreKernelOp>(
            query.take(),
            key.take(),
            scores,
            window.take(),
            scale.take());
    }
    graph.addOp<SoftmaxKernelOp>(scores, probs, -1);
    graph.addOp<MatMulKernelOp>(
        probs,
        value.take(),
        output.take(),
        false,
        false);
    return {};
}

Result<void> lower_elementwise_fusion_group(
    Graph& graph,
    const FusionGroup& group,
    ValueMap& valueMap)
{
    if (!group.root || group.root->results.size() != 1) {
        return make_error("elementwise fusion group expects one root result");
    }

    auto* rootOutput = group.root->results[0];
    auto output = add_single_result_value(graph, *group.root, valueMap);
    if (!output)
        return make_error(output.error());
    auto outputId = output.take();

    std::vector<ElementwiseInput> inputs;
    inputs.reserve(group.inputs.size());
    std::vector<ScalarNode> scalars;
    scalars.reserve(group.inputs.size() + group.ops.size());
    std::unordered_map<const mid_ir::Value*, ScalarId> scalarForValue;

    ScalarId nextScalar = 0;
    for (size_t i = 0; i < group.inputs.size(); ++i) {
        auto* input = group.inputs[i];
        if (!input)
            return make_error("elementwise fusion group has null input");
        if (!can_broadcast_to(*input, *rootOutput)) {
            return make_error("elementwise fusion input cannot broadcast to root output");
        }

        auto mapped = mapped_value(valueMap, input);
        if (!mapped)
            return make_error(mapped.error());
        inputs.push_back(ElementwiseInput{
            mapped.take(),
            broadcast_mode(*input, *rootOutput),
        });

        auto id = nextScalar++;
        scalars.push_back(ScalarNode{
            id,
            ScalarOp::Load,
            input->dtype,
            static_cast<uint32_t>(i),
            0.0,
            {},
        });
        scalarForValue[input] = id;
    }

    auto scalarForOperand = [&](const mid_ir::Value* value) -> Result<ScalarId> {
        auto it = scalarForValue.find(value);
        if (it == scalarForValue.end()) {
            return make_error("elementwise fusion operand scalar is not available");
        }
        return it->second;
    };

    for (auto* op : group.ops) {
        if (!op || op->results.size() != 1)
            return make_error("elementwise fusion op expects one result");

        auto id = nextScalar++;
        if (op->kind == mid_ir::OpKind::Constant) {
            auto attr = find_attr(*op, "value", mid_ir::AttrValue::Float);
            if (!attr)
                return make_error(attr.error());
            scalars.push_back(ScalarNode{
                id,
                ScalarOp::Constant,
                op->results[0]->dtype,
                0,
                (*attr)->floatVal,
                {},
            });
            scalarForValue[op->results[0]] = id;
            continue;
        }

        if (is_unary_elementwise(op->kind)) {
            auto input = scalarForOperand(op->operands[0]);
            if (!input)
                return make_error(input.error());
            scalars.push_back(ScalarNode{
                id,
                unary_scalar_op(op->kind),
                op->results[0]->dtype,
                0,
                0.0,
                {input.take()},
            });
            scalarForValue[op->results[0]] = id;
            continue;
        }

        if (is_binary_elementwise(op->kind)) {
            auto lhs = scalarForOperand(op->operands[0]);
            if (!lhs)
                return make_error(lhs.error());
            auto rhs = scalarForOperand(op->operands[1]);
            if (!rhs)
                return make_error(rhs.error());
            scalars.push_back(ScalarNode{
                id,
                binary_scalar_op(op->kind),
                op->results[0]->dtype,
                0,
                0.0,
                {lhs.take(), rhs.take()},
            });
            scalarForValue[op->results[0]] = id;
            continue;
        }

        return make_error("elementwise fusion group contains unsupported op");
    }

    auto result = scalarForOperand(rootOutput);
    if (!result)
        return make_error(result.error());

    graph.addOp<ElementwiseKernelOp>(
        std::move(inputs),
        outputId,
        result.take(),
        std::move(scalars));
    return {};
}

Result<void> lower_fusion_group(
    Graph& graph,
    const FusionGroup& group,
    ValueMap& valueMap)
{
    switch (group.kind) {
        case FusionKind::Elementwise:
            return lower_elementwise_fusion_group(graph, group, valueMap);
        case FusionKind::Attention:
            if (!group.root)
                return make_error("attention fusion group has no root");
            return lower_attention(graph, *group.root, valueMap, true);
    }
    return make_error("unknown fusion group kind");
}

Result<void> lower_op(
    Graph& graph,
    const mid_ir::Op& op,
    ValueMap& valueMap,
    const LoweringOptions& options)
{
    switch (op.kind) {
        case mid_ir::OpKind::Input:
            return lower_input(graph, op, valueMap);
        case mid_ir::OpKind::PagedTensorInput:
            return lower_paged_tensor_input(graph, op, valueMap);
        case mid_ir::OpKind::TensorTupleCreate:
            return lower_tensor_tuple_create(graph, op, valueMap);
        case mid_ir::OpKind::Weight:
            return lower_weight(graph, op, valueMap);
        case mid_ir::OpKind::Constant:
            return lower_constant(graph, op, valueMap);
        case mid_ir::OpKind::Linear:
            return lower_linear(graph, op, valueMap);
        case mid_ir::OpKind::Add:
        case mid_ir::OpKind::Mul:
        case mid_ir::OpKind::Div:
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
        case mid_ir::OpKind::Slice:
            return lower_layout_transform(graph, op, valueMap);
        case mid_ir::OpKind::PagedAppend:
            return lower_paged_append(graph, op, valueMap);
        case mid_ir::OpKind::SlidingQueryKeyScore:
            return lower_sliding_query_key_score(graph, op, valueMap);
        case mid_ir::OpKind::Attention:
            return lower_attention(graph, op, valueMap, false);
        case mid_ir::OpKind::Softmax:
            return lower_softmax(graph, op, valueMap);
        case mid_ir::OpKind::TopK:
            return lower_topk(graph, op, valueMap);
        case mid_ir::OpKind::Sum:
            return lower_sum(graph, op, valueMap);
        case mid_ir::OpKind::Embedding:
            return lower_embedding(graph, op, valueMap);
        case mid_ir::OpKind::MoeGather:
            return lower_moe_gather(graph, op, valueMap);
        case mid_ir::OpKind::MoeMatMul:
            return lower_moe_matmul(graph, op, valueMap);
        case mid_ir::OpKind::MoeScatterSum:
            return lower_moe_scatter_sum(graph, op, valueMap);
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

MidIRToKernelIRLowering::MidIRToKernelIRLowering(LoweringOptions options)
    : options_(std::move(options))
{}

Result<std::unique_ptr<Graph>> MidIRToKernelIRLowering::lower(
    const mid_ir::Graph& midGraph)
{
    auto graph = std::make_unique<Graph>();
    ValueMap valueMap;
    auto fusionPlan = build_fusion_plan(midGraph, options_);

    for (const auto* op : midGraph.entry()->ops) {
        const auto* group = fusionPlan.groupForRoot(op);
        if (group) {
            auto result = lower_fusion_group(*graph, *group, valueMap);
            if (!result)
                return make_error(result.error());
            continue;
        }

        if (fusionPlan.isClaimed(op))
            continue;

        auto result = lower_op(*graph, *op, valueMap, options_);
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

Result<std::unique_ptr<Graph>> lowerMidIRToKernelIR(
        const mid_ir::Graph& graph,
        const LoweringOptions& options) {
    MidIRToKernelIRLowering lowering(options);
    return lowering.lower(graph);
}

} // namespace sandy::ir::kernel_ir
