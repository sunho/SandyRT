#include "KernelIR.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace sandy::ir::kernel_ir {

namespace {

std::string value_ref(ValueId id) {
    return "%" + std::to_string(id);
}

std::string op_ref(OpId id) {
    return "op " + std::to_string(id);
}

std::string device_ref(DeviceId device) {
    return "device " + std::to_string(device);
}

std::string shape_dims_string(const core::Shape& shape) {
    std::string out;
    for (int i = 0; i < shape.rank(); ++i) {
        if (i != 0)
            out += "x";
        if (shape.is_dynamic(i)) {
            out += "?";
        } else {
            out += std::to_string(shape.dim(i));
        }
    }
    return out;
}

std::string type_string(const ValueType& type) {
    switch (type.kind) {
        case ValueKind::Tensor: {
            auto dims = shape_dims_string(type.shape);
            if (!dims.empty())
                dims += "x";
            return "tensor<" + dims + core::dtype_name(type.dtype) + ">";
        }
        case ValueKind::PagedTensor: {
            auto dims = shape_dims_string(type.shape);
            if (!dims.empty())
                dims += "x";
            return "paged_tensor<" + dims + core::dtype_name(type.dtype) +
                   ", grow_dim=" + std::to_string(type.paged.growDim) +
                   ", page_size=" + std::to_string(type.paged.pageSize) + ">";
        }
        case ValueKind::TensorTuple: {
            std::string out = "tensor_tuple<";
            for (size_t i = 0; i < type.elements.size(); i++) {
                if (i != 0)
                    out += ", ";
                out += type_string(type.elements[i]);
            }
            out += ">";
            return out;
        }
        case ValueKind::Scalar:
            return "scalar<" + std::string(core::dtype_name(type.dtype)) + ">";
    }
    return "?";
}

bool same_value_type(const ValueType& lhs, const ValueType& rhs) {
    if (lhs.kind != rhs.kind)
        return false;
    if (lhs.kind == ValueKind::TensorTuple) {
        if (lhs.elements.size() != rhs.elements.size())
            return false;
        for (size_t i = 0; i < lhs.elements.size(); i++) {
            if (!same_value_type(lhs.elements[i], rhs.elements[i]))
                return false;
        }
        return true;
    }
    if (lhs.dtype != rhs.dtype || lhs.shape != rhs.shape)
        return false;
    if (lhs.kind == ValueKind::PagedTensor) {
        return lhs.paged.growDim == rhs.paged.growDim &&
               lhs.paged.pageSize == rhs.paged.pageSize;
    }
    return true;
}

Result<void> verify_value_type_impl(const ValueType& type, const std::string& name) {
    if (type.kind == ValueKind::TensorTuple) {
        for (size_t i = 0; i < type.elements.size(); i++) {
            auto result = verify_value_type_impl(
                type.elements[i],
                name + " tuple element " + std::to_string(i));
            if (!result)
                return result;
        }
        return {};
    }
    if (type.kind != ValueKind::PagedTensor)
        return {};

    int rank = type.shape.rank();
    if (rank < 1) {
        return make_error(name + " paged tensor rank must be >= 1");
    }
    if (type.paged.growDim < 0 || type.paged.growDim >= rank) {
        return make_error(name + " paged tensor grow_dim out of range");
    }
    if (type.paged.pageSize <= 0) {
        return make_error(name + " paged tensor page_size must be > 0");
    }
    for (int i = 0; i < rank; ++i) {
        auto dim = type.shape.dim(i);
        if (dim != core::Shape::kDynamic && dim <= 0) {
            return make_error(name + " paged tensor dimensions must be positive or dynamic");
        }
    }
    return {};
}

Result<void> verify_value_type(const Value& value) {
    return verify_value_type_impl(value.type, value_ref(value.id));
}

std::string values_string(std::span<const ValueId> values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out += ", ";
        out += value_ref(values[i]);
    }
    return out;
}

std::string int_list_string(const std::vector<int64_t>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out += ", ";
        out += std::to_string(values[i]);
    }
    out += "]";
    return out;
}

bool has_scalar(const std::vector<ScalarNode>& scalars, ScalarId id) {
    return std::any_of(scalars.begin(), scalars.end(),
                       [id](const ScalarNode& scalar) { return scalar.id == id; });
}

std::string scalar_ref(ScalarId id) {
    return "s" + std::to_string(id);
}

std::string float_string(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

const char* scalar_op_name(ScalarOp op) {
    switch (op) {
        case ScalarOp::Load: return "load";
        case ScalarOp::Constant: return "constant";
        case ScalarOp::Add: return "add";
        case ScalarOp::Sub: return "sub";
        case ScalarOp::Mul: return "mul";
        case ScalarOp::Div: return "div";
        case ScalarOp::Max: return "max";
        case ScalarOp::Min: return "min";
        case ScalarOp::Neg: return "neg";
        case ScalarOp::Sqrt: return "sqrt";
        case ScalarOp::Rsqrt: return "rsqrt";
        case ScalarOp::Exp: return "exp";
        case ScalarOp::Log: return "log";
        case ScalarOp::Tanh: return "tanh";
        case ScalarOp::ReLU: return "relu";
        case ScalarOp::Cast: return "cast";
    }
    return "?";
}

std::string scalar_operands_string(const ScalarNode& scalar) {
    std::string out;
    for (size_t i = 0; i < scalar.operands.size(); ++i) {
        if (i != 0)
            out += ", ";
        out += scalar_ref(scalar.operands[i]);
    }
    return out;
}

void dump_elementwise_body(const ElementwiseKernelOp& op) {
    std::cout << "  {\n";
    for (const auto& scalar : op.scalars()) {
        std::cout << "    " << scalar_ref(scalar.id) << " = ";

        if (scalar.op == ScalarOp::Load) {
            std::cout << "load input" << scalar.inputIndex;
        } else if (scalar.op == ScalarOp::Constant) {
            std::cout << "constant " << float_string(scalar.constant);
        } else if (scalar.op == ScalarOp::Cast) {
            std::cout << "cast " << scalar_operands_string(scalar)
                      << " to " << core::dtype_name(scalar.dtype);
        } else {
            std::cout << scalar_op_name(scalar.op);
            auto operands = scalar_operands_string(scalar);
            if (!operands.empty())
                std::cout << " " << operands;
        }

        std::cout << "\n";
    }

    std::cout << "    store " << value_ref(op.output())
              << ", " << scalar_ref(op.result()) << "\n";
    std::cout << "  }\n";
}

Result<void> verify_values_exist(
    const Graph& graph,
    std::span<const ValueId> values,
    const char* role,
    const Op& op)
{
    for (auto value : values) {
        if (!graph.hasValue(value)) {
            return make_error(op_ref(op.id()) + " " + op.name() + " has invalid " +
                              role + " value " + std::to_string(value));
        }
    }
    return {};
}

Result<void> verify_common_op_shape(
    const Graph& graph,
    const Op& op,
    size_t expectedInputs,
    size_t expectedOutputs)
{
    if (op.inputs().size() != expectedInputs) {
        return make_error(op_ref(op.id()) + " " + op.name() + " expects " +
                          std::to_string(expectedInputs) + " input(s), got " +
                          std::to_string(op.inputs().size()));
    }
    if (op.outputs().size() != expectedOutputs) {
        return make_error(op_ref(op.id()) + " " + op.name() + " expects " +
                          std::to_string(expectedOutputs) + " output(s), got " +
                          std::to_string(op.outputs().size()));
    }
    if (auto result = verify_values_exist(graph, op.inputs(), "input", op); !result) {
        return result;
    }
    return verify_values_exist(graph, op.outputs(), "output", op);
}

Result<void> verify_device_placement(const Graph& graph, const Op& op) {
    if (op.kind() == OpKind::DeviceTransfer) {
        const auto& transfer = static_cast<const DeviceTransferOp&>(op);
        if (op.device() != transfer.sourceDevice()) {
            return make_error(op_ref(op.id()) + " device transfer op placement must match source device");
        }
        if (graph.value(transfer.inputs()[0]).device != transfer.sourceDevice()) {
            return make_error(op_ref(op.id()) + " device transfer input is not on source device");
        }
        if (graph.value(transfer.outputs()[0]).device != transfer.targetDevice()) {
            return make_error(op_ref(op.id()) + " device transfer output is not on target device");
        }
        return {};
    }

    for (auto input : op.inputs()) {
        if (graph.value(input).device != op.device()) {
            return make_error(op_ref(op.id()) + " input " + value_ref(input) +
                              " is on " + device_ref(graph.value(input).device) +
                              ", expected " + device_ref(op.device()));
        }
    }

    for (auto output : op.outputs()) {
        if (graph.value(output).device != op.device()) {
            return make_error(op_ref(op.id()) + " output " + value_ref(output) +
                              " is on " + device_ref(graph.value(output).device) +
                              ", expected " + device_ref(op.device()));
        }
    }

    return {};
}

std::string input_source_string(const InputSource& source) {
    switch (source.kind) {
        case InputSourceKind::Argument:
            if (source.tupleElement >= 0) {
                return "arg(" + std::to_string(source.index) +
                       ")[" + std::to_string(source.tupleElement) + "]";
            }
            return "arg(" + std::to_string(source.index) + ")";
        case InputSourceKind::Weight:
            return "weight(\"" + source.name + "\")";
        case InputSourceKind::External:
            return "external(\"" + source.name + "\")";
    }
    return "?";
}

const char* layout_transform_name(LayoutTransformKind kind) {
    switch (kind) {
        case LayoutTransformKind::Reshape: return "reshape";
        case LayoutTransformKind::Transpose: return "transpose";
        case LayoutTransformKind::Permute: return "permute";
        case LayoutTransformKind::Contiguous: return "contiguous";
    }
    return "?";
}

const char* norm_kind_name(NormKind kind) {
    switch (kind) {
        case NormKind::RMSNorm: return "rms_norm";
        case NormKind::LayerNorm: return "layer_norm";
    }
    return "?";
}

const char* reduce_op_name(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum: return "sum";
        case ReduceOp::Max: return "max";
        case ReduceOp::Min: return "min";
        case ReduceOp::Prod: return "prod";
        case ReduceOp::Mean: return "mean";
    }
    return "?";
}

std::string op_attr_string(const Op& op) {
    switch (op.kind()) {
        case OpKind::Input: {
            const auto& input = static_cast<const InputOp&>(op);
            return " source=" + input_source_string(input.source());
        }
        case OpKind::DeviceTransfer: {
            const auto& transfer = static_cast<const DeviceTransferOp&>(op);
            return " source_device=" + std::to_string(transfer.sourceDevice()) +
                   " target_device=" + std::to_string(transfer.targetDevice());
        }
        case OpKind::LayoutTransform: {
            const auto& layout = static_cast<const LayoutTransformOp&>(op);
            return " kind=" + std::string(layout_transform_name(layout.transform())) +
                   " dims=" + int_list_string(layout.dims());
        }
        case OpKind::ElementwiseKernel: {
            const auto& elementwise = static_cast<const ElementwiseKernelOp&>(op);
            return " scalars=" + std::to_string(elementwise.scalars().size()) +
                   " result=" + scalar_ref(elementwise.result());
        }
        case OpKind::ReductionKernel: {
            const auto& reduction = static_cast<const ReductionKernelOp&>(op);
            return " reduce=" + std::string(reduce_op_name(reduction.reduce())) +
                   " axes=" + int_list_string(reduction.axes()) +
                   " keep_dims=" + std::to_string(reduction.keepDims() ? 1 : 0);
        }
        case OpKind::MatMulKernel: {
            const auto& matmul = static_cast<const MatMulKernelOp&>(op);
            return " transpose_lhs=" +
                   std::to_string(matmul.transposeLhs() ? 1 : 0) +
                   " transpose_rhs=" +
                   std::to_string(matmul.transposeRhs() ? 1 : 0);
        }
        case OpKind::SoftmaxKernel: {
            const auto& softmax = static_cast<const SoftmaxKernelOp&>(op);
            return " axis=" + std::to_string(softmax.axis());
        }
        case OpKind::NormKernel: {
            const auto& norm = static_cast<const NormKernelOp&>(op);
            return " kind=" + std::string(norm_kind_name(norm.norm())) +
                   " epsilon=" + std::to_string(norm.epsilon());
        }
        case OpKind::RoPEKernel: {
            const auto& rope = static_cast<const RoPEKernelOp&>(op);
            return " theta=" + std::to_string(rope.theta()) +
                   " rotary_dim=" + std::to_string(rope.rotaryDim()) +
                   " split_half=" + std::to_string(rope.splitHalf() ? 1 : 0);
        }
        case OpKind::SlidingQueryKeyScoreKernel: {
            const auto& score = static_cast<const SlidingQueryKeyScoreKernelOp&>(op);
            return " window=" + std::to_string(score.window()) +
                   " scale=" + std::to_string(score.scale());
        }
        case OpKind::CustomKernel: {
            const auto& custom = static_cast<const CustomKernelOp&>(op);
            return " name=\"" + custom.customName() + "\"";
        }
        default:
            return "";
    }
}

} // namespace

const char* op_kind_name(OpKind kind) {
    switch (kind) {
        case OpKind::Input: return "input";
        case OpKind::TensorTupleCreate: return "tensor_tuple_create";
        case OpKind::DeviceTransfer: return "device_transfer";
        case OpKind::LayoutTransform: return "layout_transform";
        case OpKind::ElementwiseKernel: return "elementwise_kernel";
        case OpKind::ReductionKernel: return "reduction_kernel";
        case OpKind::MatMulKernel: return "matmul_kernel";
        case OpKind::GatherKernel: return "gather_kernel";
        case OpKind::SoftmaxKernel: return "softmax_kernel";
        case OpKind::NormKernel: return "norm_kernel";
        case OpKind::RoPEKernel: return "rope_kernel";
        case OpKind::SlidingQueryKeyScoreKernel:
            return "sliding_query_key_score_kernel";
        case OpKind::CustomKernel: return "custom_kernel";
    }
    return "?";
}

ValueId Graph::addValue(ValueType type, std::string debugName, DeviceId device) {
    auto id = nextValueId_++;
    values_.push_back(Value{id, std::move(type), device, {}, {}, std::move(debugName)});
    return id;
}

ValueId Graph::addValue(ValueType type, DeviceId device) {
    return addValue(std::move(type), "", device);
}

bool Graph::hasValue(ValueId id) const {
    return id < values_.size() && values_[id].id == id;
}

bool Graph::hasOp(OpId id) const {
    return id < ops_.size() && ops_[id] && ops_[id]->id() == id;
}

const Value& Graph::value(ValueId id) const {
    return values_.at(id);
}

Value& Graph::value(ValueId id) {
    return values_.at(id);
}

const Op& Graph::op(OpId id) const {
    return *ops_.at(id);
}

Op& Graph::op(OpId id) {
    return *ops_.at(id);
}

void Graph::setOutputs(std::vector<ValueId> outputs) {
    outputs_ = std::move(outputs);
}

Result<void> Graph::verify() const {
    for (size_t i = 0; i < values_.size(); ++i) {
        const auto& val = values_[i];
        if (val.id != i) {
            return make_error("value id mismatch at index " + std::to_string(i));
        }
        if (val.def.op == kInvalidOpId) {
            return make_error(value_ref(val.id) + " has no defining op");
        }
        if (auto result = verify_value_type(val); !result) {
            return result;
        }
        if (!hasOp(val.def.op)) {
            return make_error(value_ref(val.id) + " has invalid defining op " +
                              std::to_string(val.def.op));
        }
        auto defOutputs = op(val.def.op).outputs();
        if (val.def.result >= defOutputs.size() ||
            defOutputs[val.def.result] != val.id) {
            return make_error(value_ref(val.id) + " def does not point back to value");
        }

        for (const auto& use : val.uses) {
            if (!hasOp(use.op)) {
                return make_error(value_ref(val.id) + " has invalid use op " +
                                  std::to_string(use.op));
            }
            auto useInputs = op(use.op).inputs();
            if (use.operand >= useInputs.size() ||
                useInputs[use.operand] != val.id) {
                return make_error(value_ref(val.id) + " use does not point back to value");
            }
        }
    }

    for (size_t i = 0; i < ops_.size(); ++i) {
        if (!ops_[i]) {
            return make_error("null op at index " + std::to_string(i));
        }
        const auto& current = *ops_[i];
        if (current.id() != i) {
            return make_error("op id mismatch at index " + std::to_string(i));
        }

        if (auto result = current.verify(*this); !result) {
            return result;
        }
        if (auto result = verify_device_placement(*this, current); !result) {
            return result;
        }

        auto inputs = current.inputs();
        for (uint32_t operand = 0; operand < inputs.size(); ++operand) {
            auto input = inputs[operand];
            if (!hasValue(input)) {
                return make_error(op_ref(current.id()) + " references invalid input " +
                                  std::to_string(input));
            }

            const auto& inputValue = value(input);
            if (inputValue.def.op >= current.id()) {
                return make_error(op_ref(current.id()) + " uses " + value_ref(input) +
                                  " before it is defined");
            }

            bool found = std::any_of(
                inputValue.uses.begin(), inputValue.uses.end(),
                [&](const Use& use) {
                    return use.op == current.id() && use.operand == operand;
                });
            if (!found) {
                return make_error(op_ref(current.id()) + " input " + value_ref(input) +
                                  " is missing matching use record");
            }
        }

        auto outputs = current.outputs();
        for (uint32_t result = 0; result < outputs.size(); ++result) {
            auto output = outputs[result];
            if (!hasValue(output)) {
                return make_error(op_ref(current.id()) + " references invalid output " +
                                  std::to_string(output));
            }

            const auto& outputValue = value(output);
            if (outputValue.def.op != current.id() ||
                outputValue.def.result != result) {
                return make_error(op_ref(current.id()) + " output " +
                                  value_ref(output) +
                                  " is missing matching def record");
            }
        }
    }

    for (auto output : outputs_) {
        if (!hasValue(output)) {
            return make_error("graph output references invalid value " +
                              std::to_string(output));
        }
    }

    return {};
}

void Graph::dump() const {
    std::cout << "kernel_ir.graph {\n";

    for (const auto& opPtr : ops_) {
        const auto& current = *opPtr;
        auto outputs = current.outputs();
        std::cout << "  ";
        if (!outputs.empty()) {
            std::cout << values_string(outputs) << " = ";
        }
        std::cout << current.name() << "(" << values_string(current.inputs()) << ")";

        if (!outputs.empty()) {
            std::cout << " : ";
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (i != 0)
                    std::cout << ", ";
                if (hasValue(outputs[i])) {
                    std::cout << type_string(value(outputs[i]).type);
                } else {
                    std::cout << "<invalid>";
                }
            }
        }

        std::cout << op_attr_string(current) << "\n";

        if (current.kind() == OpKind::ElementwiseKernel) {
            dump_elementwise_body(static_cast<const ElementwiseKernelOp&>(current));
        }
    }

    std::cout << "  outputs: " << values_string(outputs_) << "\n";
    std::cout << "}\n";
}

InputOp::InputOp(OpId id, InputSource source, ValueId output, DeviceId device)
    : Op(id, OpKind::Input, device),
      source_(std::move(source)),
      outputs_{output}
{}

Result<void> InputOp::verify(const Graph& graph) const {
    if (auto result = verify_common_op_shape(graph, *this, 0, 1); !result) {
        return result;
    }

    switch (source_.kind) {
        case InputSourceKind::Argument:
            if (source_.index < 0) {
                return make_error(op_ref(id()) + " input argument has negative index");
            }
            break;
        case InputSourceKind::Weight:
        case InputSourceKind::External:
            if (source_.name.empty()) {
                return make_error(op_ref(id()) + " input source name is empty");
            }
            break;
    }
    return {};
}

TensorTupleCreateOp::TensorTupleCreateOp(
        OpId id,
        std::vector<ValueId> inputs,
        ValueId output,
        DeviceId device)
    : Op(id, OpKind::TensorTupleCreate, device),
      inputs_(std::move(inputs)),
      outputs_{output}
{}

Result<void> TensorTupleCreateOp::verify(const Graph& graph) const {
    if (auto result = verify_values_exist(graph, inputs(), "input", *this); !result) {
        return result;
    }
    if (auto result = verify_values_exist(graph, outputs(), "output", *this); !result) {
        return result;
    }
    const auto& outputType = graph.value(outputs_[0]).type;
    if (outputType.kind != ValueKind::TensorTuple) {
        return make_error(op_ref(id()) + " tensor tuple create output must be tuple");
    }
    if (outputType.elements.size() != inputs_.size()) {
        return make_error(op_ref(id()) + " tensor tuple create type arity mismatch");
    }
    for (size_t i = 0; i < inputs_.size(); i++) {
        const auto& inputType = graph.value(inputs_[i]).type;
        if (inputType.kind == ValueKind::TensorTuple) {
            return make_error(op_ref(id()) + " nested tensor tuples are not supported");
        }
        if (!same_value_type(inputType, outputType.elements[i])) {
            return make_error(op_ref(id()) + " tensor tuple create element type mismatch");
        }
    }
    return {};
}

DeviceTransferOp::DeviceTransferOp(
    OpId id,
    DeviceId sourceDevice,
    DeviceId targetDevice,
    ValueId input,
    ValueId output)
    : Op(id, OpKind::DeviceTransfer, sourceDevice),
      sourceDevice_(sourceDevice),
      targetDevice_(targetDevice),
      inputs_{input},
      outputs_{output}
{}

Result<void> DeviceTransferOp::verify(const Graph& graph) const {
    if (auto result = verify_common_op_shape(graph, *this, 1, 1); !result) {
        return result;
    }
    if (sourceDevice_ == targetDevice_) {
        return make_error(op_ref(id()) + " device transfer source and target are equal");
    }
    if (!same_value_type(graph.value(inputs_[0]).type, graph.value(outputs_[0]).type)) {
        return make_error(op_ref(id()) + " device transfer input/output types differ");
    }
    return {};
}

LayoutTransformOp::LayoutTransformOp(
    OpId id,
    LayoutTransformKind transform,
    ValueId input,
    ValueId output,
    std::vector<int64_t> dims,
    DeviceId device)
    : Op(id, OpKind::LayoutTransform, device),
      transform_(transform),
      inputs_{input},
      outputs_{output},
      dims_(std::move(dims))
{}

Result<void> LayoutTransformOp::verify(const Graph& graph) const {
    if (auto result = verify_common_op_shape(graph, *this, 1, 1); !result) {
        return result;
    }
    const auto& inputValue = graph.value(inputs_[0]);
    if (inputValue.type.kind == ValueKind::PagedTensor) {
        return make_error(op_ref(id()) + " layout transform cannot consume paged input");
    }
    if (transform_ == LayoutTransformKind::Permute && dims_.empty()) {
        return make_error(op_ref(id()) + " permute transform requires dims");
    }
    return {};
}

ElementwiseKernelOp::ElementwiseKernelOp(
    OpId id,
    std::vector<ElementwiseInput> elementwiseInputs,
    ValueId output,
    ScalarId result,
    std::vector<ScalarNode> scalars,
    DeviceId device)
    : Op(id, OpKind::ElementwiseKernel, device),
      elementwiseInputs_(std::move(elementwiseInputs)),
      outputs_{output},
      output_(output),
      result_(result),
      scalars_(std::move(scalars))
{
    inputs_.reserve(elementwiseInputs_.size());
    for (const auto& input : elementwiseInputs_) {
        inputs_.push_back(input.value);
    }
}

std::span<const ValueId> ElementwiseKernelOp::inputs() const {
    return {inputs_.data(), inputs_.size()};
}

Result<void> ElementwiseKernelOp::verify(const Graph& graph) const {
    if (auto result = verify_values_exist(graph, inputs(), "input", *this); !result) {
        return result;
    }
    if (auto result = verify_values_exist(graph, outputs(), "output", *this); !result) {
        return result;
    }

    for (const auto& scalar : scalars_) {
        if (scalar.op == ScalarOp::Load && scalar.inputIndex >= inputs_.size()) {
            return make_error(op_ref(id()) + " scalar load references invalid input");
        }
        for (auto operand : scalar.operands) {
            if (!has_scalar(scalars_, operand)) {
                return make_error(op_ref(id()) + " scalar references invalid operand");
            }
        }
    }

    if (!has_scalar(scalars_, result_)) {
        return make_error(op_ref(id()) + " result references invalid scalar");
    }

    return {};
}

ReductionKernelOp::ReductionKernelOp(
    OpId id,
    ReduceOp reduce,
    ValueId input,
    ValueId output,
    std::vector<int64_t> axes,
    bool keepDims,
    DeviceId device)
    : Op(id, OpKind::ReductionKernel, device),
      reduce_(reduce),
      inputs_{input},
      outputs_{output},
      axes_(std::move(axes)),
      keepDims_(keepDims)
{}

Result<void> ReductionKernelOp::verify(const Graph& graph) const {
    if (auto result = verify_common_op_shape(graph, *this, 1, 1); !result) {
        return result;
    }
    if (axes_.empty()) {
        return make_error(op_ref(id()) + " reduction kernel requires axes");
    }
    return {};
}

MatMulKernelOp::MatMulKernelOp(
    OpId id,
    ValueId lhs,
    ValueId rhs,
    ValueId output,
    bool transposeLhs,
    bool transposeRhs,
    DeviceId device)
    : Op(id, OpKind::MatMulKernel, device),
      inputs_{lhs, rhs},
      outputs_{output},
      transposeLhs_(transposeLhs),
      transposeRhs_(transposeRhs)
{}

Result<void> MatMulKernelOp::verify(const Graph& graph) const {
    return verify_common_op_shape(graph, *this, 2, 1);
}

GatherKernelOp::GatherKernelOp(
    OpId id,
    ValueId ids,
    ValueId table,
    ValueId output,
    DeviceId device)
    : Op(id, OpKind::GatherKernel, device),
      inputs_{ids, table},
      outputs_{output}
{}

Result<void> GatherKernelOp::verify(const Graph& graph) const {
    return verify_common_op_shape(graph, *this, 2, 1);
}

SoftmaxKernelOp::SoftmaxKernelOp(
    OpId id,
    ValueId input,
    ValueId output,
    int64_t axis,
    DeviceId device)
    : Op(id, OpKind::SoftmaxKernel, device),
      inputs_{input},
      outputs_{output},
      axis_(axis)
{}

Result<void> SoftmaxKernelOp::verify(const Graph& graph) const {
    return verify_common_op_shape(graph, *this, 1, 1);
}

NormKernelOp::NormKernelOp(
    OpId id,
    NormKind norm,
    std::vector<ValueId> inputs,
    ValueId output,
    double epsilon,
    DeviceId device)
    : Op(id, OpKind::NormKernel, device),
      norm_(norm),
      inputs_(std::move(inputs)),
      outputs_{output},
      epsilon_(epsilon)
{}

std::span<const ValueId> NormKernelOp::inputs() const {
    return {inputs_.data(), inputs_.size()};
}

Result<void> NormKernelOp::verify(const Graph& graph) const {
    if (auto result = verify_values_exist(graph, inputs(), "input", *this); !result) {
        return result;
    }
    if (auto result = verify_values_exist(graph, outputs(), "output", *this); !result) {
        return result;
    }

    if (norm_ == NormKind::RMSNorm && inputs_.size() != 1 && inputs_.size() != 2) {
        return make_error(op_ref(id()) + " rms_norm expects 1 or 2 inputs");
    }
    if (norm_ == NormKind::LayerNorm && inputs_.size() != 3) {
        return make_error(op_ref(id()) + " layer_norm expects 3 inputs");
    }
    if (epsilon_ < 0.0) {
        return make_error(op_ref(id()) + " norm epsilon must be non-negative");
    }
    return {};
}

RoPEKernelOp::RoPEKernelOp(
    OpId id,
    ValueId input,
    ValueId output,
    double theta,
    int64_t rotaryDim,
    bool splitHalf,
    DeviceId device)
    : Op(id, OpKind::RoPEKernel, device),
      inputs_{input},
      outputs_{output},
      theta_(theta),
      rotaryDim_(rotaryDim),
      splitHalf_(splitHalf)
{}

Result<void> RoPEKernelOp::verify(const Graph& graph) const {
    if (auto result = verify_common_op_shape(graph, *this, 1, 1); !result) {
        return result;
    }
    if (theta_ <= 0.0) {
        return make_error(op_ref(id()) + " rope theta must be positive");
    }
    return {};
}

SlidingQueryKeyScoreKernelOp::SlidingQueryKeyScoreKernelOp(
    OpId id,
    ValueId query,
    ValueId key,
    ValueId output,
    int64_t window,
    double scale,
    DeviceId device)
    : Op(id, OpKind::SlidingQueryKeyScoreKernel, device),
      inputs_{query, key},
      outputs_{output},
      window_(window),
      scale_(scale)
{}

Result<void> SlidingQueryKeyScoreKernelOp::verify(const Graph& graph) const {
    if (auto result = verify_common_op_shape(graph, *this, 2, 1); !result) {
        return result;
    }
    if (window_ < 0) {
        return make_error(op_ref(id()) + " sliding score window must be non-negative");
    }
    return {};
}

CustomKernelOp::CustomKernelOp(
    OpId id,
    std::string customName,
    std::vector<ValueId> inputs,
    std::vector<ValueId> outputs,
    mid_ir::AttrMap attrs,
    DeviceId device)
    : Op(id, OpKind::CustomKernel, device),
      customName_(std::move(customName)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      attrs_(std::move(attrs))
{}

std::span<const ValueId> CustomKernelOp::inputs() const {
    return {inputs_.data(), inputs_.size()};
}

std::span<const ValueId> CustomKernelOp::outputs() const {
    return {outputs_.data(), outputs_.size()};
}

Result<void> CustomKernelOp::verify(const Graph& graph) const {
    if (customName_.empty()) {
        return make_error(op_ref(id()) + " custom kernel name is empty");
    }
    if (outputs_.empty()) {
        return make_error(op_ref(id()) + " custom kernel has no outputs");
    }
    if (auto result = verify_values_exist(graph, inputs(), "input", *this); !result) {
        return result;
    }
    return verify_values_exist(graph, outputs(), "output", *this);
}

} // namespace sandy::ir::kernel_ir
