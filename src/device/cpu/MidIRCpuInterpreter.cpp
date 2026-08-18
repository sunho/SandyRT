#include "MidIRCpuInterpreter.h"

#include <string>
#include <vector>

namespace sandy::device::debug {

namespace {

Result<int64_t> attr_int_or(
        const ir::mid_ir::AttrMap& attrs,
        const std::string& name,
        int64_t fallback) {
    auto it = attrs.find(name);
    if (it == attrs.end())
        return fallback;
    if (it->second.kind != ir::mid_ir::AttrValue::Int)
        return make_error("attr '" + name + "' must be int");
    return it->second.intVal;
}

Result<float> attr_float_or(
        const ir::mid_ir::AttrMap& attrs,
        const std::string& name,
        float fallback) {
    auto it = attrs.find(name);
    if (it == attrs.end())
        return fallback;
    if (it->second.kind != ir::mid_ir::AttrValue::Float)
        return make_error("attr '" + name + "' must be float");
    return static_cast<float>(it->second.floatVal);
}

Result<std::vector<int64_t>> attr_int_list(
        const ir::mid_ir::AttrMap& attrs,
        const std::string& name) {
    auto it = attrs.find(name);
    if (it == attrs.end() || it->second.kind != ir::mid_ir::AttrValue::IntList)
        return make_error("missing int list attr '" + name + "'");
    return it->second.intListVal;
}

Result<void> check_arity(
        const ir::mid_ir::Op& op,
        std::span<const core::TensorRef> inputs,
        std::span<const core::MutableTensorRef> outputs) {
    if (inputs.size() != op.operands.size())
        return make_error("debug MidIR CPU interpreter input arity mismatch");
    if (outputs.size() != op.results.size())
        return make_error("debug MidIR CPU interpreter output arity mismatch");
    return {};
}

} // namespace

Result<void> runMidIROpOnCpu(
        const ir::mid_ir::Op& op,
        std::span<const core::TensorRef> inputs,
        std::span<const core::MutableTensorRef> outputs) {
    auto arity = check_arity(op, inputs, outputs);
    if (!arity)
        return make_error(arity.error());

    switch (op.kind) {
        case ir::mid_ir::OpKind::Constant: {
            auto value = attr_float_or(op.attrs, "value", 0.0f);
            if (!value) return make_error(value.error());
            outputs[0].store_float(0, value.take());
            return {};
        }
        case ir::mid_ir::OpKind::Linear:
            return core::linear(inputs[0], inputs[1], inputs[2], outputs[0]);
        case ir::mid_ir::OpKind::ReLU:
            return core::relu(inputs[0], outputs[0]);
        case ir::mid_ir::OpKind::Add:
            return core::add(inputs[0], inputs[1], outputs[0]);
        case ir::mid_ir::OpKind::Mul:
            return core::mul(inputs[0], inputs[1], outputs[0]);
        case ir::mid_ir::OpKind::Div:
            return core::div(inputs[0], inputs[1], outputs[0]);
        case ir::mid_ir::OpKind::Sqrt:
            return core::sqrt(inputs[0], outputs[0]);
        case ir::mid_ir::OpKind::Tanh:
            return core::tanh(inputs[0], outputs[0]);
        case ir::mid_ir::OpKind::MatMul: {
            auto transposeLhs = attr_int_or(op.attrs, "transpose_lhs", 0);
            if (!transposeLhs) return make_error(transposeLhs.error());
            auto transposeRhs = attr_int_or(op.attrs, "transpose_rhs", 0);
            if (!transposeRhs) return make_error(transposeRhs.error());
            return core::matmul(
                inputs[0],
                inputs[1],
                transposeLhs.take() != 0,
                transposeRhs.take() != 0,
                outputs[0]);
        }
        case ir::mid_ir::OpKind::Transpose:
            return core::transpose(inputs[0], outputs[0]);
        case ir::mid_ir::OpKind::Reshape:
            return core::reshape(inputs[0], outputs[0]);
        case ir::mid_ir::OpKind::Permute: {
            auto dims = attr_int_list(op.attrs, "dims");
            if (!dims) return make_error(dims.error());
            auto dimsValue = dims.take();
            return core::permute(inputs[0], dimsValue, outputs[0]);
        }
        case ir::mid_ir::OpKind::SlidingQueryKeyScore: {
            auto window = attr_int_or(op.attrs, "window", 0);
            if (!window) return make_error(window.error());
            auto scale = attr_float_or(op.attrs, "scale", -1.0f);
            if (!scale) return make_error(scale.error());
            return core::sliding_query_key_score(
                inputs[0],
                inputs[1],
                window.take(),
                scale.take(),
                outputs[0]);
        }
        case ir::mid_ir::OpKind::Attention: {
            auto window = attr_int_or(op.attrs, "window", 0);
            if (!window) return make_error(window.error());
            auto scale = attr_float_or(op.attrs, "scale", -1.0f);
            if (!scale) return make_error(scale.error());
            if (inputs.size() == 4) {
                return core::attention(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    window.take(),
                    scale.take(),
                    outputs[0]);
            }
            return core::attention(
                inputs[0],
                inputs[1],
                inputs[2],
                window.take(),
                scale.take(),
                outputs[0]);
        }
        case ir::mid_ir::OpKind::Softmax: {
            auto dim = attr_int_or(op.attrs, "dim", -1);
            if (!dim) return make_error(dim.error());
            return core::softmax(inputs[0], dim.take(), outputs[0]);
        }
        case ir::mid_ir::OpKind::Embedding:
            return core::embedding(inputs[0], inputs[1], outputs[0]);
        case ir::mid_ir::OpKind::TopK:
        case ir::mid_ir::OpKind::Sum:
        case ir::mid_ir::OpKind::MoeGather:
        case ir::mid_ir::OpKind::MoeMatMul:
        case ir::mid_ir::OpKind::MoeScatterSum:
            return make_error("debug MidIR CPU interpreter cannot run routing or MoE op");
        case ir::mid_ir::OpKind::RoPE: {
            auto theta = attr_float_or(op.attrs, "rope_theta", 10000.0f);
            if (!theta) return make_error(theta.error());
            auto rotaryDim = attr_int_or(op.attrs, "rotary_dim", -1);
            if (!rotaryDim) return make_error(rotaryDim.error());
            auto splitHalf = attr_int_or(op.attrs, "split_half", 0);
            if (!splitHalf) return make_error(splitHalf.error());
            if (inputs.size() == 2) {
                return core::rope(
                    inputs[0],
                    inputs[1],
                    theta.take(),
                    rotaryDim.take(),
                    splitHalf.take() != 0,
                    outputs[0]);
            }
            return core::rope(
                    inputs[0],
                    theta.take(),
                    rotaryDim.take(),
                    splitHalf.take() != 0,
                    outputs[0]);
        }
        case ir::mid_ir::OpKind::RMSNorm: {
            auto epsilon = attr_float_or(op.attrs, "epsilon", 1.0e-6f);
            if (!epsilon) return make_error(epsilon.error());
            if (inputs.size() == 1)
                return core::rms_norm(inputs[0], epsilon.take(), outputs[0]);
            if (inputs.size() != 2)
                return make_error("rms_norm expects 1 or 2 inputs");
            return core::rms_norm(inputs[0], inputs[1], epsilon.take(), outputs[0]);
        }
        case ir::mid_ir::OpKind::LayerNorm: {
            auto epsilon = attr_float_or(op.attrs, "epsilon", 1.0e-5f);
            if (!epsilon) return make_error(epsilon.error());
            return core::layer_norm(inputs[0], inputs[1], inputs[2], epsilon.take(), outputs[0]);
        }
        case ir::mid_ir::OpKind::Input:
        case ir::mid_ir::OpKind::Weight:
        case ir::mid_ir::OpKind::NUM_KINDS:
            return make_error("debug MidIR CPU interpreter cannot run boundary or invalid op");
    }

    return make_error("debug MidIR CPU interpreter cannot run unknown op kind");
}

} // namespace sandy::device::debug
