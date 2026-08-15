#include "CpuDevice.h"

#include "ShapeUtil.h"
#include "TensorCalc.h"

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sandy::engine {

namespace {

class CpuTensorBuffer final : public core::TensorBuffer {
public:
    CpuTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

Result<size_t> tensor_byte_size(const core::TensorDesc& desc) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error("cpu device buffer must have static shape");
    return static_cast<size_t>(numel) * core::dtype_size(desc.dtype);
}

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

} // namespace

Result<DeviceProgramId> CpuDevice::compile(const ir::mid_ir::Op& op) {
    switch (op.kind) {
        case ir::mid_ir::OpKind::Input:
        case ir::mid_ir::OpKind::Weight:
            return make_error("cpu device cannot compile data boundary ops");
        case ir::mid_ir::OpKind::Reshape:
            return make_error("cpu device cannot compile reshape");
        case ir::mid_ir::OpKind::NUM_KINDS:
            return make_error("cpu device cannot compile invalid op kind");
        default:
            break;
    }

    CpuDeviceProgram program;
    program.kind = op.kind;
    program.attrs = op.attrs;
    program.inputDescs.reserve(op.operands.size());
    for (const auto* operand : op.operands)
        program.inputDescs.push_back(core::TensorDesc(operand->shape, operand->dtype));
    program.outputDescs.reserve(op.results.size());
    for (const auto* result : op.results)
        program.outputDescs.push_back(core::TensorDesc(result->shape, result->dtype));

    auto id = nextProgramId_++;
    programs_[id] = std::move(program);
    return id;
}

Result<DeviceBufferId> CpuDevice::alloc(core::TensorDesc desc) {
    auto bytes = tensor_byte_size(desc);
    if (!bytes)
        return make_error(bytes.error());

    CpuDeviceBuffer buffer;
    buffer.desc = std::move(desc);
    buffer.data.resize(bytes.take());

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<void> CpuDevice::dealloc(DeviceBufferId buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    buffers_.erase(it);
    return {};
}

Result<DeviceBufferId> CpuDevice::load(core::TensorBuffer& src) {
    auto access = src.access();
    if (!access)
        return make_error(access.error());

    CpuDeviceBuffer buffer;
    buffer.desc = (*access).desc();
    auto data = (*access).data();
    buffer.data.assign(data.begin(), data.end());

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<TensorBufferPtr> CpuDevice::read(DeviceBufferId src) {
    auto it = buffers_.find(src);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    TensorBufferPtr buffer = std::make_shared<CpuTensorBuffer>(
        it->second.desc,
        it->second.data);
    return buffer;
}

Result<void> CpuDevice::run(
        DeviceProgramId programId,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) {
    auto programIt = programs_.find(programId);
    if (programIt == programs_.end())
        return make_error("cpu device program not found");
    const auto& program = programIt->second;

    if (inputs.size() != program.inputDescs.size())
        return make_error("cpu device input arity mismatch");
    if (outputs.size() != program.outputDescs.size())
        return make_error("cpu device output arity mismatch");

    auto inputRef = [&](size_t index) -> Result<core::TensorRef> {
        auto it = buffers_.find(inputs[index]);
        if (it == buffers_.end())
            return make_error("cpu device input buffer not found");
        return core::make_tensor_ref(it->second.desc, it->second.data);
    };

    auto outputRef = [&](size_t index) -> Result<core::MutableTensorRef> {
        auto it = buffers_.find(outputs[index]);
        if (it == buffers_.end())
            return make_error("cpu device output buffer not found");
        return core::make_mutable_tensor_ref(it->second.desc, it->second.data);
    };

    switch (program.kind) {
        case ir::mid_ir::OpKind::Constant: {
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto value = attr_float_or(program.attrs, "value", 0.0f);
            if (!value) return make_error(value.error());
            out->store_float(0, value.take());
            return {};
        }
        case ir::mid_ir::OpKind::Linear: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto bias = inputRef(2);
            if (!bias) return make_error(bias.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::linear(*x, *weight, *bias, *out);
        }
        case ir::mid_ir::OpKind::ReLU: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::relu(*x, *out);
        }
        case ir::mid_ir::OpKind::Add: {
            auto lhs = inputRef(0);
            if (!lhs) return make_error(lhs.error());
            auto rhs = inputRef(1);
            if (!rhs) return make_error(rhs.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::add(*lhs, *rhs, *out);
        }
        case ir::mid_ir::OpKind::Mul: {
            auto lhs = inputRef(0);
            if (!lhs) return make_error(lhs.error());
            auto rhs = inputRef(1);
            if (!rhs) return make_error(rhs.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::mul(*lhs, *rhs, *out);
        }
        case ir::mid_ir::OpKind::Sqrt: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::sqrt(*x, *out);
        }
        case ir::mid_ir::OpKind::Tanh: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::tanh(*x, *out);
        }
        case ir::mid_ir::OpKind::MatMul: {
            auto lhs = inputRef(0);
            if (!lhs) return make_error(lhs.error());
            auto rhs = inputRef(1);
            if (!rhs) return make_error(rhs.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::matmul(*lhs, *rhs, *out);
        }
        case ir::mid_ir::OpKind::Transpose: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::transpose(*x, *out);
        }
        case ir::mid_ir::OpKind::Permute: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto dims = attr_int_list(program.attrs, "dims");
            if (!dims) return make_error(dims.error());
            auto dimsValue = dims.take();
            return core::permute(*x, dimsValue, *out);
        }
        case ir::mid_ir::OpKind::SlidingQueryKeyScore: {
            auto q = inputRef(0);
            if (!q) return make_error(q.error());
            auto k = inputRef(1);
            if (!k) return make_error(k.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto window = attr_int_or(program.attrs, "window", 0);
            if (!window) return make_error(window.error());
            return core::sliding_query_key_score(*q, *k, window.take(), *out);
        }
        case ir::mid_ir::OpKind::Softmax: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto dim = attr_int_or(program.attrs, "dim", -1);
            if (!dim) return make_error(dim.error());
            return core::softmax(*x, dim.take(), *out);
        }
        case ir::mid_ir::OpKind::Embedding: {
            auto ids = inputRef(0);
            if (!ids) return make_error(ids.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::embedding(*ids, *weight, *out);
        }
        case ir::mid_ir::OpKind::RoPE: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto theta = attr_float_or(program.attrs, "rope_theta", 10000.0f);
            if (!theta) return make_error(theta.error());
            return core::rope(*x, theta.take(), *out);
        }
        case ir::mid_ir::OpKind::RMSNorm: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto epsilon = attr_float_or(program.attrs, "epsilon", 1.0e-6f);
            if (!epsilon) return make_error(epsilon.error());
            return core::rms_norm(*x, *weight, epsilon.take(), *out);
        }
        case ir::mid_ir::OpKind::LayerNorm: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto bias = inputRef(2);
            if (!bias) return make_error(bias.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            auto epsilon = attr_float_or(program.attrs, "epsilon", 1.0e-5f);
            if (!epsilon) return make_error(epsilon.error());
            return core::layer_norm(*x, *weight, *bias, epsilon.take(), *out);
        }
        case ir::mid_ir::OpKind::Input:
        case ir::mid_ir::OpKind::Weight:
        case ir::mid_ir::OpKind::Reshape:
        case ir::mid_ir::OpKind::NUM_KINDS:
            return make_error("cpu device cannot run unsupported op kind");
    }

    return make_error("cpu device cannot run unknown op kind");
}

} // namespace sandy::engine
