#include "CpuInterpreterBackend.h"

#include "MidIR.h"
#include "TensorCalc.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sandy::engine::backend {

namespace {

class CpuProgram final : public Program {
public:
    explicit CpuProgram(const ir::mid_ir::Graph& graph)
        : graph_(&graph) {}

    const ir::mid_ir::Graph& graph() const { return *graph_; }

private:
    const ir::mid_ir::Graph* graph_;
};

class CpuBackendBuffer final : public BackendBuffer {
public:
    CpuBackendBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
        : desc_(std::move(desc)), data_(std::move(data)) {}

    const core::TensorDesc& desc() const override { return desc_; }
    std::span<const uint8_t> data() const override { return data_; }

private:
    core::TensorDesc desc_;
    std::vector<uint8_t> data_;
};

std::string attr_string(const ir::mid_ir::Op& op, const std::string& name) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end() || it->second.kind != ir::mid_ir::AttrValue::String) {
        fprintf(stderr, "missing string attr '%s'\n", name.c_str());
        abort();
    }
    return it->second.strVal;
}

float attr_float_or(const ir::mid_ir::Op& op, const std::string& name, float fallback) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end()) return fallback;
    if (it->second.kind != ir::mid_ir::AttrValue::Float) {
        fprintf(stderr, "attr '%s' must be float\n", name.c_str());
        abort();
    }
    return static_cast<float>(it->second.floatVal);
}

int64_t attr_int_or(const ir::mid_ir::Op& op, const std::string& name, int64_t fallback) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end()) return fallback;
    if (it->second.kind != ir::mid_ir::AttrValue::Int) {
        fprintf(stderr, "attr '%s' must be int\n", name.c_str());
        abort();
    }
    return it->second.intVal;
}

std::vector<int64_t> attr_int_list(const ir::mid_ir::Op& op, const std::string& name) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end() || it->second.kind != ir::mid_ir::AttrValue::IntList) {
        fprintf(stderr, "missing int list attr '%s'\n", name.c_str());
        abort();
    }
    return it->second.intListVal;
}

Result<const BackendBuffer*> lookup_buffer(
        const BackendBufferMap& buffers,
        const std::string& name,
        const std::string& kind) {
    auto it = buffers.find(name);
    if (it == buffers.end())
        return make_error("missing " + kind + " buffer: " + name);
    if (!it->second)
        return make_error("null " + kind + " buffer: " + name);
    return it->second.get();
}

Result<const BackendBuffer*> lookup_value(
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values,
        const ir::mid_ir::Value* value) {
    auto it = values.find(value);
    if (it == values.end())
        return make_error("value not available during cpu interpretation");
    return it->second;
}

std::unique_ptr<CpuBackendBuffer> make_cpu_buffer(core::OwnedTensor tensor) {
    return std::make_unique<CpuBackendBuffer>(
        std::move(tensor.desc), std::move(tensor.data));
}

Result<void> bind_single_result(
        const ir::mid_ir::Op& op,
        const BackendBuffer* buffer,
        std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.results.size() != 1)
        return make_error("cpu interpreter expects single-result ops");
    values[op.results[0]] = buffer;
    return {};
}

Result<core::OwnedTensor> eval_linear(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 3)
        return make_error("linear expects three operands");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());
    auto bias = lookup_value(values, op.operands[2]);
    if (!bias) return make_error(bias.error());

    return core::linear_f32(
        (*x)->data(), (*x)->desc(),
        (*weight)->data(), (*weight)->desc(),
        (*bias)->data(), (*bias)->desc());
}

Result<core::OwnedTensor> eval_relu(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 1)
        return make_error("relu expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    return core::relu_f32((*x)->data(), (*x)->desc());
}

Result<core::OwnedTensor> eval_add(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 2)
        return make_error("add expects two operands");

    auto lhs = lookup_value(values, op.operands[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = lookup_value(values, op.operands[1]);
    if (!rhs) return make_error(rhs.error());

    return core::add_f32(
        (*lhs)->data(), (*lhs)->desc(),
        (*rhs)->data(), (*rhs)->desc());
}

Result<core::OwnedTensor> eval_mul(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 2)
        return make_error("mul expects two operands");

    auto lhs = lookup_value(values, op.operands[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = lookup_value(values, op.operands[1]);
    if (!rhs) return make_error(rhs.error());

    return core::mul_f32(
        (*lhs)->data(), (*lhs)->desc(),
        (*rhs)->data(), (*rhs)->desc());
}

Result<core::OwnedTensor> eval_sqrt(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 1)
        return make_error("sqrt expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    return core::sqrt_f32((*x)->data(), (*x)->desc());
}

Result<core::OwnedTensor> eval_matmul(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 2)
        return make_error("matmul expects two operands");

    auto lhs = lookup_value(values, op.operands[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = lookup_value(values, op.operands[1]);
    if (!rhs) return make_error(rhs.error());

    return core::matmul_f32(
        (*lhs)->data(), (*lhs)->desc(),
        (*rhs)->data(), (*rhs)->desc());
}

Result<core::OwnedTensor> eval_transpose(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 1)
        return make_error("transpose expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    return core::transpose_f32((*x)->data(), (*x)->desc());
}

Result<core::OwnedTensor> eval_reshape(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 1)
        return make_error("reshape expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto shape = attr_int_list(op, "shape");
    return core::reshape_f32((*x)->data(), (*x)->desc(), core::Shape(std::move(shape)));
}

Result<core::OwnedTensor> eval_permute(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 1)
        return make_error("permute expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto dims = attr_int_list(op, "dims");
    return core::permute_f32((*x)->data(), (*x)->desc(), dims);
}

Result<core::OwnedTensor> eval_sliding_query_key_score(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 2)
        return make_error("sliding_query_key_score expects two operands");

    auto q = lookup_value(values, op.operands[0]);
    if (!q) return make_error(q.error());
    auto k = lookup_value(values, op.operands[1]);
    if (!k) return make_error(k.error());

    return core::sliding_query_key_score_f32(
        (*q)->data(), (*q)->desc(),
        (*k)->data(), (*k)->desc(),
        attr_int_or(op, "window", 0));
}

Result<core::OwnedTensor> eval_embedding(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 2)
        return make_error("embedding expects two operands");

    auto ids = lookup_value(values, op.operands[0]);
    if (!ids) return make_error(ids.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());

    return core::embedding_f32(
        (*ids)->data(), (*ids)->desc(),
        (*weight)->data(), (*weight)->desc());
}

Result<core::OwnedTensor> eval_rms_norm(
        const ir::mid_ir::Op& op,
        const std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*>& values) {
    if (op.operands.size() != 2)
        return make_error("rms_norm expects two operands");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());

    return core::rms_norm_f32(
        (*x)->data(), (*x)->desc(),
        (*weight)->data(), (*weight)->desc(),
        attr_float_or(op, "epsilon", 1.0e-6f));
}

Result<void> copy_output(BackendBufferMap& outputs,
                         const std::string& name,
                         const BackendBuffer& source) {
    auto desc = source.desc();
    desc.name = name;
    auto view = source.data();
    outputs[name] = std::make_unique<CpuBackendBuffer>(
        std::move(desc),
        std::vector<uint8_t>(view.begin(), view.end()));
    return {};
}

Result<BackendRunResult> interpret_graph(
        const ir::mid_ir::Graph& graph,
        BackendBufferMap inputs,
        BackendBufferMap weights) {
    std::unordered_map<const ir::mid_ir::Value*, const BackendBuffer*> values;
    std::vector<std::unique_ptr<CpuBackendBuffer>> temporaries;

    for (auto* op : graph.entry()->ops) {
        switch (op->kind) {
            case ir::mid_ir::OpKind::Input: {
                auto name = attr_string(*op, "name");
                auto buffer = lookup_buffer(inputs, name, "input");
                if (!buffer) return make_error(buffer.error());
                auto bind = bind_single_result(*op, *buffer, values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Weight: {
                auto name = attr_string(*op, "name");
                auto buffer = lookup_buffer(weights, name, "weight");
                if (!buffer) return make_error(buffer.error());
                auto bind = bind_single_result(*op, *buffer, values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Linear: {
                auto tensor = eval_linear(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::ReLU: {
                auto tensor = eval_relu(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Add: {
                auto tensor = eval_add(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Mul: {
                auto tensor = eval_mul(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Sqrt: {
                auto tensor = eval_sqrt(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::MatMul: {
                auto tensor = eval_matmul(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Transpose: {
                auto tensor = eval_transpose(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Reshape: {
                auto tensor = eval_reshape(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Permute: {
                auto tensor = eval_permute(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::SlidingQueryKeyScore: {
                auto tensor = eval_sliding_query_key_score(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Embedding: {
                auto tensor = eval_embedding(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::RMSNorm: {
                auto tensor = eval_rms_norm(*op, values);
                if (!tensor) return make_error(tensor.error());
                temporaries.push_back(make_cpu_buffer(tensor.take()));
                auto bind = bind_single_result(*op, temporaries.back().get(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::NUM_KINDS:
                return make_error("invalid op kind");
        }
    }

    BackendRunResult outputs;
    const auto& graphOutputs = graph.outputs();
    for (size_t i = 0; i < graphOutputs.size(); i++) {
        auto value = lookup_value(values, graphOutputs[i]);
        if (!value) return make_error(value.error());
        auto copy = copy_output(outputs, "output" + std::to_string(i), **value);
        if (!copy) return make_error(copy.error());
    }
    return outputs;
}

} // namespace

Result<BackendBufferPtr> CpuInterpreterBackend::create_buffer(core::TensorBuffer& buffer) {
    auto access = buffer.access();
    if (!access)
        return make_error(access.error());

    auto view = (*access).data();
    BackendBufferPtr backendBuffer = std::make_unique<CpuBackendBuffer>(
        (*access).desc(),
        std::vector<uint8_t>(view.begin(), view.end()));
    return backendBuffer;
}

Result<std::unique_ptr<Program>> CpuInterpreterBackend::compile(
        const ir::mid_ir::Graph& graph) {
    std::unique_ptr<Program> program = std::make_unique<CpuProgram>(graph);
    return program;
}

Result<BackendRunResult> CpuInterpreterBackend::run(
        const Program& program,
        BackendBufferMap inputs,
        BackendBufferMap weights,
        const RunOptions&) {
    const auto& cpuProgram = static_cast<const CpuProgram&>(program);
    return interpret_graph(cpuProgram.graph(), std::move(inputs), std::move(weights));
}

} // namespace sandy::engine::backend
