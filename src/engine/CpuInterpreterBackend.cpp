#include "CpuInterpreterBackend.h"

#include "MidIR.h"
#include "ShapeUtil.h"
#include "TensorCalc.h"

#include <cstring>
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
    std::span<uint8_t> mutable_data() { return data_; }

private:
    core::TensorDesc desc_;
    std::vector<uint8_t> data_;
};

class CpuBufferStore {
public:
    size_t add_external(const BackendBuffer* buffer) {
        buffers_.push_back(buffer);
        mutableBuffers_.push_back(nullptr);
        return buffers_.size() - 1;
    }

    size_t add_owned(std::unique_ptr<CpuBackendBuffer> buffer) {
        auto* ptr = buffer.get();
        owned_.push_back(std::move(buffer));
        buffers_.push_back(ptr);
        mutableBuffers_.push_back(ptr);
        return buffers_.size() - 1;
    }

    const BackendBuffer& get(size_t id) const {
        if (id >= buffers_.size() || buffers_[id] == nullptr) {
            fprintf(stderr, "invalid cpu buffer id: %zu\n", id);
            abort();
        }
        return *buffers_[id];
    }

    CpuBackendBuffer& get_mutable(size_t id) {
        if (id >= mutableBuffers_.size() || mutableBuffers_[id] == nullptr) {
            fprintf(stderr, "cpu buffer id is not mutable: %zu\n", id);
            abort();
        }
        return *mutableBuffers_[id];
    }

private:
    std::vector<const BackendBuffer*> buffers_;
    std::vector<CpuBackendBuffer*> mutableBuffers_;
    std::vector<std::unique_ptr<CpuBackendBuffer>> owned_;
};

struct TensorRef {
    size_t bufferId = 0;
    core::TensorDesc desc;
    std::vector<int64_t> strides;
};

using ValueMap = std::unordered_map<const ir::mid_ir::Value*, TensorRef>;

std::vector<int64_t> contiguous_strides(const core::Shape& shape) {
    std::vector<int64_t> strides(static_cast<size_t>(shape.rank()), 1);
    int64_t stride = 1;
    for (int i = shape.rank() - 1; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= shape.dim(i);
    }
    return strides;
}

TensorRef make_ref(size_t bufferId, core::TensorDesc desc) {
    auto strides = contiguous_strides(desc.shape);
    return TensorRef{bufferId, std::move(desc), std::move(strides)};
}

std::span<const uint8_t> ref_data(const CpuBufferStore& store, const TensorRef& ref) {
    return store.get(ref.bufferId).data();
}

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

int64_t attr_int(const ir::mid_ir::Op& op, const std::string& name) {
    auto it = op.attrs.find(name);
    if (it == op.attrs.end() || it->second.kind != ir::mid_ir::AttrValue::Int) {
        fprintf(stderr, "missing int attr '%s'\n", name.c_str());
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

Result<TensorRef> lookup_value(
        const ValueMap& values,
        const ir::mid_ir::Value* value) {
    auto it = values.find(value);
    if (it == values.end())
        return make_error("value not available during cpu interpretation");
    return it->second;
}

Result<void> bind_single_result(
        const ir::mid_ir::Op& op,
        TensorRef ref,
        ValueMap& values) {
    if (op.results.size() != 1)
        return make_error("cpu interpreter expects single-result ops");
    values[op.results[0]] = std::move(ref);
    return {};
}

bool op_allocates_result(ir::mid_ir::OpKind kind) {
    switch (kind) {
        case ir::mid_ir::OpKind::Input:
        case ir::mid_ir::OpKind::Weight:
        case ir::mid_ir::OpKind::Reshape:
        case ir::mid_ir::OpKind::NUM_KINDS:
            return false;
        default:
            return true;
    }
}

Result<core::TensorDesc> single_result_desc(const ir::mid_ir::Op& op) {
    if (op.results.size() != 1)
        return make_error("cpu interpreter expects single-result ops");
    return core::TensorDesc(op.results[0]->shape, op.results[0]->dtype);
}

Result<core::TensorRef> calc_ref(const CpuBufferStore& store, const TensorRef& ref) {
    return core::make_tensor_ref(ref.desc, ref_data(store, ref));
}

Result<core::MutableTensorRef> mutable_calc_ref(CpuBufferStore& store, const TensorRef& ref) {
    auto& buffer = store.get_mutable(ref.bufferId);
    return core::make_mutable_tensor_ref(ref.desc, buffer.mutable_data());
}

Result<void> preallocate_results(
        const ir::mid_ir::Graph& graph,
        CpuBufferStore& store,
        ValueMap& allocations) {
    for (auto* op : graph.entry()->ops) {
        if (!op_allocates_result(op->kind))
            continue;

        auto desc = single_result_desc(*op);
        if (!desc) return make_error(desc.error());

        int64_t numel = desc->shape.numel();
        if (numel < 0)
            return make_error("cpu interpreter result must have static shape");

        std::vector<uint8_t> data(static_cast<size_t>(numel) * core::dtype_size(desc->dtype));
        auto bufferId = store.add_owned(
            std::make_unique<CpuBackendBuffer>(desc.take(), std::move(data)));
        allocations[op->results[0]] = make_ref(bufferId, store.get(bufferId).desc());
    }
    return {};
}

Result<TensorRef> eval_linear(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 3)
        return make_error("linear expects three operands");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());
    auto bias = lookup_value(values, op.operands[2]);
    if (!bias) return make_error(bias.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto weightRef = calc_ref(store, *weight);
    if (!weightRef) return make_error(weightRef.error());
    auto biasRef = calc_ref(store, *bias);
    if (!biasRef) return make_error(biasRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::linear(*xRef, *weightRef, *biasRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_relu(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("relu expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::relu(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_add(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 2)
        return make_error("add expects two operands");

    auto lhs = lookup_value(values, op.operands[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = lookup_value(values, op.operands[1]);
    if (!rhs) return make_error(rhs.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto lhsRef = calc_ref(store, *lhs);
    if (!lhsRef) return make_error(lhsRef.error());
    auto rhsRef = calc_ref(store, *rhs);
    if (!rhsRef) return make_error(rhsRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::add(*lhsRef, *rhsRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_mul(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 2)
        return make_error("mul expects two operands");

    auto lhs = lookup_value(values, op.operands[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = lookup_value(values, op.operands[1]);
    if (!rhs) return make_error(rhs.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto lhsRef = calc_ref(store, *lhs);
    if (!lhsRef) return make_error(lhsRef.error());
    auto rhsRef = calc_ref(store, *rhs);
    if (!rhsRef) return make_error(rhsRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::mul(*lhsRef, *rhsRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_sqrt(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("sqrt expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::sqrt(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_tanh(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("tanh expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::tanh(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_matmul(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 2)
        return make_error("matmul expects two operands");

    auto lhs = lookup_value(values, op.operands[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = lookup_value(values, op.operands[1]);
    if (!rhs) return make_error(rhs.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto lhsRef = calc_ref(store, *lhs);
    if (!lhsRef) return make_error(lhsRef.error());
    auto rhsRef = calc_ref(store, *rhs);
    if (!rhsRef) return make_error(rhsRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::matmul(*lhsRef, *rhsRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_transpose(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("transpose expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::transpose(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_reshape(
        const ir::mid_ir::Op& op,
        const ValueMap& values) {
    if (op.operands.size() != 1)
        return make_error("reshape expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto shape = attr_int_list(op, "shape");
    auto inferred = core::infer_reshape_shape(x->desc.shape, core::Shape(std::move(shape)));
    if (!inferred) return make_error(inferred.error());

    auto desc = x->desc;
    desc.shape = inferred.take();
    return make_ref(x->bufferId, std::move(desc));
}

Result<TensorRef> eval_permute(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("permute expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto dims = attr_int_list(op, "dims");
    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::permute(*xRef, dims, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_sliding_query_key_score(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 2)
        return make_error("sliding_query_key_score expects two operands");

    auto q = lookup_value(values, op.operands[0]);
    if (!q) return make_error(q.error());
    auto k = lookup_value(values, op.operands[1]);
    if (!k) return make_error(k.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto qRef = calc_ref(store, *q);
    if (!qRef) return make_error(qRef.error());
    auto kRef = calc_ref(store, *k);
    if (!kRef) return make_error(kRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::sliding_query_key_score(
        *qRef, *kRef, attr_int_or(op, "window", 0), *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_softmax(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("softmax expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::softmax(*xRef, attr_int_or(op, "dim", -1), *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_embedding(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 2)
        return make_error("embedding expects two operands");

    auto ids = lookup_value(values, op.operands[0]);
    if (!ids) return make_error(ids.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto idsRef = calc_ref(store, *ids);
    if (!idsRef) return make_error(idsRef.error());
    auto weightRef = calc_ref(store, *weight);
    if (!weightRef) return make_error(weightRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::embedding(*idsRef, *weightRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_rope(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 1)
        return make_error("rope expects one operand");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::rope(*xRef, attr_float_or(op, "rope_theta", 10000.0f), *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_rms_norm(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 2)
        return make_error("rms_norm expects two operands");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto weightRef = calc_ref(store, *weight);
    if (!weightRef) return make_error(weightRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::rms_norm(
        *xRef, *weightRef, attr_float_or(op, "epsilon", 1.0e-6f), *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TensorRef> eval_layer_norm(
        const ir::mid_ir::Op& op,
        const ValueMap& values,
        const ValueMap& allocations,
        CpuBufferStore& store) {
    if (op.operands.size() != 3)
        return make_error("layer_norm expects three operands");

    auto x = lookup_value(values, op.operands[0]);
    if (!x) return make_error(x.error());
    auto weight = lookup_value(values, op.operands[1]);
    if (!weight) return make_error(weight.error());
    auto bias = lookup_value(values, op.operands[2]);
    if (!bias) return make_error(bias.error());

    auto out = lookup_value(allocations, op.results[0]);
    if (!out) return make_error(out.error());
    auto xRef = calc_ref(store, *x);
    if (!xRef) return make_error(xRef.error());
    auto weightRef = calc_ref(store, *weight);
    if (!weightRef) return make_error(weightRef.error());
    auto biasRef = calc_ref(store, *bias);
    if (!biasRef) return make_error(biasRef.error());
    auto outRef = mutable_calc_ref(store, *out);
    if (!outRef) return make_error(outRef.error());
    auto result = core::layer_norm(
        *xRef, *weightRef, *biasRef, attr_float_or(op, "epsilon", 1.0e-5f), *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<void> copy_output(BackendBufferMap& outputs,
                         const std::string& name,
                         const TensorRef& source,
                         const CpuBufferStore& store) {
    auto desc = source.desc;
    desc.name = name;
    auto view = ref_data(store, source);
    outputs[name] = std::make_unique<CpuBackendBuffer>(
        std::move(desc),
        std::vector<uint8_t>(view.begin(), view.end()));
    return {};
}

Result<BackendRunResult> interpret_graph(
        const ir::mid_ir::Graph& graph,
        BackendBufferMap inputs,
        BackendBufferMap weights) {
    ValueMap values;
    ValueMap allocations;
    CpuBufferStore store;

    auto allocate = preallocate_results(graph, store, allocations);
    if (!allocate) return make_error(allocate.error());

    for (auto* op : graph.entry()->ops) {
        switch (op->kind) {
            case ir::mid_ir::OpKind::Input: {
                auto name = std::to_string(attr_int(*op, "index"));
                auto buffer = lookup_buffer(inputs, name, "input");
                if (!buffer) return make_error(buffer.error());
                auto bufferId = store.add_external(*buffer);
                auto bind = bind_single_result(
                    *op, make_ref(bufferId, (*buffer)->desc()), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Weight: {
                auto name = attr_string(*op, "name");
                auto buffer = lookup_buffer(weights, name, "weight");
                if (!buffer) return make_error(buffer.error());
                auto bufferId = store.add_external(*buffer);
                auto bind = bind_single_result(
                    *op, make_ref(bufferId, (*buffer)->desc()), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Constant: {
                auto ref = lookup_value(allocations, op->results[0]);
                if (!ref) return make_error(ref.error());
                auto outRef = mutable_calc_ref(store, *ref);
                if (!outRef) return make_error(outRef.error());
                outRef->store_float(0, attr_float_or(*op, "value", 0.0f));
                auto bind = bind_single_result(
                    *op, ref.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Linear: {
                auto tensor = eval_linear(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::ReLU: {
                auto tensor = eval_relu(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Add: {
                auto tensor = eval_add(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Mul: {
                auto tensor = eval_mul(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Sqrt: {
                auto tensor = eval_sqrt(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Tanh: {
                auto tensor = eval_tanh(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::MatMul: {
                auto tensor = eval_matmul(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Transpose: {
                auto tensor = eval_transpose(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Reshape: {
                auto ref = eval_reshape(*op, values);
                if (!ref) return make_error(ref.error());
                auto bind = bind_single_result(*op, ref.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Permute: {
                auto tensor = eval_permute(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::SlidingQueryKeyScore: {
                auto tensor = eval_sliding_query_key_score(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Softmax: {
                auto tensor = eval_softmax(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::Embedding: {
                auto tensor = eval_embedding(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::RoPE: {
                auto tensor = eval_rope(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::RMSNorm: {
                auto tensor = eval_rms_norm(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
                if (!bind) return make_error(bind.error());
                break;
            }
            case ir::mid_ir::OpKind::LayerNorm: {
                auto tensor = eval_layer_norm(*op, values, allocations, store);
                if (!tensor) return make_error(tensor.error());
                auto bind = bind_single_result(*op, tensor.take(), values);
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
        auto copy = copy_output(outputs, "output" + std::to_string(i), *value, store);
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
