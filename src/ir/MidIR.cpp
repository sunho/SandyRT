#include "MidIR.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace sandy::ir::mid_ir {

// === OpKind ===

const char* op_kind_name(OpKind kind) {
    switch (kind) {
        case OpKind::Input:     return "input";
        case OpKind::Weight:    return "weight";
        case OpKind::Linear:    return "linear";
        case OpKind::ReLU:      return "relu";
        case OpKind::RMSNorm:   return "rms_norm";
        case OpKind::NUM_KINDS: return "?";
    }
    return "?";
}

// === AttrValue ===

AttrValue AttrValue::make_int(int64_t v) {
    AttrValue a; a.kind = Int; a.intVal = v; return a;
}

AttrValue AttrValue::make_float(double v) {
    AttrValue a; a.kind = Float; a.floatVal = v; return a;
}

AttrValue AttrValue::make_string(const std::string& v) {
    AttrValue a; a.kind = String; a.strVal = v; return a;
}

// === OpRegistry ===

OpRegistry& OpRegistry::global() {
    static OpRegistry instance;
    return instance;
}

void OpRegistry::add(const OpDef* def) {
    defs_[static_cast<int>(def->kind())] = def;
}

const OpDef* OpRegistry::lookup(OpKind kind) const {
    return defs_[static_cast<int>(kind)];
}

// === Concrete OpDefs ===

namespace {

class ReLUOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::ReLU; }
    const char* name() const override { return "relu"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        return {{operands[0]->shape, operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        if (operands.size() != 1) {
            fprintf(stderr, "relu expects 1 operand, got %zu\n", operands.size());
            abort();
        }
    }
};

class LinearOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::Linear; }
    const char* name() const override { return "linear"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        // y = x @ W^T + b
        // x: [..., in_features], weight: [out_features, in_features]
        // result: [..., out_features]
        auto dims = operands[0]->shape.dims();
        dims.back() = operands[1]->shape.dim(0);
        return {{core::Shape(dims), operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        if (operands.size() != 3) {
            fprintf(stderr, "linear expects 3 operands (x, weight, bias), got %zu\n",
                    operands.size());
            abort();
        }
    }
};

class RMSNormOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::RMSNorm; }
    const char* name() const override { return "rms_norm"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        return {{operands[0]->shape, operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap& attrs) const override
    {
        if (operands.size() != 2) {
            fprintf(stderr, "rms_norm expects 2 operands (x, weight), got %zu\n",
                    operands.size());
            abort();
        }
        auto epsilon = attrs.find("epsilon");
        if (epsilon != attrs.end() && epsilon->second.kind != AttrValue::Float) {
            fprintf(stderr, "rms_norm epsilon attr must be float\n");
            abort();
        }
        if (operands[0]->shape.rank() < 1) {
            fprintf(stderr, "rms_norm input must have rank >= 1\n");
            abort();
        }
        if (operands[1]->shape.rank() != 1) {
            fprintf(stderr, "rms_norm weight must have rank 1\n");
            abort();
        }
        int64_t hidden = operands[0]->shape.dim(operands[0]->shape.rank() - 1);
        int64_t weightDim = operands[1]->shape.dim(0);
        if (hidden >= 0 && weightDim >= 0 && hidden != weightDim) {
            fprintf(stderr, "rms_norm weight dimension mismatch\n");
            abort();
        }
    }
};

} // anonymous namespace

void register_all_ops() {
    static ReLUOpDef relu_def;
    static LinearOpDef linear_def;
    static RMSNormOpDef rms_norm_def;

    auto& reg = OpRegistry::global();
    reg.add(&relu_def);
    reg.add(&linear_def);
    reg.add(&rms_norm_def);
}

// === Graph ===

Graph::Graph() {
    auto& block = blocks_.emplace_back();
    block.parent = this;
}

Block* Graph::entry() { return &blocks_.front(); }
const Block* Graph::entry() const { return &blocks_.front(); }
const std::vector<Value*>& Graph::outputs() const { return outputs_; }

Value* Graph::newValue(core::Shape shape, core::DType dtype) {
    auto& v = values_.emplace_back();
    v.id = nextId_++;
    v.shape = std::move(shape);
    v.dtype = dtype;
    return &v;
}

static void printAttrVal(const AttrValue& a) {
    switch (a.kind) {
        case AttrValue::Int:    std::cout << a.intVal; break;
        case AttrValue::Float:  std::cout << a.floatVal; break;
        case AttrValue::String: std::cout << "\"" << a.strVal << "\""; break;
    }
}

void Graph::dump() const {
    for (auto* op : entry()->ops) {
        for (size_t i = 0; i < op->results.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << "%" << op->results[i]->id;
        }
        std::cout << " = " << op_kind_name(op->kind) << "(";

        bool first = true;
        for (auto* v : op->operands) {
            if (!first) std::cout << ", ";
            std::cout << "%" << v->id;
            first = false;
        }
        for (auto& [aname, val] : op->attrs) {
            if (!first) std::cout << ", ";
            std::cout << aname << "=";
            printAttrVal(val);
            first = false;
        }
        std::cout << ") : ";

        for (size_t i = 0; i < op->results.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << core::dtype_name(op->results[i]->dtype)
                      << op->results[i]->shape.str();
        }
        std::cout << "\n";
    }

    if (!outputs_.empty()) {
        std::cout << "return ";
        for (size_t i = 0; i < outputs_.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << "%" << outputs_[i]->id;
        }
        std::cout << "\n";
    }
}

// === Builder ===

Builder::Builder(Graph& graph)
    : graph_(graph), block_(graph.entry()), registry_(OpRegistry::global()) {}

Builder::Builder(Graph& graph, Block* block)
    : graph_(graph), block_(block), registry_(OpRegistry::global()) {}

std::vector<Value*> Builder::createOp(OpKind kind,
                                       std::span<Value* const> operands,
                                       const AttrMap& attrs,
                                       int numResults) {
    const OpDef* def = registry_.lookup(kind);
    if (!def) {
        fprintf(stderr, "no OpDef registered for %s\n", op_kind_name(kind));
        abort();
    }

    def->verify(operands, attrs);
    auto result_types = def->infer_types(operands, attrs);

    auto& op = graph_.ops_.emplace_back();
    op.kind = kind;
    op.def = def;
    op.operands.assign(operands.begin(), operands.end());
    op.attrs = attrs;
    op.parent = block_;

    for (int i = 0; i < (int)operands.size(); i++)
        operands[i]->uses.push_back({&op, i});

    std::vector<Value*> results;
    for (auto& rt : result_types) {
        auto* v = graph_.newValue(std::move(rt.shape), rt.dtype);
        v->def = &op;
        op.results.push_back(v);
        results.push_back(v);
    }

    block_->ops.push_back(&op);
    return results;
}

Value* Builder::createInput(const std::string& name, core::Shape shape, core::DType dtype) {
    auto& op = graph_.ops_.emplace_back();
    op.kind = OpKind::Input;
    op.attrs["name"] = AttrValue::make_string(name);
    op.parent = block_;

    auto* v = graph_.newValue(std::move(shape), dtype);
    v->def = &op;
    op.results.push_back(v);

    block_->ops.push_back(&op);
    return v;
}

Value* Builder::createWeight(const std::string& name, core::Shape shape, core::DType dtype) {
    auto& op = graph_.ops_.emplace_back();
    op.kind = OpKind::Weight;
    op.attrs["name"] = AttrValue::make_string(name);
    op.parent = block_;

    auto* v = graph_.newValue(std::move(shape), dtype);
    v->def = &op;
    op.results.push_back(v);

    block_->ops.push_back(&op);
    return v;
}

Value* Builder::createLinear(Value* x, Value* weight, Value* bias) {
    Value* operands[] = {x, weight, bias};
    return createOp(OpKind::Linear, operands)[0];
}

Value* Builder::createReLU(Value* x) {
    Value* operands[] = {x};
    return createOp(OpKind::ReLU, operands)[0];
}

Value* Builder::createRMSNorm(Value* x, Value* weight, float epsilon) {
    Value* operands[] = {x, weight};
    AttrMap attrs;
    attrs["epsilon"] = AttrValue::make_float(epsilon);
    return createOp(OpKind::RMSNorm, operands, attrs)[0];
}

void Builder::setOutputs(std::span<Value* const> outputs) {
    graph_.outputs_.assign(outputs.begin(), outputs.end());
}

} // namespace sandy::ir::mid_ir
