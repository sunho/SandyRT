#include "MidIR.h"
#include "ShapeUtil.h"

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
        case OpKind::Add:       return "add";
        case OpKind::Mul:       return "mul";
        case OpKind::Sqrt:      return "sqrt";
        case OpKind::MatMul:    return "matmul";
        case OpKind::Transpose: return "transpose";
        case OpKind::Reshape:   return "reshape";
        case OpKind::Permute:   return "permute";
        case OpKind::Embedding: return "embedding";
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

AttrValue AttrValue::make_int_list(std::vector<int64_t> v) {
    AttrValue a; a.kind = IntList; a.intListVal = std::move(v); return a;
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

class BinaryElementwiseOpDef : public OpDef {
public:
    BinaryElementwiseOpDef(OpKind kind, const char* name)
        : kind_(kind), name_(name) {}

    OpKind kind() const override { return kind_; }
    const char* name() const override { return name_; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        auto shape = core::broadcast_shape(operands[0]->shape, operands[1]->shape);
        if (!shape) {
            fprintf(stderr, "%s: %s\n", name_, shape.error().c_str());
            abort();
        }
        return {{shape.take(), operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        if (operands.size() != 2) {
            fprintf(stderr, "%s expects 2 operands, got %zu\n",
                    name_, operands.size());
            abort();
        }
        if (operands[0]->dtype != operands[1]->dtype) {
            fprintf(stderr, "%s operands must have the same dtype\n", name_);
            abort();
        }
        auto shape = core::broadcast_shape(operands[0]->shape, operands[1]->shape);
        if (!shape) {
            fprintf(stderr, "%s: %s\n", name_, shape.error().c_str());
            abort();
        }
    }

private:
    OpKind kind_;
    const char* name_;
};

class SqrtOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::Sqrt; }
    const char* name() const override { return "sqrt"; }
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
            fprintf(stderr, "sqrt expects 1 operand, got %zu\n", operands.size());
            abort();
        }
    }
};

class MatMulOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::MatMul; }
    const char* name() const override { return "matmul"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        const auto& lhsShape = operands[0]->shape;
        const auto& rhsShape = operands[1]->shape;
        auto lhsDims = lhsShape.dims();
        auto rhsDims = rhsShape.dims();
        core::Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
        core::Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
        auto batchShape = core::broadcast_shape(lhsBatch, rhsBatch);
        if (!batchShape) {
            fprintf(stderr, "matmul: %s\n", batchShape.error().c_str());
            abort();
        }
        auto outDims = batchShape.take().dims();
        outDims.push_back(lhsShape.dim(lhsShape.rank() - 2));
        outDims.push_back(rhsShape.dim(rhsShape.rank() - 1));
        return {{core::Shape(outDims), operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        if (operands.size() != 2) {
            fprintf(stderr, "matmul expects 2 operands, got %zu\n", operands.size());
            abort();
        }
        if (operands[0]->dtype != operands[1]->dtype) {
            fprintf(stderr, "matmul operands must have the same dtype\n");
            abort();
        }
        if (operands[0]->shape.rank() < 2 || operands[1]->shape.rank() < 2) {
            fprintf(stderr, "matmul operands must have rank >= 2\n");
            abort();
        }

        int64_t lhsK = operands[0]->shape.dim(operands[0]->shape.rank() - 1);
        int64_t rhsK = operands[1]->shape.dim(operands[1]->shape.rank() - 2);
        if (lhsK >= 0 && rhsK >= 0 && lhsK != rhsK) {
            fprintf(stderr, "matmul contracting dimension mismatch\n");
            abort();
        }
        auto lhsDims = operands[0]->shape.dims();
        auto rhsDims = operands[1]->shape.dims();
        core::Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
        core::Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
        auto batchShape = core::broadcast_shape(lhsBatch, rhsBatch);
        if (!batchShape) {
            fprintf(stderr, "matmul: %s\n", batchShape.error().c_str());
            abort();
        }
    }
};

class TransposeOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::Transpose; }
    const char* name() const override { return "transpose"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        auto dims = operands[0]->shape.dims();
        std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
        return {{core::Shape(dims), operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        if (operands.size() != 1) {
            fprintf(stderr, "transpose expects 1 operand, got %zu\n", operands.size());
            abort();
        }
        if (operands[0]->shape.rank() != 2) {
            fprintf(stderr, "transpose input must have rank 2\n");
            abort();
        }
    }
};

class ReshapeOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::Reshape; }
    const char* name() const override { return "reshape"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap& attrs) const override
    {
        return {{core::Shape(attrs.at("shape").intListVal), operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap& attrs) const override
    {
        if (operands.size() != 1) {
            fprintf(stderr, "reshape expects 1 operand, got %zu\n", operands.size());
            abort();
        }

        auto it = attrs.find("shape");
        if (it == attrs.end() || it->second.kind != AttrValue::IntList) {
            fprintf(stderr, "reshape shape attr must be int list\n");
            abort();
        }

        for (int64_t dim : it->second.intListVal) {
            if (dim < core::Shape::kDynamic) {
                fprintf(stderr, "reshape dimensions must be >= -1\n");
                abort();
            }
        }

        core::Shape outShape(it->second.intListVal);
        int64_t inputNumel = operands[0]->shape.numel();
        int64_t outputNumel = outShape.numel();
        if (inputNumel >= 0 && outputNumel >= 0 && inputNumel != outputNumel) {
            fprintf(stderr, "reshape element count mismatch\n");
            abort();
        }
    }
};

class PermuteOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::Permute; }
    const char* name() const override { return "permute"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap& attrs) const override
    {
        const auto& dims = attrs.at("dims").intListVal;
        std::vector<int64_t> outDims;
        outDims.reserve(dims.size());
        for (int64_t axis : dims)
            outDims.push_back(operands[0]->shape.dim(static_cast<int>(axis)));
        return {{core::Shape(outDims), operands[0]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap& attrs) const override
    {
        if (operands.size() != 1) {
            fprintf(stderr, "permute expects 1 operand, got %zu\n", operands.size());
            abort();
        }

        auto it = attrs.find("dims");
        if (it == attrs.end() || it->second.kind != AttrValue::IntList) {
            fprintf(stderr, "permute dims attr must be int list\n");
            abort();
        }

        const auto& dims = it->second.intListVal;
        int rank = operands[0]->shape.rank();
        if (static_cast<int>(dims.size()) != rank) {
            fprintf(stderr, "permute dims size must match input rank\n");
            abort();
        }

        std::vector<bool> seen(static_cast<size_t>(rank), false);
        for (int64_t axis : dims) {
            if (axis < 0 || axis >= rank) {
                fprintf(stderr, "permute axis out of range\n");
                abort();
            }
            auto index = static_cast<size_t>(axis);
            if (seen[index]) {
                fprintf(stderr, "permute dims must not contain duplicates\n");
                abort();
            }
            seen[index] = true;
        }
    }
};

class EmbeddingOpDef : public OpDef {
public:
    OpKind kind() const override { return OpKind::Embedding; }
    const char* name() const override { return "embedding"; }
    std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        auto dims = operands[0]->shape.dims();
        dims.push_back(operands[1]->shape.dim(1));
        return {{core::Shape(dims), operands[1]->dtype}};
    }
    void verify(
        std::span<Value* const> operands,
        const AttrMap&) const override
    {
        if (operands.size() != 2) {
            fprintf(stderr, "embedding expects 2 operands (ids, weight), got %zu\n",
                    operands.size());
            abort();
        }
        if (operands[0]->dtype != core::DType::I32 &&
            operands[0]->dtype != core::DType::I64) {
            fprintf(stderr, "embedding ids must be i32 or i64\n");
            abort();
        }
        if (operands[1]->dtype != core::DType::F32) {
            fprintf(stderr, "embedding weight must be f32\n");
            abort();
        }
        if (operands[1]->shape.rank() != 2) {
            fprintf(stderr, "embedding weight must have rank 2\n");
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
    static BinaryElementwiseOpDef add_def(OpKind::Add, "add");
    static BinaryElementwiseOpDef mul_def(OpKind::Mul, "mul");
    static SqrtOpDef sqrt_def;
    static MatMulOpDef matmul_def;
    static TransposeOpDef transpose_def;
    static ReshapeOpDef reshape_def;
    static PermuteOpDef permute_def;
    static EmbeddingOpDef embedding_def;
    static RMSNormOpDef rms_norm_def;

    auto& reg = OpRegistry::global();
    reg.add(&relu_def);
    reg.add(&linear_def);
    reg.add(&add_def);
    reg.add(&mul_def);
    reg.add(&sqrt_def);
    reg.add(&matmul_def);
    reg.add(&transpose_def);
    reg.add(&reshape_def);
    reg.add(&permute_def);
    reg.add(&embedding_def);
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
        case AttrValue::IntList:
            std::cout << "[";
            for (size_t i = 0; i < a.intListVal.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << a.intListVal[i];
            }
            std::cout << "]";
            break;
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

Value* Builder::createAdd(Value* lhs, Value* rhs) {
    Value* operands[] = {lhs, rhs};
    return createOp(OpKind::Add, operands)[0];
}

Value* Builder::createMul(Value* lhs, Value* rhs) {
    Value* operands[] = {lhs, rhs};
    return createOp(OpKind::Mul, operands)[0];
}

Value* Builder::createSqrt(Value* x) {
    Value* operands[] = {x};
    return createOp(OpKind::Sqrt, operands)[0];
}

Value* Builder::createMatMul(Value* lhs, Value* rhs) {
    Value* operands[] = {lhs, rhs};
    return createOp(OpKind::MatMul, operands)[0];
}

Value* Builder::createTranspose(Value* x) {
    Value* operands[] = {x};
    return createOp(OpKind::Transpose, operands)[0];
}

Value* Builder::createReshape(Value* x, std::vector<int64_t> shape) {
    Value* operands[] = {x};
    AttrMap attrs;
    attrs["shape"] = AttrValue::make_int_list(std::move(shape));
    return createOp(OpKind::Reshape, operands, attrs)[0];
}

Value* Builder::createPermute(Value* x, std::vector<int64_t> dims) {
    Value* operands[] = {x};
    AttrMap attrs;
    attrs["dims"] = AttrValue::make_int_list(std::move(dims));
    return createOp(OpKind::Permute, operands, attrs)[0];
}

Value* Builder::createEmbedding(Value* ids, Value* weight) {
    Value* operands[] = {ids, weight};
    return createOp(OpKind::Embedding, operands)[0];
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
