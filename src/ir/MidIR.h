#pragma once

#include "Tensor.h"
#include <array>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::ir::mid_ir {

// === OpKind ===

enum class OpKind {
    Input,
    Weight,
    Linear,
    ReLU,
    Add,
    Mul,
    Sqrt,
    MatMul,
    Transpose,
    Reshape,
    Permute,
    SlidingQueryKeyScore,
    Softmax,
    Embedding,
    RMSNorm,

    NUM_KINDS
};

const char* op_kind_name(OpKind kind);

// === Attrs ===

struct AttrValue {
    enum Kind { Int, Float, String, IntList };
    Kind kind;
    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string strVal;
    std::vector<int64_t> intListVal;

    static AttrValue make_int(int64_t v);
    static AttrValue make_float(double v);
    static AttrValue make_string(const std::string& v);
    static AttrValue make_int_list(std::vector<int64_t> v);
};

using AttrMap = std::unordered_map<std::string, AttrValue>;

// === OpDef ===

struct Value;

struct ValueType {
    core::Shape shape;
    core::DType dtype;
};

class OpDef {
public:
    virtual ~OpDef() = default;
    virtual OpKind kind() const = 0;
    virtual const char* name() const = 0;
    virtual std::vector<ValueType> infer_types(
        std::span<Value* const> operands,
        const AttrMap& attrs) const = 0;
    virtual void verify(
        std::span<Value* const> operands,
        const AttrMap& attrs) const {}
};

class OpRegistry {
public:
    static OpRegistry& global();
    void add(const OpDef* def);
    const OpDef* lookup(OpKind kind) const;

private:
    static constexpr int kNumKinds = static_cast<int>(OpKind::NUM_KINDS);
    std::array<const OpDef*, kNumKinds> defs_{};
};

void register_all_ops();

// === IR nodes ===

struct Op;
struct Block;
class Graph;

struct Use {
    Op* op;
    int operand;
};

struct Value {
    int id;
    core::Shape shape;
    core::DType dtype;
    Op* def = nullptr;
    std::vector<Use> uses;
};

struct Op {
    OpKind kind;
    const OpDef* def = nullptr;
    std::vector<Value*> operands;
    std::vector<Value*> results;
    AttrMap attrs;
    Block* parent = nullptr;
};

struct Block {
    std::vector<Op*> ops;
    Graph* parent = nullptr;
};

class Graph {
public:
    Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;

    Block* entry();
    const Block* entry() const;
    const std::vector<Value*>& outputs() const;


    void dump() const;

    friend class Builder;

private:
    std::deque<Value> values_;
    std::deque<Op> ops_;
    std::deque<Block> blocks_;
    std::vector<Value*> outputs_;
    int nextId_ = 0;

    Value* newValue(core::Shape shape, core::DType dtype);
};

// === Builder ===

class Builder {
public:
    Builder(Graph& graph);
    Builder(Graph& graph, Block* block);

    std::vector<Value*> createOp(OpKind kind,
                                 std::span<Value* const> operands,
                                 const AttrMap& attrs = {},
                                 int numResults = 1);

    Value* createInput(const std::string& name, core::Shape shape, core::DType dtype);
    Value* createWeight(const std::string& name, core::Shape shape, core::DType dtype);

    Value* createLinear(Value* x, Value* weight, Value* bias);
    Value* createReLU(Value* x);
    Value* createAdd(Value* lhs, Value* rhs);
    Value* createMul(Value* lhs, Value* rhs);
    Value* createSqrt(Value* x);
    Value* createMatMul(Value* lhs, Value* rhs);
    Value* createTranspose(Value* x);
    Value* createReshape(Value* x, std::vector<int64_t> shape);
    Value* createPermute(Value* x, std::vector<int64_t> dims);
    Value* createSlidingQueryKeyScore(Value* q, Value* k, int64_t window = 0);
    Value* createSoftmax(Value* x, int64_t dim = -1);
    Value* createEmbedding(Value* ids, Value* weight);
    Value* createRMSNorm(Value* x, Value* weight, float epsilon = 1.0e-6f);

    void setOutputs(std::span<Value* const> outputs);

private:
    Graph& graph_;
    Block* block_;
    const OpRegistry& registry_;
};

} // namespace sandy::ir::mid_ir
