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

    NUM_KINDS
};

const char* op_kind_name(OpKind kind);

// === Attrs ===

struct AttrValue {
    enum Kind { Int, Float, String };
    Kind kind;
    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string strVal;

    static AttrValue make_int(int64_t v);
    static AttrValue make_float(double v);
    static AttrValue make_string(const std::string& v);
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

    void setOutputs(std::span<Value* const> outputs);

private:
    Graph& graph_;
    Block* block_;
    const OpRegistry& registry_;
};

} // namespace sandy::ir::mid_ir
