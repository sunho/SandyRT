#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace sandy::ir::high_ir {

enum class Type { Tensor, Int, Float, String, IntList };

const char* typeName(Type type);

struct Op;

struct Value {
    int id;
    Type type;
    Op* def = nullptr;
};

enum class InputKind {
    Tensor,
    PagedTensor,
};

struct Attr {
    std::string name;
    Type type;
    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string strVal;
    std::vector<int64_t> intListVal;

    static Attr fromInt(const std::string& name, int64_t v);
    static Attr fromFloat(const std::string& name, double v);
    static Attr fromString(const std::string& name, const std::string& v);
    static Attr fromIntList(const std::string& name, std::vector<int64_t> v);
};

struct Op {
    enum Kind { Input, Builtin, Weight, IntConst, FloatConst, StringConst };
    Kind kind;
    std::vector<Value*> results;

    std::string name;
    std::vector<Value*> operands;
    std::vector<Attr> attrs;

    std::string weightName;
    std::string inputName;
    InputKind inputKind = InputKind::Tensor;
    std::vector<int64_t> inputPagedTensorDims;
    int64_t inputPagedTensorPageSize = -1;

    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string strVal;
};

class Graph {
public:
    Value* addInput(const std::string& name);
    Value* addPagedTensorInput(const std::string& name,
                               std::vector<int64_t> dims,
                               int64_t pageSize);
    Value* addWeight(const std::string& name);
    Value* addIntConst(int64_t val);
    Value* addFloatConst(double val);
    Value* addStringConst(const std::string& val);
    std::vector<Value*> addBuiltin(const std::string& name,
                                   const std::vector<Value*>& operands,
                                   const std::vector<Attr>& attrs,
                                   int numResults);

    void setOutputs(const std::vector<Value*>& outputs);
    void dump() const;

    const std::deque<Op>& ops() const { return ops_; }
    const std::vector<Value*>& outputs() const { return outputs_; }

private:
    std::deque<Value> values_;
    std::deque<Op> ops_;
    std::vector<Value*> outputs_;
    int nextId_ = 0;

    Value* newValue(Type type);
};

} // namespace sandy::ir::high_ir
