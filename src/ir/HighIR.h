#pragma once

#include "Tensor.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace sandy::ir::high_ir {

enum class Type { Tensor, TensorTuple, Int, Float, DType, String, IntList };

const char* typeName(Type type);

struct Op;

enum class TensorKind {
    Tensor,
    PagedTensor,
};

struct TensorType {
    TensorKind kind = TensorKind::Tensor;
    std::vector<int64_t> dims;
    std::string dtype;
    int64_t pageSize = -1;
};

struct Value {
    int id;
    Type type;
    TensorType tensorType;
    std::vector<TensorType> tupleElements;
    Op* def = nullptr;
};

enum class InputKind {
    Tensor,
    PagedTensor,
    TensorTuple,
};

struct Attr {
    std::string name;
    Type type;
    int64_t intVal = 0;
    double floatVal = 0.0;
    core::DType dtypeVal = core::DType::F32;
    std::string strVal;
    std::vector<int64_t> intListVal;

    static Attr fromInt(const std::string& name, int64_t v);
    static Attr fromFloat(const std::string& name, double v);
    static Attr fromDType(const std::string& name, core::DType v);
    static Attr fromString(const std::string& name, const std::string& v);
    static Attr fromIntList(const std::string& name, std::vector<int64_t> v);
};

struct Op {
    enum Kind {
        Input,
        Builtin,
        Weight,
        IntConst,
        FloatConst,
        StringConst,
        TensorTupleCreate,
        TensorTupleAppend,
        TensorTupleGet,
    };
    Kind kind;
    std::vector<Value*> results;

    std::string name;
    std::vector<Value*> operands;
    std::vector<Attr> attrs;

    std::string weightName;
    std::string inputName;
    InputKind inputKind = InputKind::Tensor;
    std::vector<int64_t> inputTensorDims;
    std::string inputTensorDType;
    std::vector<int64_t> inputPagedTensorDims;
    std::string inputPagedTensorDType;
    int64_t inputPagedTensorPageSize = -1;
    std::vector<TensorType> inputTensorTupleElements;
    int64_t tupleIndex = -1;

    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string strVal;
};

class Graph {
public:
    Value* addInput(const std::string& name);
    Value* addTensorInput(const std::string& name,
                          std::vector<int64_t> dims,
                          std::string dtype);
    Value* addPagedTensorInput(const std::string& name,
                               std::vector<int64_t> dims,
                               int64_t pageSize);
    Value* addPagedTensorInput(const std::string& name,
                               std::vector<int64_t> dims,
                               std::string dtype,
                               int64_t pageSize);
    Value* addTensorTupleInput(const std::string& name,
                               std::vector<TensorType> elements);
    Value* addWeight(const std::string& name);
    Value* addIntConst(int64_t val);
    Value* addFloatConst(double val);
    Value* addStringConst(const std::string& val);
    std::vector<Value*> addBuiltin(const std::string& name,
                                   const std::vector<Value*>& operands,
                                   const std::vector<Attr>& attrs,
                                   int numResults);
    Value* addTensorTupleCreate(const std::vector<Value*>& elements);
    Value* addTensorTupleAppend(Value* tuple, Value* element);
    Value* addTensorTupleGet(Value* tuple, int64_t index);

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
