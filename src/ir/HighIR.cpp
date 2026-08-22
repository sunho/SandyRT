#include "HighIR.h"
#include <iostream>
#include <utility>

namespace sandy::ir::high_ir {

const char* typeName(Type type) {
    switch (type) {
        case Type::Tensor: return "tensor";
        case Type::TensorTuple: return "tensor_tuple";
        case Type::Int: return "int";
        case Type::Float: return "float";
        case Type::DType: return "dtype";
        case Type::String: return "string";
        case Type::IntList: return "int_list";
    }
    return "unknown";
}

Attr Attr::fromInt(const std::string& name, int64_t v) {
    Attr a; a.name = name; a.type = Type::Int; a.intVal = v; return a;
}

Attr Attr::fromFloat(const std::string& name, double v) {
    Attr a; a.name = name; a.type = Type::Float; a.floatVal = v; return a;
}

Attr Attr::fromDType(const std::string& name, core::DType v) {
    Attr a; a.name = name; a.type = Type::DType; a.dtypeVal = v; return a;
}

Attr Attr::fromString(const std::string& name, const std::string& v) {
    Attr a; a.name = name; a.type = Type::String; a.strVal = v; return a;
}

Attr Attr::fromIntList(const std::string& name, std::vector<int64_t> v) {
    Attr a; a.name = name; a.type = Type::IntList; a.intListVal = std::move(v); return a;
}

Value* Graph::newValue(Type type) {
    auto& v = values_.emplace_back();
    v.id = nextId_++;
    v.type = type;
    return &v;
}

Value* Graph::addInput(const std::string& name) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Input;
    op.inputName = name;
    op.inputKind = InputKind::Tensor;
    auto* v = newValue(Type::Tensor);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addTensorInput(
        const std::string& name,
        std::vector<int64_t> dims,
        std::string dtype) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Input;
    op.inputName = name;
    op.inputKind = InputKind::Tensor;
    op.inputTensorDims = std::move(dims);
    op.inputTensorDType = std::move(dtype);
    auto* v = newValue(Type::Tensor);
    v->tensorType.kind = TensorKind::Tensor;
    v->tensorType.dims = op.inputTensorDims;
    v->tensorType.dtype = op.inputTensorDType;
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addPagedTensorInput(
        const std::string& name,
        std::vector<int64_t> dims,
        int64_t pageSize) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Input;
    op.inputName = name;
    op.inputKind = InputKind::PagedTensor;
    op.inputPagedTensorDims = std::move(dims);
    op.inputPagedTensorPageSize = pageSize;
    auto* v = newValue(Type::Tensor);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addPagedTensorInput(
        const std::string& name,
        std::vector<int64_t> dims,
        std::string dtype,
        int64_t pageSize) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Input;
    op.inputName = name;
    op.inputKind = InputKind::PagedTensor;
    op.inputPagedTensorDims = std::move(dims);
    op.inputPagedTensorDType = std::move(dtype);
    op.inputPagedTensorPageSize = pageSize;
    auto* v = newValue(Type::Tensor);
    v->tensorType.kind = TensorKind::PagedTensor;
    v->tensorType.dims = op.inputPagedTensorDims;
    v->tensorType.dtype = op.inputPagedTensorDType;
    v->tensorType.pageSize = op.inputPagedTensorPageSize;
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addTensorTupleInput(
        const std::string& name,
        std::vector<TensorType> elements) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Input;
    op.inputName = name;
    op.inputKind = InputKind::TensorTuple;
    op.inputTensorTupleElements = std::move(elements);
    auto* v = newValue(Type::TensorTuple);
    v->tupleElements = op.inputTensorTupleElements;
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addWeight(const std::string& name) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Weight;
    op.weightName = name;
    auto* v = newValue(Type::Tensor);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addIntConst(int64_t val) {
    auto& op = ops_.emplace_back();
    op.kind = Op::IntConst;
    op.intVal = val;
    auto* v = newValue(Type::Int);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addFloatConst(double val) {
    auto& op = ops_.emplace_back();
    op.kind = Op::FloatConst;
    op.floatVal = val;
    auto* v = newValue(Type::Float);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addStringConst(const std::string& val) {
    auto& op = ops_.emplace_back();
    op.kind = Op::StringConst;
    op.strVal = val;
    auto* v = newValue(Type::String);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

std::vector<Value*> Graph::addBuiltin(const std::string& name,
                                      const std::vector<Value*>& operands,
                                      const std::vector<Attr>& attrs,
                                      int numResults) {
    auto& op = ops_.emplace_back();
    op.kind = Op::Builtin;
    op.name = name;
    op.operands = operands;
    op.attrs = attrs;
    std::vector<Value*> results;
    for (int i = 0; i < numResults; i++) {
        auto* v = newValue(Type::Tensor);
        v->def = &op;
        op.results.push_back(v);
        results.push_back(v);
    }
    return results;
}

Value* Graph::addTensorTupleCreate(const std::vector<Value*>& elements) {
    auto& op = ops_.emplace_back();
    op.kind = Op::TensorTupleCreate;
    op.operands = elements;
    auto* v = newValue(Type::TensorTuple);
    v->tupleElements.reserve(elements.size());
    for (auto* element : elements)
        v->tupleElements.push_back(element->tensorType);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addTensorTupleAppend(Value* tuple, Value* element) {
    auto& op = ops_.emplace_back();
    op.kind = Op::TensorTupleAppend;
    op.operands = {tuple, element};
    auto* v = newValue(Type::TensorTuple);
    v->tupleElements = tuple->tupleElements;
    v->tupleElements.push_back(element->tensorType);
    v->def = &op;
    op.results.push_back(v);
    return v;
}

Value* Graph::addTensorTupleGet(Value* tuple, int64_t index) {
    auto& op = ops_.emplace_back();
    op.kind = Op::TensorTupleGet;
    op.operands = {tuple};
    op.tupleIndex = index;
    auto* v = newValue(Type::Tensor);
    if (index >= 0 && static_cast<size_t>(index) < tuple->tupleElements.size())
        v->tensorType = tuple->tupleElements[static_cast<size_t>(index)];
    v->def = &op;
    op.results.push_back(v);
    return v;
}

void Graph::setOutputs(const std::vector<Value*>& outputs) {
    outputs_ = outputs;
}

static void printAttrVal(const Attr& a) {
    switch (a.type) {
        case Type::Int: std::cout << a.intVal; break;
        case Type::Float: std::cout << a.floatVal; break;
        case Type::DType: std::cout << core::dtype_name(a.dtypeVal); break;
        case Type::String: std::cout << "\"" << a.strVal << "\""; break;
        case Type::IntList:
            std::cout << "[";
            for (size_t i = 0; i < a.intListVal.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << a.intListVal[i];
            }
            std::cout << "]";
            break;
        default: std::cout << "?"; break;
    }
}

void Graph::dump() const {
    for (auto& op : ops_) {
        switch (op.kind) {
            case Op::Input:
                std::cout << "%" << op.results[0]->id
                          << " = ";
                if (op.inputKind == InputKind::TensorTuple) {
                    std::cout << "tensor_tuple_input(\"" << op.inputName
                              << "\", len=" << op.inputTensorTupleElements.size();
                } else if (op.inputKind == InputKind::PagedTensor) {
                    std::cout << "paged_tensor_input(\"" << op.inputName << "\", dims=";
                    std::cout << "[";
                    for (size_t i = 0; i < op.inputPagedTensorDims.size(); i++) {
                        if (i > 0) std::cout << ", ";
                        std::cout << op.inputPagedTensorDims[i];
                    }
                    std::cout << "]";
                    if (!op.inputPagedTensorDType.empty())
                        std::cout << ", dtype=" << op.inputPagedTensorDType;
                    std::cout << ", page_size=" << op.inputPagedTensorPageSize;
                } else {
                    std::cout << "input(\"" << op.inputName << "\"";
                    if (!op.inputTensorDims.empty()) {
                        std::cout << ", dims=[";
                        for (size_t i = 0; i < op.inputTensorDims.size(); i++) {
                            if (i > 0) std::cout << ", ";
                            std::cout << op.inputTensorDims[i];
                        }
                        std::cout << "]";
                    }
                    if (!op.inputTensorDType.empty())
                        std::cout << ", dtype=" << op.inputTensorDType;
                }
                std::cout << ") : tensor\n";
                break;
            case Op::Weight:
                std::cout << "%" << op.results[0]->id
                          << " = weight(\"" << op.weightName << "\") : tensor\n";
                break;
            case Op::IntConst:
                std::cout << "%" << op.results[0]->id
                          << " = int(" << op.intVal << ")\n";
                break;
            case Op::FloatConst:
                std::cout << "%" << op.results[0]->id
                          << " = float(" << op.floatVal << ")\n";
                break;
            case Op::StringConst:
                std::cout << "%" << op.results[0]->id
                          << " = string(\"" << op.strVal << "\")\n";
                break;
            case Op::Builtin:
                for (size_t i = 0; i < op.results.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << "%" << op.results[i]->id;
                }
                std::cout << " = builtin(\"" << op.name << "\"";
                for (auto* v : op.operands)
                    std::cout << ", %" << v->id;
                for (auto& a : op.attrs) {
                    std::cout << ", " << a.name << "=";
                    printAttrVal(a);
                }
                std::cout << ") : ";
                for (size_t i = 0; i < op.results.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << typeName(op.results[i]->type);
                }
                std::cout << "\n";
                break;
            case Op::TensorTupleCreate:
                std::cout << "%" << op.results[0]->id << " = tensor_tuple_create(";
                for (size_t i = 0; i < op.operands.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << "%" << op.operands[i]->id;
                }
                std::cout << ") : tensor_tuple\n";
                break;
            case Op::TensorTupleAppend:
                std::cout << "%" << op.results[0]->id << " = tensor_tuple_append(%"
                          << op.operands[0]->id << ", %" << op.operands[1]->id
                          << ") : tensor_tuple\n";
                break;
            case Op::TensorTupleGet:
                std::cout << "%" << op.results[0]->id << " = tensor_tuple_get(%"
                          << op.operands[0]->id << ", index=" << op.tupleIndex
                          << ") : tensor\n";
                break;
        }
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

} // namespace sandy::ir::high_ir
