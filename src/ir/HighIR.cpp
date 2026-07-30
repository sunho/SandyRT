#include "HighIR.h"
#include <iostream>
#include <sstream>

namespace high_ir {

const char* typeName(Type type) {
    switch (type) {
        case Type::Node: return "node";
        case Type::Int: return "int";
        case Type::Float: return "float";
        case Type::String: return "string";
    }
    return "unknown";
}

Attr Attr::fromInt(const std::string& name, int64_t v) {
    Attr a;
    a.name = name;
    a.type = Type::Int;
    a.intVal = v;
    return a;
}

Attr Attr::fromFloat(const std::string& name, double v) {
    Attr a;
    a.name = name;
    a.type = Type::Float;
    a.floatVal = v;
    return a;
}

Attr Attr::fromString(const std::string& name, const std::string& v) {
    Attr a;
    a.name = name;
    a.type = Type::String;
    a.strVal = v;
    return a;
}

Value Graph::addInput(const std::string& name) {
    Op op;
    op.kind = Op::Input;
    op.inputName = name;
    auto v = newValue(Type::Node);
    op.results.push_back(v);
    ops_.push_back(std::move(op));
    return v;
}

Value Graph::addWeight(const std::string& name) {
    Op op;
    op.kind = Op::Weight;
    op.weightName = name;
    auto v = newValue(Type::Node);
    op.results.push_back(v);
    ops_.push_back(std::move(op));
    return v;
}

Value Graph::addIntConst(int64_t val) {
    Op op;
    op.kind = Op::IntConst;
    op.intVal = val;
    auto v = newValue(Type::Int);
    op.results.push_back(v);
    ops_.push_back(std::move(op));
    return v;
}

Value Graph::addFloatConst(double val) {
    Op op;
    op.kind = Op::FloatConst;
    op.floatVal = val;
    auto v = newValue(Type::Float);
    op.results.push_back(v);
    ops_.push_back(std::move(op));
    return v;
}

Value Graph::addStringConst(const std::string& val) {
    Op op;
    op.kind = Op::StringConst;
    op.strVal = val;
    auto v = newValue(Type::String);
    op.results.push_back(v);
    ops_.push_back(std::move(op));
    return v;
}

std::vector<Value> Graph::addBuiltin(const std::string& name,
                                     const std::vector<Value>& operands,
                                     const std::vector<Attr>& attrs,
                                     int numResults) {
    Op op;
    op.kind = Op::Builtin;
    op.name = name;
    op.operands = operands;
    op.attrs = attrs;
    std::vector<Value> results;
    for (int i = 0; i < numResults; i++) {
        auto v = newValue(Type::Node);
        op.results.push_back(v);
        results.push_back(v);
    }
    ops_.push_back(std::move(op));
    return results;
}

void Graph::setOutputs(const std::vector<Value>& outputs) {
    outputs_ = outputs;
}

static void printValue(const Value& v) {
    std::cout << "%" << v.id;
}

static void printAttrVal(const Attr& a) {
    switch (a.type) {
        case Type::Int: std::cout << a.intVal; break;
        case Type::Float: std::cout << a.floatVal; break;
        case Type::String: std::cout << "\"" << a.strVal << "\""; break;
        default: std::cout << "?"; break;
    }
}

void Graph::dump() const {
    for (auto& op : ops_) {
        switch (op.kind) {
            case Op::Input: {
                printValue(op.results[0]);
                std::cout << " = input(\"" << op.inputName << "\") : node\n";
                break;
            }
            case Op::Weight: {
                printValue(op.results[0]);
                std::cout << " = weight(\"" << op.weightName << "\") : node\n";
                break;
            }
            case Op::IntConst: {
                printValue(op.results[0]);
                std::cout << " = int(" << op.intVal << ")\n";
                break;
            }
            case Op::FloatConst: {
                printValue(op.results[0]);
                std::cout << " = float(" << op.floatVal << ")\n";
                break;
            }
            case Op::StringConst: {
                printValue(op.results[0]);
                std::cout << " = string(\"" << op.strVal << "\")\n";
                break;
            }
            case Op::Builtin: {
                for (size_t i = 0; i < op.results.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    printValue(op.results[i]);
                }
                std::cout << " = builtin(\"" << op.name << "\"";
                for (auto& v : op.operands) {
                    std::cout << ", ";
                    printValue(v);
                }
                for (auto& a : op.attrs) {
                    std::cout << ", " << a.name << "=";
                    printAttrVal(a);
                }
                std::cout << ") : ";
                for (size_t i = 0; i < op.results.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << typeName(op.results[i].type);
                }
                std::cout << "\n";
                break;
            }
        }
    }

    if (!outputs_.empty()) {
        std::cout << "return ";
        for (size_t i = 0; i < outputs_.size(); i++) {
            if (i > 0) std::cout << ", ";
            printValue(outputs_[i]);
        }
        std::cout << "\n";
    }
}

} // namespace high_ir
