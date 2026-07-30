#pragma once

#include "HighIR.h"
#include "Shape.h"

#include <string>
#include <vector>

namespace mid_ir {

struct Value {
    int id;
    ir::Shape shape;
    ir::DType dtype;
};

struct Op {
    enum Kind { Input, Builtin, Weight };
    Kind kind;
    std::vector<Value> results;

    std::string name;
    std::vector<Value> operands;
    std::vector<high_ir::Attr> attrs;

    std::string weightName;
    std::string inputName;
};

class Graph {
public:
    void dump() const;

    const std::vector<Op>& ops() const { return ops_; }
    const std::vector<Value>& outputs() const { return outputs_; }

private:
    std::vector<Op> ops_;
    std::vector<Value> outputs_;
    int nextId_ = 0;

    friend class Builder;
};

} // namespace mid_ir
