#pragma once

#include "AST.h"
#include "HighIR.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace sandygo {

struct RuntimeValue {
    enum Kind { Void, Int, Float, String, NodeVal, Tuple };
    Kind kind = Void;
    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string strVal;
    high_ir::Value nodeVal = {-1, high_ir::Type::Node};
    std::vector<RuntimeValue> tupleVals;

    static RuntimeValue makeInt(int64_t v) {
        RuntimeValue rv; rv.kind = Int; rv.intVal = v; return rv;
    }
    static RuntimeValue makeFloat(double v) {
        RuntimeValue rv; rv.kind = Float; rv.floatVal = v; return rv;
    }
    static RuntimeValue makeString(const std::string& v) {
        RuntimeValue rv; rv.kind = String; rv.strVal = v; return rv;
    }
    static RuntimeValue makeNode(high_ir::Value v) {
        RuntimeValue rv; rv.kind = NodeVal; rv.nodeVal = v; return rv;
    }
    static RuntimeValue makeTuple(std::vector<RuntimeValue> vals) {
        RuntimeValue rv; rv.kind = Tuple; rv.tupleVals = std::move(vals); return rv;
    }
    static RuntimeValue makeVoid() { return RuntimeValue{}; }
};

class Interpreter {
public:
    Interpreter(const Program& program, high_ir::Graph& graph);
    void interpret();

private:
    const Program& program_;
    high_ir::Graph& graph_;

    std::unordered_map<std::string, const FuncDecl*> funcTable_;

    using Env = std::unordered_map<std::string, RuntimeValue>;
    std::vector<Env> envStack_;

    std::vector<std::string> weightScope_;

    bool hasReturn_ = false;
    RuntimeValue returnValue_;
    int expectedResults_ = 1;

    RuntimeValue evalExpr(const Expr& expr);
    RuntimeValue evalCall(const Expr& expr);
    RuntimeValue evalBinary(const Expr& expr);
    RuntimeValue evalUnary(const Expr& expr);

    void execStmt(const Stmt& stmt);
    void execBlock(const std::vector<StmtPtr>& stmts);
    void execAssign(const Stmt& stmt);
    void execVarDecl(const Stmt& stmt);
    void execReturn(const Stmt& stmt);
    void execFor(const Stmt& stmt);
    void execIf(const Stmt& stmt);
    void execWeightScope(const Stmt& stmt);

    RuntimeValue callFunc(const FuncDecl& func, const std::vector<RuntimeValue>& args);

    void setVar(const std::string& name, const RuntimeValue& val);
    RuntimeValue getVar(const std::string& name);
    void pushEnv();
    void popEnv();

    std::string resolveWeight(const std::string& localName);
    std::string interpolateString(const std::string& s);
    high_ir::Value toGraphValue(const RuntimeValue& val);

    [[noreturn]] void error(const std::string& msg);
};

} // namespace sandygo
