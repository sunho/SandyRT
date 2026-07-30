#include "Interpreter.h"
#include <cstdio>
#include <cstdlib>

namespace sandygo {

Interpreter::Interpreter(const Program& program, high_ir::Graph& graph)
    : program_(program), graph_(graph) {
    for (auto& func : program_.funcs) {
        funcTable_[func.name] = &func;
    }
}

void Interpreter::interpret() {
    auto it = funcTable_.find("main");
    if (it == funcTable_.end()) error("no 'main' function found");

    const FuncDecl& mainFunc = *it->second;

    std::vector<RuntimeValue> mainArgs;
    for (auto& param : mainFunc.params) {
        auto input = graph_.addInput(param.name);
        mainArgs.push_back(RuntimeValue::makeNode(input));
    }

    RuntimeValue result = callFunc(mainFunc, mainArgs);

    if (result.kind == RuntimeValue::Tuple) {
        std::vector<high_ir::Value> outputs;
        for (auto& v : result.tupleVals)
            outputs.push_back(toGraphValue(v));
        graph_.setOutputs(outputs);
    } else if (result.kind == RuntimeValue::NodeVal) {
        graph_.setOutputs({result.nodeVal});
    }
}

RuntimeValue Interpreter::callFunc(const FuncDecl& func,
                                   const std::vector<RuntimeValue>& args) {
    if (args.size() != func.params.size())
        error("argument count mismatch for '" + func.name + "'");

    bool savedReturn = hasReturn_;
    RuntimeValue savedReturnVal = returnValue_;

    pushEnv();
    for (size_t i = 0; i < args.size(); i++)
        setVar(func.params[i].name, args[i]);

    hasReturn_ = false;
    returnValue_ = RuntimeValue::makeVoid();

    execBlock(func.body);

    RuntimeValue result = returnValue_;

    popEnv();
    hasReturn_ = savedReturn;
    returnValue_ = savedReturnVal;

    return result;
}

void Interpreter::execBlock(const std::vector<StmtPtr>& stmts) {
    for (auto& stmt : stmts) {
        if (hasReturn_) return;
        execStmt(*stmt);
    }
}

void Interpreter::execStmt(const Stmt& stmt) {
    switch (stmt.kind) {
        case Stmt::Assign:      execAssign(stmt); break;
        case Stmt::VarDecl:     execVarDecl(stmt); break;
        case Stmt::Return:      execReturn(stmt); break;
        case Stmt::For:         execFor(stmt); break;
        case Stmt::If:          execIf(stmt); break;
        case Stmt::WeightScope: execWeightScope(stmt); break;
        case Stmt::ExprStmt:    evalExpr(*stmt.expr); break;
    }
}

void Interpreter::execAssign(const Stmt& stmt) {
    int numTargets = (int)stmt.targets.size();

    int saved = expectedResults_;
    expectedResults_ = numTargets;
    RuntimeValue val = evalExpr(*stmt.value);
    expectedResults_ = saved;

    if (numTargets == 1) {
        if (stmt.targets[0]->kind != Expr::Ident)
            error("assignment target must be an identifier");
        setVar(stmt.targets[0]->sval, val);
    } else {
        if (val.kind != RuntimeValue::Tuple ||
            (int)val.tupleVals.size() != numTargets)
            error("expected " + std::to_string(numTargets) + " values");
        for (int i = 0; i < numTargets; i++) {
            if (stmt.targets[i]->kind != Expr::Ident)
                error("assignment target must be an identifier");
            setVar(stmt.targets[i]->sval, val.tupleVals[i]);
        }
    }
}

void Interpreter::execVarDecl(const Stmt& stmt) {
    setVar(stmt.name, RuntimeValue::makeVoid());
}

void Interpreter::execReturn(const Stmt& stmt) {
    if (stmt.values.empty()) {
        returnValue_ = RuntimeValue::makeVoid();
    } else if (stmt.values.size() == 1) {
        returnValue_ = evalExpr(*stmt.values[0]);
    } else {
        std::vector<RuntimeValue> vals;
        for (auto& v : stmt.values)
            vals.push_back(evalExpr(*v));
        returnValue_ = RuntimeValue::makeTuple(std::move(vals));
    }
    hasReturn_ = true;
}

void Interpreter::execFor(const Stmt& stmt) {
    if (!stmt.iterExpr || stmt.iterExpr->kind != Expr::Call ||
        !stmt.iterExpr->left || stmt.iterExpr->left->kind != Expr::Ident ||
        stmt.iterExpr->left->sval != "range")
        error("for loop requires range()");

    auto& callExpr = *stmt.iterExpr;
    int64_t start = 0, end = 0;

    if (callExpr.args.size() == 1) {
        auto v = evalExpr(*callExpr.args[0]);
        if (v.kind != RuntimeValue::Int) error("range() arg must be int");
        end = v.intVal;
    } else if (callExpr.args.size() == 2) {
        auto v1 = evalExpr(*callExpr.args[0]);
        auto v2 = evalExpr(*callExpr.args[1]);
        if (v1.kind != RuntimeValue::Int || v2.kind != RuntimeValue::Int)
            error("range() args must be int");
        start = v1.intVal;
        end = v2.intVal;
    } else {
        error("range() expects 1 or 2 arguments");
    }

    for (int64_t i = start; i < end; i++) {
        setVar(stmt.name, RuntimeValue::makeInt(i));
        execBlock(stmt.stmts);
        if (hasReturn_) return;
    }
}

void Interpreter::execIf(const Stmt& stmt) {
    auto cond = evalExpr(*stmt.cond);
    if (cond.kind != RuntimeValue::Int)
        error("if condition must be int");

    if (cond.intVal != 0) {
        execBlock(stmt.stmts);
    } else {
        execBlock(stmt.elseStmts);
    }
}

void Interpreter::execWeightScope(const Stmt& stmt) {
    auto scope = evalExpr(*stmt.scopeExpr);
    if (scope.kind != RuntimeValue::String)
        error("weight_scope must be a string");

    weightScope_.push_back(interpolateString(scope.strVal));
    execBlock(stmt.stmts);
    weightScope_.pop_back();
}

RuntimeValue Interpreter::evalExpr(const Expr& expr) {
    switch (expr.kind) {
        case Expr::Ident:     return getVar(expr.sval);
        case Expr::IntLit:    return RuntimeValue::makeInt(expr.ival);
        case Expr::FloatLit:  return RuntimeValue::makeFloat(expr.fval);
        case Expr::StringLit: return RuntimeValue::makeString(expr.sval);
        case Expr::WeightLit: {
            auto v = graph_.addWeight(resolveWeight(expr.sval));
            return RuntimeValue::makeNode(v);
        }
        case Expr::Binary:  return evalBinary(expr);
        case Expr::Unary:   return evalUnary(expr);
        case Expr::Call:    return evalCall(expr);
        case Expr::Index:   error("indexing not supported in interpreter");
    }
    error("unknown expression kind");
}

RuntimeValue Interpreter::evalCall(const Expr& expr) {
    if (!expr.left || expr.left->kind != Expr::Ident)
        error("callee must be an identifier");
    std::string name = expr.left->sval;

    if (name.size() > 2 && name[0] == '_' && name[1] == '_') {
        std::string builtinName = name.substr(2);
        int numResults = expectedResults_;

        int saved = expectedResults_;
        expectedResults_ = 1;

        std::vector<high_ir::Value> operands;
        for (auto& arg : expr.args) {
            RuntimeValue val = evalExpr(*arg);
            operands.push_back(toGraphValue(val));
        }

        std::vector<high_ir::Attr> attrs;
        for (auto& na : expr.namedArgs) {
            RuntimeValue val = evalExpr(*na.value);
            switch (val.kind) {
                case RuntimeValue::Int:
                    attrs.push_back(high_ir::Attr::fromInt(na.name, val.intVal));
                    break;
                case RuntimeValue::Float:
                    attrs.push_back(high_ir::Attr::fromFloat(na.name, val.floatVal));
                    break;
                case RuntimeValue::String:
                    attrs.push_back(high_ir::Attr::fromString(na.name, val.strVal));
                    break;
                default:
                    error("named arg '" + na.name + "' must be compile-time");
            }
        }

        expectedResults_ = saved;

        auto results = graph_.addBuiltin(builtinName, operands, attrs, numResults);
        if (numResults == 1) {
            return RuntimeValue::makeNode(results[0]);
        }
        std::vector<RuntimeValue> tuple;
        for (auto& r : results)
            tuple.push_back(RuntimeValue::makeNode(r));
        return RuntimeValue::makeTuple(std::move(tuple));
    }

    auto it = funcTable_.find(name);
    if (it == funcTable_.end())
        error("undefined function '" + name + "'");

    int saved = expectedResults_;
    expectedResults_ = 1;
    std::vector<RuntimeValue> args;
    for (auto& arg : expr.args)
        args.push_back(evalExpr(*arg));
    expectedResults_ = saved;

    return callFunc(*it->second, args);
}

RuntimeValue Interpreter::evalBinary(const Expr& expr) {
    auto left = evalExpr(*expr.left);
    auto right = evalExpr(*expr.right);

    if (left.kind == RuntimeValue::Int && right.kind == RuntimeValue::Int) {
        int64_t l = left.intVal, r = right.intVal;
        if (expr.op == "+")  return RuntimeValue::makeInt(l + r);
        if (expr.op == "-")  return RuntimeValue::makeInt(l - r);
        if (expr.op == "*")  return RuntimeValue::makeInt(l * r);
        if (expr.op == "/")  return RuntimeValue::makeInt(l / r);
        if (expr.op == "%")  return RuntimeValue::makeInt(l % r);
        if (expr.op == "==") return RuntimeValue::makeInt(l == r ? 1 : 0);
        if (expr.op == "!=") return RuntimeValue::makeInt(l != r ? 1 : 0);
        if (expr.op == "<")  return RuntimeValue::makeInt(l < r ? 1 : 0);
        if (expr.op == ">")  return RuntimeValue::makeInt(l > r ? 1 : 0);
        if (expr.op == "<=") return RuntimeValue::makeInt(l <= r ? 1 : 0);
        if (expr.op == ">=") return RuntimeValue::makeInt(l >= r ? 1 : 0);
    }

    if (left.kind == RuntimeValue::Float && right.kind == RuntimeValue::Float) {
        double l = left.floatVal, r = right.floatVal;
        if (expr.op == "+") return RuntimeValue::makeFloat(l + r);
        if (expr.op == "-") return RuntimeValue::makeFloat(l - r);
        if (expr.op == "*") return RuntimeValue::makeFloat(l * r);
        if (expr.op == "/") return RuntimeValue::makeFloat(l / r);
    }

    if (left.kind == RuntimeValue::Int && right.kind == RuntimeValue::Float) {
        double l = (double)left.intVal, r = right.floatVal;
        if (expr.op == "+") return RuntimeValue::makeFloat(l + r);
        if (expr.op == "-") return RuntimeValue::makeFloat(l - r);
        if (expr.op == "*") return RuntimeValue::makeFloat(l * r);
        if (expr.op == "/") return RuntimeValue::makeFloat(l / r);
    }

    if (left.kind == RuntimeValue::Float && right.kind == RuntimeValue::Int) {
        double l = left.floatVal, r = (double)right.intVal;
        if (expr.op == "+") return RuntimeValue::makeFloat(l + r);
        if (expr.op == "-") return RuntimeValue::makeFloat(l - r);
        if (expr.op == "*") return RuntimeValue::makeFloat(l * r);
        if (expr.op == "/") return RuntimeValue::makeFloat(l / r);
    }

    error("unsupported binary operation '" + expr.op + "'");
}

RuntimeValue Interpreter::evalUnary(const Expr& expr) {
    auto operand = evalExpr(*expr.left);
    if (expr.op == "-") {
        if (operand.kind == RuntimeValue::Int)
            return RuntimeValue::makeInt(-operand.intVal);
        if (operand.kind == RuntimeValue::Float)
            return RuntimeValue::makeFloat(-operand.floatVal);
    }
    error("unsupported unary '" + expr.op + "'");
}

void Interpreter::pushEnv() { envStack_.emplace_back(); }
void Interpreter::popEnv()  { envStack_.pop_back(); }

void Interpreter::setVar(const std::string& name, const RuntimeValue& val) {
    envStack_.back()[name] = val;
}

RuntimeValue Interpreter::getVar(const std::string& name) {
    auto it = envStack_.back().find(name);
    if (it != envStack_.back().end()) return it->second;
    error("undefined variable '" + name + "'");
}

std::string Interpreter::resolveWeight(const std::string& localName) {
    std::string prefix;
    for (auto& s : weightScope_) {
        if (!prefix.empty()) prefix += ".";
        prefix += s;
    }
    if (!prefix.empty()) return prefix + "." + localName;
    return localName;
}

std::string Interpreter::interpolateString(const std::string& s) {
    std::string result;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '{') {
            size_t end = s.find('}', i + 1);
            if (end == std::string::npos) error("unterminated '{' in string");
            std::string var = s.substr(i + 1, end - i - 1);
            auto val = getVar(var);
            if (val.kind == RuntimeValue::Int)
                result += std::to_string(val.intVal);
            else if (val.kind == RuntimeValue::String)
                result += val.strVal;
            else
                error("interpolation var '" + var + "' must be int or string");
            i = end;
        } else {
            result += s[i];
        }
    }
    return result;
}

high_ir::Value Interpreter::toGraphValue(const RuntimeValue& val) {
    switch (val.kind) {
        case RuntimeValue::NodeVal: return val.nodeVal;
        case RuntimeValue::Int:     return graph_.addIntConst(val.intVal);
        case RuntimeValue::Float:   return graph_.addFloatConst(val.floatVal);
        case RuntimeValue::String:  return graph_.addStringConst(val.strVal);
        default: error("cannot convert to graph value");
    }
}

void Interpreter::error(const std::string& msg) {
    fprintf(stderr, "interpreter: %s\n", msg.c_str());
    abort();
}

} // namespace sandygo
