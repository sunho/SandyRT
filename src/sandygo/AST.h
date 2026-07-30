#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sandygo {

struct Expr;
struct Stmt;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

struct TypeExpr {
    enum Kind { Simple, Slice };
    Kind kind = Simple;
    std::string name;

    static TypeExpr simple(const std::string& n) { return {Simple, n}; }
    static TypeExpr slice(const std::string& elem) { return {Slice, elem}; }
};

struct Param {
    std::string name;
    TypeExpr type;
};

struct NamedArg {
    std::string name;
    ExprPtr value;
};

struct Expr {
    enum Kind {
        Ident, IntLit, FloatLit, StringLit, WeightLit,
        Binary, Unary, Call, Index
    };
    Kind kind;
    int line = 0;
    int col = 0;

    std::string sval;
    int64_t ival = 0;
    double fval = 0.0;

    std::string op;
    ExprPtr left;
    ExprPtr right;

    std::vector<ExprPtr> args;
    std::vector<NamedArg> namedArgs;
};

struct Stmt {
    enum Kind {
        Assign, VarDecl, Return, For, If, WeightScope, ExprStmt
    };
    Kind kind;
    int line = 0;

    std::vector<ExprPtr> targets;
    ExprPtr value;
    bool isDecl = false;

    std::string name;
    TypeExpr type;

    std::vector<ExprPtr> values;

    ExprPtr iterExpr;

    ExprPtr cond;
    std::vector<StmtPtr> elseStmts;

    ExprPtr scopeExpr;

    ExprPtr expr;

    std::vector<StmtPtr> stmts;
};

struct FuncDecl {
    std::string name;
    std::vector<Param> params;
    std::vector<TypeExpr> returnTypes;
    std::vector<StmtPtr> body;
    int line = 0;
};

struct Program {
    std::vector<FuncDecl> funcs;
};

inline ExprPtr makeIdent(const std::string& name, int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Ident;
    e->sval = name;
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeIntLit(int64_t val, int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::IntLit;
    e->ival = val;
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeFloatLit(double val, int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::FloatLit;
    e->fval = val;
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeStringLit(const std::string& val, int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::StringLit;
    e->sval = val;
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeWeightLit(const std::string& path, int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::WeightLit;
    e->sval = path;
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeBinary(const std::string& op, ExprPtr left, ExprPtr right,
                          int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Binary;
    e->op = op;
    e->left = std::move(left);
    e->right = std::move(right);
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeUnary(const std::string& op, ExprPtr operand,
                         int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Unary;
    e->op = op;
    e->left = std::move(operand);
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeCall(ExprPtr callee, std::vector<ExprPtr> args,
                        std::vector<NamedArg> namedArgs,
                        int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Call;
    e->left = std::move(callee);
    e->args = std::move(args);
    e->namedArgs = std::move(namedArgs);
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr makeIndex(ExprPtr target, ExprPtr index,
                         int line = 0, int col = 0) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Index;
    e->left = std::move(target);
    e->right = std::move(index);
    e->line = line;
    e->col = col;
    return e;
}

} // namespace sandygo
