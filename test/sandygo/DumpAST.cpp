#include "Lexer.h"
#include "Parser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace sandy::sandygo;

static void indent(int depth) {
    for (int i = 0; i < depth; i++) std::cout << "  ";
}

static void dumpType(const TypeExpr& t) {
    if (t.kind == TypeExpr::Slice) {
        std::cout << "[]" << t.name;
    } else {
        std::cout << t.name;
    }
}

static void dumpExpr(const Expr& e, int depth) {
    indent(depth);
    switch (e.kind) {
        case Expr::Ident:
            std::cout << "Ident(" << e.sval << ")\n";
            break;
        case Expr::IntLit:
            std::cout << "Int(" << e.ival << ")\n";
            break;
        case Expr::FloatLit:
            std::cout << "Float(" << e.fval << ")\n";
            break;
        case Expr::StringLit:
            std::cout << "String(\"" << e.sval << "\")\n";
            break;
        case Expr::WeightLit:
            std::cout << "Weight(@" << e.sval << ")\n";
            break;
        case Expr::IntListLit:
            std::cout << "IntList[";
            for (size_t i = 0; i < e.intListVal.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << e.intListVal[i];
            }
            std::cout << "]\n";
            break;
        case Expr::Binary:
            std::cout << "Binary(" << e.op << ")\n";
            dumpExpr(*e.left, depth + 1);
            dumpExpr(*e.right, depth + 1);
            break;
        case Expr::Unary:
            std::cout << "Unary(" << e.op << ")\n";
            dumpExpr(*e.left, depth + 1);
            break;
        case Expr::Call:
            std::cout << "Call\n";
            indent(depth + 1); std::cout << "callee:\n";
            dumpExpr(*e.left, depth + 2);
            if (!e.args.empty()) {
                indent(depth + 1); std::cout << "args:\n";
                for (auto& a : e.args) dumpExpr(*a, depth + 2);
            }
            if (!e.namedArgs.empty()) {
                indent(depth + 1); std::cout << "named:\n";
                for (auto& na : e.namedArgs) {
                    indent(depth + 2);
                    std::cout << na.name << " =\n";
                    dumpExpr(*na.value, depth + 3);
                }
            }
            break;
        case Expr::Index:
            std::cout << "Index\n";
            dumpExpr(*e.left, depth + 1);
            indent(depth + 1); std::cout << "[\n";
            for (const auto& selector : e.indexSelectors) {
                if (selector.kind == IndexSelector::Full) {
                    indent(depth + 2); std::cout << ":\n";
                } else {
                    dumpExpr(*selector.index, depth + 2);
                }
            }
            indent(depth + 1); std::cout << "]\n";
            break;
    }
}

static void dumpStmt(const Stmt& s, int depth);

static void dumpBlock(const std::vector<StmtPtr>& stmts, int depth) {
    for (auto& s : stmts) dumpStmt(*s, depth);
}

static void dumpStmt(const Stmt& s, int depth) {
    indent(depth);
    switch (s.kind) {
        case Stmt::Assign:
            if (s.isDecl) std::cout << "ShortVarDecl\n";
            else std::cout << "Assign\n";
            indent(depth + 1); std::cout << "targets:\n";
            for (auto& t : s.targets) dumpExpr(*t, depth + 2);
            indent(depth + 1); std::cout << "value:\n";
            dumpExpr(*s.value, depth + 2);
            break;
        case Stmt::VarDecl:
            std::cout << "VarDecl " << s.name << " ";
            dumpType(s.type);
            std::cout << "\n";
            break;
        case Stmt::Return:
            std::cout << "Return\n";
            for (auto& v : s.values) dumpExpr(*v, depth + 1);
            break;
        case Stmt::For:
            std::cout << "For " << s.name << " :=\n";
            indent(depth + 1); std::cout << "iter:\n";
            dumpExpr(*s.iterExpr, depth + 2);
            indent(depth + 1); std::cout << "body:\n";
            dumpBlock(s.stmts, depth + 2);
            break;
        case Stmt::If:
            std::cout << "If\n";
            indent(depth + 1); std::cout << "cond:\n";
            dumpExpr(*s.cond, depth + 2);
            indent(depth + 1); std::cout << "then:\n";
            dumpBlock(s.stmts, depth + 2);
            if (!s.elseStmts.empty()) {
                indent(depth + 1); std::cout << "else:\n";
                dumpBlock(s.elseStmts, depth + 2);
            }
            break;
        case Stmt::WeightScope:
            std::cout << "WeightScope\n";
            indent(depth + 1); std::cout << "scope:\n";
            dumpExpr(*s.scopeExpr, depth + 2);
            indent(depth + 1); std::cout << "body:\n";
            dumpBlock(s.stmts, depth + 2);
            break;
        case Stmt::ExprStmt:
            std::cout << "ExprStmt\n";
            dumpExpr(*s.expr, depth + 1);
            break;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: dump_ast <file.sandy.go>\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "cannot open: " << argv[1] << "\n";
        return 1;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    if (lexer.hasError()) {
        std::cerr << "lexer error: " << lexer.errorMessage() << "\n";
        return 1;
    }

    Parser parser(std::move(tokens));
    auto prog = parser.parse();
    if (parser.hasError()) {
        std::cerr << "parser error: " << parser.errorMessage() << "\n";
        return 1;
    }

    for (auto& import : prog.imports) {
        std::cout << "Import \"" << import.path << "\"\n";
    }

    for (auto& fn : prog.funcs) {
        std::cout << "Func " << fn.name << "(";
        for (size_t i = 0; i < fn.params.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << fn.params[i].name << " ";
            dumpType(fn.params[i].type);
        }
        std::cout << ")";
        if (!fn.returnTypes.empty()) {
            std::cout << " -> (";
            for (size_t i = 0; i < fn.returnTypes.size(); i++) {
                if (i > 0) std::cout << ", ";
                dumpType(fn.returnTypes[i]);
            }
            std::cout << ")";
        }
        std::cout << "\n";
        dumpBlock(fn.body, 1);
        std::cout << "\n";
    }

    return 0;
}
