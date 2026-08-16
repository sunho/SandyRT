#pragma once

#include "AST.h"
#include "Lexer.h"
#include <vector>

namespace sandy::sandygo {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    Program parse();

    bool hasError() const { return hasError_; }
    const std::string& errorMessage() const { return error_; }

private:
    ImportDecl parseImportDecl();
    FuncDecl parseFuncDecl();
    std::vector<Param> parseParams();
    TypeExpr parseType();
    std::vector<int64_t> parseTypeDimList();
    std::vector<TypeExpr> parseReturnTypes();

    std::vector<StmtPtr> parseBlock();
    StmtPtr parseStmt();
    StmtPtr parseVarDecl();
    StmtPtr parseReturn();
    StmtPtr parseFor();
    StmtPtr parseIf();
    StmtPtr parseWeightScope();
    StmtPtr parseAssignOrExpr();

    ExprPtr parseExpr();
    ExprPtr parseComparison();
    ExprPtr parseAddition();
    ExprPtr parseMultiplication();
    ExprPtr parseUnary();
    ExprPtr parsePostfix();
    ExprPtr parsePrimary();

    std::vector<ExprPtr> parseCallArgs(std::vector<NamedArg>& namedArgs);

    const Token& peek() const;
    const Token& previous() const;
    Token advance();
    bool check(TokenKind kind) const;
    bool match(TokenKind kind);
    Token expect(TokenKind kind, const std::string& msg);

    void reportError(const std::string& msg);

    std::vector<Token> tokens_;
    size_t pos_ = 0;

    std::string error_;
    bool hasError_ = false;
};

} // namespace sandy::sandygo
