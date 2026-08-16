#include "Parser.h"
#include <cstdlib>
#include <sstream>

namespace sandy::sandygo {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

const Token& Parser::peek() const {
    return tokens_[pos_];
}

const Token& Parser::previous() const {
    return tokens_[pos_ - 1];
}

Token Parser::advance() {
    Token tok = tokens_[pos_];
    if (tok.kind != TokenKind::Eof) pos_++;
    return tok;
}

bool Parser::check(TokenKind kind) const {
    return peek().kind == kind;
}

bool Parser::match(TokenKind kind) {
    if (check(kind)) {
        advance();
        return true;
    }
    return false;
}

Token Parser::expect(TokenKind kind, const std::string& msg) {
    if (check(kind)) return advance();
    reportError(msg);
    return {TokenKind::Error, "", peek().line, peek().col};
}

void Parser::reportError(const std::string& msg) {
    if (!hasError_) {
        std::ostringstream oss;
        oss << "line " << peek().line << ":" << peek().col << ": " << msg
            << " (got '" << tokenKindName(peek().kind) << "'";
        if (!peek().value.empty()) oss << " \"" << peek().value << "\"";
        oss << ")";
        error_ = oss.str();
        hasError_ = true;
    }
}

Program Parser::parse() {
    Program prog;
    while (!check(TokenKind::Eof) && !hasError_) {
        while (match(TokenKind::Semicolon)) {}
        if (check(TokenKind::Eof)) break;
        if (check(TokenKind::Import)) {
            prog.imports.push_back(parseImportDecl());
        } else {
            prog.funcs.push_back(parseFuncDecl());
        }
    }
    return prog;
}

ImportDecl Parser::parseImportDecl() {
    ImportDecl decl;
    decl.line = peek().line;

    expect(TokenKind::Import, "expected 'import'");
    if (hasError_) return decl;

    Token path = expect(TokenKind::StringLit, "expected import path string");
    if (hasError_) return decl;
    decl.path = path.value;
    return decl;
}

FuncDecl Parser::parseFuncDecl() {
    FuncDecl decl;
    decl.line = peek().line;

    expect(TokenKind::Func, "expected 'func'");
    if (hasError_) return decl;

    Token name = expect(TokenKind::Ident, "expected function name");
    if (hasError_) return decl;
    decl.name = name.value;

    expect(TokenKind::LParen, "expected '('");
    if (hasError_) return decl;
    decl.params = parseParams();
    if (hasError_) return decl;
    expect(TokenKind::RParen, "expected ')'");
    if (hasError_) return decl;

    decl.returnTypes = parseReturnTypes();
    if (hasError_) return decl;

    decl.body = parseBlock();
    return decl;
}

std::vector<Param> Parser::parseParams() {
    std::vector<Param> params;
    if (check(TokenKind::RParen)) return params;

    while (true) {
        Param p;
        Token name = expect(TokenKind::Ident, "expected parameter name");
        if (hasError_) return params;
        p.name = name.value;
        p.type = parseType();
        if (hasError_) return params;
        params.push_back(std::move(p));
        if (!match(TokenKind::Comma)) break;
    }
    return params;
}

TypeExpr Parser::parseType() {
    if (check(TokenKind::LBracket)) {
        advance();
        expect(TokenKind::RBracket, "expected ']' in slice type");
        if (hasError_) return TypeExpr::simple("");
        Token elem = expect(TokenKind::Ident, "expected element type");
        if (hasError_) return TypeExpr::simple("");
        return TypeExpr::slice(elem.value);
    }

    Token name = expect(TokenKind::Ident, "expected type name");
    if (hasError_) return TypeExpr::simple("");
    auto type = TypeExpr::simple(name.value);
    if (match(TokenKind::LBracket)) {
        expect(TokenKind::LBracket, "expected '[' in type argument");
        if (hasError_) return type;
        type.dims = parseTypeDimList();
        if (hasError_) return type;
        expect(TokenKind::RBracket, "expected ']' after type argument");
        if (hasError_) return type;
        if (name.value == "PagedTensor") {
            expect(TokenKind::Comma, "expected ', page_size=...' in PagedTensor type");
            if (hasError_) return type;
            Token pageSizeName = expect(TokenKind::Ident, "expected page_size in PagedTensor type");
            if (hasError_) return type;
            if (pageSizeName.value != "page_size") {
                reportError("expected page_size in PagedTensor type");
                return type;
            }
            expect(TokenKind::Assign, "expected '=' after page_size in PagedTensor type");
            if (hasError_) return type;
            Token pageSize = expect(TokenKind::IntLit, "expected integer page size in PagedTensor type");
            if (hasError_) return type;
            type.pageSize = std::strtoll(pageSize.value.c_str(), nullptr, 10);
        } else if (match(TokenKind::Comma)) {
            reportError("unexpected type argument");
            return type;
        }
        expect(TokenKind::RBracket, "expected ']' after type arguments");
    }
    return type;
}

std::vector<int64_t> Parser::parseTypeDimList() {
    std::vector<int64_t> dims;
    if (check(TokenKind::RBracket)) return dims;

    while (true) {
        int64_t sign = 1;
        if (match(TokenKind::Minus))
            sign = -1;
        Token value = expect(TokenKind::IntLit, "expected integer in type dimension list");
        if (hasError_) return dims;
        dims.push_back(sign * std::strtoll(value.value.c_str(), nullptr, 10));
        if (!match(TokenKind::Comma)) break;
        if (check(TokenKind::RBracket)) break;
    }
    return dims;
}

std::vector<TypeExpr> Parser::parseReturnTypes() {
    std::vector<TypeExpr> types;

    if (check(TokenKind::LBrace)) return types;

    if (check(TokenKind::LParen)) {
        advance();
        while (!check(TokenKind::RParen) && !hasError_) {
            types.push_back(parseType());
            if (!match(TokenKind::Comma)) break;
        }
        expect(TokenKind::RParen, "expected ')' in return type");
        return types;
    }

    types.push_back(parseType());
    return types;
}

std::vector<StmtPtr> Parser::parseBlock() {
    expect(TokenKind::LBrace, "expected '{'");
    if (hasError_) return {};

    while (match(TokenKind::Semicolon)) {}

    std::vector<StmtPtr> stmts;
    while (!check(TokenKind::RBrace) && !check(TokenKind::Eof) && !hasError_) {
        stmts.push_back(parseStmt());
        while (match(TokenKind::Semicolon)) {}
    }

    expect(TokenKind::RBrace, "expected '}'");
    return stmts;
}

StmtPtr Parser::parseStmt() {
    if (check(TokenKind::Var)) return parseVarDecl();
    if (check(TokenKind::Return)) return parseReturn();
    if (check(TokenKind::For)) return parseFor();
    if (check(TokenKind::If)) return parseIf();
    if (check(TokenKind::WeightScope)) return parseWeightScope();
    return parseAssignOrExpr();
}

StmtPtr Parser::parseVarDecl() {
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::VarDecl;
    stmt->line = peek().line;
    advance();

    Token name = expect(TokenKind::Ident, "expected variable name after 'var'");
    if (hasError_) return stmt;
    stmt->name = name.value;
    stmt->type = parseType();
    return stmt;
}

StmtPtr Parser::parseReturn() {
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::Return;
    stmt->line = peek().line;
    advance();

    if (!check(TokenKind::Semicolon) && !check(TokenKind::RBrace)) {
        stmt->values.push_back(parseExpr());
        while (match(TokenKind::Comma)) {
            stmt->values.push_back(parseExpr());
        }
    }
    return stmt;
}

StmtPtr Parser::parseFor() {
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::For;
    stmt->line = peek().line;
    advance();

    Token var = expect(TokenKind::Ident, "expected variable in for loop");
    if (hasError_) return stmt;
    stmt->name = var.value;

    expect(TokenKind::ColonAssign, "expected ':=' in for loop");
    if (hasError_) return stmt;

    stmt->iterExpr = parseExpr();
    if (hasError_) return stmt;

    stmt->stmts = parseBlock();
    return stmt;
}

StmtPtr Parser::parseIf() {
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::If;
    stmt->line = peek().line;
    advance();

    stmt->cond = parseExpr();
    if (hasError_) return stmt;

    stmt->stmts = parseBlock();
    if (hasError_) return stmt;

    if (match(TokenKind::Else)) {
        if (check(TokenKind::If)) {
            auto elseIf = parseIf();
            stmt->elseStmts.push_back(std::move(elseIf));
        } else {
            stmt->elseStmts = parseBlock();
        }
    }
    return stmt;
}

StmtPtr Parser::parseWeightScope() {
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::WeightScope;
    stmt->line = peek().line;
    advance();

    stmt->scopeExpr = parseExpr();
    if (hasError_) return stmt;

    stmt->stmts = parseBlock();
    return stmt;
}

StmtPtr Parser::parseAssignOrExpr() {
    auto stmt = std::make_unique<Stmt>();
    stmt->line = peek().line;

    ExprPtr first = parseExpr();
    if (hasError_) return stmt;

    if (check(TokenKind::Comma)) {
        std::vector<ExprPtr> targets;
        targets.push_back(std::move(first));
        while (match(TokenKind::Comma)) {
            targets.push_back(parseExpr());
            if (hasError_) return stmt;
        }

        if (match(TokenKind::ColonAssign)) {
            stmt->kind = Stmt::Assign;
            stmt->isDecl = true;
            stmt->targets = std::move(targets);
            stmt->value = parseExpr();
        } else if (match(TokenKind::Assign)) {
            stmt->kind = Stmt::Assign;
            stmt->isDecl = false;
            stmt->targets = std::move(targets);
            stmt->value = parseExpr();
        } else {
            reportError("expected ':=' or '=' after expression list");
        }
        return stmt;
    }

    if (match(TokenKind::ColonAssign)) {
        stmt->kind = Stmt::Assign;
        stmt->isDecl = true;
        stmt->targets.push_back(std::move(first));
        stmt->value = parseExpr();
        return stmt;
    }

    if (match(TokenKind::Assign)) {
        stmt->kind = Stmt::Assign;
        stmt->isDecl = false;
        stmt->targets.push_back(std::move(first));
        stmt->value = parseExpr();
        return stmt;
    }

    stmt->kind = Stmt::ExprStmt;
    stmt->expr = std::move(first);
    return stmt;
}

ExprPtr Parser::parseExpr() {
    return parseComparison();
}

ExprPtr Parser::parseComparison() {
    ExprPtr left = parseAddition();
    if (hasError_) return left;

    while (check(TokenKind::Eq) || check(TokenKind::NotEq) ||
           check(TokenKind::Lt) || check(TokenKind::Gt) ||
           check(TokenKind::LtEq) || check(TokenKind::GtEq)) {
        Token op = advance();
        ExprPtr right = parseAddition();
        if (hasError_) return left;
        left = makeBinary(tokenKindName(op.kind), std::move(left), std::move(right),
                          op.line, op.col);
    }
    return left;
}

ExprPtr Parser::parseAddition() {
    ExprPtr left = parseMultiplication();
    if (hasError_) return left;

    while (check(TokenKind::Plus) || check(TokenKind::Minus)) {
        Token op = advance();
        ExprPtr right = parseMultiplication();
        if (hasError_) return left;
        left = makeBinary(tokenKindName(op.kind), std::move(left), std::move(right),
                          op.line, op.col);
    }
    return left;
}

ExprPtr Parser::parseMultiplication() {
    ExprPtr left = parseUnary();
    if (hasError_) return left;

    while (check(TokenKind::Star) || check(TokenKind::Slash) ||
           check(TokenKind::Percent)) {
        Token op = advance();
        ExprPtr right = parseUnary();
        if (hasError_) return left;
        left = makeBinary(tokenKindName(op.kind), std::move(left), std::move(right),
                          op.line, op.col);
    }
    return left;
}

ExprPtr Parser::parseUnary() {
    if (check(TokenKind::Minus)) {
        Token op = advance();
        ExprPtr operand = parseUnary();
        return makeUnary("-", std::move(operand), op.line, op.col);
    }
    return parsePostfix();
}

ExprPtr Parser::parsePostfix() {
    ExprPtr expr = parsePrimary();
    if (hasError_) return expr;

    while (!hasError_) {
        if (match(TokenKind::LParen)) {
            int callLine = previous().line;
            int callCol = previous().col;
            std::vector<NamedArg> namedArgs;
            std::vector<ExprPtr> args = parseCallArgs(namedArgs);
            if (hasError_) return expr;
            expect(TokenKind::RParen, "expected ')'");
            if (hasError_) return expr;
            expr = makeCall(std::move(expr), std::move(args), std::move(namedArgs),
                            callLine, callCol);
        } else if (match(TokenKind::LBracket)) {
            int idxLine = previous().line;
            int idxCol = previous().col;
            ExprPtr index = parseExpr();
            if (hasError_) return expr;
            expect(TokenKind::RBracket, "expected ']'");
            if (hasError_) return expr;
            expr = makeIndex(std::move(expr), std::move(index), idxLine, idxCol);
        } else {
            break;
        }
    }
    return expr;
}

ExprPtr Parser::parsePrimary() {
    if (check(TokenKind::IntLit)) {
        Token tok = advance();
        return makeIntLit(std::strtoll(tok.value.c_str(), nullptr, 10),
                          tok.line, tok.col);
    }
    if (check(TokenKind::FloatLit)) {
        Token tok = advance();
        return makeFloatLit(std::strtod(tok.value.c_str(), nullptr),
                            tok.line, tok.col);
    }
    if (check(TokenKind::StringLit)) {
        Token tok = advance();
        return makeStringLit(tok.value, tok.line, tok.col);
    }
    if (check(TokenKind::WeightLit)) {
        Token tok = advance();
        return makeWeightLit(tok.value, tok.line, tok.col);
    }
    if (check(TokenKind::Ident)) {
        Token tok = advance();
        return makeIdent(tok.value, tok.line, tok.col);
    }
    if (check(TokenKind::LBracket)) {
        Token start = advance();
        std::vector<int64_t> values;
        if (!check(TokenKind::RBracket)) {
            while (true) {
                int64_t sign = 1;
                if (match(TokenKind::Minus))
                    sign = -1;
                Token value = expect(TokenKind::IntLit, "expected integer in int list literal");
                if (hasError_) return std::make_unique<Expr>();
                values.push_back(sign * std::strtoll(value.value.c_str(), nullptr, 10));
                if (!match(TokenKind::Comma)) break;
                if (check(TokenKind::RBracket)) break;
            }
        }
        expect(TokenKind::RBracket, "expected ']'");
        if (hasError_) return std::make_unique<Expr>();
        return makeIntListLit(std::move(values), start.line, start.col);
    }
    if (match(TokenKind::LParen)) {
        ExprPtr expr = parseExpr();
        if (hasError_) return expr;
        expect(TokenKind::RParen, "expected ')'");
        return expr;
    }

    reportError("expected expression");
    return std::make_unique<Expr>();
}

std::vector<ExprPtr> Parser::parseCallArgs(std::vector<NamedArg>& namedArgs) {
    std::vector<ExprPtr> args;
    if (check(TokenKind::RParen)) return args;

    while (true) {
        ExprPtr expr = parseExpr();
        if (hasError_) return args;

        if (expr->kind == Expr::Ident && check(TokenKind::Assign)) {
            advance();
            NamedArg na;
            na.name = expr->sval;
            na.value = parseExpr();
            if (hasError_) return args;
            namedArgs.push_back(std::move(na));
        } else {
            args.push_back(std::move(expr));
        }

        if (!match(TokenKind::Comma)) break;
        if (check(TokenKind::RParen)) break;
    }
    return args;
}

} // namespace sandy::sandygo
