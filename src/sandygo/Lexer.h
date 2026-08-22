#pragma once

#include <string>
#include <vector>

namespace sandy::sandygo {

enum class TokenKind {
    Func, Return, For, If, Else, Var, Import, WeightScope, Config, Const,

    Ident, IntLit, FloatLit, StringLit, WeightLit,

    ColonAssign, Assign,
    Plus, Minus, Star, Slash, Percent,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,

    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,
    Comma,

    Semicolon,
    Eof,
    Error,
};

const char* tokenKindName(TokenKind kind);

struct Token {
    TokenKind kind;
    std::string value;
    int line;
    int col;
};

class Lexer {
public:
    explicit Lexer(std::string source);

    std::vector<Token> tokenize();

    bool hasError() const { return hasError_; }
    const std::string& errorMessage() const { return error_; }

private:
    Token scanToken();
    Token scanIdent();
    Token scanNumber();
    Token scanString();
    Token scanWeightLit();
    void skipLineComment();

    char peek() const;
    char peekNext() const;
    char advance();
    bool match(char expected);
    bool isAtEnd() const;

    Token makeToken(TokenKind kind, const std::string& value = "");
    Token errorToken(const std::string& msg);

    static bool shouldInsertSemicolon(TokenKind kind);

    std::string source_;
    size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;
    int tokenLine_ = 1;
    int tokenCol_ = 1;

    std::string error_;
    bool hasError_ = false;
};

} // namespace sandy::sandygo
