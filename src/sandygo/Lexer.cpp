#include "Lexer.h"
#include <cctype>

namespace sandy::sandygo {

const char* tokenKindName(TokenKind kind) {
    switch (kind) {
        case TokenKind::Func: return "func";
        case TokenKind::Return: return "return";
        case TokenKind::For: return "for";
        case TokenKind::If: return "if";
        case TokenKind::Else: return "else";
        case TokenKind::Var: return "var";
        case TokenKind::Import: return "import";
        case TokenKind::WeightScope: return "weight_scope";
        case TokenKind::Config: return "config";
        case TokenKind::Const: return "const";
        case TokenKind::Ident: return "IDENT";
        case TokenKind::IntLit: return "INT";
        case TokenKind::FloatLit: return "FLOAT";
        case TokenKind::StringLit: return "STRING";
        case TokenKind::WeightLit: return "WEIGHT";
        case TokenKind::ColonAssign: return ":=";
        case TokenKind::Assign: return "=";
        case TokenKind::Plus: return "+";
        case TokenKind::Minus: return "-";
        case TokenKind::Star: return "*";
        case TokenKind::Slash: return "/";
        case TokenKind::Percent: return "%";
        case TokenKind::Eq: return "==";
        case TokenKind::NotEq: return "!=";
        case TokenKind::Lt: return "<";
        case TokenKind::Gt: return ">";
        case TokenKind::LtEq: return "<=";
        case TokenKind::GtEq: return ">=";
        case TokenKind::LParen: return "(";
        case TokenKind::RParen: return ")";
        case TokenKind::LBrace: return "{";
        case TokenKind::RBrace: return "}";
        case TokenKind::LBracket: return "[";
        case TokenKind::RBracket: return "]";
        case TokenKind::Comma: return ",";
        case TokenKind::Semicolon: return ";";
        case TokenKind::Eof: return "EOF";
        case TokenKind::Error: return "ERROR";
    }
    return "UNKNOWN";
}

Lexer::Lexer(std::string source) : source_(std::move(source)) {}

char Lexer::peek() const {
    if (isAtEnd()) return '\0';
    return source_[pos_];
}

char Lexer::peekNext() const {
    if (pos_ + 1 >= source_.size()) return '\0';
    return source_[pos_ + 1];
}

char Lexer::advance() {
    char c = source_[pos_++];
    if (c == '\n') {
        line_++;
        col_ = 1;
    } else {
        col_++;
    }
    return c;
}

bool Lexer::match(char expected) {
    if (isAtEnd() || source_[pos_] != expected) return false;
    advance();
    return true;
}

bool Lexer::isAtEnd() const {
    return pos_ >= source_.size();
}

Token Lexer::makeToken(TokenKind kind, const std::string& value) {
    return {kind, value, tokenLine_, tokenCol_};
}

Token Lexer::errorToken(const std::string& msg) {
    hasError_ = true;
    if (error_.empty()) error_ = msg;
    return {TokenKind::Error, msg, tokenLine_, tokenCol_};
}

bool Lexer::shouldInsertSemicolon(TokenKind kind) {
    switch (kind) {
        case TokenKind::Ident:
        case TokenKind::IntLit:
        case TokenKind::FloatLit:
        case TokenKind::StringLit:
        case TokenKind::WeightLit:
        case TokenKind::Return:
        case TokenKind::RParen:
        case TokenKind::RBracket:
        case TokenKind::RBrace:
            return true;
        default:
            return false;
    }
}

void Lexer::skipLineComment() {
    while (!isAtEnd() && peek() != '\n') {
        advance();
    }
}

Token Lexer::scanIdent() {
    size_t start = pos_ - 1;
    while (!isAtEnd() && (std::isalnum(peek()) || peek() == '_')) {
        advance();
    }
    std::string word = source_.substr(start, pos_ - start);

    if (word == "func") return makeToken(TokenKind::Func, word);
    if (word == "return") return makeToken(TokenKind::Return, word);
    if (word == "for") return makeToken(TokenKind::For, word);
    if (word == "if") return makeToken(TokenKind::If, word);
    if (word == "else") return makeToken(TokenKind::Else, word);
    if (word == "var") return makeToken(TokenKind::Var, word);
    if (word == "import") return makeToken(TokenKind::Import, word);
    if (word == "weight_scope") return makeToken(TokenKind::WeightScope, word);
    if (word == "config") return makeToken(TokenKind::Config, word);
    if (word == "const") return makeToken(TokenKind::Const, word);

    return makeToken(TokenKind::Ident, word);
}

Token Lexer::scanNumber() {
    size_t start = pos_ - 1;
    while (!isAtEnd() && std::isdigit(peek())) {
        advance();
    }

    if (peek() == '.' && std::isdigit(peekNext())) {
        advance();
        while (!isAtEnd() && std::isdigit(peek())) {
            advance();
        }
        return makeToken(TokenKind::FloatLit, source_.substr(start, pos_ - start));
    }

    return makeToken(TokenKind::IntLit, source_.substr(start, pos_ - start));
}

Token Lexer::scanString() {
    std::string value;
    while (!isAtEnd() && peek() != '"') {
        if (peek() == '\n') {
            return errorToken("unterminated string literal");
        }
        if (peek() == '\\') {
            advance();
            if (isAtEnd()) return errorToken("unterminated string literal");
            switch (peek()) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case '\\': value += '\\'; break;
                case '"': value += '"'; break;
                default:
                    return errorToken(std::string("invalid escape sequence: \\") + peek());
            }
            advance();
        } else {
            value += advance();
        }
    }
    if (isAtEnd()) {
        return errorToken("unterminated string literal");
    }
    advance();
    return makeToken(TokenKind::StringLit, value);
}

Token Lexer::scanWeightLit() {
    size_t start = pos_;
    while (!isAtEnd() && (std::isalnum(peek()) || peek() == '_' || peek() == '.')) {
        advance();
    }
    if (pos_ == start) {
        return errorToken("expected weight name after '@'");
    }
    return makeToken(TokenKind::WeightLit, source_.substr(start, pos_ - start));
}

Token Lexer::scanToken() {
    tokenLine_ = line_;
    tokenCol_ = col_;

    char c = advance();

    if (std::isalpha(c) || c == '_') return scanIdent();
    if (std::isdigit(c)) return scanNumber();

    switch (c) {
        case '"': return scanString();
        case '@': return scanWeightLit();
        case '(': return makeToken(TokenKind::LParen);
        case ')': return makeToken(TokenKind::RParen);
        case '{': return makeToken(TokenKind::LBrace);
        case '}': return makeToken(TokenKind::RBrace);
        case '[': return makeToken(TokenKind::LBracket);
        case ']': return makeToken(TokenKind::RBracket);
        case ',': return makeToken(TokenKind::Comma);
        case '+': return makeToken(TokenKind::Plus);
        case '-': return makeToken(TokenKind::Minus);
        case '*': return makeToken(TokenKind::Star);
        case '/': return makeToken(TokenKind::Slash);
        case '%': return makeToken(TokenKind::Percent);
        case ':':
            if (match('=')) return makeToken(TokenKind::ColonAssign);
            return errorToken("unexpected ':'");
        case '=':
            if (match('=')) return makeToken(TokenKind::Eq);
            return makeToken(TokenKind::Assign);
        case '!':
            if (match('=')) return makeToken(TokenKind::NotEq);
            return errorToken("unexpected '!'");
        case '<':
            if (match('=')) return makeToken(TokenKind::LtEq);
            return makeToken(TokenKind::Lt);
        case '>':
            if (match('=')) return makeToken(TokenKind::GtEq);
            return makeToken(TokenKind::Gt);
        default:
            return errorToken(std::string("unexpected character: ") + c);
    }
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    TokenKind lastKind = TokenKind::Eof;

    while (!isAtEnd() && !hasError_) {
        while (!isAtEnd() && (peek() == ' ' || peek() == '\t' || peek() == '\r')) {
            advance();
        }

        if (isAtEnd()) break;

        if (peek() == '\n') {
            advance();
            if (shouldInsertSemicolon(lastKind)) {
                tokens.push_back({TokenKind::Semicolon, ";", line_ - 1, 0});
                lastKind = TokenKind::Semicolon;
            }
            continue;
        }

        if (peek() == '/' && peekNext() == '/') {
            skipLineComment();
            continue;
        }

        Token tok = scanToken();
        if (tok.kind == TokenKind::Error) break;

        tokens.push_back(tok);
        lastKind = tok.kind;
    }

    if (shouldInsertSemicolon(lastKind)) {
        tokens.push_back({TokenKind::Semicolon, ";", line_, col_});
    }

    tokens.push_back({TokenKind::Eof, "", line_, col_});
    return tokens;
}

} // namespace sandy::sandygo
