#include "Lexer.h"
#include <gtest/gtest.h>

using namespace sandy::sandygo;

static std::vector<Token> lex(const std::string& src) {
    Lexer lexer(src);
    return lexer.tokenize();
}

TEST(Lexer, Keywords) {
    auto tokens = lex("func return for if else var import weight_scope");
    ASSERT_GE(tokens.size(), 8u);
    EXPECT_EQ(tokens[0].kind, TokenKind::Func);
    EXPECT_EQ(tokens[1].kind, TokenKind::Return);
    EXPECT_EQ(tokens[2].kind, TokenKind::For);
    EXPECT_EQ(tokens[3].kind, TokenKind::If);
    EXPECT_EQ(tokens[4].kind, TokenKind::Else);
    EXPECT_EQ(tokens[5].kind, TokenKind::Var);
    EXPECT_EQ(tokens[6].kind, TokenKind::Import);
    EXPECT_EQ(tokens[7].kind, TokenKind::WeightScope);
}

TEST(Lexer, Identifiers) {
    auto tokens = lex("x __rms_norm Tensor gemma_layer");
    EXPECT_EQ(tokens[0].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[0].value, "x");
    EXPECT_EQ(tokens[1].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[1].value, "__rms_norm");
    EXPECT_EQ(tokens[2].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[2].value, "Tensor");
    EXPECT_EQ(tokens[3].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[3].value, "gemma_layer");
}

TEST(Lexer, IntLiterals) {
    auto tokens = lex("0 42 1536 1000000");
    EXPECT_EQ(tokens[0].kind, TokenKind::IntLit);
    EXPECT_EQ(tokens[0].value, "0");
    EXPECT_EQ(tokens[1].kind, TokenKind::IntLit);
    EXPECT_EQ(tokens[1].value, "42");
    EXPECT_EQ(tokens[2].kind, TokenKind::IntLit);
    EXPECT_EQ(tokens[2].value, "1536");
    EXPECT_EQ(tokens[3].kind, TokenKind::IntLit);
    EXPECT_EQ(tokens[3].value, "1000000");
}

TEST(Lexer, FloatLiterals) {
    auto tokens = lex("3.14 1000000.0 0.5");
    EXPECT_EQ(tokens[0].kind, TokenKind::FloatLit);
    EXPECT_EQ(tokens[0].value, "3.14");
    EXPECT_EQ(tokens[1].kind, TokenKind::FloatLit);
    EXPECT_EQ(tokens[1].value, "1000000.0");
    EXPECT_EQ(tokens[2].kind, TokenKind::FloatLit);
    EXPECT_EQ(tokens[2].value, "0.5");
}

TEST(Lexer, StringLiterals) {
    auto tokens = lex(R"("hello" "layers.{i}" "")");
    EXPECT_EQ(tokens[0].kind, TokenKind::StringLit);
    EXPECT_EQ(tokens[0].value, "hello");
    EXPECT_EQ(tokens[1].kind, TokenKind::StringLit);
    EXPECT_EQ(tokens[1].value, "layers.{i}");
    EXPECT_EQ(tokens[2].kind, TokenKind::StringLit);
    EXPECT_EQ(tokens[2].value, "");
}

TEST(Lexer, WeightLiterals) {
    auto tokens = lex("@input_layernorm.weight @self_attn.q_proj.weight");
    EXPECT_EQ(tokens[0].kind, TokenKind::WeightLit);
    EXPECT_EQ(tokens[0].value, "input_layernorm.weight");
    EXPECT_EQ(tokens[1].kind, TokenKind::WeightLit);
    EXPECT_EQ(tokens[1].value, "self_attn.q_proj.weight");
}

TEST(Lexer, Operators) {
    auto tokens = lex(":= = + - * / % == != < > <= >=");
    EXPECT_EQ(tokens[0].kind, TokenKind::ColonAssign);
    EXPECT_EQ(tokens[1].kind, TokenKind::Assign);
    EXPECT_EQ(tokens[2].kind, TokenKind::Plus);
    EXPECT_EQ(tokens[3].kind, TokenKind::Minus);
    EXPECT_EQ(tokens[4].kind, TokenKind::Star);
    EXPECT_EQ(tokens[5].kind, TokenKind::Slash);
    EXPECT_EQ(tokens[6].kind, TokenKind::Percent);
    EXPECT_EQ(tokens[7].kind, TokenKind::Eq);
    EXPECT_EQ(tokens[8].kind, TokenKind::NotEq);
    EXPECT_EQ(tokens[9].kind, TokenKind::Lt);
    EXPECT_EQ(tokens[10].kind, TokenKind::Gt);
    EXPECT_EQ(tokens[11].kind, TokenKind::LtEq);
    EXPECT_EQ(tokens[12].kind, TokenKind::GtEq);
}

TEST(Lexer, Delimiters) {
    auto tokens = lex("( ) { } [ ] ,");
    EXPECT_EQ(tokens[0].kind, TokenKind::LParen);
    EXPECT_EQ(tokens[1].kind, TokenKind::RParen);
    EXPECT_EQ(tokens[2].kind, TokenKind::LBrace);
    EXPECT_EQ(tokens[3].kind, TokenKind::RBrace);
    EXPECT_EQ(tokens[4].kind, TokenKind::LBracket);
    EXPECT_EQ(tokens[5].kind, TokenKind::RBracket);
    EXPECT_EQ(tokens[6].kind, TokenKind::Comma);
}

TEST(Lexer, SemicolonInsertion) {
    auto tokens = lex("x\ny\n");
    // x ; y ; EOF
    ASSERT_GE(tokens.size(), 5u);
    EXPECT_EQ(tokens[0].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[1].kind, TokenKind::Semicolon);
    EXPECT_EQ(tokens[2].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[3].kind, TokenKind::Semicolon);
    EXPECT_EQ(tokens[4].kind, TokenKind::Eof);
}

TEST(Lexer, NoSemicolonAfterComma) {
    auto tokens = lex("f(a,\nb)\n");
    // f ( a , b ) ; EOF
    ASSERT_GE(tokens.size(), 7u);
    EXPECT_EQ(tokens[0].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[1].kind, TokenKind::LParen);
    EXPECT_EQ(tokens[2].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[3].kind, TokenKind::Comma);
    EXPECT_EQ(tokens[4].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[5].kind, TokenKind::RParen);
    EXPECT_EQ(tokens[6].kind, TokenKind::Semicolon);
    EXPECT_EQ(tokens[7].kind, TokenKind::Eof);
}

TEST(Lexer, SemicolonAfterClosingTokens) {
    auto tokens = lex(")\n]\n}\n");
    EXPECT_EQ(tokens[0].kind, TokenKind::RParen);
    EXPECT_EQ(tokens[1].kind, TokenKind::Semicolon);
    EXPECT_EQ(tokens[2].kind, TokenKind::RBracket);
    EXPECT_EQ(tokens[3].kind, TokenKind::Semicolon);
    EXPECT_EQ(tokens[4].kind, TokenKind::RBrace);
    EXPECT_EQ(tokens[5].kind, TokenKind::Semicolon);
}

TEST(Lexer, Comments) {
    auto tokens = lex("x // comment\ny");
    EXPECT_EQ(tokens[0].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[0].value, "x");
    EXPECT_EQ(tokens[1].kind, TokenKind::Semicolon);
    EXPECT_EQ(tokens[2].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[2].value, "y");
}

TEST(Lexer, VarDecl) {
    auto tokens = lex("var sliding_kv Tensor\n");
    EXPECT_EQ(tokens[0].kind, TokenKind::Var);
    EXPECT_EQ(tokens[1].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[1].value, "sliding_kv");
    EXPECT_EQ(tokens[2].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[2].value, "Tensor");
    EXPECT_EQ(tokens[3].kind, TokenKind::Semicolon);
}

TEST(Lexer, FuncSignature) {
    auto tokens = lex("func f(x Tensor, i int) (Tensor, Tensor) {\n");
    EXPECT_EQ(tokens[0].kind, TokenKind::Func);
    EXPECT_EQ(tokens[1].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[1].value, "f");
    EXPECT_EQ(tokens[2].kind, TokenKind::LParen);
    EXPECT_EQ(tokens[3].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[4].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[5].kind, TokenKind::Comma);
    EXPECT_EQ(tokens[6].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[7].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[8].kind, TokenKind::RParen);
    EXPECT_EQ(tokens[9].kind, TokenKind::LParen);
    EXPECT_EQ(tokens[10].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[11].kind, TokenKind::Comma);
    EXPECT_EQ(tokens[12].kind, TokenKind::Ident);
    EXPECT_EQ(tokens[13].kind, TokenKind::RParen);
    EXPECT_EQ(tokens[14].kind, TokenKind::LBrace);
}

TEST(Lexer, ErrorUnterminatedString) {
    Lexer lexer("\"hello");
    auto tokens = lexer.tokenize();
    EXPECT_TRUE(lexer.hasError());
}

TEST(Lexer, ErrorUnexpectedChar) {
    Lexer lexer("~");
    auto tokens = lexer.tokenize();
    EXPECT_TRUE(lexer.hasError());
}
