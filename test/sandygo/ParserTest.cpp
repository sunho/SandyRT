#include "Lexer.h"
#include "Parser.h"
#include <gtest/gtest.h>

using namespace sandy::sandygo;

static Program parseSource(const std::string& src) {
    Lexer lexer(src);
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.hasError()) << lexer.errorMessage();
    Parser parser(std::move(tokens));
    auto prog = parser.parse();
    EXPECT_FALSE(parser.hasError()) << parser.errorMessage();
    return prog;
}

TEST(Parser, SimpleFunction) {
    auto prog = parseSource(R"(
func f() Tensor {
    return x
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    EXPECT_EQ(prog.funcs[0].name, "f");
    EXPECT_EQ(prog.funcs[0].params.size(), 0u);
    ASSERT_EQ(prog.funcs[0].returnTypes.size(), 1u);
    EXPECT_EQ(prog.funcs[0].returnTypes[0].name, "Tensor");
    ASSERT_EQ(prog.funcs[0].body.size(), 1u);
    EXPECT_EQ(prog.funcs[0].body[0]->kind, Stmt::Return);
}

TEST(Parser, ImportDecl) {
    auto prog = parseSource(R"(
import "layers.sandy.go"

func f() Tensor {
    return x
}
)");
    ASSERT_EQ(prog.imports.size(), 1u);
    EXPECT_EQ(prog.imports[0].path, "layers.sandy.go");
    EXPECT_EQ(prog.imports[0].line, 2);
    ASSERT_EQ(prog.funcs.size(), 1u);
    EXPECT_EQ(prog.funcs[0].name, "f");
}

TEST(Parser, FunctionWithParams) {
    auto prog = parseSource(R"(
func f(x Tensor, i int, s string) Tensor {
    return x
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    auto& fn = prog.funcs[0];
    ASSERT_EQ(fn.params.size(), 3u);
    EXPECT_EQ(fn.params[0].name, "x");
    EXPECT_EQ(fn.params[0].type.name, "Tensor");
    EXPECT_EQ(fn.params[1].name, "i");
    EXPECT_EQ(fn.params[1].type.name, "int");
    EXPECT_EQ(fn.params[2].name, "s");
    EXPECT_EQ(fn.params[2].type.name, "string");
}

TEST(Parser, PagedTensorParamWithShape) {
    auto prog = parseSource(R"(
func main(k PagedTensor[[2, -1, 128], page_size=16], v PagedTensor[[2, -1, 128], page_size=32]) Tensor {
    return k
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    auto& fn = prog.funcs[0];
    ASSERT_EQ(fn.params.size(), 2u);
    EXPECT_EQ(fn.params[0].name, "k");
    EXPECT_EQ(fn.params[0].type.name, "PagedTensor");
    EXPECT_EQ(fn.params[0].type.dims, (std::vector<int64_t>{2, -1, 128}));
    EXPECT_EQ(fn.params[0].type.pageSize, 16);
    EXPECT_EQ(fn.params[1].name, "v");
    EXPECT_EQ(fn.params[1].type.name, "PagedTensor");
    EXPECT_EQ(fn.params[1].type.dims, (std::vector<int64_t>{2, -1, 128}));
    EXPECT_EQ(fn.params[1].type.pageSize, 32);
    ASSERT_EQ(fn.returnTypes.size(), 1u);
    EXPECT_EQ(fn.returnTypes[0].name, "Tensor");
}

TEST(Parser, PagedTensorParamRequiresPageSize) {
    Lexer lexer(R"(
func main(k PagedTensor[[2, -1, 128]]) Tensor {
    return k
}
)");
    auto tokens = lexer.tokenize();
    ASSERT_FALSE(lexer.hasError()) << lexer.errorMessage();

    Parser parser(std::move(tokens));
    (void)parser.parse();
    EXPECT_TRUE(parser.hasError());
}

TEST(Parser, MultiReturn) {
    auto prog = parseSource(R"(
func f(x Tensor) (Tensor, Tensor) {
    return x, x
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    auto& fn = prog.funcs[0];
    ASSERT_EQ(fn.returnTypes.size(), 2u);
    EXPECT_EQ(fn.returnTypes[0].name, "Tensor");
    EXPECT_EQ(fn.returnTypes[1].name, "Tensor");
    ASSERT_EQ(fn.body.size(), 1u);
    EXPECT_EQ(fn.body[0]->kind, Stmt::Return);
    EXPECT_EQ(fn.body[0]->values.size(), 2u);
}

TEST(Parser, SliceParam) {
    auto prog = parseSource(R"(
func f(xs []Tensor) Tensor {
    return xs
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    ASSERT_EQ(prog.funcs[0].params.size(), 1u);
    EXPECT_EQ(prog.funcs[0].params[0].type.kind, TypeExpr::Slice);
    EXPECT_EQ(prog.funcs[0].params[0].type.name, "Tensor");
}

TEST(Parser, FixedTensorTupleParamWithExplicitElementType) {
    auto prog = parseSource(R"(
func main(k [8]PagedTensor[[128], bf16, page_size=16]) []Tensor {
    return k
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    ASSERT_EQ(prog.funcs[0].params.size(), 1u);
    const auto& type = prog.funcs[0].params[0].type;
    EXPECT_EQ(type.kind, TypeExpr::FixedTensorTuple);
    EXPECT_EQ(type.tupleLen, 8);
    EXPECT_EQ(type.name, "PagedTensor");
    EXPECT_EQ(type.dims, (std::vector<int64_t>{128}));
    EXPECT_EQ(type.dtype, "bf16");
    EXPECT_EQ(type.pageSize, 16);
    ASSERT_EQ(prog.funcs[0].returnTypes.size(), 1u);
    EXPECT_EQ(prog.funcs[0].returnTypes[0].kind, TypeExpr::Slice);
}

TEST(Parser, VarDecl) {
    auto prog = parseSource(R"(
func f() {
    var x Tensor
    var y int
}
)");
    ASSERT_EQ(prog.funcs[0].body.size(), 2u);
    EXPECT_EQ(prog.funcs[0].body[0]->kind, Stmt::VarDecl);
    EXPECT_EQ(prog.funcs[0].body[0]->name, "x");
    EXPECT_EQ(prog.funcs[0].body[0]->type.name, "Tensor");
    EXPECT_EQ(prog.funcs[0].body[1]->kind, Stmt::VarDecl);
    EXPECT_EQ(prog.funcs[0].body[1]->name, "y");
    EXPECT_EQ(prog.funcs[0].body[1]->type.name, "int");
}

TEST(Parser, ShortVarDecl) {
    auto prog = parseSource(R"(
func f() {
    x := 42
}
)");
    ASSERT_EQ(prog.funcs[0].body.size(), 1u);
    auto& stmt = prog.funcs[0].body[0];
    EXPECT_EQ(stmt->kind, Stmt::Assign);
    EXPECT_TRUE(stmt->isDecl);
    ASSERT_EQ(stmt->targets.size(), 1u);
    EXPECT_EQ(stmt->targets[0]->kind, Expr::Ident);
    EXPECT_EQ(stmt->targets[0]->sval, "x");
    EXPECT_EQ(stmt->value->kind, Expr::IntLit);
    EXPECT_EQ(stmt->value->ival, 42);
}

TEST(Parser, MultiAssign) {
    auto prog = parseSource(R"(
func f() {
    x, y := call()
}
)");
    ASSERT_EQ(prog.funcs[0].body.size(), 1u);
    auto& stmt = prog.funcs[0].body[0];
    EXPECT_EQ(stmt->kind, Stmt::Assign);
    EXPECT_TRUE(stmt->isDecl);
    ASSERT_EQ(stmt->targets.size(), 2u);
    EXPECT_EQ(stmt->targets[0]->sval, "x");
    EXPECT_EQ(stmt->targets[1]->sval, "y");
    EXPECT_EQ(stmt->value->kind, Expr::Call);
}

TEST(Parser, ForRange) {
    auto prog = parseSource(R"(
func f() {
    for i := range(10) {
        x := i
    }
}
)");
    ASSERT_EQ(prog.funcs[0].body.size(), 1u);
    auto& stmt = prog.funcs[0].body[0];
    EXPECT_EQ(stmt->kind, Stmt::For);
    EXPECT_EQ(stmt->name, "i");
    ASSERT_NE(stmt->iterExpr, nullptr);
    EXPECT_EQ(stmt->iterExpr->kind, Expr::Call);
    ASSERT_EQ(stmt->stmts.size(), 1u);
}

TEST(Parser, ForRangeTwoArgs) {
    auto prog = parseSource(R"(
func f() {
    for i := range(15, 35) {
        x := i
    }
}
)");
    auto& stmt = prog.funcs[0].body[0];
    EXPECT_EQ(stmt->kind, Stmt::For);
    EXPECT_EQ(stmt->iterExpr->kind, Expr::Call);
    EXPECT_EQ(stmt->iterExpr->args.size(), 2u);
    EXPECT_EQ(stmt->iterExpr->args[0]->ival, 15);
    EXPECT_EQ(stmt->iterExpr->args[1]->ival, 35);
}

TEST(Parser, IfElse) {
    auto prog = parseSource(R"(
func f() {
    if x == 1 {
        y := 2
    } else {
        y := 3
    }
}
)");
    auto& stmt = prog.funcs[0].body[0];
    EXPECT_EQ(stmt->kind, Stmt::If);
    ASSERT_NE(stmt->cond, nullptr);
    EXPECT_EQ(stmt->cond->kind, Expr::Binary);
    EXPECT_EQ(stmt->cond->op, "==");
    ASSERT_EQ(stmt->stmts.size(), 1u);
    ASSERT_EQ(stmt->elseStmts.size(), 1u);
}

TEST(Parser, WeightScope) {
    auto prog = parseSource(R"(
func f() {
    weight_scope "model.layers.{i}" {
        x := @input_layernorm.weight
    }
}
)");
    auto& stmt = prog.funcs[0].body[0];
    EXPECT_EQ(stmt->kind, Stmt::WeightScope);
    ASSERT_NE(stmt->scopeExpr, nullptr);
    EXPECT_EQ(stmt->scopeExpr->kind, Expr::StringLit);
    EXPECT_EQ(stmt->scopeExpr->sval, "model.layers.{i}");
    ASSERT_EQ(stmt->stmts.size(), 1u);
}

TEST(Parser, FunctionCallWithNamedArgs) {
    auto prog = parseSource(R"(
func f() {
    x := __attention(q, k, v, window=512)
}
)");
    auto& stmt = prog.funcs[0].body[0];
    auto& call = stmt->value;
    ASSERT_EQ(call->kind, Expr::Call);
    EXPECT_EQ(call->left->sval, "__attention");
    ASSERT_EQ(call->args.size(), 3u);
    EXPECT_EQ(call->args[0]->sval, "q");
    EXPECT_EQ(call->args[1]->sval, "k");
    EXPECT_EQ(call->args[2]->sval, "v");
    ASSERT_EQ(call->namedArgs.size(), 1u);
    EXPECT_EQ(call->namedArgs[0].name, "window");
    EXPECT_EQ(call->namedArgs[0].value->ival, 512);
}

TEST(Parser, FunctionCallWithIntListNamedArg) {
    auto prog = parseSource(R"(
func f() {
    x := __reshape(x, shape=[-1, 16, 12, 64])
}
)");
    auto& call = prog.funcs[0].body[0]->value;
    ASSERT_EQ(call->kind, Expr::Call);
    ASSERT_EQ(call->namedArgs.size(), 1u);
    EXPECT_EQ(call->namedArgs[0].name, "shape");
    ASSERT_EQ(call->namedArgs[0].value->kind, Expr::IntListLit);
    EXPECT_EQ(call->namedArgs[0].value->intListVal, (std::vector<int64_t>{-1, 16, 12, 64}));
}

TEST(Parser, IndexExpr) {
    auto prog = parseSource(R"(
func f() {
    x := kvs[i - 1]
}
)");
    auto& val = prog.funcs[0].body[0]->value;
    EXPECT_EQ(val->kind, Expr::Index);
    EXPECT_EQ(val->left->sval, "kvs");
    EXPECT_EQ(val->right->kind, Expr::Binary);
    EXPECT_EQ(val->right->op, "-");
}

TEST(Parser, ArithmeticPrecedence) {
    auto prog = parseSource(R"(
func f() {
    x := a + b * c
}
)");
    // Should parse as a + (b * c)
    auto& val = prog.funcs[0].body[0]->value;
    EXPECT_EQ(val->kind, Expr::Binary);
    EXPECT_EQ(val->op, "+");
    EXPECT_EQ(val->left->kind, Expr::Ident);
    EXPECT_EQ(val->right->kind, Expr::Binary);
    EXPECT_EQ(val->right->op, "*");
}

TEST(Parser, ModuloInCondition) {
    auto prog = parseSource(R"(
func f() {
    if i % 5 == 4 {
        x := 1
    }
}
)");
    auto& cond = prog.funcs[0].body[0]->cond;
    EXPECT_EQ(cond->kind, Expr::Binary);
    EXPECT_EQ(cond->op, "==");
    // LHS should be i % 5
    EXPECT_EQ(cond->left->kind, Expr::Binary);
    EXPECT_EQ(cond->left->op, "%");
}

TEST(Parser, NestedCalls) {
    auto prog = parseSource(R"(
func f() {
    x := __mul(x, __sqrt(1536))
}
)");
    auto& call = prog.funcs[0].body[0]->value;
    EXPECT_EQ(call->kind, Expr::Call);
    EXPECT_EQ(call->left->sval, "__mul");
    ASSERT_EQ(call->args.size(), 2u);
    EXPECT_EQ(call->args[1]->kind, Expr::Call);
    EXPECT_EQ(call->args[1]->left->sval, "__sqrt");
}

TEST(Parser, MultipleFunctions) {
    auto prog = parseSource(R"(
func a() {
}

func b(x Tensor) Tensor {
    return x
}
)");
    ASSERT_EQ(prog.funcs.size(), 2u);
    EXPECT_EQ(prog.funcs[0].name, "a");
    EXPECT_EQ(prog.funcs[1].name, "b");
}

TEST(Parser, GemmaLayerStructure) {
    auto prog = parseSource(R"(
func gemma_kv_layer(x Tensor, i int, window int, head_dim int, rope_theta float) (Tensor, Tensor) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = __attention(q, k, v, window=window)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)
        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = __gated_mlp(h,
            @mlp.gate_proj.weight,
            @mlp.up_proj.weight,
            @mlp.down_proj.weight,
            act="gelu",
        )
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)
    }
    return x, kv
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    auto& fn = prog.funcs[0];
    EXPECT_EQ(fn.name, "gemma_kv_layer");
    ASSERT_EQ(fn.params.size(), 5u);
    ASSERT_EQ(fn.returnTypes.size(), 2u);
    // Body: weight_scope { ... }, return
    ASSERT_EQ(fn.body.size(), 2u);
    EXPECT_EQ(fn.body[0]->kind, Stmt::WeightScope);
    EXPECT_EQ(fn.body[1]->kind, Stmt::Return);
    // weight_scope body has 8 statements
    EXPECT_EQ(fn.body[0]->stmts.size(), 8u);
}

TEST(Parser, MainFunction) {
    auto prog = parseSource(R"(
func main(input_ids Tensor) Tensor {
    weight_scope "language_model.model" {
        x := __embedding(input_ids, @embed_tokens.weight)
        x = __mul(x, __sqrt(1536))
        var sliding_kv Tensor
        var full_kv Tensor
        for i := range(15) {
            if i % 5 == 4 {
                x, full_kv = gemma_kv_layer(x, i, 0, 512, 1000000.0)
            } else {
                x, sliding_kv = gemma_kv_layer(x, i, 512, 256, 10000.0)
            }
        }
        for i := range(15, 35) {
            if i % 5 == 4 {
                x = gemma_layer(x, i, full_kv, 0, 512, 1000000.0)
            } else {
                x = gemma_layer(x, i, sliding_kv, 512, 256, 10000.0)
            }
        }
        x = __rms_norm(x, @norm.weight)
        logits := __matmul(x, @embed_tokens.weight)
        logits = __softcap(logits, 30.0)
    }
    return logits
}
)");
    ASSERT_EQ(prog.funcs.size(), 1u);
    auto& fn = prog.funcs[0];
    EXPECT_EQ(fn.name, "main");
    ASSERT_EQ(fn.params.size(), 1u);
    EXPECT_EQ(fn.params[0].name, "input_ids");
    ASSERT_EQ(fn.returnTypes.size(), 1u);
    // Body: weight_scope, return
    ASSERT_EQ(fn.body.size(), 2u);
    EXPECT_EQ(fn.body[0]->kind, Stmt::WeightScope);
    EXPECT_EQ(fn.body[1]->kind, Stmt::Return);
}
