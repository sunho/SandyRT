#include "MidIR.h"
#include "MidIRPass.h"
#include <gtest/gtest.h>

class MidIRTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sandy::ir::mid_ir::register_all_ops();
    }
};

TEST_F(MidIRTest, BuildMNISTGraph) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({-1, 784}), sandy::core::DType::F32);
    auto* w1 = builder.createWeight("fc1.weight", sandy::core::Shape({128, 784}), sandy::core::DType::F32);
    auto* b1 = builder.createWeight("fc1.bias", sandy::core::Shape({128}), sandy::core::DType::F32);
    auto* l1 = builder.createLinear(x, w1, b1);
    auto* r1 = builder.createReLU(l1);
    auto* w2 = builder.createWeight("fc2.weight", sandy::core::Shape({10, 128}), sandy::core::DType::F32);
    auto* b2 = builder.createWeight("fc2.bias", sandy::core::Shape({10}), sandy::core::DType::F32);
    auto* l2 = builder.createLinear(r1, w2, b2);

    sandy::ir::mid_ir::Value* outs[] = {l2};
    builder.setOutputs(outs);

    EXPECT_EQ(graph.entry()->ops.size(), 8u);
    EXPECT_EQ(graph.outputs().size(), 1u);
    EXPECT_EQ(graph.outputs()[0], l2);

    graph.dump();
}

TEST_F(MidIRTest, LinearTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({-1, 784}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({128, 784}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({128}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);

    EXPECT_EQ(out->shape.rank(), 2);
    EXPECT_EQ(out->shape.dim(0), -1);
    EXPECT_EQ(out->shape.dim(1), 128);
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
}

TEST_F(MidIRTest, ReLUTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({4, 128}), sandy::core::DType::F32);
    auto* out = builder.createReLU(x);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
}

TEST_F(MidIRTest, BinaryElementwiseAllowsScalarDTypeMismatch) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({4, 128}), sandy::core::DType::BF16);
    auto* scale = builder.createConstantF32(0.5f);
    auto* out = builder.createMul(x, scale);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, sandy::core::DType::BF16);
}

TEST_F(MidIRTest, RMSNormTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("norm.weight", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createRMSNorm(x, weight);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::RMSNorm);
}

TEST_F(MidIRTest, RMSNormWithoutScaleTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createRMSNorm(x);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::RMSNorm);
    ASSERT_EQ(out->def->operands.size(), 1u);
    EXPECT_EQ(out->def->operands[0], x);
}

TEST_F(MidIRTest, LayerNormTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 4, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("ln.weight", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("ln.bias", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createLayerNorm(x, weight, bias, 1.0e-5f);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::LayerNorm);
}

TEST_F(MidIRTest, BinaryElementwiseBroadcastTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput(0, sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({3, 1}), sandy::core::DType::F32);
    auto* add = builder.createAdd(lhs, rhs);
    auto* mul = builder.createMul(lhs, rhs);

    EXPECT_EQ(add->shape, sandy::core::Shape({2, 3, 4}));
    EXPECT_EQ(add->dtype, sandy::core::DType::F32);
    EXPECT_EQ(add->def->kind, sandy::ir::mid_ir::OpKind::Add);
    EXPECT_EQ(mul->shape, sandy::core::Shape({2, 3, 4}));
    EXPECT_EQ(mul->dtype, sandy::core::DType::F32);
    EXPECT_EQ(mul->def->kind, sandy::ir::mid_ir::OpKind::Mul);
}

TEST_F(MidIRTest, SqrtTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createSqrt(x);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Sqrt);
}

TEST_F(MidIRTest, ConstantAndTanhTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* c = builder.createConstantF32(1.25f);
    auto* out = builder.createTanh(c);

    EXPECT_EQ(c->shape, sandy::core::Shape({}));
    EXPECT_EQ(c->dtype, sandy::core::DType::F32);
    EXPECT_EQ(c->def->kind, sandy::ir::mid_ir::OpKind::Constant);
    EXPECT_EQ(c->def->attrs.at("value").floatVal, 1.25);
    EXPECT_EQ(out->shape, c->shape);
    EXPECT_EQ(out->dtype, c->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Tanh);
}

TEST_F(MidIRTest, MatMulTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput(0, sandy::core::Shape({2, 4, 3}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({1, 3, 5}), sandy::core::DType::F32);
    auto* out = builder.createMatMul(lhs, rhs);

    EXPECT_EQ(out->shape, sandy::core::Shape({2, 4, 5}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::MatMul);
}

TEST_F(MidIRTest, MatMulTransposedRhsTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput(0, sandy::core::Shape({2, 4, 3}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({1, 5, 3}), sandy::core::DType::F32);
    auto* out = builder.createMatMul(lhs, rhs, false, true);

    EXPECT_EQ(out->shape, sandy::core::Shape({2, 4, 5}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::MatMul);
    ASSERT_TRUE(out->def->attrs.contains("transpose_rhs"));
    EXPECT_EQ(out->def->attrs.at("transpose_rhs").intVal, 1);
}

TEST_F(MidIRTest, Transpose2DTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({3, 5}), sandy::core::DType::F32);
    auto* out = builder.createTranspose(x);

    EXPECT_EQ(out->shape, sandy::core::Shape({5, 3}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Transpose);
}

TEST_F(MidIRTest, FuseTransposeIntoMatMulPassFusesDeadRhsTranspose) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("w", sandy::core::Shape({5, 3}), sandy::core::DType::F32);
    auto* transposed = builder.createTranspose(weight);
    auto* out = builder.createMatMul(lhs, transposed);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto pass = sandy::ir::mid_ir::createFuseTransposeIntoMatMulPass();
    auto result = pass->run(graph);
    ASSERT_TRUE(result) << result.error();
    EXPECT_TRUE(result->changed);

    ASSERT_EQ(out->def->operands.size(), 2u);
    EXPECT_EQ(out->def->operands[1], weight);
    ASSERT_TRUE(out->def->attrs.contains("transpose_rhs"));
    EXPECT_EQ(out->def->attrs.at("transpose_rhs").intVal, 1);
    EXPECT_TRUE(transposed->uses.empty());
    EXPECT_EQ(graph.entry()->ops.size(), 3u);
}

TEST_F(MidIRTest, FuseTransposeIntoMatMulPassKeepsSharedTranspose) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("w", sandy::core::Shape({5, 3}), sandy::core::DType::F32);
    auto* transposed = builder.createTranspose(weight);
    auto* out = builder.createMatMul(lhs, transposed);
    auto* alsoOut = builder.createTanh(transposed);
    sandy::ir::mid_ir::Value* outputs[] = {out, alsoOut};
    builder.setOutputs(outputs);

    auto pass = sandy::ir::mid_ir::createFuseTransposeIntoMatMulPass();
    auto result = pass->run(graph);
    ASSERT_TRUE(result) << result.error();
    EXPECT_TRUE(result->changed);

    EXPECT_EQ(out->def->operands[1], weight);
    ASSERT_TRUE(out->def->attrs.contains("transpose_rhs"));
    EXPECT_EQ(out->def->attrs.at("transpose_rhs").intVal, 1);
    EXPECT_FALSE(transposed->uses.empty());
    EXPECT_EQ(graph.entry()->ops.size(), 5u);
}

TEST_F(MidIRTest, DeadCodeEliminationPreservesPagedAppendSideEffect) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* cache = builder.createPagedTensorInput(
        0,
        sandy::core::Shape({1, 1, -1, 4}),
        sandy::core::DType::BF16,
        2,
        16);
    auto* chunk = builder.createInput(
        1,
        sandy::core::Shape({1, 1, 1, 4}),
        sandy::core::DType::BF16);
    auto* output = builder.createInput(
        2,
        sandy::core::Shape({1}),
        sandy::core::DType::F32);
    builder.createPagedAppend(cache, chunk);
    sandy::ir::mid_ir::Value* outputs[] = {output};
    builder.setOutputs(outputs);

    auto pass = sandy::ir::mid_ir::createDeadCodeEliminationPass();
    auto result = pass->run(graph);
    ASSERT_TRUE(result) << result.error();

    bool foundAppend = false;
    for (auto* op : graph.entry()->ops) {
        if (op && op->kind == sandy::ir::mid_ir::OpKind::PagedAppend)
            foundAppend = true;
    }
    EXPECT_TRUE(foundAppend);
}

TEST_F(MidIRTest, ReshapeTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {2, 12});

    EXPECT_EQ(out->shape, sandy::core::Shape({2, 12}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Reshape);
    ASSERT_EQ(out->def->attrs.at("shape").intListVal.size(), 2u);
    EXPECT_EQ(out->def->attrs.at("shape").intListVal[1], 12);
}

TEST_F(MidIRTest, ReshapeInfersNegativeOneDimension) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {-1, 4});

    EXPECT_EQ(out->shape, sandy::core::Shape({6, 4}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    ASSERT_EQ(out->def->attrs.at("shape").intListVal.size(), 2u);
    EXPECT_EQ(out->def->attrs.at("shape").intListVal[0], -1);
}

TEST_F(MidIRTest, ReshapePreservesMultipleDynamicDimensions) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({-1, -1, 2048}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {-1, -1, 8, 256});

    EXPECT_EQ(out->shape, sandy::core::Shape({-1, -1, 8, 256}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    ASSERT_EQ(out->def->attrs.at("shape").intListVal.size(), 4u);
    EXPECT_EQ(out->def->attrs.at("shape").intListVal[0], -1);
    EXPECT_EQ(out->def->attrs.at("shape").intListVal[1], -1);
}

TEST_F(MidIRTest, ReshapeResolvesMultipleLeadingDynamicDimensionsFromStaticInput) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({1, 32, 2048}), sandy::core::DType::F32);
    auto* q = builder.createReshape(x, {-1, -1, 8, 256});

    EXPECT_EQ(q->shape, sandy::core::Shape({1, 32, 8, 256}));

    auto* flat = builder.createReshape(q, {-1, -1, 2048});
    EXPECT_EQ(flat->shape, sandy::core::Shape({1, 32, 2048}));
}

TEST_F(MidIRTest, GemmaAttentionShapeInferenceSupportsDynamicBatchAndSeq) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* qFlat = builder.createInput(0, sandy::core::Shape({-1, -1, 2048}), sandy::core::DType::F32);
    auto* kFlat = builder.createInput(1, sandy::core::Shape({-1, -1, 512}), sandy::core::DType::F32);
    auto* vFlat = builder.createInput(2, sandy::core::Shape({-1, -1, 512}), sandy::core::DType::F32);

    auto* q = builder.createPermute(
        builder.createReshape(qFlat, {-1, -1, 8, 256}),
        {0, 2, 1, 3});
    auto* k = builder.createPermute(
        builder.createReshape(kFlat, {-1, -1, 2, 256}),
        {0, 2, 1, 3});
    auto* v = builder.createPermute(
        builder.createReshape(vFlat, {-1, -1, 2, 256}),
        {0, 2, 1, 3});

    EXPECT_EQ(q->shape, sandy::core::Shape({-1, 8, -1, 256}));
    EXPECT_EQ(k->shape, sandy::core::Shape({-1, 2, -1, 256}));

    auto* scores = builder.createSlidingQueryKeyScore(q, k, 512, 1.0f);
    EXPECT_EQ(scores->shape, sandy::core::Shape({-1, 8, -1, -1}));

    auto* probs = builder.createSoftmax(scores, -1);
    auto* context = builder.createMatMul(probs, v);
    EXPECT_EQ(context->shape, sandy::core::Shape({-1, 8, -1, 256}));

    auto* contextSeqMajor = builder.createPermute(context, {0, 2, 1, 3});
    auto* contextFlat = builder.createReshape(contextSeqMajor, {-1, -1, 2048});
    EXPECT_EQ(contextFlat->shape, sandy::core::Shape({-1, -1, 2048}));
}

TEST_F(MidIRTest, PermuteTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4, 5}), sandy::core::DType::F32);
    auto* out = builder.createPermute(x, {0, 2, 1, 3});

    EXPECT_EQ(out->shape, sandy::core::Shape({2, 4, 3, 5}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Permute);
    ASSERT_EQ(out->def->attrs.at("dims").intListVal.size(), 4u);
    EXPECT_EQ(out->def->attrs.at("dims").intListVal[1], 2);
}

TEST_F(MidIRTest, SlidingQueryKeyScoreTypeInferenceSupportsRank3AndRank4) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* q = builder.createInput(0, sandy::core::Shape({4, 3, 8}), sandy::core::DType::F32);
    auto* k = builder.createInput(1, sandy::core::Shape({2, 5, 8}), sandy::core::DType::F32);
    auto* out = builder.createSlidingQueryKeyScore(q, k, 2);

    EXPECT_EQ(out->shape, sandy::core::Shape({4, 3, 5}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::SlidingQueryKeyScore);
    EXPECT_EQ(out->def->attrs.at("window").intVal, 2);

    auto* bq = builder.createInput(2, sandy::core::Shape({2, 4, 3, 8}), sandy::core::DType::F32);
    auto* bk = builder.createInput(3, sandy::core::Shape({2, 2, 5, 8}), sandy::core::DType::F32);
    auto* bout = builder.createSlidingQueryKeyScore(bq, bk);

    EXPECT_EQ(bout->shape, sandy::core::Shape({2, 4, 3, 5}));
    EXPECT_EQ(bout->dtype, sandy::core::DType::F32);
    EXPECT_EQ(bout->def->kind, sandy::ir::mid_ir::OpKind::SlidingQueryKeyScore);
}

TEST_F(MidIRTest, AttentionTypeInferenceSupportsGroupedKVHeads) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* q = builder.createInput(0, sandy::core::Shape({2, 8, 3, 4}), sandy::core::DType::F32);
    auto* k = builder.createInput(1, sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    auto* v = builder.createInput(2, sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    auto* out = builder.createAttention(q, k, v, 512, 1.0f);

    EXPECT_EQ(out->shape, q->shape);
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Attention);
    EXPECT_EQ(out->def->attrs.at("window").intVal, 512);
    EXPECT_EQ(out->def->attrs.at("scale").floatVal, 1.0);
}

TEST_F(MidIRTest, AttentionTypeInferenceSupportsBatchedPositionOffsets) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* q = builder.createInput(0, sandy::core::Shape({2, 8, 1, 4}), sandy::core::DType::F32);
    auto* k = builder.createInput(1, sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    auto* v = builder.createInput(2, sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    auto* positionOffsets = builder.createInput(3, sandy::core::Shape({2}), sandy::core::DType::I64);
    auto* out = builder.createAttention(q, k, v, positionOffsets, 512, 1.0f);

    EXPECT_EQ(out->shape, q->shape);
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Attention);
    ASSERT_EQ(out->def->operands.size(), 4u);
    EXPECT_EQ(out->def->operands[3], positionOffsets);
}

TEST_F(MidIRTest, SoftmaxTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* out = builder.createSoftmax(x, -1);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Softmax);
    EXPECT_EQ(out->def->attrs.at("dim").intVal, -1);
}

TEST_F(MidIRTest, EmbeddingTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* ids = builder.createInput(0, sandy::core::Shape({2, 4}), sandy::core::DType::I32);
    auto* weight = builder.createWeight("embed_tokens.weight", sandy::core::Shape({10, 3}), sandy::core::DType::F32);
    auto* out = builder.createEmbedding(ids, weight);

    EXPECT_EQ(out->shape, sandy::core::Shape({2, 4, 3}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Embedding);
}

TEST_F(MidIRTest, RoPETypeInferenceSupportsArbitraryRank) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4, 5, 6}), sandy::core::DType::F32);
    auto* out = builder.createRoPE(x, 10000.0f);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::RoPE);
    EXPECT_EQ(out->def->attrs.at("rope_theta").floatVal, 10000.0);
}

TEST_F(MidIRTest, RoPETypeInferenceSupportsPartialRotaryDim) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4, 8}), sandy::core::DType::F32);
    auto* out = builder.createRoPE(x, 1000000.0f, 4);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::RoPE);
    EXPECT_EQ(out->def->attrs.at("rope_theta").floatVal, 1000000.0);
    EXPECT_EQ(out->def->attrs.at("rotary_dim").intVal, 4);
}

TEST_F(MidIRTest, UseDefChains) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({-1, 784}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({128, 784}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({128}), sandy::core::DType::F32);
    auto* l = builder.createLinear(x, w, b);
    auto* r = builder.createReLU(l);

    EXPECT_EQ(x->uses.size(), 1u);
    EXPECT_EQ(x->uses[0].op, l->def);
    EXPECT_EQ(x->uses[0].operand, 0);

    EXPECT_EQ(l->uses.size(), 1u);
    EXPECT_EQ(l->uses[0].op, r->def);

    EXPECT_EQ(l->def->kind, sandy::ir::mid_ir::OpKind::Linear);
    EXPECT_EQ(r->def->kind, sandy::ir::mid_ir::OpKind::ReLU);
}

TEST_F(MidIRTest, InputWeightAttrs) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({1, 784}), sandy::core::DType::F32);
    auto* w = builder.createWeight("fc1.weight", sandy::core::Shape({128, 784}), sandy::core::DType::F32);

    EXPECT_EQ(x->def->attrs.at("index").intVal, 0);
    EXPECT_EQ(x->def->attrs.count("name"), 0u);
    EXPECT_EQ(w->def->attrs.at("name").strVal, "fc1.weight");
}

TEST_F(MidIRTest, PagedTensorInputAttrs) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* cache = builder.createPagedTensorInput(
        1,
        sandy::core::Shape({2, 4, -1, 128}),
        sandy::core::DType::BF16,
        2,
        16);

    EXPECT_EQ(cache->shape, sandy::core::Shape({2, 4, -1, 128}));
    EXPECT_EQ(cache->dtype, sandy::core::DType::BF16);
    ASSERT_NE(cache->def, nullptr);
    EXPECT_EQ(cache->def->kind, sandy::ir::mid_ir::OpKind::PagedTensorInput);
    EXPECT_EQ(cache->def->attrs.at("index").intVal, 1);
    EXPECT_EQ(cache->def->attrs.at("grow_dim").intVal, 2);
    EXPECT_EQ(cache->def->attrs.at("page_size").intVal, 16);

    const auto& dims = cache->def->attrs.at("dims").intListVal;
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 4);
    EXPECT_EQ(dims[2], -1);
    EXPECT_EQ(dims[3], 128);
}
