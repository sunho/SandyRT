#include "MidIR.h"
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

    auto* x = builder.createInput("x", sandy::core::Shape({-1, 784}), sandy::core::DType::F32);
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

    auto* x = builder.createInput("x", sandy::core::Shape({-1, 784}), sandy::core::DType::F32);
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

    auto* x = builder.createInput("x", sandy::core::Shape({4, 128}), sandy::core::DType::F32);
    auto* out = builder.createReLU(x);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
}

TEST_F(MidIRTest, RMSNormTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput("x", sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("norm.weight", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createRMSNorm(x, weight);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::RMSNorm);
}

TEST_F(MidIRTest, BinaryElementwiseBroadcastTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput("lhs", sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* rhs = builder.createInput("rhs", sandy::core::Shape({3, 1}), sandy::core::DType::F32);
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

    auto* x = builder.createInput("x", sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createSqrt(x);

    EXPECT_EQ(out->shape, x->shape);
    EXPECT_EQ(out->dtype, x->dtype);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Sqrt);
}

TEST_F(MidIRTest, MatMulTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* lhs = builder.createInput("lhs", sandy::core::Shape({2, 4, 3}), sandy::core::DType::F32);
    auto* rhs = builder.createInput("rhs", sandy::core::Shape({1, 3, 5}), sandy::core::DType::F32);
    auto* out = builder.createMatMul(lhs, rhs);

    EXPECT_EQ(out->shape, sandy::core::Shape({2, 4, 5}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::MatMul);
}

TEST_F(MidIRTest, Transpose2DTypeInference) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput("x", sandy::core::Shape({3, 5}), sandy::core::DType::F32);
    auto* out = builder.createTranspose(x);

    EXPECT_EQ(out->shape, sandy::core::Shape({5, 3}));
    EXPECT_EQ(out->dtype, sandy::core::DType::F32);
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Transpose);
}

TEST_F(MidIRTest, UseDefChains) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput("x", sandy::core::Shape({-1, 784}), sandy::core::DType::F32);
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

    auto* x = builder.createInput("x", sandy::core::Shape({1, 784}), sandy::core::DType::F32);
    auto* w = builder.createWeight("fc1.weight", sandy::core::Shape({128, 784}), sandy::core::DType::F32);

    EXPECT_EQ(x->def->attrs.at("name").strVal, "x");
    EXPECT_EQ(w->def->attrs.at("name").strVal, "fc1.weight");
}
