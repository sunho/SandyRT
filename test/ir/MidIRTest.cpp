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
