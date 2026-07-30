#include "Compiler.h"
#include "SafeTensorWeights.h"
#include <gtest/gtest.h>

TEST(MaterializeTest, MNISTEndToEnd) {
    sandy::Compiler compiler;
    auto high_graph = compiler.load_sandygo("../src/models/mnist.sandy.go");

    std::cout << "=== HighIR ===" << std::endl;
    high_graph.dump();

    auto weights = sandy::weight::EagerSafeTensorWeights::load("../experiments/mnist/mnist.safetensors");

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::ir::TensorDesc(sandy::ir::Shape({1, 784}), sandy::ir::DType::F32);

    auto result = compiler.materialize_mid_ir(high_graph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto mid_graph = result.take();

    std::cout << "\n=== MidIR ===" << std::endl;
    mid_graph.dump();

    auto* entry = mid_graph.entry();
    EXPECT_EQ(entry->ops.size(), 8u);
    EXPECT_EQ(mid_graph.outputs().size(), 1u);

    auto* out = mid_graph.outputs()[0];
    EXPECT_EQ(out->def->kind, sandy::ir::mid_ir::OpKind::Linear);
    EXPECT_EQ(out->shape.rank(), 2);
    EXPECT_EQ(out->shape.dim(0), 1);
    EXPECT_EQ(out->shape.dim(1), 10);
}

TEST(MaterializeTest, MissingInputShape) {
    sandy::Compiler compiler;
    auto high_graph = compiler.load_sandygo("../src/models/mnist.sandy.go");
    auto weights = sandy::weight::EagerSafeTensorWeights::load("../experiments/mnist/mnist.safetensors");

    auto result = compiler.materialize_mid_ir(high_graph, weights);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("no shape provided"), std::string::npos);
}
