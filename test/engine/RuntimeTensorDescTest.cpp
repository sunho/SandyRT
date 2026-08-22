#include "RuntimeTensorDesc.h"

#include <gtest/gtest.h>

namespace {

namespace kir = sandy::ir::kernel_ir;

kir::ValueType tensorType(
        std::initializer_list<int64_t> dims,
        sandy::core::DType dtype = sandy::core::DType::F32) {
    kir::ValueType type;
    type.kind = kir::ValueKind::Tensor;
    type.dtype = dtype;
    type.shape = sandy::core::Shape(dims);
    return type;
}

kir::InputSource argument(int64_t index) {
    kir::InputSource source;
    source.kind = kir::InputSourceKind::Argument;
    source.index = index;
    return source;
}

TEST(RuntimeTensorDescTest, InfersWholeDynamicMatmulGraphFromInvocationInputs) {
    kir::Graph graph;
    auto lhs = graph.addValue(tensorType({-1, 4}), "lhs");
    auto rhs = graph.addValue(tensorType({4, 8}), "rhs");
    auto output = graph.addValue(tensorType({-1, 8}), "output");
    graph.addOp<kir::InputOp>(argument(0), lhs);
    graph.addOp<kir::InputOp>(argument(1), rhs);
    graph.addOp<kir::MatMulKernelOp>(lhs, rhs, output, false, false);
    graph.setOutputs({output});

    sandy::engine::RuntimeTensorDescs inputs(graph.values().size());
    ASSERT_TRUE(inputs.set(lhs, {{3, 4}, sandy::core::DType::F32}));
    ASSERT_TRUE(inputs.set(rhs, {{4, 8}, sandy::core::DType::F32}));

    auto inferred = sandy::engine::inferRuntimeTensorDescs(graph, std::move(inputs));
    ASSERT_TRUE(inferred) << inferred.error();
    EXPECT_EQ(inferred->get(output).shape, sandy::core::Shape({3, 8}));
}

TEST(RuntimeTensorDescTest, SimulatesPagedAppendBeforeExecution) {
    kir::Graph graph;
    auto cacheType = tensorType({1, 2, -1, 4});
    cacheType.kind = kir::ValueKind::PagedTensor;
    cacheType.paged = {.growDim = 2, .pageSize = 32};
    auto cache = graph.addValue(cacheType, "cache");
    auto chunk = graph.addValue(tensorType({1, 2, -1, 4}), "chunk");
    graph.addOp<kir::InputOp>(argument(0), cache);
    graph.addOp<kir::InputOp>(argument(1), chunk);
    graph.addOp<kir::PagedAppendOp>(cache, chunk);

    sandy::engine::RuntimeTensorDescs inputs(graph.values().size());
    ASSERT_TRUE(inputs.set(cache, {{1, 2, 7, 4}, sandy::core::DType::F32}));
    ASSERT_TRUE(inputs.set(chunk, {{1, 2, 5, 4}, sandy::core::DType::F32}));

    auto inferred = sandy::engine::inferRuntimeTensorDescs(graph, std::move(inputs));
    ASSERT_TRUE(inferred) << inferred.error();
    EXPECT_EQ(inferred->get(cache).shape, sandy::core::Shape({1, 2, 12, 4}));
}

TEST(RuntimeTensorDescTest, InfersDynamicNegativeIndexSlice) {
    kir::Graph graph;
    auto input = graph.addValue(tensorType({1, -1, 4}), "input");
    auto output = graph.addValue(tensorType({1, 4}), "output");
    graph.addOp<kir::InputOp>(argument(0), input);
    graph.addOp<kir::LayoutTransformOp>(
        kir::LayoutTransformKind::Slice,
        input,
        output,
        std::vector<int64_t>{0, 1, 0},
        std::vector<int64_t>{0, -1, 0});
    graph.setOutputs({output});

    sandy::engine::RuntimeTensorDescs inputs(graph.values().size());
    ASSERT_TRUE(inputs.set(input, {{1, 7, 4}, sandy::core::DType::F32}));

    auto inferred = sandy::engine::inferRuntimeTensorDescs(graph, std::move(inputs));
    ASSERT_TRUE(inferred) << inferred.error();
    EXPECT_EQ(inferred->get(output).shape, sandy::core::Shape({1, 4}));
}

TEST(RuntimeTensorDescTest, RejectsConcreteInputThatViolatesKernelIRContract) {
    kir::Graph graph;
    auto input = graph.addValue(tensorType({-1, 4}), "input");
    graph.addOp<kir::InputOp>(argument(0), input);

    sandy::engine::RuntimeTensorDescs inputs(graph.values().size());
    ASSERT_TRUE(inputs.set(input, {{3, 5}, sandy::core::DType::F32}));

    auto inferred = sandy::engine::inferRuntimeTensorDescs(graph, std::move(inputs));
    ASSERT_FALSE(inferred);
    EXPECT_NE(inferred.error().find("dimension 1 mismatch"), std::string::npos);
}

} // namespace
