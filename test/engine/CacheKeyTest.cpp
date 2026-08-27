#include "CacheKey.h"
#include "InvocationCacheKey.h"

#include <gtest/gtest.h>

namespace {

namespace kir = sandy::ir::kernel_ir;

TEST(CacheKeyTest, CanonicalEncodingIncludesDomainShapeAndDtype) {
    auto make = [](std::string_view domain, sandy::core::DType dtype, int64_t length) {
        sandy::core::CacheKeyBuilder key(domain);
        key.addTensorDesc(sandy::core::TensorDesc({1, length}, dtype));
        return std::move(key).finish();
    };

    EXPECT_EQ(make("scratch", sandy::core::DType::BF16, 16),
              make("scratch", sandy::core::DType::BF16, 16));
    EXPECT_NE(make("scratch", sandy::core::DType::BF16, 16),
              make("jit", sandy::core::DType::BF16, 16));
    EXPECT_NE(make("scratch", sandy::core::DType::BF16, 16),
              make("scratch", sandy::core::DType::F32, 16));
    EXPECT_NE(make("scratch", sandy::core::DType::BF16, 16),
              make("scratch", sandy::core::DType::BF16, 17));
}

TEST(CacheKeyTest, InvocationKeyIncludesProgramAndPagedLength) {
    kir::Graph graph;
    kir::ValueType pagedType;
    pagedType.kind = kir::ValueKind::PagedTensor;
    pagedType.dtype = sandy::core::DType::BF16;
    pagedType.shape = sandy::core::Shape({1, 2, sandy::core::Shape::kDynamic, 64});
    pagedType.paged = {2, 32};
    auto cache = graph.addValue(pagedType);
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, "", 3},
        cache);

    auto inputDescs = [&](int64_t length) {
        sandy::engine::RuntimeTensorDescs descs(graph.values().size());
        EXPECT_TRUE(descs.set(
            cache,
            sandy::core::TensorDesc({1, 2, length, 64}, sandy::core::DType::BF16)));
        return descs;
    };

    auto a = sandy::engine::buildInvocationCacheKey(
        "scratch-layout-v1", 7, graph, inputDescs(1024));
    auto b = sandy::engine::buildInvocationCacheKey(
        "scratch-layout-v1", 7, graph, inputDescs(1024));
    auto differentLength = sandy::engine::buildInvocationCacheKey(
        "scratch-layout-v1", 7, graph, inputDescs(1025));
    auto differentProgram = sandy::engine::buildInvocationCacheKey(
        "scratch-layout-v1", 8, graph, inputDescs(1024));

    ASSERT_TRUE(a) << a.error();
    ASSERT_TRUE(b) << b.error();
    ASSERT_TRUE(differentLength) << differentLength.error();
    ASSERT_TRUE(differentProgram) << differentProgram.error();
    EXPECT_EQ(*a, *b);
    EXPECT_NE(*a, *differentLength);
    EXPECT_NE(*a, *differentProgram);
}

} // namespace

