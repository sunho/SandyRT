#include "CpuInterpreterBackend.h"
#include "Engine.h"
#include "MidIR.h"
#include "ShapeUtil.h"
#include "TensorCalc.h"
#include "TensorBuffer.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

class TestTensorBuffer final : public sandy::core::TensorBuffer {
public:
    TestTensorBuffer(sandy::core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

    int loadCount = 0;
    int unloadCount = 0;

private:
    Result<void> load() override {
        loadCount++;
        return {};
    }

    void unload() override {
        unloadCount++;
    }

    std::span<const uint8_t> data() const override {
        if (!is_mounted()) {
            fprintf(stderr, "TestTensorBuffer::data() called while unmounted\n");
            abort();
        }
        return data_;
    }

    std::vector<uint8_t> data_;
};

std::shared_ptr<TestTensorBuffer> make_buffer(const std::string& name) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, sandy::core::Shape({1}), sandy::core::DType::F32),
        std::vector<uint8_t>{0, 0, 0, 0});
}

std::vector<uint8_t> f32_bytes(std::initializer_list<float> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    size_t index = 0;
    for (float value : values) {
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
        index++;
    }
    return bytes;
}

std::vector<uint8_t> i32_bytes(std::initializer_list<int32_t> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(int32_t));
    size_t index = 0;
    for (int32_t value : values) {
        std::memcpy(bytes.data() + index * sizeof(int32_t), &value, sizeof(int32_t));
        index++;
    }
    return bytes;
}

std::vector<uint8_t> i64_bytes(std::initializer_list<int64_t> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    size_t index = 0;
    for (int64_t value : values) {
        std::memcpy(bytes.data() + index * sizeof(int64_t), &value, sizeof(int64_t));
        index++;
    }
    return bytes;
}

float read_f32(std::span<const uint8_t> bytes, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
    return value;
}

std::shared_ptr<TestTensorBuffer> make_f32_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        std::initializer_list<float> values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::F32),
        f32_bytes(values));
}

std::shared_ptr<TestTensorBuffer> make_i32_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        std::initializer_list<int32_t> values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::I32),
        i32_bytes(values));
}

} // namespace

TEST(TensorBufferTest, MountsAreStacked) {
    auto buffer = make_buffer("x");

    auto first = buffer->mount();
    ASSERT_TRUE(first) << first.error();
    EXPECT_EQ(buffer->loadCount, 1);
    EXPECT_EQ(buffer->unloadCount, 0);

    auto second = buffer->mount();
    ASSERT_TRUE(second) << second.error();
    EXPECT_EQ(buffer->loadCount, 1);

    buffer->unmount();
    EXPECT_EQ(buffer->unloadCount, 0);

    buffer->unmount();
    EXPECT_EQ(buffer->unloadCount, 1);
}

TEST(TensorBufferTest, AccessMountsAndUnmounts) {
    auto buffer = make_buffer("x");

    {
        auto accessResult = buffer->access();
        ASSERT_TRUE(accessResult) << accessResult.error();
        auto access = accessResult.take();
        EXPECT_EQ(buffer->loadCount, 1);
        EXPECT_EQ(access.desc().name, "x");
        EXPECT_EQ(access.data().size(), 4u);
    }

    EXPECT_EQ(buffer->unloadCount, 1);
}

TEST(EngineTest, CreatePlanAndRunWithDummyCpuInterpreter) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({1, 1}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({1, 1}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer("x", sandy::core::Shape({1, 1}), {1.0f});

    sandy::engine::TensorMap weights;
    weights["w"] = make_f32_buffer("w", sandy::core::Shape({1, 1}), {2.0f});
    weights["b"] = make_f32_buffer("b", sandy::core::Shape({1}), {3.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    EXPECT_TRUE(runResult) << runResult.error();
}

TEST(TensorCalcTest, LinearF32) {
    auto x = f32_bytes({1.0f, 2.0f});
    auto w = f32_bytes({3.0f, 4.0f, 5.0f, 6.0f});
    auto b = f32_bytes({7.0f, 8.0f});

    auto result = sandy::core::linear_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({1, 2}), sandy::core::DType::F32),
        w, sandy::core::TensorDesc(sandy::core::Shape({2, 2}), sandy::core::DType::F32),
        b, sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 18.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 25.0f);
}

TEST(TensorCalcTest, ReLUF32) {
    auto x = f32_bytes({-2.0f, 0.5f, 3.0f});

    auto result = sandy::core::relu_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({1, 3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 0.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 0.5f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 3.0f);
}

TEST(ShapeUtilTest, BroadcastShapeRightAligned) {
    auto result = sandy::core::broadcast_shape(
        sandy::core::Shape({2, 3, 4}),
        sandy::core::Shape({3, 1}));

    ASSERT_TRUE(result) << result.error();
    EXPECT_EQ(result.take(), sandy::core::Shape({2, 3, 4}));
}

TEST(TensorCalcTest, AddF32BroadcastsRightAligned) {
    auto lhs = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto rhs = f32_bytes({10.0f, 20.0f, 30.0f});

    auto result = sandy::core::add_f32(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 3}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 11.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 22.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 33.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 14.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 25.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 36.0f);
}

TEST(TensorCalcTest, MulF32BroadcastsMiddleDim) {
    auto lhs = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto rhs = f32_bytes({10.0f, 20.0f});

    auto result = sandy::core::mul_f32(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({2, 1, 3}), sandy::core::DType::F32),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({2, 1, 1}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 1, 3}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 10.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 20.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 30.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 80.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 100.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 120.0f);
}

TEST(TensorCalcTest, SqrtF32) {
    auto x = f32_bytes({1.0f, 4.0f, 9.0f, 16.0f});

    auto result = sandy::core::sqrt_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 4.0f);
}

TEST(TensorCalcTest, TanhF32) {
    auto x = f32_bytes({-1.0f, 0.0f, 1.0f});

    auto result = sandy::core::tanh_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({3}));
    EXPECT_NEAR(read_f32(out.data, 0), std::tanh(-1.0f), 1.0e-6f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 0.0f);
    EXPECT_NEAR(read_f32(out.data, 2), std::tanh(1.0f), 1.0e-6f);
}

TEST(TensorCalcTest, MatMulF32UsesTorchLayout) {
    auto lhs = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto rhs = f32_bytes({
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,
    });

    auto result = sandy::core::matmul_f32(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({3, 2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 58.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 64.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 139.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 154.0f);
}

TEST(TensorCalcTest, MatMulF32BroadcastsBatchDims) {
    auto lhs = f32_bytes({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    });
    auto rhs = f32_bytes({
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f,
    });

    auto result = sandy::core::matmul_f32(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({2, 2, 2}), sandy::core::DType::F32),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({1, 2, 3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2, 3}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 90.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 120.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 150.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 190.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 260.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 330.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 6), 290.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 7), 400.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 8), 510.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 9), 390.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 10), 540.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 11), 690.0f);
}

TEST(TensorCalcTest, TransposeF32Requires2D) {
    auto x = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });

    auto result = sandy::core::transpose_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({3, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 6.0f);

    auto rank3 = sandy::core::transpose_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({1, 2, 3}), sandy::core::DType::F32));
    EXPECT_FALSE(rank3);
    EXPECT_NE(rank3.error().find("rank 2"), std::string::npos);
}

TEST(TensorCalcTest, ReshapeF32ChangesShapeOnly) {
    auto x = f32_bytes({
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
    });

    auto result = sandy::core::reshape_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 6}), sandy::core::DType::F32),
        sandy::core::Shape({2, 3, 2}));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 3, 2}));
    for (size_t i = 0; i < 12; i++)
        EXPECT_FLOAT_EQ(read_f32(out.data, i), static_cast<float>(i));
}

TEST(TensorCalcTest, ReshapeF32InfersNegativeOneDimension) {
    auto x = f32_bytes({
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
    });

    auto result = sandy::core::reshape_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 6}), sandy::core::DType::F32),
        sandy::core::Shape({-1, 3, 2}));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 3, 2}));
    for (size_t i = 0; i < 12; i++)
        EXPECT_FLOAT_EQ(read_f32(out.data, i), static_cast<float>(i));
}

TEST(TensorCalcTest, PermuteF32ReordersAxes) {
    auto x = f32_bytes({
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    });
    std::vector<int64_t> dims = {0, 2, 1};

    auto result = sandy::core::permute_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32),
        dims);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 4, 3}));

    std::vector<float> expected = {
        0.0f, 4.0f, 8.0f,
        1.0f, 5.0f, 9.0f,
        2.0f, 6.0f, 10.0f,
        3.0f, 7.0f, 11.0f,
        12.0f, 16.0f, 20.0f,
        13.0f, 17.0f, 21.0f,
        14.0f, 18.0f, 22.0f,
        15.0f, 19.0f, 23.0f,
    };
    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_FLOAT_EQ(read_f32(out.data, i), expected[i]);
}

TEST(TensorCalcTest, SlidingQueryKeyScoreF32CausalUnbatched) {
    auto q = f32_bytes({
        1.0f, 0.0f,
        0.0f, 2.0f,
        1.0f, 1.0f,
    });
    auto k = f32_bytes({
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    });

    auto result = sandy::core::sliding_query_key_score_f32(
        q, sandy::core::TensorDesc(sandy::core::Shape({1, 3, 2}), sandy::core::DType::F32),
        k, sandy::core::TensorDesc(sandy::core::Shape({1, 3, 2}), sandy::core::DType::F32),
        0);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 3, 3}));

    float scale = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(read_f32(out.data, 0), 1.0f * scale, 1.0e-6f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 1)) && read_f32(out.data, 1) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 2)) && read_f32(out.data, 2) < 0.0f);
    EXPECT_NEAR(read_f32(out.data, 3), 0.0f, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 4), 2.0f * scale, 1.0e-6f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 5)) && read_f32(out.data, 5) < 0.0f);
    EXPECT_NEAR(read_f32(out.data, 6), 1.0f * scale, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 7), 1.0f * scale, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 8), 2.0f * scale, 1.0e-6f);
}

TEST(TensorCalcTest, SlidingQueryKeyScoreF32AppliesSlidingWindow) {
    auto q = f32_bytes({
        1.0f,
        2.0f,
        3.0f,
        4.0f,
    });
    auto k = f32_bytes({
        10.0f,
        20.0f,
        30.0f,
        40.0f,
    });

    auto result = sandy::core::sliding_query_key_score_f32(
        q, sandy::core::TensorDesc(sandy::core::Shape({1, 4, 1}), sandy::core::DType::F32),
        k, sandy::core::TensorDesc(sandy::core::Shape({1, 4, 1}), sandy::core::DType::F32),
        2);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 4, 4}));

    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 10.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 1)) && read_f32(out.data, 1) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 2)) && read_f32(out.data, 2) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 3)) && read_f32(out.data, 3) < 0.0f);

    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 20.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 40.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 6)) && read_f32(out.data, 6) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 7)) && read_f32(out.data, 7) < 0.0f);

    EXPECT_TRUE(std::isinf(read_f32(out.data, 8)) && read_f32(out.data, 8) < 0.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 9), 60.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 10), 90.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 11)) && read_f32(out.data, 11) < 0.0f);

    EXPECT_TRUE(std::isinf(read_f32(out.data, 12)) && read_f32(out.data, 12) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 13)) && read_f32(out.data, 13) < 0.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 14), 120.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 15), 160.0f);
}

TEST(TensorCalcTest, SlidingQueryKeyScoreF32GroupedQueryBatched) {
    auto q = f32_bytes({
        1.0f, 2.0f,
        3.0f, 4.0f,
    });
    auto k = f32_bytes({
        10.0f,
        20.0f,
    });

    auto result = sandy::core::sliding_query_key_score_f32(
        q, sandy::core::TensorDesc(sandy::core::Shape({1, 2, 1, 2}), sandy::core::DType::F32),
        k, sandy::core::TensorDesc(sandy::core::Shape({1, 1, 2, 1}), sandy::core::DType::F32),
        0);

    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("head dimension mismatch"), std::string::npos);

    auto validK = f32_bytes({
        10.0f, 1.0f,
        20.0f, 2.0f,
    });
    result = sandy::core::sliding_query_key_score_f32(
        q, sandy::core::TensorDesc(sandy::core::Shape({1, 2, 1, 2}), sandy::core::DType::F32),
        validK, sandy::core::TensorDesc(sandy::core::Shape({1, 1, 2, 2}), sandy::core::DType::F32),
        0);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 2, 1, 2}));

    float scale = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(read_f32(out.data, 0), 12.0f * scale, 1.0e-6f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 1)) && read_f32(out.data, 1) < 0.0f);
    EXPECT_NEAR(read_f32(out.data, 2), 34.0f * scale, 1.0e-6f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 3)) && read_f32(out.data, 3) < 0.0f);
}

TEST(TensorCalcTest, SoftmaxF32LastDim) {
    auto x = f32_bytes({
        1.0f, 1.0f,
        1.0f, 2.0f,
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    });

    auto result = sandy::core::softmax_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({3, 2}), sandy::core::DType::F32),
        -1);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({3, 2}));

    float inv = 1.0f / (1.0f + std::exp(1.0f));
    EXPECT_NEAR(read_f32(out.data, 0), 0.5f, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 1), 0.5f, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 2), inv, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 3), 1.0f - inv, 1.0e-6f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 0.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 0.0f);
}

TEST(TensorCalcTest, SoftmaxF32MiddleDim) {
    auto x = f32_bytes({
        1.0f, 10.0f,
        1.0f, 20.0f,
        1.0f, 30.0f,
    });

    auto result = sandy::core::softmax_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({1, 3, 2}), sandy::core::DType::F32),
        1);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 3, 2}));

    EXPECT_NEAR(read_f32(out.data, 0), 1.0f / 3.0f, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 2), 1.0f / 3.0f, 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 4), 1.0f / 3.0f, 1.0e-6f);

    double e0 = std::exp(-20.0);
    double e1 = std::exp(-10.0);
    double sum = e0 + e1 + 1.0;
    EXPECT_NEAR(read_f32(out.data, 1), static_cast<float>(e0 / sum), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 3), static_cast<float>(e1 / sum), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 5), static_cast<float>(1.0 / sum), 1.0e-6f);
}

TEST(TensorCalcTest, EmbeddingF32WithI32Ids) {
    auto ids = i32_bytes({2, 0, 3});
    auto weight = f32_bytes({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    });

    auto result = sandy::core::embedding_f32(
        ids, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::I32),
        weight, sandy::core::TensorDesc(sandy::core::Shape({4, 2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({3, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 6.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 7.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 8.0f);
}

TEST(TensorCalcTest, EmbeddingF32WithI64IdsKeepsLeadingDims) {
    auto ids = i64_bytes({1, 3, 0, 2});
    auto weight = f32_bytes({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    });

    auto result = sandy::core::embedding_f32(
        ids, sandy::core::TensorDesc(sandy::core::Shape({2, 2}), sandy::core::DType::I64),
        weight, sandy::core::TensorDesc(sandy::core::Shape({4, 2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 7.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 8.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 4), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 5), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 6), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 7), 6.0f);
}

TEST(TensorCalcTest, RMSNormF32) {
    auto x = f32_bytes({1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f});
    auto weight = f32_bytes({1.0f, 10.0f, -1.0f});
    constexpr float eps = 1.0e-6f;

    auto result = sandy::core::rms_norm_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32),
        weight, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::F32),
        eps);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 3}));

    float row0Scale = 1.0f / std::sqrt(3.0f + eps);
    float row1Scale = 1.0f / std::sqrt((25.0f / 3.0f) + eps);
    EXPECT_NEAR(read_f32(out.data, 0), 1.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 1), 20.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 2), -2.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 4), 30.0f * row1Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 5), -4.0f * row1Scale, 1.0e-5f);
}

TEST(TensorCalcTest, RMSNormF32Rank3NormalizesLastDim) {
    auto x = f32_bytes({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    });
    auto weight = f32_bytes({2.0f, -1.0f});
    constexpr float eps = 1.0e-6f;

    auto result = sandy::core::rms_norm_f32(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 2, 2}), sandy::core::DType::F32),
        weight, sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32),
        eps);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2, 2}));

    for (size_t row = 0; row < 4; row++) {
        float a = static_cast<float>(row * 2 + 1);
        float b = static_cast<float>(row * 2 + 2);
        float scale = 1.0f / std::sqrt(((a * a) + (b * b)) / 2.0f + eps);
        EXPECT_NEAR(read_f32(out.data, row * 2), a * scale * 2.0f, 1.0e-5f);
        EXPECT_NEAR(read_f32(out.data, row * 2 + 1), b * scale * -1.0f, 1.0e-5f);
    }
}

TEST(CpuInterpretTest, EngineRunReturnsOutput0) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({2, 2}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer("x", sandy::core::Shape({1, 2}), {1.0f, 2.0f});

    sandy::engine::TensorMap weights;
    weights["w"] = make_f32_buffer("w", sandy::core::Shape({2, 2}), {3.0f, 4.0f, 5.0f, 6.0f});
    weights["b"] = make_f32_buffer("b", sandy::core::Shape({2}), {7.0f, 8.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 2}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 18.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 25.0f);
}

TEST(CpuInterpretTest, RMSNorm) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("norm.weight", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createRMSNorm(x, weight);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f});

    sandy::engine::TensorMap weights;
    weights["norm.weight"] = make_f32_buffer(
        "norm.weight", sandy::core::Shape({3}), {1.0f, 10.0f, -1.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({2, 3}));

    constexpr float eps = 1.0e-6f;
    float row0Scale = 1.0f / std::sqrt(3.0f + eps);
    float row1Scale = 1.0f / std::sqrt((25.0f / 3.0f) + eps);
    EXPECT_NEAR(read_f32(it->second->data(), 0), 1.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 1), 20.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 2), -2.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 4), 30.0f * row1Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 5), -4.0f * row1Scale, 1.0e-5f);
}

TEST(CpuInterpretTest, AddMulSqrt) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("bias", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* scale = builder.createWeight("scale", sandy::core::Shape({2, 1}), sandy::core::DType::F32);
    auto* y = builder.createAdd(x, bias);
    auto* z = builder.createMul(y, scale);
    auto* out = builder.createSqrt(z);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f});

    sandy::engine::TensorMap weights;
    weights["bias"] = make_f32_buffer("bias", sandy::core::Shape({3}), {0.0f, 5.0f, 7.0f});
    weights["scale"] = make_f32_buffer("scale", sandy::core::Shape({2, 1}), {1.0f, 4.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({2, 3}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 2), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 3), 8.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 4), std::sqrt(120.0f));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 5), std::sqrt(172.0f));
}

TEST(CpuInterpretTest, ConstantBroadcastAndTanh) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* offset = builder.createConstantF32(1.0f);
    auto* shifted = builder.createAdd(x, offset);
    auto* out = builder.createTanh(shifted);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer("x", sandy::core::Shape({3}), {-2.0f, -1.0f, 0.0f});

    sandy::engine::TensorMap weights;

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({3}));
    EXPECT_NEAR(read_f32(it->second->data(), 0), std::tanh(-1.0f), 1.0e-6f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 0.0f);
    EXPECT_NEAR(read_f32(it->second->data(), 2), std::tanh(1.0f), 1.0e-6f);
}

TEST(CpuInterpretTest, GemmaStyleMatMulWithTransposedWeight) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("embed_tokens.weight", sandy::core::Shape({4, 3}), sandy::core::DType::F32);
    auto* weightT = builder.createTranspose(weight);
    auto* logits = builder.createMatMul(x, weightT);
    sandy::ir::mid_ir::Value* outputs[] = {logits};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    sandy::engine::TensorMap weights;
    weights["embed_tokens.weight"] = make_f32_buffer(
        "embed_tokens.weight", sandy::core::Shape({4, 3}), {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
        });

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({2, 4}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 3), 6.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 4), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 5), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 6), 6.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 7), 15.0f);
}

TEST(CpuInterpretTest, ReshapeProjectionLayout) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({1, 2, 6}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {1, 2, 3, 2});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer(
        "x", sandy::core::Shape({1, 2, 6}), {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        });

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 2, 3, 2}));
    for (size_t i = 0; i < 12; i++)
        EXPECT_FLOAT_EQ(read_f32(it->second->data(), i), static_cast<float>(i));
}

TEST(CpuInterpretTest, PermuteAttentionLayout) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput("x", sandy::core::Shape({1, 2, 3, 2}), sandy::core::DType::F32);
    auto* out = builder.createPermute(x, {0, 2, 1, 3});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["x"] = make_f32_buffer(
        "x", sandy::core::Shape({1, 2, 3, 2}), {
            0.0f, 1.0f,
            2.0f, 3.0f,
            4.0f, 5.0f,
            6.0f, 7.0f,
            8.0f, 9.0f,
            10.0f, 11.0f,
        });

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 3, 2, 2}));

    std::vector<float> expected = {
        0.0f, 1.0f, 6.0f, 7.0f,
        2.0f, 3.0f, 8.0f, 9.0f,
        4.0f, 5.0f, 10.0f, 11.0f,
    };
    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_FLOAT_EQ(read_f32(it->second->data(), i), expected[i]);
}

TEST(CpuInterpretTest, SlidingQueryKeyScore) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* q = builder.createInput("q", sandy::core::Shape({1, 1, 3, 1}), sandy::core::DType::F32);
    auto* k = builder.createInput("k", sandy::core::Shape({1, 1, 3, 1}), sandy::core::DType::F32);
    auto* out = builder.createSlidingQueryKeyScore(q, k, 2);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["q"] = make_f32_buffer("q", sandy::core::Shape({1, 1, 3, 1}), {
        1.0f, 2.0f, 3.0f,
    });
    inputs["k"] = make_f32_buffer("k", sandy::core::Shape({1, 1, 3, 1}), {
        10.0f, 20.0f, 30.0f,
    });

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 1, 3, 3}));

    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 10.0f);
    EXPECT_TRUE(std::isinf(read_f32(it->second->data(), 1)) &&
                read_f32(it->second->data(), 1) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(it->second->data(), 2)) &&
                read_f32(it->second->data(), 2) < 0.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 3), 20.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 4), 40.0f);
    EXPECT_TRUE(std::isinf(read_f32(it->second->data(), 5)) &&
                read_f32(it->second->data(), 5) < 0.0f);
    EXPECT_TRUE(std::isinf(read_f32(it->second->data(), 6)) &&
                read_f32(it->second->data(), 6) < 0.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 7), 60.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 8), 90.0f);
}

TEST(CpuInterpretTest, SoftmaxAfterSlidingQueryKeyScore) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* q = builder.createInput("q", sandy::core::Shape({1, 1, 2, 1}), sandy::core::DType::F32);
    auto* k = builder.createInput("k", sandy::core::Shape({1, 1, 2, 1}), sandy::core::DType::F32);
    auto* scores = builder.createSlidingQueryKeyScore(q, k);
    auto* out = builder.createSoftmax(scores, -1);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["q"] = make_f32_buffer("q", sandy::core::Shape({1, 1, 2, 1}), {
        1.0f, 1.0f,
    });
    inputs["k"] = make_f32_buffer("k", sandy::core::Shape({1, 1, 2, 1}), {
        1.0f, 2.0f,
    });

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 1, 2, 2}));

    float inv = 1.0f / (1.0f + std::exp(1.0f));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 0.0f);
    EXPECT_NEAR(read_f32(it->second->data(), 2), inv, 1.0e-6f);
    EXPECT_NEAR(read_f32(it->second->data(), 3), 1.0f - inv, 1.0e-6f);
}

TEST(CpuInterpretTest, EmbeddingLoadsRowsFromFullWeightBuffer) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* ids = builder.createInput("input_ids", sandy::core::Shape({2, 2}), sandy::core::DType::I32);
    auto* weight = builder.createWeight("embed_tokens.weight", sandy::core::Shape({4, 2}), sandy::core::DType::F32);
    auto* out = builder.createEmbedding(ids, weight);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());

    auto planResult = engine.create_plan(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    sandy::engine::TensorMap inputs;
    inputs["input_ids"] = make_i32_buffer(
        "input_ids", sandy::core::Shape({2, 2}), {3, 1, 0, 2});

    sandy::engine::TensorMap weights;
    weights["embed_tokens.weight"] = make_f32_buffer(
        "embed_tokens.weight", sandy::core::Shape({4, 2}), {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
        });

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = runResult.take();

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({2, 2, 2}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 7.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 8.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 3), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 4), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 5), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 6), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 7), 6.0f);
}
