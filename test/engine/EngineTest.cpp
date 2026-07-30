#include "CpuInterpreterBackend.h"
#include "Engine.h"
#include "MidIR.h"
#include "TensorCalc.h"
#include "TensorBuffer.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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
