#include "CpuInterpreterBackend.h"
#include "Engine.h"
#include "MidIR.h"
#include "TensorBuffer.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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
    auto* x = builder.createInput("x", sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({1}), sandy::core::DType::F32);
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
    inputs["x"] = make_buffer("x");

    sandy::engine::TensorMap weights;
    weights["w"] = make_buffer("w");
    weights["b"] = make_buffer("b");

    auto runResult = engine.run(*plan, inputs, weights);
    EXPECT_TRUE(runResult) << runResult.error();
}
