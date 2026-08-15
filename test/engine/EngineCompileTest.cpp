#include "Device.h"
#include "Engine.h"
#include "MidIR.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeTensorBuffer final : public sandy::core::TensorBuffer {
public:
    explicit FakeTensorBuffer(sandy::core::TensorDesc desc)
        : TensorBuffer(std::move(desc)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return {}; }
};

class FakeDevice final : public sandy::engine::Device {
public:
    Result<sandy::engine::DeviceProgramId> compile(const sandy::ir::mid_ir::Op& op) override {
        compiledOps.push_back(&op);
        if (failCompile)
            return make_error("fake compile failed");
        return nextProgramId++;
    }

    Result<sandy::engine::DeviceBufferId> alloc(sandy::core::TensorDesc desc) override {
        allocDescs.push_back(std::move(desc));
        return nextBufferId++;
    }

    Result<void> dealloc(sandy::engine::DeviceBufferId buffer) override {
        deallocs.push_back(buffer);
        return {};
    }

    Result<sandy::engine::DeviceBufferId> load(sandy::core::TensorBuffer& src) override {
        loads.push_back(src.desc());
        return nextBufferId++;
    }

    Result<void> run(
            sandy::engine::DeviceProgramId program,
            std::span<const sandy::engine::DeviceBufferId> inputs,
            std::span<const sandy::engine::DeviceBufferId> outputs) override {
        runs.push_back({program, std::vector<sandy::engine::DeviceBufferId>(inputs.begin(), inputs.end()),
                        std::vector<sandy::engine::DeviceBufferId>(outputs.begin(), outputs.end())});
        return {};
    }

    Result<sandy::engine::TensorBufferPtr> read(sandy::engine::DeviceBufferId src) override {
        reads.push_back(src);
        sandy::engine::TensorBufferPtr buffer = std::make_shared<FakeTensorBuffer>(
            sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::F32));
        return buffer;
    }

    struct RunCall {
        sandy::engine::DeviceProgramId program = 0;
        std::vector<sandy::engine::DeviceBufferId> inputs;
        std::vector<sandy::engine::DeviceBufferId> outputs;
    };

    bool failCompile = false;
    sandy::engine::DeviceProgramId nextProgramId = 100;
    sandy::engine::DeviceBufferId nextBufferId = 200;
    std::vector<const sandy::ir::mid_ir::Op*> compiledOps;
    std::vector<sandy::core::TensorDesc> allocDescs;
    std::vector<sandy::engine::DeviceBufferId> deallocs;
    std::vector<sandy::core::TensorDesc> loads;
    std::vector<RunCall> runs;
    std::vector<sandy::engine::DeviceBufferId> reads;
};

class EngineCompileTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sandy::ir::mid_ir::register_all_ops();
    }
};

} // namespace

TEST_F(EngineCompileTest, CompileStoresDeviceProgramsInInvocPlan) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* reshaped = builder.createReshape(x, {2});
    auto* tanh = builder.createTanh(reshaped);
    auto* out = builder.createReLU(tanh);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto fake = std::make_unique<FakeDevice>();
    auto* fakePtr = fake.get();
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    ASSERT_EQ(fakePtr->compiledOps.size(), 2u);
    EXPECT_EQ(fakePtr->compiledOps[0], tanh->def);
    EXPECT_EQ(fakePtr->compiledOps[1], out->def);

    ASSERT_EQ(plan->programs.size(), 2u);
    EXPECT_EQ(plan->programs[0].id, 0u);
    EXPECT_EQ(plan->programs[0].device, 0u);
    EXPECT_EQ(plan->programs[0].deviceProgram, 100u);
    EXPECT_EQ(plan->programs[1].id, 1u);
    EXPECT_EQ(plan->programs[1].device, 0u);
    EXPECT_EQ(plan->programs[1].deviceProgram, 101u);

    EXPECT_EQ(plan->outputs, std::vector<sandy::engine::InvocValueId>({2}));
    ASSERT_FALSE(plan->instructions.empty());
    EXPECT_EQ(plan->instructions.back().kind, sandy::engine::InvocInstructionKind::StoreOutputs);
}

TEST_F(EngineCompileTest, CompileFailsWithoutDevices) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    builder.setOutputs(std::span<sandy::ir::mid_ir::Value* const>(&x, 1));

    sandy::engine::Engine engine(std::vector<std::unique_ptr<sandy::engine::Device>>{});

    auto planResult = engine.compile(graph);
    EXPECT_FALSE(planResult);
    EXPECT_NE(planResult.error().find("no devices"), std::string::npos);
}

TEST_F(EngineCompileTest, CompilePropagatesDeviceCompileFailure) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* out = builder.createTanh(x);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto fake = std::make_unique<FakeDevice>();
    fake->failCompile = true;
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    auto planResult = engine.compile(graph);
    EXPECT_FALSE(planResult);
    EXPECT_EQ(planResult.error(), "fake compile failed");
}

TEST_F(EngineCompileTest, RunInterpretsInvocPlanWithFakeDevice) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* out = builder.createAdd(x, w);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto fake = std::make_unique<FakeDevice>();
    auto* fakePtr = fake.get();
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));

    sandy::engine::TensorMap weights;
    weights["w"] = std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("w", sandy::core::Shape({1}), sandy::core::DType::F32));

    auto outputsResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    auto runOutputs = outputsResult.take();

    ASSERT_EQ(runOutputs.size(), 1u);
    EXPECT_NE(runOutputs[0], nullptr);

    ASSERT_EQ(fakePtr->loads.size(), 2u);
    EXPECT_EQ(fakePtr->loads[0].name, "x");
    EXPECT_EQ(fakePtr->loads[1].name, "w");

    ASSERT_EQ(fakePtr->allocDescs.size(), 1u);
    EXPECT_EQ(fakePtr->allocDescs[0].shape, sandy::core::Shape({1}));

    ASSERT_EQ(fakePtr->runs.size(), 1u);
    EXPECT_EQ(fakePtr->runs[0].program, 100u);
    EXPECT_EQ(fakePtr->runs[0].inputs, std::vector<sandy::engine::DeviceBufferId>({200, 201}));
    EXPECT_EQ(fakePtr->runs[0].outputs, std::vector<sandy::engine::DeviceBufferId>({202}));

    EXPECT_EQ(fakePtr->reads, std::vector<sandy::engine::DeviceBufferId>({202}));
    EXPECT_EQ(fakePtr->deallocs, std::vector<sandy::engine::DeviceBufferId>({200, 201, 202}));
}

TEST_F(EngineCompileTest, RunUsesStoreOutputsOrder) {
    auto fake = std::make_unique<FakeDevice>();
    auto* fakePtr = fake.get();
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    sandy::engine::InvocPlan plan;
    plan.instructions.push_back(sandy::engine::InvocInstruction::alloc({
        0,
        0,
        sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::F32),
    }));
    plan.instructions.push_back(sandy::engine::InvocInstruction::alloc({
        0,
        1,
        sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::F32),
    }));
    plan.instructions.push_back(sandy::engine::InvocInstruction::store_outputs({0, {1, 0}, {}}));
    plan.outputs = {0, 1};

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    sandy::engine::TensorMap weights;

    auto result = engine.run(plan, inputs, weights);
    ASSERT_TRUE(result) << result.error();
    EXPECT_EQ(result->size(), 2u);
    EXPECT_EQ(fakePtr->reads, std::vector<sandy::engine::DeviceBufferId>({201, 200}));
    EXPECT_EQ(fakePtr->deallocs, std::vector<sandy::engine::DeviceBufferId>({201, 200}));
}

TEST_F(EngineCompileTest, RunFailsForMissingInputIndex) {
    auto fake = std::make_unique<FakeDevice>();
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    sandy::engine::InvocPlan plan;
    plan.instructions.push_back(sandy::engine::InvocInstruction::load_input({0, 1, 0}));
    plan.instructions.push_back(sandy::engine::InvocInstruction::store_outputs({0, {0}, {}}));
    plan.outputs = {0};

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    auto result = engine.run(plan, inputs, weights);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("input index out of range"), std::string::npos);
}

TEST_F(EngineCompileTest, RunFailsForMissingWeight) {
    auto fake = std::make_unique<FakeDevice>();
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    sandy::engine::InvocPlan plan;
    plan.instructions.push_back(sandy::engine::InvocInstruction::load_weight({0, "w", 0}));
    plan.instructions.push_back(sandy::engine::InvocInstruction::store_outputs({0, {0}, {}}));
    plan.outputs = {0};

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    sandy::engine::TensorMap weights;

    auto result = engine.run(plan, inputs, weights);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("missing weight buffer: w"), std::string::npos);
}

TEST_F(EngineCompileTest, RunFailsForMissingProgram) {
    auto fake = std::make_unique<FakeDevice>();
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    sandy::engine::InvocPlan plan;
    plan.instructions.push_back(sandy::engine::InvocInstruction::alloc({
        0,
        0,
        sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::F32),
    }));
    plan.instructions.push_back(sandy::engine::InvocInstruction::run_kernel({
        0,
        42,
        {},
        {0},
    }));
    plan.instructions.push_back(sandy::engine::InvocInstruction::store_outputs({0, {0}, {}}));
    plan.outputs = {0};

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    sandy::engine::TensorMap weights;

    auto result = engine.run(plan, inputs, weights);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("missing program: 42"), std::string::npos);
}
