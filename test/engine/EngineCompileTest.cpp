#include "Device.h"
#include "Engine.h"
#include "KernelIR.h"
#include "MidIR.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void expect_same_buffers(
        const std::vector<sandy::device::DeviceBufferId>& actual,
        std::initializer_list<sandy::device::DeviceBufferId> expected) {
    EXPECT_EQ(actual.size(), expected.size());
    EXPECT_TRUE(std::is_permutation(
        actual.begin(), actual.end(), expected.begin(), expected.end()));
}

class FakeTensorBuffer final : public sandy::core::TensorBuffer {
public:
    explicit FakeTensorBuffer(sandy::core::TensorDesc desc)
        : TensorBuffer(std::move(desc)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return {}; }
};

class FakeDevice final : public sandy::device::Device {
public:
    Result<sandy::device::DeviceCompiledGraphId> compile(
            const sandy::ir::kernel_ir::Graph& graph) override {
        compiledOpKinds.clear();
        compiledOutputs = graph.outputs();
        for (const auto& op : graph.ops())
            compiledOpKinds.push_back(op->kind());
        if (failCompile)
            return make_error("fake compile failed");
        return nextGraphId++;
    }

    Result<sandy::device::DeviceBufferId> alloc(sandy::core::TensorDesc desc) override {
        allocDescs.push_back(desc);
        auto id = nextBufferId++;
        bufferDescs[id] = std::move(desc);
        return id;
    }

    Result<void> dealloc(sandy::device::DeviceBufferId buffer) override {
        deallocs.push_back(buffer);
        bufferDescs.erase(buffer);
        return {};
    }

    Result<sandy::device::DeviceBufferId> load(sandy::core::TensorBuffer& src) override {
        loads.push_back(src.desc());
        auto id = nextBufferId++;
        bufferDescs[id] = src.desc();
        return id;
    }

    Result<void> run(
            sandy::device::DeviceCompiledGraphId graph,
            sandy::ir::kernel_ir::OpId op,
            std::span<const sandy::device::DeviceRunValue> inputs,
            std::span<const sandy::device::DeviceRunValue> outputs) override {
        std::vector<sandy::device::DeviceBufferId> inputBuffers;
        inputBuffers.reserve(inputs.size());
        for (const auto& input : inputs)
            inputBuffers.push_back(std::get<sandy::device::DeviceTensorView>(input).buffer);
        std::vector<sandy::device::DeviceBufferId> outputBuffers;
        outputBuffers.reserve(outputs.size());
        for (const auto& output : outputs)
            outputBuffers.push_back(std::get<sandy::device::DeviceTensorView>(output).buffer);
        runs.push_back({
            graph,
            op,
            std::move(inputBuffers),
            std::move(outputBuffers),
        });
        return {};
    }

    Result<sandy::engine::TensorBufferPtr> read(
            sandy::device::DeviceTensorView src) override {
        reads.push_back(src.buffer);
        auto it = bufferDescs.find(src.buffer);
        if (it == bufferDescs.end())
            return make_error("fake buffer not found");
        sandy::engine::TensorBufferPtr buffer =
            std::make_shared<FakeTensorBuffer>(src.view.desc);
        return buffer;
    }

    struct RunCall {
        sandy::device::DeviceCompiledGraphId graph = 0;
        sandy::ir::kernel_ir::OpId op = 0;
        std::vector<sandy::device::DeviceBufferId> inputs;
        std::vector<sandy::device::DeviceBufferId> outputs;
    };

    bool failCompile = false;
    sandy::device::DeviceCompiledGraphId nextGraphId = 100;
    sandy::device::DeviceBufferId nextBufferId = 200;
    std::vector<sandy::ir::kernel_ir::OpKind> compiledOpKinds;
    std::vector<sandy::ir::kernel_ir::ValueId> compiledOutputs;
    std::vector<sandy::core::TensorDesc> allocDescs;
    std::vector<sandy::device::DeviceBufferId> deallocs;
    std::vector<sandy::core::TensorDesc> loads;
    std::vector<RunCall> runs;
    std::vector<sandy::device::DeviceBufferId> reads;
    std::unordered_map<sandy::device::DeviceBufferId, sandy::core::TensorDesc> bufferDescs;
};

class EngineCompileTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sandy::ir::mid_ir::register_all_ops();
    }
};

sandy::engine::Engine make_engine(FakeDevice** fakePtr) {
    auto fake = std::make_unique<FakeDevice>();
    *fakePtr = fake.get();
    std::vector<std::unique_ptr<sandy::device::Device>> devices;
    devices.push_back(std::move(fake));
    return sandy::engine::Engine(std::move(devices));
}

sandy::engine::Engine make_two_device_engine(FakeDevice** firstPtr, FakeDevice** secondPtr) {
    auto first = std::make_unique<FakeDevice>();
    auto second = std::make_unique<FakeDevice>();
    *firstPtr = first.get();
    *secondPtr = second.get();
    std::vector<std::unique_ptr<sandy::device::Device>> devices;
    devices.push_back(std::move(first));
    devices.push_back(std::move(second));
    return sandy::engine::Engine(std::move(devices));
}

} // namespace

TEST_F(EngineCompileTest, CompileLowersAndCompilesKernelGraph) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* reshaped = builder.createReshape(x, {2});
    auto* tanh = builder.createTanh(reshaped);
    auto* out = builder.createReLU(tanh);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);

    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    EXPECT_EQ(compiled->device, 0u);
    EXPECT_EQ(compiled->deviceGraph, 100u);
    EXPECT_EQ(fakePtr->compiledOpKinds, std::vector<sandy::ir::kernel_ir::OpKind>({
        sandy::ir::kernel_ir::OpKind::Input,
        sandy::ir::kernel_ir::OpKind::LayoutTransform,
        sandy::ir::kernel_ir::OpKind::ElementwiseKernel,
        sandy::ir::kernel_ir::OpKind::ElementwiseKernel,
    }));
    EXPECT_EQ(fakePtr->compiledOutputs, std::vector<sandy::ir::kernel_ir::ValueId>({3}));
    ASSERT_NE(compiled->graph, nullptr);
    EXPECT_EQ(compiled->graph->outputs(), std::vector<sandy::ir::kernel_ir::ValueId>({3}));
}

TEST_F(EngineCompileTest, CompileFailsWithoutDevices) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    builder.setOutputs(std::span<sandy::ir::mid_ir::Value* const>(&x, 1));

    sandy::engine::Engine engine(std::vector<std::unique_ptr<sandy::device::Device>>{});

    auto result = engine.compile(graph);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("no devices"), std::string::npos);
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
    std::vector<std::unique_ptr<sandy::device::Device>> devices;
    devices.push_back(std::move(fake));
    sandy::engine::Engine engine(std::move(devices));

    auto result = engine.compile(graph);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error(), "fake compile failed");
}

TEST_F(EngineCompileTest, RunExecutesKernelGraphWithFakeDevice) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* out = builder.createAdd(x, w);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));

    sandy::engine::TensorMap weights;
    weights["w"] = std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("w", sandy::core::Shape({1}), sandy::core::DType::F32));

    auto outputsResult = engine.run(*compiled, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);

    ASSERT_EQ(fakePtr->loads.size(), 2u);
    EXPECT_EQ(fakePtr->loads[0].name, "x");
    EXPECT_EQ(fakePtr->loads[1].name, "w");

    ASSERT_EQ(fakePtr->allocDescs.size(), 1u);
    EXPECT_EQ(fakePtr->allocDescs[0].shape, sandy::core::Shape({1}));

    ASSERT_EQ(fakePtr->runs.size(), 1u);
    EXPECT_EQ(fakePtr->runs[0].graph, 100u);
    EXPECT_EQ(fakePtr->runs[0].op, 2u);
    EXPECT_EQ(fakePtr->runs[0].inputs, std::vector<sandy::device::DeviceBufferId>({200, 201}));
    EXPECT_EQ(fakePtr->runs[0].outputs, std::vector<sandy::device::DeviceBufferId>({202}));

    EXPECT_EQ(fakePtr->reads, std::vector<sandy::device::DeviceBufferId>({202}));
    expect_same_buffers(fakePtr->deallocs, {200, 201, 202});
}

TEST_F(EngineCompileTest, RunEmbeddingWithRank1WeightAllocatesScalarLookupShape) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* ids = builder.createInput(0, sandy::core::Shape({1, 1, 8}), sandy::core::DType::I64);
    auto* weight = builder.createWeight("scale", sandy::core::Shape({128}), sandy::core::DType::BF16);
    auto* out = builder.createEmbedding(ids, weight);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("ids", sandy::core::Shape({1, 1, 8}), sandy::core::DType::I64)));

    sandy::engine::TensorMap weights;
    weights["scale"] = std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("scale", sandy::core::Shape({128}), sandy::core::DType::BF16));

    auto outputsResult = engine.run(*compiled, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);

    ASSERT_EQ(fakePtr->allocDescs.size(), 1u);
    EXPECT_EQ(fakePtr->allocDescs[0].shape, sandy::core::Shape({1, 1, 8}));
    EXPECT_EQ(fakePtr->allocDescs[0].dtype, sandy::core::DType::BF16);
    ASSERT_EQ(fakePtr->runs.size(), 1u);
    EXPECT_EQ(fakePtr->runs[0].outputs, std::vector<sandy::device::DeviceBufferId>({202}));
}

TEST_F(EngineCompileTest, RunValuesReturnsTensorTupleOutput) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* y = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    sandy::ir::mid_ir::Value* tupleElements[] = {x, y};
    auto* tuple = builder.createTensorTupleCreate(tupleElements);
    sandy::ir::mid_ir::Value* outputs[] = {tuple};
    builder.setOutputs(outputs);

    FakeDevice* fake = nullptr;
    auto engine = make_engine(&fake);
    auto compiled = engine.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    std::vector<sandy::engine::RunInput> inputs = {
        std::make_shared<FakeTensorBuffer>(
            sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::F32)),
        std::make_shared<FakeTensorBuffer>(
            sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32)),
    };
    auto outputsResult = engine.runValues(**compiled, inputs, sandy::device::TensorMap{});
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);
    auto* tupleOutput = std::get_if<sandy::engine::RunTensorTuple>(&(*outputsResult)[0]);
    ASSERT_NE(tupleOutput, nullptr);
    EXPECT_EQ(tupleOutput->elements.size(), 2u);
    EXPECT_EQ(fake->runs.size(), 0u);
    EXPECT_EQ(fake->loads.size(), 2u);
    EXPECT_EQ(fake->reads.size(), 2u);
}

TEST_F(EngineCompileTest, RunValuesKeepsTupleOutputElementsAliveUntilReadAndThenDeallocates) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* y = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* xOut = builder.createTanh(x);
    auto* yOut = builder.createTanh(y);
    sandy::ir::mid_ir::Value* tupleElements[] = {xOut, yOut};
    auto* tuple = builder.createTensorTupleCreate(tupleElements);
    sandy::ir::mid_ir::Value* outputs[] = {tuple};
    builder.setOutputs(outputs);

    FakeDevice* fake = nullptr;
    auto engine = make_engine(&fake);
    auto compiled = engine.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    std::vector<sandy::engine::RunInput> inputs = {
        std::make_shared<FakeTensorBuffer>(
            sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::F32)),
        std::make_shared<FakeTensorBuffer>(
            sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32)),
    };

    auto outputsResult = engine.runValues(**compiled, inputs, sandy::device::TensorMap{});
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);
    auto* tupleOutput = std::get_if<sandy::engine::RunTensorTuple>(&(*outputsResult)[0]);
    ASSERT_NE(tupleOutput, nullptr);
    ASSERT_EQ(tupleOutput->elements.size(), 2u);

    EXPECT_EQ(fake->reads,
              std::vector<sandy::device::DeviceBufferId>({202, 203}));
    expect_same_buffers(fake->deallocs, {200, 201, 202, 203});
    EXPECT_TRUE(fake->bufferDescs.empty());
}

TEST_F(EngineCompileTest, RunReshapeUsesDeviceTensorViewWithoutKernelLaunch) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {3, 2});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({2, 3}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    auto outputsResult = engine.run(*compiled, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);

    ASSERT_EQ(fakePtr->loads.size(), 1u);
    EXPECT_TRUE(fakePtr->allocDescs.empty());
    EXPECT_TRUE(fakePtr->runs.empty());
    EXPECT_EQ(fakePtr->reads, std::vector<sandy::device::DeviceBufferId>({200}));
    EXPECT_EQ(fakePtr->deallocs, std::vector<sandy::device::DeviceBufferId>({200}));
}

TEST_F(EngineCompileTest, RunPermuteUsesDeviceTensorViewWithoutKernelLaunch) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* out = builder.createPermute(x, {1, 0, 2});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    auto outputsResult = engine.run(*compiled, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);
    EXPECT_EQ((*outputsResult)[0]->desc().shape, sandy::core::Shape({3, 2, 4}));

    ASSERT_EQ(fakePtr->loads.size(), 1u);
    EXPECT_TRUE(fakePtr->allocDescs.empty());
    EXPECT_TRUE(fakePtr->runs.empty());
    EXPECT_EQ(fakePtr->reads, std::vector<sandy::device::DeviceBufferId>({200}));
    EXPECT_EQ(fakePtr->deallocs, std::vector<sandy::device::DeviceBufferId>({200}));
}

TEST_F(EngineCompileTest, RunSliceAliasesInputWithoutAllocationOrKernelLaunch) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(
        0, sandy::core::Shape({1, 7, 4}), sandy::core::DType::F32);
    auto* out = builder.createSlice(x, {0, 1, 0}, {0, -1, 0});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1, 7, 4}), sandy::core::DType::F32)));

    auto outputsResult = engine.run(
        *compiled, inputs, sandy::device::TensorMap{});
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);
    EXPECT_EQ((*outputsResult)[0]->desc().shape, sandy::core::Shape({1, 4}));
    EXPECT_TRUE(fakePtr->allocDescs.empty());
    EXPECT_TRUE(fakePtr->runs.empty());
    EXPECT_EQ(fakePtr->reads, std::vector<sandy::device::DeviceBufferId>({200}));
}

TEST_F(EngineCompileTest, RunExecutesDeviceTransferThroughHost) {
    using namespace sandy::ir::kernel_ir;

    auto graph = std::make_unique<Graph>();
    auto input = graph->addValue(
        ValueType{ValueKind::Tensor, sandy::core::DType::F32, sandy::core::Shape({1})},
        "x");
    graph->addOp<InputOp>(
        InputSource{InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph->addValue(
        ValueType{ValueKind::Tensor, sandy::core::DType::F32, sandy::core::Shape({1})},
        "x_on_device_1",
        1);
    graph->addOp<DeviceTransferOp>(0, 1, input, output);
    graph->setOutputs({output});

    auto verify = graph->verify();
    ASSERT_TRUE(verify) << verify.error();

    sandy::engine::CompiledKernelGraph compiled;
    compiled.graph = std::move(graph);
    compiled.device = 0;
    compiled.deviceGraph = 100;

    FakeDevice* first = nullptr;
    FakeDevice* second = nullptr;
    auto engine = make_two_device_engine(&first, &second);

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    auto outputsResult = engine.run(compiled, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);

    EXPECT_EQ(first->loads.size(), 1u);
    EXPECT_EQ(first->reads, std::vector<sandy::device::DeviceBufferId>({200}));
    EXPECT_EQ(first->deallocs, std::vector<sandy::device::DeviceBufferId>({200}));

    EXPECT_EQ(second->loads.size(), 1u);
    EXPECT_EQ(second->loads[0].name, "x");
    EXPECT_TRUE(second->runs.empty());
    EXPECT_EQ(second->reads, std::vector<sandy::device::DeviceBufferId>({200}));
    EXPECT_EQ(second->deallocs, std::vector<sandy::device::DeviceBufferId>({200}));
}

TEST_F(EngineCompileTest, RunExecutesKernelOnDeviceSelectedByTransferredInput) {
    using namespace sandy::ir::kernel_ir;

    auto graph = std::make_unique<Graph>();
    auto input = graph->addValue(
        ValueType{ValueKind::Tensor, sandy::core::DType::F32, sandy::core::Shape({1})},
        "x");
    graph->addOp<InputOp>(
        InputSource{InputSourceKind::Argument, 0, ""},
        input);

    auto transferred = graph->addValue(
        ValueType{ValueKind::Tensor, sandy::core::DType::F32, sandy::core::Shape({1})},
        "x_on_device_1",
        1);
    graph->addOp<DeviceTransferOp>(0, 1, input, transferred);

    auto output = graph->addValue(
        ValueType{ValueKind::Tensor, sandy::core::DType::F32, sandy::core::Shape({1})},
        "out",
        1);
    auto* elementwise = graph->addOp<ElementwiseKernelOp>(
        std::vector<ElementwiseInput>{ElementwiseInput{transferred, BroadcastMode::None}},
        output,
        1,
        std::vector<ScalarNode>{
            ScalarNode{0, ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            ScalarNode{1, ScalarOp::Tanh, sandy::core::DType::F32, 0, 0.0, {0}},
        },
        1);
    graph->setOutputs({output});

    auto verify = graph->verify();
    ASSERT_TRUE(verify) << verify.error();

    sandy::engine::CompiledKernelGraph compiled;
    compiled.graph = std::move(graph);
    compiled.defaultDevice = 0;
    compiled.device = 0;
    compiled.deviceGraphs[1] = 900;

    FakeDevice* first = nullptr;
    FakeDevice* second = nullptr;
    auto engine = make_two_device_engine(&first, &second);

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    auto outputsResult = engine.run(compiled, inputs, weights);
    ASSERT_TRUE(outputsResult) << outputsResult.error();
    ASSERT_EQ(outputsResult->size(), 1u);

    EXPECT_TRUE(first->runs.empty());
    EXPECT_EQ(first->reads, std::vector<sandy::device::DeviceBufferId>({200}));
    EXPECT_EQ(first->deallocs, std::vector<sandy::device::DeviceBufferId>({200}));

    ASSERT_EQ(second->allocDescs.size(), 1u);
    EXPECT_EQ(second->allocDescs[0].shape, sandy::core::Shape({1}));
    ASSERT_EQ(second->runs.size(), 1u);
    EXPECT_EQ(second->runs[0].graph, 900u);
    EXPECT_EQ(second->runs[0].op, elementwise->id());
    EXPECT_EQ(second->runs[0].inputs, std::vector<sandy::device::DeviceBufferId>({200}));
    EXPECT_EQ(second->runs[0].outputs, std::vector<sandy::device::DeviceBufferId>({201}));
    EXPECT_EQ(second->reads, std::vector<sandy::device::DeviceBufferId>({201}));
    expect_same_buffers(second->deallocs, {200, 201});
}

TEST_F(EngineCompileTest, RunReportsProfileEventsForKernels) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* out = builder.createTanh(x);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    std::vector<sandy::engine::EngineProfileEvent> events;
    sandy::engine::EngineRunOptions options;
    options.profileKernel = [&](const sandy::engine::EngineProfileEvent& event) {
        events.push_back(event);
    };

    auto outputsResult = engine.run(*compiled, inputs, weights, &options);
    ASSERT_TRUE(outputsResult) << outputsResult.error();

    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].opIndex, 1u);
    EXPECT_EQ(events[0].op, 1u);
    EXPECT_EQ(events[0].device, 0u);
    EXPECT_EQ(events[0].deviceGraph, 100u);
    EXPECT_EQ(events[0].opKind, sandy::ir::kernel_ir::OpKind::ElementwiseKernel);
    EXPECT_EQ(events[0].inputCount, 1u);
    EXPECT_EQ(events[0].outputCount, 1u);
    EXPECT_GE(events[0].elapsedMs, 0.0);
}

TEST_F(EngineCompileTest, RunFailsForMissingInputIndex) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(1, sandy::core::Shape({1}), sandy::core::DType::F32);
    builder.setOutputs(std::span<sandy::ir::mid_ir::Value* const>(&x, 1));

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(std::make_shared<FakeTensorBuffer>(
        sandy::core::TensorDesc("x", sandy::core::Shape({1}), sandy::core::DType::F32)));
    sandy::engine::TensorMap weights;

    auto result = engine.run(*compiled, inputs, weights);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("input index out of range"), std::string::npos);
}

TEST_F(EngineCompileTest, RunFailsForMissingWeight) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* w = builder.createWeight("w", sandy::core::Shape({1}), sandy::core::DType::F32);
    builder.setOutputs(std::span<sandy::ir::mid_ir::Value* const>(&w, 1));

    FakeDevice* fakePtr = nullptr;
    auto engine = make_engine(&fakePtr);
    auto compiledResult = engine.compile(graph);
    ASSERT_TRUE(compiledResult) << compiledResult.error();
    auto compiled = compiledResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    sandy::engine::TensorMap weights;

    auto result = engine.run(*compiled, inputs, weights);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("missing weight buffer: w"), std::string::npos);
}
