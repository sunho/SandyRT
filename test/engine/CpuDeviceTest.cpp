#include "CpuDevice.h"
#include "MidIRCpuInterpreter.h"
#include "MidIRToKernelIR.h"
#include "MidIR.h"

#include <gtest/gtest.h>

#include <cmath>
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

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

class CpuDeviceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sandy::ir::mid_ir::register_all_ops();
    }
};

std::vector<uint8_t> f32_bytes(std::initializer_list<float> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    size_t index = 0;
    for (float value : values) {
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
        index++;
    }
    return bytes;
}

std::shared_ptr<TestTensorBuffer> make_f32_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        std::initializer_list<float> values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::F32),
        f32_bytes(values));
}

float read_f32(std::span<const uint8_t> bytes, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
    return value;
}

void expect_f32_output(
        sandy::device::CpuDevice& device,
        sandy::device::DeviceBufferId buffer,
        std::initializer_list<float> expected) {
    auto read = device.read(buffer);
    ASSERT_TRUE(read) << read.error();
    auto access = (*read)->access();
    ASSERT_TRUE(access) << access.error();
    auto data = (*access).data();
    ASSERT_EQ(data.size(), expected.size() * sizeof(float));

    size_t index = 0;
    for (float value : expected) {
        EXPECT_NEAR(read_f32(data, index), value, 1.0e-5f);
        index++;
    }
}

sandy::ir::kernel_ir::ValueType tensor_type(
        sandy::core::Shape shape,
        sandy::core::DType dtype = sandy::core::DType::F32) {
    return sandy::ir::kernel_ir::ValueType{
        sandy::ir::kernel_ir::ValueKind::Tensor,
        dtype,
        std::move(shape),
    };
}

sandy::device::DeviceTensorView tensor_view(
        sandy::device::CpuDevice& device,
        sandy::device::DeviceBufferId buffer,
        sandy::core::TensorDesc desc) {
    auto view = device.defaultView(std::move(desc));
    return sandy::device::DeviceTensorView{
        buffer,
        view.take(),
    };
}

} // namespace

TEST_F(CpuDeviceTest, LoadReadAndDeallocBuffer) {
    sandy::device::CpuDevice device;
    auto host = make_f32_buffer("x", sandy::core::Shape({2}), {1.0f, 2.0f});

    auto loaded = device.load(*host);
    ASSERT_TRUE(loaded) << loaded.error();
    expect_f32_output(device, *loaded, {1.0f, 2.0f});

    auto dealloc = device.dealloc(*loaded);
    EXPECT_TRUE(dealloc) << dealloc.error();
    auto readAfterDealloc = device.read(*loaded);
    EXPECT_FALSE(readAfterDealloc);
}

TEST_F(CpuDeviceTest, DefaultViewRejectsDynamicShape) {
    sandy::device::CpuDevice device;
    auto view = device.defaultView(
        sandy::core::TensorDesc(sandy::core::Shape({sandy::core::Shape::kDynamic, 2}),
                                sandy::core::DType::F32));
    EXPECT_FALSE(view);
    EXPECT_NE(view.error().find("dynamic shape"), std::string::npos);
}

TEST_F(CpuDeviceTest, RunReshapeF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {3, 2});
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_TRUE(device.dealloc(*xBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, RunAddF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* lhs = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createAdd(lhs, rhs);
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer("lhs", sandy::core::Shape({2}), {1.0f, 2.0f});
    auto rhsHost = make_f32_buffer("rhs", sandy::core::Shape({2}), {3.0f, 4.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *lhsBuffer, lhsHost->desc()),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {4.0f, 6.0f});
    EXPECT_TRUE(device.dealloc(*lhsBuffer));
    EXPECT_TRUE(device.dealloc(*rhsBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, RunLinearF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("w", sandy::core::Shape({2, 2}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("b", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, weight, bias);
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({1, 2}), {1.0f, 2.0f});
    auto weightHost = make_f32_buffer("w", sandy::core::Shape({2, 2}), {3.0f, 4.0f, 5.0f, 6.0f});
    auto biasHost = make_f32_buffer("b", sandy::core::Shape({2}), {7.0f, 8.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto weightBuffer = device.load(*weightHost);
    ASSERT_TRUE(weightBuffer) << weightBuffer.error();
    auto biasBuffer = device.load(*biasHost);
    ASSERT_TRUE(biasBuffer) << biasBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *weightBuffer, weightHost->desc()),
        tensor_view(device, *biasBuffer, biasHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {18.0f, 25.0f});
    EXPECT_TRUE(device.dealloc(*xBuffer));
    EXPECT_TRUE(device.dealloc(*weightBuffer));
    EXPECT_TRUE(device.dealloc(*biasBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, RunTanhF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createTanh(x);
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({2}), {0.0f, 1.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {0.0f, std::tanh(1.0f)});
    EXPECT_TRUE(device.dealloc(*xBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, CompileRejectsChainedUnaryElementwiseKernel) {
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({2}));
    graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{input, kir::BroadcastMode::None},
        },
        output,
        2,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Sqrt, sandy::core::DType::F32, 0, 0.0, {0}},
            kir::ScalarNode{2, kir::ScalarOp::Tanh, sandy::core::DType::F32, 0, 0.0, {1}},
        });
    graph.setOutputs({output});

    sandy::device::CpuDevice device;
    auto compiled = device.compile(graph);
    EXPECT_FALSE(compiled);
    EXPECT_NE(compiled.error().find("unary kernel must be a single op"), std::string::npos);
}

TEST_F(CpuDeviceTest, CompileRejectsChainedBinaryElementwiseKernel) {
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({2}));
    graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{lhs, kir::BroadcastMode::None},
            kir::ElementwiseInput{rhs, kir::BroadcastMode::None},
        },
        output,
        3,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Load, sandy::core::DType::F32, 1, 0.0, {}},
            kir::ScalarNode{2, kir::ScalarOp::Add, sandy::core::DType::F32, 0, 0.0, {0, 1}},
            kir::ScalarNode{3, kir::ScalarOp::Mul, sandy::core::DType::F32, 0, 0.0, {2, 1}},
        });
    graph.setOutputs({output});

    sandy::device::CpuDevice device;
    auto compiled = device.compile(graph);
    EXPECT_FALSE(compiled);
    EXPECT_NE(compiled.error().find("binary kernel must be a single op"), std::string::npos);
}

TEST_F(CpuDeviceTest, DebugMidIRCpuInterpreterRunsOldPerOpDispatch) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* lhs = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createAdd(lhs, rhs);

    auto lhsBytes = f32_bytes({1.0f, 2.0f});
    auto rhsBytes = f32_bytes({3.0f, 4.0f});
    std::vector<uint8_t> outBytes(2 * sizeof(float));

    auto lhsRef = sandy::core::make_tensor_ref(
        sandy::core::TensorDesc(lhs->shape, lhs->dtype),
        lhsBytes);
    ASSERT_TRUE(lhsRef) << lhsRef.error();
    auto rhsRef = sandy::core::make_tensor_ref(
        sandy::core::TensorDesc(rhs->shape, rhs->dtype),
        rhsBytes);
    ASSERT_TRUE(rhsRef) << rhsRef.error();
    auto outRef = sandy::core::make_mutable_tensor_ref(
        sandy::core::TensorDesc(out->shape, out->dtype),
        outBytes);
    ASSERT_TRUE(outRef) << outRef.error();

    std::vector<sandy::core::TensorRef> inputs = {*lhsRef, *rhsRef};
    std::vector<sandy::core::MutableTensorRef> outputs = {*outRef};
    auto run = sandy::device::debug::runMidIROpOnCpu(*out->def, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    EXPECT_FLOAT_EQ(read_f32(outBytes, 0), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(outBytes, 1), 6.0f);
}
