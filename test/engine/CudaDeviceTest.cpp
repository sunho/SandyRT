#include "CudaDevice.h"
#include "KernelIR.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
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

std::shared_ptr<TestTensorBuffer> make_i64_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        std::initializer_list<int64_t> values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::I64),
        i64_bytes(values));
}

float read_f32(std::span<const uint8_t> bytes, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
    return value;
}

std::string cuda_device_skip_reason() {
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        std::string reason = cudaGetErrorString(status);
        cudaGetLastError();
        return "CUDA runtime has no usable device: " + reason;
    }
    if (deviceCount == 0)
        return "CUDA runtime reports no devices";
    return "";
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
        sandy::device::CudaDevice& device,
        sandy::device::DeviceBufferId buffer,
        sandy::core::TensorDesc desc) {
    auto view = device.defaultView(std::move(desc));
    EXPECT_TRUE(view) << view.error();
    return sandy::device::DeviceTensorView{
        buffer,
        view.take(),
    };
}

void expect_f32_output(
        sandy::device::CudaDevice& device,
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

} // namespace

TEST(CudaDeviceTest, RunChainedElementwiseF32) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({3}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
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

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto host = make_f32_buffer("x", sandy::core::Shape({3}), {0.0f, 1.0f, 4.0f});
    auto inputBuffer = device.load(*host);
    ASSERT_TRUE(inputBuffer) << inputBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({3}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *inputBuffer, host->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({3}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {0.0f, std::tanh(1.0f), std::tanh(2.0f)});
}

TEST(CudaDeviceTest, RunSuffixBroadcastAddF32) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({2, 3}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{lhs, kir::BroadcastMode::None},
            kir::ElementwiseInput{rhs, kir::BroadcastMode::RightAligned},
        },
        output,
        2,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Load, sandy::core::DType::F32, 1, 0.0, {}},
            kir::ScalarNode{2, kir::ScalarOp::Add, sandy::core::DType::F32, 0, 0.0, {0, 1}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer(
        "lhs",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f});
    auto rhsHost = make_f32_buffer(
        "rhs",
        sandy::core::Shape({3}),
        {100.0f, 200.0f, 300.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *lhsBuffer, lhsHost->desc()),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {101.0f, 202.0f, 303.0f, 110.0f, 220.0f, 330.0f});
}

TEST(CudaDeviceTest, RunGatherF32WithI32Ids) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto ids = graph.addValue(tensor_type({3}, sandy::core::DType::I32));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        ids);
    auto table = graph.addValue(tensor_type({4, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        table);
    auto output = graph.addValue(tensor_type({3, 2}));
    auto* op = graph.addOp<kir::GatherKernelOp>(ids, table, output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto idsHost = make_i32_buffer("ids", sandy::core::Shape({3}), {2, 0, 3});
    auto tableHost = make_f32_buffer(
        "table",
        sandy::core::Shape({4, 2}),
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
        });
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto tableBuffer = device.load(*tableHost);
    ASSERT_TRUE(tableBuffer) << tableBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({3, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *tableBuffer, tableHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({3, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outputBuffer, {5.0f, 6.0f, 1.0f, 2.0f, 7.0f, 8.0f});
}

TEST(CudaDeviceTest, RunGatherF32WithI64IdsKeepsLeadingDims) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto ids = graph.addValue(tensor_type({2, 2}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        ids);
    auto table = graph.addValue(tensor_type({4, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        table);
    auto output = graph.addValue(tensor_type({2, 2, 2}));
    auto* op = graph.addOp<kir::GatherKernelOp>(ids, table, output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto idsHost = make_i64_buffer("ids", sandy::core::Shape({2, 2}), {1, 3, 0, 2});
    auto tableHost = make_f32_buffer(
        "table",
        sandy::core::Shape({4, 2}),
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
        });
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto tableBuffer = device.load(*tableHost);
    ASSERT_TRUE(tableBuffer) << tableBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({2, 2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *tableBuffer, tableHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {3.0f, 4.0f, 7.0f, 8.0f, 1.0f, 2.0f, 5.0f, 6.0f});
}

TEST(CudaDeviceTest, RunGatherReportsOutOfRangeId) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto ids = graph.addValue(tensor_type({2}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        ids);
    auto table = graph.addValue(tensor_type({2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        table);
    auto output = graph.addValue(tensor_type({2, 2}));
    auto* op = graph.addOp<kir::GatherKernelOp>(ids, table, output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto idsHost = make_i64_buffer("ids", sandy::core::Shape({2}), {0, 2});
    auto tableHost = make_f32_buffer(
        "table",
        sandy::core::Shape({2, 2}),
        {1.0f, 2.0f, 3.0f, 4.0f});
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto tableBuffer = device.load(*tableHost);
    ASSERT_TRUE(tableBuffer) << tableBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *tableBuffer, tableHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("embedding id out of range"), std::string::npos);
}

TEST(CudaDeviceTest, RunMatMulF32) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({3, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({2, 2}));
    auto* op = graph.addOp<kir::MatMulKernelOp>(
        lhs,
        rhs,
        output,
        false,
        false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer(
        "lhs",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto rhsHost = make_f32_buffer(
        "rhs",
        sandy::core::Shape({3, 2}),
        {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *lhsBuffer, lhsHost->desc()),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outputBuffer, {58.0f, 64.0f, 139.0f, 154.0f});
}

TEST(CudaDeviceTest, RunMatMulF32TransposedRhs) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({2, 2}));
    auto* op = graph.addOp<kir::MatMulKernelOp>(
        lhs,
        rhs,
        output,
        false,
        true);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer(
        "lhs",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto rhsHost = make_f32_buffer(
        "rhs",
        sandy::core::Shape({2, 3}),
        {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *lhsBuffer, lhsHost->desc()),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outputBuffer, {50.0f, 68.0f, 122.0f, 167.0f});
}

TEST(CudaDeviceTest, RunMatMulF32GroupedBatchHeads) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({1, 4, 2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({1, 2, 3, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({1, 4, 2, 2}));
    auto* op = graph.addOp<kir::MatMulKernelOp>(
        lhs,
        rhs,
        output,
        false,
        false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer(
        "lhs",
        sandy::core::Shape({1, 4, 2, 3}),
        {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f,
            1.0f, 2.0f, 0.0f,
            0.0f, 1.0f, 1.0f,
            2.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 1.0f,
        });
    auto rhsHost = make_f32_buffer(
        "rhs",
        sandy::core::Shape({1, 2, 3, 2}),
        {
            10.0f, 100.0f,
            20.0f, 200.0f,
            30.0f, 300.0f,
            1.0f, 10.0f,
            2.0f, 20.0f,
            3.0f, 30.0f,
        });
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({1, 4, 2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *lhsBuffer, lhsHost->desc()),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 4, 2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            10.0f, 100.0f,
            20.0f, 200.0f,
            30.0f, 300.0f,
            30.0f, 300.0f,
            5.0f, 50.0f,
            5.0f, 50.0f,
            5.0f, 50.0f,
            4.0f, 40.0f,
        });
}

TEST(CudaDeviceTest, RunSoftmaxF32LastDim) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({2, 3}));
    auto* op = graph.addOp<kir::SoftmaxKernelOp>(x, output, -1);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    float exp0 = std::exp(-2.0f);
    float exp1 = std::exp(-1.0f);
    float inv = 1.0f / (exp0 + exp1 + 1.0f);
    expect_f32_output(
        device,
        *outputBuffer,
        {
            exp0 * inv,
            exp1 * inv,
            inv,
            1.0f / 3.0f,
            1.0f / 3.0f,
            1.0f / 3.0f,
        });
}

TEST(CudaDeviceTest, RunSoftmaxRejectsNonLastDim) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({2, 3}));
    auto* op = graph.addOp<kir::SoftmaxKernelOp>(x, output, 0);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("only supports last dimension"), std::string::npos);
}

TEST(CudaDeviceTest, RunRMSNormF32WithWeight) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto weight = graph.addValue(tensor_type({4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        weight);
    auto output = graph.addValue(tensor_type({2, 4}));
    auto* op = graph.addOp<kir::NormKernelOp>(
        kir::NormKind::RMSNorm,
        std::vector<kir::ValueId>{x, weight},
        output,
        1.0e-6);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 4}),
        {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 0.0f, -2.0f, 4.0f});
    auto weightHost = make_f32_buffer(
        "weight",
        sandy::core::Shape({4}),
        {1.0f, 0.5f, 2.0f, -1.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto weightBuffer = device.load(*weightHost);
    ASSERT_TRUE(weightBuffer) << weightBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *weightBuffer, weightHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    float inv0 = 1.0f / std::sqrt(30.0f / 4.0f + 1.0e-6f);
    float inv1 = 1.0f / std::sqrt(24.0f / 4.0f + 1.0e-6f);
    expect_f32_output(
        device,
        *outputBuffer,
        {
            1.0f * inv0,
            2.0f * inv0 * 0.5f,
            3.0f * inv0 * 2.0f,
            4.0f * inv0 * -1.0f,
            2.0f * inv1,
            0.0f,
            -2.0f * inv1 * 2.0f,
            4.0f * inv1 * -1.0f,
        });
}

TEST(CudaDeviceTest, RunRMSNormF32WithoutWeight) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({1, 4}));
    auto* op = graph.addOp<kir::NormKernelOp>(
        kir::NormKind::RMSNorm,
        std::vector<kir::ValueId>{x},
        output,
        1.0e-6);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 4}),
        {1.0f, -1.0f, 3.0f, -3.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    float inv = 1.0f / std::sqrt(20.0f / 4.0f + 1.0e-6f);
    expect_f32_output(
        device,
        *outputBuffer,
        {1.0f * inv, -1.0f * inv, 3.0f * inv, -3.0f * inv});
}

TEST(CudaDeviceTest, RunLayerNormF32) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto weight = graph.addValue(tensor_type({3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        weight);
    auto bias = graph.addValue(tensor_type({3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        bias);
    auto output = graph.addValue(tensor_type({2, 3}));
    auto* op = graph.addOp<kir::NormKernelOp>(
        kir::NormKind::LayerNorm,
        std::vector<kir::ValueId>{x, weight, bias},
        output,
        1.0e-5);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 4.0f});
    auto weightHost = make_f32_buffer(
        "weight",
        sandy::core::Shape({3}),
        {1.0f, 2.0f, -1.0f});
    auto biasHost = make_f32_buffer(
        "bias",
        sandy::core::Shape({3}),
        {0.5f, -0.5f, 1.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto weightBuffer = device.load(*weightHost);
    ASSERT_TRUE(weightBuffer) << weightBuffer.error();
    auto biasBuffer = device.load(*biasHost);
    ASSERT_TRUE(biasBuffer) << biasBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *weightBuffer, weightHost->desc()),
        tensor_view(device, *biasBuffer, biasHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    float mean0 = 2.0f;
    float inv0 = 1.0f / std::sqrt(2.0f / 3.0f + 1.0e-5f);
    float mean1 = 10.0f / 3.0f;
    float inv1 = 1.0f / std::sqrt(8.0f / 9.0f + 1.0e-5f);
    expect_f32_output(
        device,
        *outputBuffer,
        {
            (1.0f - mean0) * inv0 + 0.5f,
            (2.0f - mean0) * inv0 * 2.0f - 0.5f,
            (3.0f - mean0) * inv0 * -1.0f + 1.0f,
            (2.0f - mean1) * inv1 + 0.5f,
            (4.0f - mean1) * inv1 * 2.0f - 0.5f,
            (4.0f - mean1) * inv1 * -1.0f + 1.0f,
        });
}
