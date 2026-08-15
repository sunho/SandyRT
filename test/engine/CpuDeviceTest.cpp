#include "CpuDevice.h"
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
        sandy::engine::CpuDevice& device,
        sandy::engine::DeviceBufferId buffer,
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

TEST_F(CpuDeviceTest, LoadReadAndDeallocBuffer) {
    sandy::engine::CpuDevice device;
    auto host = make_f32_buffer("x", sandy::core::Shape({2}), {1.0f, 2.0f});

    auto loaded = device.load(*host);
    ASSERT_TRUE(loaded) << loaded.error();
    expect_f32_output(device, *loaded, {1.0f, 2.0f});

    auto dealloc = device.dealloc(*loaded);
    EXPECT_TRUE(dealloc) << dealloc.error();
    auto readAfterDealloc = device.read(*loaded);
    EXPECT_FALSE(readAfterDealloc);
}

TEST_F(CpuDeviceTest, CompileRejectsReshape) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {3, 2});

    sandy::engine::CpuDevice device;
    auto program = device.compile(*out->def);
    EXPECT_FALSE(program);
    EXPECT_NE(program.error().find("reshape"), std::string::npos);
}

TEST_F(CpuDeviceTest, RunAddF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* lhs = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createAdd(lhs, rhs);

    sandy::engine::CpuDevice device;
    auto program = device.compile(*out->def);
    ASSERT_TRUE(program) << program.error();

    auto lhsHost = make_f32_buffer("lhs", sandy::core::Shape({2}), {1.0f, 2.0f});
    auto rhsHost = make_f32_buffer("rhs", sandy::core::Shape({2}), {3.0f, 4.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::engine::DeviceBufferId> inputs = {*lhsBuffer, *rhsBuffer};
    std::vector<sandy::engine::DeviceBufferId> outputs = {*outBuffer};
    auto run = device.run(*program, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {4.0f, 6.0f});
}

TEST_F(CpuDeviceTest, RunLinearF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("w", sandy::core::Shape({2, 2}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("b", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, weight, bias);

    sandy::engine::CpuDevice device;
    auto program = device.compile(*out->def);
    ASSERT_TRUE(program) << program.error();

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

    std::vector<sandy::engine::DeviceBufferId> inputs = {*xBuffer, *weightBuffer, *biasBuffer};
    std::vector<sandy::engine::DeviceBufferId> outputs = {*outBuffer};
    auto run = device.run(*program, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {18.0f, 25.0f});
}

TEST_F(CpuDeviceTest, RunTanhF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createTanh(x);

    sandy::engine::CpuDevice device;
    auto program = device.compile(*out->def);
    ASSERT_TRUE(program) << program.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({2}), {0.0f, 1.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::engine::DeviceBufferId> inputs = {*xBuffer};
    std::vector<sandy::engine::DeviceBufferId> outputs = {*outBuffer};
    auto run = device.run(*program, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {0.0f, std::tanh(1.0f)});
}
