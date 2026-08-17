#include "CudaDevice.h"
#include "KernelIR.h"
#include "TensorCalc.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
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

std::vector<uint8_t> f32_bytes(std::span<const float> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    for (size_t index = 0; index < values.size(); index++) {
        std::memcpy(bytes.data() + index * sizeof(float), &values[index], sizeof(float));
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

std::shared_ptr<TestTensorBuffer> make_f32_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        const std::vector<float>& values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::F32),
        f32_bytes(std::span<const float>(values.data(), values.size())));
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

void expect_f32_output_near(
        sandy::device::CudaDevice& device,
        sandy::device::DeviceBufferId buffer,
        const std::vector<float>& expected,
        float tolerance = 1.0e-4f) {
    auto read = device.read(buffer);
    ASSERT_TRUE(read) << read.error();
    auto access = (*read)->access();
    ASSERT_TRUE(access) << access.error();
    auto data = (*access).data();
    ASSERT_EQ(data.size(), expected.size() * sizeof(float));

    for (size_t index = 0; index < expected.size(); index++) {
        EXPECT_NEAR(read_f32(data, index), expected[index], tolerance)
            << "at flat index " << index;
    }
}

std::vector<float> make_pattern(size_t count, float scale, float bias) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; i++) {
        int centered = static_cast<int>((i * 17 + 13) % 29) - 14;
        values[i] = bias + scale * static_cast<float>(centered);
    }
    return values;
}

std::vector<float> make_uniform_values(
        std::mt19937& rng,
        size_t count,
        float minValue,
        float maxValue) {
    std::uniform_real_distribution<float> values(minValue, maxValue);
    std::vector<float> result(count);
    for (float& value : result)
        value = values(rng);
    return result;
}

Result<std::vector<float>> attention_reference(
        const std::vector<float>& q,
        const std::vector<float>& k,
        const std::vector<float>& v,
        int64_t batch,
        int64_t qHeads,
        int64_t kvHeads,
        int64_t tq,
        int64_t tk,
        int64_t headDim,
        int64_t window,
        float scale,
        const std::optional<std::vector<int64_t>>& positionOffsets = std::nullopt) {
    auto qDesc = sandy::core::TensorDesc({batch, qHeads, tq, headDim}, sandy::core::DType::F32);
    auto kDesc = sandy::core::TensorDesc({batch, kvHeads, tk, headDim}, sandy::core::DType::F32);
    auto vDesc = sandy::core::TensorDesc({batch, kvHeads, tk, headDim}, sandy::core::DType::F32);
    auto outDesc = sandy::core::TensorDesc({batch, qHeads, tq, headDim}, sandy::core::DType::F32);

    auto qBytes = f32_bytes(std::span<const float>(q.data(), q.size()));
    auto kBytes = f32_bytes(std::span<const float>(k.data(), k.size()));
    auto vBytes = f32_bytes(std::span<const float>(v.data(), v.size()));
    std::vector<uint8_t> outBytes(
        static_cast<size_t>(batch * qHeads * tq * headDim) * sizeof(float));

    auto qRef = sandy::core::make_tensor_ref(qDesc, qBytes);
    if (!qRef) return make_error(qRef.error());
    auto kRef = sandy::core::make_tensor_ref(kDesc, kBytes);
    if (!kRef) return make_error(kRef.error());
    auto vRef = sandy::core::make_tensor_ref(vDesc, vBytes);
    if (!vRef) return make_error(vRef.error());
    auto outRef = sandy::core::make_mutable_tensor_ref(outDesc, outBytes);
    if (!outRef) return make_error(outRef.error());

    if (positionOffsets) {
        std::vector<uint8_t> positionBytes(positionOffsets->size() * sizeof(int64_t));
        for (size_t i = 0; i < positionOffsets->size(); i++) {
            std::memcpy(
                positionBytes.data() + i * sizeof(int64_t),
                &(*positionOffsets)[i],
                sizeof(int64_t));
        }
        auto positionRef = sandy::core::make_tensor_ref(
            sandy::core::TensorDesc({batch}, sandy::core::DType::I64),
            positionBytes);
        if (!positionRef) return make_error(positionRef.error());
        auto result = sandy::core::attention(
            *qRef,
            *kRef,
            *vRef,
            *positionRef,
            window,
            scale,
            *outRef);
        if (!result) return make_error(result.error());
    } else {
        auto result = sandy::core::attention(
            *qRef,
            *kRef,
            *vRef,
            window,
            scale,
            *outRef);
        if (!result) return make_error(result.error());
    }

    std::vector<float> output(static_cast<size_t>(batch * qHeads * tq * headDim));
    for (size_t i = 0; i < output.size(); i++)
        output[i] = read_f32(outBytes, i);
    return output;
}

void run_attention_test(
        int64_t batch,
        int64_t qHeads,
        int64_t kvHeads,
        int64_t tq,
        int64_t tk,
        int64_t headDim,
        int64_t window,
        float scale,
        const std::vector<float>& qValues,
        const std::vector<float>& kValues,
        const std::vector<float>& vValues,
        const std::optional<std::vector<int64_t>>& positionOffsets = std::nullopt,
        float tolerance = 1.0e-4f) {
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto q = graph.addValue(tensor_type({batch, qHeads, tq, headDim}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        q);
    auto k = graph.addValue(tensor_type({batch, kvHeads, tk, headDim}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        k);
    auto v = graph.addValue(tensor_type({batch, kvHeads, tk, headDim}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        v);
    auto output = graph.addValue(tensor_type({batch, qHeads, tq, headDim}));

    kir::AttentionKernelOp* op = nullptr;
    if (positionOffsets) {
        auto positions = graph.addValue(tensor_type({batch}, sandy::core::DType::I64));
        graph.addOp<kir::InputOp>(
            kir::InputSource{kir::InputSourceKind::Argument, 3, ""},
            positions);
        op = graph.addOp<kir::AttentionKernelOp>(
            q,
            k,
            v,
            positions,
            output,
            window,
            scale);
    } else {
        op = graph.addOp<kir::AttentionKernelOp>(
            q,
            k,
            v,
            output,
            window,
            scale);
    }
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto qHost = make_f32_buffer("q", sandy::core::Shape({batch, qHeads, tq, headDim}), qValues);
    auto kHost = make_f32_buffer("k", sandy::core::Shape({batch, kvHeads, tk, headDim}), kValues);
    auto vHost = make_f32_buffer("v", sandy::core::Shape({batch, kvHeads, tk, headDim}), vValues);
    auto qBuffer = device.load(*qHost);
    ASSERT_TRUE(qBuffer) << qBuffer.error();
    auto kBuffer = device.load(*kHost);
    ASSERT_TRUE(kBuffer) << kBuffer.error();
    auto vBuffer = device.load(*vHost);
    ASSERT_TRUE(vBuffer) << vBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({batch, qHeads, tq, headDim}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *qBuffer, qHost->desc()),
        tensor_view(device, *kBuffer, kHost->desc()),
        tensor_view(device, *vBuffer, vHost->desc()),
    };
    std::shared_ptr<TestTensorBuffer> positionHost;
    sandy::device::DeviceBufferId positionBuffer = 0;
    if (positionOffsets) {
        std::vector<uint8_t> bytes(positionOffsets->size() * sizeof(int64_t));
        for (size_t i = 0; i < positionOffsets->size(); i++) {
            std::memcpy(
                bytes.data() + i * sizeof(int64_t),
                &(*positionOffsets)[i],
                sizeof(int64_t));
        }
        positionHost = std::make_shared<TestTensorBuffer>(
            sandy::core::TensorDesc({batch}, sandy::core::DType::I64),
            std::move(bytes));
        auto loadedPosition = device.load(*positionHost);
        ASSERT_TRUE(loadedPosition) << loadedPosition.error();
        positionBuffer = *loadedPosition;
        inputs.push_back(tensor_view(device, positionBuffer, positionHost->desc()));
    }

    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({batch, qHeads, tq, headDim}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    auto expected = attention_reference(
        qValues,
        kValues,
        vValues,
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        window,
        scale,
        positionOffsets);
    ASSERT_TRUE(expected) << expected.error();
    expect_f32_output_near(device, *outputBuffer, expected.take(), tolerance);
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

TEST(CudaDeviceTest, RunAttentionPrefillF32FullCausalHeadDim64MatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 1;
    int64_t qHeads = 1;
    int64_t kvHeads = 1;
    int64_t tq = 4;
    int64_t tk = 4;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        0,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.01f, 0.02f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.015f, -0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.02f, 0.03f));
}

TEST(CudaDeviceTest, RunAttentionPrefillF32SlidingWindowHeadDim64MatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 1;
    int64_t qHeads = 1;
    int64_t kvHeads = 1;
    int64_t tq = 6;
    int64_t tk = 6;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        2,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.011f, 0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.013f, -0.02f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.017f, 0.04f));
}

TEST(CudaDeviceTest, RunAttentionPrefillF32GroupedQueryHeadsMatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 2;
    int64_t qHeads = 4;
    int64_t kvHeads = 2;
    int64_t tq = 5;
    int64_t tk = 5;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        3,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.007f, -0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.009f, 0.02f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.012f, -0.03f));
}

TEST(CudaDeviceTest, RunAttentionPrefillF32BatchedPositionOffsetsMatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 2;
    int64_t qHeads = 2;
    int64_t kvHeads = 1;
    int64_t tq = 3;
    int64_t tk = 8;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        4,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.008f, 0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.01f, -0.02f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.014f, 0.03f),
        std::vector<int64_t>{2, 5});
}

TEST(CudaDeviceTest, RunAttentionPrefillF32GemmaLocalHeadDim256MatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 1;
    int64_t qHeads = 8;
    int64_t kvHeads = 1;
    int64_t tq = 4;
    int64_t tk = 4;
    int64_t headDim = 256;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        512,
        1.0f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.0015f, 0.0f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.0012f, 0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.003f, -0.02f));
}

TEST(CudaDeviceTest, RunAttentionPrefillF32GemmaGlobalHeadDim512MatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 1;
    int64_t qHeads = 8;
    int64_t kvHeads = 2;
    int64_t tq = 3;
    int64_t tk = 3;
    int64_t headDim = 512;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        0,
        1.0f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.001f, 0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.0011f, -0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.0025f, 0.0f));
}

TEST(CudaDeviceTest, RunAttentionDecoderF32LongKvFullContextMatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 1;
    int64_t qHeads = 1;
    int64_t kvHeads = 1;
    int64_t tq = 1;
    int64_t tk = 300;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        0,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.006f, 0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.004f, -0.005f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.005f, 0.02f),
        std::vector<int64_t>{tk - 1});
}

TEST(CudaDeviceTest, RunAttentionDecoderF32LongKvSlidingWindowMatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 1;
    int64_t qHeads = 2;
    int64_t kvHeads = 1;
    int64_t tq = 1;
    int64_t tk = 257;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        17,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.005f, -0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.003f, 0.02f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.006f, -0.03f),
        std::vector<int64_t>{tk - 1});
}

TEST(CudaDeviceTest, RunAttentionDecoderF32LongKvGroupedQueryHeadsMatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    int64_t batch = 2;
    int64_t qHeads = 4;
    int64_t kvHeads = 2;
    int64_t tq = 1;
    int64_t tk = 260;
    int64_t headDim = 64;
    run_attention_test(
        batch,
        qHeads,
        kvHeads,
        tq,
        tk,
        headDim,
        0,
        0.125f,
        make_pattern(static_cast<size_t>(batch * qHeads * tq * headDim), 0.004f, 0.0f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.003f, 0.01f),
        make_pattern(static_cast<size_t>(batch * kvHeads * tk * headDim), 0.005f, -0.02f),
        std::vector<int64_t>{tk - 1, tk - 3});
}

TEST(CudaDeviceTest, RunAttentionF32RandomUniformFuzzMatchesCpu) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    constexpr int kIterations = 1000;
    constexpr uint32_t kSeed = 0x5a17d00du;
    std::mt19937 rng(kSeed);

    const std::vector<int64_t> headDims = {64, 64, 64, 128, 256, 512};
    const std::vector<int64_t> groups = {1, 1, 2, 4};
    std::uniform_int_distribution<int> percent(0, 99);

    for (int iteration = 0; iteration < kIterations; iteration++) {
        int64_t headDim = headDims[static_cast<size_t>(percent(rng)) % headDims.size()];
        int64_t batch = percent(rng) < 25 ? 2 : 1;
        int64_t kvHeads = percent(rng) < 20 ? 2 : 1;
        int64_t group = groups[static_cast<size_t>(percent(rng)) % groups.size()];
        if (headDim >= 256)
            group = std::min<int64_t>(group, 2);
        int64_t qHeads = kvHeads * group;

        bool decoder = percent(rng) < 35;
        int64_t tq = decoder
            ? 1
            : static_cast<int64_t>(2 + (percent(rng) % (headDim >= 256 ? 3 : 5)));
        int64_t tk = decoder
            ? static_cast<int64_t>(1 + (percent(rng) % 384))
            : static_cast<int64_t>(1 + (percent(rng) % (headDim >= 256 ? 8 : 12)));

        bool slidingWindow = percent(rng) < 70;
        int64_t window = slidingWindow
            ? static_cast<int64_t>(1 + (percent(rng) % (tk + tq + 4)))
            : 0;
        float scale = (0.5f + static_cast<float>(percent(rng)) / 99.0f * 1.5f) /
            std::sqrt(static_cast<float>(headDim));

        bool usePositionOffsets = decoder || percent(rng) < 50;
        std::optional<std::vector<int64_t>> positionOffsets;
        if (usePositionOffsets) {
            positionOffsets = std::vector<int64_t>(static_cast<size_t>(batch));
            int64_t maxOffset = tk + tq + 4;
            for (int64_t b = 0; b < batch; b++)
                (*positionOffsets)[static_cast<size_t>(b)] =
                    static_cast<int64_t>(percent(rng) % (maxOffset + 1));
        }

        SCOPED_TRACE(
            ::testing::Message()
            << "seed=" << kSeed
            << " iteration=" << iteration
            << " batch=" << batch
            << " q_heads=" << qHeads
            << " kv_heads=" << kvHeads
            << " tq=" << tq
            << " tk=" << tk
            << " head_dim=" << headDim
            << " window=" << window
            << " positions=" << usePositionOffsets);

        run_attention_test(
            batch,
            qHeads,
            kvHeads,
            tq,
            tk,
            headDim,
            window,
            scale,
            make_uniform_values(
                rng,
                static_cast<size_t>(batch * qHeads * tq * headDim),
                -0.35f,
                0.35f),
            make_uniform_values(
                rng,
                static_cast<size_t>(batch * kvHeads * tk * headDim),
                -0.35f,
                0.35f),
            make_uniform_values(
                rng,
                static_cast<size_t>(batch * kvHeads * tk * headDim),
                -0.75f,
                0.75f),
            positionOffsets,
            5.0e-4f);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST(CudaDeviceTest, RunAttentionRejectsUnsupportedHeadDim) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto q = graph.addValue(tensor_type({1, 1, 1, 32}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        q);
    auto k = graph.addValue(tensor_type({1, 1, 1, 32}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        k);
    auto v = graph.addValue(tensor_type({1, 1, 1, 32}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        v);
    auto output = graph.addValue(tensor_type({1, 1, 1, 32}));
    auto* op = graph.addOp<kir::AttentionKernelOp>(q, k, v, output, 0, 1.0);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto qHost = make_f32_buffer(
        "q",
        sandy::core::Shape({1, 1, 1, 32}),
        make_pattern(32, 0.01f, 0.0f));
    auto kHost = make_f32_buffer(
        "k",
        sandy::core::Shape({1, 1, 1, 32}),
        make_pattern(32, 0.01f, 0.0f));
    auto vHost = make_f32_buffer(
        "v",
        sandy::core::Shape({1, 1, 1, 32}),
        make_pattern(32, 0.01f, 0.0f));
    auto qBuffer = device.load(*qHost);
    ASSERT_TRUE(qBuffer) << qBuffer.error();
    auto kBuffer = device.load(*kHost);
    ASSERT_TRUE(kBuffer) << kBuffer.error();
    auto vBuffer = device.load(*vHost);
    ASSERT_TRUE(vBuffer) << vBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({1, 1, 1, 32}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *qBuffer, qHost->desc()),
        tensor_view(device, *kBuffer, kHost->desc()),
        tensor_view(device, *vBuffer, vHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 1, 1, 32}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("cuda attention unsupported head dimension"), std::string::npos);
}

TEST(CudaDeviceTest, RunRoPEF32ImplicitPositions) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 3, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({1, 3, 4}));
    auto* op = graph.addOp<kir::RoPEKernelOp>(x, output, 10000.0, -1, false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 3, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 0.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 3, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 3, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            std::cos(1.0f), std::sin(1.0f), -std::sin(0.01f), std::cos(0.01f),
            -std::sin(2.0f), std::cos(2.0f), std::cos(0.02f), std::sin(0.02f),
        });
}

TEST(CudaDeviceTest, RunRoPEF32PartialRotaryDimZerosTail) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 2, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({1, 2, 4}));
    auto* op = graph.addOp<kir::RoPEKernelOp>(x, output, 10000.0, 2, false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 2, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 0.0f, 5.0f, 6.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 2, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 2, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            1.0f, 2.0f, 0.0f, 0.0f,
            std::cos(1.0f), std::sin(1.0f), 0.0f, 0.0f,
        });
}

TEST(CudaDeviceTest, RunRoPEF32RuntimePositionIds) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 1, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto positionIds = graph.addValue(tensor_type({1}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        positionIds);
    auto output = graph.addValue(tensor_type({1, 1, 4}));
    auto* op = graph.addOp<kir::RoPEKernelOp>(
        std::vector<kir::ValueId>{x, positionIds},
        output,
        10000.0,
        -1,
        false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 1, 4}),
        {1.0f, 0.0f, 0.0f, 1.0f});
    auto positionHost = make_i64_buffer("position_ids", sandy::core::Shape({1}), {3});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto positionBuffer = device.load(*positionHost);
    ASSERT_TRUE(positionBuffer) << positionBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 1, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *positionBuffer, positionHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 1, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            std::cos(3.0f),
            std::sin(3.0f),
            -std::sin(0.03f),
            std::cos(0.03f),
        });
}

TEST(CudaDeviceTest, RunRoPEF32SplitHalf) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 2, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({1, 2, 4}));
    auto* op = graph.addOp<kir::RoPEKernelOp>(x, output, 10000.0, -1, true);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 2, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 2, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 2, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    float c1 = std::cos(1.0f);
    float s1 = std::sin(1.0f);
    float c001 = std::cos(0.01f);
    float s001 = std::sin(0.01f);
    expect_f32_output(
        device,
        *outputBuffer,
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f * c1 - 3.0f * s1,
            2.0f * c001 - 4.0f * s001,
            1.0f * s1 + 3.0f * c1,
            2.0f * s001 + 4.0f * c001,
        });
}

TEST(CudaDeviceTest, RunRoPERejectsNegativePositionIds) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 1, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto positionIds = graph.addValue(tensor_type({1}, sandy::core::DType::I32));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        positionIds);
    auto output = graph.addValue(tensor_type({1, 1, 4}));
    auto* op = graph.addOp<kir::RoPEKernelOp>(
        std::vector<kir::ValueId>{x, positionIds},
        output,
        10000.0,
        -1,
        false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 1, 4}),
        {1.0f, 0.0f, 0.0f, 1.0f});
    auto positionHost = make_i32_buffer("position_ids", sandy::core::Shape({1}), {-1});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto positionBuffer = device.load(*positionHost);
    ASSERT_TRUE(positionBuffer) << positionBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 1, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *positionBuffer, positionHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 1, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("position_ids must be non-negative"), std::string::npos);
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
