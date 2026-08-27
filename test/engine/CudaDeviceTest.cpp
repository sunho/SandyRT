#include "CudaDevice.h"
#include "CudaElementwiseJit.h"
#include "CudaJit.h"
#include "CudaJitEmbeddedSources.h"
#include "KernelIR.h"
#include "TensorCalc.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <future>
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

uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

float bf16_bits_to_f32(uint16_t bits) {
    uint32_t fbits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &fbits, sizeof(float));
    return value;
}

std::vector<uint8_t> bf16_bytes(std::initializer_list<float> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(uint16_t));
    size_t index = 0;
    for (float value : values) {
        uint16_t bits = f32_to_bf16_bits(value);
        std::memcpy(bytes.data() + index * sizeof(uint16_t), &bits, sizeof(uint16_t));
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

std::shared_ptr<TestTensorBuffer> make_f32_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        const std::vector<float>& values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::F32),
        f32_bytes(std::span<const float>(values.data(), values.size())));
}

std::shared_ptr<TestTensorBuffer> make_bf16_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        std::initializer_list<float> values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::BF16),
        bf16_bytes(values));
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

float read_bf16(std::span<const uint8_t> bytes, size_t index) {
    uint16_t bits = 0;
    std::memcpy(&bits, bytes.data() + index * sizeof(uint16_t), sizeof(uint16_t));
    return bf16_bits_to_f32(bits);
}

Result<std::vector<float>> read_f32_values(
        sandy::device::CudaDevice& device,
        sandy::device::DeviceBufferId buffer) {
    auto read = device.read(buffer);
    if (!read)
        return make_error(read.error());
    auto access = (*read)->access();
    if (!access)
        return make_error(access.error());
    auto data = (*access).data();
    if (data.size() % sizeof(float) != 0)
        return make_error("test output byte size is not a multiple of float");
    std::vector<float> values(data.size() / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = read_f32(data, i);
    return values;
}

int64_t read_i64(std::span<const uint8_t> bytes, size_t index) {
    int64_t value = 0;
    std::memcpy(&value, bytes.data() + index * sizeof(int64_t), sizeof(int64_t));
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

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(const char* name, const char* value)
        : name_(name) {
        if (const char* old = std::getenv(name))
            oldValue_ = std::string(old);
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
    }

    ~ScopedEnvironmentVariable() {
        if (oldValue_)
            setenv(name_.c_str(), oldValue_->c_str(), 1);
        else
            unsetenv(name_.c_str());
    }

private:
    std::string name_;
    std::optional<std::string> oldValue_;
};

sandy::ir::kernel_ir::ValueType tensor_type(
        sandy::core::Shape shape,
        sandy::core::DType dtype = sandy::core::DType::F32) {
    return sandy::ir::kernel_ir::ValueType{
        sandy::ir::kernel_ir::ValueKind::Tensor,
        dtype,
        std::move(shape),
    };
}

sandy::ir::kernel_ir::ValueType paged_tensor_type(
        sandy::core::Shape shape,
        sandy::core::DType dtype,
        int64_t growDim,
        int64_t pageSize) {
    return sandy::ir::kernel_ir::ValueType{
        sandy::ir::kernel_ir::ValueKind::PagedTensor,
        dtype,
        std::move(shape),
        sandy::ir::kernel_ir::PagedTensorMeta{growDim, pageSize},
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

sandy::device::DevicePagedTensorView paged_tensor_view(
        sandy::device::CudaDevice& device,
        sandy::device::DevicePagedTensorId tensor) {
    auto meta = device.pagedMeta(tensor);
    EXPECT_TRUE(meta) << meta.error();
    return sandy::device::DevicePagedTensorView{tensor, meta.take()};
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

void expect_bf16_output_near(
        sandy::device::CudaDevice& device,
        sandy::device::DeviceBufferId buffer,
        std::initializer_list<float> expected,
        float tolerance = 1.0e-2f) {
    auto read = device.read(buffer);
    ASSERT_TRUE(read) << read.error();
    auto access = (*read)->access();
    ASSERT_TRUE(access) << access.error();
    auto data = (*access).data();
    ASSERT_EQ(data.size(), expected.size() * sizeof(uint16_t));

    size_t index = 0;
    for (float value : expected) {
        EXPECT_NEAR(read_bf16(data, index), value, tolerance)
            << "at flat index " << index;
        index++;
    }
}

void expect_i64_output(
        sandy::device::CudaDevice& device,
        sandy::device::DeviceBufferId buffer,
        std::initializer_list<int64_t> expected) {
    auto read = device.read(buffer);
    ASSERT_TRUE(read) << read.error();
    auto access = (*read)->access();
    ASSERT_TRUE(access) << access.error();
    auto data = (*access).data();
    ASSERT_EQ(data.size(), expected.size() * sizeof(int64_t));

    size_t index = 0;
    for (int64_t value : expected) {
        EXPECT_EQ(read_i64(data, index), value);
        index++;
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

TEST(CudaDeviceTest, JitCacheCompilesHighlightedTemplateOnce) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    sandy::device::CudaJitRequest request;
    request.sourceName = "CudaJitElementwiseKernel.cu";
    request.source = sandy::device::embeddedElementwiseKernelSource();
    request.headers = sandy::device::embeddedElementwiseHeaders();
    request.entryName = "sandy_jit_cache_smoke";
    request.options = {"-DSANDY_JIT_ENTRY_NAME=sandy_jit_cache_smoke"};

    sandy::device::CudaJitCache cache;
    auto first = cache.getOrCompile(0, request);
    ASSERT_TRUE(first) << first.error();
    auto second = cache.getOrCompile(0, request);
    ASSERT_TRUE(second) << second.error();
    EXPECT_EQ(*first, *second);
    auto stats = cache.stats();
    EXPECT_EQ(stats.misses, 1u);
    EXPECT_EQ(stats.hits, 1u);
    EXPECT_EQ(stats.entries, 1u);
    EXPECT_GT(stats.compileMilliseconds, 0.0);
}

TEST(CudaDeviceTest, ElementwiseJitEmitterProducesStraightLineExpressions) {
    namespace kir = sandy::ir::kernel_ir;
    sandy::device::CudaElementwiseProgram program{
        {kir::ElementwiseInput{7, kir::BroadcastMode::None}},
        8,
        3,
        {
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Constant, sandy::core::DType::F32, 0, 2.0, {}},
            kir::ScalarNode{2, kir::ScalarOp::Mul, sandy::core::DType::F32, 0, 0.0, {0, 1}},
            kir::ScalarNode{3, kir::ScalarOp::ReLU, sandy::core::DType::F32, 0, 0.0, {2}},
        },
    };
    auto emitted = sandy::device::emitCudaElementwiseJitSource(program);
    ASSERT_TRUE(emitted) << emitted.error();
    EXPECT_NE(emitted->evaluatorSource.find("const float s0 = loader.load(0);"),
              std::string::npos);
    EXPECT_NE(emitted->evaluatorSource.find("const float s2 = s0 * s1;"),
              std::string::npos);
    EXPECT_NE(emitted->evaluatorSource.find("const float s3 = fmaxf(s2, 0.0f);"),
              std::string::npos);
    EXPECT_EQ(emitted->evaluatorSource.find("switch"), std::string::npos);
}

TEST(CudaDeviceTest, JitKeyInvalidatesOnAbiAndTemplateChanges) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    sandy::device::CudaJitRequest request;
    request.sourceName = "CudaJitElementwiseKernel.cu";
    request.source = sandy::device::embeddedElementwiseKernelSource();
    request.headers = sandy::device::embeddedElementwiseHeaders();
    request.entryName = "sandy_jit_key_test";
    request.options = {"-DSANDY_JIT_ENTRY_NAME=sandy_jit_key_test"};
    auto original = sandy::device::buildCudaJitCacheKey(0, request);
    ASSERT_TRUE(original) << original.error();

    request.abiVersion++;
    auto changedAbi = sandy::device::buildCudaJitCacheKey(0, request);
    ASSERT_TRUE(changedAbi) << changedAbi.error();
    EXPECT_NE(*original, *changedAbi);

    request.abiVersion--;
    request.headers.front().source += "\n// changed template";
    auto changedHeader = sandy::device::buildCudaJitCacheKey(0, request);
    ASSERT_TRUE(changedHeader) << changedHeader.error();
    EXPECT_NE(*original, *changedHeader);
}

TEST(CudaDeviceTest, JitCacheCoalescesConcurrentSameKeyCompiles) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    sandy::device::CudaJitRequest request;
    request.sourceName = "CudaJitElementwiseKernel.cu";
    request.source = sandy::device::embeddedElementwiseKernelSource();
    request.headers = sandy::device::embeddedElementwiseHeaders();
    request.entryName = "sandy_jit_concurrent_test";
    request.options = {"-DSANDY_JIT_ENTRY_NAME=sandy_jit_concurrent_test"};
    sandy::device::CudaJitCache cache;

    std::vector<std::future<Result<sandy::device::CudaJitCache::KernelPtr>>> futures;
    for (int i = 0; i < 4; ++i) {
        futures.push_back(std::async(std::launch::async, [&] {
            return cache.getOrCompile(0, request);
        }));
    }
    sandy::device::CudaJitCache::KernelPtr first;
    for (auto& future : futures) {
        auto result = future.get();
        ASSERT_TRUE(result) << result.error();
        if (!first)
            first = *result;
        EXPECT_EQ(first, *result);
    }
    auto stats = cache.stats();
    EXPECT_EQ(stats.misses, 1u);
    EXPECT_EQ(stats.hits, 3u);
    EXPECT_EQ(stats.entries, 1u);
}

TEST(CudaDeviceTest, JitCompileFailureNamesGeneratedHeader) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    sandy::device::CudaJitRequest request;
    request.sourceName = "CudaJitElementwiseKernel.cu";
    request.source = sandy::device::embeddedElementwiseKernelSource();
    request.headers = sandy::device::embeddedElementwiseHeaders();
    for (auto& header : request.headers) {
        if (header.name == "generated/ElementwiseEvaluator.cuh")
            header.source = "#pragma once\nthis is invalid cuda source;\n";
    }
    request.entryName = "sandy_jit_diagnostic_test";
    request.options = {"-DSANDY_JIT_ENTRY_NAME=sandy_jit_diagnostic_test"};

    sandy::device::CudaJitCache cache;
    auto result = cache.getOrCompile(0, request);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("generated/ElementwiseEvaluator.cuh"),
              std::string::npos) << result.error();
}

TEST(CudaDeviceTest, ScratchAllocatorFinalizesReusablePlacementLayout) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;

    sandy::device::CudaDevice device;
    auto allocator = device.createScratchAllocator();
    ASSERT_NE(allocator, nullptr);

    ASSERT_TRUE(allocator->alloc(
        7, sandy::core::TensorDesc(
               sandy::core::Shape({3}), sandy::core::DType::F32)));
    ASSERT_TRUE(allocator->free(7));
    ASSERT_TRUE(allocator->alloc(
        11, sandy::core::TensorDesc(
                sandy::core::Shape({5}), sandy::core::DType::BF16)));
    ASSERT_TRUE(allocator->free(11));

    auto layout = allocator->finalizeLayout();
    ASSERT_TRUE(layout) << layout.error();
    ASSERT_GT(layout->bytes, 0u);
    ASSERT_EQ(layout->placements.size(), 2u);
    EXPECT_EQ(layout->placements.at(7).byteOffset, 0u);
    EXPECT_EQ(layout->placements.at(11).byteOffset, 0u);
    EXPECT_EQ(layout->placements.at(7).desc.shape,
              sandy::core::Shape({3}));
    EXPECT_EQ(layout->placements.at(11).desc.shape,
              sandy::core::Shape({5}));
}

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

TEST(CudaDeviceTest, RunEveryElementwiseScalarOperationF32AndReuseJit) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""}, input);
    auto output = graph.addValue(tensor_type({3}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{{input, kir::BroadcastMode::None}},
        output,
        16,
        std::vector<kir::ScalarNode>{
            {0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            {1, kir::ScalarOp::Constant, sandy::core::DType::F32, 0, 2.0, {}},
            {2, kir::ScalarOp::Add, sandy::core::DType::F32, 0, 0.0, {0, 1}},
            {3, kir::ScalarOp::Sub, sandy::core::DType::F32, 0, 0.0, {2, 1}},
            {4, kir::ScalarOp::Mul, sandy::core::DType::F32, 0, 0.0, {3, 1}},
            {5, kir::ScalarOp::Div, sandy::core::DType::F32, 0, 0.0, {4, 1}},
            {6, kir::ScalarOp::Max, sandy::core::DType::F32, 0, 0.0, {5, 1}},
            {7, kir::ScalarOp::Min, sandy::core::DType::F32, 0, 0.0, {6, 2}},
            {8, kir::ScalarOp::Neg, sandy::core::DType::F32, 0, 0.0, {7}},
            {9, kir::ScalarOp::Neg, sandy::core::DType::F32, 0, 0.0, {8}},
            {10, kir::ScalarOp::Sqrt, sandy::core::DType::F32, 0, 0.0, {9}},
            {11, kir::ScalarOp::Rsqrt, sandy::core::DType::F32, 0, 0.0, {10}},
            {12, kir::ScalarOp::Exp, sandy::core::DType::F32, 0, 0.0, {11}},
            {13, kir::ScalarOp::Log, sandy::core::DType::F32, 0, 0.0, {12}},
            {14, kir::ScalarOp::Tanh, sandy::core::DType::F32, 0, 0.0, {13}},
            {15, kir::ScalarOp::ReLU, sandy::core::DType::F32, 0, 0.0, {14}},
            {16, kir::ScalarOp::Cast, sandy::core::DType::F32, 0, 0.0, {15}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();
    auto compiledAgain = device.compile(graph);
    ASSERT_TRUE(compiledAgain) << compiledAgain.error();
    auto stats = device.jitCacheStats();
    EXPECT_EQ(stats.misses, 1u);
    EXPECT_EQ(stats.hits, 1u);
    EXPECT_EQ(stats.entries, 1u);

    auto host = make_f32_buffer("x", sandy::core::Shape({3}), {0.25f, 2.0f, 4.0f});
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

    std::vector<float> expected;
    for (float x : {0.25f, 2.0f, 4.0f}) {
        float value = x + 2.0f;
        value -= 2.0f;
        value *= 2.0f;
        value /= 2.0f;
        value = std::max(value, 2.0f);
        value = std::min(value, x + 2.0f);
        value = -value;
        value = -value;
        value = std::sqrt(value);
        value = 1.0f / std::sqrt(value);
        value = std::exp(value);
        value = std::log(value);
        value = std::tanh(value);
        value = std::max(value, 0.0f);
        expected.push_back(value);
    }
    expect_f32_output_near(device, *outputBuffer, expected, 1.0e-5f);
}

TEST(CudaDeviceTest, RunElementwiseBF16WithStridedInput) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({3, 2}, sandy::core::DType::BF16));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""}, input);
    auto output = graph.addValue(tensor_type({3, 2}, sandy::core::DType::BF16));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{{input, kir::BroadcastMode::None}},
        output,
        2,
        std::vector<kir::ScalarNode>{
            {0, kir::ScalarOp::Load, sandy::core::DType::BF16, 0, 0.0, {}},
            {1, kir::ScalarOp::Constant, sandy::core::DType::BF16, 0, 0.5, {}},
            {2, kir::ScalarOp::Add, sandy::core::DType::BF16, 0, 0.0, {0, 1}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();
    auto host = make_bf16_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto inputBuffer = device.load(*host);
    ASSERT_TRUE(inputBuffer) << inputBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({3, 2}, sandy::core::DType::BF16));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    sandy::device::TensorViewDesc transposed;
    transposed.desc = sandy::core::TensorDesc({3, 2}, sandy::core::DType::BF16);
    transposed.strides = {1, 3};
    std::vector<sandy::device::DeviceRunValue> inputs = {
        sandy::device::DeviceTensorView{*inputBuffer, transposed},
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device, *outputBuffer,
            sandy::core::TensorDesc({3, 2}, sandy::core::DType::BF16)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();
    expect_bf16_output_near(device, *outputBuffer, {1.5f, 4.5f, 2.5f, 5.5f, 3.5f, 6.5f});
}

TEST(CudaDeviceTest, RunZeroSizedElementwiseOutput) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({0}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""}, input);
    auto output = graph.addValue(tensor_type({0}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{{input, kir::BroadcastMode::None}},
        output,
        0,
        std::vector<kir::ScalarNode>{
            {0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();
    auto host = make_f32_buffer(
        "empty", sandy::core::Shape({0}), std::initializer_list<float>{});
    auto inputBuffer = device.load(*host);
    ASSERT_TRUE(inputBuffer) << inputBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({0}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();
    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *inputBuffer, host->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({0}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();
    auto values = read_f32_values(device, *outputBuffer);
    ASSERT_TRUE(values) << values.error();
    EXPECT_TRUE(values->empty());
}

TEST(CudaDeviceTest, InterpreterFallbackMatchesElementwiseJit) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""}, input);
    auto output = graph.addValue(tensor_type({4}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{{input, kir::BroadcastMode::None}},
        output,
        4,
        std::vector<kir::ScalarNode>{
            {0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            {1, kir::ScalarOp::Constant, sandy::core::DType::F32, 0, 1.25, {}},
            {2, kir::ScalarOp::Mul, sandy::core::DType::F32, 0, 0.0, {0, 1}},
            {3, kir::ScalarOp::Neg, sandy::core::DType::F32, 0, 0.0, {2}},
            {4, kir::ScalarOp::Tanh, sandy::core::DType::F32, 0, 0.0, {3}},
        });
    graph.setOutputs({output});
    auto host = make_f32_buffer(
        "x", sandy::core::Shape({4}), {-2.0f, -0.5f, 0.25f, 3.0f});

    auto runGraph = [&](sandy::device::CudaDevice& device) -> Result<std::vector<float>> {
        auto compiled = device.compile(graph);
        if (!compiled)
            return make_error(compiled.error());
        auto inputBuffer = device.load(*host);
        if (!inputBuffer)
            return make_error(inputBuffer.error());
        auto outputBuffer = device.alloc(
            sandy::core::TensorDesc({4}, sandy::core::DType::F32));
        if (!outputBuffer)
            return make_error(outputBuffer.error());
        std::vector<sandy::device::DeviceRunValue> inputs = {
            tensor_view(device, *inputBuffer, host->desc()),
        };
        std::vector<sandy::device::DeviceRunValue> outputs = {
            tensor_view(
                device, *outputBuffer,
                sandy::core::TensorDesc({4}, sandy::core::DType::F32)),
        };
        auto run = device.run(*compiled, op->id(), inputs, outputs);
        if (!run)
            return make_error(run.error());
        return read_f32_values(device, *outputBuffer);
    };

    sandy::device::CudaDevice jitDevice;
    auto jitValues = runGraph(jitDevice);
    ASSERT_TRUE(jitValues) << jitValues.error();
    EXPECT_EQ(jitDevice.jitCacheStats().entries, 1u);

    ScopedEnvironmentVariable disableJit("SANDY_CUDA_ELEMENTWISE_JIT", "0");
    sandy::device::CudaDevice interpreterDevice;
    auto interpreterValues = runGraph(interpreterDevice);
    ASSERT_TRUE(interpreterValues) << interpreterValues.error();
    EXPECT_EQ(interpreterDevice.jitCacheStats().entries, 0u);
    ASSERT_EQ(jitValues->size(), interpreterValues->size());
    for (size_t i = 0; i < jitValues->size(); ++i)
        EXPECT_FLOAT_EQ((*jitValues)[i], (*interpreterValues)[i]) << "at index " << i;
}

TEST(CudaDeviceTest, RunLayoutTransformMaterializesStridedF32View) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({3, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({3, 2}));
    auto* op = graph.addOp<kir::LayoutTransformOp>(
        kir::LayoutTransformKind::Contiguous,
        input,
        output,
        std::vector<int64_t>{});
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto host = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto inputBuffer = device.load(*host);
    ASSERT_TRUE(inputBuffer) << inputBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({3, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    sandy::device::TensorViewDesc transposedView;
    transposedView.desc = sandy::core::TensorDesc({3, 2}, sandy::core::DType::F32);
    transposedView.strides = {1, 3};

    std::vector<sandy::device::DeviceRunValue> inputs = {
        sandy::device::DeviceTensorView{*inputBuffer, transposedView},
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({3, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST(CudaDeviceTest, CompileAcceptsPagedTensorKernelInput) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({1, 1, sandy::core::Shape::kDynamic, 64}),
            sandy::core::DType::F32,
            2,
            16));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({1, 1, sandy::core::Shape::kDynamic, 64}));
    graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{input, kir::BroadcastMode::None},
        },
        output,
        0,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();
}

TEST(CudaDeviceTest, RunElementwiseReadsPagedTensorF32) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({2, sandy::core::Shape::kDynamic, 4}),
            sandy::core::DType::F32,
            1,
            2));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({2, sandy::core::Shape::kDynamic, 4}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{input, kir::BroadcastMode::None},
        },
        output,
        2,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Constant, sandy::core::DType::F32, 0, 1.0, {}},
            kir::ScalarNode{2, kir::ScalarOp::Add, sandy::core::DType::F32, 0, 0.0, {0, 1}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(
        sandy::core::Shape({2, sandy::core::Shape::kDynamic, 4}),
        sandy::core::DType::F32);
    poolDesc.growDim = 1;
    poolDesc.pageSize = 2;
    auto pool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(pool) << pool.error();
    auto paged = device.allocPaged(*pool, sandy::core::Shape({2, 0, 4}));
    ASSERT_TRUE(paged) << paged.error();

    std::vector<float> values(26, -1.0f);
    for (size_t i = 0; i < 24; i++)
        values[i + 2] = static_cast<float>(i);
    auto chunk = make_f32_buffer("chunk", sandy::core::Shape({26}), values);
    auto chunkBuffer = device.load(*chunk);
    ASSERT_TRUE(chunkBuffer) << chunkBuffer.error();
    auto chunkView = device.defaultView(
        sandy::core::TensorDesc({2, 3, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(chunkView) << chunkView.error();
    chunkView->storageOffset = 2;
    auto append = device.appendPaged(
        *paged,
        sandy::device::DeviceTensorView{*chunkBuffer, chunkView.take()});
    ASSERT_TRUE(append) << append.error();

    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({2, 3, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        paged_tensor_view(device, *paged),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    std::vector<float> expected(24);
    for (size_t i = 0; i < expected.size(); i++)
        expected[i] = static_cast<float>(i + 1);
    expect_f32_output_near(device, *outputBuffer, expected);

    EXPECT_TRUE(device.dealloc(*chunkBuffer));
    EXPECT_TRUE(device.deallocPaged(*paged));
    EXPECT_TRUE(device.destroyPagedPool(*pool));
}

TEST(CudaDeviceTest, RunElementwiseRefreshesPagedTensorMetaAtLaunch) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({2, sandy::core::Shape::kDynamic, 4}),
            sandy::core::DType::F32,
            1,
            2));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({2, sandy::core::Shape::kDynamic, 4}));
    auto* op = graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{input, kir::BroadcastMode::None},
        },
        output,
        0,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
        });
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(
        sandy::core::Shape({2, sandy::core::Shape::kDynamic, 4}),
        sandy::core::DType::F32);
    poolDesc.growDim = 1;
    poolDesc.pageSize = 2;
    auto pool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(pool) << pool.error();
    auto paged = device.allocPaged(*pool, sandy::core::Shape({2, 0, 4}));
    ASSERT_TRUE(paged) << paged.error();
    auto staleView = paged_tensor_view(device, *paged);

    auto first = make_f32_buffer(
        "first",
        sandy::core::Shape({2, 1, 4}),
        {0.0f, 1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f, 7.0f});
    auto second = make_f32_buffer(
        "second",
        sandy::core::Shape({2, 2, 4}),
        {8.0f, 9.0f, 10.0f, 11.0f,
         12.0f, 13.0f, 14.0f, 15.0f,
         16.0f, 17.0f, 18.0f, 19.0f,
         20.0f, 21.0f, 22.0f, 23.0f});
    ASSERT_TRUE(device.appendPaged(*paged, *first));
    ASSERT_TRUE(device.appendPaged(*paged, *second));

    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({2, 3, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {staleView};
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3, 4}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    std::vector<float> expected = {
        0.0f, 1.0f, 2.0f, 3.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    expect_f32_output_near(device, *outputBuffer, expected);

    EXPECT_TRUE(device.deallocPaged(*paged));
    EXPECT_TRUE(device.destroyPagedPool(*pool));
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
            6.0f, 15.0f,
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

TEST(CudaDeviceTest, RunGatherF32Rank1WeightReturnsScalarLookup) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto ids = graph.addValue(tensor_type({1, 1, 8}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        ids);
    auto table = graph.addValue(tensor_type({8}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        table);
    auto output = graph.addValue(tensor_type({1, 1, 8}));
    auto* op = graph.addOp<kir::GatherKernelOp>(ids, table, output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto idsHost = make_i64_buffer(
        "ids",
        sandy::core::Shape({1, 1, 8}),
        {7, 1, 3, 0, 2, 4, 5, 6});
    auto tableHost = make_f32_buffer(
        "table",
        sandy::core::Shape({8}),
        {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f});
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto tableBuffer = device.load(*tableHost);
    ASSERT_TRUE(tableBuffer) << tableBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({1, 1, 8}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *tableBuffer, tableHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 1, 8}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {80.0f, 20.0f, 40.0f, 10.0f, 30.0f, 50.0f, 60.0f, 70.0f});
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

TEST(CudaDeviceTest, RunMatMulAcceptsOffsetSliceViewWithSingletonLeadingAxis) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({1, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""}, lhs);
    auto rhs = graph.addValue(tensor_type({2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""}, rhs);
    auto output = graph.addValue(tensor_type({1, 2}));
    auto* op = graph.addOp<kir::MatMulKernelOp>(
        lhs, rhs, output, false, false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer(
        "lhs", sandy::core::Shape({1, 3, 2}),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto rhsHost = make_f32_buffer(
        "rhs", sandy::core::Shape({2, 2}),
        {1.0f, 0.0f, 0.0f, 1.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({1, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    sandy::device::TensorViewDesc slicedLhs;
    slicedLhs.desc = sandy::core::TensorDesc({1, 2}, sandy::core::DType::F32);
    slicedLhs.strides = {6, 1};
    slicedLhs.storageOffset = 4;
    std::vector<sandy::device::DeviceRunValue> inputs = {
        sandy::device::DeviceTensorView{*lhsBuffer, std::move(slicedLhs)},
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outputBuffer, {5.0f, 6.0f});
}

TEST(CudaDeviceTest, RunMatMulRejectsPagedOperand) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({sandy::core::Shape::kDynamic, 3}),
            sandy::core::DType::F32,
            0,
            2));
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

    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(
        sandy::core::Shape({sandy::core::Shape::kDynamic, 3}),
        sandy::core::DType::F32);
    poolDesc.growDim = 0;
    poolDesc.pageSize = 2;
    auto pool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(pool) << pool.error();
    auto paged = device.allocPaged(*pool, sandy::core::Shape({2, 3}));
    ASSERT_TRUE(paged) << paged.error();
    auto chunk = make_f32_buffer(
        "lhs",
        sandy::core::Shape({2, 3}),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ASSERT_TRUE(device.appendPaged(*paged, *chunk));

    auto rhsHost = make_f32_buffer(
        "rhs",
        sandy::core::Shape({3, 2}),
        {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        paged_tensor_view(device, *paged),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("matmul does not support paged tensor"), std::string::npos);

    EXPECT_TRUE(device.deallocPaged(*paged));
    EXPECT_TRUE(device.destroyPagedPool(*pool));
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

TEST(CudaDeviceTest, RunMoeMatMulF32GroupedExperts) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({5, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto expertOffsets = graph.addValue(tensor_type({4}, sandy::core::DType::I32));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        expertOffsets);
    auto weight = graph.addValue(tensor_type({3, 2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        weight);
    auto output = graph.addValue(tensor_type({5, 2}));
    auto* op = graph.addOp<kir::MoeMatMulKernelOp>(
        x,
        expertOffsets,
        weight,
        output,
        true);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({5, 3}),
        {
            1.0f, 0.0f, 2.0f,
            0.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            2.0f, 0.0f, 1.0f,
            0.0f, 2.0f, 3.0f,
        });
    auto offsetsHost = make_i32_buffer(
        "expert_offsets",
        sandy::core::Shape({4}),
        {0, 2, 2, 5});
    auto weightHost = make_f32_buffer(
        "weight",
        sandy::core::Shape({3, 2, 3}),
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            10.0f, 20.0f, 30.0f,
            40.0f, 50.0f, 60.0f,
            2.0f, 0.0f, 1.0f,
            0.0f, 3.0f, 1.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto offsetsBuffer = device.load(*offsetsHost);
    ASSERT_TRUE(offsetsBuffer) << offsetsBuffer.error();
    auto weightBuffer = device.load(*weightHost);
    ASSERT_TRUE(weightBuffer) << weightBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({5, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *offsetsBuffer, offsetsHost->desc()),
        tensor_view(device, *weightBuffer, weightHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({5, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            7.0f, 16.0f,
            5.0f, 11.0f,
            3.0f, 4.0f,
            5.0f, 1.0f,
            3.0f, 9.0f,
        });
}

TEST(CudaDeviceTest, RunMoeGatherF32Unbatched) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto ids = graph.addValue(tensor_type({2, 2}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        ids);
    auto weights = graph.addValue(tensor_type({2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        weights);
    auto packedX = graph.addValue(tensor_type({4, 2}));
    auto packedWeights = graph.addValue(tensor_type({4}));
    auto tokenIds = graph.addValue(tensor_type({4}, sandy::core::DType::I64));
    auto expertOffsets = graph.addValue(tensor_type({5}, sandy::core::DType::I64));
    auto* op = graph.addOp<kir::MoeGatherKernelOp>(
        x,
        ids,
        weights,
        packedX,
        packedWeights,
        tokenIds,
        expertOffsets,
        4,
        2);
    graph.setOutputs({packedX, packedWeights, tokenIds, expertOffsets});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 2}),
        {1.0f, 2.0f, 3.0f, 4.0f});
    auto idsHost = make_i64_buffer(
        "ids",
        sandy::core::Shape({2, 2}),
        {0, 1, 2, 3});
    auto weightsHost = make_f32_buffer(
        "weights",
        sandy::core::Shape({2, 2}),
        {0.1f, 0.2f, 0.3f, 0.4f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto weightsBuffer = device.load(*weightsHost);
    ASSERT_TRUE(weightsBuffer) << weightsBuffer.error();
    auto packedXBuffer = device.alloc(sandy::core::TensorDesc({4, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(packedXBuffer) << packedXBuffer.error();
    auto packedWeightsBuffer = device.alloc(sandy::core::TensorDesc({4}, sandy::core::DType::F32));
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.alloc(sandy::core::TensorDesc({4}, sandy::core::DType::I64));
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto offsetsBuffer = device.alloc(sandy::core::TensorDesc({5}, sandy::core::DType::I64));
    ASSERT_TRUE(offsetsBuffer) << offsetsBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *weightsBuffer, weightsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *packedXBuffer, sandy::core::TensorDesc({4, 2}, sandy::core::DType::F32)),
        tensor_view(device, *packedWeightsBuffer, sandy::core::TensorDesc({4}, sandy::core::DType::F32)),
        tensor_view(device, *tokenIdsBuffer, sandy::core::TensorDesc({4}, sandy::core::DType::I64)),
        tensor_view(device, *offsetsBuffer, sandy::core::TensorDesc({5}, sandy::core::DType::I64)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *packedXBuffer, {1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f});
    expect_f32_output(device, *packedWeightsBuffer, {0.1f, 0.2f, 0.3f, 0.4f});
    expect_i64_output(device, *tokenIdsBuffer, {0, 0, 1, 1});
    expect_i64_output(device, *offsetsBuffer, {0, 1, 2, 3, 4});
}

TEST(CudaDeviceTest, RunMoeGatherF32Batched) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto ids = graph.addValue(tensor_type({2, 2, 2}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        ids);
    auto weights = graph.addValue(tensor_type({2, 2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        weights);
    auto packedX = graph.addValue(tensor_type({2, 4, 2}));
    auto packedWeights = graph.addValue(tensor_type({2, 4}));
    auto tokenIds = graph.addValue(tensor_type({2, 4}, sandy::core::DType::I64));
    auto expertOffsets = graph.addValue(tensor_type({2, 5}, sandy::core::DType::I64));
    auto* op = graph.addOp<kir::MoeGatherKernelOp>(
        x,
        ids,
        weights,
        packedX,
        packedWeights,
        tokenIds,
        expertOffsets,
        4,
        2);
    graph.setOutputs({packedX, packedWeights, tokenIds, expertOffsets});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 2, 2}),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto idsHost = make_i64_buffer(
        "ids",
        sandy::core::Shape({2, 2, 2}),
        {
            0, 1,
            2, 3,
            3, 2,
            1, 0,
        });
    auto weightsHost = make_f32_buffer(
        "weights",
        sandy::core::Shape({2, 2, 2}),
        {
            0.1f, 0.2f,
            0.3f, 0.4f,
            0.5f, 0.6f,
            0.7f, 0.8f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto weightsBuffer = device.load(*weightsHost);
    ASSERT_TRUE(weightsBuffer) << weightsBuffer.error();
    auto packedXBuffer = device.alloc(sandy::core::TensorDesc({2, 4, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(packedXBuffer) << packedXBuffer.error();
    auto packedWeightsBuffer = device.alloc(sandy::core::TensorDesc({2, 4}, sandy::core::DType::F32));
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.alloc(sandy::core::TensorDesc({2, 4}, sandy::core::DType::I64));
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto offsetsBuffer = device.alloc(sandy::core::TensorDesc({2, 5}, sandy::core::DType::I64));
    ASSERT_TRUE(offsetsBuffer) << offsetsBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *weightsBuffer, weightsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *packedXBuffer, sandy::core::TensorDesc({2, 4, 2}, sandy::core::DType::F32)),
        tensor_view(device, *packedWeightsBuffer, sandy::core::TensorDesc({2, 4}, sandy::core::DType::F32)),
        tensor_view(device, *tokenIdsBuffer, sandy::core::TensorDesc({2, 4}, sandy::core::DType::I64)),
        tensor_view(device, *offsetsBuffer, sandy::core::TensorDesc({2, 5}, sandy::core::DType::I64)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *packedXBuffer,
        {
            1.0f, 2.0f,
            1.0f, 2.0f,
            3.0f, 4.0f,
            3.0f, 4.0f,
            7.0f, 8.0f,
            7.0f, 8.0f,
            5.0f, 6.0f,
            5.0f, 6.0f,
        });
    expect_f32_output(device, *packedWeightsBuffer, {0.1f, 0.2f, 0.3f, 0.4f, 0.8f, 0.7f, 0.6f, 0.5f});
    expect_i64_output(device, *tokenIdsBuffer, {0, 0, 1, 1, 1, 1, 0, 0});
    expect_i64_output(device, *offsetsBuffer, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
}

TEST(CudaDeviceTest, RunMoeGatherRejectsInvalidExpertId) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto ids = graph.addValue(tensor_type({1, 2}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        ids);
    auto weights = graph.addValue(tensor_type({1, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        weights);
    auto packedX = graph.addValue(tensor_type({2, 2}));
    auto packedWeights = graph.addValue(tensor_type({2}));
    auto tokenIds = graph.addValue(tensor_type({2}, sandy::core::DType::I64));
    auto expertOffsets = graph.addValue(tensor_type({3}, sandy::core::DType::I64));
    auto* op = graph.addOp<kir::MoeGatherKernelOp>(
        x,
        ids,
        weights,
        packedX,
        packedWeights,
        tokenIds,
        expertOffsets,
        2,
        2);
    graph.setOutputs({packedX, packedWeights, tokenIds, expertOffsets});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({1, 2}), {1.0f, 2.0f});
    auto idsHost = make_i64_buffer("ids", sandy::core::Shape({1, 2}), {0, 2});
    auto weightsHost = make_f32_buffer("weights", sandy::core::Shape({1, 2}), {0.1f, 0.2f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto idsBuffer = device.load(*idsHost);
    ASSERT_TRUE(idsBuffer) << idsBuffer.error();
    auto weightsBuffer = device.load(*weightsHost);
    ASSERT_TRUE(weightsBuffer) << weightsBuffer.error();
    auto packedXBuffer = device.alloc(sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(packedXBuffer) << packedXBuffer.error();
    auto packedWeightsBuffer = device.alloc(sandy::core::TensorDesc({2}, sandy::core::DType::F32));
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.alloc(sandy::core::TensorDesc({2}, sandy::core::DType::I64));
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto offsetsBuffer = device.alloc(sandy::core::TensorDesc({3}, sandy::core::DType::I64));
    ASSERT_TRUE(offsetsBuffer) << offsetsBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *idsBuffer, idsHost->desc()),
        tensor_view(device, *weightsBuffer, weightsHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *packedXBuffer, sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32)),
        tensor_view(device, *packedWeightsBuffer, sandy::core::TensorDesc({2}, sandy::core::DType::F32)),
        tensor_view(device, *tokenIdsBuffer, sandy::core::TensorDesc({2}, sandy::core::DType::I64)),
        tensor_view(device, *offsetsBuffer, sandy::core::TensorDesc({3}, sandy::core::DType::I64)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("expert id out of range"), std::string::npos);
}

TEST(CudaDeviceTest, RunMoeScatterSumF32Unbatched) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto packedOut = graph.addValue(tensor_type({4, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        packedOut);
    auto packedWeights = graph.addValue(tensor_type({4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        packedWeights);
    auto tokenIds = graph.addValue(tensor_type({4}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        tokenIds);
    auto reference = graph.addValue(tensor_type({2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 3, ""},
        reference);
    auto output = graph.addValue(tensor_type({2, 2}));
    auto* op = graph.addOp<kir::MoeScatterSumKernelOp>(
        packedOut,
        packedWeights,
        tokenIds,
        reference,
        output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto packedOutHost = make_f32_buffer(
        "packed_out",
        sandy::core::Shape({4, 2}),
        {
            10.0f, 100.0f,
            20.0f, 200.0f,
            30.0f, 300.0f,
            40.0f, 400.0f,
        });
    auto packedWeightsHost = make_f32_buffer(
        "packed_weights",
        sandy::core::Shape({4}),
        {0.5f, 1.0f, 0.25f, 0.75f});
    auto tokenIdsHost = make_i64_buffer(
        "token_ids",
        sandy::core::Shape({4}),
        {0, 1, 0, 1});
    auto referenceHost = make_f32_buffer(
        "reference",
        sandy::core::Shape({2, 2}),
        {9.0f, 9.0f, 9.0f, 9.0f});
    auto packedOutBuffer = device.load(*packedOutHost);
    ASSERT_TRUE(packedOutBuffer) << packedOutBuffer.error();
    auto packedWeightsBuffer = device.load(*packedWeightsHost);
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.load(*tokenIdsHost);
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto referenceBuffer = device.load(*referenceHost);
    ASSERT_TRUE(referenceBuffer) << referenceBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *packedOutBuffer, packedOutHost->desc()),
        tensor_view(device, *packedWeightsBuffer, packedWeightsHost->desc()),
        tensor_view(device, *tokenIdsBuffer, tokenIdsHost->desc()),
        tensor_view(device, *referenceBuffer, referenceHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            12.5f, 125.0f,
            50.0f, 500.0f,
        });
}

TEST(CudaDeviceTest, RunMoeScatterSumF32Batched) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto packedOut = graph.addValue(tensor_type({2, 4, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        packedOut);
    auto packedWeights = graph.addValue(tensor_type({2, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        packedWeights);
    auto tokenIds = graph.addValue(tensor_type({2, 4}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        tokenIds);
    auto reference = graph.addValue(tensor_type({2, 2, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 3, ""},
        reference);
    auto output = graph.addValue(tensor_type({2, 2, 2}));
    auto* op = graph.addOp<kir::MoeScatterSumKernelOp>(
        packedOut,
        packedWeights,
        tokenIds,
        reference,
        output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto packedOutHost = make_f32_buffer(
        "packed_out",
        sandy::core::Shape({2, 4, 2}),
        {
            1.0f, 10.0f,
            2.0f, 20.0f,
            3.0f, 30.0f,
            4.0f, 40.0f,
            5.0f, 50.0f,
            6.0f, 60.0f,
            7.0f, 70.0f,
            8.0f, 80.0f,
        });
    auto packedWeightsHost = make_f32_buffer(
        "packed_weights",
        sandy::core::Shape({2, 4}),
        {
            1.0f, 0.5f, 0.25f, 2.0f,
            0.5f, 1.0f, 1.5f, 0.25f,
        });
    auto tokenIdsHost = make_i64_buffer(
        "token_ids",
        sandy::core::Shape({2, 4}),
        {
            0, 1, 0, 1,
            1, 0, 1, 0,
        });
    auto referenceHost = make_f32_buffer(
        "reference",
        sandy::core::Shape({2, 2, 2}),
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    auto packedOutBuffer = device.load(*packedOutHost);
    ASSERT_TRUE(packedOutBuffer) << packedOutBuffer.error();
    auto packedWeightsBuffer = device.load(*packedWeightsHost);
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.load(*tokenIdsHost);
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto referenceBuffer = device.load(*referenceHost);
    ASSERT_TRUE(referenceBuffer) << referenceBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 2, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *packedOutBuffer, packedOutHost->desc()),
        tensor_view(device, *packedWeightsBuffer, packedWeightsHost->desc()),
        tensor_view(device, *tokenIdsBuffer, tokenIdsHost->desc()),
        tensor_view(device, *referenceBuffer, referenceHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({2, 2, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            1.75f, 17.5f,
            9.0f, 90.0f,
            8.0f, 80.0f,
            13.0f, 130.0f,
        });
}

TEST(CudaDeviceTest, RunMoeScatterSumBF16Unbatched) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto packedOut = graph.addValue(tensor_type({4, 2}, sandy::core::DType::BF16));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        packedOut);
    auto packedWeights = graph.addValue(tensor_type({4}, sandy::core::DType::BF16));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        packedWeights);
    auto tokenIds = graph.addValue(tensor_type({4}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        tokenIds);
    auto reference = graph.addValue(tensor_type({2, 2}, sandy::core::DType::BF16));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 3, ""},
        reference);
    auto output = graph.addValue(tensor_type({2, 2}, sandy::core::DType::BF16));
    auto* op = graph.addOp<kir::MoeScatterSumKernelOp>(
        packedOut,
        packedWeights,
        tokenIds,
        reference,
        output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto packedOutHost = make_bf16_buffer(
        "packed_out",
        sandy::core::Shape({4, 2}),
        {
            8.0f, 16.0f,
            4.0f, 12.0f,
            2.0f, 6.0f,
            10.0f, 14.0f,
        });
    auto packedWeightsHost = make_bf16_buffer(
        "packed_weights",
        sandy::core::Shape({4}),
        {0.5f, 1.0f, 0.25f, 0.5f});
    auto tokenIdsHost = make_i64_buffer(
        "token_ids",
        sandy::core::Shape({4}),
        {0, 1, 0, 1});
    auto referenceHost = make_bf16_buffer(
        "reference",
        sandy::core::Shape({2, 2}),
        {0.0f, 0.0f, 0.0f, 0.0f});
    auto packedOutBuffer = device.load(*packedOutHost);
    ASSERT_TRUE(packedOutBuffer) << packedOutBuffer.error();
    auto packedWeightsBuffer = device.load(*packedWeightsHost);
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.load(*tokenIdsHost);
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto referenceBuffer = device.load(*referenceHost);
    ASSERT_TRUE(referenceBuffer) << referenceBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 2}, sandy::core::DType::BF16));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *packedOutBuffer, packedOutHost->desc()),
        tensor_view(device, *packedWeightsBuffer, packedWeightsHost->desc()),
        tensor_view(device, *tokenIdsBuffer, tokenIdsHost->desc()),
        tensor_view(device, *referenceBuffer, referenceHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({2, 2}, sandy::core::DType::BF16)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_bf16_output_near(
        device,
        *outputBuffer,
        {
            4.5f, 9.5f,
            9.0f, 19.0f,
        });
}

TEST(CudaDeviceTest, RunMoeScatterSumRejectsInvalidTokenId) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto packedOut = graph.addValue(tensor_type({1, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        packedOut);
    auto packedWeights = graph.addValue(tensor_type({1}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        packedWeights);
    auto tokenIds = graph.addValue(tensor_type({1}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        tokenIds);
    auto reference = graph.addValue(tensor_type({1, 2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 3, ""},
        reference);
    auto output = graph.addValue(tensor_type({1, 2}));
    auto* op = graph.addOp<kir::MoeScatterSumKernelOp>(
        packedOut,
        packedWeights,
        tokenIds,
        reference,
        output);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto packedOutHost = make_f32_buffer("packed_out", sandy::core::Shape({1, 2}), {1.0f, 2.0f});
    auto packedWeightsHost = make_f32_buffer("packed_weights", sandy::core::Shape({1}), {1.0f});
    auto tokenIdsHost = make_i64_buffer("token_ids", sandy::core::Shape({1}), {1});
    auto referenceHost = make_f32_buffer("reference", sandy::core::Shape({1, 2}), {0.0f, 0.0f});
    auto packedOutBuffer = device.load(*packedOutHost);
    ASSERT_TRUE(packedOutBuffer) << packedOutBuffer.error();
    auto packedWeightsBuffer = device.load(*packedWeightsHost);
    ASSERT_TRUE(packedWeightsBuffer) << packedWeightsBuffer.error();
    auto tokenIdsBuffer = device.load(*tokenIdsHost);
    ASSERT_TRUE(tokenIdsBuffer) << tokenIdsBuffer.error();
    auto referenceBuffer = device.load(*referenceHost);
    ASSERT_TRUE(referenceBuffer) << referenceBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *packedOutBuffer, packedOutHost->desc()),
        tensor_view(device, *packedWeightsBuffer, packedWeightsHost->desc()),
        tensor_view(device, *tokenIdsBuffer, tokenIdsHost->desc()),
        tensor_view(device, *referenceBuffer, referenceHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(device, *outputBuffer, sandy::core::TensorDesc({1, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("token id out of range"), std::string::npos);
}

TEST(CudaDeviceTest, RunMoeMatMulF32BatchedGroupedExperts) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto expertOffsets = graph.addValue(tensor_type({2, 3}, sandy::core::DType::I32));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        expertOffsets);
    auto weight = graph.addValue(tensor_type({2, 2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        weight);
    auto output = graph.addValue(tensor_type({2, 3, 2}));
    auto* op = graph.addOp<kir::MoeMatMulKernelOp>(
        x,
        expertOffsets,
        weight,
        output,
        true);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3, 3}),
        {
            1.0f, 0.0f, 2.0f,
            0.0f, 1.0f, 1.0f,
            2.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f,
            2.0f, 0.0f, 1.0f,
            0.0f, 2.0f, 3.0f,
        });
    auto offsetsHost = make_i32_buffer(
        "expert_offsets",
        sandy::core::Shape({2, 3}),
        {
            0, 2, 3,
            0, 1, 3,
        });
    auto weightHost = make_f32_buffer(
        "weight",
        sandy::core::Shape({2, 2, 3}),
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            2.0f, 0.0f, 1.0f,
            0.0f, 3.0f, 1.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto offsetsBuffer = device.load(*offsetsHost);
    ASSERT_TRUE(offsetsBuffer) << offsetsBuffer.error();
    auto weightBuffer = device.load(*weightHost);
    ASSERT_TRUE(weightBuffer) << weightBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({2, 3, 2}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *offsetsBuffer, offsetsHost->desc()),
        tensor_view(device, *weightBuffer, weightHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3, 2}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            7.0f, 16.0f,
            5.0f, 11.0f,
            4.0f, 3.0f,
            6.0f, 15.0f,
            5.0f, 1.0f,
            3.0f, 9.0f,
        });
}

TEST(CudaDeviceTest, RunReductionSumF32LastDimKeepDims) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({2, 3, 1}));
    auto* op = graph.addOp<kir::ReductionKernelOp>(
        kir::ReduceOp::Sum,
        x,
        output,
        std::vector<int64_t>{-1},
        true);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            0.5f, 1.5f, 2.5f, 3.5f,
            -1.0f, 2.0f, -3.0f, 4.0f,
            10.0f, 0.0f, 1.0f, -2.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({2, 3, 1}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({2, 3, 1}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            10.0f,
            26.0f,
            8.0f,
            2.0f,
            9.0f,
            2.5f,
        });
}

TEST(CudaDeviceTest, RunReductionRejectsKeepDimsFalse) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3, 4}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({2, 3}));
    auto* op = graph.addOp<kir::ReductionKernelOp>(
        kir::ReduceOp::Sum,
        x,
        output,
        std::vector<int64_t>{-1},
        false);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 3, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            0.5f, 1.5f, 2.5f, 3.5f,
            -1.0f, 2.0f, -3.0f, 4.0f,
            10.0f, 0.0f, 1.0f, -2.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
        });
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
    EXPECT_NE(run.error().find("keepDims=true"), std::string::npos);
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

TEST(CudaDeviceTest, RunTopKF32LastDim) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 5}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto values = graph.addValue(tensor_type({2, 3}));
    auto indices = graph.addValue(tensor_type({2, 3}, sandy::core::DType::I64));
    auto* op = graph.addOp<kir::TopKKernelOp>(x, values, indices, 3, -1);
    graph.setOutputs({values, indices});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({2, 5}),
        {
            1.0f, 3.0f, 2.0f, 3.0f, 0.0f,
            -1.0f, 5.0f, 4.0f, 5.0f, 6.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto valuesBuffer = device.alloc(sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32));
    ASSERT_TRUE(valuesBuffer) << valuesBuffer.error();
    auto indicesBuffer = device.alloc(sandy::core::TensorDesc({2, 3}, sandy::core::DType::I64));
    ASSERT_TRUE(indicesBuffer) << indicesBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *valuesBuffer,
            sandy::core::TensorDesc({2, 3}, sandy::core::DType::F32)),
        tensor_view(
            device,
            *indicesBuffer,
            sandy::core::TensorDesc({2, 3}, sandy::core::DType::I64)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *valuesBuffer, {3.0f, 3.0f, 2.0f, 6.0f, 5.0f, 5.0f});
    expect_i64_output(device, *indicesBuffer, {1, 3, 2, 4, 1, 3});
}

TEST(CudaDeviceTest, RunTopKBF16LastDim) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 4}, sandy::core::DType::BF16));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto values = graph.addValue(tensor_type({1, 2}, sandy::core::DType::BF16));
    auto indices = graph.addValue(tensor_type({1, 2}, sandy::core::DType::I64));
    auto* op = graph.addOp<kir::TopKKernelOp>(x, values, indices, 2, -1);
    graph.setOutputs({values, indices});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_bf16_buffer("x", sandy::core::Shape({1, 4}), {0.25f, 2.0f, 1.0f, -1.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto valuesBuffer = device.alloc(sandy::core::TensorDesc({1, 2}, sandy::core::DType::BF16));
    ASSERT_TRUE(valuesBuffer) << valuesBuffer.error();
    auto indicesBuffer = device.alloc(sandy::core::TensorDesc({1, 2}, sandy::core::DType::I64));
    ASSERT_TRUE(indicesBuffer) << indicesBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *valuesBuffer,
            sandy::core::TensorDesc({1, 2}, sandy::core::DType::BF16)),
        tensor_view(
            device,
            *indicesBuffer,
            sandy::core::TensorDesc({1, 2}, sandy::core::DType::I64)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_bf16_output_near(device, *valuesBuffer, {2.0f, 1.0f});
    expect_i64_output(device, *indicesBuffer, {1, 2});
}

TEST(CudaDeviceTest, RunTopKF32LargeLastDimKRange) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    constexpr int64_t rows = 2;
    constexpr int64_t axis = 8193;
    std::vector<float> input(static_cast<size_t>(rows * axis));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < axis; ++column) {
            // Deliberate ties exercise the token-index tie-break across tiles.
            input[static_cast<size_t>(row * axis + column)] =
                static_cast<float>((column * 7919 + row * 17) % 997);
        }
    }

    for (int64_t k : {2, 32, 33, 64}) {
        SCOPED_TRACE("k=" + std::to_string(k));

        kir::Graph graph;
        auto x = graph.addValue(tensor_type({rows, axis}));
        graph.addOp<kir::InputOp>(
            kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
            x);
        auto values = graph.addValue(tensor_type({rows, k}));
        auto indices = graph.addValue(tensor_type({rows, k}, sandy::core::DType::I64));
        auto* op = graph.addOp<kir::TopKKernelOp>(x, values, indices, k, -1);
        graph.setOutputs({values, indices});

        sandy::device::CudaDevice device;
        auto compiled = device.compile(graph);
        ASSERT_TRUE(compiled) << compiled.error();
        auto xHost = make_f32_buffer("x", sandy::core::Shape({rows, axis}), input);
        auto xBuffer = device.load(*xHost);
        ASSERT_TRUE(xBuffer) << xBuffer.error();
        auto valuesBuffer = device.alloc(
            sandy::core::TensorDesc({rows, k}, sandy::core::DType::F32));
        ASSERT_TRUE(valuesBuffer) << valuesBuffer.error();
        auto indicesBuffer = device.alloc(
            sandy::core::TensorDesc({rows, k}, sandy::core::DType::I64));
        ASSERT_TRUE(indicesBuffer) << indicesBuffer.error();

        std::vector<sandy::device::DeviceRunValue> inputs = {
            tensor_view(device, *xBuffer, xHost->desc()),
        };
        std::vector<sandy::device::DeviceRunValue> outputs = {
            tensor_view(
                device,
                *valuesBuffer,
                sandy::core::TensorDesc({rows, k}, sandy::core::DType::F32)),
            tensor_view(
                device,
                *indicesBuffer,
                sandy::core::TensorDesc({rows, k}, sandy::core::DType::I64)),
        };
        auto run = device.run(*compiled, op->id(), inputs, outputs);
        ASSERT_TRUE(run) << run.error();

        auto actualValues = device.read(*valuesBuffer);
        ASSERT_TRUE(actualValues) << actualValues.error();
        auto actualIndices = device.read(*indicesBuffer);
        ASSERT_TRUE(actualIndices) << actualIndices.error();
        auto valueAccess = (*actualValues)->access();
        ASSERT_TRUE(valueAccess) << valueAccess.error();
        auto indexAccess = (*actualIndices)->access();
        ASSERT_TRUE(indexAccess) << indexAccess.error();

        for (int64_t row = 0; row < rows; ++row) {
            std::vector<std::pair<float, int64_t>> reference;
            reference.reserve(static_cast<size_t>(axis));
            for (int64_t column = 0; column < axis; ++column) {
                reference.emplace_back(
                    input[static_cast<size_t>(row * axis + column)],
                    column);
            }
            std::sort(
                reference.begin(),
                reference.end(),
                [](const auto& lhs, const auto& rhs) {
                    if (lhs.first != rhs.first)
                        return lhs.first > rhs.first;
                    return lhs.second < rhs.second;
                });
            for (int64_t rank = 0; rank < k; ++rank) {
                size_t outputIndex = static_cast<size_t>(row * k + rank);
                EXPECT_FLOAT_EQ(
                    read_f32((*valueAccess).data(), outputIndex),
                    reference[static_cast<size_t>(rank)].first)
                    << "row=" << row << " rank=" << rank;
                EXPECT_EQ(
                    read_i64((*indexAccess).data(), outputIndex),
                    reference[static_cast<size_t>(rank)].second)
                    << "row=" << row << " rank=" << rank;
            }
        }
    }
}

TEST(CudaDeviceTest, RunTopKRejectsNonLastDim) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto values = graph.addValue(tensor_type({1, 3}));
    auto indices = graph.addValue(tensor_type({1, 3}, sandy::core::DType::I64));
    auto* op = graph.addOp<kir::TopKKernelOp>(x, values, indices, 1, 0);
    graph.setOutputs({values, indices});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto valuesBuffer = device.alloc(sandy::core::TensorDesc({1, 3}, sandy::core::DType::F32));
    ASSERT_TRUE(valuesBuffer) << valuesBuffer.error();
    auto indicesBuffer = device.alloc(sandy::core::TensorDesc({1, 3}, sandy::core::DType::I64));
    ASSERT_TRUE(indicesBuffer) << indicesBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *valuesBuffer,
            sandy::core::TensorDesc({1, 3}, sandy::core::DType::F32)),
        tensor_view(
            device,
            *indicesBuffer,
            sandy::core::TensorDesc({1, 3}, sandy::core::DType::I64)),
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

TEST(CudaDeviceTest, RunAttentionReadsPagedKeyValueCacheF32) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    int64_t batch = 1;
    int64_t qHeads = 2;
    int64_t kvHeads = 1;
    int64_t tq = 1;
    int64_t tk = 3;
    int64_t headDim = 64;
    float scale = 0.125f;

    kir::Graph graph;
    auto q = graph.addValue(tensor_type({batch, qHeads, tq, headDim}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        q);
    auto k = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({batch, kvHeads, sandy::core::Shape::kDynamic, headDim}),
            sandy::core::DType::F32,
            2,
            2));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        k);
    auto v = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({batch, kvHeads, sandy::core::Shape::kDynamic, headDim}),
            sandy::core::DType::F32,
            2,
            2));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 2, ""},
        v);
    auto positions = graph.addValue(tensor_type({batch}, sandy::core::DType::I64));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 3, ""},
        positions);
    auto output = graph.addValue(tensor_type({batch, qHeads, tq, headDim}));
    auto* op = graph.addOp<kir::AttentionKernelOp>(
        q,
        k,
        v,
        positions,
        output,
        0,
        scale);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto qValues = make_pattern(
        static_cast<size_t>(batch * qHeads * tq * headDim),
        0.006f,
        0.01f);
    auto kValues = make_pattern(
        static_cast<size_t>(batch * kvHeads * tk * headDim),
        0.004f,
        -0.005f);
    auto vValues = make_pattern(
        static_cast<size_t>(batch * kvHeads * tk * headDim),
        0.005f,
        0.02f);

    auto qHost = make_f32_buffer("q", sandy::core::Shape({batch, qHeads, tq, headDim}), qValues);
    auto qBuffer = device.load(*qHost);
    ASSERT_TRUE(qBuffer) << qBuffer.error();

    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(
        sandy::core::Shape({batch, kvHeads, sandy::core::Shape::kDynamic, headDim}),
        sandy::core::DType::F32);
    poolDesc.growDim = 2;
    poolDesc.pageSize = 2;
    auto kPool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(kPool) << kPool.error();
    auto vPool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(vPool) << vPool.error();
    auto kPaged = device.allocPaged(*kPool, sandy::core::Shape({batch, kvHeads, 0, headDim}));
    ASSERT_TRUE(kPaged) << kPaged.error();
    auto vPaged = device.allocPaged(*vPool, sandy::core::Shape({batch, kvHeads, 0, headDim}));
    ASSERT_TRUE(vPaged) << vPaged.error();

    auto kChunk = make_f32_buffer(
        "k",
        sandy::core::Shape({batch, kvHeads, tk, headDim}),
        kValues);
    auto vChunk = make_f32_buffer(
        "v",
        sandy::core::Shape({batch, kvHeads, tk, headDim}),
        vValues);
    ASSERT_TRUE(device.appendPaged(*kPaged, *kChunk));
    ASSERT_TRUE(device.appendPaged(*vPaged, *vChunk));

    auto positionHost = make_i64_buffer("positions", sandy::core::Shape({batch}), {tk - 1});
    auto positionBuffer = device.load(*positionHost);
    ASSERT_TRUE(positionBuffer) << positionBuffer.error();
    auto outputBuffer = device.alloc(
        sandy::core::TensorDesc({batch, qHeads, tq, headDim}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *qBuffer, qHost->desc()),
        paged_tensor_view(device, *kPaged),
        paged_tensor_view(device, *vPaged),
        tensor_view(device, *positionBuffer, positionHost->desc()),
    };
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
        0,
        scale,
        std::vector<int64_t>{tk - 1});
    ASSERT_TRUE(expected) << expected.error();
    expect_f32_output_near(device, *outputBuffer, expected.take());

    EXPECT_TRUE(device.deallocPaged(*kPaged));
    EXPECT_TRUE(device.deallocPaged(*vPaged));
    EXPECT_TRUE(device.destroyPagedPool(*kPool));
    EXPECT_TRUE(device.destroyPagedPool(*vPool));
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

TEST(CudaDeviceTest, RunRoPEF32PartialRotaryDimCopiesTail) {
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
            1.0f, 2.0f, 3.0f, 4.0f,
            std::cos(1.0f), std::sin(1.0f), 5.0f, 6.0f,
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

TEST(CudaDeviceTest, RunRoPEF32SplitHalfPartialRotaryDimCopiesTail) {
    if (auto reason = cuda_device_skip_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto x = graph.addValue(tensor_type({1, 2, 8}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        x);
    auto output = graph.addValue(tensor_type({1, 2, 8}));
    auto* op = graph.addOp<kir::RoPEKernelOp>(x, output, 10000.0, 4, true);
    graph.setOutputs({output});

    sandy::device::CudaDevice device;
    auto compiled = device.compile(graph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer(
        "x",
        sandy::core::Shape({1, 2, 8}),
        {
            1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f,
            1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f,
        });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outputBuffer = device.alloc(sandy::core::TensorDesc({1, 2, 8}, sandy::core::DType::F32));
    ASSERT_TRUE(outputBuffer) << outputBuffer.error();

    std::vector<sandy::device::DeviceRunValue> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceRunValue> outputs = {
        tensor_view(
            device,
            *outputBuffer,
            sandy::core::TensorDesc({1, 2, 8}, sandy::core::DType::F32)),
    };
    auto run = device.run(*compiled, op->id(), inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(
        device,
        *outputBuffer,
        {
            1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f,
            1.0f * std::cos(1.0f) - 3.0f * std::sin(1.0f),
            2.0f * std::cos(0.1f) - 4.0f * std::sin(0.1f),
            10.0f,
            20.0f,
            1.0f * std::sin(1.0f) + 3.0f * std::cos(1.0f),
            2.0f * std::sin(0.1f) + 4.0f * std::cos(0.1f),
            30.0f,
            40.0f,
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
