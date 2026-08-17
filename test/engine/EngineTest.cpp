#include "CpuDevice.h"
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
#include <string>
#include <unordered_map>
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

struct TestTensor {
    sandy::core::TensorDesc desc;
    std::vector<uint8_t> data;
};

class CapturedTensor {
public:
    CapturedTensor(sandy::core::TensorDesc desc, std::vector<uint8_t> data)
        : desc_(std::move(desc)), data_(std::move(data)) {}

    const sandy::core::TensorDesc& desc() const { return desc_; }
    std::span<const uint8_t> data() const { return data_; }

private:
    sandy::core::TensorDesc desc_;
    std::vector<uint8_t> data_;
};

sandy::engine::Engine make_cpu_engine() {
    std::vector<std::unique_ptr<sandy::device::Device>> devices;
    devices.push_back(std::make_unique<sandy::device::CpuDevice>());
    return sandy::engine::Engine(std::move(devices));
}

std::unordered_map<std::string, std::shared_ptr<CapturedTensor>> outputs_to_map(
        const std::vector<sandy::engine::TensorBufferPtr>& outputs) {
    std::unordered_map<std::string, std::shared_ptr<CapturedTensor>> map;
    for (size_t index = 0; index < outputs.size(); index++) {
        auto accessResult = outputs[index]->access();
        if (!accessResult)
            return {};
        auto access = accessResult.take();
        auto data = access.data();
        map["output" + std::to_string(index)] = std::make_shared<CapturedTensor>(
            access.desc(),
            std::vector<uint8_t>(data.begin(), data.end()));
    }
    return map;
}

Result<sandy::core::TensorRef> tensor_ref(
        std::span<const uint8_t> data,
        sandy::core::TensorDesc desc) {
    return sandy::core::make_tensor_ref(std::move(desc), data);
}

Result<sandy::core::MutableTensorRef> mutable_tensor_ref(TestTensor& tensor) {
    return sandy::core::make_mutable_tensor_ref(tensor.desc, tensor.data);
}

Result<TestTensor> make_output(sandy::core::TensorDesc desc) {
    int64_t numel = desc.shape.numel();
    if (numel < 0)
        return make_error("test output must have static shape");
    TestTensor out;
    out.desc = std::move(desc);
    out.data.resize(static_cast<size_t>(numel) * sandy::core::dtype_size(out.desc.dtype));
    return out;
}

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

std::vector<uint8_t> f32_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    for (size_t index = 0; index < values.size(); index++) {
        float value = values[index];
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
    }
    return bytes;
}

std::vector<uint8_t> bf16_bytes(std::initializer_list<float> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(sandy::core::BFloat16));
    size_t index = 0;
    for (float value : values) {
        auto bf16 = sandy::core::bfloat16_from_float(value);
        std::memcpy(
            bytes.data() + index * sizeof(sandy::core::BFloat16),
            &bf16,
            sizeof(sandy::core::BFloat16));
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

float read_bf16(std::span<const uint8_t> bytes, size_t index) {
    sandy::core::BFloat16 value;
    std::memcpy(
        &value,
        bytes.data() + index * sizeof(sandy::core::BFloat16),
        sizeof(sandy::core::BFloat16));
    return sandy::core::bfloat16_to_float(value);
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

Result<TestTensor> linear_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        std::span<const uint8_t> weight,
        sandy::core::TensorDesc weightDesc,
        std::span<const uint8_t> bias,
        sandy::core::TensorDesc biasDesc) {
    auto outDims = xDesc.shape.dims();
    outDims.back() = weightDesc.shape.dim(0);
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto weightRef = tensor_ref(weight, weightDesc);
    if (!weightRef) return make_error(weightRef.error());
    auto biasRef = tensor_ref(bias, biasDesc);
    if (!biasRef) return make_error(biasRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::linear(*xRef, *weightRef, *biasRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> unary_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        Result<void> (*op)(sandy::core::TensorRef, sandy::core::MutableTensorRef)) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = op(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> binary_calc(
        std::span<const uint8_t> lhs,
        sandy::core::TensorDesc lhsDesc,
        std::span<const uint8_t> rhs,
        sandy::core::TensorDesc rhsDesc,
        Result<void> (*op)(
            sandy::core::TensorRef,
            sandy::core::TensorRef,
            sandy::core::MutableTensorRef)) {
    auto shape = sandy::core::broadcast_shape(lhsDesc.shape, rhsDesc.shape);
    if (!shape) return make_error(shape.error());
    auto out = make_output(sandy::core::TensorDesc(shape.take(), lhsDesc.dtype));
    if (!out) return make_error(out.error());
    auto lhsRef = tensor_ref(lhs, lhsDesc);
    if (!lhsRef) return make_error(lhsRef.error());
    auto rhsRef = tensor_ref(rhs, rhsDesc);
    if (!rhsRef) return make_error(rhsRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = op(*lhsRef, *rhsRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> relu_calc(std::span<const uint8_t> x, sandy::core::TensorDesc xDesc) {
    return unary_calc(x, std::move(xDesc), sandy::core::relu);
}

Result<TestTensor> sqrt_calc(std::span<const uint8_t> x, sandy::core::TensorDesc xDesc) {
    return unary_calc(x, std::move(xDesc), sandy::core::sqrt);
}

Result<TestTensor> tanh_calc(std::span<const uint8_t> x, sandy::core::TensorDesc xDesc) {
    return unary_calc(x, std::move(xDesc), sandy::core::tanh);
}

Result<TestTensor> add_calc(
        std::span<const uint8_t> lhs,
        sandy::core::TensorDesc lhsDesc,
        std::span<const uint8_t> rhs,
        sandy::core::TensorDesc rhsDesc) {
    return binary_calc(lhs, std::move(lhsDesc), rhs, std::move(rhsDesc), sandy::core::add);
}

Result<TestTensor> mul_calc(
        std::span<const uint8_t> lhs,
        sandy::core::TensorDesc lhsDesc,
        std::span<const uint8_t> rhs,
        sandy::core::TensorDesc rhsDesc) {
    return binary_calc(lhs, std::move(lhsDesc), rhs, std::move(rhsDesc), sandy::core::mul);
}

Result<TestTensor> matmul_calc(
        std::span<const uint8_t> lhs,
        sandy::core::TensorDesc lhsDesc,
        std::span<const uint8_t> rhs,
        sandy::core::TensorDesc rhsDesc,
        bool transposeLhs = false,
        bool transposeRhs = false) {
    int lhsRank = lhsDesc.shape.rank();
    int rhsRank = rhsDesc.shape.rank();
    auto lhsDims = lhsDesc.shape.dims();
    auto rhsDims = rhsDesc.shape.dims();
    sandy::core::Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
    sandy::core::Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
    auto batch = sandy::core::matmul_batch_shape(lhsBatch, rhsBatch);
    if (!batch) return make_error(batch.error());
    auto outDims = batch.take().dims();
    outDims.push_back(lhsDesc.shape.dim(lhsRank - (transposeLhs ? 1 : 2)));
    outDims.push_back(rhsDesc.shape.dim(rhsRank - (transposeRhs ? 2 : 1)));
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), lhsDesc.dtype));
    if (!out) return make_error(out.error());
    auto lhsRef = tensor_ref(lhs, lhsDesc);
    if (!lhsRef) return make_error(lhsRef.error());
    auto rhsRef = tensor_ref(rhs, rhsDesc);
    if (!rhsRef) return make_error(rhsRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::matmul(*lhsRef, *rhsRef, transposeLhs, transposeRhs, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> transpose_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc) {
    auto outDims = xDesc.shape.dims();
    if (outDims.size() >= 2)
        std::swap(outDims[outDims.size() - 1], outDims[outDims.size() - 2]);
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::transpose(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> reshape_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        sandy::core::Shape shape) {
    auto inferred = sandy::core::infer_reshape_shape(xDesc.shape, std::move(shape));
    if (!inferred) return make_error(inferred.error());
    auto out = make_output(sandy::core::TensorDesc(inferred.take(), xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::reshape(*xRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> permute_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        std::span<const int64_t> dims) {
    std::vector<int64_t> outDims;
    outDims.reserve(dims.size());
    for (int64_t dim : dims)
        outDims.push_back(xDesc.shape.dim(static_cast<int>(dim)));
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::permute(*xRef, dims, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> sliding_query_key_score_calc(
        std::span<const uint8_t> q,
        sandy::core::TensorDesc qDesc,
        std::span<const uint8_t> k,
        sandy::core::TensorDesc kDesc,
        int64_t window) {
    int rank = qDesc.shape.rank();
    std::vector<int64_t> outDims;
    if (rank == 4)
        outDims.push_back(qDesc.shape.dim(0));
    outDims.push_back(qDesc.shape.dim(rank - 3));
    outDims.push_back(qDesc.shape.dim(rank - 2));
    outDims.push_back(kDesc.shape.dim(rank - 2));
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), qDesc.dtype));
    if (!out) return make_error(out.error());
    auto qRef = tensor_ref(q, qDesc);
    if (!qRef) return make_error(qRef.error());
    auto kRef = tensor_ref(k, kDesc);
    if (!kRef) return make_error(kRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::sliding_query_key_score(*qRef, *kRef, window, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> sliding_query_key_score_calc(
        std::span<const uint8_t> q,
        sandy::core::TensorDesc qDesc,
        std::span<const uint8_t> k,
        sandy::core::TensorDesc kDesc,
        std::span<const uint8_t> positionIds,
        sandy::core::TensorDesc positionDesc,
        int64_t window) {
    int rank = qDesc.shape.rank();
    std::vector<int64_t> outDims;
    if (rank == 4)
        outDims.push_back(qDesc.shape.dim(0));
    outDims.push_back(qDesc.shape.dim(rank - 3));
    outDims.push_back(qDesc.shape.dim(rank - 2));
    outDims.push_back(kDesc.shape.dim(rank - 2));
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), qDesc.dtype));
    if (!out) return make_error(out.error());
    auto qRef = tensor_ref(q, qDesc);
    if (!qRef) return make_error(qRef.error());
    auto kRef = tensor_ref(k, kDesc);
    if (!kRef) return make_error(kRef.error());
    auto positionRef = tensor_ref(positionIds, positionDesc);
    if (!positionRef) return make_error(positionRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::sliding_query_key_score(
        *qRef,
        *kRef,
        *positionRef,
        window,
        -1.0f,
        *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> softmax_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        int64_t dim) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::softmax(*xRef, dim, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> embedding_calc(
        std::span<const uint8_t> ids,
        sandy::core::TensorDesc idsDesc,
        std::span<const uint8_t> weight,
        sandy::core::TensorDesc weightDesc) {
    auto outDims = idsDesc.shape.dims();
    outDims.push_back(weightDesc.shape.dim(1));
    auto out = make_output(sandy::core::TensorDesc(sandy::core::Shape(outDims), weightDesc.dtype));
    if (!out) return make_error(out.error());
    auto idsRef = tensor_ref(ids, idsDesc);
    if (!idsRef) return make_error(idsRef.error());
    auto weightRef = tensor_ref(weight, weightDesc);
    if (!weightRef) return make_error(weightRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::embedding(*idsRef, *weightRef, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> rope_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        float theta) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::rope(*xRef, theta, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> rope_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        float theta,
        int64_t rotaryDim) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::rope(*xRef, theta, rotaryDim, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> rope_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        std::span<const uint8_t> positionIds,
        sandy::core::TensorDesc positionDesc,
        float theta,
        int64_t rotaryDim,
        bool splitHalf) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto positionRef = tensor_ref(positionIds, positionDesc);
    if (!positionRef) return make_error(positionRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::rope(
        *xRef,
        *positionRef,
        theta,
        rotaryDim,
        splitHalf,
        *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> rms_norm_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        std::span<const uint8_t> weight,
        sandy::core::TensorDesc weightDesc,
        float epsilon) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto weightRef = tensor_ref(weight, weightDesc);
    if (!weightRef) return make_error(weightRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::rms_norm(*xRef, *weightRef, epsilon, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> rms_norm_unscaled_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        float epsilon) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::rms_norm(*xRef, epsilon, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
}

Result<TestTensor> layer_norm_calc(
        std::span<const uint8_t> x,
        sandy::core::TensorDesc xDesc,
        std::span<const uint8_t> weight,
        sandy::core::TensorDesc weightDesc,
        std::span<const uint8_t> bias,
        sandy::core::TensorDesc biasDesc,
        float epsilon) {
    auto out = make_output(sandy::core::TensorDesc(xDesc.shape, xDesc.dtype));
    if (!out) return make_error(out.error());
    auto xRef = tensor_ref(x, xDesc);
    if (!xRef) return make_error(xRef.error());
    auto weightRef = tensor_ref(weight, weightDesc);
    if (!weightRef) return make_error(weightRef.error());
    auto biasRef = tensor_ref(bias, biasDesc);
    if (!biasRef) return make_error(biasRef.error());
    auto outRef = mutable_tensor_ref(*out);
    if (!outRef) return make_error(outRef.error());
    auto result = sandy::core::layer_norm(*xRef, *weightRef, *biasRef, epsilon, *outRef);
    if (!result) return make_error(result.error());
    return out.take();
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

TEST(EngineTest, CompileAndRunWithCpuDevice) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 1}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({1, 1}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({1}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("x", sandy::core::Shape({1, 1}), {1.0f}));

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

    auto result = linear_calc(
        x, sandy::core::TensorDesc(sandy::core::Shape({1, 2}), sandy::core::DType::F32),
        w, sandy::core::TensorDesc(sandy::core::Shape({2, 2}), sandy::core::DType::F32),
        b, sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 2}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 18.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 25.0f);
}

TEST(TensorCalcTest, LinearF32Rank3FlattensLeadingDims) {
    auto x = f32_bytes({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    });
    auto w = f32_bytes({
        10.0f, 100.0f,
        -1.0f, 1.0f,
        2.0f, 3.0f,
    });
    auto b = f32_bytes({1.0f, 2.0f, 3.0f});

    auto result = linear_calc(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 2, 2}), sandy::core::DType::F32),
        w, sandy::core::TensorDesc(sandy::core::Shape({3, 2}), sandy::core::DType::F32),
        b, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2, 3}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 211.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 11.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 9), 871.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 10), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 11), 41.0f);
}

TEST(TensorCalcTest, ReLUF32) {
    auto x = f32_bytes({-2.0f, 0.5f, 3.0f});

    auto result = relu_calc(
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

TEST(ShapeUtilTest, MatMulBatchShapeAllowsGroupedHeads) {
    auto result = sandy::core::matmul_batch_shape(
        sandy::core::Shape({1, 8}),
        sandy::core::Shape({1, 2}));

    ASSERT_TRUE(result) << result.error();
    EXPECT_EQ(result.take(), sandy::core::Shape({1, 8}));
}

TEST(TensorCalcTest, AddF32BroadcastsRightAligned) {
    auto lhs = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto rhs = f32_bytes({10.0f, 20.0f, 30.0f});

    auto result = add_calc(
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

    auto result = mul_calc(
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

    auto result = sqrt_calc(
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

    auto result = tanh_calc(
        x, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({3}));
    EXPECT_NEAR(read_f32(out.data, 0), std::tanh(-1.0f), 1.0e-6f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 0.0f);
    EXPECT_NEAR(read_f32(out.data, 2), std::tanh(1.0f), 1.0e-6f);
}

TEST(TensorCalcTest, BFloat16ConvertsThroughFloat) {
    auto one = sandy::core::bfloat16_from_float(1.0f);
    EXPECT_EQ(sandy::core::bfloat16_bits(one), 0x3f80);
    EXPECT_FLOAT_EQ(sandy::core::bfloat16_to_float(one), 1.0f);

    auto third = sandy::core::bfloat16_from_float(1.0f / 3.0f);
    EXPECT_NEAR(sandy::core::bfloat16_to_float(third), 1.0f / 3.0f, 0.002f);
}

TEST(TensorCalcTest, BFloat16RefsLoadAndStoreComputeValues) {
    TestTensor tensor;
    tensor.desc = sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::BF16);
    tensor.data.resize(2 * sizeof(sandy::core::BFloat16));
    auto mutableRef = mutable_tensor_ref(tensor);
    ASSERT_TRUE(mutableRef) << mutableRef.error();
    mutableRef->store_float(0, 1.0f);
    mutableRef->store_float(1, 1.0f / 3.0f);

    auto ref = tensor_ref(tensor.data, tensor.desc);
    ASSERT_TRUE(ref) << ref.error();
    EXPECT_FLOAT_EQ(ref->load_float(0), 1.0f);
    EXPECT_NEAR(ref->load_float(1), 1.0f / 3.0f, 0.002f);

    sandy::core::BFloat16 raw = sandy::core::bfloat16_from_bits(0);
    std::memcpy(&raw, tensor.data.data(), sizeof(raw));
    EXPECT_EQ(sandy::core::bfloat16_bits(raw), 0x3f80);
}

TEST(TensorCalcTest, BFloat16ElementwiseUsesTensorRefStorageAccessors) {
    auto lhs = bf16_bytes({1.0f, 2.0f, 3.0f});
    auto rhs = bf16_bytes({10.0f, 20.0f, 30.0f});

    auto result = add_calc(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::BF16),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::BF16));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.dtype, sandy::core::DType::BF16);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 0), 11.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 1), 22.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 2), 33.0f);
}

TEST(TensorCalcTest, BFloat16ElementwiseAcceptsF32Scalar) {
    auto lhs = bf16_bytes({1.0f, 2.0f, 3.0f});
    auto rhs = f32_bytes({2.0f});

    auto result = mul_calc(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({3}), sandy::core::DType::BF16),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.dtype, sandy::core::DType::BF16);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 0), 2.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 1), 4.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 2), 6.0f);
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

    auto result = matmul_calc(
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

TEST(TensorCalcTest, MatMulF32SupportsTransposedRhsFlag) {
    auto lhs = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto rhs = f32_bytes({
        7.0f, 9.0f, 11.0f,
        8.0f, 10.0f, 12.0f,
    });

    auto result = matmul_calc(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32),
        false,
        true);

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

    auto result = matmul_calc(
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

TEST(TensorCalcTest, MatMulF32SupportsGroupedBatchHeads) {
    auto lhs = f32_bytes({
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,

        0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f,

        1.0f, 2.0f, 0.0f,
        0.0f, 1.0f, 1.0f,

        2.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
    });
    auto rhs = f32_bytes({
        10.0f, 100.0f,
        20.0f, 200.0f,
        30.0f, 300.0f,

        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
    });

    auto result = matmul_calc(
        lhs, sandy::core::TensorDesc(sandy::core::Shape({1, 4, 2, 3}), sandy::core::DType::F32),
        rhs, sandy::core::TensorDesc(sandy::core::Shape({1, 2, 3, 2}), sandy::core::DType::F32));

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 4, 2, 2}));

    std::vector<float> expected = {
        10.0f, 100.0f,
        20.0f, 200.0f,
        30.0f, 300.0f,
        30.0f, 300.0f,
        5.0f, 50.0f,
        5.0f, 50.0f,
        5.0f, 50.0f,
        4.0f, 40.0f,
    };
    for (size_t i = 0; i < expected.size(); i++)
        EXPECT_FLOAT_EQ(read_f32(out.data, i), expected[i]);
}

TEST(TensorCalcTest, TransposeF32Requires2D) {
    auto x = f32_bytes({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });

    auto result = transpose_calc(
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

    auto rank3 = transpose_calc(
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

    auto result = reshape_calc(
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

    auto result = reshape_calc(
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

    auto result = permute_calc(
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

    auto result = sliding_query_key_score_calc(
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

    auto result = sliding_query_key_score_calc(
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

    auto result = sliding_query_key_score_calc(
        q, sandy::core::TensorDesc(sandy::core::Shape({1, 2, 1, 2}), sandy::core::DType::F32),
        k, sandy::core::TensorDesc(sandy::core::Shape({1, 1, 2, 1}), sandy::core::DType::F32),
        0);

    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("head dimension mismatch"), std::string::npos);

    auto validK = f32_bytes({
        10.0f, 1.0f,
        20.0f, 2.0f,
    });
    result = sliding_query_key_score_calc(
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

TEST(TensorCalcTest, SlidingQueryKeyScoreUsesRuntimePositionIdsForCausalMask) {
    auto q = f32_bytes({1.0f});
    auto k = f32_bytes({10.0f, 20.0f, 30.0f, 40.0f});
    auto position = i64_bytes({2});

    auto result = sliding_query_key_score_calc(
        q, sandy::core::TensorDesc(sandy::core::Shape({1, 1, 1, 1}), sandy::core::DType::F32),
        k, sandy::core::TensorDesc(sandy::core::Shape({1, 1, 4, 1}), sandy::core::DType::F32),
        position, sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::I64),
        0);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 1, 1, 4}));
    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 10.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 20.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 30.0f);
    EXPECT_TRUE(std::isinf(read_f32(out.data, 3)) && read_f32(out.data, 3) < 0.0f);
}

TEST(TensorCalcTest, SoftmaxF32LastDim) {
    auto x = f32_bytes({
        1.0f, 1.0f,
        1.0f, 2.0f,
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    });

    auto result = softmax_calc(
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

TEST(TensorCalcTest, SoftmaxAllowsZeroElementAxis) {
    std::vector<uint8_t> x;
    auto result = softmax_calc(
        x, sandy::core::TensorDesc(sandy::core::Shape({1, 8, 1, 0}), sandy::core::DType::F32),
        -1);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 8, 1, 0}));
    EXPECT_TRUE(out.data.empty());
}

TEST(TensorCalcTest, SoftmaxF32MiddleDim) {
    auto x = f32_bytes({
        1.0f, 10.0f,
        1.0f, 20.0f,
        1.0f, 30.0f,
    });

    auto result = softmax_calc(
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

    auto result = embedding_calc(
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

    auto result = embedding_calc(
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

TEST(TensorCalcTest, TensorRefAllowsZeroElementStaticShape) {
    std::vector<uint8_t> bytes;
    auto ref = sandy::core::make_tensor_ref(
        sandy::core::TensorDesc(
            sandy::core::Shape({1, 1, 0, 256}),
            sandy::core::DType::BF16),
        bytes);
    ASSERT_TRUE(ref) << ref.error();
}

TEST(TensorCalcTest, RoPEF32AppliesToLastDimForArbitraryRank) {
    std::vector<float> values(2 * 2 * 3 * 4, 0.0f);
    auto setVector = [&](size_t vector, std::initializer_list<float> row) {
        size_t index = vector * 4;
        for (float value : row)
            values[index++] = value;
    };
    setVector(0, {1.0f, 2.0f, 3.0f, 4.0f});
    setVector(1, {1.0f, 0.0f, 0.0f, 1.0f});
    setVector(2, {0.0f, 1.0f, 1.0f, 0.0f});
    setVector(7, {2.0f, 0.0f, 0.0f, 2.0f});

    auto result = rope_calc(
        f32_bytes(values),
        sandy::core::TensorDesc(sandy::core::Shape({2, 2, 3, 4}), sandy::core::DType::F32),
        10000.0f);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2, 3, 4}));

    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 4.0f);

    EXPECT_NEAR(read_f32(out.data, 4), std::cos(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 5), std::sin(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 6), -std::sin(0.01f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 7), std::cos(0.01f), 1.0e-6f);

    EXPECT_NEAR(read_f32(out.data, 8), -std::sin(2.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 9), std::cos(2.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 10), std::cos(0.02f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 11), std::sin(0.02f), 1.0e-6f);

    EXPECT_NEAR(read_f32(out.data, 28), 2.0f * std::cos(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 29), 2.0f * std::sin(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 30), -2.0f * std::sin(0.01f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 31), 2.0f * std::cos(0.01f), 1.0e-6f);
}

TEST(TensorCalcTest, RoPEF32AppliesPartialRotaryDimAndCopiesTail) {
    auto result = rope_calc(
        f32_bytes({
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 0.0f, 5.0f, 6.0f,
        }),
        sandy::core::TensorDesc(sandy::core::Shape({1, 2, 4}), sandy::core::DType::F32),
        10000.0f,
        2);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 2, 4}));

    EXPECT_FLOAT_EQ(read_f32(out.data, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 1), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 3), 4.0f);
    EXPECT_NEAR(read_f32(out.data, 4), std::cos(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 5), std::sin(1.0f), 1.0e-6f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 6), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(out.data, 7), 6.0f);
}

TEST(TensorCalcTest, RoPEF32UsesRuntimePositionIds) {
    auto result = rope_calc(
        f32_bytes({1.0f, 0.0f, 0.0f, 1.0f}),
        sandy::core::TensorDesc(sandy::core::Shape({1, 1, 4}), sandy::core::DType::F32),
        i64_bytes({3}),
        sandy::core::TensorDesc(sandy::core::Shape({1}), sandy::core::DType::I64),
        10000.0f,
        -1,
        false);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 1, 4}));
    EXPECT_NEAR(read_f32(out.data, 0), std::cos(3.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 1), std::sin(3.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 2), -std::sin(0.03f), 1.0e-6f);
    EXPECT_NEAR(read_f32(out.data, 3), std::cos(0.03f), 1.0e-6f);
}

TEST(TensorCalcTest, RoPEBF16UsesTensorRefStorageAccessors) {
    auto result = rope_calc(
        bf16_bytes({
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
        }),
        sandy::core::TensorDesc(sandy::core::Shape({1, 2, 4}), sandy::core::DType::BF16),
        10000.0f);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({1, 2, 4}));
    EXPECT_EQ(out.desc.dtype, sandy::core::DType::BF16);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 1), 2.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 2), 3.0f);
    EXPECT_FLOAT_EQ(read_bf16(out.data, 3), 4.0f);
    EXPECT_NEAR(read_bf16(out.data, 4), std::cos(1.0f), 0.003f);
    EXPECT_NEAR(read_bf16(out.data, 5), std::sin(1.0f), 0.003f);
    EXPECT_NEAR(read_bf16(out.data, 6), -std::sin(0.01f), 0.003f);
    EXPECT_NEAR(read_bf16(out.data, 7), std::cos(0.01f), 0.003f);
}

TEST(TensorCalcTest, RMSNormF32) {
    auto x = f32_bytes({1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f});
    auto weight = f32_bytes({1.0f, 10.0f, -1.0f});
    constexpr float eps = 1.0e-6f;

    auto result = rms_norm_calc(
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

    auto result = rms_norm_calc(
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

TEST(TensorCalcTest, RMSNormF32WithoutScale) {
    auto x = f32_bytes({1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f});
    constexpr float eps = 1.0e-6f;

    auto result = rms_norm_unscaled_calc(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 3}), sandy::core::DType::F32),
        eps);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 3}));

    float row0Scale = 1.0f / std::sqrt(3.0f + eps);
    float row1Scale = 1.0f / std::sqrt((25.0f / 3.0f) + eps);
    EXPECT_NEAR(read_f32(out.data, 0), 1.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 1), 2.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 2), 2.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 4), 3.0f * row1Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(out.data, 5), 4.0f * row1Scale, 1.0e-5f);
}

TEST(TensorCalcTest, LayerNormF32Rank3NormalizesLastDimWithBias) {
    auto x = f32_bytes({
        1.0f, 2.0f,
        3.0f, 5.0f,
        10.0f, 14.0f,
        -2.0f, 2.0f,
    });
    auto weight = f32_bytes({2.0f, -1.0f});
    auto bias = f32_bytes({0.5f, 10.0f});
    constexpr float eps = 1.0e-5f;

    auto result = layer_norm_calc(
        x, sandy::core::TensorDesc(sandy::core::Shape({2, 2, 2}), sandy::core::DType::F32),
        weight, sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32),
        bias, sandy::core::TensorDesc(sandy::core::Shape({2}), sandy::core::DType::F32),
        eps);

    ASSERT_TRUE(result) << result.error();
    auto out = result.take();
    EXPECT_EQ(out.desc.shape, sandy::core::Shape({2, 2, 2}));

    for (size_t row = 0; row < 4; row++) {
        float a = read_f32(x, row * 2);
        float b = read_f32(x, row * 2 + 1);
        float mean = (a + b) / 2.0f;
        float var = ((a - mean) * (a - mean) + (b - mean) * (b - mean)) / 2.0f;
        float invStd = 1.0f / std::sqrt(var + eps);
        EXPECT_NEAR(read_f32(out.data, row * 2), (a - mean) * invStd * 2.0f + 0.5f, 1.0e-5f);
        EXPECT_NEAR(read_f32(out.data, row * 2 + 1), (b - mean) * invStd * -1.0f + 10.0f, 1.0e-5f);
    }
}

TEST(CpuInterpretTest, EngineRunReturnsOutput0) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({2, 2}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("x", sandy::core::Shape({1, 2}), {1.0f, 2.0f}));

    sandy::engine::TensorMap weights;
    weights["w"] = make_f32_buffer("w", sandy::core::Shape({2, 2}), {3.0f, 4.0f, 5.0f, 6.0f});
    weights["b"] = make_f32_buffer("b", sandy::core::Shape({2}), {7.0f, 8.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 2}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 18.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 25.0f);
}

TEST(CpuInterpretTest, LinearRank3) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 2, 2}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({3, 2}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({2, 2, 2}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}));

    sandy::engine::TensorMap weights;
    weights["w"] = make_f32_buffer(
        "w", sandy::core::Shape({3, 2}), {10.0f, 100.0f, -1.0f, 1.0f, 2.0f, 3.0f});
    weights["b"] = make_f32_buffer("b", sandy::core::Shape({3}), {1.0f, 2.0f, 3.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({2, 2, 3}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 211.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 11), 41.0f);
}

TEST(CpuInterpretTest, RMSNorm) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("norm.weight", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createRMSNorm(x, weight);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f}));

    sandy::engine::TensorMap weights;
    weights["norm.weight"] = make_f32_buffer(
        "norm.weight", sandy::core::Shape({3}), {1.0f, 10.0f, -1.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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

TEST(CpuInterpretTest, RMSNormWithoutScale) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createRMSNorm(x);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f}));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({2, 3}));

    constexpr float eps = 1.0e-6f;
    float row0Scale = 1.0f / std::sqrt(3.0f + eps);
    float row1Scale = 1.0f / std::sqrt((25.0f / 3.0f) + eps);
    EXPECT_NEAR(read_f32(it->second->data(), 0), 1.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 1), 2.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 2), 2.0f * row0Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 4), 3.0f * row1Scale, 1.0e-5f);
    EXPECT_NEAR(read_f32(it->second->data(), 5), 4.0f * row1Scale, 1.0e-5f);
}

TEST(CpuInterpretTest, LayerNorm) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2, 2}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("ln.weight", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("ln.bias", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createLayerNorm(x, weight, bias, 1.0e-5f);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("x", sandy::core::Shape({1, 2, 2}), {1.0f, 2.0f, 3.0f, 5.0f}));

    sandy::engine::TensorMap weights;
    weights["ln.weight"] = make_f32_buffer("ln.weight", sandy::core::Shape({2}), {2.0f, -1.0f});
    weights["ln.bias"] = make_f32_buffer("ln.bias", sandy::core::Shape({2}), {0.5f, 10.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 2, 2}));
    EXPECT_NEAR(read_f32(it->second->data(), 0), -1.49996f, 1.0e-4f);
    EXPECT_NEAR(read_f32(it->second->data(), 1), 9.00002f, 1.0e-4f);
    EXPECT_NEAR(read_f32(it->second->data(), 2), -1.49999f, 1.0e-4f);
    EXPECT_NEAR(read_f32(it->second->data(), 3), 9.00001f, 1.0e-4f);
}

TEST(CpuInterpretTest, AddMulSqrt) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("bias", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* scale = builder.createWeight("scale", sandy::core::Shape({2, 1}), sandy::core::DType::F32);
    auto* y = builder.createAdd(x, bias);
    auto* z = builder.createMul(y, scale);
    auto* out = builder.createSqrt(z);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f}));

    sandy::engine::TensorMap weights;
    weights["bias"] = make_f32_buffer("bias", sandy::core::Shape({3}), {0.0f, 5.0f, 7.0f});
    weights["scale"] = make_f32_buffer("scale", sandy::core::Shape({2, 1}), {1.0f, 4.0f});

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* x = builder.createInput(0, sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* offset = builder.createConstantF32(1.0f);
    auto* shifted = builder.createAdd(x, offset);
    auto* out = builder.createTanh(shifted);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("x", sandy::core::Shape({3}), {-2.0f, -1.0f, 0.0f}));

    sandy::engine::TensorMap weights;

    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("embed_tokens.weight", sandy::core::Shape({4, 3}), sandy::core::DType::F32);
    auto* weightT = builder.createTranspose(weight);
    auto* logits = builder.createMatMul(x, weightT);
    sandy::ir::mid_ir::Value* outputs[] = {logits};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

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
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2, 6}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {1, 2, 3, 2});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({1, 2, 6}), {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        }));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2, 3, 2}), sandy::core::DType::F32);
    auto* out = builder.createPermute(x, {0, 2, 1, 3});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer(
        "x", sandy::core::Shape({1, 2, 3, 2}), {
            0.0f, 1.0f,
            2.0f, 3.0f,
            4.0f, 5.0f,
            6.0f, 7.0f,
            8.0f, 9.0f,
            10.0f, 11.0f,
        }));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* q = builder.createInput(0, sandy::core::Shape({1, 1, 3, 1}), sandy::core::DType::F32);
    auto* k = builder.createInput(1, sandy::core::Shape({1, 1, 3, 1}), sandy::core::DType::F32);
    auto* out = builder.createSlidingQueryKeyScore(q, k, 2);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("q", sandy::core::Shape({1, 1, 3, 1}), {
        1.0f, 2.0f, 3.0f,
    }));
    inputs.push_back(make_f32_buffer("k", sandy::core::Shape({1, 1, 3, 1}), {
        10.0f, 20.0f, 30.0f,
    }));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* q = builder.createInput(0, sandy::core::Shape({1, 1, 2, 1}), sandy::core::DType::F32);
    auto* k = builder.createInput(1, sandy::core::Shape({1, 1, 2, 1}), sandy::core::DType::F32);
    auto* scores = builder.createSlidingQueryKeyScore(q, k);
    auto* out = builder.createSoftmax(scores, -1);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("q", sandy::core::Shape({1, 1, 2, 1}), {
        1.0f, 1.0f,
    }));
    inputs.push_back(make_f32_buffer("k", sandy::core::Shape({1, 1, 2, 1}), {
        1.0f, 2.0f,
    }));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

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
    auto* ids = builder.createInput(0, sandy::core::Shape({2, 2}), sandy::core::DType::I32);
    auto* weight = builder.createWeight("embed_tokens.weight", sandy::core::Shape({4, 2}), sandy::core::DType::F32);
    auto* out = builder.createEmbedding(ids, weight);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_i32_buffer(
        "input_ids", sandy::core::Shape({2, 2}), {3, 1, 0, 2}));

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
    auto outputsMap = outputs_to_map(runResult.take());

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

TEST(CpuInterpretTest, RoPEReceivesThetaAttr) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2, 4}), sandy::core::DType::F32);
    auto* out = builder.createRoPE(x, 10000.0f);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("x", sandy::core::Shape({1, 2, 4}), {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 0.0f, 0.0f, 1.0f,
    }));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 2, 4}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 3), 4.0f);
    EXPECT_NEAR(read_f32(it->second->data(), 4), std::cos(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(it->second->data(), 5), std::sin(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(it->second->data(), 6), -std::sin(0.01f), 1.0e-6f);
    EXPECT_NEAR(read_f32(it->second->data(), 7), std::cos(0.01f), 1.0e-6f);
}

TEST(CpuInterpretTest, RoPEReceivesRotaryDimAttr) {
    sandy::ir::mid_ir::register_all_ops();

    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2, 4}), sandy::core::DType::F32);
    auto* out = builder.createRoPE(x, 10000.0f, 2);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    auto engine = make_cpu_engine();

    auto planResult = engine.compile(graph);
    ASSERT_TRUE(planResult) << planResult.error();
    auto plan = planResult.take();

    std::vector<sandy::engine::TensorBufferPtr> inputs;
    inputs.push_back(make_f32_buffer("x", sandy::core::Shape({1, 2, 4}), {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 0.0f, 5.0f, 6.0f,
    }));

    sandy::engine::TensorMap weights;
    auto runResult = engine.run(*plan, inputs, weights);
    ASSERT_TRUE(runResult) << runResult.error();
    auto outputsMap = outputs_to_map(runResult.take());

    auto it = outputsMap.find("output0");
    ASSERT_NE(it, outputsMap.end());
    ASSERT_NE(it->second, nullptr);
    EXPECT_EQ(it->second->desc().shape, sandy::core::Shape({1, 2, 4}));
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 0), 1.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 1), 2.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 2), 3.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 3), 4.0f);
    EXPECT_NEAR(read_f32(it->second->data(), 4), std::cos(1.0f), 1.0e-6f);
    EXPECT_NEAR(read_f32(it->second->data(), 5), std::sin(1.0f), 1.0e-6f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 6), 5.0f);
    EXPECT_FLOAT_EQ(read_f32(it->second->data(), 7), 6.0f);
}
