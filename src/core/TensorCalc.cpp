#include "TensorCalc.h"
#include "ShapeUtil.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace sandy::core {

namespace {

Result<void> require_f32(const TensorDesc& desc, const std::string& name) {
    if (desc.dtype != DType::F32)
        return make_error(name + " must be f32");
    return {};
}

Result<void> require_bytes(std::span<const uint8_t> data,
                           const TensorDesc& desc,
                           const std::string& name) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error(name + " must have static shape");

    auto expected = static_cast<size_t>(numel) * dtype_size(desc.dtype);
    if (data.size() != expected)
        return make_error(name + " byte size mismatch");

    return {};
}

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

int64_t read_index(std::span<const uint8_t> data, DType dtype, size_t index) {
    if (dtype == DType::I32) {
        int32_t value = 0;
        std::memcpy(&value, data.data() + index * sizeof(int32_t), sizeof(int32_t));
        return value;
    }

    int64_t value = 0;
    std::memcpy(&value, data.data() + index * sizeof(int64_t), sizeof(int64_t));
    return value;
}

void write_f32(std::vector<uint8_t>& data, size_t index, float value) {
    std::memcpy(data.data() + index * sizeof(float), &value, sizeof(float));
}

std::vector<int64_t> strides_for(const Shape& shape) {
    std::vector<int64_t> strides(static_cast<size_t>(shape.rank()), 1);
    int64_t stride = 1;
    for (int i = shape.rank() - 1; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= shape.dim(i);
    }
    return strides;
}

size_t broadcast_source_index(
        size_t outIndex,
        const Shape& outShape,
        const Shape& sourceShape,
        const std::vector<int64_t>& sourceStrides) {
    if (sourceShape.rank() == 0) return 0;

    size_t sourceIndex = 0;
    size_t remaining = outIndex;
    int rankOffset = outShape.rank() - sourceShape.rank();
    for (int outDimIndex = outShape.rank() - 1; outDimIndex >= 0; outDimIndex--) {
        int64_t outDim = outShape.dim(outDimIndex);
        size_t coord = remaining % static_cast<size_t>(outDim);
        remaining /= static_cast<size_t>(outDim);

        int sourceDimIndex = outDimIndex - rankOffset;
        if (sourceDimIndex < 0) continue;
        if (sourceShape.dim(sourceDimIndex) != 1) {
            sourceIndex += coord *
                static_cast<size_t>(sourceStrides[static_cast<size_t>(sourceDimIndex)]);
        }
    }
    return sourceIndex;
}

size_t broadcast_batch_offset(
        size_t batchIndex,
        const Shape& batchShape,
        const Shape& sourceShape,
        const std::vector<int64_t>& sourceStrides) {
    int sourceBatchRank = sourceShape.rank() - 2;
    if (sourceBatchRank <= 0) return 0;

    size_t sourceOffset = 0;
    size_t remaining = batchIndex;
    int rankOffset = batchShape.rank() - sourceBatchRank;
    for (int batchDimIndex = batchShape.rank() - 1; batchDimIndex >= 0; batchDimIndex--) {
        int64_t batchDim = batchShape.dim(batchDimIndex);
        size_t coord = remaining % static_cast<size_t>(batchDim);
        remaining /= static_cast<size_t>(batchDim);

        int sourceDimIndex = batchDimIndex - rankOffset;
        if (sourceDimIndex < 0) continue;
        if (sourceShape.dim(sourceDimIndex) != 1) {
            sourceOffset += coord *
                static_cast<size_t>(sourceStrides[static_cast<size_t>(sourceDimIndex)]);
        }
    }
    return sourceOffset;
}

using BinaryOp = float (*)(float, float);

Result<OwnedTensor> binary_elementwise_f32(
        std::span<const uint8_t> lhs,
        const TensorDesc& lhsDesc,
        std::span<const uint8_t> rhs,
        const TensorDesc& rhsDesc,
        const std::string& opName,
        BinaryOp op) {
    auto lhsDtype = require_f32(lhsDesc, opName + " lhs");
    if (!lhsDtype) return make_error(lhsDtype.error());

    auto rhsDtype = require_f32(rhsDesc, opName + " rhs");
    if (!rhsDtype) return make_error(rhsDtype.error());

    auto lhsBytes = require_bytes(lhs, lhsDesc, opName + " lhs");
    if (!lhsBytes) return make_error(lhsBytes.error());

    auto rhsBytes = require_bytes(rhs, rhsDesc, opName + " rhs");
    if (!rhsBytes) return make_error(rhsBytes.error());

    auto shapeResult = broadcast_shape(lhsDesc.shape, rhsDesc.shape);
    if (!shapeResult) return make_error(shapeResult.error());
    auto outShape = shapeResult.take();

    int64_t outNumel = outShape.numel();
    if (outNumel < 0)
        return make_error(opName + " output must have static shape");

    auto lhsStrides = strides_for(lhsDesc.shape);
    auto rhsStrides = strides_for(rhsDesc.shape);

    OwnedTensor out;
    out.desc = TensorDesc(outShape, DType::F32);
    out.data.resize(static_cast<size_t>(outNumel) * sizeof(float));

    for (size_t i = 0; i < static_cast<size_t>(outNumel); i++) {
        size_t lhsIndex = broadcast_source_index(i, out.desc.shape, lhsDesc.shape, lhsStrides);
        size_t rhsIndex = broadcast_source_index(i, out.desc.shape, rhsDesc.shape, rhsStrides);
        write_f32(out.data, i, op(read_f32(lhs, lhsIndex), read_f32(rhs, rhsIndex)));
    }

    return out;
}

} // namespace

Result<OwnedTensor> linear_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc,
        std::span<const uint8_t> weight,
        const TensorDesc& weightDesc,
        std::span<const uint8_t> bias,
        const TensorDesc& biasDesc) {
    auto xDtype = require_f32(xDesc, "linear input");
    if (!xDtype) return make_error(xDtype.error());

    auto weightDtype = require_f32(weightDesc, "linear weight");
    if (!weightDtype) return make_error(weightDtype.error());

    auto biasDtype = require_f32(biasDesc, "linear bias");
    if (!biasDtype) return make_error(biasDtype.error());

    if (xDesc.shape.rank() != 2)
        return make_error("linear input must have rank 2");
    if (weightDesc.shape.rank() != 2)
        return make_error("linear weight must have rank 2");
    if (biasDesc.shape.rank() != 1)
        return make_error("linear bias must have rank 1");

    int64_t batch = xDesc.shape.dim(0);
    int64_t inFeatures = xDesc.shape.dim(1);
    int64_t outFeatures = weightDesc.shape.dim(0);
    if (batch < 0 || inFeatures < 0 || outFeatures < 0)
        return make_error("linear inputs must have static shape");
    if (weightDesc.shape.dim(1) != inFeatures)
        return make_error("linear weight input dimension mismatch");
    if (biasDesc.shape.dim(0) != outFeatures)
        return make_error("linear bias dimension mismatch");

    auto xBytes = require_bytes(x, xDesc, "linear input");
    if (!xBytes) return make_error(xBytes.error());

    auto weightBytes = require_bytes(weight, weightDesc, "linear weight");
    if (!weightBytes) return make_error(weightBytes.error());

    auto biasBytes = require_bytes(bias, biasDesc, "linear bias");
    if (!biasBytes) return make_error(biasBytes.error());

    OwnedTensor out;
    out.desc = TensorDesc(Shape({batch, outFeatures}), DType::F32);
    out.data.resize(static_cast<size_t>(batch * outFeatures) * sizeof(float));

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t o = 0; o < outFeatures; o++) {
            float acc = read_f32(bias, static_cast<size_t>(o));
            for (int64_t i = 0; i < inFeatures; i++) {
                float xv = read_f32(x, static_cast<size_t>(b * inFeatures + i));
                float wv = read_f32(weight, static_cast<size_t>(o * inFeatures + i));
                acc += xv * wv;
            }
            write_f32(out.data, static_cast<size_t>(b * outFeatures + o), acc);
        }
    }

    return out;
}

Result<OwnedTensor> relu_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc) {
    auto xDtype = require_f32(xDesc, "relu input");
    if (!xDtype) return make_error(xDtype.error());

    auto xBytes = require_bytes(x, xDesc, "relu input");
    if (!xBytes) return make_error(xBytes.error());

    OwnedTensor out;
    out.desc = xDesc;
    out.data.resize(x.size());

    size_t count = x.size() / sizeof(float);
    for (size_t i = 0; i < count; i++)
        write_f32(out.data, i, std::max(0.0f, read_f32(x, i)));

    return out;
}

Result<OwnedTensor> add_f32(
        std::span<const uint8_t> lhs,
        const TensorDesc& lhsDesc,
        std::span<const uint8_t> rhs,
        const TensorDesc& rhsDesc) {
    return binary_elementwise_f32(
        lhs, lhsDesc, rhs, rhsDesc, "add",
        [](float a, float b) { return a + b; });
}

Result<OwnedTensor> mul_f32(
        std::span<const uint8_t> lhs,
        const TensorDesc& lhsDesc,
        std::span<const uint8_t> rhs,
        const TensorDesc& rhsDesc) {
    return binary_elementwise_f32(
        lhs, lhsDesc, rhs, rhsDesc, "mul",
        [](float a, float b) { return a * b; });
}

Result<OwnedTensor> sqrt_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc) {
    auto xDtype = require_f32(xDesc, "sqrt input");
    if (!xDtype) return make_error(xDtype.error());

    auto xBytes = require_bytes(x, xDesc, "sqrt input");
    if (!xBytes) return make_error(xBytes.error());

    OwnedTensor out;
    out.desc = xDesc;
    out.data.resize(x.size());

    size_t count = x.size() / sizeof(float);
    for (size_t i = 0; i < count; i++)
        write_f32(out.data, i, std::sqrt(read_f32(x, i)));

    return out;
}

Result<OwnedTensor> matmul_f32(
        std::span<const uint8_t> lhs,
        const TensorDesc& lhsDesc,
        std::span<const uint8_t> rhs,
        const TensorDesc& rhsDesc) {
    auto lhsDtype = require_f32(lhsDesc, "matmul lhs");
    if (!lhsDtype) return make_error(lhsDtype.error());

    auto rhsDtype = require_f32(rhsDesc, "matmul rhs");
    if (!rhsDtype) return make_error(rhsDtype.error());

    if (lhsDesc.shape.rank() < 2)
        return make_error("matmul lhs must have rank >= 2");
    if (rhsDesc.shape.rank() < 2)
        return make_error("matmul rhs must have rank >= 2");

    int lhsRank = lhsDesc.shape.rank();
    int rhsRank = rhsDesc.shape.rank();
    int64_t m = lhsDesc.shape.dim(lhsRank - 2);
    int64_t lhsK = lhsDesc.shape.dim(lhsRank - 1);
    int64_t rhsK = rhsDesc.shape.dim(rhsRank - 2);
    int64_t n = rhsDesc.shape.dim(rhsRank - 1);
    if (m < 0 || n < 0 || lhsK < 0 || rhsK < 0)
        return make_error("matmul matrix dimensions must be static");
    if (lhsK != rhsK)
        return make_error("matmul contracting dimension mismatch");

    auto lhsBytes = require_bytes(lhs, lhsDesc, "matmul lhs");
    if (!lhsBytes) return make_error(lhsBytes.error());

    auto rhsBytes = require_bytes(rhs, rhsDesc, "matmul rhs");
    if (!rhsBytes) return make_error(rhsBytes.error());

    auto lhsDims = lhsDesc.shape.dims();
    auto rhsDims = rhsDesc.shape.dims();
    Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
    Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
    auto batchShapeResult = broadcast_shape(lhsBatch, rhsBatch);
    if (!batchShapeResult) return make_error(batchShapeResult.error());
    auto batchShape = batchShapeResult.take();

    auto outDims = batchShape.dims();
    outDims.push_back(m);
    outDims.push_back(n);
    Shape outShape(outDims);
    int64_t outNumel = outShape.numel();
    int64_t batchNumel = batchShape.numel();
    if (outNumel < 0 || batchNumel < 0)
        return make_error("matmul output must have static shape");

    auto lhsStrides = strides_for(lhsDesc.shape);
    auto rhsStrides = strides_for(rhsDesc.shape);
    OwnedTensor out;
    out.desc = TensorDesc(outShape, DType::F32);
    out.data.resize(static_cast<size_t>(outNumel) * sizeof(float));

    for (size_t batch = 0; batch < static_cast<size_t>(batchNumel); batch++) {
        size_t lhsBatchOffset = broadcast_batch_offset(
            batch, batchShape, lhsDesc.shape, lhsStrides);
        size_t rhsBatchOffset = broadcast_batch_offset(
            batch, batchShape, rhsDesc.shape, rhsStrides);
        size_t outBatchOffset = batch * static_cast<size_t>(m * n);

        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < n; col++) {
                float acc = 0.0f;
                for (int64_t k = 0; k < lhsK; k++) {
                    size_t lhsIndex = lhsBatchOffset +
                        static_cast<size_t>(row * lhsStrides[lhsRank - 2] +
                                            k * lhsStrides[lhsRank - 1]);
                    size_t rhsIndex = rhsBatchOffset +
                        static_cast<size_t>(k * rhsStrides[rhsRank - 2] +
                                            col * rhsStrides[rhsRank - 1]);
                    acc += read_f32(lhs, lhsIndex) * read_f32(rhs, rhsIndex);
                }
                write_f32(
                    out.data,
                    outBatchOffset + static_cast<size_t>(row * n + col),
                    acc);
            }
        }
    }

    return out;
}

Result<OwnedTensor> transpose_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc) {
    auto xDtype = require_f32(xDesc, "transpose input");
    if (!xDtype) return make_error(xDtype.error());

    if (xDesc.shape.rank() != 2)
        return make_error("transpose input must have rank 2");

    auto xBytes = require_bytes(x, xDesc, "transpose input");
    if (!xBytes) return make_error(xBytes.error());

    auto outDims = xDesc.shape.dims();
    std::swap(outDims[outDims.size() - 1], outDims[outDims.size() - 2]);
    Shape outShape(outDims);
    int64_t outNumel = outShape.numel();
    if (outNumel < 0)
        return make_error("transpose output must have static shape");

    auto inputStrides = strides_for(xDesc.shape);
    auto outputStrides = strides_for(outShape);

    OwnedTensor out;
    out.desc = TensorDesc(outShape, DType::F32);
    out.data.resize(static_cast<size_t>(outNumel) * sizeof(float));

    for (size_t outIndex = 0; outIndex < static_cast<size_t>(outNumel); outIndex++) {
        size_t row = outIndex / static_cast<size_t>(outputStrides[0]);
        size_t col = outIndex % static_cast<size_t>(outputStrides[0]);
        size_t inputIndex = col * static_cast<size_t>(inputStrides[0]) + row;
        write_f32(out.data, outIndex, read_f32(x, inputIndex));
    }

    return out;
}

Result<OwnedTensor> permute_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc,
        std::span<const int64_t> dims) {
    auto xDtype = require_f32(xDesc, "permute input");
    if (!xDtype) return make_error(xDtype.error());

    int rank = xDesc.shape.rank();
    if (static_cast<int>(dims.size()) != rank)
        return make_error("permute dims size must match input rank");

    std::vector<bool> seen(static_cast<size_t>(rank), false);
    std::vector<int64_t> outDims;
    outDims.reserve(dims.size());
    for (int64_t axis : dims) {
        if (axis < 0 || axis >= rank)
            return make_error("permute axis out of range");
        auto index = static_cast<size_t>(axis);
        if (seen[index])
            return make_error("permute dims must not contain duplicates");
        seen[index] = true;
        outDims.push_back(xDesc.shape.dim(static_cast<int>(axis)));
    }

    auto xBytes = require_bytes(x, xDesc, "permute input");
    if (!xBytes) return make_error(xBytes.error());

    Shape outShape(outDims);
    int64_t outNumel = outShape.numel();
    if (outNumel < 0)
        return make_error("permute output must have static shape");

    auto inputStrides = strides_for(xDesc.shape);
    auto outputStrides = strides_for(outShape);

    OwnedTensor out;
    out.desc = TensorDesc(outShape, DType::F32);
    out.data.resize(static_cast<size_t>(outNumel) * sizeof(float));

    for (size_t outIndex = 0; outIndex < static_cast<size_t>(outNumel); outIndex++) {
        size_t sourceIndex = 0;
        size_t remaining = outIndex;
        for (int axisIndex = 0; axisIndex < rank; axisIndex++) {
            size_t stride = static_cast<size_t>(outputStrides[static_cast<size_t>(axisIndex)]);
            size_t coord = 0;
            if (stride != 0) {
                coord = remaining / stride;
                remaining %= stride;
            }
            int inputAxis = static_cast<int>(dims[static_cast<size_t>(axisIndex)]);
            sourceIndex += coord *
                static_cast<size_t>(inputStrides[static_cast<size_t>(inputAxis)]);
        }
        write_f32(out.data, outIndex, read_f32(x, sourceIndex));
    }

    return out;
}

Result<OwnedTensor> embedding_f32(
        std::span<const uint8_t> ids,
        const TensorDesc& idsDesc,
        std::span<const uint8_t> weight,
        const TensorDesc& weightDesc) {
    if (idsDesc.dtype != DType::I32 && idsDesc.dtype != DType::I64)
        return make_error("embedding ids must be i32 or i64");

    auto weightDtype = require_f32(weightDesc, "embedding weight");
    if (!weightDtype) return make_error(weightDtype.error());

    if (weightDesc.shape.rank() != 2)
        return make_error("embedding weight must have rank 2");

    int64_t vocab = weightDesc.shape.dim(0);
    int64_t hidden = weightDesc.shape.dim(1);
    if (vocab < 0 || hidden < 0)
        return make_error("embedding weight must have static shape");

    auto idsBytes = require_bytes(ids, idsDesc, "embedding ids");
    if (!idsBytes) return make_error(idsBytes.error());

    auto weightBytes = require_bytes(weight, weightDesc, "embedding weight");
    if (!weightBytes) return make_error(weightBytes.error());

    int64_t idsNumel = idsDesc.shape.numel();
    if (idsNumel < 0)
        return make_error("embedding ids must have static shape");

    auto outDims = idsDesc.shape.dims();
    outDims.push_back(hidden);

    OwnedTensor out;
    out.desc = TensorDesc(Shape(outDims), DType::F32);
    out.data.resize(static_cast<size_t>(idsNumel * hidden) * sizeof(float));

    for (int64_t i = 0; i < idsNumel; i++) {
        int64_t tokenId = read_index(ids, idsDesc.dtype, static_cast<size_t>(i));
        if (tokenId < 0 || tokenId >= vocab)
            return make_error("embedding id out of range");

        for (int64_t h = 0; h < hidden; h++) {
            float value = read_f32(
                weight, static_cast<size_t>(tokenId * hidden + h));
            write_f32(out.data, static_cast<size_t>(i * hidden + h), value);
        }
    }

    return out;
}

Result<OwnedTensor> rms_norm_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc,
        std::span<const uint8_t> weight,
        const TensorDesc& weightDesc,
        float epsilon) {
    auto xDtype = require_f32(xDesc, "rms_norm input");
    if (!xDtype) return make_error(xDtype.error());

    auto weightDtype = require_f32(weightDesc, "rms_norm weight");
    if (!weightDtype) return make_error(weightDtype.error());

    if (xDesc.shape.rank() < 1)
        return make_error("rms_norm input must have rank >= 1");
    if (weightDesc.shape.rank() != 1)
        return make_error("rms_norm weight must have rank 1");

    int64_t hidden = xDesc.shape.dim(xDesc.shape.rank() - 1);
    if (hidden < 0)
        return make_error("rms_norm hidden dimension must be static");
    if (weightDesc.shape.dim(0) != hidden)
        return make_error("rms_norm weight dimension mismatch");

    auto xBytes = require_bytes(x, xDesc, "rms_norm input");
    if (!xBytes) return make_error(xBytes.error());

    auto weightBytes = require_bytes(weight, weightDesc, "rms_norm weight");
    if (!weightBytes) return make_error(weightBytes.error());

    int64_t total = xDesc.shape.numel();
    if (total < 0)
        return make_error("rms_norm input must have static shape");

    OwnedTensor out;
    out.desc = xDesc;
    out.data.resize(x.size());

    size_t rows = static_cast<size_t>(total / hidden);
    size_t cols = static_cast<size_t>(hidden);
    for (size_t row = 0; row < rows; row++) {
        double squareSum = 0.0;
        size_t rowOffset = row * cols;
        for (size_t col = 0; col < cols; col++) {
            float xv = read_f32(x, rowOffset + col);
            squareSum += static_cast<double>(xv) * static_cast<double>(xv);
        }

        float invRms = 1.0f / std::sqrt(
            static_cast<float>(squareSum / static_cast<double>(cols)) + epsilon);
        for (size_t col = 0; col < cols; col++) {
            float xv = read_f32(x, rowOffset + col);
            float wv = read_f32(weight, col);
            write_f32(out.data, rowOffset + col, xv * invRms * wv);
        }
    }

    return out;
}

} // namespace sandy::core
