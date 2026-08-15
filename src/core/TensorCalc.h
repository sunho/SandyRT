#pragma once

#include "Result.h"
#include "Tensor.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sandy::core {

struct OwnedTensor {
    TensorDesc desc;
    std::vector<uint8_t> data;
};

Result<OwnedTensor> linear(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    std::span<const uint8_t> bias,
    const TensorDesc& biasDesc);

Result<OwnedTensor> linear_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    std::span<const uint8_t> bias,
    const TensorDesc& biasDesc);

Result<OwnedTensor> relu(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> relu_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> add(
    std::span<const uint8_t> lhs,
    const TensorDesc& lhsDesc,
    std::span<const uint8_t> rhs,
    const TensorDesc& rhsDesc);

Result<OwnedTensor> add_f32(
    std::span<const uint8_t> lhs,
    const TensorDesc& lhsDesc,
    std::span<const uint8_t> rhs,
    const TensorDesc& rhsDesc);

Result<OwnedTensor> mul(
    std::span<const uint8_t> lhs,
    const TensorDesc& lhsDesc,
    std::span<const uint8_t> rhs,
    const TensorDesc& rhsDesc);

Result<OwnedTensor> mul_f32(
    std::span<const uint8_t> lhs,
    const TensorDesc& lhsDesc,
    std::span<const uint8_t> rhs,
    const TensorDesc& rhsDesc);

Result<OwnedTensor> sqrt(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> sqrt_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> tanh(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> tanh_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> matmul(
    std::span<const uint8_t> lhs,
    const TensorDesc& lhsDesc,
    std::span<const uint8_t> rhs,
    const TensorDesc& rhsDesc);

Result<OwnedTensor> matmul_f32(
    std::span<const uint8_t> lhs,
    const TensorDesc& lhsDesc,
    std::span<const uint8_t> rhs,
    const TensorDesc& rhsDesc);

Result<OwnedTensor> transpose(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> transpose_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> reshape(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    Shape shape);

Result<OwnedTensor> reshape_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    Shape shape);

Result<OwnedTensor> permute(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const int64_t> dims);

Result<OwnedTensor> permute_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const int64_t> dims);

Result<OwnedTensor> sliding_query_key_score(
    std::span<const uint8_t> q,
    const TensorDesc& qDesc,
    std::span<const uint8_t> k,
    const TensorDesc& kDesc,
    int64_t window);

Result<OwnedTensor> sliding_query_key_score_f32(
    std::span<const uint8_t> q,
    const TensorDesc& qDesc,
    std::span<const uint8_t> k,
    const TensorDesc& kDesc,
    int64_t window);

Result<OwnedTensor> softmax(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    int64_t dim);

Result<OwnedTensor> softmax_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    int64_t dim);

Result<OwnedTensor> embedding(
    std::span<const uint8_t> ids,
    const TensorDesc& idsDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc);

Result<OwnedTensor> embedding_f32(
    std::span<const uint8_t> ids,
    const TensorDesc& idsDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc);

Result<OwnedTensor> rope(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    float theta);

Result<OwnedTensor> rms_norm(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    float epsilon);

Result<OwnedTensor> rms_norm_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    float epsilon);

Result<OwnedTensor> layer_norm(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    std::span<const uint8_t> bias,
    const TensorDesc& biasDesc,
    float epsilon);

Result<OwnedTensor> layer_norm_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    std::span<const uint8_t> bias,
    const TensorDesc& biasDesc,
    float epsilon);

} // namespace sandy::core
