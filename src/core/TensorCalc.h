#pragma once

#include "Result.h"
#include "Tensor.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sandy::core {

struct TensorRef {
    using LoadFloatFn = float (*)(std::span<const uint8_t>, size_t);

    TensorDesc desc;
    std::span<const uint8_t> bytes;
    std::vector<int64_t> strides;
    int64_t storageOffset = 0;
    LoadFloatFn loadFloat = nullptr;

    size_t storage_index(size_t index) const;
    bool is_contiguous() const;
    float load_float(size_t index) const { return loadFloat(bytes, storage_index(index)); }
};

struct MutableTensorRef {
    using LoadFloatFn = float (*)(std::span<const uint8_t>, size_t);
    using StoreFloatFn = void (*)(std::span<uint8_t>, size_t, float);

    TensorDesc desc;
    std::span<uint8_t> bytes;
    std::vector<int64_t> strides;
    int64_t storageOffset = 0;
    LoadFloatFn loadFloat = nullptr;
    StoreFloatFn storeFloat = nullptr;

    size_t storage_index(size_t index) const;
    bool is_contiguous() const;
    float load_float(size_t index) const { return loadFloat(bytes, storage_index(index)); }
    void store_float(size_t index, float value) const {
        storeFloat(bytes, storage_index(index), value);
    }
};

Result<TensorRef> make_tensor_ref(TensorDesc desc, std::span<const uint8_t> bytes);
Result<TensorRef> make_tensor_ref(
    TensorDesc desc,
    std::span<const uint8_t> bytes,
    std::span<const int64_t> strides,
    int64_t storageOffset);
Result<MutableTensorRef> make_mutable_tensor_ref(TensorDesc desc, std::span<uint8_t> bytes);
Result<MutableTensorRef> make_mutable_tensor_ref(
    TensorDesc desc,
    std::span<uint8_t> bytes,
    std::span<const int64_t> strides,
    int64_t storageOffset);

Result<void> linear(
    TensorRef x,
    TensorRef weight,
    TensorRef bias,
    MutableTensorRef out);

Result<void> relu(TensorRef x, MutableTensorRef out);
Result<void> add(TensorRef lhs, TensorRef rhs, MutableTensorRef out);
Result<void> mul(TensorRef lhs, TensorRef rhs, MutableTensorRef out);
Result<void> sqrt(TensorRef x, MutableTensorRef out);
Result<void> tanh(TensorRef x, MutableTensorRef out);
Result<void> matmul(TensorRef lhs, TensorRef rhs, MutableTensorRef out);
Result<void> matmul(TensorRef lhs, TensorRef rhs, bool transpose_lhs, bool transpose_rhs, MutableTensorRef out);
Result<void> transpose(TensorRef x, MutableTensorRef out);
Result<void> reshape(TensorRef x, MutableTensorRef out);
Result<void> permute(TensorRef x, std::span<const int64_t> dims, MutableTensorRef out);
Result<void> sliding_query_key_score(
    TensorRef q,
    TensorRef k,
    int64_t window,
    float scale,
    MutableTensorRef out);
Result<void> sliding_query_key_score(
    TensorRef q,
    TensorRef k,
    int64_t window,
    MutableTensorRef out);
Result<void> softmax(TensorRef x, int64_t dim, MutableTensorRef out);
Result<void> embedding(TensorRef ids, TensorRef weight, MutableTensorRef out);
Result<void> rope(TensorRef x, float theta, MutableTensorRef out);
Result<void> rope(TensorRef x, float theta, int64_t rotary_dim, MutableTensorRef out);
Result<void> rope(TensorRef x, float theta, int64_t rotary_dim, bool split_half, MutableTensorRef out);
Result<void> rope(TensorRef x, TensorRef position_ids, float theta, int64_t rotary_dim, bool split_half, MutableTensorRef out);
Result<void> rms_norm(TensorRef x, float epsilon, MutableTensorRef out);
Result<void> rms_norm(TensorRef x, TensorRef weight, float epsilon, MutableTensorRef out);
Result<void> layer_norm(
    TensorRef x,
    TensorRef weight,
    TensorRef bias,
    float epsilon,
    MutableTensorRef out);

} // namespace sandy::core
