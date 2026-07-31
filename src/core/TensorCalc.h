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

Result<OwnedTensor> linear_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    std::span<const uint8_t> bias,
    const TensorDesc& biasDesc);

Result<OwnedTensor> relu_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc);

Result<OwnedTensor> rms_norm_f32(
    std::span<const uint8_t> x,
    const TensorDesc& xDesc,
    std::span<const uint8_t> weight,
    const TensorDesc& weightDesc,
    float epsilon);

} // namespace sandy::core
