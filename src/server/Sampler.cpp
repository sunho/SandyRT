#include "Sampler.h"

#include "Tensor.h"

#include <cstring>
#include <limits>

namespace sandy::server {

namespace {

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

float read_bf16(std::span<const uint8_t> data, size_t index) {
    sandy::core::BFloat16 value = sandy::core::bfloat16_from_bits(0);
    std::memcpy(&value, data.data() + index * sizeof(sandy::core::BFloat16), sizeof(value));
    return sandy::core::bfloat16_to_float(value);
}

float read_float_value(std::span<const uint8_t> data, sandy::core::DType dtype, size_t index) {
    switch (dtype) {
        case sandy::core::DType::F32:
            return read_f32(data, index);
        case sandy::core::DType::BF16:
            return read_bf16(data, index);
        default:
            return -std::numeric_limits<float>::infinity();
    }
}

} // namespace

Result<std::pair<int64_t, float>> Sampler::argmaxLast(engine::TensorBufferPtr logits) const {
    if (!logits)
        return make_error("sampler received null logits");

    auto access = logits->access();
    if (!access)
        return make_error(access.error());

    const auto& desc = access->desc();
    if (desc.dtype != core::DType::F32 && desc.dtype != core::DType::BF16)
        return make_error("logits must be F32 or BF16");
    if (desc.shape.rank() < 2)
        return make_error("logits must have rank >= 2");

    int64_t vocab = desc.shape.dim(desc.shape.rank() - 1);
    int64_t numel = desc.shape.numel();
    if (vocab <= 0 || numel <= 0)
        return make_error("logits must have static non-empty shape");

    int64_t rows = numel / vocab;
    if (rows <= 0)
        return make_error("logits has no rows");

    size_t base = static_cast<size_t>((rows - 1) * vocab);
    int64_t bestId = 0;
    float bestValue = -std::numeric_limits<float>::infinity();
    for (int64_t id = 0; id < vocab; id++) {
        float value = read_float_value(access->data(), desc.dtype, base + static_cast<size_t>(id));
        if (value > bestValue) {
            bestValue = value;
            bestId = id;
        }
    }

    return std::pair<int64_t, float>{bestId, bestValue};
}

} // namespace sandy::server
