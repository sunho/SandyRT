#include "Sampler.h"

#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

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

int64_t read_int_value(std::span<const uint8_t> data, sandy::core::DType dtype, size_t index) {
    if (dtype == sandy::core::DType::I64) {
        int64_t value = 0;
        std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
        return value;
    }
    int32_t value = 0;
    std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
    return value;
}

struct Candidate {
    int64_t id;
    float logit;
};

Result<std::pair<int64_t, float>> sample_candidates(
        std::vector<Candidate> candidates,
        const SamplingConfig& config,
        std::mt19937_64& random) {
    auto validated = resolveSamplingConfig(config);
    if (!validated)
        return make_error(validated.error());
    if (candidates.empty())
        return make_error("sampler received no candidates");
    for (const auto& candidate : candidates) {
        if (std::isnan(candidate.logit))
            return make_error("sampling logits must not contain NaN");
    }

    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.logit > rhs.logit;
        });
    if (candidates.size() > static_cast<size_t>(config.topK))
        candidates.resize(static_cast<size_t>(config.topK));

    if (config.temperature == 0.0f || !std::isfinite(candidates[0].logit))
        return std::pair<int64_t, float>{candidates[0].id, candidates[0].logit};

    const double maxScaled =
        static_cast<double>(candidates[0].logit) / config.temperature;
    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total = 0.0;
    for (const auto& candidate : candidates) {
        double weight = std::exp(
            static_cast<double>(candidate.logit) / config.temperature - maxScaled);
        weights.push_back(weight);
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total))
        return make_error("sampling probabilities are invalid");

    double cumulative = 0.0;
    size_t keep = 0;
    do {
        cumulative += weights[keep] / total;
        keep++;
    } while (keep < weights.size() && cumulative < config.topP);

    std::discrete_distribution<size_t> distribution(
        weights.begin(),
        weights.begin() + static_cast<std::ptrdiff_t>(keep));
    size_t selected = distribution(random);
    return std::pair<int64_t, float>{candidates[selected].id, candidates[selected].logit};
}

} // namespace

Result<SamplingConfig> resolveSamplingConfig(
        SamplingConfig defaults,
        const SamplingOverrides& overrides) {
    if (overrides.topP)
        defaults.topP = *overrides.topP;
    if (overrides.temperature)
        defaults.temperature = *overrides.temperature;
    if (defaults.topK <= 0)
        return make_error("top_k must be > 0");
    if (!std::isfinite(defaults.topP) || defaults.topP <= 0.0f || defaults.topP > 1.0f)
        return make_error("top_p must be > 0 and <= 1");
    if (!std::isfinite(defaults.temperature) || defaults.temperature < 0.0f)
        return make_error("temperature must be >= 0");
    return defaults;
}

Sampler::Sampler()
    : random_(std::random_device{}()) {}

Sampler::Sampler(uint64_t seed)
    : random_(seed) {}

Result<std::pair<int64_t, float>> Sampler::sampleLogitsLast(
        engine::TensorBufferPtr logits,
        const SamplingConfig& config) {
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
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(vocab));
    for (int64_t id = 0; id < vocab; id++) {
        float value = read_float_value(access->data(), desc.dtype, base + static_cast<size_t>(id));
        candidates.push_back(Candidate{id, value});
    }
    return sample_candidates(std::move(candidates), config, random_);
}

Result<std::pair<int64_t, float>> Sampler::sampleTopKLast(
        engine::TensorBufferPtr values,
        engine::TensorBufferPtr indices,
        const SamplingConfig& config) {
    if (!values || !indices)
        return make_error("sampler received null topk output");

    auto valuesAccess = values->access();
    if (!valuesAccess)
        return make_error(valuesAccess.error());
    auto indicesAccess = indices->access();
    if (!indicesAccess)
        return make_error(indicesAccess.error());

    const auto& valuesDesc = valuesAccess->desc();
    const auto& indicesDesc = indicesAccess->desc();
    if (valuesDesc.dtype != core::DType::F32 && valuesDesc.dtype != core::DType::BF16)
        return make_error("topk values must be F32 or BF16");
    if (indicesDesc.dtype != core::DType::I32 && indicesDesc.dtype != core::DType::I64)
        return make_error("topk indices must be I32 or I64");
    if (valuesDesc.shape != indicesDesc.shape)
        return make_error("topk values and indices shapes must match");
    if (valuesDesc.shape.rank() < 1)
        return make_error("topk outputs must have rank >= 1");

    int64_t width = valuesDesc.shape.dim(valuesDesc.shape.rank() - 1);
    if (width <= 0)
        return make_error("topk outputs must have a static non-empty final dimension");

    int64_t numel = valuesDesc.shape.numel();
    if (numel <= 0)
        return make_error("topk outputs must have a static non-empty shape");
    size_t base = static_cast<size_t>(numel - width);
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(width));
    for (int64_t i = 0; i < width; i++) {
        auto offset = base + static_cast<size_t>(i);
        candidates.push_back(Candidate{
            read_int_value(indicesAccess->data(), indicesDesc.dtype, offset),
            read_float_value(valuesAccess->data(), valuesDesc.dtype, offset)});
    }
    return sample_candidates(std::move(candidates), config, random_);
}

Result<std::pair<int64_t, float>> Sampler::argmaxLast(engine::TensorBufferPtr logits) {
    return sampleLogitsLast(logits, SamplingConfig{});
}

Result<std::pair<int64_t, float>> Sampler::topkLast(
        engine::TensorBufferPtr values,
        engine::TensorBufferPtr indices) {
    return sampleTopKLast(values, indices, SamplingConfig{});
}

} // namespace sandy::server
