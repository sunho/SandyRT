#pragma once

#include "EngineTypes.h"
#include "Result.h"

#include <cstdint>
#include <optional>
#include <random>
#include <utility>

namespace sandy::server {

struct SamplingConfig {
    int32_t topK = 1;
    float topP = 1.0f;
    float temperature = 0.0f;
};

struct SamplingOverrides {
    std::optional<float> topP;
    std::optional<float> temperature;
};

Result<SamplingConfig> resolveSamplingConfig(
    SamplingConfig defaults,
    const SamplingOverrides& overrides = {});

class Sampler {
public:
    Sampler();
    explicit Sampler(uint64_t seed);

    Result<std::pair<int64_t, float>> sampleLogitsLast(
        engine::TensorBufferPtr logits,
        const SamplingConfig& config);
    Result<std::pair<int64_t, float>> sampleTopKLast(
        engine::TensorBufferPtr values,
        engine::TensorBufferPtr indices,
        const SamplingConfig& config);

    Result<std::pair<int64_t, float>> argmaxLast(engine::TensorBufferPtr logits);
    Result<std::pair<int64_t, float>> topkLast(
        engine::TensorBufferPtr values,
        engine::TensorBufferPtr indices);

private:
    std::mt19937_64 random_;
};

} // namespace sandy::server
