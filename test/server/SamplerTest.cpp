#include "HostTensorBuffer.h"
#include "Sampler.h"

#include "Tensor.h"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

template <typename T>
std::shared_ptr<sandy::server::HostTensorBuffer> make_buffer(
        std::string name,
        sandy::core::Shape shape,
        sandy::core::DType dtype,
        const std::vector<T>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_shared<sandy::server::HostTensorBuffer>(
        sandy::core::TensorDesc(std::move(name), std::move(shape), dtype),
        std::move(bytes));
}

TEST(ServerSampler, ReadsLastGpuTopKRow) {
    auto values = make_buffer<float>(
        "topk_values",
        sandy::core::Shape({1, 3, 1}),
        sandy::core::DType::F32,
        {4.0f, 5.0f, 6.0f});
    auto indices = make_buffer<int64_t>(
        "topk_indices",
        sandy::core::Shape({1, 3, 1}),
        sandy::core::DType::I64,
        {10, 20, 30});

    sandy::server::Sampler sampler;
    auto sampled = sampler.topkLast(values, indices);

    ASSERT_TRUE(sampled) << sampled.error();
    EXPECT_EQ(sampled->first, 30);
    EXPECT_FLOAT_EQ(sampled->second, 6.0f);
}

TEST(ServerSampler, GreedySamplingReadsBestCandidateFromLastTopKRow) {
    auto values = make_buffer<float>(
        "topk_values",
        sandy::core::Shape({1, 2, 3}),
        sandy::core::DType::F32,
        {20.0f, 19.0f, 18.0f, 4.0f, 8.0f, 6.0f});
    auto indices = make_buffer<int64_t>(
        "topk_indices",
        sandy::core::Shape({1, 2, 3}),
        sandy::core::DType::I64,
        {1, 2, 3, 10, 20, 30});

    sandy::server::Sampler sampler;
    sandy::server::SamplingConfig config;
    config.topK = 3;
    config.temperature = 0.0f;
    auto sampled = sampler.sampleTopKLast(values, indices, config);

    ASSERT_TRUE(sampled) << sampled.error();
    EXPECT_EQ(sampled->first, 20);
    EXPECT_FLOAT_EQ(sampled->second, 8.0f);
}

TEST(ServerSampler, KeepsFullLogitsArgmaxFallback) {
    auto logits = make_buffer<float>(
        "logits",
        sandy::core::Shape({1, 2, 4}),
        sandy::core::DType::F32,
        {9.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 8.0f, 7.0f});

    sandy::server::Sampler sampler;
    auto sampled = sampler.argmaxLast(logits);

    ASSERT_TRUE(sampled) << sampled.error();
    EXPECT_EQ(sampled->first, 2);
    EXPECT_FLOAT_EQ(sampled->second, 8.0f);
}

TEST(ServerSampler, TopPCanRestrictSamplingToHighestProbabilityToken) {
    auto values = make_buffer<float>(
        "topk_values",
        sandy::core::Shape({1, 3}),
        sandy::core::DType::F32,
        {5.0f, 1.0f, 0.0f});
    auto indices = make_buffer<int64_t>(
        "topk_indices",
        sandy::core::Shape({1, 3}),
        sandy::core::DType::I64,
        {50, 10, 20});

    sandy::server::SamplingConfig config;
    config.topK = 3;
    config.topP = 0.5f;
    config.temperature = 1.0f;
    sandy::server::Sampler sampler(1234);

    for (int i = 0; i < 20; i++) {
        auto sampled = sampler.sampleTopKLast(values, indices, config);
        ASSERT_TRUE(sampled) << sampled.error();
        EXPECT_EQ(sampled->first, 50);
    }
}

TEST(ServerSampler, SeededTemperatureSamplingCanSelectMultipleCandidates) {
    auto values = make_buffer<float>(
        "topk_values",
        sandy::core::Shape({1, 2}),
        sandy::core::DType::F32,
        {0.0f, 0.0f});
    auto indices = make_buffer<int32_t>(
        "topk_indices",
        sandy::core::Shape({1, 2}),
        sandy::core::DType::I32,
        {7, 9});

    sandy::server::SamplingConfig config;
    config.topK = 2;
    config.topP = 1.0f;
    config.temperature = 1.0f;
    sandy::server::Sampler sampler(42);
    bool sawSeven = false;
    bool sawNine = false;
    for (int i = 0; i < 50; i++) {
        auto sampled = sampler.sampleTopKLast(values, indices, config);
        ASSERT_TRUE(sampled) << sampled.error();
        sawSeven |= sampled->first == 7;
        sawNine |= sampled->first == 9;
    }
    EXPECT_TRUE(sawSeven);
    EXPECT_TRUE(sawNine);
}

TEST(ServerSampler, RequestOverridesReplaceModelDefaults) {
    sandy::server::SamplingConfig defaults;
    defaults.topK = 64;
    defaults.topP = 0.95f;
    defaults.temperature = 1.0f;
    sandy::server::SamplingOverrides overrides;
    overrides.topP = 0.8f;
    overrides.temperature = 0.25f;

    auto resolved = sandy::server::resolveSamplingConfig(defaults, overrides);

    ASSERT_TRUE(resolved) << resolved.error();
    EXPECT_EQ(resolved->topK, 64);
    EXPECT_FLOAT_EQ(resolved->topP, 0.8f);
    EXPECT_FLOAT_EQ(resolved->temperature, 0.25f);
}

TEST(ServerSampler, RejectsInvalidSamplingParameters) {
    sandy::server::SamplingConfig config;
    config.topP = 0.0f;
    auto resolved = sandy::server::resolveSamplingConfig(config);
    ASSERT_FALSE(resolved);
    EXPECT_EQ(resolved.error(), "top_p must be > 0 and <= 1");
}

} // namespace
