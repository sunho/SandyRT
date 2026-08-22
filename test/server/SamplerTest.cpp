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

TEST(ServerSampler, RejectsTopKWidthOtherThanOne) {
    auto values = make_buffer<float>(
        "topk_values",
        sandy::core::Shape({1, 2}),
        sandy::core::DType::F32,
        {4.0f, 5.0f});
    auto indices = make_buffer<int64_t>(
        "topk_indices",
        sandy::core::Shape({1, 2}),
        sandy::core::DType::I64,
        {10, 20});

    sandy::server::Sampler sampler;
    auto sampled = sampler.topkLast(values, indices);

    ASSERT_FALSE(sampled);
    EXPECT_EQ(sampled.error(), "topk outputs must have a final dimension of 1");
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

} // namespace
