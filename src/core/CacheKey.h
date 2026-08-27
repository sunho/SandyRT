#pragma once

#include "Tensor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sandy::core {

// CacheKey keeps its canonical bytes so equality remains collision-free. The
// cached hash is only an unordered-container accelerator.
class CacheKey {
public:
    CacheKey() = default;
    explicit CacheKey(std::vector<uint8_t> bytes);

    std::span<const uint8_t> bytes() const { return bytes_; }
    size_t hash() const { return hash_; }
    std::string hex() const;

    bool operator==(const CacheKey& other) const { return bytes_ == other.bytes_; }

private:
    std::vector<uint8_t> bytes_;
    size_t hash_ = 0;
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const { return key.hash(); }
};

class CacheKeyBuilder {
public:
    explicit CacheKeyBuilder(std::string_view domain);

    CacheKeyBuilder& addBool(bool value);
    CacheKeyBuilder& addU32(uint32_t value);
    CacheKeyBuilder& addU64(uint64_t value);
    CacheKeyBuilder& addI64(int64_t value);
    CacheKeyBuilder& addBytes(std::span<const uint8_t> value);
    CacheKeyBuilder& addString(std::string_view value);
    CacheKeyBuilder& addDType(DType dtype);
    CacheKeyBuilder& addShape(const Shape& shape);
    CacheKeyBuilder& addTensorDesc(const TensorDesc& desc);

    CacheKey finish() &&;

private:
    void appendU64(uint64_t value);

    std::vector<uint8_t> bytes_;
};

} // namespace sandy::core

