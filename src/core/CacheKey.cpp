#include "CacheKey.h"

#include <bit>
#include <iomanip>
#include <sstream>
#include <utility>

namespace sandy::core {

namespace {

size_t stable_hash(std::span<const uint8_t> bytes) {
    constexpr uint64_t kOffset = 14695981039346656037ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t value = kOffset;
    for (auto byte : bytes) {
        value ^= byte;
        value *= kPrime;
    }
    if constexpr (sizeof(size_t) < sizeof(uint64_t))
        return static_cast<size_t>(value ^ (value >> 32));
    return static_cast<size_t>(value);
}

} // namespace

CacheKey::CacheKey(std::vector<uint8_t> bytes)
    : bytes_(std::move(bytes)), hash_(stable_hash(bytes_)) {}

std::string CacheKey::hex() const {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes_)
        out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

CacheKeyBuilder::CacheKeyBuilder(std::string_view domain) {
    addString("sandy-cache-key-v1");
    addString(domain);
}

void CacheKeyBuilder::appendU64(uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes_.push_back(static_cast<uint8_t>((value >> shift) & 0xffu));
}

CacheKeyBuilder& CacheKeyBuilder::addBool(bool value) {
    bytes_.push_back(value ? 1u : 0u);
    return *this;
}

CacheKeyBuilder& CacheKeyBuilder::addU32(uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes_.push_back(static_cast<uint8_t>((value >> shift) & 0xffu));
    return *this;
}

CacheKeyBuilder& CacheKeyBuilder::addU64(uint64_t value) {
    appendU64(value);
    return *this;
}

CacheKeyBuilder& CacheKeyBuilder::addI64(int64_t value) {
    appendU64(std::bit_cast<uint64_t>(value));
    return *this;
}

CacheKeyBuilder& CacheKeyBuilder::addBytes(std::span<const uint8_t> value) {
    appendU64(static_cast<uint64_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
    return *this;
}

CacheKeyBuilder& CacheKeyBuilder::addString(std::string_view value) {
    return addBytes(std::span(
        reinterpret_cast<const uint8_t*>(value.data()),
        value.size()));
}

CacheKeyBuilder& CacheKeyBuilder::addDType(DType dtype) {
    return addU32(static_cast<uint32_t>(dtype));
}

CacheKeyBuilder& CacheKeyBuilder::addShape(const Shape& shape) {
    addU64(static_cast<uint64_t>(shape.dims().size()));
    for (auto dim : shape.dims())
        addI64(dim);
    return *this;
}

CacheKeyBuilder& CacheKeyBuilder::addTensorDesc(const TensorDesc& desc) {
    addDType(desc.dtype);
    addShape(desc.shape);
    return *this;
}

CacheKey CacheKeyBuilder::finish() && {
    return CacheKey(std::move(bytes_));
}

} // namespace sandy::core

