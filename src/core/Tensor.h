#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace sandy::core {

enum class DType { F32, F16, BF16, I32, I64, U8 };

size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);

struct BFloat16 {
    uint16_t storage;
};

inline BFloat16 bfloat16_from_bits(uint16_t bits) {
    BFloat16 out;
    std::memcpy(&out, &bits, sizeof(bits));
    return out;
}

inline uint16_t bfloat16_bits(BFloat16 value) {
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float bfloat16_to_float(BFloat16 value) {
    uint32_t fbits = static_cast<uint32_t>(bfloat16_bits(value)) << 16;
    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(out));
    return out;
}

inline BFloat16 bfloat16_from_float(float value) {
    uint32_t fbits = 0;
    std::memcpy(&fbits, &value, sizeof(fbits));
    uint32_t lsb = (fbits >> 16) & 1u;
    uint16_t bits = static_cast<uint16_t>((fbits + 0x7fffu + lsb) >> 16);
    return bfloat16_from_bits(bits);
}

static_assert(sizeof(BFloat16) == 2, "BFloat16 must use 16-bit storage");

class Shape {
public:
    static constexpr int64_t kDynamic = -1;

    Shape() = default;
    explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

    int rank() const { return (int)dims_.size(); }
    int64_t dim(int i) const { return dims_[i]; }
    const std::vector<int64_t>& dims() const { return dims_; }
    bool is_dynamic(int i) const { return dims_[i] == kDynamic; }
    bool has_dynamic() const;

    int64_t numel() const;
    std::string str() const;

    bool operator==(const Shape& o) const { return dims_ == o.dims_; }
    bool operator!=(const Shape& o) const { return dims_ != o.dims_; }

private:
    std::vector<int64_t> dims_;
};

struct TensorDesc {
    TensorDesc() = default;
    TensorDesc(Shape shape, DType dtype) : shape(std::move(shape)), dtype(dtype) {}
    TensorDesc(std::string name, Shape shape, DType dtype)
        : name(std::move(name)), shape(std::move(shape)), dtype(dtype) {}
    std::string name;
    Shape shape;
    DType dtype;
};

} // namespace sandy::core
