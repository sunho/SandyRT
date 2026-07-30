#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace sandy::core {

enum class DType { F32, F16, BF16, I32, I64, U8 };

size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);

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
