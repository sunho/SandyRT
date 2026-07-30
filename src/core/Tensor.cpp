#include "Tensor.h"

namespace sandy::core {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::U8: return 1;
    }
    return 0;
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::I32: return "i32";
        case DType::I64: return "i64";
        case DType::U8: return "u8";
    }
    return "unknown";
}

bool Shape::has_dynamic() const {
    for (auto d : dims_)
        if (d == kDynamic) return true;
    return false;
}

int64_t Shape::numel() const {
    if (dims_.empty()) return 0;
    int64_t n = 1;
    for (auto d : dims_) {
        if (d == kDynamic) return kDynamic;
        n *= d;
    }
    return n;
}

std::string Shape::str() const {
    std::string s = "[";
    for (size_t i = 0; i < dims_.size(); i++) {
        if (i > 0) s += ", ";
        if (dims_[i] == kDynamic)
            s += "?";
        else
            s += std::to_string(dims_[i]);
    }
    s += "]";
    return s;
}

} // namespace sandy::core
