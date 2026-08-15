#pragma once

#include "Result.h"
#include "Tensor.h"

#include <string>
#include <utility>

namespace sandy::core::detail {

template <typename T>
struct ScalarTraits;

template <>
struct ScalarTraits<float> {
    using Compute = float;
    static constexpr DType dtype = DType::F32;

    static Compute to_compute(float value) { return value; }
    static float from_compute(Compute value) { return value; }
};

template <typename ResultT, typename Fn>
ResultT dispatch_float_dtype(const char* opName, DType dtype, Fn&& fn) {
    switch (dtype) {
        case ScalarTraits<float>::dtype:
            return std::forward<Fn>(fn).template operator()<float>();
        default:
            return make_error(std::string(opName) + " unsupported dtype");
    }
}

template <typename ResultT, typename Fn>
ResultT dispatch_same_float_dtype(
        const char* opName,
        DType lhs,
        DType rhs,
        Fn&& fn) {
    if (lhs != rhs)
        return make_error(std::string(opName) + " operands must have same dtype");
    return dispatch_float_dtype<ResultT>(opName, lhs, std::forward<Fn>(fn));
}

} // namespace sandy::core::detail
