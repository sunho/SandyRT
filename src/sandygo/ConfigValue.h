#pragma once

#include "Tensor.h"

#include <cstdint>
#include <variant>

namespace sandy::sandygo {

using ConfigValue = std::variant<int64_t, core::DType>;

} // namespace sandy::sandygo
