#pragma once

#include "Result.h"
#include "Tensor.h"

namespace sandy::core {

Result<Shape> broadcast_shape(const Shape& lhs, const Shape& rhs);

} // namespace sandy::core
