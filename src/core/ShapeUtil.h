#pragma once

#include "Result.h"
#include "Tensor.h"

namespace sandy::core {

Result<Shape> broadcast_shape(const Shape& lhs, const Shape& rhs);
Result<Shape> infer_reshape_shape(const Shape& inputShape, Shape requestedShape);

} // namespace sandy::core
