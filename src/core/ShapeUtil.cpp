#include "ShapeUtil.h"

#include <algorithm>
#include <string>
#include <vector>

namespace sandy::core {

Result<Shape> broadcast_shape(const Shape& lhs, const Shape& rhs) {
    int outRank = std::max(lhs.rank(), rhs.rank());
    std::vector<int64_t> dims(static_cast<size_t>(outRank), 1);

    for (int i = 0; i < outRank; i++) {
        int lhsIndex = lhs.rank() - 1 - i;
        int rhsIndex = rhs.rank() - 1 - i;
        int64_t lhsDim = lhsIndex >= 0 ? lhs.dim(lhsIndex) : 1;
        int64_t rhsDim = rhsIndex >= 0 ? rhs.dim(rhsIndex) : 1;

        int64_t outDim = Shape::kDynamic;
        if (lhsDim == rhsDim) {
            outDim = lhsDim;
        } else if (lhsDim == 1) {
            outDim = rhsDim;
        } else if (rhsDim == 1) {
            outDim = lhsDim;
        } else if (lhsDim == Shape::kDynamic || rhsDim == Shape::kDynamic) {
            outDim = Shape::kDynamic;
        } else {
            return make_error(
                "cannot broadcast shapes " + lhs.str() + " and " + rhs.str());
        }

        dims[static_cast<size_t>(outRank - 1 - i)] = outDim;
    }

    return Shape(std::move(dims));
}

} // namespace sandy::core
