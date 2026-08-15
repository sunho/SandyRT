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

Result<Shape> matmul_batch_shape(const Shape& lhs, const Shape& rhs) {
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
        } else if (lhsDim > rhsDim && lhsDim % rhsDim == 0) {
            outDim = lhsDim;
        } else if (rhsDim > lhsDim && rhsDim % lhsDim == 0) {
            outDim = rhsDim;
        } else {
            return make_error(
                "cannot broadcast matmul batch shapes " + lhs.str() + " and " + rhs.str());
        }

        dims[static_cast<size_t>(outRank - 1 - i)] = outDim;
    }

    return Shape(std::move(dims));
}

Result<Shape> infer_reshape_shape(const Shape& inputShape, Shape requestedShape) {
    for (int i = 0; i < requestedShape.rank(); i++) {
        if (requestedShape.dim(i) < Shape::kDynamic)
            return make_error("reshape dimensions must be >= -1");
    }

    int64_t inputNumel = inputShape.numel();
    if (inputNumel < 0)
        return requestedShape;

    auto dims = requestedShape.dims();
    std::vector<int> inferIndices;
    int64_t knownProduct = 1;
    for (int i = 0; i < static_cast<int>(dims.size()); i++) {
        if (dims[static_cast<size_t>(i)] == Shape::kDynamic) {
            inferIndices.push_back(i);
        } else {
            knownProduct *= dims[static_cast<size_t>(i)];
        }
    }

    if (inferIndices.size() > 1) {
        bool canResolveLeadingDims =
            static_cast<int>(inferIndices.size()) <= inputShape.rank();
        for (int i = 0; canResolveLeadingDims && i < static_cast<int>(inferIndices.size()); i++)
            canResolveLeadingDims = inferIndices[static_cast<size_t>(i)] == i &&
                                    inputShape.dim(i) != Shape::kDynamic;

        if (!canResolveLeadingDims)
            return make_error("reshape may contain at most one -1 dimension");

        for (int i : inferIndices)
            dims[static_cast<size_t>(i)] = inputShape.dim(i);
        requestedShape = Shape(std::move(dims));
        if (requestedShape.numel() != inputNumel)
            return make_error("reshape cannot infer -1 dimensions");
    } else if (inferIndices.size() == 1) {
        int inferIndex = inferIndices[0];
        if (knownProduct == 0 || inputNumel % knownProduct != 0)
            return make_error("reshape cannot infer -1 dimension");
        dims[static_cast<size_t>(inferIndex)] = inputNumel / knownProduct;
        requestedShape = Shape(std::move(dims));
    } else if (requestedShape.numel() != inputNumel) {
        return make_error("reshape element count mismatch");
    }

    return requestedShape;
}

} // namespace sandy::core
