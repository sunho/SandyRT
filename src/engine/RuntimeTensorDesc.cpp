#include "RuntimeTensorDesc.h"

#include "ShapeUtil.h"

#include <algorithm>
#include <utility>

namespace sandy::engine {

using namespace ir::kernel_ir;

bool RuntimeTensorDescs::has(ValueId value) const {
    return value < descs_.size() && descs_[value].has_value();
}

const core::TensorDesc& RuntimeTensorDescs::get(ValueId value) const {
    return *descs_[value];
}

Result<const core::TensorDesc*> RuntimeTensorDescs::lookup(ValueId value) const {
    if (!has(value))
        return make_error("missing runtime descriptor for value: " + std::to_string(value));
    return &*descs_[value];
}

Result<void> RuntimeTensorDescs::set(ValueId value, core::TensorDesc desc) {
    if (value >= descs_.size())
        return make_error("runtime descriptor value out of range: " + std::to_string(value));
    descs_[value] = std::move(desc);
    return {};
}

Result<void> verifyRuntimeTensorDesc(
        const core::TensorDesc& desc,
        const ValueType& type,
        const std::string& valueName) {
    if (desc.dtype != type.dtype) {
        return make_error(valueName + " dtype mismatch: expected " +
                          std::string(core::dtype_name(type.dtype)) + ", got " +
                          core::dtype_name(desc.dtype));
    }
    if (desc.shape.has_dynamic())
        return make_error(valueName + " runtime shape must be static");
    if (desc.shape.rank() != type.shape.rank()) {
        return make_error(valueName + " rank mismatch: expected " +
                          std::to_string(type.shape.rank()) + ", got " +
                          std::to_string(desc.shape.rank()));
    }
    for (int i = 0; i < type.shape.rank(); i++) {
        auto expected = type.shape.dim(i);
        if (expected != core::Shape::kDynamic && expected != desc.shape.dim(i)) {
            return make_error(valueName + " dimension " + std::to_string(i) +
                              " mismatch: expected " + std::to_string(expected) +
                              ", got " + std::to_string(desc.shape.dim(i)));
        }
    }
    return {};
}

namespace {

Result<core::TensorDesc> checkedDesc(
        const Graph& graph,
        ValueId value,
        core::Shape shape) {
    const auto& type = graph.value(value).type;
    core::TensorDesc desc(std::move(shape), type.dtype);
    auto verified = verifyRuntimeTensorDesc(
        desc, type, "value %" + std::to_string(value));
    if (!verified)
        return make_error(verified.error());
    return desc;
}

Result<core::TensorDesc> staticDesc(const ValueType& type) {
    if (type.shape.has_dynamic())
        return make_error("cannot resolve dynamic KernelIR value without runtime inputs");
    return core::TensorDesc(type.shape, type.dtype);
}

Result<core::TensorDesc> matmulDesc(
        const Graph& graph,
        const MatMulKernelOp& op,
        ValueId output,
        const RuntimeTensorDescs& descs) {
    auto lhs = descs.lookup(op.inputs()[0]);
    if (!lhs) return make_error(lhs.error());
    auto rhs = descs.lookup(op.inputs()[1]);
    if (!rhs) return make_error(rhs.error());
    auto lhsDims = (*lhs)->shape.dims();
    auto rhsDims = (*rhs)->shape.dims();
    if (lhsDims.size() < 2 || rhsDims.size() < 2)
        return make_error("matmul inputs must have rank >= 2");
    core::Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
    core::Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
    auto batch = core::matmul_batch_shape(lhsBatch, rhsBatch);
    if (!batch) return make_error(batch.error());
    auto outDims = batch->dims();
    const auto& lhsShape = (*lhs)->shape;
    const auto& rhsShape = (*rhs)->shape;
    outDims.push_back(lhsShape.dim(lhsShape.rank() - (op.transposeLhs() ? 1 : 2)));
    outDims.push_back(rhsShape.dim(rhsShape.rank() - (op.transposeRhs() ? 2 : 1)));
    return checkedDesc(graph, output, core::Shape(std::move(outDims)));
}

Result<core::TensorDesc> reductionDesc(
        const Graph& graph,
        const ReductionKernelOp& op,
        ValueId output,
        const RuntimeTensorDescs& descs) {
    auto input = descs.lookup(op.inputs()[0]);
    if (!input) return make_error(input.error());
    int rank = (*input)->shape.rank();
    std::vector<int> axes;
    for (auto axis : op.axes()) {
        if (axis < -rank || axis >= rank)
            return make_error("reduction axis out of range");
        int normalized = static_cast<int>(axis < 0 ? axis + rank : axis);
        if (std::find(axes.begin(), axes.end(), normalized) != axes.end())
            return make_error("reduction axes must be unique");
        axes.push_back(normalized);
    }
    auto dims = (*input)->shape.dims();
    if (op.keepDims()) {
        for (auto axis : axes) dims[static_cast<size_t>(axis)] = 1;
    } else {
        std::sort(axes.rbegin(), axes.rend());
        for (auto axis : axes) dims.erase(dims.begin() + axis);
    }
    return checkedDesc(graph, output, core::Shape(std::move(dims)));
}

Result<core::TensorDesc> topkDesc(
        const Graph& graph,
        const TopKKernelOp& op,
        ValueId output,
        const RuntimeTensorDescs& descs) {
    auto input = descs.lookup(op.inputs()[0]);
    if (!input) return make_error(input.error());
    int rank = (*input)->shape.rank();
    auto axis = op.axis();
    if (axis < -rank || axis >= rank) return make_error("topk axis out of range");
    axis = axis < 0 ? axis + rank : axis;
    auto dims = (*input)->shape.dims();
    dims[static_cast<size_t>(axis)] = op.k();
    return checkedDesc(graph, output, core::Shape(std::move(dims)));
}

Result<core::TensorDesc> moeGatherDesc(
        const Graph& graph,
        const MoeGatherKernelOp& op,
        ValueId output,
        const RuntimeTensorDescs& descs) {
    auto x = descs.lookup(op.inputs()[0]);
    if (!x) return make_error(x.error());
    const auto& shape = (*x)->shape;
    int rank = shape.rank();
    if (rank != 2 && rank != 3) return make_error("moe_gather input rank must be 2 or 3");
    auto rows = shape.dim(rank == 3 ? 1 : 0) * op.topK();
    auto hidden = shape.dim(rank - 1);
    auto outputs = op.outputs();
    core::Shape outShape;
    if (output == outputs[0])
        outShape = rank == 3 ? core::Shape({shape.dim(0), rows, hidden}) : core::Shape({rows, hidden});
    else if (output == outputs[1] || output == outputs[2])
        outShape = rank == 3 ? core::Shape({shape.dim(0), rows}) : core::Shape({rows});
    else if (output == outputs[3])
        outShape = rank == 3 ? core::Shape({shape.dim(0), op.numExperts() + 1})
                             : core::Shape({op.numExperts() + 1});
    else
        return make_error("moe_gather unknown output");
    return checkedDesc(graph, output, std::move(outShape));
}

Result<core::TensorDesc> moeMatmulDesc(
        const Graph& graph,
        const MoeMatMulKernelOp& op,
        ValueId output,
        const RuntimeTensorDescs& descs) {
    auto x = descs.lookup(op.inputs()[0]);
    if (!x) return make_error(x.error());
    auto weight = descs.lookup(op.inputs()[2]);
    if (!weight) return make_error(weight.error());
    const auto& xShape = (*x)->shape;
    const auto& wShape = (*weight)->shape;
    int rank = xShape.rank();
    if ((rank != 2 && rank != 3) || wShape.rank() != 3)
        return make_error("invalid moe_matmul runtime rank");
    auto rows = xShape.dim(rank - 2);
    auto features = wShape.dim(op.transposeRhs() ? 1 : 2);
    auto shape = rank == 3 ? core::Shape({xShape.dim(0), rows, features})
                           : core::Shape({rows, features});
    return checkedDesc(graph, output, std::move(shape));
}

Result<core::TensorDesc> inferOutput(
        const Graph& graph,
        const Op& op,
        ValueId output,
        const RuntimeTensorDescs& descs) {
    auto inputDesc = [&](size_t index) { return descs.lookup(op.inputs()[index]); };
    switch (op.kind()) {
        case OpKind::Input:
            return make_error("InputOp descriptors must be supplied by the invocation");
        case OpKind::TensorTupleCreate:
        case OpKind::PagedAppend:
            return make_error("op does not produce a dense tensor descriptor");
        case OpKind::DeviceTransfer: {
            auto input = inputDesc(0); if (!input) return make_error(input.error());
            return checkedDesc(graph, output, (*input)->shape);
        }
        case OpKind::ElementwiseKernel: {
            if (op.inputs().empty()) return staticDesc(graph.value(output).type);
            auto lhs = inputDesc(0); if (!lhs) return make_error(lhs.error());
            if (op.inputs().size() == 1) return checkedDesc(graph, output, (*lhs)->shape);
            if (op.inputs().size() == 2) {
                auto rhs = inputDesc(1); if (!rhs) return make_error(rhs.error());
                auto shape = core::broadcast_shape((*lhs)->shape, (*rhs)->shape);
                if (!shape) return make_error(shape.error());
                return checkedDesc(graph, output, shape.take());
            }
            return staticDesc(graph.value(output).type);
        }
        case OpKind::LayoutTransform: {
            const auto& layout = static_cast<const LayoutTransformOp&>(op);
            auto input = inputDesc(0); if (!input) return make_error(input.error());
            auto dims = (*input)->shape.dims();
            switch (layout.transform()) {
                case LayoutTransformKind::Reshape: {
                    auto shape = core::infer_reshape_shape((*input)->shape, core::Shape(layout.dims()));
                    if (!shape) return make_error(shape.error());
                    return checkedDesc(graph, output, shape.take());
                }
                case LayoutTransformKind::Transpose:
                    if (dims.size() < 2) return make_error("transpose input rank must be >= 2");
                    std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
                    break;
                case LayoutTransformKind::Permute: {
                    std::vector<int64_t> permuted;
                    for (auto axis : layout.dims()) permuted.push_back((*input)->shape.dim(static_cast<int>(axis)));
                    dims = std::move(permuted);
                    break;
                }
                case LayoutTransformKind::Contiguous: break;
            }
            return checkedDesc(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::MatMulKernel:
            return matmulDesc(graph, static_cast<const MatMulKernelOp&>(op), output, descs);
        case OpKind::GatherKernel: {
            auto ids = inputDesc(0); if (!ids) return make_error(ids.error());
            auto table = inputDesc(1); if (!table) return make_error(table.error());
            auto dims = (*ids)->shape.dims();
            if ((*table)->shape.rank() == 2) dims.push_back((*table)->shape.dim(1));
            else if ((*table)->shape.rank() != 1) return make_error("gather table must have rank 1 or 2");
            return checkedDesc(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::SoftmaxKernel:
        case OpKind::NormKernel:
        case OpKind::RoPEKernel: {
            auto input = inputDesc(0); if (!input) return make_error(input.error());
            return checkedDesc(graph, output, (*input)->shape);
        }
        case OpKind::SlidingQueryKeyScoreKernel: {
            auto q = inputDesc(0); if (!q) return make_error(q.error());
            auto k = inputDesc(1); if (!k) return make_error(k.error());
            int rank = (*q)->shape.rank();
            std::vector<int64_t> dims;
            if (rank == 4) dims.push_back((*q)->shape.dim(0));
            dims.push_back((*q)->shape.dim(rank - 3));
            dims.push_back((*q)->shape.dim(rank - 2));
            dims.push_back((*k)->shape.dim(rank - 2));
            return checkedDesc(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::AttentionKernel: {
            auto q = inputDesc(0); if (!q) return make_error(q.error());
            auto v = inputDesc(2); if (!v) return make_error(v.error());
            auto dims = (*q)->shape.dims();
            dims.back() = (*v)->shape.dim((*v)->shape.rank() - 1);
            return checkedDesc(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::LinearKernel: {
            auto x = inputDesc(0); if (!x) return make_error(x.error());
            auto weight = inputDesc(1); if (!weight) return make_error(weight.error());
            auto dims = (*x)->shape.dims();
            dims.back() = (*weight)->shape.dim(0);
            return checkedDesc(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::TopKKernel:
            return topkDesc(graph, static_cast<const TopKKernelOp&>(op), output, descs);
        case OpKind::MoeGatherKernel:
            return moeGatherDesc(graph, static_cast<const MoeGatherKernelOp&>(op), output, descs);
        case OpKind::MoeMatMulKernel:
            return moeMatmulDesc(graph, static_cast<const MoeMatMulKernelOp&>(op), output, descs);
        case OpKind::MoeScatterSumKernel: {
            auto reference = inputDesc(3); if (!reference) return make_error(reference.error());
            return checkedDesc(graph, output, (*reference)->shape);
        }
        case OpKind::ReductionKernel:
            return reductionDesc(graph, static_cast<const ReductionKernelOp&>(op), output, descs);
    }
    return make_error("unknown KernelIR op kind");
}

Result<void> applyPagedAppend(
        const Graph& graph,
        const PagedAppendOp& append,
        RuntimeTensorDescs& descs) {
    auto cache = descs.lookup(append.cache());
    if (!cache) return make_error(cache.error());
    auto chunk = descs.lookup(append.chunk());
    if (!chunk) return make_error(chunk.error());
    const auto& type = graph.value(append.cache()).type;
    int64_t axis = type.paged.growDim;
    if (axis < 0 || axis >= (*cache)->shape.rank())
        return make_error("paged append grow dimension out of range");
    if ((*cache)->dtype != (*chunk)->dtype ||
        (*cache)->shape.rank() != (*chunk)->shape.rank())
        return make_error("paged append runtime descriptor mismatch");
    for (int i = 0; i < (*cache)->shape.rank(); ++i) {
        if (i != axis && (*cache)->shape.dim(i) != (*chunk)->shape.dim(i))
            return make_error("paged append runtime non-grow dimension mismatch");
    }
    if ((*chunk)->shape.dim(static_cast<int>(axis)) == 0)
        return make_error("paged append runtime grow dimension must be non-zero");
    auto dims = (*cache)->shape.dims();
    dims[static_cast<size_t>(axis)] += (*chunk)->shape.dim(static_cast<int>(axis));
    auto desc = checkedDesc(graph, append.cache(), core::Shape(std::move(dims)));
    if (!desc) return make_error(desc.error());
    return descs.set(append.cache(), desc.take());
}

} // namespace

Result<RuntimeTensorDescs> inferRuntimeTensorDescs(
        const Graph& graph,
        RuntimeTensorDescs descs) {
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (op.kind() == OpKind::Input) {
            for (auto output : op.outputs()) {
                auto desc = descs.lookup(output);
                if (!desc) return make_error(desc.error());
                auto verified = verifyRuntimeTensorDesc(
                    **desc, graph.value(output).type, "value %" + std::to_string(output));
                if (!verified) return make_error(verified.error());
            }
            continue;
        }
        if (op.kind() == OpKind::TensorTupleCreate)
            continue;
        if (op.kind() == OpKind::PagedAppend) {
            auto applied = applyPagedAppend(graph, static_cast<const PagedAppendOp&>(op), descs);
            if (!applied) return make_error(applied.error());
            continue;
        }
        for (auto output : op.outputs()) {
            auto desc = inferOutput(graph, op, output, descs);
            if (!desc) return make_error(desc.error());
            auto set = descs.set(output, desc.take());
            if (!set) return make_error(set.error());
        }
    }
    return descs;
}

} // namespace sandy::engine
