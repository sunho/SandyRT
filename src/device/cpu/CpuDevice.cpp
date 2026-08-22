#include "CpuDevice.h"

#include "ShapeUtil.h"
#include "TensorCalc.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sandy::device {

namespace {

class CpuTensorBuffer final : public core::TensorBuffer {
public:
    CpuTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

Result<size_t> tensor_byte_size(const core::TensorDesc& desc) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error("cpu device buffer must have static shape");
    return static_cast<size_t>(numel) * core::dtype_size(desc.dtype);
}

Result<int64_t> checked_product(
        const core::Shape& shape,
        int begin,
        int end) {
    int64_t product = 1;
    for (int i = begin; i < end; i++) {
        auto dim = shape.dim(i);
        if (dim <= 0)
            return make_error("paged tensor pool dimensions must be static and positive");
        if (product > std::numeric_limits<int64_t>::max() / dim)
            return make_error("paged tensor element count overflow");
        product *= dim;
    }
    return product;
}

Result<void> validate_paged_pool_desc(const DevicePagedPoolDesc& desc) {
    const auto& shape = desc.templateDesc.shape;
    if (shape.rank() <= 0)
        return make_error("paged tensor pool rank must be >= 1");
    if (desc.growDim < 0 || desc.growDim >= shape.rank())
        return make_error("paged tensor pool grow_dim out of range");
    if (desc.pageSize <= 0)
        return make_error("paged tensor pool page_size must be > 0");
    if (desc.initialPages < 0)
        return make_error("paged tensor pool initial page count must be >= 0");
    if (desc.maxPages >= 0 && desc.initialPages > desc.maxPages)
        return make_error("paged tensor pool initial page count exceeds max page count");

    for (int i = 0; i < shape.rank(); i++) {
        auto dim = shape.dim(i);
        if (i == desc.growDim) {
            if (dim != core::Shape::kDynamic && dim < 0)
                return make_error("paged tensor pool grow dimension must be dynamic or non-negative");
            continue;
        }
        if (dim <= 0)
            return make_error("paged tensor pool non-grow dimensions must be static and positive");
    }
    return {};
}

Result<void> validate_paged_logical_shape(
        const DevicePagedPoolDesc& pool,
        const core::Shape& logicalShape) {
    const auto& templateShape = pool.templateDesc.shape;
    if (logicalShape.rank() != templateShape.rank())
        return make_error("paged tensor logical shape rank mismatch");
    for (int i = 0; i < logicalShape.rank(); i++) {
        auto dim = logicalShape.dim(i);
        if (dim < 0)
            return make_error("paged tensor logical shape must be concrete");
        if (i == pool.growDim)
            continue;
        if (dim != templateShape.dim(i))
            return make_error("paged tensor logical shape non-grow dimension mismatch");
    }
    return {};
}

int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

core::Shape shape_with_grow_length(core::Shape shape, int64_t growDim, int64_t growLength) {
    auto dims = shape.dims();
    dims[static_cast<size_t>(growDim)] = growLength;
    return core::Shape(std::move(dims));
}

struct SimpleElementwiseKernel {
    ir::kernel_ir::ScalarOp scalarOp = ir::kernel_ir::ScalarOp::Constant;
    double constant = 0.0;
};

const ir::kernel_ir::ScalarNode* find_scalar(
        const std::vector<ir::kernel_ir::ScalarNode>& scalars,
        ir::kernel_ir::ScalarId id) {
    auto it = std::find_if(
        scalars.begin(),
        scalars.end(),
        [&](const ir::kernel_ir::ScalarNode& scalar) {
            return scalar.id == id;
        });
    if (it == scalars.end())
        return nullptr;
    return &*it;
}

bool is_load_of(
        const ir::kernel_ir::ScalarNode* scalar,
        uint32_t inputIndex) {
    return scalar &&
           scalar->op == ir::kernel_ir::ScalarOp::Load &&
           scalar->inputIndex == inputIndex &&
           scalar->operands.empty();
}

Result<SimpleElementwiseKernel> validate_simple_elementwise_kernel(
        const ir::kernel_ir::ElementwiseKernelOp& elementwise) {
    const auto& scalars = elementwise.scalars();
    auto* terminal = find_scalar(scalars, elementwise.result());
    if (!terminal)
        return make_error("cpu device elementwise result references missing scalar");

    SimpleElementwiseKernel simple;
    simple.scalarOp = terminal->op;
    simple.constant = terminal->constant;

    switch (terminal->op) {
        case ir::kernel_ir::ScalarOp::Constant:
            if (!elementwise.inputs().empty() || scalars.size() != 1 ||
                !terminal->operands.empty()) {
                return make_error("cpu device elementwise constant must be a single scalar op");
            }
            return simple;

        case ir::kernel_ir::ScalarOp::ReLU:
        case ir::kernel_ir::ScalarOp::Sqrt:
        case ir::kernel_ir::ScalarOp::Tanh: {
            if (elementwise.inputs().size() != 1 || scalars.size() != 2 ||
                terminal->operands.size() != 1) {
                return make_error("cpu device elementwise unary kernel must be a single op");
            }
            auto* load = find_scalar(scalars, terminal->operands[0]);
            if (!is_load_of(load, 0)) {
                return make_error("cpu device elementwise unary op must consume input0 load");
            }
            return simple;
        }

        case ir::kernel_ir::ScalarOp::Add:
        case ir::kernel_ir::ScalarOp::Div:
        case ir::kernel_ir::ScalarOp::Mul: {
            if (elementwise.inputs().size() != 2 || scalars.size() != 3 ||
                terminal->operands.size() != 2) {
                return make_error("cpu device elementwise binary kernel must be a single op");
            }
            auto* lhs = find_scalar(scalars, terminal->operands[0]);
            auto* rhs = find_scalar(scalars, terminal->operands[1]);
            if (!is_load_of(lhs, 0) || !is_load_of(rhs, 1)) {
                return make_error(
                    "cpu device elementwise binary op must consume input0 and input1 loads");
            }
            return simple;
        }

        default:
            return make_error("cpu device unsupported elementwise scalar op");
    }
}

} // namespace

Result<DeviceCompiledGraphId> CpuDevice::compile(const ir::kernel_ir::Graph& graph) {
    auto verify = graph.verify();
    if (!verify)
        return make_error(verify.error());

    CpuDeviceGraph compiled;
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        CpuDeviceKernel kernel;
        kernel.kind = op.kind();
        kernel.inputCount = op.inputs().size();
        kernel.outputCount = op.outputs().size();

        switch (op.kind()) {
            case ir::kernel_ir::OpKind::Input:
            case ir::kernel_ir::OpKind::TensorTupleCreate:
            case ir::kernel_ir::OpKind::PagedAppend:
            case ir::kernel_ir::OpKind::DeviceTransfer:
                break;
            case ir::kernel_ir::OpKind::LayoutTransform: {
                const auto& layout = static_cast<const ir::kernel_ir::LayoutTransformOp&>(op);
                kernel.layoutTransform = layout.transform();
                kernel.dims = layout.dims();
                break;
            }
            case ir::kernel_ir::OpKind::ElementwiseKernel: {
                const auto& elementwise = static_cast<const ir::kernel_ir::ElementwiseKernelOp&>(op);
                auto simple = validate_simple_elementwise_kernel(elementwise);
                if (!simple)
                    return make_error(simple.error());
                kernel.scalarOp = simple->scalarOp;
                kernel.constant = simple->constant;
                break;
            }
            case ir::kernel_ir::OpKind::LinearKernel:
                break;
            case ir::kernel_ir::OpKind::MatMulKernel: {
                const auto& matmul = static_cast<const ir::kernel_ir::MatMulKernelOp&>(op);
                kernel.transposeLhs = matmul.transposeLhs();
                kernel.transposeRhs = matmul.transposeRhs();
                break;
            }
            case ir::kernel_ir::OpKind::GatherKernel:
                break;
            case ir::kernel_ir::OpKind::SoftmaxKernel: {
                const auto& softmax = static_cast<const ir::kernel_ir::SoftmaxKernelOp&>(op);
                kernel.axis = softmax.axis();
                break;
            }
            case ir::kernel_ir::OpKind::TopKKernel: {
                const auto& topk = static_cast<const ir::kernel_ir::TopKKernelOp&>(op);
                kernel.k = topk.k();
                kernel.axis = topk.axis();
                break;
            }
            case ir::kernel_ir::OpKind::NormKernel: {
                const auto& norm = static_cast<const ir::kernel_ir::NormKernelOp&>(op);
                kernel.norm = norm.norm();
                kernel.epsilon = norm.epsilon();
                break;
            }
            case ir::kernel_ir::OpKind::RoPEKernel: {
                const auto& rope = static_cast<const ir::kernel_ir::RoPEKernelOp&>(op);
                kernel.theta = rope.theta();
                kernel.rotaryDim = rope.rotaryDim();
                kernel.splitHalf = rope.splitHalf();
                break;
            }
            case ir::kernel_ir::OpKind::SlidingQueryKeyScoreKernel: {
                const auto& score =
                    static_cast<const ir::kernel_ir::SlidingQueryKeyScoreKernelOp&>(op);
                kernel.window = score.window();
                kernel.scale = score.scale();
                break;
            }
            case ir::kernel_ir::OpKind::AttentionKernel: {
                const auto& attention =
                    static_cast<const ir::kernel_ir::AttentionKernelOp&>(op);
                kernel.window = attention.window();
                kernel.scale = attention.scale();
                break;
            }
            case ir::kernel_ir::OpKind::ReductionKernel: {
                const auto& reduction = static_cast<const ir::kernel_ir::ReductionKernelOp&>(op);
                if (reduction.axes().size() != 1)
                    return make_error("cpu device reduction expects exactly one axis");
                kernel.reduce = reduction.reduce();
                kernel.axis = reduction.axes()[0];
                kernel.keepDims = reduction.keepDims();
                break;
            }
            case ir::kernel_ir::OpKind::MoeGatherKernel: {
                const auto& gather = static_cast<const ir::kernel_ir::MoeGatherKernelOp&>(op);
                kernel.numExperts = gather.numExperts();
                kernel.topK = gather.topK();
                break;
            }
            case ir::kernel_ir::OpKind::MoeMatMulKernel: {
                const auto& matmul = static_cast<const ir::kernel_ir::MoeMatMulKernelOp&>(op);
                kernel.transposeRhs = matmul.transposeRhs();
                break;
            }
            case ir::kernel_ir::OpKind::MoeScatterSumKernel:
                break;
        }

        compiled.kernels[op.id()] = std::move(kernel);
    }

    auto id = nextGraphId_++;
    graphs_[id] = std::move(compiled);
    return id;
}

Result<DeviceBufferId> CpuDevice::alloc(core::TensorDesc desc) {
    auto bytes = tensor_byte_size(desc);
    if (!bytes)
        return make_error(bytes.error());

    CpuDeviceBuffer buffer;
    buffer.desc = std::move(desc);
    buffer.data.resize(bytes.take());

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<void> CpuDevice::dealloc(DeviceBufferId buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    buffers_.erase(it);
    return {};
}

Result<DeviceBufferId> CpuDevice::load(core::TensorBuffer& src) {
    auto access = src.access();
    if (!access)
        return make_error(access.error());

    CpuDeviceBuffer buffer;
    buffer.desc = (*access).desc();
    buffer.borrowed.emplace(access.take());

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<DevicePagedPoolId> CpuDevice::createPagedPoolImpl(DevicePagedPoolDesc desc) {
    auto valid = validate_paged_pool_desc(desc);
    if (!valid)
        return make_error(valid.error());

    auto outer = checked_product(desc.templateDesc.shape, 0, static_cast<int>(desc.growDim));
    if (!outer)
        return make_error(outer.error());
    auto inner = checked_product(
        desc.templateDesc.shape,
        static_cast<int>(desc.growDim) + 1,
        desc.templateDesc.shape.rank());
    if (!inner)
        return make_error(inner.error());
    if (*outer > std::numeric_limits<int64_t>::max() / desc.pageSize)
        return make_error("paged tensor page element count overflow");
    auto outerPage = *outer * desc.pageSize;
    if (outerPage > std::numeric_limits<int64_t>::max() / *inner)
        return make_error("paged tensor page element count overflow");
    auto pageElementCount = outerPage * *inner;
    auto elementSize = core::dtype_size(desc.templateDesc.dtype);
    if (pageElementCount > static_cast<int64_t>(std::numeric_limits<size_t>::max() / elementSize))
        return make_error("paged tensor page byte size overflow");

    CpuPagedPool pool;
    pool.desc = std::move(desc);
    pool.pageElementCount = pageElementCount;
    pool.pageBytes = static_cast<size_t>(pageElementCount) * elementSize;

    auto maxPages = pool.desc.maxPages < 0 ? 0 : static_cast<size_t>(pool.desc.maxPages);
    auto initialized = pool.pages.initialize(
        pool.pageBytes,
        static_cast<size_t>(pool.desc.initialPages),
        maxPages);
    if (!initialized)
        return make_error(initialized.error());

    auto id = nextPagedPoolId_++;
    pagedPools_[id] = std::move(pool);
    return id;
}

Result<void> CpuDevice::destroyPagedPool(DevicePagedPoolId pool) {
    auto it = pagedPools_.find(pool);
    if (it == pagedPools_.end())
        return make_error("cpu device paged pool not found");
    for (const auto& item : pagedTensors_) {
        if (item.second.pool == pool)
            return make_error("cpu device paged pool still has live tensors");
    }
    pagedPools_.erase(it);
    return {};
}

Result<DevicePagedTensorId> CpuDevice::allocPaged(
        DevicePagedPoolId poolId,
        core::Shape logicalShape) {
    auto poolIt = pagedPools_.find(poolId);
    if (poolIt == pagedPools_.end())
        return make_error("cpu device paged pool not found");

    auto valid = validate_paged_logical_shape(poolIt->second.desc, logicalShape);
    if (!valid)
        return make_error(valid.error());

    CpuPagedTensor tensor;
    tensor.pool = poolId;
    tensor.growLength = logicalShape.dim(static_cast<int>(poolIt->second.desc.growDim));
    tensor.logicalShape = std::move(logicalShape);

    auto id = nextPagedTensorId_++;
    pagedTensors_[id] = std::move(tensor);

    auto requiredPages = ceil_div(
        pagedTensors_[id].growLength,
        poolIt->second.desc.pageSize);
    auto reserved = reservePaged(id, requiredPages);
    if (!reserved) {
        pagedTensors_.erase(id);
        return make_error(reserved.error());
    }
    return id;
}

Result<void> CpuDevice::deallocPaged(DevicePagedTensorId tensorId) {
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cpu device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool);
    if (poolIt == pagedPools_.end())
        return make_error("cpu device paged pool not found");

    for (auto page : tensorIt->second.pageIndices) {
        auto deallocated = poolIt->second.pages.deallocate(page);
        if (!deallocated)
            return make_error(deallocated.error());
    }
    pagedTensors_.erase(tensorIt);
    return {};
}

Result<void> CpuDevice::reservePaged(DevicePagedTensorId tensorId, int64_t pageCount) {
    if (pageCount < 0)
        return make_error("cpu device paged tensor reserve page count must be >= 0");
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cpu device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool);
    if (poolIt == pagedPools_.end())
        return make_error("cpu device paged pool not found");

    auto& pages = tensorIt->second.pageIndices;
    while (static_cast<int64_t>(pages.size()) < pageCount) {
        auto page = poolIt->second.pages.allocate();
        if (!page)
            return make_error(page.error());
        pages.push_back(*page);
    }
    return {};
}

Result<void> CpuDevice::appendPaged(
        DevicePagedTensorId tensorId,
        core::TensorBuffer& denseChunk) {
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cpu device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool);
    if (poolIt == pagedPools_.end())
        return make_error("cpu device paged pool not found");

    auto access = denseChunk.access();
    if (!access)
        return make_error(access.error());
    const auto& chunkDesc = (*access).desc();
    const auto& pool = poolIt->second;
    auto valid = validate_paged_logical_shape(pool.desc, chunkDesc.shape);
    if (!valid)
        return make_error(valid.error());
    if (chunkDesc.dtype != pool.desc.templateDesc.dtype) {
        return make_error("cpu device paged append dtype mismatch");
    }

    int growDim = static_cast<int>(pool.desc.growDim);
    auto chunkGrowLength = chunkDesc.shape.dim(growDim);
    if (chunkGrowLength <= 0)
        return make_error("cpu device paged append grow dimension must be > 0");

    auto chunkNumel = chunkDesc.shape.numel();
    if (chunkNumel < 0)
        return make_error("cpu device paged append chunk shape must be concrete");
    auto expectedBytes = static_cast<size_t>(chunkNumel) * core::dtype_size(chunkDesc.dtype);
    if ((*access).data().size() != expectedBytes)
        return make_error("cpu device paged append chunk byte size mismatch");

    auto& tensor = tensorIt->second;
    auto oldGrowLength = tensor.growLength;
    if (oldGrowLength > std::numeric_limits<int64_t>::max() - chunkGrowLength)
        return make_error("cpu device paged append grow length overflow");
    auto newGrowLength = oldGrowLength + chunkGrowLength;
    auto requiredPages = ceil_div(newGrowLength, pool.desc.pageSize);
    auto reserved = reservePaged(tensorId, requiredPages);
    if (!reserved)
        return make_error(reserved.error());

    auto prefixCount = checked_product(
        pool.desc.templateDesc.shape,
        0,
        static_cast<int>(pool.desc.growDim));
    if (!prefixCount)
        return make_error(prefixCount.error());
    auto trailingCount = checked_product(
        pool.desc.templateDesc.shape,
        static_cast<int>(pool.desc.growDim) + 1,
        pool.desc.templateDesc.shape.rank());
    if (!trailingCount)
        return make_error(trailingCount.error());

    auto elementSize = core::dtype_size(chunkDesc.dtype);
    auto trailingBytes = static_cast<size_t>(*trailingCount) * elementSize;
    auto source = (*access).data();

    for (int64_t prefix = 0; prefix < *prefixCount; prefix++) {
        for (int64_t chunkGrow = 0; chunkGrow < chunkGrowLength; chunkGrow++) {
            auto dstGrow = oldGrowLength + chunkGrow;
            auto dstPageOrdinal = dstGrow / pool.desc.pageSize;
            auto dstGrowInPage = dstGrow % pool.desc.pageSize;
            auto pageIndex = tensor.pageIndices[static_cast<size_t>(dstPageOrdinal)];

            auto page = poolIt->second.pages.page(pageIndex);
            if (!page)
                return make_error(page.error());

            auto sourceElement =
                (prefix * chunkGrowLength + chunkGrow) * *trailingCount;
            auto dstElement =
                (prefix * pool.desc.pageSize + dstGrowInPage) * *trailingCount;
            std::memcpy(
                page->data() + static_cast<size_t>(dstElement) * elementSize,
                source.data() + static_cast<size_t>(sourceElement) * elementSize,
                trailingBytes);
        }
    }

    tensor.growLength = newGrowLength;
    tensor.logicalShape = shape_with_grow_length(
        pool.desc.templateDesc.shape,
        pool.desc.growDim,
        newGrowLength);
    return {};
}

Result<DevicePagedTensorMeta> CpuDevice::pagedMeta(DevicePagedTensorId tensorId) const {
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cpu device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool);
    if (poolIt == pagedPools_.end())
        return make_error("cpu device paged pool not found");

    DevicePagedTensorMeta meta;
    meta.pool = tensorIt->second.pool;
    meta.logicalDesc = core::TensorDesc(
        tensorIt->second.logicalShape,
        poolIt->second.desc.templateDesc.dtype);
    meta.growDim = poolIt->second.desc.growDim;
    meta.pageSize = poolIt->second.desc.pageSize;
    meta.growLength = tensorIt->second.growLength;
    meta.pageCount = static_cast<int64_t>(tensorIt->second.pageIndices.size());
    meta.pageElementCount = poolIt->second.pageElementCount;
    return meta;
}

Result<TensorBufferPtr> CpuDevice::read(DeviceTensorView src) {
    auto it = buffers_.find(src.buffer);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    auto data = it->second.borrowed ? it->second.borrowed->data() : std::span<const uint8_t>(it->second.data);
    auto source = core::make_tensor_ref(
        src.view.desc,
        data,
        src.view.strides,
        src.view.storageOffset);
    if (!source)
        return make_error(source.error());

    auto numel = src.view.desc.shape.numel();
    if (numel < 0)
        return make_error("cpu device cannot read dynamic view");
    std::vector<uint8_t> outData(
        static_cast<size_t>(numel) * core::dtype_size(src.view.desc.dtype));

    auto elementSize = core::dtype_size(src.view.desc.dtype);
    for (size_t i = 0; i < static_cast<size_t>(numel); i++) {
        auto storageIndex = source->storage_index(i);
        std::memcpy(
            outData.data() + i * elementSize,
            data.data() + storageIndex * elementSize,
            elementSize);
    }

    TensorBufferPtr buffer = std::make_shared<CpuTensorBuffer>(
        src.view.desc,
        std::move(outData));
    return buffer;
}

Result<TensorBufferPtr> CpuDevice::read(DevicePagedTensorView src) {
    auto tensorIt = pagedTensors_.find(src.tensor);
    if (tensorIt == pagedTensors_.end())
        return make_error("cpu device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool);
    if (poolIt == pagedPools_.end())
        return make_error("cpu device paged pool not found");

    const auto& pool = poolIt->second;
    const auto& tensor = tensorIt->second;
    auto desc = core::TensorDesc(
        tensor.logicalShape,
        pool.desc.templateDesc.dtype);
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error("cpu device cannot read dynamic paged tensor");

    std::vector<uint8_t> outData(
        static_cast<size_t>(numel) * core::dtype_size(desc.dtype));
    if (tensor.growLength == 0) {
        return TensorBufferPtr(std::make_shared<CpuTensorBuffer>(
            std::move(desc),
            std::move(outData)));
    }

    auto prefixCount = checked_product(
        pool.desc.templateDesc.shape,
        0,
        static_cast<int>(pool.desc.growDim));
    if (!prefixCount)
        return make_error(prefixCount.error());
    auto trailingCount = checked_product(
        pool.desc.templateDesc.shape,
        static_cast<int>(pool.desc.growDim) + 1,
        pool.desc.templateDesc.shape.rank());
    if (!trailingCount)
        return make_error(trailingCount.error());

    auto elementSize = core::dtype_size(desc.dtype);
    auto trailingBytes = static_cast<size_t>(*trailingCount) * elementSize;
    for (int64_t prefix = 0; prefix < *prefixCount; prefix++) {
        for (int64_t grow = 0; grow < tensor.growLength; grow++) {
            auto pageOrdinal = grow / pool.desc.pageSize;
            auto growInPage = grow % pool.desc.pageSize;
            if (static_cast<size_t>(pageOrdinal) >= tensor.pageIndices.size())
                return make_error("cpu device paged tensor missing backing page");
            auto pageIndex = tensor.pageIndices[static_cast<size_t>(pageOrdinal)];
            auto page = pool.pages.page(pageIndex);
            if (!page)
                return make_error(page.error());

            auto srcElement =
                (prefix * pool.desc.pageSize + growInPage) * *trailingCount;
            auto dstElement =
                (prefix * tensor.growLength + grow) * *trailingCount;
            std::memcpy(
                outData.data() + static_cast<size_t>(dstElement) * elementSize,
                page->data() + static_cast<size_t>(srcElement) * elementSize,
                trailingBytes);
        }
    }

    return TensorBufferPtr(std::make_shared<CpuTensorBuffer>(
        std::move(desc),
        std::move(outData)));
}

Result<TensorBufferPtr> CpuDevice::read(DeviceBufferId src) {
    auto it = buffers_.find(src);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    auto view = defaultView(it->second.desc);
    if (!view)
        return make_error(view.error());
    return read(DeviceTensorView{src, view.take()});
}

Result<TensorBufferPtr> CpuDevice::readPaged(DevicePagedTensorId src) {
    auto meta = pagedMeta(src);
    if (!meta)
        return make_error(meta.error());
    return read(DevicePagedTensorView{src, meta.take()});
}

Result<void> CpuDevice::run(
        DeviceCompiledGraphId graphId,
        ir::kernel_ir::OpId opId,
        std::span<const DeviceTensorView> inputs,
        std::span<const DeviceTensorView> outputs) {
    std::vector<DeviceRunValue> inputValues;
    inputValues.reserve(inputs.size());
    for (auto input : inputs)
        inputValues.push_back(input);
    std::vector<DeviceRunValue> outputValues;
    outputValues.reserve(outputs.size());
    for (auto output : outputs)
        outputValues.push_back(output);
    return run(graphId, opId, inputValues, outputValues);
}

Result<void> CpuDevice::run(
        DeviceCompiledGraphId graphId,
        ir::kernel_ir::OpId opId,
        std::span<const DeviceRunValue> inputs,
        std::span<const DeviceRunValue> outputs) {
    auto graphIt = graphs_.find(graphId);
    if (graphIt == graphs_.end())
        return make_error("cpu device compiled graph not found");
    auto kernelIt = graphIt->second.kernels.find(opId);
    if (kernelIt == graphIt->second.kernels.end())
        return make_error("cpu device kernel op not found");
    const auto& kernel = kernelIt->second;

    if (inputs.size() != kernel.inputCount)
        return make_error("cpu device input arity mismatch");
    if (outputs.size() != kernel.outputCount)
        return make_error("cpu device output arity mismatch");

    std::vector<TensorBufferPtr> pagedInputBuffers;
    pagedInputBuffers.reserve(inputs.size());

    struct TempInputBuffers {
        CpuDevice& device;
        std::vector<DeviceBufferId> ids;

        ~TempInputBuffers() {
            for (auto id : ids)
                device.buffers_.erase(id);
        }
    } tempInputBuffers{*this};

    std::vector<DeviceTensorView> inputViews;
    inputViews.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto* view = std::get_if<DeviceTensorView>(&input);
        if (view) {
            inputViews.push_back(*view);
            continue;
        }

        auto* paged = std::get_if<DevicePagedTensorView>(&input);
        if (!paged)
            return make_error("cpu device kernel input must be a tensor view");
        auto dense = read(*paged);
        if (!dense)
            return make_error(dense.error());
        auto access = (*dense)->access();
        if (!access)
            return make_error(access.error());
        auto viewDesc = defaultView((*access).desc());
        if (!viewDesc)
            return make_error(viewDesc.error());

        auto bufferId = nextBufferId_++;
        CpuDeviceBuffer buffer;
        buffer.desc = (*access).desc();
        buffer.borrowed.emplace(access.take());
        buffers_[bufferId] = std::move(buffer);
        tempInputBuffers.ids.push_back(bufferId);

        pagedInputBuffers.push_back(dense.take());
        inputViews.push_back(DeviceTensorView{bufferId, viewDesc.take()});
    }

    std::vector<DeviceTensorView> outputViews;
    outputViews.reserve(outputs.size());
    for (const auto& output : outputs) {
        auto* view = std::get_if<DeviceTensorView>(&output);
        if (!view)
            return make_error("cpu device kernel output must be a dense tensor view");
        outputViews.push_back(*view);
    }

    auto inputRef = [&](size_t index) -> Result<core::TensorRef> {
        auto it = buffers_.find(inputViews[index].buffer);
        if (it == buffers_.end())
            return make_error("cpu device input buffer not found");
        auto data = it->second.borrowed ? it->second.borrowed->data() : std::span<const uint8_t>(it->second.data);
        return core::make_tensor_ref(
            inputViews[index].view.desc,
            data,
            inputViews[index].view.strides,
            inputViews[index].view.storageOffset);
    };

    auto outputRef = [&](size_t index) -> Result<core::MutableTensorRef> {
        auto it = buffers_.find(outputViews[index].buffer);
        if (it == buffers_.end())
            return make_error("cpu device output buffer not found");
        if (it->second.borrowed)
            return make_error("cpu device output buffer is not writable");
        return core::make_mutable_tensor_ref(
            outputViews[index].view.desc,
            it->second.data,
            outputViews[index].view.strides,
            outputViews[index].view.storageOffset);
    };

    switch (kernel.kind) {
        case ir::kernel_ir::OpKind::Input:
        case ir::kernel_ir::OpKind::TensorTupleCreate:
        case ir::kernel_ir::OpKind::PagedAppend:
            return make_error("cpu device cannot run input boundary op");
        case ir::kernel_ir::OpKind::DeviceTransfer:
            return make_error("cpu device cannot run device transfer boundary op");
        case ir::kernel_ir::OpKind::ElementwiseKernel: {
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            switch (kernel.scalarOp) {
                case ir::kernel_ir::ScalarOp::Constant: {
                    auto numel = out->desc.shape.numel();
                    if (numel < 0)
                        return make_error("constant output must have static shape");
                    for (int64_t i = 0; i < numel; i++)
                        out->store_float(static_cast<size_t>(i), static_cast<float>(kernel.constant));
                    return {};
                }
                case ir::kernel_ir::ScalarOp::ReLU: {
                    auto x = inputRef(0);
                    if (!x) return make_error(x.error());
                    return core::relu(*x, *out);
                }
                case ir::kernel_ir::ScalarOp::Sqrt: {
                    auto x = inputRef(0);
                    if (!x) return make_error(x.error());
                    return core::sqrt(*x, *out);
                }
                case ir::kernel_ir::ScalarOp::Tanh: {
                    auto x = inputRef(0);
                    if (!x) return make_error(x.error());
                    return core::tanh(*x, *out);
                }
                case ir::kernel_ir::ScalarOp::Add: {
                    auto lhs = inputRef(0);
                    if (!lhs) return make_error(lhs.error());
                    auto rhs = inputRef(1);
                    if (!rhs) return make_error(rhs.error());
                    return core::add(*lhs, *rhs, *out);
                }
                case ir::kernel_ir::ScalarOp::Mul: {
                    auto lhs = inputRef(0);
                    if (!lhs) return make_error(lhs.error());
                    auto rhs = inputRef(1);
                    if (!rhs) return make_error(rhs.error());
                    return core::mul(*lhs, *rhs, *out);
                }
                case ir::kernel_ir::ScalarOp::Div: {
                    auto lhs = inputRef(0);
                    if (!lhs) return make_error(lhs.error());
                    auto rhs = inputRef(1);
                    if (!rhs) return make_error(rhs.error());
                    return core::div(*lhs, *rhs, *out);
                }
                default:
                    return make_error("cpu device unsupported elementwise scalar op");
            }
        }
        case ir::kernel_ir::OpKind::LinearKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto bias = inputRef(2);
            if (!bias) return make_error(bias.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::linear(*x, *weight, *bias, *out);
        }
        case ir::kernel_ir::OpKind::MatMulKernel: {
            auto lhs = inputRef(0);
            if (!lhs) return make_error(lhs.error());
            auto rhs = inputRef(1);
            if (!rhs) return make_error(rhs.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::matmul(*lhs, *rhs, kernel.transposeLhs, kernel.transposeRhs, *out);
        }
        case ir::kernel_ir::OpKind::LayoutTransform: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            switch (kernel.layoutTransform) {
                case ir::kernel_ir::LayoutTransformKind::Reshape:
                    return core::reshape(*x, *out);
                case ir::kernel_ir::LayoutTransformKind::Transpose:
                    return core::transpose(*x, *out);
                case ir::kernel_ir::LayoutTransformKind::Permute:
                    return core::permute(*x, kernel.dims, *out);
                case ir::kernel_ir::LayoutTransformKind::Slice:
                    return make_error("slice layout aliases must not launch on CPU");
                case ir::kernel_ir::LayoutTransformKind::Contiguous:
                    return core::reshape(*x, *out);
            }
            return make_error("cpu device unsupported layout transform");
        }
        case ir::kernel_ir::OpKind::SlidingQueryKeyScoreKernel: {
            auto q = inputRef(0);
            if (!q) return make_error(q.error());
            auto k = inputRef(1);
            if (!k) return make_error(k.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            if (inputs.size() == 3) {
                auto positionIds = inputRef(2);
                if (!positionIds) return make_error(positionIds.error());
                return core::sliding_query_key_score(
                    *q,
                    *k,
                    *positionIds,
                    kernel.window,
                    static_cast<float>(kernel.scale),
                    *out);
            }
            return core::sliding_query_key_score(
                *q,
                *k,
                kernel.window,
                static_cast<float>(kernel.scale),
                *out);
        }
        case ir::kernel_ir::OpKind::AttentionKernel: {
            auto q = inputRef(0);
            if (!q) return make_error(q.error());
            auto k = inputRef(1);
            if (!k) return make_error(k.error());
            auto v = inputRef(2);
            if (!v) return make_error(v.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            if (inputs.size() == 4) {
                auto positionOffsets = inputRef(3);
                if (!positionOffsets) return make_error(positionOffsets.error());
                return core::attention(
                    *q,
                    *k,
                    *v,
                    *positionOffsets,
                    kernel.window,
                    static_cast<float>(kernel.scale),
                    *out);
            }
            return core::attention(
                *q,
                *k,
                *v,
                kernel.window,
                static_cast<float>(kernel.scale),
                *out);
        }
        case ir::kernel_ir::OpKind::SoftmaxKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::softmax(*x, kernel.axis, *out);
        }
        case ir::kernel_ir::OpKind::TopKKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto values = outputRef(0);
            if (!values) return make_error(values.error());
            auto indices = outputRef(1);
            if (!indices) return make_error(indices.error());
            return core::topk(*x, kernel.k, kernel.axis, *values, *indices);
        }
        case ir::kernel_ir::OpKind::ReductionKernel: {
            if (kernel.reduce != ir::kernel_ir::ReduceOp::Sum)
                return make_error("cpu device only supports sum reduction");
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::sum(*x, kernel.axis, kernel.keepDims, *out);
        }
        case ir::kernel_ir::OpKind::GatherKernel: {
            auto ids = inputRef(0);
            if (!ids) return make_error(ids.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::embedding(*ids, *weight, *out);
        }
        case ir::kernel_ir::OpKind::RoPEKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            if (inputs.size() == 2) {
                auto positionIds = inputRef(1);
                if (!positionIds) return make_error(positionIds.error());
                return core::rope(
                    *x,
                    *positionIds,
                    static_cast<float>(kernel.theta),
                    kernel.rotaryDim,
                    kernel.splitHalf,
                    *out);
            }
            return core::rope(
                *x,
                static_cast<float>(kernel.theta),
                kernel.rotaryDim,
                kernel.splitHalf,
                *out);
        }
        case ir::kernel_ir::OpKind::NormKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            if (kernel.norm == ir::kernel_ir::NormKind::RMSNorm) {
                if (inputs.size() == 1)
                    return core::rms_norm(*x, static_cast<float>(kernel.epsilon), *out);
                if (inputs.size() != 2)
                    return make_error("rms_norm expects 1 or 2 inputs");
                auto weight = inputRef(1);
                if (!weight) return make_error(weight.error());
                return core::rms_norm(*x, *weight, static_cast<float>(kernel.epsilon), *out);
            }
            if (inputs.size() != 3)
                return make_error("layer_norm expects 3 inputs");
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto bias = inputRef(2);
            if (!bias) return make_error(bias.error());
            return core::layer_norm(*x, *weight, *bias, static_cast<float>(kernel.epsilon), *out);
        }
        case ir::kernel_ir::OpKind::MoeGatherKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto topkIds = inputRef(1);
            if (!topkIds) return make_error(topkIds.error());
            auto topkWeights = inputRef(2);
            if (!topkWeights) return make_error(topkWeights.error());
            auto packedX = outputRef(0);
            if (!packedX) return make_error(packedX.error());
            auto packedWeights = outputRef(1);
            if (!packedWeights) return make_error(packedWeights.error());
            auto tokenIds = outputRef(2);
            if (!tokenIds) return make_error(tokenIds.error());
            auto expertOffsets = outputRef(3);
            if (!expertOffsets) return make_error(expertOffsets.error());
            return core::moe_gather(
                *x,
                *topkIds,
                *topkWeights,
                kernel.numExperts,
                kernel.topK,
                *packedX,
                *packedWeights,
                *tokenIds,
                *expertOffsets);
        }
        case ir::kernel_ir::OpKind::MoeMatMulKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto expertOffsets = inputRef(1);
            if (!expertOffsets) return make_error(expertOffsets.error());
            auto weight = inputRef(2);
            if (!weight) return make_error(weight.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::moe_matmul(*x, *expertOffsets, *weight, kernel.transposeRhs, *out);
        }
        case ir::kernel_ir::OpKind::MoeScatterSumKernel: {
            auto packedOut = inputRef(0);
            if (!packedOut) return make_error(packedOut.error());
            auto packedWeights = inputRef(1);
            if (!packedWeights) return make_error(packedWeights.error());
            auto tokenIds = inputRef(2);
            if (!tokenIds) return make_error(tokenIds.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::moe_scatter_sum(*packedOut, *packedWeights, *tokenIds, *out);
        }
    }

    return make_error("cpu device cannot run unknown op kind");
}

} // namespace sandy::device
