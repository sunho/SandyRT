#include "CudaDevice.h"

#include "CudaRuntime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace sandy::device {

namespace {

constexpr uintptr_t kPagedVectorLoadAlignment = 16;

} // namespace

namespace {

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

struct PagedAppendLayout {
    int64_t chunkGrowLength = 0;
    int64_t newGrowLength = 0;
    int64_t requiredPages = 0;
    int64_t prefixCount = 0;
    int64_t trailingCount = 0;
    size_t elementSize = 0;
    size_t expectedBytes = 0;
    size_t trailingBytes = 0;
};

Result<PagedAppendLayout> plan_paged_append(
        const DevicePagedPoolDesc& pool,
        const core::TensorDesc& chunk,
        int64_t oldGrowLength) {
    auto valid = validate_paged_logical_shape(pool, chunk.shape);
    if (!valid)
        return make_error(valid.error());
    if (chunk.dtype != pool.templateDesc.dtype)
        return make_error("cuda device paged append dtype mismatch");

    auto chunkGrowLength = chunk.shape.dim(static_cast<int>(pool.growDim));
    if (chunkGrowLength <= 0)
        return make_error("cuda device paged append grow dimension must be > 0");
    auto chunkNumel = chunk.shape.numel();
    if (chunkNumel < 0)
        return make_error("cuda device paged append chunk shape must be concrete");
    if (oldGrowLength > std::numeric_limits<int64_t>::max() - chunkGrowLength)
        return make_error("cuda device paged append grow length overflow");

    auto prefixCount = checked_product(
        pool.templateDesc.shape,
        0,
        static_cast<int>(pool.growDim));
    if (!prefixCount)
        return make_error(prefixCount.error());
    auto trailingCount = checked_product(
        pool.templateDesc.shape,
        static_cast<int>(pool.growDim) + 1,
        pool.templateDesc.shape.rank());
    if (!trailingCount)
        return make_error(trailingCount.error());

    PagedAppendLayout layout;
    layout.chunkGrowLength = chunkGrowLength;
    layout.newGrowLength = oldGrowLength + chunkGrowLength;
    layout.requiredPages = ceil_div(layout.newGrowLength, pool.pageSize);
    layout.prefixCount = *prefixCount;
    layout.trailingCount = *trailingCount;
    layout.elementSize = core::dtype_size(chunk.dtype);
    layout.expectedBytes = static_cast<size_t>(chunkNumel) * layout.elementSize;
    layout.trailingBytes = static_cast<size_t>(*trailingCount) * layout.elementSize;
    return layout;
}

} // namespace

CudaPagedTensorPool::CudaPagedTensorPool(int cudaDevice)
    : cudaDevice_(cudaDevice), initialized_(true) {}

CudaPagedTensorPool::~CudaPagedTensorPool() {
    release();
}

CudaPagedTensorPool::CudaPagedTensorPool(CudaPagedTensorPool&& other) noexcept
    : cudaDevice_(other.cudaDevice_),
      initialized_(other.initialized_),
      desc_(std::move(other.desc_)),
      pageElementCount_(other.pageElementCount_),
      pageBytes_(other.pageBytes_),
      pages_(std::move(other.pages_)),
      freePages_(std::move(other.freePages_)) {
    other.initialized_ = false;
    other.pages_.clear();
}

CudaPagedTensorPool& CudaPagedTensorPool::operator=(CudaPagedTensorPool&& other) noexcept {
    if (this == &other)
        return *this;
    release();
    cudaDevice_ = other.cudaDevice_;
    initialized_ = other.initialized_;
    desc_ = std::move(other.desc_);
    pageElementCount_ = other.pageElementCount_;
    pageBytes_ = other.pageBytes_;
    pages_ = std::move(other.pages_);
    freePages_ = std::move(other.freePages_);
    other.initialized_ = false;
    other.pages_.clear();
    return *this;
}

Result<CudaPagedTensorPool> CudaPagedTensorPool::create(
        int cudaDevice,
        DevicePagedPoolDesc desc,
        cudaStream_t stream) {
    CudaPagedTensorPool pool(cudaDevice);
    auto initialized = pool.initialize(std::move(desc), stream);
    if (!initialized)
        return make_error(initialized.error());
    return std::move(pool);
}

Result<void> CudaPagedTensorPool::initialize(
        DevicePagedPoolDesc desc,
        cudaStream_t stream) {
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

    auto set = cuda_check(cudaSetDevice(cudaDevice_), "cudaSetDevice");
    if (!set)
        return make_error(set.error());

    desc_ = std::move(desc);
    pageElementCount_ = pageElementCount;
    pageBytes_ = static_cast<size_t>(pageElementCount) * elementSize;

    for (int64_t i = 0; i < desc_.initialPages; i++) {
        auto page = allocate_page(stream);
        if (!page)
            return make_error(page.error());
        auto freed = free_page(*page);
        if (!freed)
            return make_error(freed.error());
    }
    return {};
}

Result<uint32_t> CudaPagedTensorPool::allocate_page(cudaStream_t stream) {
    auto set = cuda_check(cudaSetDevice(cudaDevice_), "cudaSetDevice");
    if (!set)
        return make_error(set.error());

    if (!freePages_.empty()) {
        auto page = freePages_.back();
        freePages_.pop_back();
        return page;
    }

    if (desc_.maxPages >= 0 && static_cast<int64_t>(pages_.size()) >= desc_.maxPages)
        return make_error("cuda device paged pool capacity exceeded");
    if (pages_.size() > std::numeric_limits<uint32_t>::max())
        return make_error("cuda device paged pool page index overflow");

    void* data = nullptr;
    auto allocated = cuda_malloc_stream_ordered(
        &data,
        pageBytes_,
        stream,
        "cudaMallocAsync paged pool page");
    if (!allocated)
        return make_error(allocated.error());
    if ((reinterpret_cast<uintptr_t>(data) & (kPagedVectorLoadAlignment - 1)) != 0) {
        (void)cuda_free_stream_ordered(
            data,
            stream,
            "cudaFreeAsync misaligned paged pool page");
        return make_error("CUDA paged pool page is not vector-load aligned");
    }
    pages_.push_back(Page{data});
    return static_cast<uint32_t>(pages_.size() - 1);
}

Result<void> CudaPagedTensorPool::free_page(uint32_t page) {
    if (page >= pages_.size())
        return make_error("cuda device paged pool page index out of range");
    if (std::find(freePages_.begin(), freePages_.end(), page) != freePages_.end())
        return make_error("cuda device paged pool page already freed");
    freePages_.push_back(page);
    return {};
}

Result<void*> CudaPagedTensorPool::page_data(uint32_t page) const {
    if (page >= pages_.size())
        return make_error("cuda device paged pool page index out of range");
    return pages_[page].data;
}

void CudaPagedTensorPool::release() {
    if (!initialized_)
        return;
    cudaSetDevice(cudaDevice_);
    for (auto& page : pages_) {
        if (page.data)
            cudaFree(page.data);
        page.data = nullptr;
    }
    pages_.clear();
    freePages_.clear();
    initialized_ = false;
}

CudaPagedTensor::CudaPagedTensor(
        int cudaDevice,
        DevicePagedPoolId pool,
        core::Shape logicalShape,
        int64_t growLength)
    : cudaDevice_(cudaDevice),
      pool_(pool),
      logicalShape_(std::move(logicalShape)),
      growLength_(growLength) {}

CudaPagedTensor::~CudaPagedTensor() {
    release_table();
}

CudaPagedTensor::CudaPagedTensor(CudaPagedTensor&& other) noexcept
    : cudaDevice_(other.cudaDevice_),
      pool_(other.pool_),
      logicalShape_(std::move(other.logicalShape_)),
      growLength_(other.growLength_),
      pageIndices_(std::move(other.pageIndices_)),
      pageTable_(other.pageTable_),
      pageTableCapacity_(other.pageTableCapacity_),
      hostPageTable_(other.hostPageTable_),
      hostPageTableCapacity_(other.hostPageTableCapacity_),
      syncedPageCount_(other.syncedPageCount_),
      retiredHostPageTables_(std::move(other.retiredHostPageTables_)) {
    other.pageTable_ = nullptr;
    other.pageTableCapacity_ = 0;
    other.hostPageTable_ = nullptr;
    other.hostPageTableCapacity_ = 0;
    other.syncedPageCount_ = 0;
}

CudaPagedTensor& CudaPagedTensor::operator=(CudaPagedTensor&& other) noexcept {
    if (this == &other)
        return *this;
    release_table();
    cudaDevice_ = other.cudaDevice_;
    pool_ = other.pool_;
    logicalShape_ = std::move(other.logicalShape_);
    growLength_ = other.growLength_;
    pageIndices_ = std::move(other.pageIndices_);
    pageTable_ = other.pageTable_;
    pageTableCapacity_ = other.pageTableCapacity_;
    hostPageTable_ = other.hostPageTable_;
    hostPageTableCapacity_ = other.hostPageTableCapacity_;
    syncedPageCount_ = other.syncedPageCount_;
    retiredHostPageTables_ = std::move(other.retiredHostPageTables_);
    other.pageTable_ = nullptr;
    other.pageTableCapacity_ = 0;
    other.hostPageTable_ = nullptr;
    other.hostPageTableCapacity_ = 0;
    other.syncedPageCount_ = 0;
    return *this;
}

Result<void> CudaPagedTensor::sync_page_table(
        const CudaPagedTensorPool& pool,
        cudaStream_t stream) {
    auto set = cuda_check(cudaSetDevice(cudaDevice_), "cudaSetDevice");
    if (!set)
        return make_error(set.error());

    auto pageCount = static_cast<int64_t>(pageIndices_.size());
    if (pageCount <= syncedPageCount_)
        return {};

    if (pageCount > hostPageTableCapacity_) {
        auto newCapacity = std::max<int64_t>(
            pageCount,
            std::max<int64_t>(1, hostPageTableCapacity_ * 2));
        void** hostTable = nullptr;
        auto allocated = cuda_check(
            cudaHostAlloc(
                reinterpret_cast<void**>(&hostTable),
                static_cast<size_t>(newCapacity) * sizeof(void*),
                cudaHostAllocDefault),
            "cudaHostAlloc paged tensor host table");
        if (!allocated)
            return make_error(allocated.error());
        if (hostPageTable_ && syncedPageCount_ > 0) {
            std::copy_n(hostPageTable_, syncedPageCount_, hostTable);
            retiredHostPageTables_.push_back(hostPageTable_);
        } else if (hostPageTable_) {
            auto freed = cuda_check(
                cudaFreeHost(hostPageTable_),
                "cudaFreeHost paged tensor host table");
            if (!freed) {
                cudaFreeHost(hostTable);
                return make_error(freed.error());
            }
        }
        hostPageTable_ = hostTable;
        hostPageTableCapacity_ = newCapacity;
    }

    for (int64_t i = syncedPageCount_; i < pageCount; i++) {
        auto data = pool.page_data(pageIndices_[static_cast<size_t>(i)]);
        if (!data)
            return make_error(data.error());
        hostPageTable_[i] = *data;
    }

    if (pageCount > pageTableCapacity_) {
        void** table = nullptr;
        auto newCapacity = std::max<int64_t>(
            pageCount,
            std::max<int64_t>(1, pageTableCapacity_ * 2));
        auto bytes = static_cast<size_t>(newCapacity) * sizeof(void*);
        if (bytes > 0) {
            auto allocated = cuda_malloc_stream_ordered(
                &table,
                bytes,
                stream,
                "cudaMallocAsync paged tensor table");
            if (!allocated)
                return make_error(allocated.error());
        }
        auto copied = cuda_check(
            cudaMemcpyAsync(
                table,
                hostPageTable_,
                static_cast<size_t>(pageCount) * sizeof(void*),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync paged tensor table");
        if (!copied) {
            (void)cuda_free_stream_ordered(
                table,
                stream,
                "cudaFreeAsync paged tensor table");
            return make_error(copied.error());
        }
        if (pageTable_) {
            auto freed = cuda_free_stream_ordered(
                pageTable_,
                stream,
                "cudaFreeAsync paged tensor table");
            if (!freed)
                return make_error(freed.error());
        }
        pageTable_ = table;
        pageTableCapacity_ = newCapacity;
    } else {
        auto copied = cuda_check(
            cudaMemcpyAsync(
                pageTable_ + syncedPageCount_,
                hostPageTable_ + syncedPageCount_,
                static_cast<size_t>(pageCount - syncedPageCount_) * sizeof(void*),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync paged tensor table entries");
        if (!copied)
            return make_error(copied.error());
    }

    syncedPageCount_ = pageCount;
    return {};
}

void CudaPagedTensor::release_table() {
    cudaSetDevice(cudaDevice_);
    if (pageTable_)
        cudaFree(pageTable_);
    pageTable_ = nullptr;
    pageTableCapacity_ = 0;
    if (hostPageTable_)
        cudaFreeHost(hostPageTable_);
    hostPageTable_ = nullptr;
    hostPageTableCapacity_ = 0;
    for (auto* table : retiredHostPageTables_)
        cudaFreeHost(table);
    retiredHostPageTables_.clear();
    syncedPageCount_ = 0;
}

Result<DevicePagedPoolId> CudaDevice::createPagedPoolImpl(DevicePagedPoolDesc desc) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());

    auto pool = CudaPagedTensorPool::create(cudaDevice_, std::move(desc), stream_);
    if (!pool)
        return make_error(pool.error());

    auto id = nextPagedPoolId_++;
    pagedPools_[id] = pool.take();
    return id;
}

Result<void> CudaDevice::destroyPagedPool(DevicePagedPoolId poolId) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());

    auto poolIt = pagedPools_.find(poolId);
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");
    for (const auto& item : pagedTensors_) {
        if (item.second.pool() == poolId)
            return make_error("cuda device paged pool still has live tensors");
    }
    auto synced = cuda_check(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize destroy paged pool");
    if (!synced)
        return make_error(synced.error());
    pagedPools_.erase(poolIt);
    return {};
}

Result<DevicePagedTensorId> CudaDevice::allocPaged(
        DevicePagedPoolId poolId,
        core::Shape logicalShape) {
    auto poolIt = pagedPools_.find(poolId);
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    auto valid = validate_paged_logical_shape(poolIt->second.desc(), logicalShape);
    if (!valid)
        return make_error(valid.error());

    auto growLength = logicalShape.dim(static_cast<int>(poolIt->second.desc().growDim));
    auto id = nextPagedTensorId_++;
    pagedTensors_[id] = CudaPagedTensor(
        cudaDevice_,
        poolId,
        std::move(logicalShape),
        growLength);

    auto requiredPages = ceil_div(growLength, poolIt->second.desc().pageSize);
    auto reserved = reservePaged(id, requiredPages);
    if (!reserved) {
        pagedTensors_.erase(id);
        return make_error(reserved.error());
    }
    return id;
}

Result<void> CudaDevice::deallocPaged(DevicePagedTensorId tensorId) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());

    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    auto synced = cuda_check(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize dealloc paged tensor");
    if (!synced)
        return make_error(synced.error());

    for (auto page : tensorIt->second.page_indices()) {
        auto freed = poolIt->second.free_page(page);
        if (!freed)
            return make_error(freed.error());
    }
    pagedTensors_.erase(tensorIt);
    return {};
}

Result<void> CudaDevice::reservePaged(DevicePagedTensorId tensorId, int64_t pageCount) {
    if (pageCount < 0)
        return make_error("cuda device paged tensor reserve page count must be >= 0");
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    auto& pages = tensorIt->second.page_indices();
    while (static_cast<int64_t>(pages.size()) < pageCount) {
        auto page = poolIt->second.allocate_page(stream_);
        if (!page)
            return make_error(page.error());
        pages.push_back(*page);
    }
    return tensorIt->second.sync_page_table(poolIt->second, stream_);
}

Result<void> CudaDevice::appendPaged(
        DevicePagedTensorId tensorId,
        core::TensorBuffer& denseChunk) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    auto access = denseChunk.access();
    if (!access)
        return make_error(access.error());
    const auto& chunkDesc = (*access).desc();
    const auto& pool = poolIt->second;
    auto& tensor = tensorIt->second;
    auto oldGrowLength = tensor.grow_length();
    auto layout = plan_paged_append(pool.desc(), chunkDesc, oldGrowLength);
    if (!layout)
        return make_error(layout.error());
    if ((*access).data().size() != layout->expectedBytes)
        return make_error("cuda device paged append chunk byte size mismatch");

    auto reserved = reservePaged(tensorId, layout->requiredPages);
    if (!reserved)
        return make_error(reserved.error());
    auto source = (*access).data();

    for (int64_t prefix = 0; prefix < layout->prefixCount; prefix++) {
        for (int64_t chunkGrow = 0; chunkGrow < layout->chunkGrowLength; chunkGrow++) {
            auto dstGrow = oldGrowLength + chunkGrow;
            auto dstPageOrdinal = dstGrow / pool.desc().pageSize;
            auto dstGrowInPage = dstGrow % pool.desc().pageSize;
            auto pageIndex = tensor.page_indices()[static_cast<size_t>(dstPageOrdinal)];
            auto pageData = pool.page_data(pageIndex);
            if (!pageData)
                return make_error(pageData.error());

            auto sourceElement =
                (prefix * layout->chunkGrowLength + chunkGrow) * layout->trailingCount;
            auto dstElement =
                (prefix * pool.desc().pageSize + dstGrowInPage) * layout->trailingCount;
            auto copied = cuda_check(
                cudaMemcpyAsync(
                    static_cast<uint8_t*>(*pageData) +
                        static_cast<size_t>(dstElement) * layout->elementSize,
                    source.data() + static_cast<size_t>(sourceElement) * layout->elementSize,
                    layout->trailingBytes,
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync paged append host to device");
            if (!copied)
                return make_error(copied.error());
        }
    }

    auto synced = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    if (!synced)
        return make_error(synced.error());

    tensor.set_grow_length(layout->newGrowLength);
    tensor.set_logical_shape(shape_with_grow_length(
        pool.desc().templateDesc.shape,
        pool.desc().growDim,
        layout->newGrowLength));
    return {};
}

Result<void> CudaDevice::appendPaged(
        DevicePagedTensorId tensorId,
        DeviceTensorView denseChunk) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    auto sourceBuffer = buffer_view(denseChunk.buffer, false);
    if (!sourceBuffer)
        return make_error(sourceBuffer.error());
    auto contiguous = isDefaultView(denseChunk.view);
    if (!contiguous)
        return make_error(contiguous.error());
    if (!*contiguous)
        return make_error("cuda device paged append requires a contiguous source view");
    if (denseChunk.view.storageOffset < 0)
        return make_error("cuda device paged append source has a negative storage offset");

    const auto& pool = poolIt->second;
    auto& tensor = tensorIt->second;
    auto oldGrowLength = tensor.grow_length();
    auto layout = plan_paged_append(pool.desc(), denseChunk.view.desc, oldGrowLength);
    if (!layout)
        return make_error(layout.error());

    auto storageOffset = static_cast<size_t>(denseChunk.view.storageOffset);
    if (storageOffset > std::numeric_limits<size_t>::max() / layout->elementSize)
        return make_error("cuda device paged append source offset overflow");
    auto sourceByteOffset = storageOffset * layout->elementSize;
    if (sourceByteOffset > sourceBuffer->bytes ||
        layout->expectedBytes > sourceBuffer->bytes - sourceByteOffset) {
        return make_error("cuda device paged append source view exceeds its buffer");
    }

    auto reserved = reservePaged(tensorId, layout->requiredPages);
    if (!reserved)
        return make_error(reserved.error());

    auto* source = static_cast<const uint8_t*>(sourceBuffer->data) + sourceByteOffset;
    for (int64_t prefix = 0; prefix < layout->prefixCount; prefix++) {
        int64_t chunkGrow = 0;
        while (chunkGrow < layout->chunkGrowLength) {
            auto dstGrow = oldGrowLength + chunkGrow;
            auto dstPageOrdinal = dstGrow / pool.desc().pageSize;
            auto dstGrowInPage = dstGrow % pool.desc().pageSize;
            auto copyGrowLength = std::min(
                layout->chunkGrowLength - chunkGrow,
                pool.desc().pageSize - dstGrowInPage);
            auto pageIndex = tensor.page_indices()[static_cast<size_t>(dstPageOrdinal)];
            auto pageData = pool.page_data(pageIndex);
            if (!pageData)
                return make_error(pageData.error());

            auto sourceElement =
                (prefix * layout->chunkGrowLength + chunkGrow) * layout->trailingCount;
            auto dstElement =
                (prefix * pool.desc().pageSize + dstGrowInPage) * layout->trailingCount;
            auto copyBytes = static_cast<size_t>(copyGrowLength) * layout->trailingBytes;
            auto copied = cuda_check(
                cudaMemcpyAsync(
                    static_cast<uint8_t*>(*pageData) +
                        static_cast<size_t>(dstElement) * layout->elementSize,
                    source + static_cast<size_t>(sourceElement) * layout->elementSize,
                    copyBytes,
                    cudaMemcpyDeviceToDevice,
                    stream_),
                "cudaMemcpyAsync paged append device to device");
            if (!copied)
                return make_error(copied.error());
            chunkGrow += copyGrowLength;
        }
    }

    tensor.set_grow_length(layout->newGrowLength);
    tensor.set_logical_shape(shape_with_grow_length(
        pool.desc().templateDesc.shape,
        pool.desc().growDim,
        layout->newGrowLength));
    return {};
}

Result<DevicePagedTensorMeta> CudaDevice::pagedMeta(DevicePagedTensorId tensorId) const {
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    DevicePagedTensorMeta meta;
    meta.pool = tensorIt->second.pool();
    meta.logicalDesc = core::TensorDesc(
        tensorIt->second.logical_shape(),
        poolIt->second.desc().templateDesc.dtype);
    meta.growDim = poolIt->second.desc().growDim;
    meta.pageSize = poolIt->second.desc().pageSize;
    meta.growLength = tensorIt->second.grow_length();
    meta.pageCount = static_cast<int64_t>(tensorIt->second.page_indices().size());
    meta.pageElementCount = poolIt->second.page_element_count();
    return meta;
}

Result<void> CudaDevice::sync_paged_tensor_table(DevicePagedTensorId tensorId) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");
    return tensorIt->second.sync_page_table(poolIt->second, stream_);
}

Result<CudaDevicePagedTensorView> CudaDevice::paged_tensor_view(
        DevicePagedTensorId tensorId) const {
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

    auto meta = pagedMeta(tensorId);
    if (!meta)
        return make_error(meta.error());
    return CudaDevicePagedTensorView{
        tensorIt->second.page_table(),
        meta.take(),
        poolIt->second.page_bytes(),
    };
}

} // namespace sandy::device
