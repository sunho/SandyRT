#include "CudaDevice.h"

#include "CudaRuntime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace sandy::device {

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
        DevicePagedPoolDesc desc) {
    CudaPagedTensorPool pool(cudaDevice);
    auto initialized = pool.initialize(std::move(desc));
    if (!initialized)
        return make_error(initialized.error());
    return std::move(pool);
}

Result<void> CudaPagedTensorPool::initialize(DevicePagedPoolDesc desc) {
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
        auto page = allocate_page();
        if (!page)
            return make_error(page.error());
        auto freed = free_page(*page);
        if (!freed)
            return make_error(freed.error());
    }
    return {};
}

Result<uint32_t> CudaPagedTensorPool::allocate_page() {
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
    auto allocated = cuda_check(cudaMalloc(&data, pageBytes_), "cudaMalloc paged pool page");
    if (!allocated)
        return make_error(allocated.error());
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
      pageTableCapacity_(other.pageTableCapacity_) {
    other.pageTable_ = nullptr;
    other.pageTableCapacity_ = 0;
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
    other.pageTable_ = nullptr;
    other.pageTableCapacity_ = 0;
    return *this;
}

Result<void> CudaPagedTensor::sync_page_table(
        const CudaPagedTensorPool& pool,
        cudaStream_t stream) {
    auto set = cuda_check(cudaSetDevice(cudaDevice_), "cudaSetDevice");
    if (!set)
        return make_error(set.error());

    auto pageCount = static_cast<int64_t>(pageIndices_.size());
    if (pageCount > pageTableCapacity_) {
        void** table = nullptr;
        auto bytes = static_cast<size_t>(pageCount) * sizeof(void*);
        if (bytes > 0) {
            auto allocated = cuda_check(cudaMalloc(&table, bytes), "cudaMalloc paged tensor table");
            if (!allocated)
                return make_error(allocated.error());
        }
        if (pageTable_)
            cudaFree(pageTable_);
        pageTable_ = table;
        pageTableCapacity_ = pageCount;
    }
    if (pageCount == 0)
        return {};

    std::vector<void*> hostTable;
    hostTable.reserve(pageIndices_.size());
    for (auto page : pageIndices_) {
        auto data = pool.page_data(page);
        if (!data)
            return make_error(data.error());
        hostTable.push_back(*data);
    }

    auto copied = cuda_check(
        cudaMemcpyAsync(
            pageTable_,
            hostTable.data(),
            hostTable.size() * sizeof(void*),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync paged tensor table");
    if (!copied)
        return make_error(copied.error());
    return cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

void CudaPagedTensor::release_table() {
    if (!pageTable_)
        return;
    cudaSetDevice(cudaDevice_);
    cudaFree(pageTable_);
    pageTable_ = nullptr;
    pageTableCapacity_ = 0;
}

Result<DevicePagedPoolId> CudaDevice::createPagedPool(DevicePagedPoolDesc desc) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());

    auto pool = CudaPagedTensorPool::create(cudaDevice_, std::move(desc));
    if (!pool)
        return make_error(pool.error());

    auto id = nextPagedPoolId_++;
    pagedPools_[id] = pool.take();
    return id;
}

Result<void> CudaDevice::destroyPagedPool(DevicePagedPoolId poolId) {
    auto poolIt = pagedPools_.find(poolId);
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");
    for (const auto& item : pagedTensors_) {
        if (item.second.pool() == poolId)
            return make_error("cuda device paged pool still has live tensors");
    }
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
    auto tensorIt = pagedTensors_.find(tensorId);
    if (tensorIt == pagedTensors_.end())
        return make_error("cuda device paged tensor not found");
    auto poolIt = pagedPools_.find(tensorIt->second.pool());
    if (poolIt == pagedPools_.end())
        return make_error("cuda device paged pool not found");

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
        auto page = poolIt->second.allocate_page();
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
    auto valid = validate_paged_logical_shape(pool.desc(), chunkDesc.shape);
    if (!valid)
        return make_error(valid.error());
    if (chunkDesc.dtype != pool.desc().templateDesc.dtype)
        return make_error("cuda device paged append dtype mismatch");

    int growDim = static_cast<int>(pool.desc().growDim);
    auto chunkGrowLength = chunkDesc.shape.dim(growDim);
    if (chunkGrowLength <= 0)
        return make_error("cuda device paged append grow dimension must be > 0");

    auto chunkNumel = chunkDesc.shape.numel();
    if (chunkNumel < 0)
        return make_error("cuda device paged append chunk shape must be concrete");
    auto expectedBytes = static_cast<size_t>(chunkNumel) * core::dtype_size(chunkDesc.dtype);
    if ((*access).data().size() != expectedBytes)
        return make_error("cuda device paged append chunk byte size mismatch");

    auto& tensor = tensorIt->second;
    auto oldGrowLength = tensor.grow_length();
    if (oldGrowLength > std::numeric_limits<int64_t>::max() - chunkGrowLength)
        return make_error("cuda device paged append grow length overflow");
    auto newGrowLength = oldGrowLength + chunkGrowLength;
    auto requiredPages = ceil_div(newGrowLength, pool.desc().pageSize);
    auto reserved = reservePaged(tensorId, requiredPages);
    if (!reserved)
        return make_error(reserved.error());

    auto prefixCount = checked_product(
        pool.desc().templateDesc.shape,
        0,
        static_cast<int>(pool.desc().growDim));
    if (!prefixCount)
        return make_error(prefixCount.error());
    auto trailingCount = checked_product(
        pool.desc().templateDesc.shape,
        static_cast<int>(pool.desc().growDim) + 1,
        pool.desc().templateDesc.shape.rank());
    if (!trailingCount)
        return make_error(trailingCount.error());

    auto elementSize = core::dtype_size(chunkDesc.dtype);
    auto trailingBytes = static_cast<size_t>(*trailingCount) * elementSize;
    auto source = (*access).data();

    for (int64_t prefix = 0; prefix < *prefixCount; prefix++) {
        for (int64_t chunkGrow = 0; chunkGrow < chunkGrowLength; chunkGrow++) {
            auto dstGrow = oldGrowLength + chunkGrow;
            auto dstPageOrdinal = dstGrow / pool.desc().pageSize;
            auto dstGrowInPage = dstGrow % pool.desc().pageSize;
            auto pageIndex = tensor.page_indices()[static_cast<size_t>(dstPageOrdinal)];
            auto pageData = pool.page_data(pageIndex);
            if (!pageData)
                return make_error(pageData.error());

            auto sourceElement =
                (prefix * chunkGrowLength + chunkGrow) * *trailingCount;
            auto dstElement =
                (prefix * pool.desc().pageSize + dstGrowInPage) * *trailingCount;
            auto copied = cuda_check(
                cudaMemcpyAsync(
                    static_cast<uint8_t*>(*pageData) +
                        static_cast<size_t>(dstElement) * elementSize,
                    source.data() + static_cast<size_t>(sourceElement) * elementSize,
                    trailingBytes,
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

    tensor.set_grow_length(newGrowLength);
    tensor.set_logical_shape(shape_with_grow_length(
        pool.desc().templateDesc.shape,
        pool.desc().growDim,
        newGrowLength));
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
