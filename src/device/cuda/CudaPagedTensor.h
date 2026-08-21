#pragma once

#include "DeviceTypes.h"
#include "Result.h"
#include "Tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace sandy::device {

class CudaPagedTensorPool {
public:
    CudaPagedTensorPool() = default;
    ~CudaPagedTensorPool();

    CudaPagedTensorPool(const CudaPagedTensorPool&) = delete;
    CudaPagedTensorPool& operator=(const CudaPagedTensorPool&) = delete;
    CudaPagedTensorPool(CudaPagedTensorPool&& other) noexcept;
    CudaPagedTensorPool& operator=(CudaPagedTensorPool&& other) noexcept;

    static Result<CudaPagedTensorPool> create(
        int cudaDevice,
        DevicePagedPoolDesc desc,
        cudaStream_t stream);

    const DevicePagedPoolDesc& desc() const { return desc_; }
    int64_t page_element_count() const { return pageElementCount_; }
    size_t page_bytes() const { return pageBytes_; }
    size_t page_count() const { return pages_.size(); }

    Result<uint32_t> allocate_page(cudaStream_t stream);
    Result<void> free_page(uint32_t page);
    Result<void*> page_data(uint32_t page) const;

private:
    struct Page {
        void* data = nullptr;
    };

    explicit CudaPagedTensorPool(int cudaDevice);

    Result<void> initialize(DevicePagedPoolDesc desc, cudaStream_t stream);
    void release();

    int cudaDevice_ = 0;
    bool initialized_ = false;
    DevicePagedPoolDesc desc_;
    int64_t pageElementCount_ = 0;
    size_t pageBytes_ = 0;
    std::vector<Page> pages_;
    std::vector<uint32_t> freePages_;
};

class CudaPagedTensor {
public:
    CudaPagedTensor() = default;
    CudaPagedTensor(
        int cudaDevice,
        DevicePagedPoolId pool,
        core::Shape logicalShape,
        int64_t growLength);
    ~CudaPagedTensor();

    CudaPagedTensor(const CudaPagedTensor&) = delete;
    CudaPagedTensor& operator=(const CudaPagedTensor&) = delete;
    CudaPagedTensor(CudaPagedTensor&& other) noexcept;
    CudaPagedTensor& operator=(CudaPagedTensor&& other) noexcept;

    DevicePagedPoolId pool() const { return pool_; }
    const core::Shape& logical_shape() const { return logicalShape_; }
    int64_t grow_length() const { return growLength_; }
    std::vector<uint32_t>& page_indices() { return pageIndices_; }
    const std::vector<uint32_t>& page_indices() const { return pageIndices_; }
    void** page_table() const { return pageTable_; }

    void set_logical_shape(core::Shape shape) { logicalShape_ = std::move(shape); }
    void set_grow_length(int64_t growLength) { growLength_ = growLength; }

    Result<void> sync_page_table(const CudaPagedTensorPool& pool, cudaStream_t stream);

private:
    void release_table();

    int cudaDevice_ = 0;
    DevicePagedPoolId pool_ = 0;
    core::Shape logicalShape_;
    int64_t growLength_ = 0;
    std::vector<uint32_t> pageIndices_;
    void** pageTable_ = nullptr;
    int64_t pageTableCapacity_ = 0;
    void** hostPageTable_ = nullptr;
    int64_t hostPageTableCapacity_ = 0;
    int64_t syncedPageCount_ = 0;
    std::vector<void**> retiredHostPageTables_;
};

} // namespace sandy::device
