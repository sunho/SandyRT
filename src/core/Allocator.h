#pragma once

#include "Result.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sandy::core {

class BumpAllocator {
public:
    explicit BumpAllocator(size_t maxBytes = 0);

    Result<size_t> allocate(size_t bytes, size_t alignment = alignof(std::max_align_t));

    std::span<uint8_t> bytes(size_t offset, size_t size);
    std::span<const uint8_t> bytes(size_t offset, size_t size) const;

    size_t used() const { return used_; }
    size_t capacity() const { return data_.size(); }
    size_t max_bytes() const { return maxBytes_; }

private:
    size_t maxBytes_ = 0;
    size_t used_ = 0;
    std::vector<uint8_t> data_;
};

class FixedPagePool {
public:
    Result<void> initialize(size_t pageBytes, size_t initialPages, size_t maxPages = 0);

    Result<uint32_t> allocate();
    Result<void> deallocate(uint32_t page);

    Result<std::span<uint8_t>> page(uint32_t page);
    Result<std::span<const uint8_t>> page(uint32_t page) const;

    size_t page_bytes() const { return pageBytes_; }
    size_t page_count() const { return pageOffsets_.size(); }
    size_t free_page_count() const { return freePages_.size(); }
    size_t max_pages() const { return maxPages_; }

private:
    Result<uint32_t> add_page();

    size_t pageBytes_ = 0;
    size_t maxPages_ = 0;
    BumpAllocator arena_;
    std::vector<size_t> pageOffsets_;
    std::vector<uint32_t> freePages_;
};

} // namespace sandy::core
