#include "Allocator.h"

#include <algorithm>
#include <limits>

namespace sandy::core {

namespace {

Result<size_t> align_up(size_t value, size_t alignment) {
    if (alignment == 0)
        return make_error("allocator alignment must be > 0");
    if ((alignment & (alignment - 1)) != 0)
        return make_error("allocator alignment must be a power of two");
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1))
        return make_error("allocator size overflow");
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

BumpAllocator::BumpAllocator(size_t maxBytes)
    : maxBytes_(maxBytes) {
    if (maxBytes_ > 0)
        data_.reserve(maxBytes_);
}

Result<size_t> BumpAllocator::allocate(size_t bytes, size_t alignment) {
    auto aligned = align_up(used_, alignment);
    if (!aligned)
        return make_error(aligned.error());
    if (*aligned > std::numeric_limits<size_t>::max() - bytes)
        return make_error("allocator size overflow");

    size_t end = *aligned + bytes;
    if (maxBytes_ > 0 && end > maxBytes_)
        return make_error("allocator capacity exceeded");
    if (end > data_.size())
        data_.resize(end);

    used_ = end;
    return *aligned;
}

std::span<uint8_t> BumpAllocator::bytes(size_t offset, size_t size) {
    if (offset > data_.size() || size > data_.size() - offset)
        return {};
    return std::span<uint8_t>(data_.data() + offset, size);
}

std::span<const uint8_t> BumpAllocator::bytes(size_t offset, size_t size) const {
    if (offset > data_.size() || size > data_.size() - offset)
        return {};
    return std::span<const uint8_t>(data_.data() + offset, size);
}

Result<void> FixedPagePool::initialize(
        size_t pageBytes,
        size_t initialPages,
        size_t maxPages) {
    if (pageBytes == 0)
        return make_error("fixed page pool page size must be > 0");
    if (maxPages > 0 && initialPages > maxPages)
        return make_error("fixed page pool initial page count exceeds max page count");

    pageBytes_ = pageBytes;
    maxPages_ = maxPages;
    arena_ = BumpAllocator(maxPages_ == 0 ? 0 : maxPages_ * pageBytes_);
    pageOffsets_.clear();
    freePages_.clear();

    for (size_t i = 0; i < initialPages; i++) {
        auto added = add_page();
        if (!added)
            return make_error(added.error());
        freePages_.push_back(*added);
    }
    return {};
}

Result<uint32_t> FixedPagePool::allocate() {
    if (!freePages_.empty()) {
        auto page = freePages_.back();
        freePages_.pop_back();
        return page;
    }
    return add_page();
}

Result<void> FixedPagePool::deallocate(uint32_t page) {
    if (page >= pageOffsets_.size())
        return make_error("fixed page pool page index out of range");
    if (std::find(freePages_.begin(), freePages_.end(), page) != freePages_.end())
        return make_error("fixed page pool page already freed");
    freePages_.push_back(page);
    return {};
}

Result<std::span<uint8_t>> FixedPagePool::page(uint32_t page) {
    if (page >= pageOffsets_.size())
        return make_error("fixed page pool page index out of range");
    return arena_.bytes(pageOffsets_[page], pageBytes_);
}

Result<std::span<const uint8_t>> FixedPagePool::page(uint32_t page) const {
    if (page >= pageOffsets_.size())
        return make_error("fixed page pool page index out of range");
    return arena_.bytes(pageOffsets_[page], pageBytes_);
}

Result<uint32_t> FixedPagePool::add_page() {
    if (maxPages_ > 0 && pageOffsets_.size() >= maxPages_)
        return make_error("fixed page pool capacity exceeded");
    if (pageOffsets_.size() > std::numeric_limits<uint32_t>::max())
        return make_error("fixed page pool page index overflow");

    auto offset = arena_.allocate(pageBytes_, alignof(std::max_align_t));
    if (!offset)
        return make_error(offset.error());
    pageOffsets_.push_back(*offset);
    return static_cast<uint32_t>(pageOffsets_.size() - 1);
}

} // namespace sandy::core
