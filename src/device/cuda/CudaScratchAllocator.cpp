#include "CudaScratchAllocator.h"

#include "CudaDevice.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace sandy::device {

namespace {

constexpr size_t kScratchAlignment = 256;
constexpr size_t kVectorLoadAlignment = 16;
static_assert(kScratchAlignment % kVectorLoadAlignment == 0);

Result<size_t> tensorBytes(const core::TensorDesc& desc) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error("CUDA scratch tensor shape must be static");
    auto elementBytes = core::dtype_size(desc.dtype);
    if (static_cast<size_t>(numel) > std::numeric_limits<size_t>::max() / elementBytes)
        return make_error("CUDA scratch tensor size overflow");
    return static_cast<size_t>(numel) * elementBytes;
}

Result<size_t> alignUp(size_t value) {
    if (value > std::numeric_limits<size_t>::max() - (kScratchAlignment - 1))
        return make_error("CUDA scratch offset overflow");
    return (value + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
}

} // namespace

Result<void> CudaScratchAllocator::alloc(
        ir::kernel_ir::ValueId value,
        core::TensorDesc desc) {
    if (finalized_)
        return make_error("CUDA scratch allocator is already finalized");
    if (placements_.contains(value))
        return make_error("CUDA scratch value was allocated more than once");
    auto bytes = tensorBytes(desc);
    if (!bytes) return make_error(bytes.error());
    auto reservedBytes = alignUp(*bytes);
    if (!reservedBytes) return make_error(reservedBytes.error());

    auto best = freeRanges_.end();
    for (auto it = freeRanges_.begin(); it != freeRanges_.end(); ++it) {
        if (it->bytes < *reservedBytes)
            continue;
        if (best == freeRanges_.end() || it->bytes < best->bytes)
            best = it;
    }

    size_t offset = 0;
    if (best != freeRanges_.end()) {
        offset = best->byteOffset;
        best->byteOffset += *reservedBytes;
        best->bytes -= *reservedBytes;
        if (best->bytes == 0)
            freeRanges_.erase(best);
    } else {
        auto alignedCursor = alignUp(cursor_);
        if (!alignedCursor) return make_error(alignedCursor.error());
        offset = *alignedCursor;
    }
    if (offset > std::numeric_limits<size_t>::max() - *reservedBytes)
        return make_error("CUDA scratch pool size overflow");
    placements_[value] = Placement{std::move(desc), offset, *reservedBytes};
    live_[value] = true;
    cursor_ = std::max(cursor_, offset + *reservedBytes);
    return {};
}

Result<void> CudaScratchAllocator::free(ir::kernel_ir::ValueId value) {
    if (finalized_)
        return make_error("CUDA scratch allocator is already finalized");
    auto live = live_.find(value);
    if (live == live_.end() || !live->second)
        return make_error("CUDA scratch value is not live");
    live->second = false;

    const auto& placement = placements_.at(value);
    freeRanges_.push_back({placement.byteOffset, placement.reservedBytes});
    std::sort(
        freeRanges_.begin(),
        freeRanges_.end(),
        [](const FreeRange& lhs, const FreeRange& rhs) {
            return lhs.byteOffset < rhs.byteOffset;
        });
    std::vector<FreeRange> merged;
    merged.reserve(freeRanges_.size());
    for (const auto& range : freeRanges_) {
        if (!merged.empty() &&
            merged.back().byteOffset + merged.back().bytes == range.byteOffset) {
            merged.back().bytes += range.bytes;
        } else {
            merged.push_back(range);
        }
    }
    freeRanges_ = std::move(merged);
    return {};
}

Result<DeviceScratchAllocation> CudaScratchAllocator::finalize() {
    if (finalized_)
        return make_error("CUDA scratch allocator is already finalized");
    finalized_ = true;

    DeviceScratchAllocation result;
    if (placements_.empty())
        return result;
    auto buffer = device_.alloc(core::TensorDesc(
        core::Shape({static_cast<int64_t>(cursor_)}),
        core::DType::U8));
    if (!buffer) return make_error(buffer.error());
    result.buffer = buffer.take();

    for (const auto& [value, placement] : placements_) {
        auto view = device_.defaultView(placement.desc);
        if (!view) {
            auto deallocated = device_.dealloc(result.buffer);
            if (!deallocated) return make_error(deallocated.error());
            return make_error(view.error());
        }
        auto elementBytes = core::dtype_size(placement.desc.dtype);
        if (placement.byteOffset % kVectorLoadAlignment != 0) {
            auto deallocated = device_.dealloc(result.buffer);
            if (!deallocated) return make_error(deallocated.error());
            return make_error("CUDA scratch placement is not vector-load aligned");
        }
        if (placement.byteOffset % elementBytes != 0) {
            auto deallocated = device_.dealloc(result.buffer);
            if (!deallocated) return make_error(deallocated.error());
            return make_error("CUDA scratch placement is not element aligned");
        }
        auto viewValue = view.take();
        viewValue.storageOffset = static_cast<int64_t>(placement.byteOffset / elementBytes);
        result.views[value] = DeviceTensorView{result.buffer, std::move(viewValue)};
    }
    return result;
}

} // namespace sandy::device
