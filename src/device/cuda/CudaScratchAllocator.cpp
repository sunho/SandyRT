#include "CudaScratchAllocator.h"

#include "CudaDevice.h"

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
    auto offset = alignUp(cursor_);
    if (!offset) return make_error(offset.error());
    auto bytes = tensorBytes(desc);
    if (!bytes) return make_error(bytes.error());
    if (*offset > std::numeric_limits<size_t>::max() - *bytes)
        return make_error("CUDA scratch pool size overflow");
    placements_[value] = Placement{std::move(desc), *offset};
    live_[value] = true;
    cursor_ = *offset + *bytes;
    return {};
}

Result<void> CudaScratchAllocator::free(ir::kernel_ir::ValueId value) {
    if (finalized_)
        return make_error("CUDA scratch allocator is already finalized");
    auto live = live_.find(value);
    if (live == live_.end() || !live->second)
        return make_error("CUDA scratch value is not live");
    live->second = false;
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
