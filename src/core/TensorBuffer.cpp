#include "TensorBuffer.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace sandy::core {

TensorBuffer::Access::Access(TensorBuffer& buffer)
    : buffer_(&buffer) {}

TensorBuffer::Access::Access(Access&& other) noexcept
    : buffer_(other.buffer_) {
    other.buffer_ = nullptr;
}

TensorBuffer::Access& TensorBuffer::Access::operator=(Access&& other) noexcept {
    if (this == &other) return *this;
    if (buffer_) buffer_->unmount();
    buffer_ = other.buffer_;
    other.buffer_ = nullptr;
    return *this;
}

TensorBuffer::Access::~Access() {
    if (buffer_) buffer_->unmount();
}

const TensorDesc& TensorBuffer::Access::desc() const {
    return buffer_->desc();
}

std::span<const uint8_t> TensorBuffer::Access::data() const {
    return buffer_->data();
}

TensorBuffer::TensorBuffer(TensorDesc desc)
    : desc_(std::move(desc)) {}

const TensorDesc& TensorBuffer::desc() const {
    return desc_;
}

Result<void> TensorBuffer::mount() {
    if (mountDepth_ == 0) {
        auto result = load();
        if (!result) return result;
    }
    mountDepth_++;
    return {};
}

void TensorBuffer::unmount() {
    if (mountDepth_ == 0) {
        fprintf(stderr, "TensorBuffer::unmount() called without mount\n");
        abort();
    }

    mountDepth_--;
    if (mountDepth_ == 0)
        unload();
}

Result<TensorBuffer::Access> TensorBuffer::access() {
    auto result = mount();
    if (!result) return make_error(result.error());
    return Access(*this);
}

bool TensorBuffer::is_mounted() const {
    return mountDepth_ > 0;
}

} // namespace sandy::core
