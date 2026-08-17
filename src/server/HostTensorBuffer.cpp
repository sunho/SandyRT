#include "HostTensorBuffer.h"

#include <utility>

namespace sandy::server {

HostTensorBuffer::HostTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
    : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

Result<void> HostTensorBuffer::load() {
    return {};
}

void HostTensorBuffer::unload() {}

std::span<const uint8_t> HostTensorBuffer::data() const {
    return data_;
}

} // namespace sandy::server
