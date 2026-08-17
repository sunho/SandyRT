#pragma once

#include "TensorBuffer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sandy::server {

class HostTensorBuffer final : public core::TensorBuffer {
public:
    HostTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data);

private:
    Result<void> load() override;
    void unload() override;
    std::span<const uint8_t> data() const override;

    std::vector<uint8_t> data_;
};

} // namespace sandy::server
