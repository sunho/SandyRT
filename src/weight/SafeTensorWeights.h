#pragma once

#include "Weight.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace weight {

class EagerSafeTensorWeights : public Weights {
public:
    static EagerSafeTensorWeights load(const std::string& path);

    std::vector<ir::TensorDesc> get_descriptors() const override;
    ir::TensorDesc get_descriptor(const std::string& name) const override;
    bool has(const std::string& name) const override;
    std::span<const uint8_t> get_buffer(const std::string& name) const override;

private:
    struct TensorInfo {
        ir::Shape shape;
        ir::DType dtype;
        size_t offset;
        size_t size;
    };

    std::vector<uint8_t> data_;
    size_t dataOffset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::vector<std::string> names_;
};

} // namespace weight
