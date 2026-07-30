#pragma once

#include "Shape.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace weight {

class Weights {
public:
    virtual ~Weights() = default;
    virtual std::vector<ir::TensorDesc> get_descriptors() const = 0;
    virtual ir::TensorDesc get_descriptor(const std::string& name) const = 0;
    virtual bool has(const std::string& name) const = 0;
    virtual std::span<const uint8_t> get_buffer(const std::string& name) const = 0;
};

} // namespace weight
