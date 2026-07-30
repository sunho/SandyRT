#pragma once

#include "Tensor.h"
#include "TensorBuffer.h"

#include <memory>
#include <string>
#include <vector>

namespace sandy::weight {

class Weights {
public:
    virtual ~Weights() = default;
    virtual std::vector<core::TensorDesc> descriptors() const = 0;
    virtual std::shared_ptr<core::TensorBuffer> get_tensor(
        const std::string& name) const = 0;
};

} // namespace sandy::weight
