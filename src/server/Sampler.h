#pragma once

#include "EngineTypes.h"
#include "Result.h"

#include <cstdint>
#include <utility>

namespace sandy::server {

class Sampler {
public:
    Result<std::pair<int64_t, float>> argmaxLast(engine::TensorBufferPtr logits) const;
    Result<std::pair<int64_t, float>> topkLast(
        engine::TensorBufferPtr values,
        engine::TensorBufferPtr indices) const;
};

} // namespace sandy::server
