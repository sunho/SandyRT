#pragma once

#include "EngineTypes.h"
#include "Result.h"

#include <cstdint>
#include <utility>

namespace sandy::server {

class Sampler {
public:
    Result<std::pair<int64_t, float>> argmaxLast(engine::TensorBufferPtr logits) const;
};

} // namespace sandy::server
