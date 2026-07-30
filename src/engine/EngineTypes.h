#pragma once

#include "TensorBuffer.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace sandy::engine {

struct RunOptions {};

using TensorBufferPtr = std::shared_ptr<core::TensorBuffer>;
using TensorMap = std::unordered_map<std::string, TensorBufferPtr>;

} // namespace sandy::engine
