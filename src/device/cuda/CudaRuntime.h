#pragma once

#include "Result.h"

#include <cuda_runtime.h>

#include <string>

namespace sandy::device {

Result<void> cuda_check(cudaError_t status, const std::string& context);

} // namespace sandy::device
