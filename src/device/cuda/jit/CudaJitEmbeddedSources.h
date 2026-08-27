#pragma once
#include "CudaJit.h"

#include <string_view>
#include <vector>

namespace sandy::device {

std::string_view embeddedElementwiseKernelSource();
std::vector<CudaJitHeader> embeddedElementwiseHeaders();
std::string_view embeddedReductionKernelSource();
std::vector<CudaJitHeader> embeddedReductionHeaders();

} // namespace sandy::device
