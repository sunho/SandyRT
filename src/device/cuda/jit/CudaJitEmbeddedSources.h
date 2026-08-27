#pragma once
#include "CudaJit.h"

#include <string_view>
#include <vector>

namespace sandy::device {

std::string_view embeddedElementwiseKernelSource();
std::vector<CudaJitHeader> embeddedElementwiseHeaders();
std::string_view embeddedReductionKernelSource();
std::vector<CudaJitHeader> embeddedReductionHeaders();
std::string_view embeddedLayoutTransformKernelSource();
std::vector<CudaJitHeader> embeddedLayoutTransformHeaders();
std::string_view embeddedGatherKernelSource();
std::vector<CudaJitHeader> embeddedGatherHeaders();
std::string_view embeddedSoftmaxKernelSource();
std::vector<CudaJitHeader> embeddedSoftmaxHeaders();
std::string_view embeddedNormKernelSource();
std::vector<CudaJitHeader> embeddedNormHeaders();
std::string_view embeddedRoPEKernelSource();
std::vector<CudaJitHeader> embeddedRoPEHeaders();
std::string_view embeddedAttentionDecodeKernelSource();
std::vector<CudaJitHeader> embeddedAttentionDecodeHeaders();
std::string_view embeddedAttentionPrefillKernelSource();
std::vector<CudaJitHeader> embeddedAttentionPrefillHeaders();

} // namespace sandy::device
