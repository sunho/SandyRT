#include "CudaJitEmbeddedSources.h"

#include "CudaJitAbiSource.generated.h"
#include "CudaJitElementwiseKernelSource.generated.h"
#include "CudaJitElementwiseTemplateSource.generated.h"
#include "CudaJitEvaluatorStubSource.generated.h"
#include "CudaJitLayoutTransformAbiSource.generated.h"
#include "CudaJitLayoutTransformKernelSource.generated.h"
#include "CudaJitReductionAbiSource.generated.h"
#include "CudaJitReductionKernelSource.generated.h"
#include "CudaJitTensorAccessSource.generated.h"

namespace sandy::device {

std::string_view embeddedElementwiseKernelSource() {
    return cuda_jit_embedded::kElementwiseKernel;
}

std::vector<CudaJitHeader> embeddedElementwiseHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitElementwiseTemplate.cuh", cuda_jit_embedded::kElementwiseTemplate},
        {"generated/ElementwiseEvaluator.cuh", cuda_jit_embedded::kEvaluatorStub},
    };
}

std::string_view embeddedReductionKernelSource() {
    return cuda_jit_embedded::kReductionKernel;
}

std::vector<CudaJitHeader> embeddedReductionHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitReductionAbi.cuh", cuda_jit_embedded::kReductionAbi},
    };
}

std::string_view embeddedLayoutTransformKernelSource() {
    return cuda_jit_embedded::kLayoutTransformKernel;
}

std::vector<CudaJitHeader> embeddedLayoutTransformHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitLayoutTransformAbi.cuh", cuda_jit_embedded::kLayoutTransformAbi},
    };
}

} // namespace sandy::device
