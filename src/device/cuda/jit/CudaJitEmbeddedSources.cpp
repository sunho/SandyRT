#include "CudaJitEmbeddedSources.h"

#include "CudaJitAbiSource.generated.h"
#include "CudaJitElementwiseKernelSource.generated.h"
#include "CudaJitElementwiseConfigStubSource.generated.h"
#include "CudaJitElementwiseTemplateSource.generated.h"
#include "CudaJitEvaluatorStubSource.generated.h"
#include "CudaJitGatherAbiSource.generated.h"
#include "CudaJitGatherKernelSource.generated.h"
#include "CudaJitLayoutTransformAbiSource.generated.h"
#include "CudaJitLayoutTransformKernelSource.generated.h"
#include "CudaJitNormAbiSource.generated.h"
#include "CudaJitNormKernelSource.generated.h"
#include "CudaJitReductionAbiSource.generated.h"
#include "CudaJitReductionKernelSource.generated.h"
#include "CudaJitRoPEAbiSource.generated.h"
#include "CudaJitRoPEKernelSource.generated.h"
#include "CudaJitSoftmaxAbiSource.generated.h"
#include "CudaJitSoftmaxKernelSource.generated.h"
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
        {"generated/ElementwiseConfig.cuh", cuda_jit_embedded::kElementwiseConfigStub},
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

std::string_view embeddedGatherKernelSource() {
    return cuda_jit_embedded::kGatherKernel;
}

std::vector<CudaJitHeader> embeddedGatherHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitGatherAbi.cuh", cuda_jit_embedded::kGatherAbi},
    };
}

std::string_view embeddedSoftmaxKernelSource() {
    return cuda_jit_embedded::kSoftmaxKernel;
}

std::vector<CudaJitHeader> embeddedSoftmaxHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitSoftmaxAbi.cuh", cuda_jit_embedded::kSoftmaxAbi},
    };
}

std::string_view embeddedNormKernelSource() {
    return cuda_jit_embedded::kNormKernel;
}

std::vector<CudaJitHeader> embeddedNormHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitNormAbi.cuh", cuda_jit_embedded::kNormAbi},
    };
}

std::string_view embeddedRoPEKernelSource() {
    return cuda_jit_embedded::kRoPEKernel;
}

std::vector<CudaJitHeader> embeddedRoPEHeaders() {
    return {
        {"CudaJitAbi.cuh", cuda_jit_embedded::kAbi},
        {"CudaJitTensorAccess.cuh", cuda_jit_embedded::kTensorAccess},
        {"CudaJitRoPEAbi.cuh", cuda_jit_embedded::kRoPEAbi},
    };
}

} // namespace sandy::device
