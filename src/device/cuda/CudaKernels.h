#pragma once

#include "CudaRuntime.h"
#include "DeviceTypes.h"
#include "KernelIR.h"
#include "Result.h"
#include "Tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace sandy::device {

struct CudaDeviceBufferView {
    void* data = nullptr;
    TensorViewDesc view;
    size_t bytes = 0;
    bool paged = false;
    int64_t growDim = -1;
    int64_t pageSize = -1;
    int64_t pageCount = 0;
    int64_t pageElementCount = 0;
};

struct CudaDevicePagedTensorView {
    void** pageTable = nullptr;
    DevicePagedTensorMeta meta;
    size_t pageBytes = 0;
};

struct CudaLaunchContext {
    int cudaDevice = 0;
    cudaStream_t stream = nullptr;
    ir::kernel_ir::OpId op = ir::kernel_ir::kInvalidOpId;
    std::span<const CudaDeviceBufferView> inputs;
    std::span<const CudaDeviceBufferView> outputs;
};

struct CudaElementwiseProgram {
    std::vector<ir::kernel_ir::ElementwiseInput> elementwiseInputs;
    ir::kernel_ir::ValueId output = 0;
    ir::kernel_ir::ScalarId result = 0;
    std::vector<ir::kernel_ir::ScalarNode> scalars;
};

struct CudaLayoutTransformProgram {
    ir::kernel_ir::LayoutTransformKind transform =
        ir::kernel_ir::LayoutTransformKind::Contiguous;
    std::vector<int64_t> dims;
};

struct CudaMatMulProgram {
    bool transposeLhs = false;
    bool transposeRhs = false;
};

struct CudaReductionProgram {
    ir::kernel_ir::ReduceOp reduce = ir::kernel_ir::ReduceOp::Sum;
    std::vector<int64_t> axes;
    bool keepDims = false;
};

struct CudaSoftmaxProgram {
    int64_t axis = -1;
};

struct CudaNormProgram {
    ir::kernel_ir::NormKind norm = ir::kernel_ir::NormKind::RMSNorm;
    double epsilon = 0.0;
};

struct CudaRoPEProgram {
    double theta = 10000.0;
    int64_t rotaryDim = -1;
    bool splitHalf = false;
};

struct CudaSlidingQueryKeyScoreProgram {
    int64_t window = 0;
    double scale = -1.0;
};

struct CudaAttentionProgram {
    int64_t window = 0;
    double scale = -1.0;
};

struct CudaCustomProgram {
    std::string customName;
};

Result<void> launch_cuda_elementwise(
    const CudaLaunchContext& context,
    const CudaElementwiseProgram& program);

Result<void> launch_cuda_layout_transform(
    const CudaLaunchContext& context,
    const CudaLayoutTransformProgram& program);

Result<void> launch_cuda_matmul(
    const CudaLaunchContext& context,
    const CudaMatMulProgram& program);

Result<void> launch_cuda_gather(const CudaLaunchContext& context);

Result<void> launch_cuda_softmax(
    const CudaLaunchContext& context,
    const CudaSoftmaxProgram& program);

Result<void> launch_cuda_norm(
    const CudaLaunchContext& context,
    const CudaNormProgram& program);

Result<void> launch_cuda_rope(
    const CudaLaunchContext& context,
    const CudaRoPEProgram& program);

Result<void> launch_cuda_sliding_query_key_score(
    const CudaLaunchContext& context,
    const CudaSlidingQueryKeyScoreProgram& program);

Result<void> launch_cuda_attention(
    const CudaLaunchContext& context,
    const CudaAttentionProgram& program);

Result<void> launch_cuda_reduction(
    const CudaLaunchContext& context,
    const CudaReductionProgram& program);

Result<void> launch_cuda_custom(
    const CudaLaunchContext& context,
    const CudaCustomProgram& program);

} // namespace sandy::device
