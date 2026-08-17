#include "CudaKernels.h"
#include "CudaKernelUtils.cuh"
#include "ShapeUtil.h"

#include <cublas_v2.h>

#include <cmath>
#include <limits>

namespace sandy::device {

namespace {

constexpr int kMaxScalars = 32;

struct DeviceScalarNode {
    int id = 0;
    ir::kernel_ir::ScalarOp op = ir::kernel_ir::ScalarOp::Constant;
    core::DType dtype = core::DType::F32;
    uint32_t inputIndex = 0;
    double constant = 0.0;
    int operands[2]{};
    int operandCount = 0;
};

struct DeviceElementwiseProgram {
    cuda_kernel::TensorInputs<> inputs;
    ir::kernel_ir::BroadcastMode inputBroadcasts[cuda_kernel::kMaxInputs]{};
    cuda_kernel::TensorArg output;
    DeviceScalarNode scalars[kMaxScalars]{};
    int scalarCount = 0;
    int result = 0;
    int64_t numel = 0;
};

Result<void> validate_context(
        const CudaLaunchContext& context,
        size_t minInputs,
        size_t minOutputs,
        const char* kernelName) {
    if (!context.stream)
        return make_error(std::string("cuda ") + kernelName + " launch has null stream");
    if (context.inputs.size() < minInputs)
        return make_error(std::string("cuda ") + kernelName + " input arity mismatch");
    if (context.outputs.size() < minOutputs)
        return make_error(std::string("cuda ") + kernelName + " output arity mismatch");
    for (const auto& input : context.inputs) {
        if (!input.data && input.bytes != 0)
            return make_error(std::string("cuda ") + kernelName + " input buffer is null");
    }
    for (const auto& output : context.outputs) {
        if (!output.data && output.bytes != 0)
            return make_error(std::string("cuda ") + kernelName + " output buffer is null");
    }
    return {};
}

Result<void> unimplemented(const char* kernelName) {
    return make_error(std::string("cuda ") + kernelName + " kernel is not implemented yet");
}

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

Result<void> cublas_check(cublasStatus_t status, const std::string& context) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return {};
    return make_error(context + ": " + cublas_status_name(status));
}

Result<cudaDataType_t> cublas_data_type_for(core::DType dtype) {
    switch (dtype) {
        case core::DType::F32:
            return CUDA_R_32F;
        case core::DType::BF16:
            return CUDA_R_16BF;
        default:
            return make_error("cuda matmul unsupported dtype");
    }
}

bool cublas_int_dims_ok(
        int64_t rows,
        int64_t n,
        int64_t k,
        int64_t lhsLd,
        int64_t rhsLd,
        int64_t outLd,
        int64_t batchNumel) {
    int64_t maxInt = std::numeric_limits<int>::max();
    return rows <= maxInt && n <= maxInt && k <= maxInt &&
           lhsLd <= maxInt && rhsLd <= maxInt && outLd <= maxInt &&
           batchNumel <= maxInt;
}

void* byte_offset(void* ptr, int64_t elementOffset, core::DType dtype) {
    auto* bytes = static_cast<char*>(ptr);
    return bytes + elementOffset * static_cast<int64_t>(core::dtype_size(dtype));
}

bool is_float_compute_dtype(core::DType dtype) {
    return dtype == core::DType::F32 || dtype == core::DType::BF16;
}

Result<void> validate_elementwise_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda elementwise ") + name + " unsupported dtype");
    return {};
}

Result<DeviceElementwiseProgram> pack_elementwise_program(
        const CudaLaunchContext& context,
        const CudaElementwiseProgram& program) {
    if (context.inputs.size() > cuda_kernel::kMaxInputs)
        return make_error("cuda elementwise input count exceeds kernel max inputs");
    if (program.scalars.size() > kMaxScalars)
        return make_error("cuda elementwise scalar count exceeds kernel max scalars");
    if (program.result >= kMaxScalars)
        return make_error("cuda elementwise result scalar id exceeds kernel max scalars");

    auto outputDtype = validate_elementwise_dtype(context.outputs[0], "output");
    if (!outputDtype)
        return make_error(outputDtype.error());

    DeviceElementwiseProgram packed;
    auto output = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!output)
        return make_error(output.error());
    packed.output = output.take();
    packed.numel = packed.output.numel;
    packed.result = static_cast<int>(program.result);

    packed.inputs.count = static_cast<int>(context.inputs.size());
    for (int i = 0; i < packed.inputs.count; ++i) {
        auto dtype = validate_elementwise_dtype(context.inputs[static_cast<size_t>(i)], "input");
        if (!dtype)
            return make_error(dtype.error());
        auto input = cuda_kernel::pack_tensor_arg(context.inputs[static_cast<size_t>(i)]);
        if (!input)
            return make_error(input.error());
        packed.inputs.items[i] = input.take();
        packed.inputBroadcasts[i] =
            program.elementwiseInputs[static_cast<size_t>(i)].broadcast;
    }

    bool seen[kMaxScalars]{};
    packed.scalarCount = static_cast<int>(program.scalars.size());
    for (int i = 0; i < packed.scalarCount; ++i) {
        const auto& src = program.scalars[static_cast<size_t>(i)];
        if (src.id >= kMaxScalars)
            return make_error("cuda elementwise scalar id exceeds kernel max scalars");
        if (seen[src.id])
            return make_error("cuda elementwise duplicate scalar id");
        if (src.operands.size() > 2)
            return make_error("cuda elementwise scalar op has too many operands");
        if (src.op == ir::kernel_ir::ScalarOp::Load &&
            src.inputIndex >= context.inputs.size()) {
            return make_error("cuda elementwise scalar load references invalid input");
        }

        auto& dst = packed.scalars[i];
        dst.id = static_cast<int>(src.id);
        dst.op = src.op;
        dst.dtype = src.dtype;
        dst.inputIndex = src.inputIndex;
        dst.constant = src.constant;
        dst.operandCount = static_cast<int>(src.operands.size());

        for (int j = 0; j < dst.operandCount; ++j) {
            auto operand = src.operands[static_cast<size_t>(j)];
            if (operand >= kMaxScalars)
                return make_error("cuda elementwise operand id exceeds kernel max scalars");
            if (!seen[operand])
                return make_error("cuda elementwise scalar operands must be in dependency order");
            dst.operands[j] = static_cast<int>(operand);
        }

        seen[src.id] = true;
    }

    if (!seen[program.result])
        return make_error("cuda elementwise result references missing scalar");

    return packed;
}

__device__ float apply_scalar(
        const DeviceScalarNode& node,
        const float* values) {
    float a = node.operandCount > 0 ? values[node.operands[0]] : 0.0f;
    float b = node.operandCount > 1 ? values[node.operands[1]] : 0.0f;

    switch (node.op) {
        case ir::kernel_ir::ScalarOp::Constant:
            return static_cast<float>(node.constant);
        case ir::kernel_ir::ScalarOp::Add:
            return a + b;
        case ir::kernel_ir::ScalarOp::Sub:
            return a - b;
        case ir::kernel_ir::ScalarOp::Mul:
            return a * b;
        case ir::kernel_ir::ScalarOp::Div:
            return a / b;
        case ir::kernel_ir::ScalarOp::Max:
            return fmaxf(a, b);
        case ir::kernel_ir::ScalarOp::Min:
            return fminf(a, b);
        case ir::kernel_ir::ScalarOp::Neg:
            return -a;
        case ir::kernel_ir::ScalarOp::Sqrt:
            return sqrtf(a);
        case ir::kernel_ir::ScalarOp::Rsqrt:
            return rsqrtf(a);
        case ir::kernel_ir::ScalarOp::Exp:
            return expf(a);
        case ir::kernel_ir::ScalarOp::Log:
            return logf(a);
        case ir::kernel_ir::ScalarOp::Tanh:
            return tanhf(a);
        case ir::kernel_ir::ScalarOp::ReLU:
            return fmaxf(a, 0.0f);
        case ir::kernel_ir::ScalarOp::Cast:
            return a;
        case ir::kernel_ir::ScalarOp::Load:
            return 0.0f;
    }

    return 0.0f;
}

__global__ void elementwise_kernel(DeviceElementwiseProgram program) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.numel)
        return;

    float values[kMaxScalars]{};
    for (int i = 0; i < program.scalarCount; ++i) {
        const auto& node = program.scalars[i];
        if (node.op == ir::kernel_ir::ScalarOp::Load) {
            const auto& input = program.inputs.items[node.inputIndex];
            int64_t inputLinear = linear;
            if (program.inputBroadcasts[node.inputIndex] ==
                ir::kernel_ir::BroadcastMode::RightAligned) {
                inputLinear = input.numel == 0 ? 0 : linear % input.numel;
            }
            values[node.id] = cuda_kernel::load_float(
                input,
                inputLinear);
        } else {
            values[node.id] = apply_scalar(node, values);
        }
    }

    cuda_kernel::store_float(program.output, linear, values[program.result]);
}

struct MatMulShape {
    int lhsRank = 0;
    int rhsRank = 0;
    int outputRank = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    core::Shape batchShape;
    int64_t batchNumel = 0;
};

Result<MatMulShape> validate_matmul(
        const CudaLaunchContext& context,
        const CudaMatMulProgram& program) {
    const auto& lhs = context.inputs[0].view;
    const auto& rhs = context.inputs[1].view;
    const auto& output = context.outputs[0].view;

    if (lhs.desc.dtype != rhs.desc.dtype || lhs.desc.dtype != output.desc.dtype)
        return make_error("cuda matmul operands must have same dtype");
    if (!is_float_compute_dtype(lhs.desc.dtype))
        return make_error("cuda matmul unsupported dtype");
    if (!cuda_kernel::is_contiguous(lhs) ||
        !cuda_kernel::is_contiguous(rhs) ||
        !cuda_kernel::is_contiguous(output)) {
        return make_error("cuda matmul requires contiguous tensor views");
    }

    MatMulShape shape;
    shape.lhsRank = lhs.desc.shape.rank();
    shape.rhsRank = rhs.desc.shape.rank();
    shape.outputRank = output.desc.shape.rank();
    if (shape.lhsRank < 2)
        return make_error("cuda matmul lhs must have rank >= 2");
    if (shape.rhsRank < 2)
        return make_error("cuda matmul rhs must have rank >= 2");

    shape.m = lhs.desc.shape.dim(shape.lhsRank - (program.transposeLhs ? 1 : 2));
    int64_t lhsK = lhs.desc.shape.dim(shape.lhsRank - (program.transposeLhs ? 2 : 1));
    int64_t rhsK = rhs.desc.shape.dim(shape.rhsRank - (program.transposeRhs ? 1 : 2));
    shape.n = rhs.desc.shape.dim(shape.rhsRank - (program.transposeRhs ? 2 : 1));
    if (shape.m < 0 || shape.n < 0 || lhsK < 0 || rhsK < 0)
        return make_error("cuda matmul matrix dimensions must be static");
    if (lhsK != rhsK)
        return make_error("cuda matmul contracting dimension mismatch");
    shape.k = lhsK;

    auto lhsDims = lhs.desc.shape.dims();
    auto rhsDims = rhs.desc.shape.dims();
    core::Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
    core::Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
    auto batchShape = core::matmul_batch_shape(lhsBatch, rhsBatch);
    if (!batchShape)
        return make_error(batchShape.error());
    shape.batchShape = batchShape.take();
    shape.batchNumel = shape.batchShape.numel();
    if (shape.batchNumel < 0)
        return make_error("cuda matmul output must have static shape");

    auto outDims = shape.batchShape.dims();
    outDims.push_back(shape.m);
    outDims.push_back(shape.n);
    if (output.desc.shape != core::Shape(outDims))
        return make_error("cuda matmul output shape mismatch");

    int64_t lhsLd = lhs.desc.shape.dim(shape.lhsRank - 1);
    int64_t rhsLd = rhs.desc.shape.dim(shape.rhsRank - 1);
    int64_t maxRows = shape.batchNumel * shape.m;
    if (!cublas_int_dims_ok(maxRows, shape.n, shape.k, lhsLd, rhsLd, shape.n, shape.batchNumel))
        return make_error("cuda matmul dimensions exceed cuBLAS int limits");

    return shape;
}

int64_t matmul_batch_offset(
        int64_t batchIndex,
        const core::Shape& batchShape,
        const cuda_kernel::TensorArg& source) {
    int sourceBatchRank = source.rank - 2;
    if (sourceBatchRank <= 0)
        return 0;

    int64_t sourceOffset = 0;
    int64_t remaining = batchIndex;
    int rankOffset = batchShape.rank() - sourceBatchRank;
    for (int batchDimIndex = batchShape.rank() - 1; batchDimIndex >= 0; --batchDimIndex) {
        int64_t batchDim = batchShape.dim(batchDimIndex);
        int64_t coord = remaining % batchDim;
        remaining /= batchDim;

        int sourceDimIndex = batchDimIndex - rankOffset;
        if (sourceDimIndex < 0)
            continue;

        int64_t sourceDim = source.dims[sourceDimIndex];
        if (sourceDim == 1)
            continue;
        if (sourceDim != batchDim)
            coord /= batchDim / sourceDim;
        sourceOffset += coord * source.strides[sourceDimIndex];
    }
    return sourceOffset;
}

Result<void> run_cublas_gemm(
        cublasHandle_t handle,
        const cuda_kernel::TensorArg& lhs,
        const cuda_kernel::TensorArg& rhs,
        const cuda_kernel::TensorArg& output,
        const MatMulShape& shape,
        const CudaMatMulProgram& program,
        int64_t rows,
        int64_t lhsOffset,
        int64_t rhsOffset,
        int64_t outputOffset) {
    auto dataType = cublas_data_type_for(lhs.dtype);
    if (!dataType)
        return make_error(dataType.error());
    cudaDataType_t cudaType = dataType.take();

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasOperation_t lhsOp = program.transposeLhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t rhsOp = program.transposeRhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    int64_t lhsLd = lhs.dims[shape.lhsRank - 1];
    int64_t rhsLd = rhs.dims[shape.rhsRank - 1];

    auto status = cublasGemmEx(
        handle,
        rhsOp,
        lhsOp,
        static_cast<int>(shape.n),
        static_cast<int>(rows),
        static_cast<int>(shape.k),
        &alpha,
        byte_offset(rhs.data, rhs.storageOffset + rhsOffset, rhs.dtype),
        cudaType,
        static_cast<int>(rhsLd),
        byte_offset(lhs.data, lhs.storageOffset + lhsOffset, lhs.dtype),
        cudaType,
        static_cast<int>(lhsLd),
        &beta,
        byte_offset(output.data, output.storageOffset + outputOffset, output.dtype),
        cudaType,
        static_cast<int>(shape.n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    return cublas_check(status, "cublasGemmEx");
}

} // namespace

Result<void> launch_cuda_elementwise(
        const CudaLaunchContext& context,
        const CudaElementwiseProgram& program) {
    auto valid = validate_context(
        context,
        program.elementwiseInputs.size(),
        1,
        "elementwise");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_elementwise_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.numel == 0)
        return {};

    int blocks = static_cast<int>(
        (launchProgram.numel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    elementwise_kernel<<<blocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
        launchProgram);
    return cuda_check(cudaGetLastError(), "cuda elementwise launch");
}

Result<void> launch_cuda_layout_transform(
        const CudaLaunchContext& context,
        const CudaLayoutTransformProgram&) {
    auto valid = validate_context(context, 1, 1, "layout_transform");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("layout_transform");
}

Result<void> launch_cuda_matmul(
        const CudaLaunchContext& context,
        const CudaMatMulProgram& program) {
    auto valid = validate_context(context, 2, 1, "matmul");
    if (!valid)
        return make_error(valid.error());

    auto shape = validate_matmul(context, program);
    if (!shape)
        return make_error(shape.error());
    auto matmulShape = shape.take();
    if (matmulShape.batchNumel == 0 || matmulShape.m == 0 || matmulShape.n == 0)
        return {};

    auto lhsArg = cuda_kernel::pack_tensor_arg(context.inputs[0]);
    if (!lhsArg)
        return make_error(lhsArg.error());
    auto rhsArg = cuda_kernel::pack_tensor_arg(context.inputs[1]);
    if (!rhsArg)
        return make_error(rhsArg.error());
    auto outputArg = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!outputArg)
        return make_error(outputArg.error());
    auto lhs = lhsArg.take();
    auto rhs = rhsArg.take();
    auto output = outputArg.take();

    cublasHandle_t handle = nullptr;
    auto create = cublas_check(cublasCreate(&handle), "cublasCreate");
    if (!create)
        return make_error(create.error());
    auto destroyHandle = [&]() {
        if (handle)
            cublasDestroy(handle);
    };

    auto stream = cublas_check(cublasSetStream(handle, context.stream), "cublasSetStream");
    if (!stream) {
        destroyHandle();
        return make_error(stream.error());
    }

    bool flattenSharedRhs = !program.transposeLhs && matmulShape.rhsRank == 2;
    if (flattenSharedRhs) {
        int64_t rows = matmulShape.batchNumel * matmulShape.m;
        auto gemm = run_cublas_gemm(
            handle,
            lhs,
            rhs,
            output,
            matmulShape,
            program,
            rows,
            0,
            0,
            0);
        destroyHandle();
        if (!gemm)
            return make_error(gemm.error());
        return {};
    }

    for (int64_t batch = 0; batch < matmulShape.batchNumel; ++batch) {
        int64_t lhsOffset = matmul_batch_offset(batch, matmulShape.batchShape, lhs);
        int64_t rhsOffset = matmul_batch_offset(batch, matmulShape.batchShape, rhs);
        int64_t outputOffset = matmul_batch_offset(batch, matmulShape.batchShape, output);

        auto gemm = run_cublas_gemm(
            handle,
            lhs,
            rhs,
            output,
            matmulShape,
            program,
            matmulShape.m,
            lhsOffset,
            rhsOffset,
            outputOffset);
        if (!gemm) {
            destroyHandle();
            return make_error(gemm.error());
        }
    }

    destroyHandle();
    return {};
}

Result<void> launch_cuda_gather(const CudaLaunchContext& context) {
    auto valid = validate_context(context, 2, 1, "gather");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("gather");
}

Result<void> launch_cuda_softmax(
        const CudaLaunchContext& context,
        const CudaSoftmaxProgram&) {
    auto valid = validate_context(context, 1, 1, "softmax");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("softmax");
}

Result<void> launch_cuda_norm(
        const CudaLaunchContext& context,
        const CudaNormProgram&) {
    auto valid = validate_context(context, 1, 1, "norm");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("norm");
}

Result<void> launch_cuda_rope(
        const CudaLaunchContext& context,
        const CudaRoPEProgram&) {
    auto valid = validate_context(context, 1, 1, "rope");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("rope");
}

Result<void> launch_cuda_sliding_query_key_score(
        const CudaLaunchContext& context,
        const CudaSlidingQueryKeyScoreProgram&) {
    auto valid = validate_context(context, 2, 1, "sliding_query_key_score");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("sliding_query_key_score");
}

Result<void> launch_cuda_reduction(
        const CudaLaunchContext& context,
        const CudaReductionProgram&) {
    auto valid = validate_context(context, 1, 1, "reduction");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("reduction");
}

Result<void> launch_cuda_custom(
        const CudaLaunchContext& context,
        const CudaCustomProgram& program) {
    auto valid = validate_context(context, 0, 1, "custom");
    if (!valid)
        return make_error(valid.error());
    return unimplemented(program.customName.empty() ? "custom" : program.customName.c_str());
}

} // namespace sandy::device
