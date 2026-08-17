#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "ShapeUtil.h"

#include <cublas_v2.h>

#include <limits>

namespace sandy::device {

namespace {

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

} // namespace sandy::device
