#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "ShapeUtil.h"

#include <cublas_v2.h>

#include <limits>
#include <vector>

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

    if (context.inputs[0].paged || context.inputs[1].paged)
        return make_error("cuda matmul does not support paged tensor operands");
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

    if (!context.cublas)
        return make_error("cuda matmul missing cuBLAS handle");

    bool flattenSharedRhs = !program.transposeLhs && matmulShape.rhsRank == 2;
    if (flattenSharedRhs) {
        int64_t rows = matmulShape.batchNumel * matmulShape.m;
        auto gemm = run_cublas_gemm(
            context.cublas,
            lhs,
            rhs,
            output,
            matmulShape,
            program,
            rows,
            0,
            0,
            0);
        if (!gemm)
            return make_error(gemm.error());
        return {};
    }

    for (int64_t batch = 0; batch < matmulShape.batchNumel; ++batch) {
        int64_t lhsOffset = matmul_batch_offset(batch, matmulShape.batchShape, lhs);
        int64_t rhsOffset = matmul_batch_offset(batch, matmulShape.batchShape, rhs);
        int64_t outputOffset = matmul_batch_offset(batch, matmulShape.batchShape, output);

        auto gemm = run_cublas_gemm(
            context.cublas,
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
            return make_error(gemm.error());
        }
    }

    return {};
}

Result<void> launch_cuda_moe_matmul(
        const CudaLaunchContext& context,
        const CudaMoeMatMulProgram& program) {
#if CUDART_VERSION < 12050
    (void)context;
    (void)program;
    return make_error("cuda moe_matmul grouped GEMM requires CUDA toolkit 12.5 or newer");
#else
    auto valid = validate_context(context, 3, 1, "moe_matmul");
    if (!valid)
        return make_error(valid.error());

    const auto& xView = context.inputs[0];
    const auto& offsetView = context.inputs[1];
    const auto& weightView = context.inputs[2];
    const auto& outputView = context.outputs[0];

    if (xView.paged || offsetView.paged || weightView.paged)
        return make_error("cuda moe_matmul does not support paged tensor operands");
    if (xView.view.desc.dtype != weightView.view.desc.dtype ||
        xView.view.desc.dtype != outputView.view.desc.dtype) {
        return make_error("cuda moe_matmul x, weight, and output must have same dtype");
    }
    if (!is_float_compute_dtype(xView.view.desc.dtype))
        return make_error("cuda moe_matmul unsupported dtype");
    if (offsetView.view.desc.dtype != core::DType::I32 &&
        offsetView.view.desc.dtype != core::DType::I64) {
        return make_error("cuda moe_matmul expert_offsets must be i32 or i64");
    }
    if (!cuda_kernel::is_contiguous(xView.view) ||
        !cuda_kernel::is_contiguous(offsetView.view) ||
        !cuda_kernel::is_contiguous(weightView.view) ||
        !cuda_kernel::is_contiguous(outputView.view)) {
        return make_error("cuda moe_matmul requires contiguous tensor views");
    }

    const auto& xShape = xView.view.desc.shape;
    const auto& offsetShape = offsetView.view.desc.shape;
    const auto& weightShape = weightView.view.desc.shape;
    const auto& outputShape = outputView.view.desc.shape;
    int xRank = xShape.rank();
    if ((xRank != 2 && xRank != 3) || offsetShape.rank() != xRank - 1 ||
        weightShape.rank() != 3 || outputShape.rank() != xRank) {
        return make_error(
            "cuda moe_matmul expects ranks [N,K], [E+1], [E,M,K] or [B,N,K], [B,E+1], [E,M,K]");
    }

    int64_t batch = xRank == 3 ? xShape.dim(0) : 1;
    int64_t rows = xShape.dim(xRank - 2);
    int64_t inFeatures = xShape.dim(xRank - 1);
    int64_t experts = weightShape.dim(0);
    int64_t outFeatures = program.transposeRhs ? weightShape.dim(1) : weightShape.dim(2);
    int64_t rhsK = program.transposeRhs ? weightShape.dim(2) : weightShape.dim(1);
    if (batch <= 0 || rows < 0 || inFeatures <= 0 || experts <= 0 || outFeatures <= 0)
        return make_error("cuda moe_matmul expects static positive dimensions");
    if (rhsK != inFeatures)
        return make_error("cuda moe_matmul contracting dimension mismatch");
    if (offsetShape.dim(xRank - 2) != experts + 1)
        return make_error("cuda moe_matmul expert_offsets shape mismatch");
    if (xRank == 3 && offsetShape.dim(0) != batch)
        return make_error("cuda moe_matmul batch dimension mismatch");
    if ((xRank == 2 && (outputShape.dim(0) != rows || outputShape.dim(1) != outFeatures)) ||
        (xRank == 3 && (outputShape.dim(0) != batch ||
                        outputShape.dim(1) != rows ||
                        outputShape.dim(2) != outFeatures))) {
        return make_error("cuda moe_matmul output shape mismatch");
    }

    int64_t maxInt = std::numeric_limits<int>::max();
    if (batch > maxInt || rows > maxInt || inFeatures > maxInt ||
        outFeatures > maxInt || experts + 1 > maxInt) {
        return make_error("cuda moe_matmul dimensions exceed cuBLAS int limits");
    }

    if (rows == 0)
        return {};
    if (!context.cublas)
        return make_error("cuda moe_matmul missing cuBLAS handle");

    auto dataType = cublas_data_type_for(xView.view.desc.dtype);
    if (!dataType)
        return make_error(dataType.error());
    cudaDataType_t cudaType = dataType.take();

    auto xArg = cuda_kernel::pack_tensor_arg(xView);
    if (!xArg)
        return make_error(xArg.error());
    auto offsetsArg = cuda_kernel::pack_tensor_arg(offsetView);
    if (!offsetsArg)
        return make_error(offsetsArg.error());
    auto weightArg = cuda_kernel::pack_tensor_arg(weightView);
    if (!weightArg)
        return make_error(weightArg.error());
    auto outputArg = cuda_kernel::pack_tensor_arg(outputView);
    if (!outputArg)
        return make_error(outputArg.error());

    auto x = xArg.take();
    auto offsets = offsetsArg.take();
    auto weight = weightArg.take();
    auto output = outputArg.take();

    int64_t offsetCount = batch * (experts + 1);
    std::vector<uint8_t> offsetBytes(
        static_cast<size_t>(offsetCount) * core::dtype_size(offsets.dtype));
    auto copied = cuda_check(
        cudaMemcpyAsync(
            offsetBytes.data(),
            byte_offset(offsets.data, offsets.storageOffset, offsets.dtype),
            offsetBytes.size(),
            cudaMemcpyDeviceToHost,
            context.stream),
        "cudaMemcpyAsync moe_matmul expert_offsets");
    if (!copied)
        return make_error(copied.error());
    auto synced = cuda_check(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize");
    if (!synced)
        return make_error(synced.error());

    std::vector<int64_t> hostOffsets(static_cast<size_t>(offsetCount));
    if (offsets.dtype == core::DType::I32) {
        auto* values = reinterpret_cast<const int32_t*>(offsetBytes.data());
        for (int64_t i = 0; i < offsetCount; ++i)
            hostOffsets[static_cast<size_t>(i)] = values[i];
    } else {
        auto* values = reinterpret_cast<const int64_t*>(offsetBytes.data());
        for (int64_t i = 0; i < offsetCount; ++i)
            hostOffsets[static_cast<size_t>(i)] = values[i];
    }

    std::vector<cublasOperation_t> transA;
    std::vector<cublasOperation_t> transB;
    std::vector<int> mArray;
    std::vector<int> nArray;
    std::vector<int> kArray;
    std::vector<float> alphaArray;
    std::vector<float> betaArray;
    std::vector<const void*> aArray;
    std::vector<const void*> bArray;
    std::vector<void*> cArray;
    std::vector<int> ldaArray;
    std::vector<int> ldbArray;
    std::vector<int> ldcArray;
    std::vector<int> groupSize;

    int64_t weightSlice = weight.dims[1] * weight.dims[2];
    for (int64_t b = 0; b < batch; ++b) {
        int64_t offsetBase = b * (experts + 1);
        if (hostOffsets[static_cast<size_t>(offsetBase)] != 0 ||
            hostOffsets[static_cast<size_t>(offsetBase + experts)] != rows) {
            return make_error("cuda moe_matmul expert_offsets must cover all rows per batch");
        }
        int64_t xBatchBase = b * rows * inFeatures;
        int64_t outputBatchBase = b * rows * outFeatures;
        for (int64_t expert = 0; expert < experts; ++expert) {
            int64_t begin = hostOffsets[static_cast<size_t>(offsetBase + expert)];
            int64_t end = hostOffsets[static_cast<size_t>(offsetBase + expert + 1)];
            if (begin < 0 || end < begin || end > rows)
                return make_error("cuda moe_matmul invalid expert offsets");
            int64_t expertRows = end - begin;
            if (expertRows == 0)
                continue;
            if (expertRows > maxInt)
                return make_error("cuda moe_matmul expert row count exceeds cuBLAS int limit");

            transA.push_back(program.transposeRhs ? CUBLAS_OP_T : CUBLAS_OP_N);
            transB.push_back(CUBLAS_OP_N);
            mArray.push_back(static_cast<int>(outFeatures));
            nArray.push_back(static_cast<int>(expertRows));
            kArray.push_back(static_cast<int>(inFeatures));
            alphaArray.push_back(1.0f);
            betaArray.push_back(0.0f);
            aArray.push_back(byte_offset(
                weight.data,
                weight.storageOffset + expert * weightSlice,
                weight.dtype));
            bArray.push_back(byte_offset(
                x.data,
                x.storageOffset + xBatchBase + begin * inFeatures,
                x.dtype));
            cArray.push_back(byte_offset(
                output.data,
                output.storageOffset + outputBatchBase + begin * outFeatures,
                output.dtype));
            ldaArray.push_back(static_cast<int>(weight.dims[2]));
            ldbArray.push_back(static_cast<int>(inFeatures));
            ldcArray.push_back(static_cast<int>(outFeatures));
            groupSize.push_back(1);
        }
    }

    if (aArray.empty())
        return {};

    void** deviceAArray = nullptr;
    void** deviceBArray = nullptr;
    void** deviceCArray = nullptr;
    size_t pointerBytes = aArray.size() * sizeof(void*);

    auto allocA = cuda_check(cudaMalloc(&deviceAArray, pointerBytes), "cudaMalloc moe_matmul Aarray");
    if (!allocA)
        return make_error(allocA.error());
    auto allocB = cuda_check(cudaMalloc(&deviceBArray, pointerBytes), "cudaMalloc moe_matmul Barray");
    if (!allocB) {
        cudaFree(deviceAArray);
        return make_error(allocB.error());
    }
    auto allocC = cuda_check(cudaMalloc(&deviceCArray, pointerBytes), "cudaMalloc moe_matmul Carray");
    if (!allocC) {
        cudaFree(deviceAArray);
        cudaFree(deviceBArray);
        return make_error(allocC.error());
    }

    auto cleanup = [&]() -> Result<void> {
        auto freeA = cuda_check(cudaFree(deviceAArray), "cudaFree moe_matmul Aarray");
        auto freeB = cuda_check(cudaFree(deviceBArray), "cudaFree moe_matmul Barray");
        auto freeC = cuda_check(cudaFree(deviceCArray), "cudaFree moe_matmul Carray");
        if (!freeA) return make_error(freeA.error());
        if (!freeB) return make_error(freeB.error());
        if (!freeC) return make_error(freeC.error());
        return {};
    };

    auto copyA = cuda_check(
        cudaMemcpyAsync(
            deviceAArray,
            aArray.data(),
            pointerBytes,
            cudaMemcpyHostToDevice,
            context.stream),
        "cudaMemcpyAsync moe_matmul Aarray");
    if (!copyA) {
        auto cleaned = cleanup();
        return make_error(cleaned ? copyA.error() : copyA.error() + "; " + cleaned.error());
    }
    auto copyB = cuda_check(
        cudaMemcpyAsync(
            deviceBArray,
            bArray.data(),
            pointerBytes,
            cudaMemcpyHostToDevice,
            context.stream),
        "cudaMemcpyAsync moe_matmul Barray");
    if (!copyB) {
        auto cleaned = cleanup();
        return make_error(cleaned ? copyB.error() : copyB.error() + "; " + cleaned.error());
    }
    auto copyC = cuda_check(
        cudaMemcpyAsync(
            deviceCArray,
            cArray.data(),
            pointerBytes,
            cudaMemcpyHostToDevice,
            context.stream),
        "cudaMemcpyAsync moe_matmul Carray");
    if (!copyC) {
        auto cleaned = cleanup();
        return make_error(cleaned ? copyC.error() : copyC.error() + "; " + cleaned.error());
    }

    auto status = cublasGemmGroupedBatchedEx(
        context.cublas,
        transA.data(),
        transB.data(),
        mArray.data(),
        nArray.data(),
        kArray.data(),
        alphaArray.data(),
        reinterpret_cast<const void* const*>(deviceAArray),
        cudaType,
        ldaArray.data(),
        reinterpret_cast<const void* const*>(deviceBArray),
        cudaType,
        ldbArray.data(),
        betaArray.data(),
        reinterpret_cast<void* const*>(deviceCArray),
        cudaType,
        ldcArray.data(),
        static_cast<int>(aArray.size()),
        groupSize.data(),
        CUBLAS_COMPUTE_32F);
    auto gemm = cublas_check(status, "cublasGemmGroupedBatchedEx");
    auto cleaned = cleanup();
    if (!gemm)
        return make_error(cleaned ? gemm.error() : gemm.error() + "; " + cleaned.error());
    if (!cleaned)
        return make_error(cleaned.error());
    return {};
#endif
}

} // namespace sandy::device
