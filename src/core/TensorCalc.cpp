#include "TensorCalc.h"
#include "ShapeUtil.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(SANDY_MATMUL_FAST) && defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#if defined(SANDY_MATMUL_CUBLAS)
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace sandy::core {

namespace {

std::vector<int64_t> strides_for(const Shape& shape);

float load_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

void store_f32(std::span<uint8_t> data, size_t index, float value) {
    std::memcpy(data.data() + index * sizeof(float), &value, sizeof(float));
}

float load_bf16(std::span<const uint8_t> data, size_t index) {
    BFloat16 value = bfloat16_from_bits(0);
    std::memcpy(&value, data.data() + index * sizeof(BFloat16), sizeof(BFloat16));
    return bfloat16_to_float(value);
}

void store_bf16(std::span<uint8_t> data, size_t index, float value) {
    BFloat16 storage = bfloat16_from_float(value);
    std::memcpy(data.data() + index * sizeof(BFloat16), &storage, sizeof(BFloat16));
}

float unsupported_float_load(std::span<const uint8_t>, size_t) {
    return 0.0f;
}

void unsupported_float_store(std::span<uint8_t>, size_t, float) {}

bool is_float_compute_dtype(DType dtype) {
    return dtype == DType::F32 || dtype == DType::BF16;
}

TensorRef::LoadFloatFn float_loader_for(DType dtype) {
    switch (dtype) {
        case DType::F32: return load_f32;
        case DType::BF16: return load_bf16;
        default: return unsupported_float_load;
    }
}

MutableTensorRef::StoreFloatFn float_storer_for(DType dtype) {
    switch (dtype) {
        case DType::F32: return store_f32;
        case DType::BF16: return store_bf16;
        default: return unsupported_float_store;
    }
}

Result<void> require_bytes(
        std::span<const uint8_t> data,
        const TensorDesc& desc,
        std::span<const int64_t> strides,
        int64_t storageOffset,
        const std::string& name) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error(name + " must have static shape");

    if (static_cast<int>(strides.size()) != desc.shape.rank())
        return make_error(name + " strides rank mismatch");
    if (storageOffset < 0)
        return make_error(name + " storage offset must be non-negative");

    if (numel == 0)
        return {};

    int64_t maxStorageIndex = storageOffset;
    for (int i = 0; i < desc.shape.rank(); i++) {
        if (strides[static_cast<size_t>(i)] < 0)
            return make_error(name + " strides must be non-negative");
        if (desc.shape.dim(i) > 0) {
            maxStorageIndex += (desc.shape.dim(i) - 1) *
                strides[static_cast<size_t>(i)];
        }
    }

    auto expected = static_cast<size_t>(maxStorageIndex + 1) * dtype_size(desc.dtype);
    if (data.size() < expected)
        return make_error(name + " byte size mismatch");

    return {};
}

Result<void> require_bytes(
        std::span<const uint8_t> data,
        const TensorDesc& desc,
        const std::string& name) {
    return require_bytes(data, desc, strides_for(desc.shape), 0, name);
}

Result<void> require_float_tensor(const TensorRef& ref, const std::string& name) {
    if (!is_float_compute_dtype(ref.desc.dtype))
        return make_error(name + " unsupported dtype");
    return {};
}

Result<void> require_float_tensor(const MutableTensorRef& ref, const std::string& name) {
    if (!is_float_compute_dtype(ref.desc.dtype))
        return make_error(name + " unsupported dtype");
    return {};
}

Result<void> require_same_dtype(DType lhs, DType rhs, const std::string& opName) {
    if (lhs != rhs)
        return make_error(opName + " operands must have same dtype");
    return {};
}

Result<void> require_output(
        const MutableTensorRef& out,
        const Shape& shape,
        DType dtype,
        const std::string& opName) {
    if (out.desc.shape != shape)
        return make_error(opName + " output shape mismatch");
    if (out.desc.dtype != dtype)
        return make_error(opName + " output dtype mismatch");
    return require_float_tensor(out, opName + " output");
}

int64_t numel_or_error(const Shape& shape, const std::string& name) {
    int64_t numel = shape.numel();
    if (numel < 0)
        return -1;
    return numel;
}

int64_t read_index(const TensorRef& ids, size_t index) {
    size_t storageIndex = ids.storage_index(index);
    if (ids.desc.dtype == DType::I32) {
        int32_t value = 0;
        std::memcpy(&value, ids.bytes.data() + storageIndex * sizeof(int32_t), sizeof(int32_t));
        return value;
    }

    int64_t value = 0;
    std::memcpy(&value, ids.bytes.data() + storageIndex * sizeof(int64_t), sizeof(int64_t));
    return value;
}

std::vector<int64_t> strides_for(const Shape& shape) {
    std::vector<int64_t> strides(static_cast<size_t>(shape.rank()), 1);
    int64_t stride = 1;
    for (int i = shape.rank() - 1; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= shape.dim(i);
    }
    return strides;
}

size_t broadcast_source_index(
        size_t outIndex,
        const Shape& outShape,
        const Shape& sourceShape,
        const std::vector<int64_t>& sourceStrides) {
    if (sourceShape.rank() == 0) return 0;

    size_t sourceIndex = 0;
    size_t remaining = outIndex;
    int rankOffset = outShape.rank() - sourceShape.rank();
    for (int outDimIndex = outShape.rank() - 1; outDimIndex >= 0; outDimIndex--) {
        int64_t outDim = outShape.dim(outDimIndex);
        size_t coord = remaining % static_cast<size_t>(outDim);
        remaining /= static_cast<size_t>(outDim);

        int sourceDimIndex = outDimIndex - rankOffset;
        if (sourceDimIndex < 0) continue;
        if (sourceShape.dim(sourceDimIndex) != 1) {
            sourceIndex += coord *
                static_cast<size_t>(sourceStrides[static_cast<size_t>(sourceDimIndex)]);
        }
    }
    return sourceIndex;
}

size_t matmul_batch_offset(
        size_t batchIndex,
        const Shape& batchShape,
        const Shape& sourceShape,
        const std::vector<int64_t>& sourceStrides) {
    int sourceBatchRank = sourceShape.rank() - 2;
    if (sourceBatchRank <= 0) return 0;

    size_t sourceOffset = 0;
    size_t remaining = batchIndex;
    int rankOffset = batchShape.rank() - sourceBatchRank;
    for (int batchDimIndex = batchShape.rank() - 1; batchDimIndex >= 0; batchDimIndex--) {
        int64_t batchDim = batchShape.dim(batchDimIndex);
        size_t coord = remaining % static_cast<size_t>(batchDim);
        remaining /= static_cast<size_t>(batchDim);

        int sourceDimIndex = batchDimIndex - rankOffset;
        if (sourceDimIndex < 0) continue;

        int64_t sourceDim = sourceShape.dim(sourceDimIndex);
        if (sourceDim == 1) continue;
        if (sourceDim != batchDim)
            coord /= static_cast<size_t>(batchDim / sourceDim);
        sourceOffset += coord *
            static_cast<size_t>(sourceStrides[static_cast<size_t>(sourceDimIndex)]);
    }
    return sourceOffset;
}

#if defined(SANDY_MATMUL_CUBLAS)
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

Result<void> cuda_check(cudaError_t status, const std::string& context) {
    if (status == cudaSuccess)
        return {};
    return make_error(context + ": " + cudaGetErrorString(status));
}

Result<void> cublas_check(cublasStatus_t status, const std::string& context) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return {};
    return make_error(context + ": " + cublas_status_name(status));
}

class CudaAllocation {
public:
    CudaAllocation() = default;
    CudaAllocation(const CudaAllocation&) = delete;
    CudaAllocation& operator=(const CudaAllocation&) = delete;
    ~CudaAllocation() {
        if (ptr_)
            cudaFree(ptr_);
    }

    Result<void> alloc(size_t bytes) {
        if (bytes == 0)
            return {};
        return cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }

    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() = default;
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    ~CublasHandle() {
        if (handle_)
            cublasDestroy(handle_);
    }

    Result<void> create() {
        return cublas_check(cublasCreate(&handle_), "cublasCreate");
    }

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

bool cublas_runtime_available() {
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return deviceCount > 0;
}

bool cublas_int_dims_ok(int64_t rows, int64_t n, int64_t k, int64_t lhsLd, int64_t rhsLd) {
    int64_t maxInt = std::numeric_limits<int>::max();
    return rows <= maxInt && n <= maxInt && k <= maxInt &&
           lhsLd <= maxInt && rhsLd <= maxInt;
}

Result<cudaDataType_t> cublas_data_type_for(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return CUDA_R_32F;
        case DType::BF16:
            return CUDA_R_16BF;
        default:
            return make_error("cuBLAS matmul unsupported dtype");
    }
}

bool cublas_can_fallback(cublasStatus_t status) {
    return status == CUBLAS_STATUS_NOT_SUPPORTED ||
           status == CUBLAS_STATUS_ARCH_MISMATCH;
}

void log_cublas_skip_once(const char* reason) {
    static bool logged = false;
    if (logged) return;
    logged = true;
    std::fprintf(stderr, "[sandy] cuBLAS matmul skipped: %s\n", reason);
}

void log_cublas_use_once(
        DType dtype,
        int64_t rows,
        int64_t k,
        int64_t n,
        int64_t batchNumel) {
    static bool logged = false;
    if (logged) return;
    logged = true;
    std::fprintf(
        stderr,
        "[sandy] cuBLAS matmul active: dtype=%s rows=%ld k=%ld n=%ld batch=%ld\n",
        dtype_name(dtype),
        static_cast<long>(rows),
        static_cast<long>(k),
        static_cast<long>(n),
        static_cast<long>(batchNumel));
}

Result<bool> matmul_cublas(
        TensorRef lhs,
        TensorRef rhs,
        bool transpose_lhs,
        bool transpose_rhs,
        int64_t m,
        int64_t k,
        int64_t n,
        int64_t batchNumel,
        MutableTensorRef out) {
    if (lhs.desc.dtype != rhs.desc.dtype || lhs.desc.dtype != out.desc.dtype) {
        log_cublas_skip_once("dtype mismatch");
        return false;
    }
    if (!lhs.is_contiguous() || !rhs.is_contiguous() || !out.is_contiguous()) {
        log_cublas_skip_once("non-contiguous tensor");
        return false;
    }
    auto dataType = cublas_data_type_for(lhs.desc.dtype);
    if (!dataType) {
        log_cublas_skip_once("unsupported dtype");
        return false;
    }
    cudaDataType_t cudaType = dataType.take();
    if (!cublas_runtime_available()) {
        log_cublas_skip_once("no CUDA device available");
        return false;
    }

    int lhsRank = lhs.desc.shape.rank();
    int rhsRank = rhs.desc.shape.rank();
    bool flattenSharedRhs = !transpose_lhs && rhsRank == 2;
    if (!flattenSharedRhs && batchNumel != 1) {
        log_cublas_skip_once("batched matmul with non-shared rhs");
        return false;
    }

    int64_t rows = flattenSharedRhs ? batchNumel * m : m;
    int64_t lhsLd = lhs.desc.shape.dim(lhsRank - 1);
    int64_t rhsLd = rhs.desc.shape.dim(rhsRank - 1);
    if (!cublas_int_dims_ok(rows, n, k, lhsLd, rhsLd)) {
        log_cublas_skip_once("matrix dimensions exceed cuBLAS int limits");
        return false;
    }

    CudaAllocation lhsDevice;
    CudaAllocation rhsDevice;
    CudaAllocation outDevice;
    auto lhsAlloc = lhsDevice.alloc(lhs.bytes.size());
    if (!lhsAlloc) return make_error(lhsAlloc.error());
    auto rhsAlloc = rhsDevice.alloc(rhs.bytes.size());
    if (!rhsAlloc) return make_error(rhsAlloc.error());
    auto outAlloc = outDevice.alloc(out.bytes.size());
    if (!outAlloc) return make_error(outAlloc.error());

    auto lhsCopy = cuda_check(
        cudaMemcpy(lhsDevice.get(), lhs.bytes.data(), lhs.bytes.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy lhs host-to-device");
    if (!lhsCopy) return make_error(lhsCopy.error());
    auto rhsCopy = cuda_check(
        cudaMemcpy(rhsDevice.get(), rhs.bytes.data(), rhs.bytes.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy rhs host-to-device");
    if (!rhsCopy) return make_error(rhsCopy.error());

    CublasHandle handle;
    auto handleCreate = handle.create();
    if (!handleCreate) return make_error(handleCreate.error());

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasOperation_t lhsOp = transpose_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t rhsOp = transpose_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

    auto gemmStatus = cublasGemmEx(
        handle.get(),
        rhsOp,
        lhsOp,
        static_cast<int>(n),
        static_cast<int>(rows),
        static_cast<int>(k),
        &alpha,
        rhsDevice.get(),
        cudaType,
        static_cast<int>(rhsLd),
        lhsDevice.get(),
        cudaType,
        static_cast<int>(lhsLd),
        &beta,
        outDevice.get(),
        cudaType,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (cublas_can_fallback(gemmStatus)) {
        log_cublas_skip_once(cublas_status_name(gemmStatus));
        return false;
    }
    auto gemm = cublas_check(gemmStatus, "cublasGemmEx");
    if (!gemm)
        return make_error(gemm.error());
    log_cublas_use_once(lhs.desc.dtype, rows, k, n, batchNumel);

    auto outCopy = cuda_check(
        cudaMemcpy(out.bytes.data(), outDevice.get(), out.bytes.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy output device-to-host");
    if (!outCopy) return make_error(outCopy.error());

    return true;
}
#endif

#if defined(SANDY_MATMUL_FAST) && defined(__APPLE__)
bool can_use_fast_matmul(
        const TensorRef& lhs,
        const TensorRef& rhs,
        bool transpose_lhs,
        int64_t batchNumel) {
    if (transpose_lhs)
        return false;
    if (rhs.desc.shape.rank() != 2)
        return false;
    if (batchNumel != 1)
        return false;
    return lhs.desc.dtype == DType::F32 || lhs.desc.dtype == DType::BF16;
}

void copy_tensor_to_f32(TensorRef src, std::vector<float>& dst) {
    auto numel = src.desc.shape.numel();
    dst.resize(static_cast<size_t>(numel));
    if (src.desc.dtype == DType::F32 && src.is_contiguous()) {
        std::memcpy(dst.data(), src.bytes.data(), dst.size() * sizeof(float));
        return;
    }
    for (size_t i = 0; i < dst.size(); i++)
        dst[i] = src.load_float(i);
}

void store_f32_to_output(std::span<const float> src, MutableTensorRef out) {
    if (out.desc.dtype == DType::F32 && out.is_contiguous()) {
        std::memcpy(out.bytes.data(), src.data(), src.size() * sizeof(float));
        return;
    }
    for (size_t i = 0; i < src.size(); i++)
        out.store_float(i, src[i]);
}

Result<bool> matmul_fast(
        TensorRef lhs,
        TensorRef rhs,
        bool transpose_lhs,
        bool transpose_rhs,
        int64_t rows,
        int64_t k,
        int64_t n,
        int64_t batchNumel,
        MutableTensorRef out) {
    if (!can_use_fast_matmul(lhs, rhs, transpose_lhs, batchNumel))
        return false;

    std::vector<float> lhsF32;
    std::vector<float> rhsF32;
    std::vector<float> outF32(static_cast<size_t>(rows * n));
    copy_tensor_to_f32(lhs, lhsF32);
    copy_tensor_to_f32(rhs, rhsF32);

    CBLAS_TRANSPOSE rhsTranspose = transpose_rhs ? CblasTrans : CblasNoTrans;
    int rhsLeadingDim = static_cast<int>(transpose_rhs ? k : n);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        rhsTranspose,
        static_cast<int>(rows),
        static_cast<int>(n),
        static_cast<int>(k),
        1.0f,
        lhsF32.data(),
        static_cast<int>(k),
        rhsF32.data(),
        rhsLeadingDim,
        0.0f,
        outF32.data(),
        static_cast<int>(n));

    store_f32_to_output(outF32, out);
    return true;
}
#endif

using UnaryOp = float (*)(float);
using BinaryOp = float (*)(float, float);

Result<void> unary_elementwise(
        TensorRef x,
        MutableTensorRef out,
        const std::string& opName,
        UnaryOp op) {
    auto inputFloat = require_float_tensor(x, opName + " input");
    if (!inputFloat) return make_error(inputFloat.error());
    auto output = require_output(out, x.desc.shape, x.desc.dtype, opName);
    if (!output) return make_error(output.error());

    int64_t count = numel_or_error(x.desc.shape, opName + " input");
    if (count < 0)
        return make_error(opName + " input must have static shape");

    for (size_t i = 0; i < static_cast<size_t>(count); i++)
        out.store_float(i, op(x.load_float(i)));

    return {};
}

Result<void> binary_elementwise(
        TensorRef lhs,
        TensorRef rhs,
        MutableTensorRef out,
        const std::string& opName,
        BinaryOp op) {
    auto lhsFloat = require_float_tensor(lhs, opName + " lhs");
    if (!lhsFloat) return make_error(lhsFloat.error());
    auto rhsFloat = require_float_tensor(rhs, opName + " rhs");
    if (!rhsFloat) return make_error(rhsFloat.error());
    bool lhsScalar = lhs.desc.shape.rank() == 0;
    bool rhsScalar = rhs.desc.shape.rank() == 0;
    if (lhs.desc.dtype != rhs.desc.dtype && !lhsScalar && !rhsScalar)
        return make_error(opName + " operands must have same dtype");

    auto shapeResult = broadcast_shape(lhs.desc.shape, rhs.desc.shape);
    if (!shapeResult) return make_error(shapeResult.error());
    auto outShape = shapeResult.take();
    auto outDtype = lhs.desc.dtype == rhs.desc.dtype
        ? lhs.desc.dtype
        : (lhsScalar ? rhs.desc.dtype : lhs.desc.dtype);
    auto output = require_output(out, outShape, outDtype, opName);
    if (!output) return make_error(output.error());

    int64_t outNumel = numel_or_error(outShape, opName + " output");
    if (outNumel < 0)
        return make_error(opName + " output must have static shape");

    auto lhsStrides = strides_for(lhs.desc.shape);
    auto rhsStrides = strides_for(rhs.desc.shape);
    for (size_t i = 0; i < static_cast<size_t>(outNumel); i++) {
        size_t lhsIndex = broadcast_source_index(i, out.desc.shape, lhs.desc.shape, lhsStrides);
        size_t rhsIndex = broadcast_source_index(i, out.desc.shape, rhs.desc.shape, rhsStrides);
        out.store_float(i, op(lhs.load_float(lhsIndex), rhs.load_float(rhsIndex)));
    }

    return {};
}

} // namespace

size_t TensorRef::storage_index(size_t index) const {
    if (desc.shape.rank() == 0)
        return static_cast<size_t>(storageOffset);

    size_t storage = static_cast<size_t>(storageOffset);
    size_t remaining = index;
    auto logicalStrides = strides_for(desc.shape);
    for (int axis = 0; axis < desc.shape.rank(); axis++) {
        auto logicalStride = static_cast<size_t>(logicalStrides[static_cast<size_t>(axis)]);
        size_t coord = logicalStride == 0 ? 0 : remaining / logicalStride;
        if (logicalStride != 0)
            remaining %= logicalStride;
        storage += coord * static_cast<size_t>(strides[static_cast<size_t>(axis)]);
    }
    return storage;
}

bool TensorRef::is_contiguous() const {
    return storageOffset == 0 && strides == strides_for(desc.shape);
}

size_t MutableTensorRef::storage_index(size_t index) const {
    if (desc.shape.rank() == 0)
        return static_cast<size_t>(storageOffset);

    size_t storage = static_cast<size_t>(storageOffset);
    size_t remaining = index;
    auto logicalStrides = strides_for(desc.shape);
    for (int axis = 0; axis < desc.shape.rank(); axis++) {
        auto logicalStride = static_cast<size_t>(logicalStrides[static_cast<size_t>(axis)]);
        size_t coord = logicalStride == 0 ? 0 : remaining / logicalStride;
        if (logicalStride != 0)
            remaining %= logicalStride;
        storage += coord * static_cast<size_t>(strides[static_cast<size_t>(axis)]);
    }
    return storage;
}

bool MutableTensorRef::is_contiguous() const {
    return storageOffset == 0 && strides == strides_for(desc.shape);
}

Result<TensorRef> make_tensor_ref(TensorDesc desc, std::span<const uint8_t> bytes) {
    auto strides = strides_for(desc.shape);
    auto byteCheck = require_bytes(bytes, desc, strides, 0, "tensor ref");
    if (!byteCheck) return make_error(byteCheck.error());
    auto dtype = desc.dtype;
    return TensorRef{std::move(desc), bytes, std::move(strides), 0, float_loader_for(dtype)};
}

Result<TensorRef> make_tensor_ref(
        TensorDesc desc,
        std::span<const uint8_t> bytes,
        std::span<const int64_t> strides,
        int64_t storageOffset) {
    auto byteCheck = require_bytes(bytes, desc, strides, storageOffset, "tensor ref");
    if (!byteCheck) return make_error(byteCheck.error());
    auto dtype = desc.dtype;
    return TensorRef{
        std::move(desc),
        bytes,
        std::vector<int64_t>(strides.begin(), strides.end()),
        storageOffset,
        float_loader_for(dtype)};
}

Result<MutableTensorRef> make_mutable_tensor_ref(
        TensorDesc desc,
        std::span<uint8_t> bytes) {
    auto strides = strides_for(desc.shape);
    auto byteCheck = require_bytes(bytes, desc, strides, 0, "mutable tensor ref");
    if (!byteCheck) return make_error(byteCheck.error());
    auto dtype = desc.dtype;
    return MutableTensorRef{
        std::move(desc),
        bytes,
        std::move(strides),
        0,
        float_loader_for(dtype),
        float_storer_for(dtype)};
}

Result<MutableTensorRef> make_mutable_tensor_ref(
        TensorDesc desc,
        std::span<uint8_t> bytes,
        std::span<const int64_t> strides,
        int64_t storageOffset) {
    auto byteCheck = require_bytes(bytes, desc, strides, storageOffset, "mutable tensor ref");
    if (!byteCheck) return make_error(byteCheck.error());
    auto dtype = desc.dtype;
    return MutableTensorRef{
        std::move(desc),
        bytes,
        std::vector<int64_t>(strides.begin(), strides.end()),
        storageOffset,
        float_loader_for(dtype),
        float_storer_for(dtype)};
}

Result<void> linear(
        TensorRef x,
        TensorRef weight,
        TensorRef bias,
        MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "linear input");
    if (!xFloat) return make_error(xFloat.error());
    auto weightFloat = require_float_tensor(weight, "linear weight");
    if (!weightFloat) return make_error(weightFloat.error());
    auto biasFloat = require_float_tensor(bias, "linear bias");
    if (!biasFloat) return make_error(biasFloat.error());
    if (x.desc.dtype != weight.desc.dtype || x.desc.dtype != bias.desc.dtype)
        return make_error("linear operands must have same dtype");

    if (x.desc.shape.rank() < 2)
        return make_error("linear input must have rank >= 2");
    if (weight.desc.shape.rank() != 2)
        return make_error("linear weight must have rank 2");
    if (bias.desc.shape.rank() != 1)
        return make_error("linear bias must have rank 1");

    int xRank = x.desc.shape.rank();
    int64_t inFeatures = x.desc.shape.dim(xRank - 1);
    int64_t outFeatures = weight.desc.shape.dim(0);
    if (inFeatures < 0 || outFeatures < 0)
        return make_error("linear inputs must have static shape");
    if (weight.desc.shape.dim(1) != inFeatures)
        return make_error("linear weight input dimension mismatch");
    if (bias.desc.shape.dim(0) != outFeatures)
        return make_error("linear bias dimension mismatch");

    auto outDims = x.desc.shape.dims();
    outDims.back() = outFeatures;
    Shape outShape(outDims);
    auto output = require_output(out, outShape, x.desc.dtype, "linear");
    if (!output) return make_error(output.error());

    int64_t rows = x.desc.shape.numel() / inFeatures;
    for (int64_t b = 0; b < rows; b++) {
        for (int64_t o = 0; o < outFeatures; o++) {
            float acc = bias.load_float(static_cast<size_t>(o));
            for (int64_t i = 0; i < inFeatures; i++) {
                float xv = x.load_float(static_cast<size_t>(b * inFeatures + i));
                float wv = weight.load_float(static_cast<size_t>(o * inFeatures + i));
                acc += xv * wv;
            }
            out.store_float(static_cast<size_t>(b * outFeatures + o), acc);
        }
    }

    return {};
}

Result<void> relu(TensorRef x, MutableTensorRef out) {
    auto op = [](float v) { return std::max(0.0f, v); };
    return unary_elementwise(x, out, "relu", op);
}

Result<void> add(TensorRef lhs, TensorRef rhs, MutableTensorRef out) {
    return binary_elementwise(lhs, rhs, out, "add", [](float a, float b) { return a + b; });
}

Result<void> mul(TensorRef lhs, TensorRef rhs, MutableTensorRef out) {
    return binary_elementwise(lhs, rhs, out, "mul", [](float a, float b) { return a * b; });
}

Result<void> sqrt(TensorRef x, MutableTensorRef out) {
    return unary_elementwise(x, out, "sqrt", [](float v) { return std::sqrt(v); });
}

Result<void> tanh(TensorRef x, MutableTensorRef out) {
    return unary_elementwise(x, out, "tanh", [](float v) { return std::tanh(v); });
}

Result<void> matmul(TensorRef lhs, TensorRef rhs, MutableTensorRef out) {
    return matmul(lhs, rhs, false, false, out);
}

Result<void> matmul(
        TensorRef lhs,
        TensorRef rhs,
        bool transpose_lhs,
        bool transpose_rhs,
        MutableTensorRef out) {
    auto lhsFloat = require_float_tensor(lhs, "matmul lhs");
    if (!lhsFloat) return make_error(lhsFloat.error());
    auto rhsFloat = require_float_tensor(rhs, "matmul rhs");
    if (!rhsFloat) return make_error(rhsFloat.error());
    auto same = require_same_dtype(lhs.desc.dtype, rhs.desc.dtype, "matmul");
    if (!same) return make_error(same.error());

    if (lhs.desc.shape.rank() < 2)
        return make_error("matmul lhs must have rank >= 2");
    if (rhs.desc.shape.rank() < 2)
        return make_error("matmul rhs must have rank >= 2");

    int lhsRank = lhs.desc.shape.rank();
    int rhsRank = rhs.desc.shape.rank();
    int64_t m = lhs.desc.shape.dim(lhsRank - (transpose_lhs ? 1 : 2));
    int64_t lhsK = lhs.desc.shape.dim(lhsRank - (transpose_lhs ? 2 : 1));
    int64_t rhsK = rhs.desc.shape.dim(rhsRank - (transpose_rhs ? 1 : 2));
    int64_t n = rhs.desc.shape.dim(rhsRank - (transpose_rhs ? 2 : 1));
    if (m < 0 || n < 0 || lhsK < 0 || rhsK < 0)
        return make_error("matmul matrix dimensions must be static");
    if (lhsK != rhsK)
        return make_error("matmul contracting dimension mismatch");

    auto lhsDims = lhs.desc.shape.dims();
    auto rhsDims = rhs.desc.shape.dims();
    Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
    Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
    auto batchShapeResult = matmul_batch_shape(lhsBatch, rhsBatch);
    if (!batchShapeResult) return make_error(batchShapeResult.error());
    auto batchShape = batchShapeResult.take();

    auto outDims = batchShape.dims();
    outDims.push_back(m);
    outDims.push_back(n);
    Shape outShape(outDims);
    auto output = require_output(out, outShape, lhs.desc.dtype, "matmul");
    if (!output) return make_error(output.error());

    int64_t batchNumel = batchShape.numel();
    if (batchNumel < 0)
        return make_error("matmul output must have static shape");

#if defined(SANDY_MATMUL_CUBLAS)
    auto cublas = matmul_cublas(
        lhs,
        rhs,
        transpose_lhs,
        transpose_rhs,
        m,
        lhsK,
        n,
        batchNumel,
        out);
    if (!cublas)
        return make_error(cublas.error());
    if (cublas.take())
        return {};
#endif

#if defined(SANDY_MATMUL_FAST) && defined(__APPLE__)
    int64_t rows = batchNumel * m;
    auto fast = matmul_fast(
        lhs,
        rhs,
        transpose_lhs,
        transpose_rhs,
        rows,
        lhsK,
        n,
        batchNumel,
        out);
    if (!fast)
        return make_error(fast.error());
    if (fast.take())
        return {};
#endif

    auto lhsStrides = strides_for(lhs.desc.shape);
    auto rhsStrides = strides_for(rhs.desc.shape);

    for (size_t batch = 0; batch < static_cast<size_t>(batchNumel); batch++) {
        size_t lhsBatchOffset = matmul_batch_offset(
            batch, batchShape, lhs.desc.shape, lhsStrides);
        size_t rhsBatchOffset = matmul_batch_offset(
            batch, batchShape, rhs.desc.shape, rhsStrides);
        size_t outBatchOffset = batch * static_cast<size_t>(m * n);

        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < n; col++) {
                float acc = 0.0f;
                for (int64_t k = 0; k < lhsK; k++) {
                    int64_t lhsRow = transpose_lhs ? k : row;
                    int64_t lhsCol = transpose_lhs ? row : k;
                    int64_t rhsRow = transpose_rhs ? col : k;
                    int64_t rhsCol = transpose_rhs ? k : col;
                    size_t lhsIndex = lhsBatchOffset +
                        static_cast<size_t>(lhsRow * lhsStrides[lhsRank - 2] +
                                            lhsCol * lhsStrides[lhsRank - 1]);
                    size_t rhsIndex = rhsBatchOffset +
                        static_cast<size_t>(rhsRow * rhsStrides[rhsRank - 2] +
                                            rhsCol * rhsStrides[rhsRank - 1]);
                    acc += lhs.load_float(lhsIndex) * rhs.load_float(rhsIndex);
                }
                out.store_float(outBatchOffset + static_cast<size_t>(row * n + col), acc);
            }
        }
    }

    return {};
}

Result<void> transpose(TensorRef x, MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "transpose input");
    if (!xFloat) return make_error(xFloat.error());
    if (x.desc.shape.rank() != 2)
        return make_error("transpose input must have rank 2");

    auto outDims = x.desc.shape.dims();
    std::swap(outDims[outDims.size() - 1], outDims[outDims.size() - 2]);
    Shape outShape(outDims);
    auto output = require_output(out, outShape, x.desc.dtype, "transpose");
    if (!output) return make_error(output.error());

    int64_t outNumel = outShape.numel();
    if (outNumel < 0)
        return make_error("transpose output must have static shape");

    auto inputStrides = strides_for(x.desc.shape);
    auto outputStrides = strides_for(outShape);
    for (size_t outIndex = 0; outIndex < static_cast<size_t>(outNumel); outIndex++) {
        size_t row = outIndex / static_cast<size_t>(outputStrides[0]);
        size_t col = outIndex % static_cast<size_t>(outputStrides[0]);
        size_t inputIndex = col * static_cast<size_t>(inputStrides[0]) + row;
        out.store_float(outIndex, x.load_float(inputIndex));
    }

    return {};
}

Result<void> reshape(TensorRef x, MutableTensorRef out) {
    if (x.desc.dtype != out.desc.dtype)
        return make_error("reshape output dtype mismatch");

    int64_t inputNumel = x.desc.shape.numel();
    int64_t outputNumel = out.desc.shape.numel();
    if (inputNumel < 0 || outputNumel < 0)
        return make_error("reshape tensors must have static shape");
    if (inputNumel != outputNumel)
        return make_error("reshape element count mismatch");
    if (x.desc.shape.numel() != out.desc.shape.numel())
        return make_error("reshape byte size mismatch");

    for (size_t i = 0; i < static_cast<size_t>(inputNumel); i++)
        out.store_float(i, x.load_float(i));
    return {};
}

Result<void> permute(TensorRef x, std::span<const int64_t> dims, MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "permute input");
    if (!xFloat) return make_error(xFloat.error());

    int rank = x.desc.shape.rank();
    if (static_cast<int>(dims.size()) != rank)
        return make_error("permute dims size must match input rank");

    std::vector<bool> seen(static_cast<size_t>(rank), false);
    std::vector<int64_t> outDims;
    outDims.reserve(dims.size());
    for (int64_t axis : dims) {
        if (axis < 0 || axis >= rank)
            return make_error("permute axis out of range");
        auto index = static_cast<size_t>(axis);
        if (seen[index])
            return make_error("permute dims must not contain duplicates");
        seen[index] = true;
        outDims.push_back(x.desc.shape.dim(static_cast<int>(axis)));
    }

    Shape outShape(outDims);
    auto output = require_output(out, outShape, x.desc.dtype, "permute");
    if (!output) return make_error(output.error());

    int64_t outNumel = outShape.numel();
    if (outNumel < 0)
        return make_error("permute output must have static shape");

    auto inputStrides = strides_for(x.desc.shape);
    auto outputStrides = strides_for(outShape);

    for (size_t outIndex = 0; outIndex < static_cast<size_t>(outNumel); outIndex++) {
        size_t sourceIndex = 0;
        size_t remaining = outIndex;
        for (int axisIndex = 0; axisIndex < rank; axisIndex++) {
            size_t stride = static_cast<size_t>(outputStrides[static_cast<size_t>(axisIndex)]);
            size_t coord = 0;
            if (stride != 0) {
                coord = remaining / stride;
                remaining %= stride;
            }
            int inputAxis = static_cast<int>(dims[static_cast<size_t>(axisIndex)]);
            sourceIndex += coord *
                static_cast<size_t>(inputStrides[static_cast<size_t>(inputAxis)]);
        }
        out.store_float(outIndex, x.load_float(sourceIndex));
    }

    return {};
}

Result<void> sliding_query_key_score_impl(
        TensorRef q,
        TensorRef k,
        const TensorRef* positionIds,
        int64_t window,
        float scale,
        MutableTensorRef out) {
    auto qFloat = require_float_tensor(q, "sliding_query_key_score q");
    if (!qFloat) return make_error(qFloat.error());
    auto kFloat = require_float_tensor(k, "sliding_query_key_score k");
    if (!kFloat) return make_error(kFloat.error());
    auto same = require_same_dtype(q.desc.dtype, k.desc.dtype, "sliding_query_key_score");
    if (!same) return make_error(same.error());

    int rank = q.desc.shape.rank();
    if ((rank != 3 && rank != 4) || k.desc.shape.rank() != rank)
        return make_error("sliding_query_key_score operands must both have rank 3 or rank 4");
    if (window < 0)
        return make_error("sliding_query_key_score window must be >= 0");

    int64_t batch = rank == 4 ? q.desc.shape.dim(0) : 1;
    int64_t kBatch = rank == 4 ? k.desc.shape.dim(0) : 1;
    int64_t heads = q.desc.shape.dim(rank - 3);
    int64_t kvHeads = k.desc.shape.dim(rank - 3);
    int64_t tq = q.desc.shape.dim(rank - 2);
    int64_t tk = k.desc.shape.dim(rank - 2);
    int64_t headDim = q.desc.shape.dim(rank - 1);
    int64_t kHeadDim = k.desc.shape.dim(rank - 1);
    if (batch < 0 || kBatch < 0 || heads < 0 || kvHeads < 0 ||
        tq < 0 || tk < 0 || headDim < 0 || kHeadDim < 0) {
        return make_error("sliding_query_key_score inputs must have static shape");
    }
    if (batch != kBatch)
        return make_error("sliding_query_key_score batch dimension mismatch");
    if (headDim != kHeadDim)
        return make_error("sliding_query_key_score head dimension mismatch");
    if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0)
        return make_error("sliding_query_key_score heads must be divisible by kv_heads");
    int64_t positionCount = 0;
    if (positionIds) {
        if (positionIds->desc.dtype != DType::I32 &&
            positionIds->desc.dtype != DType::I64) {
            return make_error("sliding_query_key_score position_ids must be i32 or i64");
        }
        positionCount = positionIds->desc.shape.numel();
        if (positionCount < 0)
            return make_error("sliding_query_key_score position_ids must have static shape");
        if (positionCount != 1 && positionCount != tq)
            return make_error("sliding_query_key_score position_ids numel must be 1 or query length");
    }

    std::vector<int64_t> outDims;
    if (rank == 4)
        outDims.push_back(batch);
    outDims.push_back(heads);
    outDims.push_back(tq);
    outDims.push_back(tk);
    auto output = require_output(out, Shape(outDims), q.desc.dtype, "sliding_query_key_score");
    if (!output) return make_error(output.error());

    int64_t headsPerKv = heads / kvHeads;
    if (scale <= 0.0f)
        scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    float masked = -std::numeric_limits<float>::infinity();

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < heads; h++) {
            int64_t kh = h / headsPerKv;
            for (int64_t qi = 0; qi < tq; qi++) {
                int64_t queryPosition = qi;
                if (positionIds) {
                    auto positionIndex = positionCount == 1 ? 0 : qi;
                    queryPosition = read_index(*positionIds, static_cast<size_t>(positionIndex));
                    if (queryPosition < 0)
                        return make_error("sliding_query_key_score position_ids must be non-negative");
                }
                int64_t minKey = 0;
                if (window > 0)
                    minKey = std::max<int64_t>(0, queryPosition + 1 - window);
                for (int64_t ki = 0; ki < tk; ki++) {
                    size_t outIndex = rank == 4
                        ? static_cast<size_t>(((b * heads + h) * tq + qi) * tk + ki)
                        : static_cast<size_t>((h * tq + qi) * tk + ki);

                    if (ki > queryPosition || ki < minKey) {
                        out.store_float(outIndex, masked);
                        continue;
                    }

                    float acc = 0.0f;
                    for (int64_t d = 0; d < headDim; d++) {
                        size_t qIndex = 0;
                        size_t kIndex = 0;
                        if (rank == 4) {
                            qIndex = static_cast<size_t>(
                                ((b * heads + h) * tq + qi) * headDim + d);
                            kIndex = static_cast<size_t>(
                                ((b * kvHeads + kh) * tk + ki) * headDim + d);
                        } else {
                            qIndex = static_cast<size_t>((h * tq + qi) * headDim + d);
                            kIndex = static_cast<size_t>((kh * tk + ki) * headDim + d);
                        }
                        acc += q.load_float(qIndex) * k.load_float(kIndex);
                    }
                    out.store_float(outIndex, acc * scale);
                }
            }
        }
    }

    return {};
}

Result<void> sliding_query_key_score(
        TensorRef q,
        TensorRef k,
        TensorRef positionIds,
        int64_t window,
        float scale,
        MutableTensorRef out) {
    return sliding_query_key_score_impl(q, k, &positionIds, window, scale, out);
}

Result<void> sliding_query_key_score(
        TensorRef q,
        TensorRef k,
        int64_t window,
        float scale,
        MutableTensorRef out) {
    return sliding_query_key_score_impl(q, k, nullptr, window, scale, out);
}

Result<void> sliding_query_key_score(
        TensorRef q,
        TensorRef k,
        int64_t window,
        MutableTensorRef out) {
    return sliding_query_key_score(q, k, window, -1.0f, out);
}

Result<void> softmax(TensorRef x, int64_t dim, MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "softmax input");
    if (!xFloat) return make_error(xFloat.error());
    auto output = require_output(out, x.desc.shape, x.desc.dtype, "softmax");
    if (!output) return make_error(output.error());

    int rank = x.desc.shape.rank();
    if (rank < 1)
        return make_error("softmax input must have rank >= 1");
    if (dim < -rank || dim >= rank)
        return make_error("softmax dim out of range");
    if (dim < 0)
        dim += rank;

    int64_t total = x.desc.shape.numel();
    if (total < 0)
        return make_error("softmax input must have static shape");
    if (total == 0)
        return {};

    int64_t axis = x.desc.shape.dim(static_cast<int>(dim));
    if (axis < 0)
        return make_error("softmax axis dimension must be static");

    int64_t inner = 1;
    for (int i = static_cast<int>(dim) + 1; i < rank; i++)
        inner *= x.desc.shape.dim(i);
    int64_t outer = total / (axis * inner);

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            size_t base = static_cast<size_t>(o * axis * inner + i);
            float maxValue = -std::numeric_limits<float>::infinity();
            for (int64_t a = 0; a < axis; a++) {
                size_t index = base + static_cast<size_t>(a * inner);
                maxValue = std::max(maxValue, x.load_float(index));
            }

            if (std::isinf(maxValue) && maxValue < 0.0f) {
                for (int64_t a = 0; a < axis; a++) {
                    size_t index = base + static_cast<size_t>(a * inner);
                    out.store_float(index, 0.0f);
                }
                continue;
            }

            double sum = 0.0;
            for (int64_t a = 0; a < axis; a++) {
                size_t index = base + static_cast<size_t>(a * inner);
                sum += static_cast<double>(std::exp(x.load_float(index) - maxValue));
            }

            float invSum = 1.0f / static_cast<float>(sum);
            for (int64_t a = 0; a < axis; a++) {
                size_t index = base + static_cast<size_t>(a * inner);
                float value = std::exp(x.load_float(index) - maxValue) * invSum;
                out.store_float(index, value);
            }
        }
    }

    return {};
}

Result<void> embedding(TensorRef ids, TensorRef weight, MutableTensorRef out) {
    if (ids.desc.dtype != DType::I32 && ids.desc.dtype != DType::I64)
        return make_error("embedding ids must be i32 or i64");
    auto weightFloat = require_float_tensor(weight, "embedding weight");
    if (!weightFloat) return make_error(weightFloat.error());

    if (weight.desc.shape.rank() != 2)
        return make_error("embedding weight must have rank 2");

    int64_t vocab = weight.desc.shape.dim(0);
    int64_t hidden = weight.desc.shape.dim(1);
    if (vocab < 0 || hidden < 0)
        return make_error("embedding weight must have static shape");

    int64_t idsNumel = ids.desc.shape.numel();
    if (idsNumel < 0)
        return make_error("embedding ids must have static shape");

    auto outDims = ids.desc.shape.dims();
    outDims.push_back(hidden);
    auto output = require_output(out, Shape(outDims), weight.desc.dtype, "embedding");
    if (!output) return make_error(output.error());

    for (int64_t i = 0; i < idsNumel; i++) {
        int64_t tokenId = read_index(ids, static_cast<size_t>(i));
        if (tokenId < 0 || tokenId >= vocab)
            return make_error("embedding id out of range");

        for (int64_t h = 0; h < hidden; h++) {
            float value = weight.load_float(static_cast<size_t>(tokenId * hidden + h));
            out.store_float(static_cast<size_t>(i * hidden + h), value);
        }
    }

    return {};
}

Result<void> rope(TensorRef x, float theta, MutableTensorRef out) {
    return rope(x, theta, -1, false, out);
}

Result<void> rope(TensorRef x, float theta, int64_t rotaryDim, MutableTensorRef out) {
    return rope(x, theta, rotaryDim, false, out);
}

namespace {

Result<void> rope_impl(
        TensorRef x,
        const TensorRef* positionIds,
        float theta,
        int64_t rotaryDim,
        bool splitHalf,
        MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "rope input");
    if (!xFloat) return make_error(xFloat.error());
    auto output = require_output(out, x.desc.shape, x.desc.dtype, "rope");
    if (!output) return make_error(output.error());
    if (positionIds) {
        if (positionIds->desc.dtype != DType::I32 &&
            positionIds->desc.dtype != DType::I64) {
            return make_error("rope position_ids must be i32 or i64");
        }
    }

    int rank = x.desc.shape.rank();
    if (rank < 2)
        return make_error("rope input must have rank >= 2");
    if (theta <= 0.0f)
        return make_error("rope theta must be > 0");

    int64_t seq = x.desc.shape.dim(rank - 2);
    int64_t dim = x.desc.shape.dim(rank - 1);
    if (seq < 0 || dim < 0)
        return make_error("rope input must have static sequence and last dimensions");
    if (dim <= 0 || dim % 2 != 0)
        return make_error("rope last dimension must be positive and even");
    if (rotaryDim < 0)
        rotaryDim = dim;
    if (rotaryDim <= 0 || rotaryDim % 2 != 0)
        return make_error("rope rotary_dim must be positive and even");
    if (rotaryDim > dim)
        return make_error("rope rotary_dim must be <= last dimension");

    int64_t total = x.desc.shape.numel();
    if (total < 0)
        return make_error("rope input must have static shape");

    int64_t vectors = total / dim;
    int64_t positionCount = 0;
    if (positionIds) {
        positionCount = positionIds->desc.shape.numel();
        if (positionCount < 0)
            return make_error("rope position_ids must have static shape");
        if (positionCount != 1 && positionCount != seq && positionCount != vectors) {
            return make_error("rope position_ids numel must be 1, sequence length, or vector count");
        }
    }

    for (int64_t vector = 0; vector < vectors; vector++) {
        int64_t position = vector % seq;
        if (positionIds) {
            int64_t positionIndex = 0;
            if (positionCount == seq) {
                positionIndex = position;
            } else if (positionCount == vectors) {
                positionIndex = vector;
            }
            position = read_index(*positionIds, static_cast<size_t>(positionIndex));
            if (position < 0)
                return make_error("rope position_ids must be non-negative");
        }
        size_t base = static_cast<size_t>(vector * dim);
        if (splitHalf) {
            int64_t half = dim / 2;
            int64_t rotatedPairs = rotaryDim / 2;
            for (int64_t pair = 0; pair < rotatedPairs; pair++) {
                float angle = static_cast<float>(position) /
                    std::pow(theta, static_cast<float>(2 * pair) / static_cast<float>(dim));
                float c = std::cos(angle);
                float s = std::sin(angle);
                size_t firstIndex = base + static_cast<size_t>(pair);
                size_t secondIndex = base + static_cast<size_t>(half + pair);
                float first = x.load_float(firstIndex);
                float second = x.load_float(secondIndex);
                out.store_float(firstIndex, first * c - second * s);
                out.store_float(secondIndex, first * s + second * c);
            }
            for (int64_t i = rotatedPairs; i < half; i++) {
                size_t firstIndex = base + static_cast<size_t>(i);
                size_t secondIndex = base + static_cast<size_t>(half + i);
                out.store_float(firstIndex, x.load_float(firstIndex));
                out.store_float(secondIndex, x.load_float(secondIndex));
            }
        } else {
            for (int64_t pair = 0; pair < rotaryDim / 2; pair++) {
                float angle = static_cast<float>(position) /
                    std::pow(theta, static_cast<float>(2 * pair) / static_cast<float>(rotaryDim));
                float c = std::cos(angle);
                float s = std::sin(angle);
                size_t evenIndex = base + static_cast<size_t>(2 * pair);
                size_t oddIndex = evenIndex + 1;
                float even = x.load_float(evenIndex);
                float odd = x.load_float(oddIndex);
                out.store_float(evenIndex, even * c - odd * s);
                out.store_float(oddIndex, even * s + odd * c);
            }
            for (int64_t i = rotaryDim; i < dim; i++) {
                size_t index = base + static_cast<size_t>(i);
                out.store_float(index, x.load_float(index));
            }
        }
    }

    return {};
}

} // namespace

Result<void> rope(TensorRef x, float theta, int64_t rotaryDim, bool splitHalf, MutableTensorRef out) {
    return rope_impl(x, nullptr, theta, rotaryDim, splitHalf, out);
}

Result<void> rope(
        TensorRef x,
        TensorRef positionIds,
        float theta,
        int64_t rotaryDim,
        bool splitHalf,
        MutableTensorRef out) {
    return rope_impl(x, &positionIds, theta, rotaryDim, splitHalf, out);
}

Result<void> rms_norm_impl(TensorRef x, const TensorRef* weight, float epsilon, MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "rms_norm input");
    if (!xFloat) return make_error(xFloat.error());
    if (weight) {
        auto weightFloat = require_float_tensor(*weight, "rms_norm weight");
        if (!weightFloat) return make_error(weightFloat.error());
        auto same = require_same_dtype(x.desc.dtype, weight->desc.dtype, "rms_norm");
        if (!same) return make_error(same.error());
    }

    if (x.desc.shape.rank() < 1)
        return make_error("rms_norm input must have rank >= 1");
    if (weight && weight->desc.shape.rank() != 1)
        return make_error("rms_norm weight must have rank 1");

    int64_t hidden = x.desc.shape.dim(x.desc.shape.rank() - 1);
    if (hidden < 0)
        return make_error("rms_norm hidden dimension must be static");
    if (weight && weight->desc.shape.dim(0) != hidden)
        return make_error("rms_norm weight dimension mismatch");
    auto output = require_output(out, x.desc.shape, x.desc.dtype, "rms_norm");
    if (!output) return make_error(output.error());

    int64_t total = x.desc.shape.numel();
    if (total < 0)
        return make_error("rms_norm input must have static shape");

    size_t rows = static_cast<size_t>(total / hidden);
    size_t cols = static_cast<size_t>(hidden);
    for (size_t row = 0; row < rows; row++) {
        double squareSum = 0.0;
        size_t rowOffset = row * cols;
        for (size_t col = 0; col < cols; col++) {
            float xv = x.load_float(rowOffset + col);
            squareSum += static_cast<double>(xv) * static_cast<double>(xv);
        }

        float invRms = 1.0f / std::sqrt(
            static_cast<float>(squareSum / static_cast<double>(cols)) + epsilon);
        for (size_t col = 0; col < cols; col++) {
            float xv = x.load_float(rowOffset + col);
            float wv = weight ? weight->load_float(col) : 1.0f;
            out.store_float(rowOffset + col, xv * invRms * wv);
        }
    }

    return {};
}

Result<void> rms_norm(TensorRef x, float epsilon, MutableTensorRef out) {
    return rms_norm_impl(x, nullptr, epsilon, out);
}

Result<void> rms_norm(TensorRef x, TensorRef weight, float epsilon, MutableTensorRef out) {
    return rms_norm_impl(x, &weight, epsilon, out);
}

Result<void> layer_norm(
        TensorRef x,
        TensorRef weight,
        TensorRef bias,
        float epsilon,
        MutableTensorRef out) {
    auto xFloat = require_float_tensor(x, "layer_norm input");
    if (!xFloat) return make_error(xFloat.error());
    auto weightFloat = require_float_tensor(weight, "layer_norm weight");
    if (!weightFloat) return make_error(weightFloat.error());
    auto biasFloat = require_float_tensor(bias, "layer_norm bias");
    if (!biasFloat) return make_error(biasFloat.error());
    if (x.desc.dtype != weight.desc.dtype || x.desc.dtype != bias.desc.dtype)
        return make_error("layer_norm operands must have same dtype");

    int rank = x.desc.shape.rank();
    if (rank < 1)
        return make_error("layer_norm input must have rank >= 1");
    if (weight.desc.shape.rank() != 1)
        return make_error("layer_norm weight must have rank 1");
    if (bias.desc.shape.rank() != 1)
        return make_error("layer_norm bias must have rank 1");

    int64_t hidden = x.desc.shape.dim(rank - 1);
    if (hidden < 0)
        return make_error("layer_norm hidden dimension must be static");
    if (weight.desc.shape.dim(0) != hidden)
        return make_error("layer_norm weight dimension mismatch");
    if (bias.desc.shape.dim(0) != hidden)
        return make_error("layer_norm bias dimension mismatch");
    auto output = require_output(out, x.desc.shape, x.desc.dtype, "layer_norm");
    if (!output) return make_error(output.error());

    int64_t total = x.desc.shape.numel();
    if (total < 0)
        return make_error("layer_norm input must have static shape");
    int64_t rows = total / hidden;

    for (int64_t row = 0; row < rows; row++) {
        size_t base = static_cast<size_t>(row * hidden);
        double mean = 0.0;
        for (int64_t i = 0; i < hidden; i++)
            mean += static_cast<double>(x.load_float(base + static_cast<size_t>(i)));
        mean /= static_cast<double>(hidden);

        double variance = 0.0;
        for (int64_t i = 0; i < hidden; i++) {
            double centered =
                static_cast<double>(x.load_float(base + static_cast<size_t>(i))) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(hidden);

        float invStd = 1.0f / std::sqrt(static_cast<float>(variance) + epsilon);
        for (int64_t i = 0; i < hidden; i++) {
            float centered = x.load_float(base + static_cast<size_t>(i)) -
                static_cast<float>(mean);
            float value = centered * invStd *
                weight.load_float(static_cast<size_t>(i)) +
                bias.load_float(static_cast<size_t>(i));
            out.store_float(base + static_cast<size_t>(i), value);
        }
    }

    return {};
}

} // namespace sandy::core
