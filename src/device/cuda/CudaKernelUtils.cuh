#pragma once

#include "CudaKernels.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sandy::device::cuda_kernel {

constexpr int kMaxRank = 8;
constexpr int kMaxInputs = 8;
constexpr int kBlockSize = 256;

__device__ inline void record_validation_failure(
        ir::kernel_ir::OpId* failure,
        ir::kernel_ir::OpId op) {
    if (!failure)
        return;
    atomicCAS(
        reinterpret_cast<unsigned int*>(failure),
        static_cast<unsigned int>(ir::kernel_ir::kInvalidOpId),
        static_cast<unsigned int>(op));
}

enum class AccessKind : int {
    Contiguous,
    Strided,
    Paged,
};

struct TensorArg {
    void* data = nullptr;
    int64_t dims[kMaxRank]{};
    int64_t strides[kMaxRank]{};
    int rank = 0;
    int64_t storageOffset = 0;
    int64_t numel = 0;
    core::DType dtype = core::DType::F32;
    AccessKind access = AccessKind::Strided;
    int64_t growDim = -1;
    int64_t pageSize = -1;
    int64_t pageCount = 0;
    int64_t pageElementCount = 0;
    int pageShift = -1;
    int64_t pageMask = 0;
    int64_t pagedInnerElements = 0;
    int64_t pagedLogicalPrefixElements = 0;
    int64_t pagedPhysicalPrefixElements = 0;
};

template <int MaxInputs = kMaxInputs>
struct TensorInputs {
    TensorArg items[MaxInputs]{};
    int count = 0;
};

__device__ inline int64_t strided_storage_index(
        const TensorArg& tensor,
        int64_t linear) {
    int64_t storage = tensor.storageOffset;
    int64_t remaining = linear;

    for (int d = tensor.rank - 1; d >= 0; --d) {
        int64_t coord = remaining % tensor.dims[d];
        remaining /= tensor.dims[d];
        storage += coord * tensor.strides[d];
    }

    return storage;
}

__device__ inline int64_t storage_index(
        const TensorArg& tensor,
        int64_t linear) {
    if (tensor.access == AccessKind::Contiguous)
        return tensor.storageOffset + linear;
    return strided_storage_index(tensor, linear);
}

__device__ inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t fbits = static_cast<uint32_t>(bits) << 16;
    return __uint_as_float(fbits);
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__device__ inline int64_t paged_storage_index(
        const TensorArg& tensor,
        int64_t linear,
        void** pageOut) {
    if (tensor.pagedInnerElements <= 0 ||
        tensor.pagedLogicalPrefixElements <= 0 ||
        tensor.pageShift < 0) {
        *pageOut = nullptr;
        return -1;
    }

    int64_t prefix = linear / tensor.pagedLogicalPrefixElements;
    int64_t prefixRemainder =
        linear - prefix * tensor.pagedLogicalPrefixElements;
    int64_t grow = prefixRemainder / tensor.pagedInnerElements;
    int64_t inner = prefixRemainder - grow * tensor.pagedInnerElements;
    int64_t pageOrdinal = grow >> tensor.pageShift;
    int64_t growInPage = grow & tensor.pageMask;
    if (pageOrdinal < 0 || pageOrdinal >= tensor.pageCount) {
        *pageOut = nullptr;
        return -1;
    }

    *pageOut = static_cast<void**>(tensor.data)[pageOrdinal];
    return prefix * tensor.pagedPhysicalPrefixElements +
           growInPage * tensor.pagedInnerElements + inner;
}

__device__ inline void* storage_data(
        const TensorArg& tensor,
        int64_t linear,
        int64_t* storageIndex) {
    if (tensor.access == AccessKind::Paged) {
        void* page = nullptr;
        *storageIndex = paged_storage_index(tensor, linear, &page);
        return page;
    }

    *storageIndex = storage_index(tensor, linear);
    return tensor.data;
}

__device__ inline float load_float_at_storage(
        const void* data,
        core::DType dtype,
        int64_t index) {
    if (!data || index < 0)
        return 0.0f;

    switch (dtype) {
        case core::DType::F32:
            return static_cast<const float*>(data)[index];
        case core::DType::BF16:
            return bf16_bits_to_float(
                static_cast<const uint16_t*>(data)[index]);
        default:
            return 0.0f;
    }
}

__device__ inline float load_float_at_storage(
        const TensorArg& tensor,
        int64_t index) {
    return load_float_at_storage(tensor.data, tensor.dtype, index);
}

__device__ inline float load_float(
        const TensorArg& tensor,
        int64_t linear) {
    int64_t index = -1;
    const void* data = storage_data(tensor, linear, &index);
    return load_float_at_storage(data, tensor.dtype, index);
}

__device__ inline int64_t load_int_at_storage(
        const void* data,
        core::DType dtype,
        int64_t index) {
    if (!data || index < 0)
        return 0;

    if (dtype == core::DType::I32)
        return static_cast<int64_t>(static_cast<const int32_t*>(data)[index]);
    if (dtype == core::DType::I64)
        return static_cast<const int64_t*>(data)[index];
    return 0;
}

__device__ inline int64_t load_int_at_storage(
        const TensorArg& tensor,
        int64_t index) {
    return load_int_at_storage(tensor.data, tensor.dtype, index);
}

__device__ inline int64_t load_int(
        const TensorArg& tensor,
        int64_t linear) {
    int64_t index = -1;
    const void* data = storage_data(tensor, linear, &index);
    return load_int_at_storage(data, tensor.dtype, index);
}

__device__ inline void store_int_at_storage(
        void* data,
        core::DType dtype,
        int64_t index,
        int64_t value) {
    if (!data || index < 0)
        return;

    switch (dtype) {
        case core::DType::I32:
            static_cast<int32_t*>(data)[index] = static_cast<int32_t>(value);
            return;
        case core::DType::I64:
            static_cast<int64_t*>(data)[index] = value;
            return;
        default:
            return;
    }
}

__device__ inline void store_int_at_storage(
        const TensorArg& tensor,
        int64_t index,
        int64_t value) {
    store_int_at_storage(tensor.data, tensor.dtype, index, value);
}

__device__ inline void store_int(
        const TensorArg& tensor,
        int64_t linear,
        int64_t value) {
    int64_t index = -1;
    void* data = storage_data(tensor, linear, &index);
    store_int_at_storage(data, tensor.dtype, index, value);
}

__device__ inline void store_float_at_storage(
        void* data,
        core::DType dtype,
        int64_t index,
        float value) {
    if (!data || index < 0)
        return;

    switch (dtype) {
        case core::DType::F32:
            static_cast<float*>(data)[index] = value;
            return;
        case core::DType::BF16:
            static_cast<uint16_t*>(data)[index] = float_to_bf16_bits(value);
            return;
        default:
            return;
    }
}

__device__ inline void store_float_at_storage(
        const TensorArg& tensor,
        int64_t index,
        float value) {
    store_float_at_storage(tensor.data, tensor.dtype, index, value);
}

__device__ inline void store_float(
        const TensorArg& tensor,
        int64_t linear,
        float value) {
    int64_t index = -1;
    void* data = storage_data(tensor, linear, &index);
    store_float_at_storage(data, tensor.dtype, index, value);
}

inline bool is_contiguous(const TensorViewDesc& view) {
    int64_t expected = 1;
    for (int d = view.desc.shape.rank() - 1; d >= 0; --d) {
        if (view.desc.shape.dim(d) != 1 &&
            view.strides[static_cast<size_t>(d)] != expected)
            return false;
        expected *= view.desc.shape.dim(d);
    }
    return true;
}

inline Result<TensorArg> pack_tensor_arg(const CudaDeviceBufferView& buffer) {
    const auto& view = buffer.view;
    int rank = view.desc.shape.rank();
    if (rank > kMaxRank)
        return make_error("cuda tensor rank exceeds kernel max rank");
    if (static_cast<int>(view.strides.size()) != rank)
        return make_error("cuda tensor strides rank mismatch");
    if (view.storageOffset < 0)
        return make_error("cuda tensor storage offset must be non-negative");

    int64_t numel = view.desc.shape.numel();
    if (numel < 0)
        return make_error("cuda tensor shape must be static");

    TensorArg arg;
    arg.data = buffer.data;
    arg.rank = rank;
    arg.storageOffset = view.storageOffset;
    arg.numel = numel;
    arg.dtype = view.desc.dtype;
    arg.access = is_contiguous(view) ? AccessKind::Contiguous
                                     : AccessKind::Strided;
    if (buffer.paged) {
        if (!is_contiguous(view))
            return make_error("cuda paged tensor view must be contiguous");
        if (view.storageOffset != 0)
            return make_error("cuda paged tensor storage offset must be zero");
        if (buffer.growDim < 0 || buffer.growDim >= rank)
            return make_error("cuda paged tensor grow_dim out of range");
        if (buffer.pageSize <= 0)
            return make_error("cuda paged tensor page_size must be > 0");
        if (!std::has_single_bit(static_cast<uint64_t>(buffer.pageSize)))
            return make_error("cuda paged tensor page_size must be a power of two");
        if (buffer.pageCount < 0)
            return make_error("cuda paged tensor page_count must be >= 0");
        if (buffer.pageElementCount <= 0)
            return make_error("cuda paged tensor page element count must be > 0");
        arg.access = AccessKind::Paged;
        arg.growDim = buffer.growDim;
        arg.pageSize = buffer.pageSize;
        arg.pageCount = buffer.pageCount;
        arg.pageElementCount = buffer.pageElementCount;
        arg.pageShift = std::countr_zero(static_cast<uint64_t>(buffer.pageSize));
        arg.pageMask = buffer.pageSize - 1;
    }

    int64_t maxStorageIndex = view.storageOffset;
    for (int i = 0; i < rank; ++i) {
        int64_t dim = view.desc.shape.dim(i);
        int64_t stride = view.strides[static_cast<size_t>(i)];
        if (dim < 0)
            return make_error("cuda tensor dim must be static");
        if (stride < 0)
            return make_error("cuda tensor strides must be non-negative");
        arg.dims[i] = dim;
        arg.strides[i] = stride;
        if (numel != 0 && dim > 0)
            maxStorageIndex += (dim - 1) * stride;
    }

    if (buffer.paged) {
        int64_t innerElements = 1;
        for (int i = static_cast<int>(buffer.growDim) + 1; i < rank; ++i) {
            auto dim = arg.dims[i];
            if (dim <= 0)
                return make_error("cuda paged tensor inner dimensions must be positive");
            if (innerElements > std::numeric_limits<int64_t>::max() / dim)
                return make_error("cuda paged tensor inner element count overflow");
            innerElements *= dim;
        }

        auto growLength = arg.dims[buffer.growDim];
        if (growLength < 0)
            return make_error("cuda paged tensor grow dimension must be static");
        if (growLength != 0 &&
            innerElements > std::numeric_limits<int64_t>::max() / growLength) {
            return make_error("cuda paged tensor logical prefix size overflow");
        }
        if (innerElements > std::numeric_limits<int64_t>::max() / buffer.pageSize)
            return make_error("cuda paged tensor physical prefix size overflow");

        arg.pagedInnerElements = innerElements;
        arg.pagedLogicalPrefixElements = growLength * innerElements;
        arg.pagedPhysicalPrefixElements = buffer.pageSize * innerElements;
    }

    if (numel != 0 && !buffer.paged) {
        size_t requiredBytes =
            static_cast<size_t>(maxStorageIndex + 1) * core::dtype_size(view.desc.dtype);
        if (buffer.bytes < requiredBytes)
            return make_error("cuda tensor view exceeds buffer storage");
    }
    if (numel != 0 && buffer.paged && !buffer.data)
        return make_error("cuda paged tensor page table is null");

    return arg;
}

} // namespace sandy::device::cuda_kernel
