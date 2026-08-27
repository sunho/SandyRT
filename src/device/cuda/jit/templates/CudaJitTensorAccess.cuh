#pragma once
#include "CudaJitAbi.cuh"

__device__ __forceinline__ float sandy_bf16_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t sandy_float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ int64_t sandy_strided_index(
        const SandyJitTensorArg& tensor,
        int64_t linear) {
    int64_t storage = tensor.storageOffset;
    for (int dim = tensor.rank - 1; dim >= 0; --dim) {
        int64_t coordinate = linear % tensor.dims[dim];
        linear /= tensor.dims[dim];
        storage += coordinate * tensor.strides[dim];
    }
    return storage;
}

__device__ __forceinline__ void* sandy_storage_data(
        const SandyJitTensorArg& tensor,
        int64_t linear,
        int64_t* storage) {
    if (tensor.access != SANDY_JIT_PAGED) {
        *storage = tensor.access == SANDY_JIT_CONTIGUOUS
            ? tensor.storageOffset + linear
            : sandy_strided_index(tensor, linear);
        return tensor.data;
    }
    int64_t prefix = linear / tensor.pagedLogicalPrefixElements;
    int64_t remainder = linear - prefix * tensor.pagedLogicalPrefixElements;
    int64_t grow = remainder / tensor.pagedInnerElements;
    int64_t inner = remainder - grow * tensor.pagedInnerElements;
    int64_t pageOrdinal = grow >> tensor.pageShift;
    int64_t growInPage = grow & tensor.pageMask;
    if (pageOrdinal < 0 || pageOrdinal >= tensor.pageCount) {
        *storage = -1;
        return nullptr;
    }
    *storage = prefix * tensor.pagedPhysicalPrefixElements +
               growInPage * tensor.pagedInnerElements + inner;
    return static_cast<void**>(tensor.data)[pageOrdinal];
}

template <int DType>
__device__ __forceinline__ float sandy_runtime_load(
        const SandyJitTensorArg& tensor,
        int64_t linear) {
    int64_t storage = -1;
    const void* data = sandy_storage_data(tensor, linear, &storage);
    if (!data || storage < 0)
        return 0.0f;
    if constexpr (DType == SANDY_JIT_F32)
        return static_cast<const float*>(data)[storage];
    else
        return sandy_bf16_to_float(static_cast<const uint16_t*>(data)[storage]);
}

template <int DType>
__device__ __forceinline__ void sandy_runtime_store(
        const SandyJitTensorArg& tensor,
        int64_t linear,
        float value) {
    int64_t storage = -1;
    void* data = sandy_storage_data(tensor, linear, &storage);
    if (!data || storage < 0)
        return;
    if constexpr (DType == SANDY_JIT_F32)
        static_cast<float*>(data)[storage] = value;
    else
        static_cast<uint16_t*>(data)[storage] = sandy_float_to_bf16(value);
}

template <typename Element>
__device__ __forceinline__ void sandy_runtime_copy_element(
        const SandyJitTensorArg& input,
        const SandyJitTensorArg& output,
        int64_t linear) {
    int64_t inputStorage = -1;
    int64_t outputStorage = -1;
    const void* inputData = sandy_storage_data(input, linear, &inputStorage);
    void* outputData = sandy_storage_data(output, linear, &outputStorage);
    if (!inputData || !outputData || inputStorage < 0 || outputStorage < 0)
        return;
    static_cast<Element*>(outputData)[outputStorage] =
        static_cast<const Element*>(inputData)[inputStorage];
}

struct SandyRuntimeLoader {
    const SandyElementwiseParams& params;
    int64_t linear;

    template <int DType>
    __device__ __forceinline__ float load(int inputIndex) const {
        const auto& input = params.inputs[inputIndex];
        int64_t inputLinear = linear;
        if (params.broadcasts[inputIndex] == SANDY_JIT_BROADCAST_RIGHT_ALIGNED)
            inputLinear = input.numel == 0 ? 0 : linear % input.numel;
        return sandy_runtime_load<DType>(input, inputLinear);
    }
};

struct SandyRuntimeStorer {
    template <int DType>
    __device__ __forceinline__ static void store(
            const SandyJitTensorArg& output,
            int64_t linear,
            float value) {
        sandy_runtime_store<DType>(output, linear, value);
    }
};
