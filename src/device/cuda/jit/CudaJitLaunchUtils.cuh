#pragma once

#include "CudaKernelUtils.cuh"
#include "templates/CudaJitAbi.cuh"

namespace sandy::device {

inline Result<SandyJitTensorArg> pack_jit_tensor_arg(
        const cuda_kernel::TensorArg& source) {
    SandyJitTensorArg target{};
    target.data = source.data;
    for (int i = 0; i < SANDY_JIT_MAX_RANK; ++i) {
        target.dims[i] = source.dims[i];
        target.strides[i] = source.strides[i];
    }
    target.storageOffset = source.storageOffset;
    target.numel = source.numel;
    target.growDim = source.growDim;
    target.pageSize = source.pageSize;
    target.pageCount = source.pageCount;
    target.pageElementCount = source.pageElementCount;
    target.pageMask = source.pageMask;
    target.pagedInnerElements = source.pagedInnerElements;
    target.pagedLogicalPrefixElements = source.pagedLogicalPrefixElements;
    target.pagedPhysicalPrefixElements = source.pagedPhysicalPrefixElements;
    target.rank = source.rank;
    target.pageShift = source.pageShift;

    switch (source.dtype) {
        case core::DType::F32:
            target.dtype = SANDY_JIT_F32;
            break;
        case core::DType::F16:
            target.dtype = SANDY_JIT_F16;
            break;
        case core::DType::BF16:
            target.dtype = SANDY_JIT_BF16;
            break;
        case core::DType::I32:
            target.dtype = SANDY_JIT_I32;
            break;
        case core::DType::I64:
            target.dtype = SANDY_JIT_I64;
            break;
        case core::DType::U8:
            target.dtype = SANDY_JIT_U8;
            break;
    }
    switch (source.access) {
        case cuda_kernel::AccessKind::Contiguous:
            target.access = SANDY_JIT_CONTIGUOUS;
            break;
        case cuda_kernel::AccessKind::Strided:
            target.access = SANDY_JIT_STRIDED;
            break;
        case cuda_kernel::AccessKind::Paged:
            target.access = SANDY_JIT_PAGED;
            break;
    }
    return target;
}

} // namespace sandy::device
