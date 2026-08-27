#pragma once
#ifdef __CUDACC_RTC__
using int32_t = int;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using int64_t = long long;
#else
#include <cstdint>
#endif

constexpr int SANDY_JIT_ABI_VERSION = 1;
constexpr int SANDY_JIT_MAX_RANK = 8;
constexpr int SANDY_JIT_MAX_INPUTS = 8;

enum SandyJitDType : int32_t { SANDY_JIT_F32 = 0, SANDY_JIT_BF16 = 1 };
enum SandyJitAccess : int32_t {
    SANDY_JIT_CONTIGUOUS = 0,
    SANDY_JIT_STRIDED = 1,
    SANDY_JIT_PAGED = 2,
};
enum SandyJitBroadcast : int32_t {
    SANDY_JIT_BROADCAST_NONE = 0,
    SANDY_JIT_BROADCAST_RIGHT_ALIGNED = 1,
};

struct SandyJitTensorArg {
    void* data;
    int64_t dims[SANDY_JIT_MAX_RANK];
    int64_t strides[SANDY_JIT_MAX_RANK];
    int64_t storageOffset;
    int64_t numel;
    int64_t growDim;
    int64_t pageSize;
    int64_t pageCount;
    int64_t pageElementCount;
    int64_t pageMask;
    int64_t pagedInnerElements;
    int64_t pagedLogicalPrefixElements;
    int64_t pagedPhysicalPrefixElements;
    int32_t rank;
    int32_t pageShift;
    int32_t dtype;
    int32_t access;
};

struct SandyElementwiseParams {
    SandyJitTensorArg inputs[SANDY_JIT_MAX_INPUTS];
    int32_t broadcasts[SANDY_JIT_MAX_INPUTS];
    SandyJitTensorArg output;
};
