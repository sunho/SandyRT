#include "CudaJitGatherAbi.cuh"
#include "CudaJitGatherConfig.cuh"
#include "CudaJitTensorAccess.cuh"

extern "C" __global__ void sandy_jit_gather(SandyGatherParams params) {
    int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= params.output.numel)
        return;

    int64_t hiddenIndex = 0;
    int64_t idLinear = linear;
    if constexpr (SandyGatherTableRank == 2) {
        hiddenIndex = linear % params.hidden;
        idLinear = linear / params.hidden;
    }
    int64_t tokenId = sandy_runtime_load_integer<
        SandyGatherIndex,
        SandyGatherIdsAccess>(
        params.ids, idLinear);
    if (tokenId < 0 || tokenId >= params.vocab) {
        if (params.errorFlag)
            atomicExch(params.errorFlag, 1);
        return;
    }

    int64_t tableLinear = tokenId;
    if constexpr (SandyGatherTableRank == 2)
        tableLinear = tokenId * params.hidden + hiddenIndex;
    float value = sandy_runtime_load<
        SandyGatherValueDType,
        SandyGatherTableAccess>(
        params.table, tableLinear);
    sandy_runtime_store<
        SandyGatherValueDType,
        SandyGatherOutputAccess>(params.output, linear, value);
}
