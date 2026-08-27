#include "CudaJitLayoutTransformAbi.cuh"
#include "CudaJitLayoutTransformConfig.cuh"
#include "CudaJitTensorAccess.cuh"

extern "C" __global__ void sandy_jit_layout_transform(
        SandyLayoutTransformParams params) {
    int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= params.output.numel)
        return;
    sandy_runtime_copy_element<
        SandyLayoutElement,
        SandyLayoutInputAccess,
        SandyLayoutOutputAccess>(
        params.input, params.output, linear);
}
