#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <cstdint>
#include <utility>

namespace sandy::device {

namespace {

struct DeviceGatherProgram {
    cuda_kernel::TensorArg ids;
    cuda_kernel::TensorArg table;
    cuda_kernel::TensorArg output;
    int64_t vocab = 0;
    int64_t hidden = 0;
    int64_t outputNumel = 0;
};

Result<DeviceGatherProgram> pack_gather_program(const CudaLaunchContext& context) {
    const auto& idsView = context.inputs[0].view;
    const auto& tableView = context.inputs[1].view;
    const auto& outputView = context.outputs[0].view;

    if (idsView.desc.dtype != core::DType::I32 && idsView.desc.dtype != core::DType::I64)
        return make_error("cuda gather ids must be i32 or i64");
    if (!is_float_compute_dtype(tableView.desc.dtype))
        return make_error("cuda gather table unsupported dtype");
    if (outputView.desc.dtype != tableView.desc.dtype)
        return make_error("cuda gather output dtype mismatch");
    if (tableView.desc.shape.rank() != 2)
        return make_error("cuda gather table must have rank 2");

    int64_t vocab = tableView.desc.shape.dim(0);
    int64_t hidden = tableView.desc.shape.dim(1);
    if (vocab < 0 || hidden < 0)
        return make_error("cuda gather table must have static shape");
    if (idsView.desc.shape.numel() < 0)
        return make_error("cuda gather ids must have static shape");

    auto outDims = idsView.desc.shape.dims();
    outDims.push_back(hidden);
    if (outputView.desc.shape != core::Shape(std::move(outDims)))
        return make_error("cuda gather output shape mismatch");

    auto ids = cuda_kernel::pack_tensor_arg(context.inputs[0]);
    if (!ids)
        return make_error(ids.error());
    auto table = cuda_kernel::pack_tensor_arg(context.inputs[1]);
    if (!table)
        return make_error(table.error());
    auto output = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!output)
        return make_error(output.error());

    DeviceGatherProgram program;
    program.ids = ids.take();
    program.table = table.take();
    program.output = output.take();
    program.vocab = vocab;
    program.hidden = hidden;
    program.outputNumel = program.output.numel;
    return program;
}

__device__ int64_t load_index_at_storage(
        const cuda_kernel::TensorArg& tensor,
        int64_t index) {
    return cuda_kernel::load_int_at_storage(tensor, index);
}

__global__ void gather_kernel(DeviceGatherProgram program, int* errorFlag) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.outputNumel)
        return;

    int64_t hiddenIndex = linear % program.hidden;
    int64_t idLinear = linear / program.hidden;
    int64_t idStorage = cuda_kernel::storage_index(program.ids, idLinear);
    int64_t tokenId = load_index_at_storage(program.ids, idStorage);
    if (tokenId < 0 || tokenId >= program.vocab) {
        atomicExch(errorFlag, 1);
        return;
    }

    int64_t tableStorage =
        program.table.storageOffset +
        tokenId * program.table.strides[0] +
        hiddenIndex * program.table.strides[1];
    float value = cuda_kernel::load_float_at_storage(program.table, tableStorage);
    cuda_kernel::store_float(program.output, linear, value);
}

} // namespace

Result<void> launch_cuda_gather(const CudaLaunchContext& context) {
    auto valid = validate_context(context, 2, 1, "gather");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_gather_program(context);
    if (!packed)
        return make_error(packed.error());
    auto program = packed.take();
    if (program.outputNumel == 0)
        return {};

    int* errorFlag = nullptr;
    auto allocated = cuda_check(cudaMalloc(&errorFlag, sizeof(int)), "cudaMalloc gather error flag");
    if (!allocated)
        return make_error(allocated.error());
    auto freeFlag = [&]() {
        if (errorFlag)
            cudaFree(errorFlag);
    };

    auto cleared = cuda_check(
        cudaMemsetAsync(errorFlag, 0, sizeof(int), context.stream),
        "cudaMemsetAsync gather error flag");
    if (!cleared) {
        freeFlag();
        return make_error(cleared.error());
    }

    int blocks = static_cast<int>(
        (program.outputNumel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    gather_kernel<<<blocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
        program,
        errorFlag);
    auto launched = cuda_check(cudaGetLastError(), "cuda gather launch");
    if (!launched) {
        freeFlag();
        return make_error(launched.error());
    }

    int hostError = 0;
    auto copied = cuda_check(
        cudaMemcpyAsync(
            &hostError,
            errorFlag,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            context.stream),
        "cudaMemcpyAsync gather error flag");
    if (!copied) {
        freeFlag();
        return make_error(copied.error());
    }

    auto synced = cuda_check(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize gather");
    if (!synced) {
        freeFlag();
        return make_error(synced.error());
    }

    freeFlag();
    if (hostError != 0)
        return make_error("embedding id out of range");
    return {};
}

} // namespace sandy::device
