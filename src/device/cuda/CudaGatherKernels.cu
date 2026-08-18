#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sandy::device {

namespace {

struct DeviceGatherProgram {
    cuda_kernel::TensorArg ids;
    cuda_kernel::TensorArg table;
    cuda_kernel::TensorArg output;
    int64_t vocab = 0;
    int64_t hidden = 0;
    int64_t outputNumel = 0;
    int tableRank = 0;
};

struct DeviceMoeGatherProgram {
    cuda_kernel::TensorArg x;
    cuda_kernel::TensorArg topkIds;
    cuda_kernel::TensorArg topkWeights;
    cuda_kernel::TensorArg packedX;
    cuda_kernel::TensorArg packedWeights;
    cuda_kernel::TensorArg tokenIds;
    cuda_kernel::TensorArg expertOffsets;
    int64_t batch = 1;
    int64_t seq = 0;
    int64_t hidden = 0;
    int64_t routesPerBatch = 0;
    int64_t totalRoutes = 0;
    int64_t topK = 0;
    int64_t numExperts = 0;
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
    int tableRank = tableView.desc.shape.rank();
    if (tableRank != 1 && tableRank != 2)
        return make_error("cuda gather table must have rank 1 or rank 2");

    int64_t vocab = tableView.desc.shape.dim(0);
    int64_t hidden = tableRank == 2 ? tableView.desc.shape.dim(1) : 1;
    if (vocab < 0 || hidden < 0)
        return make_error("cuda gather table must have static shape");
    if (idsView.desc.shape.numel() < 0)
        return make_error("cuda gather ids must have static shape");

    auto outDims = idsView.desc.shape.dims();
    if (tableRank == 2)
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
    program.tableRank = tableRank;
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

    int64_t hiddenIndex = program.tableRank == 2 ? linear % program.hidden : 0;
    int64_t idLinear = program.tableRank == 2 ? linear / program.hidden : linear;
    int64_t idStorage = cuda_kernel::storage_index(program.ids, idLinear);
    int64_t tokenId = load_index_at_storage(program.ids, idStorage);
    if (tokenId < 0 || tokenId >= program.vocab) {
        if (errorFlag)
            atomicExch(errorFlag, 1);
        return;
    }

    int64_t tableStorage = program.table.storageOffset + tokenId * program.table.strides[0];
    if (program.tableRank == 2)
        tableStorage += hiddenIndex * program.table.strides[1];
    float value = cuda_kernel::load_float_at_storage(program.table, tableStorage);
    cuda_kernel::store_float(program.output, linear, value);
}

Result<void> validate_moe_gather_float_tensor(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda moe_gather ") + name + " unsupported dtype");
    return {};
}

Result<void> validate_moe_gather_contiguous(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (buffer.paged)
        return make_error(std::string("cuda moe_gather ") + name + " cannot be paged");
    if (!cuda_kernel::is_contiguous(buffer.view))
        return make_error(std::string("cuda moe_gather ") + name + " must be contiguous");
    return {};
}

Result<DeviceMoeGatherProgram> pack_moe_gather_program(
        const CudaLaunchContext& context,
        const CudaMoeGatherProgram& program) {
    if (program.numExperts <= 0 || program.topK <= 0)
        return make_error("cuda moe_gather num_experts and top_k must be positive");

    const auto& xView = context.inputs[0];
    const auto& idsView = context.inputs[1];
    const auto& weightsView = context.inputs[2];
    const auto& packedXView = context.outputs[0];
    const auto& packedWeightsView = context.outputs[1];
    const auto& tokenIdsView = context.outputs[2];
    const auto& offsetsView = context.outputs[3];

    const CudaDeviceBufferView* allBuffers[] = {
        &xView,
        &idsView,
        &weightsView,
        &packedXView,
        &packedWeightsView,
        &tokenIdsView,
        &offsetsView,
    };
    const char* names[] = {
        "x",
        "topk_ids",
        "topk_weights",
        "packed_x",
        "packed_weights",
        "token_ids",
        "expert_offsets",
    };
    for (int i = 0; i < 7; ++i) {
        auto contiguous = validate_moe_gather_contiguous(*allBuffers[i], names[i]);
        if (!contiguous)
            return make_error(contiguous.error());
    }

    auto xDtype = validate_moe_gather_float_tensor(xView, "x");
    if (!xDtype)
        return make_error(xDtype.error());
    auto weightsDtype = validate_moe_gather_float_tensor(weightsView, "topk_weights");
    if (!weightsDtype)
        return make_error(weightsDtype.error());
    if (packedXView.view.desc.dtype != xView.view.desc.dtype)
        return make_error("cuda moe_gather packed_x dtype mismatch");
    if (packedWeightsView.view.desc.dtype != weightsView.view.desc.dtype)
        return make_error("cuda moe_gather packed_weights dtype mismatch");
    if (idsView.view.desc.dtype != core::DType::I32 &&
        idsView.view.desc.dtype != core::DType::I64) {
        return make_error("cuda moe_gather topk_ids must be i32 or i64");
    }
    if (tokenIdsView.view.desc.dtype != core::DType::I32 &&
        tokenIdsView.view.desc.dtype != core::DType::I64) {
        return make_error("cuda moe_gather token_ids must be i32 or i64");
    }
    if (offsetsView.view.desc.dtype != core::DType::I32 &&
        offsetsView.view.desc.dtype != core::DType::I64) {
        return make_error("cuda moe_gather expert_offsets must be i32 or i64");
    }

    int xRank = xView.view.desc.shape.rank();
    if ((xRank != 2 && xRank != 3) ||
        idsView.view.desc.shape.rank() != xRank ||
        weightsView.view.desc.shape.rank() != xRank) {
        return make_error("cuda moe_gather inputs must have rank 2 or rank 3");
    }

    int64_t batch = xRank == 3 ? xView.view.desc.shape.dim(0) : 1;
    int64_t seq = xRank == 3 ? xView.view.desc.shape.dim(1) : xView.view.desc.shape.dim(0);
    int64_t hidden = xView.view.desc.shape.dim(xRank - 1);
    if (batch <= 0 || seq < 0 || hidden <= 0)
        return make_error("cuda moe_gather requires static positive batch/hidden and static sequence");

    for (int axis = 0; axis < xRank - 1; ++axis) {
        if (xView.view.desc.shape.dim(axis) != idsView.view.desc.shape.dim(axis) ||
            xView.view.desc.shape.dim(axis) != weightsView.view.desc.shape.dim(axis)) {
            return make_error("cuda moe_gather leading dimension mismatch");
        }
    }
    if (idsView.view.desc.shape.dim(xRank - 1) != program.topK ||
        weightsView.view.desc.shape.dim(xRank - 1) != program.topK) {
        return make_error("cuda moe_gather top_k dimension mismatch");
    }

    int64_t routesPerBatch = seq * program.topK;
    int64_t totalRoutes = batch * routesPerBatch;
    if (routesPerBatch > std::numeric_limits<int32_t>::max())
        return make_error("cuda moe_gather routes per batch exceed int32 limit");

    core::Shape packedXShape = xRank == 3
        ? core::Shape({batch, routesPerBatch, hidden})
        : core::Shape({routesPerBatch, hidden});
    core::Shape metadataShape = xRank == 3
        ? core::Shape({batch, routesPerBatch})
        : core::Shape({routesPerBatch});
    core::Shape offsetShape = xRank == 3
        ? core::Shape({batch, program.numExperts + 1})
        : core::Shape({program.numExperts + 1});
    if (packedXView.view.desc.shape != packedXShape ||
        packedWeightsView.view.desc.shape != metadataShape ||
        tokenIdsView.view.desc.shape != metadataShape ||
        offsetsView.view.desc.shape != offsetShape) {
        return make_error("cuda moe_gather output shape mismatch");
    }

    auto x = cuda_kernel::pack_tensor_arg(xView);
    if (!x)
        return make_error(x.error());
    auto topkIds = cuda_kernel::pack_tensor_arg(idsView);
    if (!topkIds)
        return make_error(topkIds.error());
    auto topkWeights = cuda_kernel::pack_tensor_arg(weightsView);
    if (!topkWeights)
        return make_error(topkWeights.error());
    auto packedX = cuda_kernel::pack_tensor_arg(packedXView);
    if (!packedX)
        return make_error(packedX.error());
    auto packedWeights = cuda_kernel::pack_tensor_arg(packedWeightsView);
    if (!packedWeights)
        return make_error(packedWeights.error());
    auto tokenIds = cuda_kernel::pack_tensor_arg(tokenIdsView);
    if (!tokenIds)
        return make_error(tokenIds.error());
    auto expertOffsets = cuda_kernel::pack_tensor_arg(offsetsView);
    if (!expertOffsets)
        return make_error(expertOffsets.error());

    DeviceMoeGatherProgram packed;
    packed.x = x.take();
    packed.topkIds = topkIds.take();
    packed.topkWeights = topkWeights.take();
    packed.packedX = packedX.take();
    packed.packedWeights = packedWeights.take();
    packed.tokenIds = tokenIds.take();
    packed.expertOffsets = expertOffsets.take();
    packed.batch = batch;
    packed.seq = seq;
    packed.hidden = hidden;
    packed.routesPerBatch = routesPerBatch;
    packed.totalRoutes = totalRoutes;
    packed.topK = program.topK;
    packed.numExperts = program.numExperts;
    return packed;
}

__global__ void moe_gather_count_kernel(
        DeviceMoeGatherProgram program,
        int32_t* counts,
        int* errorFlag) {
    int64_t globalRoute = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (globalRoute >= program.totalRoutes)
        return;

    int64_t expert = cuda_kernel::load_int(program.topkIds, globalRoute);
    if (expert < 0 || expert >= program.numExperts) {
        atomicExch(errorFlag, 1);
        return;
    }

    int64_t batch = program.routesPerBatch == 0 ? 0 : globalRoute / program.routesPerBatch;
    atomicAdd(&counts[batch * program.numExperts + expert], 1);
}

__global__ void moe_gather_store_offsets_kernel(
        DeviceMoeGatherProgram program,
        const int32_t* offsets) {
    int64_t total = program.batch * (program.numExperts + 1);
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total)
        return;

    cuda_kernel::store_int(program.expertOffsets, linear, offsets[linear]);
}

__global__ void moe_gather_assign_kernel(
        DeviceMoeGatherProgram program,
        int32_t* cursors,
        int32_t* routeRows) {
    int64_t globalRoute = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (globalRoute >= program.totalRoutes)
        return;

    int64_t batch = globalRoute / program.routesPerBatch;
    int64_t routeInBatch = globalRoute - batch * program.routesPerBatch;
    int64_t token = routeInBatch / program.topK;
    int64_t expert = cuda_kernel::load_int(program.topkIds, globalRoute);
    int32_t row = atomicAdd(&cursors[batch * program.numExperts + expert], 1);

    routeRows[globalRoute] = row;
    int64_t metadataIndex = batch * program.routesPerBatch + row;
    cuda_kernel::store_float(
        program.packedWeights,
        metadataIndex,
        cuda_kernel::load_float(program.topkWeights, globalRoute));
    cuda_kernel::store_int(program.tokenIds, metadataIndex, token);
}

__global__ void moe_gather_copy_x_kernel(
        DeviceMoeGatherProgram program,
        const int32_t* routeRows) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = program.totalRoutes * program.hidden;
    if (linear >= total)
        return;

    int64_t globalRoute = linear / program.hidden;
    int64_t hiddenIndex = linear - globalRoute * program.hidden;
    int64_t batch = globalRoute / program.routesPerBatch;
    int64_t routeInBatch = globalRoute - batch * program.routesPerBatch;
    int64_t token = routeInBatch / program.topK;
    int64_t row = routeRows[globalRoute];

    int64_t xIndex = batch * program.seq * program.hidden + token * program.hidden + hiddenIndex;
    int64_t packedIndex =
        batch * program.routesPerBatch * program.hidden + row * program.hidden + hiddenIndex;
    cuda_kernel::store_float(
        program.packedX,
        packedIndex,
        cuda_kernel::load_float(program.x, xIndex));
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

    // Hot decode path: token ids and router expert ids are expected valid by construction.
    // Skipping the error-flag round trip avoids one cudaStreamSynchronize per gather.
    int* errorFlag = nullptr;

    int blocks = static_cast<int>(
        (program.outputNumel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    gather_kernel<<<blocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
        program,
        errorFlag);
    auto launched = cuda_check(cudaGetLastError(), "cuda gather launch");
    if (!launched)
        return make_error(launched.error());
    return {};
}

Result<void> launch_cuda_moe_gather(
        const CudaLaunchContext& context,
        const CudaMoeGatherProgram& program) {
    auto valid = validate_context(context, 3, 4, "moe_gather");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_moe_gather_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();

    int32_t* counts = nullptr;
    int32_t* cursors = nullptr;
    int32_t* routeRows = nullptr;
    int32_t* offsets = nullptr;
    int* errorFlag = nullptr;
    auto freeTemps = [&]() {
        if (counts) {
            (void)cuda_free_stream_ordered(counts, context.stream, "cudaFreeAsync moe_gather counts");
            counts = nullptr;
        }
        if (cursors) {
            (void)cuda_free_stream_ordered(cursors, context.stream, "cudaFreeAsync moe_gather cursors");
            cursors = nullptr;
        }
        if (routeRows) {
            (void)cuda_free_stream_ordered(routeRows, context.stream, "cudaFreeAsync moe_gather route rows");
            routeRows = nullptr;
        }
        if (offsets) {
            (void)cuda_free_stream_ordered(offsets, context.stream, "cudaFreeAsync moe_gather offsets");
            offsets = nullptr;
        }
        if (errorFlag) {
            (void)cuda_free_stream_ordered(errorFlag, context.stream, "cudaFreeAsync moe_gather error flag");
            errorFlag = nullptr;
        }
    };

    auto allocCounts = cuda_malloc_stream_ordered(
        &counts,
        static_cast<size_t>(launchProgram.batch * launchProgram.numExperts) * sizeof(int32_t),
        context.stream,
        "cudaMallocAsync moe_gather counts");
    if (!allocCounts)
        return make_error(allocCounts.error());
    auto allocCursors = cuda_malloc_stream_ordered(
        &cursors,
        static_cast<size_t>(launchProgram.batch * launchProgram.numExperts) * sizeof(int32_t),
        context.stream,
        "cudaMallocAsync moe_gather cursors");
    if (!allocCursors) {
        freeTemps();
        return make_error(allocCursors.error());
    }
    auto allocOffsets = cuda_malloc_stream_ordered(
        &offsets,
        static_cast<size_t>(launchProgram.batch * (launchProgram.numExperts + 1)) * sizeof(int32_t),
        context.stream,
        "cudaMallocAsync moe_gather offsets");
    if (!allocOffsets) {
        freeTemps();
        return make_error(allocOffsets.error());
    }
    if (launchProgram.totalRoutes != 0) {
        auto allocRouteRows = cuda_malloc_stream_ordered(
            &routeRows,
            static_cast<size_t>(launchProgram.totalRoutes) * sizeof(int32_t),
            context.stream,
            "cudaMallocAsync moe_gather route rows");
        if (!allocRouteRows) {
            freeTemps();
            return make_error(allocRouteRows.error());
        }
    }
    auto allocError = cuda_malloc_stream_ordered(
        &errorFlag,
        sizeof(int),
        context.stream,
        "cudaMallocAsync moe_gather error flag");
    if (!allocError) {
        freeTemps();
        return make_error(allocError.error());
    }

    auto clearCounts = cuda_check(
        cudaMemsetAsync(
            counts,
            0,
            static_cast<size_t>(launchProgram.batch * launchProgram.numExperts) * sizeof(int32_t),
            context.stream),
        "cudaMemsetAsync moe_gather counts");
    if (!clearCounts) {
        freeTemps();
        return make_error(clearCounts.error());
    }
    auto clearError = cuda_check(
        cudaMemsetAsync(errorFlag, 0, sizeof(int), context.stream),
        "cudaMemsetAsync moe_gather error flag");
    if (!clearError) {
        freeTemps();
        return make_error(clearError.error());
    }

    int routeBlocks = static_cast<int>(
        (launchProgram.totalRoutes + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    if (routeBlocks > 0) {
        moe_gather_count_kernel<<<routeBlocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
            launchProgram,
            counts,
            errorFlag);
        auto launched = cuda_check(cudaGetLastError(), "cuda moe_gather count launch");
        if (!launched) {
            freeTemps();
            return make_error(launched.error());
        }
    }

    std::vector<int32_t> hostCounts(
        static_cast<size_t>(launchProgram.batch * launchProgram.numExperts));
    int hostError = 0;
    auto copyCounts = cuda_check(
        cudaMemcpyAsync(
            hostCounts.data(),
            counts,
            hostCounts.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            context.stream),
        "cudaMemcpyAsync moe_gather counts");
    if (!copyCounts) {
        freeTemps();
        return make_error(copyCounts.error());
    }
    auto copyError = cuda_check(
        cudaMemcpyAsync(
            &hostError,
            errorFlag,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            context.stream),
        "cudaMemcpyAsync moe_gather error flag");
    if (!copyError) {
        freeTemps();
        return make_error(copyError.error());
    }
    auto counted = cuda_check(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize moe_gather count");
    if (!counted) {
        freeTemps();
        return make_error(counted.error());
    }
    if (hostError != 0) {
        freeTemps();
        return make_error("moe_gather topk expert id out of range");
    }

    std::vector<int32_t> hostOffsets(
        static_cast<size_t>(launchProgram.batch * (launchProgram.numExperts + 1)));
    std::vector<int32_t> hostCursors(
        static_cast<size_t>(launchProgram.batch * launchProgram.numExperts));
    for (int64_t batch = 0; batch < launchProgram.batch; ++batch) {
        int32_t row = 0;
        for (int64_t expert = 0; expert < launchProgram.numExperts; ++expert) {
            size_t countIndex = static_cast<size_t>(batch * launchProgram.numExperts + expert);
            size_t offsetIndex = static_cast<size_t>(batch * (launchProgram.numExperts + 1) + expert);
            hostOffsets[offsetIndex] = row;
            hostCursors[countIndex] = row;
            row += hostCounts[countIndex];
        }
        hostOffsets[static_cast<size_t>(batch * (launchProgram.numExperts + 1) + launchProgram.numExperts)] = row;
        if (row != launchProgram.routesPerBatch) {
            freeTemps();
            return make_error("cuda moe_gather routed row count mismatch");
        }
    }

    auto copyOffsets = cuda_check(
        cudaMemcpyAsync(
            offsets,
            hostOffsets.data(),
            hostOffsets.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            context.stream),
        "cudaMemcpyAsync moe_gather offsets");
    if (!copyOffsets) {
        freeTemps();
        return make_error(copyOffsets.error());
    }
    auto copyCursors = cuda_check(
        cudaMemcpyAsync(
            cursors,
            hostCursors.data(),
            hostCursors.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            context.stream),
        "cudaMemcpyAsync moe_gather cursors");
    if (!copyCursors) {
        freeTemps();
        return make_error(copyCursors.error());
    }

    int64_t offsetCount = launchProgram.batch * (launchProgram.numExperts + 1);
    int offsetBlocks = static_cast<int>(
        (offsetCount + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    moe_gather_store_offsets_kernel<<<
        offsetBlocks,
        cuda_kernel::kBlockSize,
        0,
        context.stream>>>(launchProgram, offsets);
    auto offsetLaunch = cuda_check(cudaGetLastError(), "cuda moe_gather offsets launch");
    if (!offsetLaunch) {
        freeTemps();
        return make_error(offsetLaunch.error());
    }

    if (routeBlocks > 0) {
        moe_gather_assign_kernel<<<routeBlocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
            launchProgram,
            cursors,
            routeRows);
        auto assignLaunch = cuda_check(cudaGetLastError(), "cuda moe_gather assign launch");
        if (!assignLaunch) {
            freeTemps();
            return make_error(assignLaunch.error());
        }

        if (launchProgram.totalRoutes > std::numeric_limits<int64_t>::max() / launchProgram.hidden) {
            freeTemps();
            return make_error("cuda moe_gather packed copy work exceeds int64 limit");
        }
        int64_t copyWork = launchProgram.totalRoutes * launchProgram.hidden;
        int64_t copyBlocks64 =
            (copyWork + cuda_kernel::kBlockSize - 1) / cuda_kernel::kBlockSize;
        if (copyBlocks64 > std::numeric_limits<int>::max()) {
            freeTemps();
            return make_error("cuda moe_gather packed copy grid exceeds launch limit");
        }

        moe_gather_copy_x_kernel<<<
            static_cast<int>(copyBlocks64),
            cuda_kernel::kBlockSize,
            0,
            context.stream>>>(launchProgram, routeRows);
        auto copyLaunch = cuda_check(cudaGetLastError(), "cuda moe_gather copy launch");
        if (!copyLaunch) {
            freeTemps();
            return make_error(copyLaunch.error());
        }
    }

    auto synced = cuda_check(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize moe_gather");
    if (!synced) {
        freeTemps();
        return make_error(synced.error());
    }

    freeTemps();
    return {};
}

} // namespace sandy::device
