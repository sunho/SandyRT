#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "jit/CudaJitLaunchUtils.cuh"
#include "jit/CudaGatherJit.h"
#include "jit/templates/CudaJitGatherAbi.cuh"

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

__global__ void gather_kernel(
        DeviceGatherProgram program,
        ir::kernel_ir::OpId* validationFailure,
        ir::kernel_ir::OpId op) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.outputNumel)
        return;

    int64_t hiddenIndex = program.tableRank == 2 ? linear % program.hidden : 0;
    int64_t idLinear = program.tableRank == 2 ? linear / program.hidden : linear;
    int64_t idStorage = cuda_kernel::storage_index(program.ids, idLinear);
    int64_t tokenId = load_index_at_storage(program.ids, idStorage);
    if (tokenId < 0 || tokenId >= program.vocab) {
        cuda_kernel::record_validation_failure(validationFailure, op);
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
        ir::kernel_ir::OpId* validationFailure,
        ir::kernel_ir::OpId op) {
    int64_t globalRoute = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (globalRoute >= program.totalRoutes)
        return;

    int64_t expert = cuda_kernel::load_int(program.topkIds, globalRoute);
    if (expert < 0 || expert >= program.numExperts) {
        cuda_kernel::record_validation_failure(validationFailure, op);
        return;
    }

    int64_t batch = program.routesPerBatch == 0 ? 0 : globalRoute / program.routesPerBatch;
    atomicAdd(&counts[batch * program.numExperts + expert], 1);
}

__global__ void moe_gather_prefix_kernel(
        DeviceMoeGatherProgram program,
        const int32_t* counts,
        int32_t* cursors) {
    int64_t batch = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= program.batch)
        return;

    int32_t row = 0;
    int64_t countBase = batch * program.numExperts;
    int64_t offsetBase = batch * (program.numExperts + 1);
    for (int64_t expert = 0; expert < program.numExperts; ++expert) {
        cuda_kernel::store_int(program.expertOffsets, offsetBase + expert, row);
        cursors[countBase + expert] = row;
        row += counts[countBase + expert];
    }
    cuda_kernel::store_int(
        program.expertOffsets,
        offsetBase + program.numExperts,
        row);
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
    if (expert < 0 || expert >= program.numExperts) {
        routeRows[globalRoute] = -1;
        return;
    }
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
    if (row < 0)
        return;

    int64_t xIndex = batch * program.seq * program.hidden + token * program.hidden + hiddenIndex;
    int64_t packedIndex =
        batch * program.routesPerBatch * program.hidden + row * program.hidden + hiddenIndex;
    cuda_kernel::store_float(
        program.packedX,
        packedIndex,
        cuda_kernel::load_float(program.x, xIndex));
}

} // namespace

Result<void> launch_cuda_gather(
        const CudaLaunchContext& context,
        const CudaGatherProgram& compiled) {
    auto valid = validate_context(context, 2, 1, "gather");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_gather_program(context);
    if (!packed)
        return make_error(packed.error());
    auto program = packed.take();
    if (program.outputNumel == 0)
        return {};

    int blocks = static_cast<int>(
        (program.outputNumel + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    if (compiled.jitVariants) {
        if (!context.jitCache)
            return make_error("cuda gather JIT cache is null");
        const int accesses[] = {
            jit_access_kind(program.ids.access),
            jit_access_kind(program.table.access),
            jit_access_kind(program.output.access),
        };
        auto jit = compiled.jitVariants->getOrCompile(
            cudaJitAccessKey(accesses),
            [&] {
                return compileCudaGatherJit(
                    context.cudaDevice,
                    *context.jitCache,
                    compiled.idsDtype,
                    compiled.valueDtype,
                    compiled.tableRank,
                    accesses[0],
                    accesses[1],
                    accesses[2]);
            });
        if (!jit) {
            if (!compiled.jitFallbackOnError)
                return make_error(jit.error());
        } else {
            auto ids = pack_jit_tensor_arg(program.ids);
            if (!ids)
                return make_error(ids.error());
            auto table = pack_jit_tensor_arg(program.table);
            if (!table)
                return make_error(table.error());
            auto output = pack_jit_tensor_arg(program.output);
            if (!output)
                return make_error(output.error());
            SandyGatherParams params{
                ids.take(),
                table.take(),
                output.take(),
                program.vocab,
                program.hidden,
                context.validationFailure,
                context.op,
            };
            void* arguments[] = {&params};
            return (*jit)->launch(
                dim3(static_cast<unsigned>(blocks)),
                dim3(cuda_kernel::kBlockSize),
                0,
                context.stream,
                arguments);
        }
    }
    gather_kernel<<<blocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
        program,
        context.validationFailure,
        context.op);
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
    int routeBlocks = static_cast<int>(
        (launchProgram.totalRoutes + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    if (routeBlocks > 0) {
        moe_gather_count_kernel<<<routeBlocks, cuda_kernel::kBlockSize, 0, context.stream>>>(
            launchProgram,
            counts,
            context.validationFailure,
            context.op);
        auto launched = cuda_check(cudaGetLastError(), "cuda moe_gather count launch");
        if (!launched) {
            freeTemps();
            return make_error(launched.error());
        }
    }

    int prefixBlocks = static_cast<int>(
        (launchProgram.batch + cuda_kernel::kBlockSize - 1) /
        cuda_kernel::kBlockSize);
    moe_gather_prefix_kernel<<<
        prefixBlocks,
        cuda_kernel::kBlockSize,
        0,
        context.stream>>>(launchProgram, counts, cursors);
    auto prefixLaunch = cuda_check(cudaGetLastError(), "cuda moe_gather prefix launch");
    if (!prefixLaunch) {
        freeTemps();
        return make_error(prefixLaunch.error());
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

    freeTemps();
    return {};
}

} // namespace sandy::device
