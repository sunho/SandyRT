#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <cstdint>
#include <limits>
#include <string>

namespace sandy::device {

namespace {

struct DeviceMoeScatterSumProgram {
    cuda_kernel::TensorArg packedOut;
    cuda_kernel::TensorArg packedWeights;
    cuda_kernel::TensorArg tokenIds;
    cuda_kernel::TensorArg output;
    int64_t batch = 1;
    int64_t rows = 0;
    int64_t seq = 0;
    int64_t hidden = 0;
    int64_t outputNumel = 0;
    int64_t scatterWork = 0;
};

Result<void> validate_moe_scatter_sum_contiguous(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (buffer.paged)
        return make_error(std::string("cuda moe_scatter_sum ") + name + " cannot be paged");
    if (!cuda_kernel::is_contiguous(buffer.view))
        return make_error(std::string("cuda moe_scatter_sum ") + name + " must be contiguous");
    return {};
}

Result<DeviceMoeScatterSumProgram> pack_moe_scatter_sum_program(
        const CudaLaunchContext& context) {
    const auto& packedOutView = context.inputs[0];
    const auto& packedWeightsView = context.inputs[1];
    const auto& tokenIdsView = context.inputs[2];
    const auto& referenceView = context.inputs[3];
    const auto& outputView = context.outputs[0];

    const CudaDeviceBufferView* allBuffers[] = {
        &packedOutView,
        &packedWeightsView,
        &tokenIdsView,
        &referenceView,
        &outputView,
    };
    const char* names[] = {
        "packed_out",
        "packed_weights",
        "token_ids",
        "reference",
        "output",
    };
    for (int i = 0; i < 5; ++i) {
        auto contiguous = validate_moe_scatter_sum_contiguous(*allBuffers[i], names[i]);
        if (!contiguous)
            return make_error(contiguous.error());
    }

    if (!is_float_compute_dtype(packedOutView.view.desc.dtype) ||
        !is_float_compute_dtype(packedWeightsView.view.desc.dtype)) {
        return make_error("cuda moe_scatter_sum unsupported floating dtype");
    }
    if (packedOutView.view.desc.dtype != packedWeightsView.view.desc.dtype ||
        packedOutView.view.desc.dtype != outputView.view.desc.dtype ||
        packedOutView.view.desc.dtype != referenceView.view.desc.dtype) {
        return make_error("cuda moe_scatter_sum dtype mismatch");
    }
    if (tokenIdsView.view.desc.dtype != core::DType::I32 &&
        tokenIdsView.view.desc.dtype != core::DType::I64) {
        return make_error("cuda moe_scatter_sum token_ids must be i32 or i64");
    }

    int packedRank = packedOutView.view.desc.shape.rank();
    int outRank = outputView.view.desc.shape.rank();
    if ((packedRank != 2 && packedRank != 3) ||
        packedWeightsView.view.desc.shape.rank() != packedRank - 1 ||
        tokenIdsView.view.desc.shape.rank() != packedRank - 1 ||
        (outRank != 2 && outRank != 3) ||
        packedRank != outRank ||
        referenceView.view.desc.shape.rank() != outRank) {
        return make_error(
            "cuda moe_scatter_sum expects ranks [N,H], [N], [N], [T,H] or [B,N,H], [B,N], [B,N], [B,T,H]");
    }
    if (referenceView.view.desc.shape != outputView.view.desc.shape)
        return make_error("cuda moe_scatter_sum reference shape mismatch");

    int64_t rows = packedOutView.view.desc.shape.dim(packedRank - 2);
    int64_t hidden = packedOutView.view.desc.shape.dim(packedRank - 1);
    int64_t batch = outRank == 3 ? outputView.view.desc.shape.dim(0) : 1;
    int64_t seq = outRank == 3 ? outputView.view.desc.shape.dim(1) : outputView.view.desc.shape.dim(0);
    if (rows < 0 || hidden <= 0 || batch <= 0 || seq < 0)
        return make_error("cuda moe_scatter_sum expects static shapes");

    if ((packedRank == 3 &&
         (packedOutView.view.desc.shape.dim(0) != batch ||
          packedWeightsView.view.desc.shape.dim(0) != batch ||
          tokenIdsView.view.desc.shape.dim(0) != batch)) ||
        packedWeightsView.view.desc.shape.dim(packedRank - 2) != rows ||
        tokenIdsView.view.desc.shape.dim(packedRank - 2) != rows ||
        outputView.view.desc.shape.dim(outRank - 1) != hidden) {
        return make_error("cuda moe_scatter_sum shape mismatch");
    }
    if (batch > 0 && rows > std::numeric_limits<int64_t>::max() / batch)
        return make_error("cuda moe_scatter_sum row count exceeds int64 limit");
    int64_t totalRows = batch * rows;
    if (hidden > 0 && totalRows > std::numeric_limits<int64_t>::max() / hidden)
        return make_error("cuda moe_scatter_sum work exceeds int64 limit");
    int64_t scatterWork = totalRows * hidden;
    int64_t outputNumel = outputView.view.desc.shape.numel();
    if (outputNumel < 0)
        return make_error("cuda moe_scatter_sum output must have static shape");

    auto packedOut = cuda_kernel::pack_tensor_arg(packedOutView);
    if (!packedOut)
        return make_error(packedOut.error());
    auto packedWeights = cuda_kernel::pack_tensor_arg(packedWeightsView);
    if (!packedWeights)
        return make_error(packedWeights.error());
    auto tokenIds = cuda_kernel::pack_tensor_arg(tokenIdsView);
    if (!tokenIds)
        return make_error(tokenIds.error());
    auto output = cuda_kernel::pack_tensor_arg(outputView);
    if (!output)
        return make_error(output.error());

    DeviceMoeScatterSumProgram program;
    program.packedOut = packedOut.take();
    program.packedWeights = packedWeights.take();
    program.tokenIds = tokenIds.take();
    program.output = output.take();
    program.batch = batch;
    program.rows = rows;
    program.seq = seq;
    program.hidden = hidden;
    program.outputNumel = outputNumel;
    program.scatterWork = scatterWork;
    return program;
}

Result<int> block_count(int64_t work, const char* name) {
    int64_t blocks = (work + cuda_kernel::kBlockSize - 1) / cuda_kernel::kBlockSize;
    if (blocks > std::numeric_limits<int>::max())
        return make_error(std::string("cuda moe_scatter_sum ") + name + " grid exceeds launch limit");
    return static_cast<int>(blocks);
}

__global__ void moe_scatter_sum_zero_output_kernel(DeviceMoeScatterSumProgram program) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.outputNumel)
        return;
    cuda_kernel::store_float(program.output, linear, 0.0f);
}

__global__ void moe_scatter_sum_f32_kernel(
        DeviceMoeScatterSumProgram program,
        int* errorFlag) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.scatterWork)
        return;

    int64_t h = linear % program.hidden;
    int64_t routeLinear = linear / program.hidden;
    int64_t batch = routeLinear / program.rows;
    int64_t row = routeLinear - batch * program.rows;
    int64_t metadataIndex = batch * program.rows + row;
    int64_t token = cuda_kernel::load_int(program.tokenIds, metadataIndex);
    if (token < 0 || token >= program.seq) {
        atomicExch(errorFlag, 1);
        return;
    }

    float weight = cuda_kernel::load_float(program.packedWeights, metadataIndex);
    float value = cuda_kernel::load_float(program.packedOut, linear);
    int64_t outputLinear = batch * program.seq * program.hidden + token * program.hidden + h;
    int64_t outputStorage = cuda_kernel::storage_index(program.output, outputLinear);
    atomicAdd(static_cast<float*>(program.output.data) + outputStorage, weight * value);
}

__global__ void moe_scatter_sum_to_float_kernel(
        DeviceMoeScatterSumProgram program,
        float* accum,
        int* errorFlag) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.scatterWork)
        return;

    int64_t h = linear % program.hidden;
    int64_t routeLinear = linear / program.hidden;
    int64_t batch = routeLinear / program.rows;
    int64_t row = routeLinear - batch * program.rows;
    int64_t metadataIndex = batch * program.rows + row;
    int64_t token = cuda_kernel::load_int(program.tokenIds, metadataIndex);
    if (token < 0 || token >= program.seq) {
        atomicExch(errorFlag, 1);
        return;
    }

    float weight = cuda_kernel::load_float(program.packedWeights, metadataIndex);
    float value = cuda_kernel::load_float(program.packedOut, linear);
    int64_t outputLinear = batch * program.seq * program.hidden + token * program.hidden + h;
    atomicAdd(accum + outputLinear, weight * value);
}

__global__ void moe_scatter_sum_store_accum_kernel(
        DeviceMoeScatterSumProgram program,
        const float* accum) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= program.outputNumel)
        return;
    cuda_kernel::store_float(program.output, linear, accum[linear]);
}

Result<int> copy_error_and_sync(
        const CudaLaunchContext& context,
        int* errorFlag,
        const char* name) {
    int hostError = 0;
    auto copied = cuda_check(
        cudaMemcpyAsync(
            &hostError,
            errorFlag,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            context.stream),
        std::string("cudaMemcpyAsync moe_scatter_sum ") + name + " error flag");
    if (!copied)
        return make_error(copied.error());
    auto synced = cuda_check(
        cudaStreamSynchronize(context.stream),
        std::string("cudaStreamSynchronize moe_scatter_sum ") + name);
    if (!synced)
        return make_error(synced.error());
    return hostError;
}

} // namespace

Result<void> launch_cuda_moe_scatter_sum(const CudaLaunchContext& context) {
    auto valid = validate_context(context, 4, 1, "moe_scatter_sum");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_moe_scatter_sum_program(context);
    if (!packed)
        return make_error(packed.error());
    auto program = packed.take();

    int* errorFlag = nullptr;
    float* accum = nullptr;
    auto freeTemps = [&]() {
        if (accum) {
            (void)cuda_free_stream_ordered(
                accum,
                context.stream,
                "cudaFreeAsync moe_scatter_sum accum");
            accum = nullptr;
        }
        if (errorFlag) {
            (void)cuda_free_stream_ordered(
                errorFlag,
                context.stream,
                "cudaFreeAsync moe_scatter_sum error flag");
            errorFlag = nullptr;
        }
    };

    auto allocError = cuda_malloc_stream_ordered(
        &errorFlag,
        sizeof(int),
        context.stream,
        "cudaMallocAsync moe_scatter_sum error flag");
    if (!allocError)
        return make_error(allocError.error());
    auto clearError = cuda_check(
        cudaMemsetAsync(errorFlag, 0, sizeof(int), context.stream),
        "cudaMemsetAsync moe_scatter_sum error flag");
    if (!clearError) {
        freeTemps();
        return make_error(clearError.error());
    }

    auto outputBlocks = block_count(program.outputNumel, "output");
    if (!outputBlocks) {
        freeTemps();
        return make_error(outputBlocks.error());
    }
    auto scatterBlocks = block_count(program.scatterWork, "scatter");
    if (!scatterBlocks) {
        freeTemps();
        return make_error(scatterBlocks.error());
    }

    if (program.output.dtype == core::DType::F32) {
        if (program.outputNumel > 0) {
            moe_scatter_sum_zero_output_kernel<<<
                outputBlocks.take(),
                cuda_kernel::kBlockSize,
                0,
                context.stream>>>(program);
            auto zeroLaunch = cuda_check(cudaGetLastError(), "cuda moe_scatter_sum zero launch");
            if (!zeroLaunch) {
                freeTemps();
                return make_error(zeroLaunch.error());
            }
        }

        if (program.scatterWork > 0) {
            moe_scatter_sum_f32_kernel<<<
                scatterBlocks.take(),
                cuda_kernel::kBlockSize,
                0,
                context.stream>>>(program, errorFlag);
            auto scatterLaunch = cuda_check(cudaGetLastError(), "cuda moe_scatter_sum f32 launch");
            if (!scatterLaunch) {
                freeTemps();
                return make_error(scatterLaunch.error());
            }
        }

        auto hostError = copy_error_and_sync(context, errorFlag, "f32");
        if (!hostError) {
            freeTemps();
            return make_error(hostError.error());
        }
        freeTemps();
        if (hostError.take() != 0)
            return make_error("moe_scatter_sum token id out of range");
        return {};
    }

    if (program.outputNumel > 0) {
        auto allocAccum = cuda_malloc_stream_ordered(
            &accum,
            static_cast<size_t>(program.outputNumel) * sizeof(float),
            context.stream,
            "cudaMallocAsync moe_scatter_sum accum");
        if (!allocAccum) {
            freeTemps();
            return make_error(allocAccum.error());
        }
        auto clearAccum = cuda_check(
            cudaMemsetAsync(
                accum,
                0,
                static_cast<size_t>(program.outputNumel) * sizeof(float),
                context.stream),
            "cudaMemsetAsync moe_scatter_sum accum");
        if (!clearAccum) {
            freeTemps();
            return make_error(clearAccum.error());
        }
    }

    if (program.scatterWork > 0) {
        moe_scatter_sum_to_float_kernel<<<
            scatterBlocks.take(),
            cuda_kernel::kBlockSize,
            0,
            context.stream>>>(program, accum, errorFlag);
        auto scatterLaunch = cuda_check(cudaGetLastError(), "cuda moe_scatter_sum bf16 accum launch");
        if (!scatterLaunch) {
            freeTemps();
            return make_error(scatterLaunch.error());
        }
    }

    auto hostError = copy_error_and_sync(context, errorFlag, "bf16");
    if (!hostError) {
        freeTemps();
        return make_error(hostError.error());
    }
    if (hostError.take() != 0) {
        freeTemps();
        return make_error("moe_scatter_sum token id out of range");
    }

    if (program.outputNumel > 0) {
        moe_scatter_sum_store_accum_kernel<<<
            outputBlocks.take(),
            cuda_kernel::kBlockSize,
            0,
            context.stream>>>(program, accum);
        auto storeLaunch = cuda_check(cudaGetLastError(), "cuda moe_scatter_sum store launch");
        if (!storeLaunch) {
            freeTemps();
            return make_error(storeLaunch.error());
        }
    }

    auto synced = cuda_check(
        cudaStreamSynchronize(context.stream),
        "cudaStreamSynchronize moe_scatter_sum store");
    if (!synced) {
        freeTemps();
        return make_error(synced.error());
    }

    freeTemps();
    return {};
}

} // namespace sandy::device
