#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "jit/CudaJitLaunchUtils.cuh"
#include "jit/CudaRoPEJit.h"
#include "jit/templates/CudaJitRoPEAbi.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sandy::device {

namespace {

struct DeviceRoPEProgram {
    cuda_kernel::TensorArg x;
    cuda_kernel::TensorArg positionIds;
    cuda_kernel::TensorArg output;
    int64_t seq = 0;
    int64_t dim = 0;
    int64_t rotaryDim = 0;
    int64_t vectors = 0;
    int64_t positionCount = 0;
    float theta = 10000.0f;
    bool splitHalf = false;
    bool hasPositionIds = false;
};

Result<void> validate_rope_tensor_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda rope ") + name + " unsupported dtype");
    return {};
}

Result<DeviceRoPEProgram> pack_rope_program(
        const CudaLaunchContext& context,
        const CudaRoPEProgram& program) {
    if (context.inputs.size() != 1 && context.inputs.size() != 2)
        return make_error("cuda rope expects 1 or 2 inputs");
    if (context.outputs.size() != 1)
        return make_error("cuda rope output arity mismatch");

    const auto& xView = context.inputs[0].view;
    const auto& outputView = context.outputs[0].view;

    auto xDtype = validate_rope_tensor_dtype(context.inputs[0], "input");
    if (!xDtype)
        return make_error(xDtype.error());
    auto outputDtype = validate_rope_tensor_dtype(context.outputs[0], "output");
    if (!outputDtype)
        return make_error(outputDtype.error());
    if (xView.desc.dtype != outputView.desc.dtype)
        return make_error("cuda rope output dtype mismatch");
    if (xView.desc.shape != outputView.desc.shape)
        return make_error("cuda rope output shape mismatch");

    int rank = xView.desc.shape.rank();
    if (rank < 2)
        return make_error("cuda rope input must have rank >= 2");
    if (program.theta <= 0.0)
        return make_error("cuda rope theta must be > 0");

    int64_t seq = xView.desc.shape.dim(rank - 2);
    int64_t dim = xView.desc.shape.dim(rank - 1);
    if (seq < 0 || dim < 0)
        return make_error("cuda rope input must have static sequence and last dimensions");
    if (dim <= 0 || dim % 2 != 0)
        return make_error("cuda rope last dimension must be positive and even");

    int64_t rotaryDim = program.rotaryDim < 0 ? dim : program.rotaryDim;
    if (rotaryDim <= 0 || rotaryDim % 2 != 0)
        return make_error("cuda rope rotary_dim must be positive and even");
    if (rotaryDim > dim)
        return make_error("cuda rope rotary_dim must be <= last dimension");

    int64_t total = xView.desc.shape.numel();
    if (total < 0)
        return make_error("cuda rope input must have static shape");

    DeviceRoPEProgram packed;
    packed.seq = seq;
    packed.dim = dim;
    packed.rotaryDim = rotaryDim;
    packed.vectors = total / dim;
    packed.theta = static_cast<float>(program.theta);
    packed.splitHalf = program.splitHalf;
    packed.hasPositionIds = context.inputs.size() == 2;

    if (packed.hasPositionIds) {
        const auto& positionView = context.inputs[1].view;
        if (positionView.desc.dtype != core::DType::I32 &&
            positionView.desc.dtype != core::DType::I64) {
            return make_error("cuda rope position_ids must be i32 or i64");
        }

        int64_t positionCount = positionView.desc.shape.numel();
        if (positionCount < 0)
            return make_error("cuda rope position_ids must have static shape");
        if (positionCount != 1 && positionCount != seq && positionCount != packed.vectors) {
            return make_error(
                "cuda rope position_ids numel must be 1, sequence length, or vector count");
        }
        packed.positionCount = positionCount;

        auto positionIds = cuda_kernel::pack_tensor_arg(context.inputs[1]);
        if (!positionIds)
            return make_error(positionIds.error());
        packed.positionIds = positionIds.take();
    }

    auto x = cuda_kernel::pack_tensor_arg(context.inputs[0]);
    if (!x)
        return make_error(x.error());
    auto output = cuda_kernel::pack_tensor_arg(context.outputs[0]);
    if (!output)
        return make_error(output.error());
    packed.x = x.take();
    packed.output = output.take();

    return packed;
}

__device__ int64_t load_index_at_storage(
        const cuda_kernel::TensorArg& tensor,
        int64_t index) {
    return cuda_kernel::load_int_at_storage(tensor, index);
}

__device__ bool rope_position(
        DeviceRoPEProgram program,
        int64_t vector,
        int* errorFlag,
        int64_t* positionOut) {
    int64_t seqPosition = vector % program.seq;
    int64_t position = seqPosition;
    if (!program.hasPositionIds) {
        *positionOut = position;
        return true;
    }

    int64_t positionIndex = 0;
    if (program.positionCount == 1) {
        position = load_index_at_storage(program.positionIds, 0) + seqPosition;
        if (position < 0) {
            atomicExch(errorFlag, 1);
            return false;
        }
        *positionOut = position;
        return true;
    }
    if (program.positionCount == program.seq) {
        positionIndex = seqPosition;
    } else if (program.positionCount == program.vectors) {
        positionIndex = vector;
    }

    int64_t storage = cuda_kernel::storage_index(program.positionIds, positionIndex);
    position = load_index_at_storage(program.positionIds, storage);
    if (position < 0) {
        atomicExch(errorFlag, 1);
        return false;
    }
    *positionOut = position;
    return true;
}

__device__ void store_rope_pair(
        DeviceRoPEProgram program,
        int64_t vector,
        int64_t firstCol,
        int64_t secondCol,
        float c,
        float s) {
    int64_t firstLinear = vector * program.dim + firstCol;
    int64_t secondLinear = vector * program.dim + secondCol;
    float first = cuda_kernel::load_float(program.x, firstLinear);
    float second = cuda_kernel::load_float(program.x, secondLinear);
    cuda_kernel::store_float(program.output, firstLinear, first * c - second * s);
    cuda_kernel::store_float(program.output, secondLinear, first * s + second * c);
}

__global__ void rope_kernel(DeviceRoPEProgram program, int* errorFlag) {
    int64_t rotatedPairs = program.rotaryDim / 2;
    int64_t skippedPairs = (program.dim - program.rotaryDim) / 2;
    int64_t workPerVector = rotatedPairs + skippedPairs;
    int64_t workLinear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t totalWork = program.vectors * workPerVector;
    if (workLinear >= totalWork)
        return;

    int64_t vector = workLinear / workPerVector;
    int64_t offset = workLinear % workPerVector;
    if (offset >= rotatedPairs) {
        int64_t skippedPair = offset - rotatedPairs;
        int64_t firstCol = program.rotaryDim + 2 * skippedPair;
        int64_t secondCol = firstCol + 1;
        if (program.splitHalf) {
            int64_t half = program.dim / 2;
            firstCol = rotatedPairs + skippedPair;
            secondCol = half + rotatedPairs + skippedPair;
        }

        int64_t firstLinear = vector * program.dim + firstCol;
        int64_t secondLinear = vector * program.dim + secondCol;
        cuda_kernel::store_float(
            program.output,
            firstLinear,
            cuda_kernel::load_float(program.x, firstLinear));
        cuda_kernel::store_float(
            program.output,
            secondLinear,
            cuda_kernel::load_float(program.x, secondLinear));
        return;
    }

    int64_t pair = offset;
    int64_t position = 0;
    if (!rope_position(program, vector, errorFlag, &position))
        return;

    if (program.splitHalf) {
        int64_t half = program.dim / 2;
        float exponent = static_cast<float>(2 * pair) / static_cast<float>(program.dim);
        float angle = static_cast<float>(position) / powf(program.theta, exponent);
        store_rope_pair(
            program,
            vector,
            pair,
            half + pair,
            cosf(angle),
            sinf(angle));
        return;
    }

    float exponent = static_cast<float>(2 * pair) / static_cast<float>(program.rotaryDim);
    float angle = static_cast<float>(position) / powf(program.theta, exponent);
    store_rope_pair(
        program,
        vector,
        2 * pair,
        2 * pair + 1,
        cosf(angle),
        sinf(angle));
}

} // namespace

Result<void> launch_cuda_rope(
        const CudaLaunchContext& context,
        const CudaRoPEProgram& program) {
    auto valid = validate_context(context, 1, 1, "rope");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_rope_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    int64_t rotatedPairs = launchProgram.rotaryDim / 2;
    int64_t skippedPairs = (launchProgram.dim - launchProgram.rotaryDim) / 2;
    int64_t workPerVector = rotatedPairs + skippedPairs;
    int64_t totalWork = launchProgram.vectors * workPerVector;
    if (totalWork == 0)
        return {};
    int64_t blockCount =
        (totalWork + cuda_kernel::kBlockSize - 1) / cuda_kernel::kBlockSize;
    if (blockCount > std::numeric_limits<int>::max())
        return make_error("cuda rope block count exceeds grid limit");

    int* errorFlag = nullptr;
    auto allocated = cuda_malloc_stream_ordered(
        &errorFlag,
        sizeof(int),
        context.stream,
        "cudaMallocAsync rope error flag");
    if (!allocated)
        return make_error(allocated.error());
    auto freeFlag = [&]() {
        if (errorFlag) {
            (void)cuda_free_stream_ordered(
                errorFlag,
                context.stream,
                "cudaFreeAsync rope error flag");
            errorFlag = nullptr;
        }
    };

    auto cleared = cuda_check(
        cudaMemsetAsync(errorFlag, 0, sizeof(int), context.stream),
        "cudaMemsetAsync rope error flag");
    if (!cleared) {
        freeFlag();
        return make_error(cleared.error());
    }

    Result<void> launched;
    bool usedJit = false;
    if (program.jitVariants) {
        const int accesses[] = {
            jit_access_kind(launchProgram.x.access),
            launchProgram.hasPositionIds
                ? jit_access_kind(launchProgram.positionIds.access)
                : SANDY_JIT_CONTIGUOUS,
            jit_access_kind(launchProgram.output.access),
        };
        auto jit = program.jitVariants->getOrCompile(
            cudaJitAccessKey(accesses), [&] {
                return compileCudaRoPEJit(
                    context.cudaDevice, *context.jitCache, program.dtype,
                    program.splitHalf, program.hasPositions, program.positionDtype,
                    accesses[0], accesses[1], accesses[2]);
            });
        if (!jit && !program.jitFallbackOnError) {
            freeFlag();
            return make_error(jit.error());
        }
        if (jit) {
            usedJit = true;
            auto x = pack_jit_tensor_arg(launchProgram.x);
            auto output = pack_jit_tensor_arg(launchProgram.output);
            if (!x || !output) { freeFlag(); return make_error(!x ? x.error() : output.error()); }
            SandyRoPEParams params{};
            params.input = x.take();
            params.output = output.take();
            if (launchProgram.hasPositionIds) {
                auto positions = pack_jit_tensor_arg(launchProgram.positionIds);
                if (!positions) { freeFlag(); return make_error(positions.error()); }
                params.positions = positions.take();
            }
            params.seq = launchProgram.seq; params.dim = launchProgram.dim;
            params.rotaryDim = launchProgram.rotaryDim; params.vectors = launchProgram.vectors;
            params.positionCount = launchProgram.positionCount;
            params.theta = launchProgram.theta; params.errorFlag = errorFlag;
            void* arguments[] = {&params};
            launched = (*jit)->launch(
                dim3(static_cast<unsigned>(blockCount)), dim3(cuda_kernel::kBlockSize),
                0, context.stream, arguments);
        }
    }
    if (!usedJit) {
        rope_kernel<<<static_cast<int>(blockCount), cuda_kernel::kBlockSize, 0, context.stream>>>(
            launchProgram, errorFlag);
        launched = cuda_check(cudaGetLastError(), "cuda rope launch");
    }
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
        "cudaMemcpyAsync rope error flag");
    if (!copied) {
        freeFlag();
        return make_error(copied.error());
    }

    auto synced = cuda_check(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize rope");
    if (!synced) {
        freeFlag();
        return make_error(synced.error());
    }

    freeFlag();
    if (hostError != 0)
        return make_error("cuda rope position_ids must be non-negative");
    return {};
}

} // namespace sandy::device
