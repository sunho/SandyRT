#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sandy::device {

namespace {

struct DeviceTopKProgram {
    cuda_kernel::TensorArg input;
    cuda_kernel::TensorArg values;
    cuda_kernel::TensorArg indices;
    int64_t rows = 0;
    int64_t axis = 0;
    int64_t k = 0;
};

constexpr int kTopKCapacity = 64;
// Policy choice only. The large-axis tile kernel is instantiated for several
// block sizes below; keep 256 as the initial Gemma decode default.
constexpr int kDefaultLargeTopKBlockThreads = 256;
constexpr int kTopKMergeThreads = 512;
constexpr int kTopKMergeWarps = kTopKMergeThreads / 32;
constexpr int64_t kLargeTopKAxisThreshold = 4096;

struct TopKCandidate {
    float value;
    int64_t index;
};

Result<void> validate_topk_float_tensor(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda topk ") + name + " unsupported dtype");
    return {};
}

Result<DeviceTopKProgram> pack_topk_program(
        const CudaLaunchContext& context,
        const CudaTopKProgram& program) {
    const auto& inputView = context.inputs[0];
    const auto& valuesView = context.outputs[0];
    const auto& indicesView = context.outputs[1];

    if (inputView.paged || valuesView.paged || indicesView.paged)
        return make_error("cuda topk does not support paged tensor operands");

    auto inputDtype = validate_topk_float_tensor(inputView, "input");
    if (!inputDtype)
        return make_error(inputDtype.error());
    auto valuesDtype = validate_topk_float_tensor(valuesView, "values");
    if (!valuesDtype)
        return make_error(valuesDtype.error());
    if (inputView.view.desc.dtype != valuesView.view.desc.dtype)
        return make_error("cuda topk values dtype mismatch");
    if (indicesView.view.desc.dtype != core::DType::I32 &&
        indicesView.view.desc.dtype != core::DType::I64) {
        return make_error("cuda topk indices output dtype mismatch");
    }

    int rank = inputView.view.desc.shape.rank();
    if (rank < 1)
        return make_error("cuda topk input must have rank >= 1");
    if (valuesView.view.desc.shape.rank() != rank ||
        indicesView.view.desc.shape.rank() != rank) {
        return make_error("cuda topk output rank mismatch");
    }

    int64_t axis = program.axis;
    if (axis < -rank || axis >= rank)
        return make_error("cuda topk axis out of range");
    axis = axis < 0 ? axis + rank : axis;
    if (axis != rank - 1)
        return make_error("cuda topk only supports last dimension");
    if (program.k <= 0 || program.k > kTopKCapacity)
        return make_error("cuda topk k must be between 1 and 64");

    for (int dim = 0; dim < rank; ++dim) {
        int64_t inputDim = inputView.view.desc.shape.dim(dim);
        int64_t valuesDim = valuesView.view.desc.shape.dim(dim);
        int64_t indicesDim = indicesView.view.desc.shape.dim(dim);
        if (inputDim < 0 || valuesDim < 0 || indicesDim < 0)
            return make_error("cuda topk requires static shapes");
        int64_t expected = dim == axis ? program.k : inputDim;
        if (valuesDim != expected || indicesDim != expected)
            return make_error("cuda topk output shape mismatch");
    }

    int64_t reduceDim = inputView.view.desc.shape.dim(static_cast<int>(axis));
    if (reduceDim <= 0 || program.k > reduceDim)
        return make_error("cuda topk invalid k or axis dimension");
    int64_t total = inputView.view.desc.shape.numel();
    if (total < 0)
        return make_error("cuda topk input must have static shape");

    auto input = cuda_kernel::pack_tensor_arg(inputView);
    if (!input)
        return make_error(input.error());
    auto values = cuda_kernel::pack_tensor_arg(valuesView);
    if (!values)
        return make_error(values.error());
    auto indices = cuda_kernel::pack_tensor_arg(indicesView);
    if (!indices)
        return make_error(indices.error());

    DeviceTopKProgram packed;
    packed.input = input.take();
    packed.values = values.take();
    packed.indices = indices.take();
    packed.rows = reduceDim == 0 ? 0 : total / reduceDim;
    packed.axis = reduceDim;
    packed.k = program.k;
    return packed;
}

__device__ bool better_topk_candidate(
        float lhsValue,
        int64_t lhsIndex,
        float rhsValue,
        int64_t rhsIndex) {
    if (lhsIndex < 0)
        return false;
    if (rhsIndex < 0)
        return true;
    if (lhsValue > rhsValue)
        return true;
    if (lhsValue < rhsValue)
        return false;
    return lhsIndex < rhsIndex;
}

__device__ TopKCandidate invalid_topk_candidate() {
    return TopKCandidate{-INFINITY, INT64_MAX};
}

__device__ TopKCandidate load_topk_candidate(
        DeviceTopKProgram program,
        int row,
        int64_t column) {
    if (column >= program.axis)
        return invalid_topk_candidate();
    int64_t linear = static_cast<int64_t>(row) * program.axis + column;
    return TopKCandidate{
        cuda_kernel::load_float(program.input, linear),
        column,
    };
}

__device__ bool better_topk_candidate(TopKCandidate lhs, TopKCandidate rhs) {
    return better_topk_candidate(lhs.value, lhs.index, rhs.value, rhs.index);
}

__device__ TopKCandidate shuffle_xor_topk(TopKCandidate candidate, int laneMask) {
    candidate.value = __shfl_xor_sync(0xffffffffu, candidate.value, laneMask);
    candidate.index = static_cast<int64_t>(__shfl_xor_sync(
        0xffffffffu,
        static_cast<long long>(candidate.index),
        laneMask));
    return candidate;
}

__device__ TopKCandidate select_topk_candidate(
        TopKCandidate candidate,
        TopKCandidate other,
        bool keepBetter) {
    bool otherBetter = better_topk_candidate(other, candidate);
    if ((keepBetter && otherBetter) ||
        (!keepBetter && better_topk_candidate(candidate, other))) {
        return other;
    }
    return candidate;
}

// Sort one candidate per lane. Intermediate groups alternate direction to
// construct bitonic sequences; the complete warp ends in the requested order.
__device__ TopKCandidate warp_sort_32(TopKCandidate candidate, bool descending) {
    int lane = threadIdx.x & 31;
    for (int size = 2; size <= 32; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            TopKCandidate other = shuffle_xor_topk(candidate, stride);
            bool lowerLane = (lane & stride) == 0;
            bool groupDescending = ((lane & size) == 0) ? descending : !descending;
            candidate = select_topk_candidate(
                candidate,
                other,
                lowerLane == groupDescending);
        }
    }
    return candidate;
}

// Sort 64 candidates distributed as two registers per lane, descending.
__device__ void warp_sort_64(TopKCandidate& first, TopKCandidate& second) {
    int lane = threadIdx.x & 31;
    first = warp_sort_32(first, true);
    second = warp_sort_32(second, false);

    TopKCandidate high = better_topk_candidate(first, second) ? first : second;
    TopKCandidate low = better_topk_candidate(first, second) ? second : first;
    first = high;
    second = low;

    // The distance-32 comparison above split the bitonic sequence into two
    // halves. Finish both descending 32-element bitonic merges in parallel.
    for (int stride = 16; stride > 0; stride >>= 1) {
        TopKCandidate otherFirst = shuffle_xor_topk(first, stride);
        TopKCandidate otherSecond = shuffle_xor_topk(second, stride);
        bool lowerLane = (lane & stride) == 0;
        first = select_topk_candidate(first, otherFirst, lowerLane);
        second = select_topk_candidate(second, otherSecond, lowerLane);
    }
}

// Return one rank of the descending merge of two sorted k-element lists.
__device__ TopKCandidate merged_topk_rank(
        const TopKCandidate* lhs,
        const TopKCandidate* rhs,
        int k,
        int rank) {
    int low = max(0, rank - k);
    int high = min(rank, k);
    while (low <= high) {
        int lhsCount = (low + high) >> 1;
        int rhsCount = rank - lhsCount;
        if (lhsCount > 0 && rhsCount < k &&
            better_topk_candidate(rhs[rhsCount], lhs[lhsCount - 1])) {
            high = lhsCount - 1;
            continue;
        }
        if (rhsCount > 0 && lhsCount < k &&
            better_topk_candidate(lhs[lhsCount], rhs[rhsCount - 1])) {
            low = lhsCount + 1;
            continue;
        }
        TopKCandidate lhsNext =
            lhsCount < k ? lhs[lhsCount] : invalid_topk_candidate();
        TopKCandidate rhsNext =
            rhsCount < k ? rhs[rhsCount] : invalid_topk_candidate();
        return better_topk_candidate(lhsNext, rhsNext) ? lhsNext : rhsNext;
    }
    return invalid_topk_candidate();
}

__device__ void order_topk_pair_descending(
        TopKCandidate& high,
        TopKCandidate& low) {
    if (better_topk_candidate(low, high)) {
        TopKCandidate temporary = high;
        high = low;
        low = temporary;
    }
}

// Merge two descending 64-element lists with a fixed 128-element bitonic
// network. Four logical positions live in each lane. The first two positions
// after the merge are the union's top 64 and are written to output.
__device__ void warp_merge_topk_64(
        const TopKCandidate* lhs,
        const TopKCandidate* rhs,
        TopKCandidate* output) {
    int lane = threadIdx.x & 31;
    TopKCandidate values[4] = {
        lhs[lane],
        lhs[lane + 32],
        rhs[63 - lane],
        rhs[31 - lane],
    };

    // Distances 64 and 32 pair registers owned by the same lane.
    order_topk_pair_descending(values[0], values[2]);
    order_topk_pair_descending(values[1], values[3]);
    order_topk_pair_descending(values[0], values[1]);
    order_topk_pair_descending(values[2], values[3]);

    // The remaining comparison partners live in other lanes.
    for (int stride = 16; stride > 0; stride >>= 1) {
        bool lowerLane = (lane & stride) == 0;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            TopKCandidate other = shuffle_xor_topk(values[item], stride);
            values[item] = select_topk_candidate(values[item], other, lowerLane);
        }
    }

    output[lane] = values[0];
    output[lane + 32] = values[1];
    __syncwarp();
}

__device__ void warp_merge_topk(
        const TopKCandidate* lhs,
        const TopKCandidate* rhs,
        TopKCandidate* output,
        int k) {
    if (k == kTopKCapacity) {
        warp_merge_topk_64(lhs, rhs, output);
        return;
    }
    int lane = threadIdx.x & 31;
    if (lane < k)
        output[lane] = merged_topk_rank(lhs, rhs, k, lane);
    if (lane + 32 < k)
        output[lane + 32] = merged_topk_rank(lhs, rhs, k, lane + 32);
    __syncwarp();
}

template<int WarpsPerBlock>
__global__ void topk_large_tile_kernel(
        DeviceTopKProgram program,
        TopKCandidate* partials,
        int tilesPerRow) {
    __shared__ TopKCandidate shared[2][WarpsPerBlock][kTopKCapacity];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.y;
    int tile = blockIdx.x;
    int64_t tileBase =
        static_cast<int64_t>(tile) * WarpsPerBlock * 64;
    int64_t warpBase = tileBase + static_cast<int64_t>(warp) * 64;

    TopKCandidate first = load_topk_candidate(program, row, warpBase + lane);
    TopKCandidate second = load_topk_candidate(program, row, warpBase + lane + 32);
    warp_sort_64(first, second);
    if (lane < program.k)
        shared[0][warp][lane] = first;
    if (lane + 32 < program.k)
        shared[0][warp][lane + 32] = second;
    __syncthreads();

    int sourceBank = 0;
    int destinationBank = 1;
    for (int active = WarpsPerBlock; active > 1; active >>= 1) {
        int mergeWarps = active >> 1;
        if (warp < mergeWarps) {
            warp_merge_topk(
                &shared[sourceBank][2 * warp][0],
                &shared[sourceBank][2 * warp + 1][0],
                &shared[destinationBank][warp][0],
                static_cast<int>(program.k));
        }
        __syncthreads();
        int temporary = sourceBank;
        sourceBank = destinationBank;
        destinationBank = temporary;
    }

    if (warp == 0) {
        int64_t list = static_cast<int64_t>(row) * tilesPerRow + tile;
        TopKCandidate* output = partials + list * kTopKCapacity;
        if (lane < program.k)
            output[lane] = shared[sourceBank][0][lane];
        if (lane + 32 < program.k)
            output[lane + 32] = shared[sourceBank][0][lane + 32];
    }
}

template<int WarpsPerBlock>
__global__ void topk_large_finalize_kernel(
        DeviceTopKProgram program,
        const TopKCandidate* partials,
        int listsPerRow) {
    __shared__ TopKCandidate shared[2][WarpsPerBlock][kTopKCapacity];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    int firstList = warp;
    int sourceBank = 0;
    int destinationBank = 1;

    if (firstList < listsPerRow) {
        const TopKCandidate* input =
            partials +
            (static_cast<int64_t>(row) * listsPerRow + firstList) * kTopKCapacity;
        if (lane < program.k)
            shared[0][warp][lane] = input[lane];
        if (lane + 32 < program.k)
            shared[0][warp][lane + 32] = input[lane + 32];
    } else {
        if (lane < program.k)
            shared[0][warp][lane] = invalid_topk_candidate();
        if (lane + 32 < program.k)
            shared[0][warp][lane + 32] = invalid_topk_candidate();
    }
    __syncwarp();

    for (int list = firstList + WarpsPerBlock;
         list < listsPerRow;
         list += WarpsPerBlock) {
        const TopKCandidate* input =
            partials +
            (static_cast<int64_t>(row) * listsPerRow + list) * kTopKCapacity;
        warp_merge_topk(
            &shared[sourceBank][warp][0],
            input,
            &shared[destinationBank][warp][0],
            static_cast<int>(program.k));
        int temporary = sourceBank;
        sourceBank = destinationBank;
        destinationBank = temporary;
    }

    if (sourceBank != 0) {
        if (lane < program.k)
            shared[0][warp][lane] = shared[sourceBank][warp][lane];
        if (lane + 32 < program.k)
            shared[0][warp][lane + 32] = shared[sourceBank][warp][lane + 32];
    }
    __syncthreads();

    sourceBank = 0;
    destinationBank = 1;
    for (int active = WarpsPerBlock; active > 1; active >>= 1) {
        int mergeWarps = active >> 1;
        if (warp < mergeWarps) {
            warp_merge_topk(
                &shared[sourceBank][2 * warp][0],
                &shared[sourceBank][2 * warp + 1][0],
                &shared[destinationBank][warp][0],
                static_cast<int>(program.k));
        }
        __syncthreads();
        int temporary = sourceBank;
        sourceBank = destinationBank;
        destinationBank = temporary;
    }

    if (warp == 0) {
        int64_t outputBase = static_cast<int64_t>(row) * program.k;
        if (lane < program.k) {
            TopKCandidate candidate = shared[sourceBank][0][lane];
            cuda_kernel::store_float(program.values, outputBase + lane, candidate.value);
            cuda_kernel::store_int(program.indices, outputBase + lane, candidate.index);
        }
        if (lane + 32 < program.k) {
            TopKCandidate candidate = shared[sourceBank][0][lane + 32];
            cuda_kernel::store_float(
                program.values,
                outputBase + lane + 32,
                candidate.value);
            cuda_kernel::store_int(
                program.indices,
                outputBase + lane + 32,
                candidate.index);
        }
    }
}

__global__ void topk_last_dim_kernel(DeviceTopKProgram program) {
    extern __shared__ unsigned char rawShared[];
    float* sharedValues = reinterpret_cast<float*>(rawShared);
    int64_t* sharedIndices =
        reinterpret_cast<int64_t*>(sharedValues + blockDim.x);
    int64_t* selectedIndices = sharedIndices + blockDim.x;

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= program.rows)
        return;

    for (int64_t rank = 0; rank < program.k; ++rank) {
        float localValue = -INFINITY;
        int64_t localIndex = -1;
        for (int64_t col = tid; col < program.axis; col += blockDim.x) {
            bool alreadySelected = false;
            for (int64_t selected = 0; selected < rank; ++selected) {
                if (selectedIndices[selected] == col) {
                    alreadySelected = true;
                    break;
                }
            }
            if (alreadySelected)
                continue;

            int64_t inputLinear = row * program.axis + col;
            float value = cuda_kernel::load_float(program.input, inputLinear);
            if (better_topk_candidate(value, col, localValue, localIndex)) {
                localValue = value;
                localIndex = col;
            }
        }

        sharedValues[tid] = localValue;
        sharedIndices[tid] = localIndex;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride &&
                better_topk_candidate(
                    sharedValues[tid + stride],
                    sharedIndices[tid + stride],
                    sharedValues[tid],
                    sharedIndices[tid])) {
                sharedValues[tid] = sharedValues[tid + stride];
                sharedIndices[tid] = sharedIndices[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            int64_t outputLinear = row * program.k + rank;
            selectedIndices[rank] = sharedIndices[0];
            cuda_kernel::store_float(program.values, outputLinear, sharedValues[0]);
            cuda_kernel::store_int(program.indices, outputLinear, sharedIndices[0]);
        }
        __syncthreads();
    }
}

template<int Threads>
Result<void> launch_large_topk_tiles(
        const CudaLaunchContext& context,
        DeviceTopKProgram program,
        TopKCandidate* partials,
        int tilesPerRow) {
    static_assert(Threads % 32 == 0);
    constexpr int warps = Threads / 32;
    dim3 grid(
        static_cast<unsigned>(tilesPerRow),
        static_cast<unsigned>(program.rows));
    topk_large_tile_kernel<warps><<<grid, Threads, 0, context.stream>>>(
        program,
        partials,
        tilesPerRow);
    return cuda_check(cudaGetLastError(), "cuda large topk tile launch");
}

Result<void> launch_large_topk(
        const CudaLaunchContext& context,
        DeviceTopKProgram program,
        int blockThreads) {
    if (blockThreads != 128 && blockThreads != 256 && blockThreads != 512)
        return make_error("cuda large topk block size must be 128, 256, or 512");

    int64_t elementsPerTile = static_cast<int64_t>(blockThreads) * 2;
    int64_t tiles64 = (program.axis + elementsPerTile - 1) / elementsPerTile;
    if (tiles64 <= 0 || tiles64 > std::numeric_limits<int>::max())
        return make_error("cuda large topk tile count exceeds grid limit");
    if (context.deviceProps &&
        (tiles64 > context.deviceProps->maxGridSize[0] ||
         program.rows > context.deviceProps->maxGridSize[1])) {
        return make_error("cuda large topk grid exceeds device limit");
    }

    size_t rows = static_cast<size_t>(program.rows);
    size_t tiles = static_cast<size_t>(tiles64);
    if (rows > std::numeric_limits<size_t>::max() / tiles ||
        rows * tiles > std::numeric_limits<size_t>::max() / kTopKCapacity ||
        rows * tiles * kTopKCapacity >
            std::numeric_limits<size_t>::max() / sizeof(TopKCandidate)) {
        return make_error("cuda large topk temporary size overflow");
    }
    size_t partialBytes =
        rows * tiles * kTopKCapacity * sizeof(TopKCandidate);
    TopKCandidate* partials = nullptr;
    auto allocated = cuda_malloc_stream_ordered(
        &partials,
        partialBytes,
        context.stream,
        "cudaMallocAsync large topk partials");
    if (!allocated)
        return make_error(allocated.error());

    auto freePartials = [&]() {
        if (partials) {
            (void)cuda_free_stream_ordered(
                partials,
                context.stream,
                "cudaFreeAsync large topk partials");
            partials = nullptr;
        }
    };

    Result<void> launched;
    int tilesPerRow = static_cast<int>(tiles64);
    switch (blockThreads) {
        case 128:
            launched = launch_large_topk_tiles<128>(
                context, program, partials, tilesPerRow);
            break;
        case 256:
            launched = launch_large_topk_tiles<256>(
                context, program, partials, tilesPerRow);
            break;
        case 512:
            launched = launch_large_topk_tiles<512>(
                context, program, partials, tilesPerRow);
            break;
    }
    if (!launched) {
        freePartials();
        return make_error(launched.error());
    }

    topk_large_finalize_kernel<kTopKMergeWarps>
        <<<static_cast<int>(program.rows), kTopKMergeThreads, 0, context.stream>>>(
            program,
            partials,
            tilesPerRow);
    auto finalized = cuda_check(
        cudaGetLastError(),
        "cuda large topk finalize launch");
    freePartials();
    if (!finalized)
        return make_error(finalized.error());
    return {};
}

} // namespace

Result<void> launch_cuda_topk(
        const CudaLaunchContext& context,
        const CudaTopKProgram& program) {
    auto valid = validate_context(context, 1, 2, "topk");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_topk_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.rows == 0)
        return {};
    if (launchProgram.rows > std::numeric_limits<int>::max())
        return make_error("cuda topk row count exceeds grid limit");

    if (launchProgram.axis >= kLargeTopKAxisThreshold && launchProgram.k > 1) {
        return launch_large_topk(
            context,
            launchProgram,
            kDefaultLargeTopKBlockThreads);
    }

    int blocks = static_cast<int>(launchProgram.rows);
    int threads = cuda_kernel::kBlockSize;
    size_t sharedBytes =
        static_cast<size_t>(threads) * sizeof(float) +
        static_cast<size_t>(threads + launchProgram.k) * sizeof(int64_t);
    topk_last_dim_kernel<<<blocks, threads, sharedBytes, context.stream>>>(launchProgram);
    return cuda_check(cudaGetLastError(), "cuda topk launch");
}

} // namespace sandy::device
