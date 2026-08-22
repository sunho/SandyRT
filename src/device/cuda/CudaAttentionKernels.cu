#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sandy::device {

namespace {

struct DeviceAttentionProgram {
    cuda_kernel::TensorArg q;
    cuda_kernel::TensorArg k;
    cuda_kernel::TensorArg v;
    cuda_kernel::TensorArg output;
    cuda_kernel::TensorArg positionOffsets;
    bool hasPositionOffsets = false;

    int rank = 4;
    int64_t batch = 1;
    int64_t qHeads = 0;
    int64_t kvHeads = 0;
    int64_t tq = 0;
    int64_t tk = 0;
    int64_t headDim = 0;
    int64_t window = 0;
    float scale = 1.0f;
};

Result<void> validate_attention_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda attention ") + name + " unsupported dtype");
    return {};
}

int64_t ceil_div_i64(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

__device__ float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset);
    return __shfl_sync(mask, value, 0);
}

__device__ int64_t device_max_i64(int64_t lhs, int64_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ int64_t device_min_i64(int64_t lhs, int64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ int64_t load_index(const cuda_kernel::TensorArg& tensor, int64_t linear) {
    return cuda_kernel::load_int(tensor, linear);
}

__device__ int64_t q_linear_index(
        const DeviceAttentionProgram& program,
        int64_t batch,
        int64_t head,
        int64_t query,
        int64_t dim) {
    if (program.rank == 4)
        return ((batch * program.qHeads + head) * program.tq + query) *
                   program.headDim +
               dim;
    return (head * program.tq + query) * program.headDim + dim;
}

__device__ int64_t kv_linear_index(
        const DeviceAttentionProgram& program,
        int64_t batch,
        int64_t head,
        int64_t key,
        int64_t dim) {
    if (program.rank == 4)
        return ((batch * program.kvHeads + head) * program.tk + key) *
                   program.headDim +
               dim;
    return (head * program.tk + key) * program.headDim + dim;
}

template <int HeadDim, int QueriesPerBlock, int KeyValueTileSize>
__global__ void flash_attention_prefill_kernel(DeviceAttentionProgram program) {
    constexpr int kLaneValues = (HeadDim + 31) / 32;

    extern __shared__ float shared[];
    float* kTile = shared;
    float* vTile = kTile + KeyValueTileSize * HeadDim;
    float* scores = vTile + KeyValueTileSize * HeadDim;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int64_t queryBlock = blockIdx.x;
    int64_t qHead = blockIdx.y;
    int64_t batch = blockIdx.z;
    int64_t query = queryBlock * QueriesPerBlock + warp;

    int64_t headsPerKv = program.qHeads / program.kvHeads;
    int64_t kvHead = qHead / headsPerKv;

    int64_t basePosition = 0;
    if (program.hasPositionOffsets)
        basePosition = load_index(program.positionOffsets, program.rank == 4 ? batch : 0);

    int64_t queryPosition = basePosition + query;
    int64_t minKey = program.window > 0
        ? device_max_i64(0, queryPosition + 1 - program.window)
        : 0;
    int64_t maxKey = queryPosition;

    float runningMax = -INFINITY;
    float runningSum = 0.0f;
    float output[kLaneValues];
    #pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
        output[i] = 0.0f;

    bool validQuery = query < program.tq;
    float qValues[kLaneValues];
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        qValues[frag] = validQuery && dim < HeadDim
            ? cuda_kernel::load_float(
                  program.q,
                  q_linear_index(program, batch, qHead, query, dim))
            : 0.0f;
    }

    int64_t blockFirstQuery = queryBlock * QueriesPerBlock;
    int64_t blockLastQuery = device_min_i64(
        blockFirstQuery + QueriesPerBlock - 1,
        program.tq - 1);
    int64_t blockMinKey = program.window > 0
        ? device_max_i64(0, basePosition + blockFirstQuery + 1 - program.window)
        : 0;
    int64_t blockMaxKey = basePosition + blockLastQuery;

    int64_t firstKeyTile = (blockMinKey / KeyValueTileSize) * KeyValueTileSize;
    for (int64_t keyTileStart = firstKeyTile;
         keyTileStart < program.tk && keyTileStart <= blockMaxKey;
         keyTileStart += KeyValueTileSize) {
        int tileKeys = static_cast<int>(device_min_i64(
            KeyValueTileSize,
            program.tk - keyTileStart));

        for (int idx = threadIdx.x; idx < tileKeys * HeadDim; idx += blockDim.x) {
            int keyOffset = idx / HeadDim;
            int dim = idx - keyOffset * HeadDim;
            int64_t key = keyTileStart + keyOffset;
            kTile[idx] = cuda_kernel::load_float(
                program.k,
                kv_linear_index(program, batch, kvHead, key, dim));
            vTile[idx] = cuda_kernel::load_float(
                program.v,
                kv_linear_index(program, batch, kvHead, key, dim));
        }
        __syncthreads();

        if (validQuery) {
            float tileMax = -INFINITY;
            for (int keyOffset = 0; keyOffset < tileKeys; ++keyOffset) {
                int64_t key = keyTileStart + keyOffset;
                bool visible = key <= maxKey && key >= minKey;

                float partial = 0.0f;
                #pragma unroll
                for (int frag = 0; frag < kLaneValues; ++frag) {
                    int dim = lane + frag * 32;
                    if (dim >= HeadDim)
                        continue;
                    partial += qValues[frag] * kTile[keyOffset * HeadDim + dim];
                }

                float score = warp_reduce_sum(partial) * program.scale;
                if (!visible)
                    score = -INFINITY;
                if (lane == 0)
                    scores[warp * KeyValueTileSize + keyOffset] = score;
                tileMax = fmaxf(tileMax, score);
            }

            if (!(isinf(tileMax) && tileMax < 0.0f)) {
                float nextMax = fmaxf(runningMax, tileMax);
                float oldScale = isinf(runningMax) && runningMax < 0.0f
                    ? 0.0f
                    : expf(runningMax - nextMax);
                float tileScale = expf(tileMax - nextMax);

                float localTileSum = 0.0f;
                for (int keyOffset = lane; keyOffset < tileKeys; keyOffset += 32) {
                    float score = scores[warp * KeyValueTileSize + keyOffset];
                    float probability = expf(score - tileMax);
                    scores[warp * KeyValueTileSize + keyOffset] = probability;
                    localTileSum += probability;
                }
                float tileSum = warp_reduce_sum(localTileSum);
                __syncwarp();

                #pragma unroll
                for (int frag = 0; frag < kLaneValues; ++frag) {
                    int dim = lane + frag * 32;
                    if (dim >= HeadDim)
                        continue;
                    float tileOutput = 0.0f;
                    for (int keyOffset = 0; keyOffset < tileKeys; ++keyOffset) {
                        float probability = scores[warp * KeyValueTileSize + keyOffset];
                        tileOutput +=
                            probability * vTile[keyOffset * HeadDim + dim];
                    }
                    output[frag] = oldScale * output[frag] + tileScale * tileOutput;
                }

                runningSum = oldScale * runningSum + tileScale * tileSum;
                runningMax = nextMax;
            }
        }

        __syncthreads();
    }

    if (validQuery) {
        float invSum = runningSum == 0.0f ? 0.0f : 1.0f / runningSum;
        #pragma unroll
        for (int frag = 0; frag < kLaneValues; ++frag) {
            int dim = lane + frag * 32;
            if (dim >= HeadDim)
                continue;
            cuda_kernel::store_float(
                program.output,
                q_linear_index(program, batch, qHead, query, dim),
                output[frag] * invSum);
        }
    }
}

template <int HeadDim>
__global__ void flash_attention_decode_partial_kernel(
        DeviceAttentionProgram program,
        float* partial,
        int splitSize,
        int numSplits) {
    constexpr int kLaneValues = (HeadDim + 31) / 32;

    int lane = threadIdx.x & 31;
    int split = blockIdx.x;
    int64_t qHead = blockIdx.y;
    int64_t batch = blockIdx.z;

    int64_t headsPerKv = program.qHeads / program.kvHeads;
    int64_t kvHead = qHead / headsPerKv;

    int64_t basePosition = 0;
    if (program.hasPositionOffsets)
        basePosition = load_index(program.positionOffsets, program.rank == 4 ? batch : 0);

    int64_t queryPosition = basePosition;
    int64_t minKey = program.window > 0
        ? device_max_i64(0, queryPosition + 1 - program.window)
        : 0;
    int64_t maxKey = queryPosition;

    int64_t splitStart = static_cast<int64_t>(split) * splitSize;
    int64_t splitEnd = device_min_i64(program.tk, splitStart + splitSize);

    float runningMax = -INFINITY;
    float runningSum = 0.0f;
    float output[kLaneValues];
    #pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
        output[i] = 0.0f;

    float qValues[kLaneValues];
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        qValues[frag] = dim < HeadDim
            ? cuda_kernel::load_float(
                  program.q,
                  q_linear_index(program, batch, qHead, 0, dim))
            : 0.0f;
    }

    for (int64_t key = splitStart; key < splitEnd; ++key) {
        bool visible = key <= maxKey && key >= minKey;

        float partialDot = 0.0f;
        #pragma unroll
        for (int frag = 0; frag < kLaneValues; ++frag) {
            int dim = lane + frag * 32;
            if (dim >= HeadDim)
                continue;
            float kValue = cuda_kernel::load_float(
                program.k,
                kv_linear_index(program, batch, kvHead, key, dim));
            partialDot += qValues[frag] * kValue;
        }

        float score = warp_reduce_sum(partialDot) * program.scale;
        if (!visible)
            score = -INFINITY;
        if (isinf(score) && score < 0.0f)
            continue;

        float nextMax = fmaxf(runningMax, score);
        float oldScale = isinf(runningMax) && runningMax < 0.0f
            ? 0.0f
            : expf(runningMax - nextMax);
        float newScale = expf(score - nextMax);

        #pragma unroll
        for (int frag = 0; frag < kLaneValues; ++frag) {
            int dim = lane + frag * 32;
            if (dim >= HeadDim)
                continue;
            float vValue = cuda_kernel::load_float(
                program.v,
                kv_linear_index(program, batch, kvHead, key, dim));
            output[frag] = oldScale * output[frag] + newScale * vValue;
        }
        runningSum = oldScale * runningSum + newScale;
        runningMax = nextMax;
    }

    int64_t stateIndex =
        ((batch * program.qHeads + qHead) * numSplits + split) * (HeadDim + 2);
    if (lane == 0) {
        partial[stateIndex] = runningMax;
        partial[stateIndex + 1] = runningSum;
    }
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        if (dim >= HeadDim)
            continue;
        partial[stateIndex + 2 + dim] = output[frag];
    }
}

template <int HeadDim>
__global__ void flash_attention_decode_reduce_kernel(
        DeviceAttentionProgram program,
        const float* partial,
        int numSplits) {
    constexpr int kLaneValues = (HeadDim + 31) / 32;

    int lane = threadIdx.x & 31;
    int64_t qHead = blockIdx.x;
    int64_t batch = blockIdx.y;

    float runningMax = -INFINITY;
    float runningSum = 0.0f;
    float output[kLaneValues];
    #pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
        output[i] = 0.0f;

    for (int split = 0; split < numSplits; ++split) {
        int64_t stateIndex =
            ((batch * program.qHeads + qHead) * numSplits + split) * (HeadDim + 2);
        float splitMax = partial[stateIndex];
        float splitSum = partial[stateIndex + 1];
        if (splitSum == 0.0f || (isinf(splitMax) && splitMax < 0.0f))
            continue;

        float nextMax = fmaxf(runningMax, splitMax);
        float oldScale = isinf(runningMax) && runningMax < 0.0f
            ? 0.0f
            : expf(runningMax - nextMax);
        float splitScale = expf(splitMax - nextMax);

        #pragma unroll
        for (int frag = 0; frag < kLaneValues; ++frag) {
            int dim = lane + frag * 32;
            if (dim >= HeadDim)
                continue;
            float splitOutput = partial[stateIndex + 2 + dim];
            output[frag] = oldScale * output[frag] + splitScale * splitOutput;
        }
        runningSum = oldScale * runningSum + splitScale * splitSum;
        runningMax = nextMax;
    }

    float invSum = runningSum == 0.0f ? 0.0f : 1.0f / runningSum;
    #pragma unroll
    for (int frag = 0; frag < kLaneValues; ++frag) {
        int dim = lane + frag * 32;
        if (dim >= HeadDim)
            continue;
        cuda_kernel::store_float(
            program.output,
            q_linear_index(program, batch, qHead, 0, dim),
            output[frag] * invSum);
    }
}

template <int HeadDim, int QueriesPerBlock, int KeyValueTileSize>
Result<void> launch_attention_variant(
        DeviceAttentionProgram program,
        cudaStream_t stream,
        const cudaDeviceProp& props) {
    constexpr int threads = QueriesPerBlock * 32;
    size_t sharedBytes =
        static_cast<size_t>(2 * KeyValueTileSize * HeadDim) * sizeof(float) +
        static_cast<size_t>(QueriesPerBlock * KeyValueTileSize) * sizeof(float);

    size_t maxShared = static_cast<size_t>(props.sharedMemPerBlock);
    if (props.sharedMemPerBlockOptin > 0)
        maxShared = std::max(maxShared, static_cast<size_t>(props.sharedMemPerBlockOptin));
    if (sharedBytes > maxShared) {
        return make_error("cuda attention selected tile exceeds device shared memory");
    }

    if (sharedBytes > static_cast<size_t>(props.sharedMemPerBlock)) {
        auto attr = cuda_check(
            cudaFuncSetAttribute(
                flash_attention_prefill_kernel<HeadDim, QueriesPerBlock, KeyValueTileSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(sharedBytes)),
            "cuda attention shared memory opt-in");
        if (!attr)
            return make_error(attr.error());
    }

    dim3 grid(
        static_cast<unsigned>(ceil_div_i64(program.tq, QueriesPerBlock)),
        static_cast<unsigned>(program.qHeads),
        static_cast<unsigned>(program.batch));
    dim3 block(threads);

    flash_attention_prefill_kernel<HeadDim, QueriesPerBlock, KeyValueTileSize>
        <<<grid, block, sharedBytes, stream>>>(program);
    return cuda_check(cudaGetLastError(), "cuda attention launch");
}

template <int HeadDim, int QueriesPerBlock, int LargeTile, int SmallTile>
Result<void> launch_attention_best_tile(
        DeviceAttentionProgram program,
        cudaStream_t stream,
        const cudaDeviceProp& props) {
    size_t largeShared =
        static_cast<size_t>(2 * LargeTile * HeadDim) * sizeof(float) +
        static_cast<size_t>(QueriesPerBlock * LargeTile) * sizeof(float);
    size_t maxShared = static_cast<size_t>(props.sharedMemPerBlock);
    if (props.sharedMemPerBlockOptin > 0)
        maxShared = std::max(maxShared, static_cast<size_t>(props.sharedMemPerBlockOptin));

    if (largeShared <= maxShared) {
        return launch_attention_variant<HeadDim, QueriesPerBlock, LargeTile>(
            program,
            stream,
            props);
    }
    return launch_attention_variant<HeadDim, QueriesPerBlock, SmallTile>(
        program,
        stream,
        props);
}

int choose_decoder_split_size(
        const DeviceAttentionProgram& program,
        const cudaDeviceProp& props) {
    int64_t baseBlocks = std::max<int64_t>(1, program.batch * program.qHeads);
    int64_t targetSplits = std::max<int64_t>(
        1,
        ceil_div_i64(static_cast<int64_t>(props.multiProcessorCount) * 4, baseBlocks));

    for (int splitSize = 512; splitSize >= 1; splitSize /= 2) {
        if (ceil_div_i64(program.tk, splitSize) >= targetSplits)
            return splitSize;
    }
    return 1;
}

template <int HeadDim>
Result<void> launch_attention_decoder_variant(
        DeviceAttentionProgram program,
        const CudaLaunchContext& context,
        int splitSize) {
    cudaStream_t stream = context.stream;
    int numSplits = static_cast<int>(ceil_div_i64(program.tk, splitSize));
    int64_t stateCount = program.batch * program.qHeads * numSplits;
    size_t partialElements =
        static_cast<size_t>(stateCount) * static_cast<size_t>(HeadDim + 2);
    size_t partialBytes = partialElements * sizeof(float);

    float* partial = nullptr;
    auto allocated = cuda_malloc_stream_ordered(
        &partial,
        partialBytes,
        stream,
        "cudaMallocAsync attention decoder partials");
    if (!allocated)
        return make_error(allocated.error());

    auto releasePartial = [&]() -> Result<void> {
        auto freed = cuda_free_stream_ordered(
            partial,
            stream,
            "cudaFreeAsync attention decoder partials");
        partial = nullptr;
        return freed;
    };

    dim3 partialGrid(
        static_cast<unsigned>(numSplits),
        static_cast<unsigned>(program.qHeads),
        static_cast<unsigned>(program.batch));
    dim3 partialBlock(32);

    flash_attention_decode_partial_kernel<HeadDim>
        <<<partialGrid, partialBlock, 0, stream>>>(
            program,
            partial,
            splitSize,
            numSplits);
    auto partialLaunch = cuda_check(cudaGetLastError(), "cuda attention decoder partial launch");
    if (!partialLaunch) {
        auto released = releasePartial();
        return make_error(
            released ? partialLaunch.error()
                     : partialLaunch.error() + "; " + released.error());
    }

    dim3 reduceGrid(
        static_cast<unsigned>(program.qHeads),
        static_cast<unsigned>(program.batch));
    dim3 reduceBlock(32);
    flash_attention_decode_reduce_kernel<HeadDim>
        <<<reduceGrid, reduceBlock, 0, stream>>>(
            program,
            partial,
            numSplits);
    auto reduceLaunch = cuda_check(cudaGetLastError(), "cuda attention decoder reduce launch");
    if (!reduceLaunch) {
        auto released = releasePartial();
        return make_error(
            released ? reduceLaunch.error()
                     : reduceLaunch.error() + "; " + released.error());
    }

    return releasePartial();
}

template <int HeadDim>
Result<void> launch_attention_decoder_best_split(
        DeviceAttentionProgram program,
        const CudaLaunchContext& context,
        const cudaDeviceProp& props) {
    int splitSize = choose_decoder_split_size(
        program,
        props);
    return launch_attention_decoder_variant<HeadDim>(program, context, splitSize);
}

Result<DeviceAttentionProgram> pack_attention_program(
        const CudaLaunchContext& context,
        const CudaAttentionProgram& program) {
    const auto& qView = context.inputs[0];
    const auto& kView = context.inputs[1];
    const auto& vView = context.inputs[2];
    const auto& outputView = context.outputs[0];

    auto qDtype = validate_attention_dtype(qView, "q");
    if (!qDtype) return make_error(qDtype.error());
    auto kDtype = validate_attention_dtype(kView, "k");
    if (!kDtype) return make_error(kDtype.error());
    auto vDtype = validate_attention_dtype(vView, "v");
    if (!vDtype) return make_error(vDtype.error());
    auto outDtype = validate_attention_dtype(outputView, "output");
    if (!outDtype) return make_error(outDtype.error());

    if (qView.view.desc.dtype != kView.view.desc.dtype ||
        qView.view.desc.dtype != vView.view.desc.dtype ||
        qView.view.desc.dtype != outputView.view.desc.dtype) {
        return make_error("cuda attention dtype mismatch");
    }
    if (qView.view.desc.shape != outputView.view.desc.shape)
        return make_error("cuda attention output shape mismatch");

    int rank = qView.view.desc.shape.rank();
    if ((rank != 3 && rank != 4) ||
        kView.view.desc.shape.rank() != rank ||
        vView.view.desc.shape.rank() != rank) {
        return make_error("cuda attention q, k, and v must all have rank 3 or rank 4");
    }

    int64_t batch = rank == 4 ? qView.view.desc.shape.dim(0) : 1;
    int64_t kBatch = rank == 4 ? kView.view.desc.shape.dim(0) : 1;
    int64_t vBatch = rank == 4 ? vView.view.desc.shape.dim(0) : 1;
    int64_t qHeads = qView.view.desc.shape.dim(rank - 3);
    int64_t kvHeads = kView.view.desc.shape.dim(rank - 3);
    int64_t vHeads = vView.view.desc.shape.dim(rank - 3);
    int64_t tq = qView.view.desc.shape.dim(rank - 2);
    int64_t tk = kView.view.desc.shape.dim(rank - 2);
    int64_t tv = vView.view.desc.shape.dim(rank - 2);
    int64_t headDim = qView.view.desc.shape.dim(rank - 1);
    int64_t kHeadDim = kView.view.desc.shape.dim(rank - 1);
    int64_t vHeadDim = vView.view.desc.shape.dim(rank - 1);

    if (batch < 0 || kBatch < 0 || vBatch < 0 ||
        qHeads < 0 || kvHeads < 0 || vHeads < 0 ||
        tq < 0 || tk < 0 || tv < 0 ||
        headDim < 0 || kHeadDim < 0 || vHeadDim < 0) {
        return make_error("cuda attention inputs must have static shape");
    }
    if (batch != kBatch || batch != vBatch)
        return make_error("cuda attention batch dimension mismatch");
    if (kvHeads != vHeads)
        return make_error("cuda attention k and v head count mismatch");
    if (tk != tv)
        return make_error("cuda attention k and v sequence dimension mismatch");
    if (headDim != kHeadDim || headDim != vHeadDim)
        return make_error("cuda attention head dimension mismatch");
    if (qHeads <= 0 || kvHeads <= 0 || qHeads % kvHeads != 0)
        return make_error("cuda attention heads must be divisible by kv_heads");
    if (program.window < 0)
        return make_error("cuda attention window must be non-negative");

    DeviceAttentionProgram packed;
    packed.rank = rank;
    packed.batch = batch;
    packed.qHeads = qHeads;
    packed.kvHeads = kvHeads;
    packed.tq = tq;
    packed.tk = tk;
    packed.headDim = headDim;
    packed.window = program.window;
    packed.scale = program.scale > 0.0
        ? static_cast<float>(program.scale)
        : 1.0f / std::sqrt(static_cast<float>(headDim));

    auto q = cuda_kernel::pack_tensor_arg(qView);
    if (!q) return make_error(q.error());
    auto k = cuda_kernel::pack_tensor_arg(kView);
    if (!k) return make_error(k.error());
    auto v = cuda_kernel::pack_tensor_arg(vView);
    if (!v) return make_error(v.error());
    auto output = cuda_kernel::pack_tensor_arg(outputView);
    if (!output) return make_error(output.error());
    packed.q = q.take();
    packed.k = k.take();
    packed.v = v.take();
    packed.output = output.take();

    if (context.inputs.size() == 4) {
        const auto& positionView = context.inputs[3];
        auto dtype = positionView.view.desc.dtype;
        if (dtype != core::DType::I32 && dtype != core::DType::I64)
            return make_error("cuda attention position_offsets must be i32 or i64");
        int64_t offsetsNumel = positionView.view.desc.shape.numel();
        if (offsetsNumel < 0)
            return make_error("cuda attention position_offsets must have static shape");
        if (rank == 4 && offsetsNumel != batch)
            return make_error("cuda attention position_offsets length must match batch");
        if (rank == 3 && offsetsNumel != 1)
            return make_error("cuda attention unbatched position_offsets must have one element");

        auto positionOffsets = cuda_kernel::pack_tensor_arg(positionView);
        if (!positionOffsets)
            return make_error(positionOffsets.error());
        packed.positionOffsets = positionOffsets.take();
        packed.hasPositionOffsets = true;
    }

    return packed;
}

} // namespace

Result<void> launch_cuda_attention(
        const CudaLaunchContext& context,
        const CudaAttentionProgram& program) {
    auto valid = validate_context(context, context.inputs.size() == 4 ? 4 : 3, 1, "attention");
    if (!valid)
        return make_error(valid.error());
    if (context.inputs.size() != 3 && context.inputs.size() != 4)
        return make_error("cuda attention input arity mismatch");

    auto packed = pack_attention_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.batch == 0 || launchProgram.qHeads == 0 ||
        launchProgram.tq == 0 || launchProgram.tk == 0) {
        return {};
    }
    if (ceil_div_i64(launchProgram.tq, 4) > std::numeric_limits<unsigned>::max() ||
        launchProgram.qHeads > std::numeric_limits<unsigned>::max() ||
        launchProgram.batch > std::numeric_limits<unsigned>::max()) {
        return make_error("cuda attention grid dimension exceeds CUDA limit");
    }

    if (!context.deviceProps)
        return make_error("cuda attention launch requires cached device properties");
    const cudaDeviceProp& props = *context.deviceProps;

    if (launchProgram.tq == 1) {
        int splitSize = choose_decoder_split_size(
            launchProgram,
            props);
        int64_t numSplits = ceil_div_i64(launchProgram.tk, splitSize);
        if (numSplits > std::numeric_limits<unsigned>::max()) {
            return make_error("cuda attention decoder split count exceeds CUDA grid limit");
        }

        switch (launchProgram.headDim) {
            case 64:
                return launch_attention_decoder_best_split<64>(
                    launchProgram,
                    context,
                    props);
            case 128:
                return launch_attention_decoder_best_split<128>(
                    launchProgram,
                    context,
                    props);
            case 256:
                return launch_attention_decoder_best_split<256>(
                    launchProgram,
                    context,
                    props);
            case 512:
                return launch_attention_decoder_best_split<512>(
                    launchProgram,
                    context,
                    props);
            default:
                return make_error("cuda attention unsupported head dimension");
        }
    }

    switch (launchProgram.headDim) {
        case 64:
            return launch_attention_best_tile<64, 8, 64, 32>(
                launchProgram,
                context.stream,
                props);
        case 128:
            return launch_attention_best_tile<128, 8, 64, 32>(
                launchProgram,
                context.stream,
                props);
        case 256:
            return launch_attention_best_tile<256, 8, 32, 16>(
                launchProgram,
                context.stream,
                props);
        case 512:
            return launch_attention_best_tile<512, 8, 16, 8>(
                launchProgram,
                context.stream,
                props);
        default:
            return make_error("cuda attention unsupported head dimension");
    }
}

} // namespace sandy::device
