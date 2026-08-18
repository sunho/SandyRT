#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"

#include <cmath>
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
    if (program.k <= 0)
        return make_error("cuda topk k must be positive");

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

    int blocks = static_cast<int>(launchProgram.rows);
    int threads = cuda_kernel::kBlockSize;
    size_t sharedBytes =
        static_cast<size_t>(threads) * sizeof(float) +
        static_cast<size_t>(threads + launchProgram.k) * sizeof(int64_t);
    topk_last_dim_kernel<<<blocks, threads, sharedBytes, context.stream>>>(launchProgram);
    return cuda_check(cudaGetLastError(), "cuda topk launch");
}

} // namespace sandy::device
