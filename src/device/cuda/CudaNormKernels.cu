#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"
#include "CudaKernelUtils.cuh"
#include "jit/CudaJitLaunchUtils.cuh"
#include "jit/CudaNormJit.h"
#include "jit/templates/CudaJitNormAbi.cuh"

#include <limits>

namespace sandy::device {

namespace {

struct DeviceNormProgram {
    ir::kernel_ir::NormKind norm = ir::kernel_ir::NormKind::RMSNorm;
    cuda_kernel::TensorArg x;
    cuda_kernel::TensorArg weight;
    cuda_kernel::TensorArg bias;
    cuda_kernel::TensorArg output;
    int64_t rows = 0;
    int64_t hidden = 0;
    float epsilon = 0.0f;
    bool hasWeight = false;
};

Result<void> validate_norm_tensor_dtype(
        const CudaDeviceBufferView& buffer,
        const char* name) {
    if (!is_float_compute_dtype(buffer.view.desc.dtype))
        return make_error(std::string("cuda norm ") + name + " unsupported dtype");
    return {};
}

Result<void> validate_same_shape_and_dtype(
        const CudaDeviceBufferView& lhs,
        const CudaDeviceBufferView& rhs,
        const char* name) {
    if (lhs.view.desc.dtype != rhs.view.desc.dtype)
        return make_error(std::string("cuda norm ") + name + " dtype mismatch");
    if (lhs.view.desc.shape != rhs.view.desc.shape)
        return make_error(std::string("cuda norm ") + name + " shape mismatch");
    return {};
}

Result<void> validate_norm_vector(
        const CudaDeviceBufferView& vector,
        const CudaDeviceBufferView& x,
        int64_t hidden,
        const char* name) {
    if (vector.view.desc.dtype != x.view.desc.dtype)
        return make_error(std::string("cuda norm ") + name + " dtype mismatch");
    if (vector.view.desc.shape.rank() != 1)
        return make_error(std::string("cuda norm ") + name + " must have rank 1");
    if (vector.view.desc.shape.dim(0) != hidden)
        return make_error(std::string("cuda norm ") + name + " dimension mismatch");
    return {};
}

Result<DeviceNormProgram> pack_norm_program(
        const CudaLaunchContext& context,
        const CudaNormProgram& program) {
    if (context.outputs.size() != 1)
        return make_error("cuda norm output arity mismatch");

    if (program.norm == ir::kernel_ir::NormKind::RMSNorm) {
        if (context.inputs.size() != 1 && context.inputs.size() != 2)
            return make_error("cuda rms_norm expects 1 or 2 inputs");
    } else if (context.inputs.size() != 3) {
        return make_error("cuda layer_norm expects 3 inputs");
    }

    const auto& xView = context.inputs[0];
    const auto& outputView = context.outputs[0];

    auto xDtype = validate_norm_tensor_dtype(xView, "input");
    if (!xDtype)
        return make_error(xDtype.error());
    auto outputDtype = validate_norm_tensor_dtype(outputView, "output");
    if (!outputDtype)
        return make_error(outputDtype.error());
    auto outputSame = validate_same_shape_and_dtype(xView, outputView, "output");
    if (!outputSame)
        return make_error(outputSame.error());

    int rank = xView.view.desc.shape.rank();
    if (rank < 1)
        return make_error("cuda norm input must have rank >= 1");
    int64_t hidden = xView.view.desc.shape.dim(rank - 1);
    if (hidden <= 0)
        return make_error("cuda norm hidden dimension must be static and positive");
    int64_t total = xView.view.desc.shape.numel();
    if (total < 0)
        return make_error("cuda norm input must have static shape");

    DeviceNormProgram packed;
    packed.norm = program.norm;
    packed.rows = total / hidden;
    packed.hidden = hidden;
    packed.epsilon = static_cast<float>(program.epsilon);
    packed.hasWeight =
        program.norm == ir::kernel_ir::NormKind::LayerNorm || context.inputs.size() == 2;

    auto x = cuda_kernel::pack_tensor_arg(xView);
    if (!x)
        return make_error(x.error());
    auto output = cuda_kernel::pack_tensor_arg(outputView);
    if (!output)
        return make_error(output.error());
    packed.x = x.take();
    packed.output = output.take();

    if (packed.hasWeight) {
        auto weightValid = validate_norm_vector(context.inputs[1], xView, hidden, "weight");
        if (!weightValid)
            return make_error(weightValid.error());
        auto weight = cuda_kernel::pack_tensor_arg(context.inputs[1]);
        if (!weight)
            return make_error(weight.error());
        packed.weight = weight.take();
    }

    if (program.norm == ir::kernel_ir::NormKind::LayerNorm) {
        auto biasValid = validate_norm_vector(context.inputs[2], xView, hidden, "bias");
        if (!biasValid)
            return make_error(biasValid.error());
        auto bias = cuda_kernel::pack_tensor_arg(context.inputs[2]);
        if (!bias)
            return make_error(bias.error());
        packed.bias = bias.take();
    }

    return packed;
}

__device__ float block_reduce_sum(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    return shared[0];
}

__global__ void rms_norm_kernel(DeviceNormProgram program) {
    extern __shared__ float shared[];

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= program.rows)
        return;

    float localSquareSum = 0.0f;
    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        float value = cuda_kernel::load_float(program.x, linear);
        localSquareSum += value * value;
    }

    float squareSum = block_reduce_sum(localSquareSum, shared);
    float invRms = rsqrtf(squareSum / static_cast<float>(program.hidden) + program.epsilon);

    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        float value = cuda_kernel::load_float(program.x, linear);
        float scale = program.hasWeight ? cuda_kernel::load_float(program.weight, col) : 1.0f;
        cuda_kernel::store_float(program.output, linear, value * invRms * scale);
    }
}

__global__ void layer_norm_kernel(DeviceNormProgram program) {
    extern __shared__ float shared[];

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= program.rows)
        return;

    float localSum = 0.0f;
    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        localSum += cuda_kernel::load_float(program.x, linear);
    }

    float sum = block_reduce_sum(localSum, shared);
    float mean = sum / static_cast<float>(program.hidden);

    float localVarianceSum = 0.0f;
    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        float centered = cuda_kernel::load_float(program.x, linear) - mean;
        localVarianceSum += centered * centered;
    }

    float varianceSum = block_reduce_sum(localVarianceSum, shared);
    float invStd = rsqrtf(varianceSum / static_cast<float>(program.hidden) + program.epsilon);

    for (int64_t col = tid; col < program.hidden; col += blockDim.x) {
        int64_t linear = row * program.hidden + col;
        float centered = cuda_kernel::load_float(program.x, linear) - mean;
        float scale = cuda_kernel::load_float(program.weight, col);
        float offset = cuda_kernel::load_float(program.bias, col);
        cuda_kernel::store_float(program.output, linear, centered * invStd * scale + offset);
    }
}

} // namespace

Result<void> launch_cuda_norm(
        const CudaLaunchContext& context,
        const CudaNormProgram& program) {
    auto valid = validate_context(context, 1, 1, "norm");
    if (!valid)
        return make_error(valid.error());

    auto packed = pack_norm_program(context, program);
    if (!packed)
        return make_error(packed.error());
    auto launchProgram = packed.take();
    if (launchProgram.rows == 0)
        return {};
    if (launchProgram.rows > std::numeric_limits<int>::max())
        return make_error("cuda norm row count exceeds grid limit");

    int blocks = static_cast<int>(launchProgram.rows);
    int threads = cuda_kernel::kBlockSize;
    size_t sharedBytes = static_cast<size_t>(threads) * sizeof(float);

    if (program.jitVariants) {
        if (!context.jitCache)
            return make_error("cuda norm JIT cache is null");
        std::vector<int> inputAccesses;
        inputAccesses.push_back(jit_access_kind(launchProgram.x.access));
        if (launchProgram.hasWeight)
            inputAccesses.push_back(jit_access_kind(launchProgram.weight.access));
        if (launchProgram.norm == ir::kernel_ir::NormKind::LayerNorm)
            inputAccesses.push_back(jit_access_kind(launchProgram.bias.access));
        int outputAccess = jit_access_kind(launchProgram.output.access);
        std::vector<int> accesses = inputAccesses;
        accesses.push_back(outputAccess);
        auto jit = program.jitVariants->getOrCompile(
            cudaJitAccessKey(accesses),
            [&] {
                return compileCudaNormJit(
                    context.cudaDevice,
                    *context.jitCache,
                    program.norm,
                    program.hasWeight,
                    program.dtype,
                    inputAccesses,
                    outputAccess);
            });
        if (!jit) {
            if (!program.jitFallbackOnError)
                return make_error(jit.error());
        } else {
            auto input = pack_jit_tensor_arg(launchProgram.x);
            if (!input)
                return make_error(input.error());
            SandyNormParams params{};
            params.input = input.take();
            if (launchProgram.hasWeight) {
                auto weight = pack_jit_tensor_arg(launchProgram.weight);
                if (!weight)
                    return make_error(weight.error());
                params.weight = weight.take();
            }
            if (launchProgram.norm == ir::kernel_ir::NormKind::LayerNorm) {
                auto bias = pack_jit_tensor_arg(launchProgram.bias);
                if (!bias)
                    return make_error(bias.error());
                params.bias = bias.take();
            }
            auto output = pack_jit_tensor_arg(launchProgram.output);
            if (!output)
                return make_error(output.error());
            params.output = output.take();
            params.rows = launchProgram.rows;
            params.hidden = launchProgram.hidden;
            params.epsilon = launchProgram.epsilon;
            void* arguments[] = {&params};
            return (*jit)->launch(
                dim3(static_cast<unsigned>(blocks)),
                dim3(static_cast<unsigned>(threads)),
                sharedBytes,
                context.stream,
                arguments);
        }
    }

    if (launchProgram.norm == ir::kernel_ir::NormKind::RMSNorm) {
        rms_norm_kernel<<<blocks, threads, sharedBytes, context.stream>>>(launchProgram);
        return cuda_check(cudaGetLastError(), "cuda rms_norm launch");
    }

    layer_norm_kernel<<<blocks, threads, sharedBytes, context.stream>>>(launchProgram);
    return cuda_check(cudaGetLastError(), "cuda layer_norm launch");
}

} // namespace sandy::device
