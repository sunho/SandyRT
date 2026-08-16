#include "CudaDevice.h"

#include <cstring>
#include <memory>
#include <span>
#include <utility>

namespace sandy::engine {

namespace {

class CudaHostTensorBuffer final : public core::TensorBuffer {
public:
    CudaHostTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

Result<size_t> tensor_byte_size(const core::TensorDesc& desc) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error("cuda device buffer must have static shape");
    return static_cast<size_t>(numel) * core::dtype_size(desc.dtype);
}

} // namespace

CudaDevice::CudaDevice(int cudaDevice)
    : cudaDevice_(cudaDevice) {}

CudaDevice::~CudaDevice() {
    if (stream_) {
        cudaSetDevice(cudaDevice_);
        cudaStreamDestroy(stream_);
    }
    for (auto& item : buffers_) {
        if (item.second.data) {
            cudaSetDevice(cudaDevice_);
            cudaFree(item.second.data);
        }
    }
}

Result<void> CudaDevice::set_device() const {
    return cuda_check(cudaSetDevice(cudaDevice_), "cudaSetDevice");
}

Result<void> CudaDevice::ensure_stream() {
    auto set = set_device();
    if (!set)
        return make_error(set.error());
    if (stream_)
        return {};
    return cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

Result<DeviceCompiledGraphId> CudaDevice::compile(const ir::kernel_ir::Graph& graph) {
    auto verify = graph.verify();
    if (!verify)
        return make_error(verify.error());

    CudaDeviceGraph compiled;
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (op.kind() == ir::kernel_ir::OpKind::Input ||
            op.kind() == ir::kernel_ir::OpKind::DeviceTransfer) {
            continue;
        }

        CudaDeviceKernel kernel;
        kernel.kind = op.kind();
        kernel.inputCount = op.inputs().size();
        kernel.outputCount = op.outputs().size();

        switch (op.kind()) {
            case ir::kernel_ir::OpKind::LayoutTransform: {
                const auto& layout = static_cast<const ir::kernel_ir::LayoutTransformOp&>(op);
                kernel.program = CudaLayoutTransformProgram{
                    layout.transform(),
                    layout.dims(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::ElementwiseKernel: {
                const auto& elementwise =
                    static_cast<const ir::kernel_ir::ElementwiseKernelOp&>(op);
                kernel.program = CudaElementwiseProgram{
                    elementwise.elementwiseInputs(),
                    elementwise.output(),
                    elementwise.result(),
                    elementwise.scalars(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::ReductionKernel: {
                const auto& reduction =
                    static_cast<const ir::kernel_ir::ReductionKernelOp&>(op);
                kernel.program = CudaReductionProgram{
                    reduction.reduce(),
                    reduction.axes(),
                    reduction.keepDims(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::MatMulKernel: {
                const auto& matmul = static_cast<const ir::kernel_ir::MatMulKernelOp&>(op);
                kernel.program = CudaMatMulProgram{
                    matmul.transposeLhs(),
                    matmul.transposeRhs(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::GatherKernel:
                kernel.program = std::monostate{};
                break;
            case ir::kernel_ir::OpKind::SoftmaxKernel: {
                const auto& softmax = static_cast<const ir::kernel_ir::SoftmaxKernelOp&>(op);
                kernel.program = CudaSoftmaxProgram{softmax.axis()};
                break;
            }
            case ir::kernel_ir::OpKind::NormKernel: {
                const auto& norm = static_cast<const ir::kernel_ir::NormKernelOp&>(op);
                kernel.program = CudaNormProgram{norm.norm(), norm.epsilon()};
                break;
            }
            case ir::kernel_ir::OpKind::RoPEKernel: {
                const auto& rope = static_cast<const ir::kernel_ir::RoPEKernelOp&>(op);
                kernel.program = CudaRoPEProgram{
                    rope.theta(),
                    rope.rotaryDim(),
                    rope.splitHalf(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::SlidingQueryKeyScoreKernel: {
                const auto& score =
                    static_cast<const ir::kernel_ir::SlidingQueryKeyScoreKernelOp&>(op);
                kernel.program = CudaSlidingQueryKeyScoreProgram{
                    score.window(),
                    score.scale(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::CustomKernel: {
                const auto& custom = static_cast<const ir::kernel_ir::CustomKernelOp&>(op);
                kernel.program = CudaCustomProgram{custom.customName()};
                break;
            }
            case ir::kernel_ir::OpKind::Input:
            case ir::kernel_ir::OpKind::DeviceTransfer:
                break;
        }

        compiled.kernels[op.id()] = std::move(kernel);
    }

    auto id = nextGraphId_++;
    graphs_[id] = std::move(compiled);
    return id;
}

Result<DeviceBufferId> CudaDevice::alloc(core::TensorDesc desc) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto bytes = tensor_byte_size(desc);
    if (!bytes)
        return make_error(bytes.error());

    void* data = nullptr;
    if (*bytes != 0) {
        auto allocated = cuda_check(cudaMalloc(&data, *bytes), "cudaMalloc");
        if (!allocated)
            return make_error(allocated.error());
    }

    CudaDeviceBuffer buffer;
    buffer.desc = std::move(desc);
    buffer.data = data;
    buffer.bytes = *bytes;

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<void> CudaDevice::dealloc(DeviceBufferId buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end())
        return make_error("cuda device buffer not found");
    auto set = set_device();
    if (!set)
        return make_error(set.error());
    if (it->second.data) {
        auto freed = cuda_check(cudaFree(it->second.data), "cudaFree");
        if (!freed)
            return make_error(freed.error());
    }
    buffers_.erase(it);
    return {};
}

Result<DeviceBufferId> CudaDevice::load(core::TensorBuffer& src) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto access = src.access();
    if (!access)
        return make_error(access.error());

    auto bytes = tensor_byte_size((*access).desc());
    if (!bytes)
        return make_error(bytes.error());
    if ((*access).data().size() != *bytes)
        return make_error("cuda device host buffer size mismatch");

    void* data = nullptr;
    if (*bytes != 0) {
        auto allocated = cuda_check(cudaMalloc(&data, *bytes), "cudaMalloc");
        if (!allocated)
            return make_error(allocated.error());
        auto copied = cuda_check(
            cudaMemcpyAsync(
                data,
                (*access).data().data(),
                *bytes,
                cudaMemcpyHostToDevice,
                stream_),
            "cudaMemcpyAsync host to device");
        if (!copied) {
            cudaFree(data);
            return make_error(copied.error());
        }
        auto synced = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        if (!synced) {
            cudaFree(data);
            return make_error(synced.error());
        }
    }

    CudaDeviceBuffer buffer;
    buffer.desc = (*access).desc();
    buffer.data = data;
    buffer.bytes = *bytes;

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<CudaDeviceBufferView> CudaDevice::buffer_view(DeviceBufferId buffer, bool writable) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end()) {
        return make_error(
            writable ? "cuda device output buffer not found"
                     : "cuda device input buffer not found");
    }
    return CudaDeviceBufferView{it->second.data, it->second.desc, it->second.bytes};
}

Result<void> CudaDevice::run(
        DeviceCompiledGraphId graphId,
        ir::kernel_ir::OpId opId,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());

    auto graphIt = graphs_.find(graphId);
    if (graphIt == graphs_.end())
        return make_error("cuda device compiled graph not found");
    auto kernelIt = graphIt->second.kernels.find(opId);
    if (kernelIt == graphIt->second.kernels.end())
        return make_error("cuda device kernel op not found");
    const auto& kernel = kernelIt->second;

    if (inputs.size() != kernel.inputCount)
        return make_error("cuda device input arity mismatch");
    if (outputs.size() != kernel.outputCount)
        return make_error("cuda device output arity mismatch");

    std::vector<CudaDeviceBufferView> inputViews;
    inputViews.reserve(inputs.size());
    for (auto input : inputs) {
        auto view = buffer_view(input, false);
        if (!view)
            return make_error(view.error());
        inputViews.push_back(view.take());
    }

    std::vector<CudaDeviceBufferView> outputViews;
    outputViews.reserve(outputs.size());
    for (auto output : outputs) {
        auto view = buffer_view(output, true);
        if (!view)
            return make_error(view.error());
        outputViews.push_back(view.take());
    }

    CudaLaunchContext context{
        cudaDevice_,
        stream_,
        opId,
        inputViews,
        outputViews,
    };

    switch (kernel.kind) {
        case ir::kernel_ir::OpKind::Input:
        case ir::kernel_ir::OpKind::DeviceTransfer:
            return make_error("cuda device cannot run boundary op");
        case ir::kernel_ir::OpKind::ElementwiseKernel:
            return launch_cuda_elementwise(
                context,
                std::get<CudaElementwiseProgram>(kernel.program));
        case ir::kernel_ir::OpKind::LayoutTransform:
            return launch_cuda_layout_transform(
                context,
                std::get<CudaLayoutTransformProgram>(kernel.program));
        case ir::kernel_ir::OpKind::ReductionKernel:
            return launch_cuda_reduction(
                context,
                std::get<CudaReductionProgram>(kernel.program));
        case ir::kernel_ir::OpKind::MatMulKernel:
            return launch_cuda_matmul(
                context,
                std::get<CudaMatMulProgram>(kernel.program));
        case ir::kernel_ir::OpKind::GatherKernel:
            return launch_cuda_gather(context);
        case ir::kernel_ir::OpKind::SoftmaxKernel:
            return launch_cuda_softmax(
                context,
                std::get<CudaSoftmaxProgram>(kernel.program));
        case ir::kernel_ir::OpKind::NormKernel:
            return launch_cuda_norm(
                context,
                std::get<CudaNormProgram>(kernel.program));
        case ir::kernel_ir::OpKind::RoPEKernel:
            return launch_cuda_rope(
                context,
                std::get<CudaRoPEProgram>(kernel.program));
        case ir::kernel_ir::OpKind::SlidingQueryKeyScoreKernel:
            return launch_cuda_sliding_query_key_score(
                context,
                std::get<CudaSlidingQueryKeyScoreProgram>(kernel.program));
        case ir::kernel_ir::OpKind::CustomKernel:
            return launch_cuda_custom(
                context,
                std::get<CudaCustomProgram>(kernel.program));
    }

    return make_error("cuda device cannot run unknown op kind");
}

Result<TensorBufferPtr> CudaDevice::read(DeviceBufferId src) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto it = buffers_.find(src);
    if (it == buffers_.end())
        return make_error("cuda device buffer not found");

    std::vector<uint8_t> data(it->second.bytes);
    if (it->second.bytes != 0) {
        auto copied = cuda_check(
            cudaMemcpyAsync(
                data.data(),
                it->second.data,
                it->second.bytes,
                cudaMemcpyDeviceToHost,
                stream_),
            "cudaMemcpyAsync device to host");
        if (!copied)
            return make_error(copied.error());
        auto synced = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        if (!synced)
            return make_error(synced.error());
    }

    TensorBufferPtr buffer = std::make_shared<CudaHostTensorBuffer>(
        it->second.desc,
        std::move(data));
    return buffer;
}

} // namespace sandy::engine
