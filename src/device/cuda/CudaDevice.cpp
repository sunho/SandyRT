#include "CudaDevice.h"

#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace sandy::device {

namespace {

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

Result<void> cublas_check(cublasStatus_t status, const std::string& context) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return {};
    return make_error(context + ": " + cublas_status_name(status));
}

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

Result<void*> CudaWorkspace::reserve(
        int cudaDevice,
        cudaStream_t stream,
        size_t requiredBytes) {
    if (requiredBytes == 0)
        return nullptr;
    if (data && bytes >= requiredBytes)
        return data;

    auto set = cuda_check(cudaSetDevice(cudaDevice), "cudaSetDevice");
    if (!set)
        return make_error(set.error());

    if (data) {
        auto freed = cuda_free_stream_ordered(data, stream, "cudaFreeAsync workspace");
        if (!freed)
            return make_error(freed.error());
        data = nullptr;
        bytes = 0;
    }

    void* newData = nullptr;
    auto allocated = cuda_malloc_stream_ordered(
        &newData,
        requiredBytes,
        stream,
        "cudaMallocAsync workspace");
    if (!allocated)
        return make_error(allocated.error());

    data = newData;
    bytes = requiredBytes;
    return data;
}

Result<void> CudaWorkspace::release(int cudaDevice, cudaStream_t stream) {
    if (!data)
        return {};

    auto set = cuda_check(cudaSetDevice(cudaDevice), "cudaSetDevice");
    if (!set)
        return make_error(set.error());
    auto freed = cuda_free_stream_ordered(data, stream, "cudaFreeAsync workspace");
    if (!freed)
        return make_error(freed.error());
    data = nullptr;
    bytes = 0;
    return {};
}

CudaDevice::CudaDevice(int cudaDevice)
    : cudaDevice_(cudaDevice) {}

CudaDevice::~CudaDevice() {
    cudaSetDevice(cudaDevice_);
    if (stream_)
        cudaStreamSynchronize(stream_);

    pagedTensors_.clear();
    pagedPools_.clear();

    if (stream_) {
        (void)workspace_.release(cudaDevice_, stream_);
        for (auto& item : buffers_) {
            if (item.second.data) {
                (void)cuda_free_stream_ordered(
                    item.second.data,
                    stream_,
                    "cudaFreeAsync device buffer");
                item.second.data = nullptr;
            }
        }
        cudaStreamSynchronize(stream_);
    } else {
        (void)workspace_.release(cudaDevice_, nullptr);
        for (auto& item : buffers_) {
            if (item.second.data) {
                cudaFree(item.second.data);
                item.second.data = nullptr;
            }
        }
    }
    buffers_.clear();

    if (cublasHandle_) {
        cublasDestroy(cublasHandle_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

Result<void> CudaDevice::set_device() const {
    return cuda_check(cudaSetDevice(cudaDevice_), "cudaSetDevice");
}

Result<void> CudaDevice::ensure_stream() {
    auto set = set_device();
    if (!set)
        return make_error(set.error());
    auto pool = ensure_async_memory_pool();
    if (!pool)
        return make_error(pool.error());
    if (stream_)
        return {};
    return cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

Result<void> CudaDevice::ensure_async_memory_pool() {
    if (asyncMemoryPoolConfigured_)
        return {};
    auto set = set_device();
    if (!set)
        return make_error(set.error());
    auto configured = cuda_configure_default_memory_pool(cudaDevice_);
    if (!configured)
        return make_error(configured.error());
    asyncMemoryPoolConfigured_ = true;
    return {};
}

Result<void> CudaDevice::ensure_cublas_handle() {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    if (cublasHandle_)
        return {};

    auto created = cublas_check(cublasCreate(&cublasHandle_), "cublasCreate");
    if (!created)
        return make_error(created.error());
    auto setStream = cublas_check(cublasSetStream(cublasHandle_, stream_), "cublasSetStream");
    if (!setStream) {
        cublasDestroy(cublasHandle_);
        cublasHandle_ = nullptr;
        return make_error(setStream.error());
    }
    return {};
}

Result<const cudaDeviceProp*> CudaDevice::ensure_device_properties() {
    if (deviceProps_)
        return &*deviceProps_;

    auto set = set_device();
    if (!set)
        return make_error(set.error());

    cudaDeviceProp props{};
    auto queried = cuda_check(
        cudaGetDeviceProperties(&props, cudaDevice_),
        "cudaGetDeviceProperties");
    if (!queried)
        return make_error(queried.error());

    deviceProps_ = props;
    return &*deviceProps_;
}

Result<DeviceCompiledGraphId> CudaDevice::compile(const ir::kernel_ir::Graph& graph) {
    auto verify = graph.verify();
    if (!verify)
        return make_error(verify.error());

    CudaDeviceGraph compiled;
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (op.kind() == ir::kernel_ir::OpKind::Input ||
            op.kind() == ir::kernel_ir::OpKind::TensorTupleCreate ||
            op.kind() == ir::kernel_ir::OpKind::PagedAppend ||
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
            case ir::kernel_ir::OpKind::LinearKernel:
                kernel.program = std::monostate{};
                break;
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
            case ir::kernel_ir::OpKind::TopKKernel: {
                const auto& topk = static_cast<const ir::kernel_ir::TopKKernelOp&>(op);
                kernel.program = CudaTopKProgram{topk.k(), topk.axis()};
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
            case ir::kernel_ir::OpKind::AttentionKernel: {
                const auto& attention =
                    static_cast<const ir::kernel_ir::AttentionKernelOp&>(op);
                kernel.program = CudaAttentionProgram{
                    attention.window(),
                    attention.scale(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::MoeGatherKernel: {
                const auto& gather = static_cast<const ir::kernel_ir::MoeGatherKernelOp&>(op);
                kernel.program = CudaMoeGatherProgram{
                    gather.numExperts(),
                    gather.topK(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::MoeMatMulKernel: {
                const auto& matmul = static_cast<const ir::kernel_ir::MoeMatMulKernelOp&>(op);
                kernel.program = CudaMoeMatMulProgram{
                    matmul.transposeRhs(),
                };
                break;
            }
            case ir::kernel_ir::OpKind::MoeScatterSumKernel:
                kernel.program = std::monostate{};
                break;
            case ir::kernel_ir::OpKind::Input:
            case ir::kernel_ir::OpKind::TensorTupleCreate:
            case ir::kernel_ir::OpKind::DeviceTransfer:
            case ir::kernel_ir::OpKind::PagedAppend:
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
    auto cublas = ensure_cublas_handle();
    if (!cublas)
        return make_error(cublas.error());
    auto bytes = tensor_byte_size(desc);
    if (!bytes)
        return make_error(bytes.error());

    void* data = nullptr;
    if (*bytes != 0) {
        auto allocated = cuda_malloc_stream_ordered(
            &data,
            *bytes,
            stream_,
            "cudaMallocAsync");
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
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    if (it->second.data) {
        auto freed = cuda_free_stream_ordered(
            it->second.data,
            stream_,
            "cudaFreeAsync");
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
        auto allocated = cuda_malloc_stream_ordered(
            &data,
            *bytes,
            stream_,
            "cudaMallocAsync");
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
            (void)cuda_free_stream_ordered(data, stream_, "cudaFreeAsync");
            return make_error(copied.error());
        }
        auto synced = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        if (!synced) {
            (void)cuda_free_stream_ordered(data, stream_, "cudaFreeAsync");
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
    auto view = defaultView(it->second.desc);
    if (!view)
        return make_error(view.error());
    return CudaDeviceBufferView{
        it->second.data,
        view.take(),
        it->second.bytes,
    };
}

Result<void> CudaDevice::run(
        DeviceCompiledGraphId graphId,
        ir::kernel_ir::OpId opId,
        std::span<const DeviceRunValue> inputs,
        std::span<const DeviceRunValue> outputs) {
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

    const cudaDeviceProp* deviceProps = nullptr;
    if (kernel.kind == ir::kernel_ir::OpKind::AttentionKernel) {
        auto props = ensure_device_properties();
        if (!props)
            return make_error(props.error());
        deviceProps = *props;
    }

    std::vector<CudaDeviceBufferView> inputViews;
    inputViews.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto* tensor = std::get_if<DeviceTensorView>(&input);
        if (tensor) {
            auto view = buffer_view(tensor->buffer, false);
            if (!view)
                return make_error(view.error());
            auto viewValue = view.take();
            viewValue.view = tensor->view;
            inputViews.push_back(std::move(viewValue));
            continue;
        }

        auto* paged = std::get_if<DevicePagedTensorView>(&input);
        if (!paged)
            return make_error("cuda device kernel input must be a tensor view");

        auto synced = sync_paged_tensor_table(paged->tensor);
        if (!synced)
            return make_error(synced.error());
        auto pagedView = paged_tensor_view(paged->tensor);
        if (!pagedView)
            return make_error(pagedView.error());
        auto pagedValue = pagedView.take();

        auto view = defaultView(pagedValue.meta.logicalDesc);
        if (!view)
            return make_error(view.error());
        inputViews.push_back(CudaDeviceBufferView{
            pagedValue.pageTable,
            view.take(),
            pagedValue.pageBytes * static_cast<size_t>(pagedValue.meta.pageCount),
            true,
            pagedValue.meta.growDim,
            pagedValue.meta.pageSize,
            pagedValue.meta.pageCount,
            pagedValue.meta.pageElementCount,
        });
    }

    std::vector<CudaDeviceBufferView> outputViews;
    outputViews.reserve(outputs.size());
    for (const auto& output : outputs) {
        auto* tensor = std::get_if<DeviceTensorView>(&output);
        if (!tensor)
            return make_error("cuda device kernel output must be a dense tensor view");
        auto view = buffer_view(tensor->buffer, true);
        if (!view)
            return make_error(view.error());
        auto viewValue = view.take();
        viewValue.view = tensor->view;
        outputViews.push_back(std::move(viewValue));
    }

    CudaLaunchContext context{
        cudaDevice_,
        stream_,
        cublasHandle_,
        opId,
        inputViews,
        outputViews,
        deviceProps,
        &workspace_,
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
        case ir::kernel_ir::OpKind::LinearKernel:
            return make_error("cuda linear kernel is not implemented");
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
        case ir::kernel_ir::OpKind::TopKKernel:
            return launch_cuda_topk(
                context,
                std::get<CudaTopKProgram>(kernel.program));
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
        case ir::kernel_ir::OpKind::AttentionKernel:
            return launch_cuda_attention(
                context,
                std::get<CudaAttentionProgram>(kernel.program));
        case ir::kernel_ir::OpKind::MoeGatherKernel:
            return launch_cuda_moe_gather(
                context,
                std::get<CudaMoeGatherProgram>(kernel.program));
        case ir::kernel_ir::OpKind::MoeMatMulKernel:
            return launch_cuda_moe_matmul(
                context,
                std::get<CudaMoeMatMulProgram>(kernel.program));
        case ir::kernel_ir::OpKind::MoeScatterSumKernel:
            return launch_cuda_moe_scatter_sum(context);
    }

    return make_error("cuda device cannot run unknown op kind");
}

Result<TensorBufferPtr> CudaDevice::read(DeviceTensorView src) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    auto it = buffers_.find(src.buffer);
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
        src.view.desc,
        std::move(data));
    return buffer;
}

Result<TensorBufferPtr> CudaDevice::read(DeviceBufferId src) {
    auto it = buffers_.find(src);
    if (it == buffers_.end())
        return make_error("cuda device buffer not found");
    auto view = defaultView(it->second.desc);
    if (!view)
        return make_error(view.error());
    return read(DeviceTensorView{src, view.take()});
}

} // namespace sandy::device
