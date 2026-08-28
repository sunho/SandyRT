#include "CudaDevice.h"
#include "CudaScratchAllocator.h"
#include "CudaAttentionJit.h"
#include "CudaElementwiseJit.h"
#include "CudaGatherJit.h"
#include "CudaLayoutTransformJit.h"
#include "CudaNormJit.h"
#include "CudaReductionJit.h"
#include "CudaRoPEJit.h"
#include "CudaSoftmaxJit.h"

#include <cmath>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_set>
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

bool environment_flag(const char* name, bool defaultValue) {
    const char* value = std::getenv(name);
    if (!value)
        return defaultValue;
    std::string_view text(value);
    if (text == "0" || text == "false" || text == "off" || text == "no")
        return false;
    if (text == "1" || text == "true" || text == "on" || text == "yes")
        return true;
    return defaultValue;
}

int default_jit_access(const ir::kernel_ir::ValueType& type) {
    return type.kind == ir::kernel_ir::ValueKind::PagedTensor
        ? SANDY_JIT_PAGED
        : SANDY_JIT_CONTIGUOUS;
}

} // namespace

CudaDevice::CudaDevice(int cudaDevice)
    : cudaDevice_(cudaDevice) {
    auto set = cudaSetDevice(cudaDevice_);
    if (set != cudaSuccess) {
        initializationError_ =
            std::string("cudaSetDevice: ") + cudaGetErrorString(set);
        return;
    }

    auto queried = cudaGetDeviceProperties(&deviceProps_, cudaDevice_);
    if (queried != cudaSuccess) {
        initializationError_ =
            std::string("cudaGetDeviceProperties: ") + cudaGetErrorString(queried);
        return;
    }

    auto stream = ensure_stream();
    if (!stream) {
        initializationError_ = stream.error();
        return;
    }
    auto cublas = ensure_cublas_handle();
    if (!cublas) {
        initializationError_ = cublas.error();
        return;
    }
    auto validation = cudaMalloc(
        reinterpret_cast<void**>(&validationFailure_),
        sizeof(*validationFailure_));
    if (validation != cudaSuccess)
        initializationError_ =
            std::string("cudaMalloc validation failure flag: ") +
            cudaGetErrorString(validation);
}

CudaDevice::~CudaDevice() {
    cudaSetDevice(cudaDevice_);
    if (stream_)
        cudaStreamSynchronize(stream_);

    for (auto& [_, graph] : graphs_)
        destroy_captured_regions(graph);
    graphs_.clear();

    pagedTensors_.clear();
    pagedPools_.clear();

    if (validationFailure_) {
        cudaFree(validationFailure_);
        validationFailure_ = nullptr;
    }

    if (stream_) {
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
    if (!initializationError_.empty())
        return make_error(initializationError_);
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
    if (cublasHandle_)
        return {};

    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());

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

Result<DeviceCompiledGraphId> CudaDevice::compile(const ir::kernel_ir::Graph& graph) {
    std::vector<ir::kernel_ir::OpId> ops;
    ops.reserve(graph.ops().size());
    for (const auto& op : graph.ops())
        ops.push_back(op->id());
    return compileExecutableGraph(graph, ops);
}

Result<DeviceCompiledGraphId> CudaDevice::compileExecutableGraph(
        const ir::kernel_ir::Graph& graph,
        std::span<const ir::kernel_ir::OpId> ops) {
    auto verify = graph.verify();
    if (!verify)
        return make_error(verify.error());

    CudaDeviceGraph compiled;
    std::unordered_set<ir::kernel_ir::OpId> included(ops.begin(), ops.end());
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (!included.contains(op.id()))
            continue;
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
                CudaLayoutTransformProgram program{
                    layout.transform(),
                    layout.dims(),
                };
                if (environment_flag("SANDY_CUDA_LAYOUT_TRANSFORM_JIT", true)) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_LAYOUT_TRANSFORM_JIT_FALLBACK", false);
                    const auto& inputType = graph.value(op.inputs()[0]).type;
                    const auto& outputType = graph.value(op.outputs()[0]).type;
                    const int accesses[] = {
                        default_jit_access(inputType),
                        default_jit_access(outputType),
                    };
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses),
                        [&] {
                            return compileCudaLayoutTransformJit(
                                cudaDevice_, jitCache_, inputType.dtype,
                                accesses[0], accesses[1]);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError)
                            return make_error(
                                "CUDA layout transform JIT compile failed: " + jit.error());
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
                break;
            }
            case ir::kernel_ir::OpKind::ElementwiseKernel: {
                const auto& elementwise =
                    static_cast<const ir::kernel_ir::ElementwiseKernelOp&>(op);
                CudaElementwiseProgram program{
                    elementwise.elementwiseInputs(),
                    elementwise.output(),
                    elementwise.result(),
                    elementwise.scalars(),
                };
                std::vector<int> inputAccesses;
                inputAccesses.reserve(program.elementwiseInputs.size());
                for (const auto& input : program.elementwiseInputs)
                    inputAccesses.push_back(default_jit_access(graph.value(input.value).type));
                int outputAccess =
                    default_jit_access(graph.value(program.output).type);
                std::vector<int> accesses = inputAccesses;
                accesses.push_back(outputAccess);
                auto variants = std::make_shared<CudaJitVariants>();
                auto jit = variants->getOrCompile(
                    cudaJitAccessKey(accesses),
                    [&] {
                        return compileCudaElementwiseJit(
                            cudaDevice_, jitCache_, program,
                            inputAccesses, outputAccess);
                    });
                if (!jit)
                    return make_error("CUDA elementwise JIT compile failed: " + jit.error());
                program.jitVariants = std::move(variants);
                kernel.program = std::move(program);
                break;
            }
            case ir::kernel_ir::OpKind::ReductionKernel: {
                const auto& reduction =
                    static_cast<const ir::kernel_ir::ReductionKernelOp&>(op);
                CudaReductionProgram program{
                    reduction.reduce(),
                    reduction.axes(),
                    reduction.keepDims(),
                };
                if (environment_flag("SANDY_CUDA_REDUCTION_JIT", true)) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_REDUCTION_JIT_FALLBACK", false);
                    const auto& inputType = graph.value(op.inputs()[0]).type;
                    const auto& outputType = graph.value(op.outputs()[0]).type;
                    const int accesses[] = {
                        default_jit_access(inputType),
                        default_jit_access(outputType),
                    };
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses),
                        [&] {
                            return compileCudaReductionJit(
                                cudaDevice_, jitCache_, inputType.dtype,
                                accesses[0], accesses[1]);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError)
                            return make_error("CUDA reduction JIT compile failed: " + jit.error());
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
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
            case ir::kernel_ir::OpKind::GatherKernel: {
                CudaGatherProgram program;
                if (environment_flag("SANDY_CUDA_GATHER_JIT", true)) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_GATHER_JIT_FALLBACK", false);
                    const auto inputs = op.inputs();
                    const auto& idsType = graph.value(inputs[0]).type;
                    const auto& tableType = graph.value(inputs[1]).type;
                    const auto& outputType = graph.value(op.outputs()[0]).type;
                    program.idsDtype = idsType.dtype;
                    program.valueDtype = tableType.dtype;
                    program.tableRank = tableType.shape.rank();
                    const int accesses[] = {
                        default_jit_access(idsType),
                        default_jit_access(tableType),
                        default_jit_access(outputType),
                    };
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses),
                        [&] {
                            return compileCudaGatherJit(
                                cudaDevice_, jitCache_, program.idsDtype,
                                program.valueDtype, program.tableRank,
                                accesses[0], accesses[1], accesses[2]);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError)
                            return make_error("CUDA gather JIT compile failed: " + jit.error());
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
                break;
            }
            case ir::kernel_ir::OpKind::SoftmaxKernel: {
                const auto& softmax = static_cast<const ir::kernel_ir::SoftmaxKernelOp&>(op);
                CudaSoftmaxProgram program{softmax.axis()};
                if (environment_flag("SANDY_CUDA_SOFTMAX_JIT", true)) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_SOFTMAX_JIT_FALLBACK", false);
                    const auto& inputType = graph.value(op.inputs()[0]).type;
                    const auto& outputType = graph.value(op.outputs()[0]).type;
                    program.dtype = inputType.dtype;
                    const int accesses[] = {
                        default_jit_access(inputType),
                        default_jit_access(outputType),
                    };
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses),
                        [&] {
                            return compileCudaSoftmaxJit(
                                cudaDevice_, jitCache_, program.dtype,
                                accesses[0], accesses[1]);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError)
                            return make_error("CUDA softmax JIT compile failed: " + jit.error());
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
                break;
            }
            case ir::kernel_ir::OpKind::TopKKernel: {
                const auto& topk = static_cast<const ir::kernel_ir::TopKKernelOp&>(op);
                kernel.program = CudaTopKProgram{topk.k(), topk.axis()};
                break;
            }
            case ir::kernel_ir::OpKind::NormKernel: {
                const auto& norm = static_cast<const ir::kernel_ir::NormKernelOp&>(op);
                CudaNormProgram program{norm.norm(), norm.epsilon()};
                if (environment_flag("SANDY_CUDA_NORM_JIT", true)) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_NORM_JIT_FALLBACK", false);
                    program.hasWeight = norm.norm() == ir::kernel_ir::NormKind::LayerNorm ||
                        op.inputs().size() == 2;
                    program.dtype = graph.value(op.inputs()[0]).type.dtype;
                    std::vector<int> inputAccesses;
                    inputAccesses.reserve(op.inputs().size());
                    for (auto input : op.inputs())
                        inputAccesses.push_back(default_jit_access(graph.value(input).type));
                    int outputAccess =
                        default_jit_access(graph.value(op.outputs()[0]).type);
                    std::vector<int> accesses = inputAccesses;
                    accesses.push_back(outputAccess);
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses),
                        [&] {
                            return compileCudaNormJit(
                                cudaDevice_, jitCache_, program.norm,
                                program.hasWeight, program.dtype,
                                inputAccesses, outputAccess);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError)
                            return make_error("CUDA norm JIT compile failed: " + jit.error());
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
                break;
            }
            case ir::kernel_ir::OpKind::RoPEKernel: {
                const auto& rope = static_cast<const ir::kernel_ir::RoPEKernelOp&>(op);
                CudaRoPEProgram program{
                    rope.theta(),
                    rope.rotaryDim(),
                    rope.splitHalf(),
                };
                if (environment_flag("SANDY_CUDA_ROPE_JIT", true)) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_ROPE_JIT_FALLBACK", false);
                    program.dtype = graph.value(op.inputs()[0]).type.dtype;
                    program.hasPositions = op.inputs().size() == 2;
                    if (program.hasPositions)
                        program.positionDtype = graph.value(op.inputs()[1]).type.dtype;
                    const int accesses[] = {
                        default_jit_access(graph.value(op.inputs()[0]).type),
                        program.hasPositions
                            ? default_jit_access(graph.value(op.inputs()[1]).type)
                            : SANDY_JIT_CONTIGUOUS,
                        default_jit_access(graph.value(op.outputs()[0]).type),
                    };
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses), [&] {
                            return compileCudaRoPEJit(
                                cudaDevice_, jitCache_, program.dtype,
                                program.splitHalf, program.hasPositions,
                                program.positionDtype, accesses[0], accesses[1], accesses[2]);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError)
                            return make_error("CUDA RoPE JIT compile failed: " + jit.error());
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
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
                CudaAttentionProgram program{
                    attention.window(), attention.scale()};
                const auto& qType = graph.value(op.inputs()[0]).type;
                const auto& kType = graph.value(op.inputs()[1]).type;
                const auto& vType = graph.value(op.inputs()[2]).type;
                const auto& outputType = graph.value(op.outputs()[0]).type;
                program.dtype = qType.dtype;
                program.rank = qType.shape.rank();
                program.headDim = qType.shape.dim(program.rank - 1);
                auto qHeads = qType.shape.dim(program.rank - 3);
                auto kvHeads = kType.shape.dim(program.rank - 3);
                program.queryHeadsPerKv = kvHeads > 0 ? qHeads / kvHeads : 0;
                program.hasPositionOffsets = op.inputs().size() == 4;
                if (program.hasPositionOffsets) {
                    program.positionDtype =
                        graph.value(op.inputs()[3]).type.dtype;
                }
                if (environment_flag("SANDY_CUDA_ATTENTION_DECODE_JIT", true) &&
                    qType.shape.dim(program.rank - 2) == 1 &&
                    default_jit_access(kType) == SANDY_JIT_PAGED &&
                    default_jit_access(vType) == SANDY_JIT_PAGED) {
                    program.jitFallbackOnError = environment_flag(
                        "SANDY_CUDA_ATTENTION_DECODE_JIT_FALLBACK", false);
                    const int accesses[] = {
                        default_jit_access(qType),
                        SANDY_JIT_PAGED,
                        SANDY_JIT_PAGED,
                        program.hasPositionOffsets
                            ? default_jit_access(
                                  graph.value(op.inputs()[3]).type)
                            : SANDY_JIT_CONTIGUOUS,
                        default_jit_access(outputType),
                    };
                    float scale = program.scale > 0.0
                        ? static_cast<float>(program.scale)
                        : 1.0f / std::sqrt(
                              static_cast<float>(program.headDim));
                    auto variants = std::make_shared<CudaJitVariants>();
                    auto jit = variants->getOrCompile(
                        cudaJitAccessKey(accesses), [&] {
                            return compileCudaAttentionDecodeJit(
                                cudaDevice_, jitCache_, program.dtype,
                                accesses[0], accesses[4], program.rank,
                                program.headDim, program.queryHeadsPerKv,
                                program.hasPositionOffsets,
                                program.positionDtype, accesses[3],
                                program.window, scale);
                        });
                    if (!jit) {
                        if (!program.jitFallbackOnError) {
                            return make_error(
                                "CUDA attention decode JIT compile failed: " +
                                jit.error());
                        }
                    } else {
                        program.jitVariants = std::move(variants);
                    }
                }
                kernel.program = std::move(program);
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
    if (environment_flag("SANDY_CUDA_JIT_STATS", false)) {
        auto stats = jitCache_.stats();
        std::fprintf(
            stderr,
            "cuda_jit.stats graph=%u hits=%zu misses=%zu modules=%zu compile_ms=%.3f\n",
            id,
            stats.hits,
            stats.misses,
            stats.entries,
            stats.compileMilliseconds);
    }
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
        auto allocated = cuda_malloc_stream_ordered(
            &data,
            *bytes,
            stream_,
            "cudaMallocAsync device buffer");
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

Result<void> CudaDevice::destroyCompiledGraph(DeviceCompiledGraphId graph) {
    auto found = graphs_.find(graph);
    if (found == graphs_.end())
        return make_error("cuda device compiled graph not found");
    destroy_captured_regions(found->second);
    graphs_.erase(found);
    return {};
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
            "cudaFreeAsync device buffer");
        if (!freed)
            return make_error(freed.error());
    }
    buffers_.erase(it);
    return {};
}

std::unique_ptr<DeviceScratchAllocator> CudaDevice::createScratchAllocator() {
    return std::make_unique<CudaScratchAllocator>();
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
    if (executableRunActive_)
        return run_current(graphId, opId, inputs, outputs);
    auto begun = begin_validation();
    if (!begun)
        return make_error(begun.error());
    auto ran = run_current(graphId, opId, inputs, outputs);
    if (!ran)
        return make_error(ran.error());
    return end_validation();
}

Result<void> CudaDevice::begin_validation() {
    if (!validationFailure_)
        return make_error("cuda validation failure flag is not initialized");
    return cuda_check(
        cudaMemsetAsync(
            validationFailure_,
            0xff,
            sizeof(*validationFailure_),
            stream_),
        "cudaMemsetAsync validation failure flag");
}

Result<void> CudaDevice::end_validation() {
    ir::kernel_ir::OpId failure = ir::kernel_ir::kInvalidOpId;
    auto copied = cuda_check(
        cudaMemcpyAsync(
            &failure,
            validationFailure_,
            sizeof(failure),
            cudaMemcpyDeviceToHost,
            stream_),
        "cudaMemcpyAsync validation failure op");
    if (!copied)
        return make_error(copied.error());
    auto synced = cuda_check(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize executable validation");
    if (!synced)
        return make_error(synced.error());
    if (failure != ir::kernel_ir::kInvalidOpId)
        return make_error(
            "cuda executable validation failed at op %" +
            std::to_string(failure));
    return {};
}

Result<void> CudaDevice::beginExecutableRun() {
    if (executableRunActive_)
        return make_error("cuda executable run is already active");
    auto begun = begin_validation();
    if (!begun)
        return make_error(begun.error());
    executableRunActive_ = true;
    return {};
}

Result<void> CudaDevice::endExecutableRun() {
    if (!executableRunActive_)
        return make_error("cuda executable run is not active");
    executableRunActive_ = false;
    return end_validation();
}

void CudaDevice::abortExecutableRun() {
    executableRunActive_ = false;
}

bool CudaDevice::is_capture_eligible(
        const CudaDeviceKernel& kernel,
        const DeviceRunCommand& command) const {
    if (!command.bindingsFixed)
        return false;
    switch (kernel.kind) {
        case ir::kernel_ir::OpKind::MoeMatMulKernel:
            return false;
        default:
            return true;
    }
}

Result<CudaDevice::CudaGraphRegionKey> CudaDevice::graph_region_key(
        std::span<const DeviceRunCommand> commands) const {
    CudaGraphRegionKey key;
    key.ops.reserve(commands.size());
    for (const auto& command : commands) {
        key.ops.push_back(command.op);
        auto append = [&](const DeviceRunValue& value) -> Result<void> {
            const auto* tensor = std::get_if<DeviceTensorView>(&value);
            if (!tensor)
                return make_error("cuda graph fixed binding must be a dense tensor view");
            key.bindings.push_back(CudaGraphTensorBinding{
                tensor->buffer,
                tensor->view.desc.dtype,
                tensor->view.desc.shape.dims(),
                tensor->view.strides,
                tensor->view.storageOffset,
            });
            return {};
        };
        for (const auto& input : command.inputs) {
            auto added = append(input);
            if (!added)
                return make_error(added.error());
        }
        for (const auto& output : command.outputs) {
            auto added = append(output);
            if (!added)
                return make_error(added.error());
        }
    }
    return key;
}

Result<void> CudaDevice::execute_eager_commands(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands) {
    for (const auto& command : commands) {
        auto ran = run_current(graph, command.op, command.inputs, command.outputs);
        if (!ran)
            return make_error(ran.error());
    }
    return {};
}

Result<void> CudaDevice::capture_region(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands,
        CudaCapturedRegion& region) {
    auto disableAndRunEager = [&]() -> Result<void> {
        region.captureDisabled = true;
        graphCacheStats_.captureFailures++;
        graphCacheStats_.eagerRegions++;
        (void)cudaGetLastError();
        return execute_eager_commands(graph, commands);
    };

    auto synchronized = cuda_check(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize before CUDA graph capture");
    if (!synchronized)
        return make_error(synchronized.error());

    auto begun = cuda_check(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
    if (!begun)
        return disableAndRunEager();

    auto recorded = execute_eager_commands(graph, commands);
    if (!recorded) {
        cudaGraph_t abandoned = nullptr;
        (void)cudaStreamEndCapture(stream_, &abandoned);
        if (abandoned)
            (void)cudaGraphDestroy(abandoned);
        return disableAndRunEager();
    }

    auto ended = cuda_check(
        cudaStreamEndCapture(stream_, &region.graph),
        "cudaStreamEndCapture");
    if (!ended) {
        region.graph = nullptr;
        return disableAndRunEager();
    }

    auto instantiated = cuda_check(
        cudaGraphInstantiate(
            &region.graphExec,
            region.graph,
            nullptr,
            nullptr,
            0),
        "cudaGraphInstantiate");
    if (!instantiated) {
        (void)cudaGraphDestroy(region.graph);
        region.graph = nullptr;
        return disableAndRunEager();
    }

    auto launched = cuda_check(
        cudaGraphLaunch(region.graphExec, stream_),
        "cudaGraphLaunch after capture");
    if (!launched)
        return make_error(launched.error());
    graphCacheStats_.captures++;
    return {};
}

Result<void> CudaDevice::execute_recordable_region(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands) {
    auto key = graph_region_key(commands);
    if (!key)
        return make_error(key.error());
    auto graphIt = graphs_.find(graph);
    if (graphIt == graphs_.end())
        return make_error("cuda device compiled graph not found");

    for (auto& region : graphIt->second.capturedRegions) {
        if (region.key != *key)
            continue;
        if (region.captureDisabled) {
            graphCacheStats_.eagerRegions++;
            return execute_eager_commands(graph, commands);
        }
        auto replayed = cuda_check(
            cudaGraphLaunch(region.graphExec, stream_),
            "cudaGraphLaunch");
        if (!replayed)
            return make_error(replayed.error());
        graphCacheStats_.replays++;
        return {};
    }

    CudaCapturedRegion region;
    region.key = key.take();
    graphIt->second.capturedRegions.push_back(std::move(region));
    return capture_region(
        graph,
        commands,
        graphIt->second.capturedRegions.back());
}

Result<void> CudaDevice::executeCommands(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands,
        const std::function<void(ir::kernel_ir::OpId, double)>& profileKernel) {
    auto stream = ensure_stream();
    if (!stream)
        return make_error(stream.error());
    if (profileKernel)
        return Device::executeCommands(graph, commands, profileKernel);

    auto graphIt = graphs_.find(graph);
    if (graphIt == graphs_.end())
        return make_error("cuda device compiled graph not found");

    size_t begin = 0;
    while (begin < commands.size()) {
        auto kernel = graphIt->second.kernels.find(commands[begin].op);
        if (kernel == graphIt->second.kernels.end())
            return make_error("cuda device kernel op not found");
        bool recordable = is_capture_eligible(kernel->second, commands[begin]);
        size_t end = begin + 1;
        while (end < commands.size()) {
            auto next = graphIt->second.kernels.find(commands[end].op);
            if (next == graphIt->second.kernels.end())
                return make_error("cuda device kernel op not found");
            if (is_capture_eligible(next->second, commands[end]) != recordable)
                break;
            ++end;
        }

        auto region = commands.subspan(begin, end - begin);
        Result<void> executed = recordable
            ? execute_recordable_region(graph, region)
            : execute_eager_commands(graph, region);
        if (!executed)
            return make_error(executed.error());
        if (!recordable)
            graphCacheStats_.eagerRegions++;
        begin = end;
    }
    return {};
}

void CudaDevice::destroy_captured_regions(CudaDeviceGraph& graph) {
    for (auto& region : graph.capturedRegions) {
        if (region.graphExec)
            (void)cudaGraphExecDestroy(region.graphExec);
        if (region.graph)
            (void)cudaGraphDestroy(region.graph);
        region.graphExec = nullptr;
        region.graph = nullptr;
    }
    graph.capturedRegions.clear();
}

Result<void> CudaDevice::run_current(
        DeviceCompiledGraphId graphId,
        ir::kernel_ir::OpId opId,
        std::span<const DeviceRunValue> inputs,
        std::span<const DeviceRunValue> outputs) {
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

    constexpr size_t kMaxRunInputs = 8;
    constexpr size_t kMaxRunOutputs = 4;
    if (inputs.size() > kMaxRunInputs)
        return make_error("cuda device kernel input count exceeds run capacity");
    if (outputs.size() > kMaxRunOutputs)
        return make_error("cuda device kernel output count exceeds run capacity");

    std::array<CudaDeviceBufferView, kMaxRunInputs> inputViews;
    size_t inputViewCount = 0;
    for (const auto& input : inputs) {
        auto* tensor = std::get_if<DeviceTensorView>(&input);
        if (tensor) {
            auto view = buffer_view(tensor->buffer, false);
            if (!view)
                return make_error(view.error());
            auto viewValue = view.take();
            viewValue.view = tensor->view;
            inputViews[inputViewCount++] = std::move(viewValue);
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
        inputViews[inputViewCount++] = CudaDeviceBufferView{
            pagedValue.pageTable,
            view.take(),
            pagedValue.pageBytes * static_cast<size_t>(pagedValue.meta.pageCount),
            true,
            pagedValue.meta.growDim,
            pagedValue.meta.pageSize,
            pagedValue.meta.pageCount,
            pagedValue.meta.pageElementCount,
        };
    }

    std::array<CudaDeviceBufferView, kMaxRunOutputs> outputViews;
    size_t outputViewCount = 0;
    for (const auto& output : outputs) {
        auto* tensor = std::get_if<DeviceTensorView>(&output);
        if (!tensor)
            return make_error("cuda device kernel output must be a dense tensor view");
        auto view = buffer_view(tensor->buffer, true);
        if (!view)
            return make_error(view.error());
        auto viewValue = view.take();
        viewValue.view = tensor->view;
        outputViews[outputViewCount++] = std::move(viewValue);
    }

    CudaLaunchContext context{
        cudaDevice_,
        stream_,
        cublasHandle_,
        opId,
        validationFailure_,
        std::span<const CudaDeviceBufferView>(inputViews.data(), inputViewCount),
        std::span<const CudaDeviceBufferView>(outputViews.data(), outputViewCount),
        kernel.kind == ir::kernel_ir::OpKind::AttentionKernel ? &deviceProps_ : nullptr,
        &jitCache_,
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
            return launch_cuda_gather(
                context,
                std::get<CudaGatherProgram>(kernel.program));
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

    auto expectedBytes = tensor_byte_size(src.view.desc);
    if (!expectedBytes)
        return make_error(expectedBytes.error());
    auto defaultView = isDefaultView(src.view);
    if (!defaultView)
        return make_error(defaultView.error());

    size_t sourceOffset = 0;
    if (*defaultView) {
        if (src.view.storageOffset < 0)
            return make_error("cuda device view has a negative storage offset");
        auto elementBytes = core::dtype_size(src.view.desc.dtype);
        if (static_cast<size_t>(src.view.storageOffset) >
            std::numeric_limits<size_t>::max() / elementBytes)
            return make_error("cuda device view storage offset overflow");
        sourceOffset = static_cast<size_t>(src.view.storageOffset) * elementBytes;
        if (sourceOffset > it->second.bytes ||
            *expectedBytes > it->second.bytes - sourceOffset)
            return make_error("cuda device view exceeds its backing buffer");
    } else if (src.view.storageOffset != 0 || *expectedBytes != it->second.bytes) {
        return make_error("cuda device cannot read a non-contiguous subview");
    }

    std::vector<uint8_t> data(*expectedBytes);
    if (*expectedBytes != 0) {
        auto* source = static_cast<const uint8_t*>(it->second.data) + sourceOffset;
        auto copied = cuda_check(
            cudaMemcpyAsync(
                data.data(),
                source,
                *expectedBytes,
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
