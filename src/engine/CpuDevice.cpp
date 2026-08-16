#include "CpuDevice.h"

#include "ShapeUtil.h"
#include "TensorCalc.h"

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sandy::engine {

namespace {

class CpuTensorBuffer final : public core::TensorBuffer {
public:
    CpuTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
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
        return make_error("cpu device buffer must have static shape");
    return static_cast<size_t>(numel) * core::dtype_size(desc.dtype);
}

struct SimpleElementwiseKernel {
    ir::kernel_ir::ScalarOp scalarOp = ir::kernel_ir::ScalarOp::Constant;
    double constant = 0.0;
};

const ir::kernel_ir::ScalarNode* find_scalar(
        const std::vector<ir::kernel_ir::ScalarNode>& scalars,
        ir::kernel_ir::ScalarId id) {
    auto it = std::find_if(
        scalars.begin(),
        scalars.end(),
        [&](const ir::kernel_ir::ScalarNode& scalar) {
            return scalar.id == id;
        });
    if (it == scalars.end())
        return nullptr;
    return &*it;
}

bool is_load_of(
        const ir::kernel_ir::ScalarNode* scalar,
        uint32_t inputIndex) {
    return scalar &&
           scalar->op == ir::kernel_ir::ScalarOp::Load &&
           scalar->inputIndex == inputIndex &&
           scalar->operands.empty();
}

Result<SimpleElementwiseKernel> validate_simple_elementwise_kernel(
        const ir::kernel_ir::ElementwiseKernelOp& elementwise) {
    const auto& scalars = elementwise.scalars();
    auto* terminal = find_scalar(scalars, elementwise.result());
    if (!terminal)
        return make_error("cpu device elementwise result references missing scalar");

    SimpleElementwiseKernel simple;
    simple.scalarOp = terminal->op;
    simple.constant = terminal->constant;

    switch (terminal->op) {
        case ir::kernel_ir::ScalarOp::Constant:
            if (!elementwise.inputs().empty() || scalars.size() != 1 ||
                !terminal->operands.empty()) {
                return make_error("cpu device elementwise constant must be a single scalar op");
            }
            return simple;

        case ir::kernel_ir::ScalarOp::ReLU:
        case ir::kernel_ir::ScalarOp::Sqrt:
        case ir::kernel_ir::ScalarOp::Tanh: {
            if (elementwise.inputs().size() != 1 || scalars.size() != 2 ||
                terminal->operands.size() != 1) {
                return make_error("cpu device elementwise unary kernel must be a single op");
            }
            auto* load = find_scalar(scalars, terminal->operands[0]);
            if (!is_load_of(load, 0)) {
                return make_error("cpu device elementwise unary op must consume input0 load");
            }
            return simple;
        }

        case ir::kernel_ir::ScalarOp::Add:
        case ir::kernel_ir::ScalarOp::Mul: {
            if (elementwise.inputs().size() != 2 || scalars.size() != 3 ||
                terminal->operands.size() != 2) {
                return make_error("cpu device elementwise binary kernel must be a single op");
            }
            auto* lhs = find_scalar(scalars, terminal->operands[0]);
            auto* rhs = find_scalar(scalars, terminal->operands[1]);
            if (!is_load_of(lhs, 0) || !is_load_of(rhs, 1)) {
                return make_error(
                    "cpu device elementwise binary op must consume input0 and input1 loads");
            }
            return simple;
        }

        default:
            return make_error("cpu device unsupported elementwise scalar op");
    }
}

} // namespace

Result<DeviceCompiledGraphId> CpuDevice::compile(const ir::kernel_ir::Graph& graph) {
    auto verify = graph.verify();
    if (!verify)
        return make_error(verify.error());

    CpuDeviceGraph compiled;
    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        CpuDeviceKernel kernel;
        kernel.kind = op.kind();
        kernel.inputCount = op.inputs().size();
        kernel.outputCount = op.outputs().size();

        switch (op.kind()) {
            case ir::kernel_ir::OpKind::Input:
            case ir::kernel_ir::OpKind::DeviceTransfer:
                break;
            case ir::kernel_ir::OpKind::LayoutTransform: {
                const auto& layout = static_cast<const ir::kernel_ir::LayoutTransformOp&>(op);
                kernel.layoutTransform = layout.transform();
                kernel.dims = layout.dims();
                break;
            }
            case ir::kernel_ir::OpKind::ElementwiseKernel: {
                const auto& elementwise = static_cast<const ir::kernel_ir::ElementwiseKernelOp&>(op);
                auto simple = validate_simple_elementwise_kernel(elementwise);
                if (!simple)
                    return make_error(simple.error());
                kernel.scalarOp = simple->scalarOp;
                kernel.constant = simple->constant;
                break;
            }
            case ir::kernel_ir::OpKind::MatMulKernel: {
                const auto& matmul = static_cast<const ir::kernel_ir::MatMulKernelOp&>(op);
                kernel.transposeLhs = matmul.transposeLhs();
                kernel.transposeRhs = matmul.transposeRhs();
                break;
            }
            case ir::kernel_ir::OpKind::GatherKernel:
                break;
            case ir::kernel_ir::OpKind::SoftmaxKernel: {
                const auto& softmax = static_cast<const ir::kernel_ir::SoftmaxKernelOp&>(op);
                kernel.axis = softmax.axis();
                break;
            }
            case ir::kernel_ir::OpKind::NormKernel: {
                const auto& norm = static_cast<const ir::kernel_ir::NormKernelOp&>(op);
                kernel.norm = norm.norm();
                kernel.epsilon = norm.epsilon();
                break;
            }
            case ir::kernel_ir::OpKind::RoPEKernel: {
                const auto& rope = static_cast<const ir::kernel_ir::RoPEKernelOp&>(op);
                kernel.theta = rope.theta();
                kernel.rotaryDim = rope.rotaryDim();
                kernel.splitHalf = rope.splitHalf();
                break;
            }
            case ir::kernel_ir::OpKind::SlidingQueryKeyScoreKernel: {
                const auto& score =
                    static_cast<const ir::kernel_ir::SlidingQueryKeyScoreKernelOp&>(op);
                kernel.window = score.window();
                kernel.scale = score.scale();
                break;
            }
            case ir::kernel_ir::OpKind::ReductionKernel:
                return make_error("cpu device does not support reduction kernels yet");
            case ir::kernel_ir::OpKind::CustomKernel: {
                const auto& custom = static_cast<const ir::kernel_ir::CustomKernelOp&>(op);
                kernel.customName = custom.customName();
                break;
            }
        }

        compiled.kernels[op.id()] = std::move(kernel);
    }

    auto id = nextGraphId_++;
    graphs_[id] = std::move(compiled);
    return id;
}

Result<DeviceBufferId> CpuDevice::alloc(core::TensorDesc desc) {
    auto bytes = tensor_byte_size(desc);
    if (!bytes)
        return make_error(bytes.error());

    CpuDeviceBuffer buffer;
    buffer.desc = std::move(desc);
    buffer.data.resize(bytes.take());

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<void> CpuDevice::dealloc(DeviceBufferId buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    buffers_.erase(it);
    return {};
}

Result<DeviceBufferId> CpuDevice::load(core::TensorBuffer& src) {
    auto access = src.access();
    if (!access)
        return make_error(access.error());

    CpuDeviceBuffer buffer;
    buffer.desc = (*access).desc();
    buffer.borrowed.emplace(access.take());

    auto id = nextBufferId_++;
    buffers_[id] = std::move(buffer);
    return id;
}

Result<TensorBufferPtr> CpuDevice::read(DeviceBufferId src) {
    auto it = buffers_.find(src);
    if (it == buffers_.end())
        return make_error("cpu device buffer not found");
    auto data = it->second.borrowed ? it->second.borrowed->data() : std::span<const uint8_t>(it->second.data);
    TensorBufferPtr buffer = std::make_shared<CpuTensorBuffer>(
        it->second.desc,
        std::vector<uint8_t>(data.begin(), data.end()));
    return buffer;
}

Result<void> CpuDevice::run(
        DeviceCompiledGraphId graphId,
        ir::kernel_ir::OpId opId,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) {
    auto graphIt = graphs_.find(graphId);
    if (graphIt == graphs_.end())
        return make_error("cpu device compiled graph not found");
    auto kernelIt = graphIt->second.kernels.find(opId);
    if (kernelIt == graphIt->second.kernels.end())
        return make_error("cpu device kernel op not found");
    const auto& kernel = kernelIt->second;

    if (inputs.size() != kernel.inputCount)
        return make_error("cpu device input arity mismatch");
    if (outputs.size() != kernel.outputCount)
        return make_error("cpu device output arity mismatch");

    auto inputRef = [&](size_t index) -> Result<core::TensorRef> {
        auto it = buffers_.find(inputs[index]);
        if (it == buffers_.end())
            return make_error("cpu device input buffer not found");
        auto data = it->second.borrowed ? it->second.borrowed->data() : std::span<const uint8_t>(it->second.data);
        return core::make_tensor_ref(it->second.desc, data);
    };

    auto outputRef = [&](size_t index) -> Result<core::MutableTensorRef> {
        auto it = buffers_.find(outputs[index]);
        if (it == buffers_.end())
            return make_error("cpu device output buffer not found");
        if (it->second.borrowed)
            return make_error("cpu device output buffer is not writable");
        return core::make_mutable_tensor_ref(it->second.desc, it->second.data);
    };

    switch (kernel.kind) {
        case ir::kernel_ir::OpKind::Input:
            return make_error("cpu device cannot run input boundary op");
        case ir::kernel_ir::OpKind::DeviceTransfer:
            return make_error("cpu device cannot run device transfer boundary op");
        case ir::kernel_ir::OpKind::ElementwiseKernel: {
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            switch (kernel.scalarOp) {
                case ir::kernel_ir::ScalarOp::Constant: {
                    auto numel = out->desc.shape.numel();
                    if (numel < 0)
                        return make_error("constant output must have static shape");
                    for (int64_t i = 0; i < numel; i++)
                        out->store_float(static_cast<size_t>(i), static_cast<float>(kernel.constant));
                    return {};
                }
                case ir::kernel_ir::ScalarOp::ReLU: {
                    auto x = inputRef(0);
                    if (!x) return make_error(x.error());
                    return core::relu(*x, *out);
                }
                case ir::kernel_ir::ScalarOp::Sqrt: {
                    auto x = inputRef(0);
                    if (!x) return make_error(x.error());
                    return core::sqrt(*x, *out);
                }
                case ir::kernel_ir::ScalarOp::Tanh: {
                    auto x = inputRef(0);
                    if (!x) return make_error(x.error());
                    return core::tanh(*x, *out);
                }
                case ir::kernel_ir::ScalarOp::Add: {
                    auto lhs = inputRef(0);
                    if (!lhs) return make_error(lhs.error());
                    auto rhs = inputRef(1);
                    if (!rhs) return make_error(rhs.error());
                    return core::add(*lhs, *rhs, *out);
                }
                case ir::kernel_ir::ScalarOp::Mul: {
                    auto lhs = inputRef(0);
                    if (!lhs) return make_error(lhs.error());
                    auto rhs = inputRef(1);
                    if (!rhs) return make_error(rhs.error());
                    return core::mul(*lhs, *rhs, *out);
                }
                default:
                    return make_error("cpu device unsupported elementwise scalar op");
            }
        }
        case ir::kernel_ir::OpKind::MatMulKernel: {
            auto lhs = inputRef(0);
            if (!lhs) return make_error(lhs.error());
            auto rhs = inputRef(1);
            if (!rhs) return make_error(rhs.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::matmul(*lhs, *rhs, kernel.transposeLhs, kernel.transposeRhs, *out);
        }
        case ir::kernel_ir::OpKind::LayoutTransform: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            switch (kernel.layoutTransform) {
                case ir::kernel_ir::LayoutTransformKind::Reshape:
                    return core::reshape(*x, *out);
                case ir::kernel_ir::LayoutTransformKind::Transpose:
                    return core::transpose(*x, *out);
                case ir::kernel_ir::LayoutTransformKind::Permute:
                    return core::permute(*x, kernel.dims, *out);
                case ir::kernel_ir::LayoutTransformKind::Contiguous:
                    return core::reshape(*x, *out);
            }
            return make_error("cpu device unsupported layout transform");
        }
        case ir::kernel_ir::OpKind::SlidingQueryKeyScoreKernel: {
            auto q = inputRef(0);
            if (!q) return make_error(q.error());
            auto k = inputRef(1);
            if (!k) return make_error(k.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::sliding_query_key_score(
                *q,
                *k,
                kernel.window,
                static_cast<float>(kernel.scale),
                *out);
        }
        case ir::kernel_ir::OpKind::SoftmaxKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::softmax(*x, kernel.axis, *out);
        }
        case ir::kernel_ir::OpKind::GatherKernel: {
            auto ids = inputRef(0);
            if (!ids) return make_error(ids.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::embedding(*ids, *weight, *out);
        }
        case ir::kernel_ir::OpKind::RoPEKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::rope(
                *x,
                static_cast<float>(kernel.theta),
                kernel.rotaryDim,
                kernel.splitHalf,
                *out);
        }
        case ir::kernel_ir::OpKind::NormKernel: {
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            if (kernel.norm == ir::kernel_ir::NormKind::RMSNorm) {
                if (inputs.size() == 1)
                    return core::rms_norm(*x, static_cast<float>(kernel.epsilon), *out);
                if (inputs.size() != 2)
                    return make_error("rms_norm expects 1 or 2 inputs");
                auto weight = inputRef(1);
                if (!weight) return make_error(weight.error());
                return core::rms_norm(*x, *weight, static_cast<float>(kernel.epsilon), *out);
            }
            if (inputs.size() != 3)
                return make_error("layer_norm expects 3 inputs");
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto bias = inputRef(2);
            if (!bias) return make_error(bias.error());
            return core::layer_norm(*x, *weight, *bias, static_cast<float>(kernel.epsilon), *out);
        }
        case ir::kernel_ir::OpKind::CustomKernel: {
            if (kernel.customName != "linear")
                return make_error("cpu device unsupported custom kernel: " + kernel.customName);
            auto x = inputRef(0);
            if (!x) return make_error(x.error());
            auto weight = inputRef(1);
            if (!weight) return make_error(weight.error());
            auto bias = inputRef(2);
            if (!bias) return make_error(bias.error());
            auto out = outputRef(0);
            if (!out) return make_error(out.error());
            return core::linear(*x, *weight, *bias, *out);
        }
        case ir::kernel_ir::OpKind::ReductionKernel:
            return make_error("cpu device cannot run reduction kernel");
    }

    return make_error("cpu device cannot run unknown op kind");
}

} // namespace sandy::engine
