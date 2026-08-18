#include "Engine.h"

#include "DeviceWiseCopier.h"
#include "MidIRToKernelIR.h"
#include "ShapeUtil.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace sandy::engine {

using device::Device;
using device::DeviceWiseCopier;
using device::HostBounceDeviceWiseCopier;

namespace {

using ir::kernel_ir::Graph;
using ir::kernel_ir::DeviceTransferOp;
using ir::kernel_ir::DeviceId;
using ir::kernel_ir::InputOp;
using ir::kernel_ir::InputSourceKind;
using ir::kernel_ir::LayoutTransformKind;
using ir::kernel_ir::Op;
using ir::kernel_ir::OpKind;
using ir::kernel_ir::PagedAppendOp;
using ir::kernel_ir::ValueId;
using ir::kernel_ir::ValueType;

using Clock = std::chrono::steady_clock;

void profile_stage(
        const EngineRunOptions* options,
        const std::string& stage,
        Clock::time_point start,
        Clock::time_point end,
        size_t opIndex = 0,
        ir::kernel_ir::OpId op = ir::kernel_ir::kInvalidOpId,
        OpKind opKind = OpKind::Input) {
    if (!options || !options->profileStage)
        return;
    options->profileStage({
        stage,
        opIndex,
        op,
        opKind,
        std::chrono::duration<double, std::milli>(end - start).count(),
    });
}

void profile_device_run_boundary(
        const EngineRunOptions* options,
        EngineDeviceRunBoundaryEvent::Boundary boundary,
        size_t opIndex,
        ir::kernel_ir::OpId op,
        DeviceId device,
        DeviceCompiledGraphId deviceGraph,
        OpKind opKind,
        size_t inputCount,
        size_t outputCount,
        double elapsedMs = 0.0) {
    if (!options || !options->profileDeviceRunBoundary)
        return;
    options->profileDeviceRunBoundary({
        boundary,
        opIndex,
        op,
        device,
        deviceGraph,
        opKind,
        inputCount,
        outputCount,
        elapsedMs,
    });
}

struct RuntimeState {
    std::unordered_map<ValueId, DeviceBufferId> buffers;
    std::unordered_map<ValueId, DevicePagedTensorView> pagedTensors;
    std::unordered_map<ValueId, TensorViewDesc> views;
    std::unordered_map<ValueId, uint32_t> bufferDevices;
    std::unordered_map<ValueId, std::vector<ValueId>> tensorTuples;
    std::unordered_set<ValueId> borrowedBuffers;
    std::vector<size_t> remainingUses;
    std::vector<bool> isOutput;
};

Result<Device*> lookup_device(
        std::vector<std::unique_ptr<Device>>& devices,
        uint32_t device) {
    if (device >= devices.size())
        return make_error("invalid device id: " + std::to_string(device));
    if (!devices[device])
        return make_error("null device: " + std::to_string(device));
    return devices[device].get();
}

Result<DeviceCompiledGraphId> lookup_device_graph(
        const CompiledKernelGraph& compiled,
        DeviceId device) {
    auto it = compiled.deviceGraphs.find(device);
    if (it != compiled.deviceGraphs.end())
        return it->second;
    if (device == compiled.device)
        return compiled.deviceGraph;
    return make_error("missing compiled device graph for device: " + std::to_string(device));
}

DeviceId default_runtime_device(const CompiledKernelGraph& compiled) {
    if (compiled.deviceGraphs.empty())
        return compiled.device;
    return compiled.defaultDevice;
}

DeviceId runtime_op_device(
        const CompiledKernelGraph& compiled,
        const Op& op) {
    if (compiled.deviceGraphs.empty())
        return compiled.device;
    return op.device();
}

Result<DeviceBufferId> lookup_runtime_buffer(
        const std::unordered_map<ValueId, DeviceBufferId>& buffers,
        ValueId value) {
    auto it = buffers.find(value);
    if (it == buffers.end())
        return make_error("missing runtime buffer for value: " + std::to_string(value));
    return it->second;
}

Result<core::TensorDesc> lookup_runtime_desc(
        const std::unordered_map<ValueId, TensorViewDesc>& views,
        ValueId value) {
    auto it = views.find(value);
    if (it == views.end())
        return make_error("missing runtime descriptor for value: " + std::to_string(value));
    return it->second.desc;
}

Result<TensorViewDesc> lookup_runtime_view(
        const std::unordered_map<ValueId, TensorViewDesc>& views,
        ValueId value) {
    auto it = views.find(value);
    if (it == views.end())
        return make_error("missing runtime view for value: " + std::to_string(value));
    return it->second;
}

Result<void> verify_desc_matches_type(
        const core::TensorDesc& desc,
        const ValueType& type,
        const std::string& valueName) {
    if (desc.dtype != type.dtype) {
        return make_error(valueName + " dtype mismatch: expected " +
                          std::string(core::dtype_name(type.dtype)) + ", got " +
                          core::dtype_name(desc.dtype));
    }
    if (desc.shape.has_dynamic())
        return make_error(valueName + " runtime shape must be static");
    if (desc.shape.rank() != type.shape.rank()) {
        return make_error(valueName + " rank mismatch: expected " +
                          std::to_string(type.shape.rank()) + ", got " +
                          std::to_string(desc.shape.rank()));
    }
    for (int i = 0; i < type.shape.rank(); i++) {
        int64_t expected = type.shape.dim(i);
        if (expected != core::Shape::kDynamic && expected != desc.shape.dim(i)) {
            return make_error(valueName + " dimension " + std::to_string(i) +
                              " mismatch: expected " + std::to_string(expected) +
                              ", got " + std::to_string(desc.shape.dim(i)));
        }
    }
    return {};
}

Result<core::TensorDesc> desc_from_static_type(const ValueType& type) {
    if (type.shape.has_dynamic())
        return make_error("cannot allocate unresolved dynamic KernelIR value");
    return core::TensorDesc(type.shape, type.dtype);
}

Result<core::TensorDesc> desc_with_shape(
        const Graph& graph,
        ValueId value,
        core::Shape shape) {
    const auto& type = graph.value(value).type;
    core::TensorDesc desc(std::move(shape), type.dtype);
    auto verify = verify_desc_matches_type(desc, type, "value %" + std::to_string(value));
    if (!verify)
        return make_error(verify.error());
    return desc;
}

Result<core::TensorDesc> resolve_matmul_desc(
        const Graph& graph,
        const ir::kernel_ir::MatMulKernelOp& op,
        ValueId output,
        const std::unordered_map<ValueId, TensorViewDesc>& views) {
    auto lhsDesc = lookup_runtime_desc(views, op.inputs()[0]);
    if (!lhsDesc)
        return make_error(lhsDesc.error());
    auto rhsDesc = lookup_runtime_desc(views, op.inputs()[1]);
    if (!rhsDesc)
        return make_error(rhsDesc.error());

    const auto& lhsShape = lhsDesc->shape;
    const auto& rhsShape = rhsDesc->shape;
    auto lhsDims = lhsShape.dims();
    auto rhsDims = rhsShape.dims();
    core::Shape lhsBatch(std::vector<int64_t>(lhsDims.begin(), lhsDims.end() - 2));
    core::Shape rhsBatch(std::vector<int64_t>(rhsDims.begin(), rhsDims.end() - 2));
    auto batch = core::matmul_batch_shape(lhsBatch, rhsBatch);
    if (!batch)
        return make_error(batch.error());
    auto outDims = batch.take().dims();
    outDims.push_back(lhsShape.dim(lhsShape.rank() - (op.transposeLhs() ? 1 : 2)));
    outDims.push_back(rhsShape.dim(rhsShape.rank() - (op.transposeRhs() ? 2 : 1)));
    return desc_with_shape(graph, output, core::Shape(std::move(outDims)));
}

Result<core::TensorDesc> resolve_reduction_desc(
        const Graph& graph,
        const ir::kernel_ir::ReductionKernelOp& op,
        ValueId output,
        const std::unordered_map<ValueId, TensorViewDesc>& views) {
    auto input = lookup_runtime_desc(views, op.inputs()[0]);
    if (!input)
        return make_error(input.error());

    int rank = input->shape.rank();
    std::vector<int> axes;
    axes.reserve(op.axes().size());
    for (auto axis : op.axes()) {
        if (axis < -rank || axis >= rank)
            return make_error("reduction axis out of range");
        axis = axis < 0 ? axis + rank : axis;
        for (auto existing : axes) {
            if (existing == axis)
                return make_error("reduction axes must be unique");
        }
        axes.push_back(static_cast<int>(axis));
    }

    auto dims = input->shape.dims();
    if (op.keepDims()) {
        for (auto axis : axes)
            dims[static_cast<size_t>(axis)] = 1;
    } else {
        std::sort(axes.begin(), axes.end(), [](int lhs, int rhs) {
            return lhs > rhs;
        });
        for (auto axis : axes)
            dims.erase(dims.begin() + axis);
    }
    return desc_with_shape(graph, output, core::Shape(std::move(dims)));
}

Result<core::TensorDesc> resolve_topk_desc(
        const Graph& graph,
        const ir::kernel_ir::TopKKernelOp& op,
        ValueId output,
        const std::unordered_map<ValueId, TensorViewDesc>& views) {
    auto input = lookup_runtime_desc(views, op.inputs()[0]);
    if (!input)
        return make_error(input.error());

    int rank = input->shape.rank();
    int64_t axis = op.axis();
    if (axis < -rank || axis >= rank)
        return make_error("topk axis out of range");
    axis = axis < 0 ? axis + rank : axis;

    auto dims = input->shape.dims();
    dims[static_cast<size_t>(axis)] = op.k();
    return desc_with_shape(graph, output, core::Shape(std::move(dims)));
}

Result<core::TensorDesc> resolve_moe_gather_desc(
        const Graph& graph,
        const ir::kernel_ir::MoeGatherKernelOp& op,
        ValueId output,
        const std::unordered_map<ValueId, TensorViewDesc>& views) {
    auto x = lookup_runtime_desc(views, op.inputs()[0]);
    if (!x)
        return make_error(x.error());

    int rank = x->shape.rank();
    if (rank != 2 && rank != 3)
        return make_error("moe_gather input rank must be 2 or 3");

    int64_t seq = x->shape.dim(rank == 3 ? 1 : 0);
    int64_t hidden = x->shape.dim(rank - 1);
    if (seq < 0 || hidden < 0)
        return make_error("moe_gather runtime dimensions must be static");
    int64_t rows = seq * op.topK();

    auto outputs = op.outputs();
    if (output == outputs[0]) {
        if (rank == 3) {
            return desc_with_shape(
                graph,
                output,
                core::Shape({x->shape.dim(0), rows, hidden}));
        }
        return desc_with_shape(graph, output, core::Shape({rows, hidden}));
    }
    if (output == outputs[1] || output == outputs[2]) {
        if (rank == 3) {
            return desc_with_shape(
                graph,
                output,
                core::Shape({x->shape.dim(0), rows}));
        }
        return desc_with_shape(graph, output, core::Shape({rows}));
    }
    if (output == outputs[3]) {
        if (rank == 3) {
            return desc_with_shape(
                graph,
                output,
                core::Shape({x->shape.dim(0), op.numExperts() + 1}));
        }
        return desc_with_shape(graph, output, core::Shape({op.numExperts() + 1}));
    }

    return make_error("moe_gather unknown output");
}

Result<core::TensorDesc> resolve_moe_matmul_desc(
        const Graph& graph,
        const ir::kernel_ir::MoeMatMulKernelOp& op,
        ValueId output,
        const std::unordered_map<ValueId, TensorViewDesc>& views) {
    auto x = lookup_runtime_desc(views, op.inputs()[0]);
    if (!x)
        return make_error(x.error());
    auto weight = lookup_runtime_desc(views, op.inputs()[2]);
    if (!weight)
        return make_error(weight.error());

    int rank = x->shape.rank();
    if (rank != 2 && rank != 3)
        return make_error("moe_matmul input rank must be 2 or 3");
    if (weight->shape.rank() != 3)
        return make_error("moe_matmul weight rank must be 3");

    int64_t rows = x->shape.dim(rank - 2);
    int64_t outFeatures = weight->shape.dim(op.transposeRhs() ? 1 : 2);
    if (rank == 3) {
        return desc_with_shape(
            graph,
            output,
            core::Shape({x->shape.dim(0), rows, outFeatures}));
    }
    return desc_with_shape(graph, output, core::Shape({rows, outFeatures}));
}

Result<core::TensorDesc> resolve_output_desc(
        const Graph& graph,
        const Op& op,
        ValueId output,
        const std::unordered_map<ValueId, TensorViewDesc>& views) {
    switch (op.kind()) {
        case OpKind::Input:
            return make_error("input op output descriptor is bound externally");
        case OpKind::DeviceTransfer: {
            auto input = lookup_runtime_desc(views, op.inputs()[0]);
            if (!input)
                return make_error(input.error());
            return desc_with_shape(graph, output, input->shape);
        }
        case OpKind::ElementwiseKernel: {
            auto inputs = op.inputs();
            if (inputs.empty())
                return desc_from_static_type(graph.value(output).type);
            if (inputs.size() == 1) {
                auto desc = lookup_runtime_desc(views, inputs[0]);
                if (!desc)
                    return make_error(desc.error());
                return desc_with_shape(graph, output, desc->shape);
            }
            if (inputs.size() == 2) {
                auto lhs = lookup_runtime_desc(views, inputs[0]);
                if (!lhs)
                    return make_error(lhs.error());
                auto rhs = lookup_runtime_desc(views, inputs[1]);
                if (!rhs)
                    return make_error(rhs.error());
                auto shape = core::broadcast_shape(lhs->shape, rhs->shape);
                if (!shape)
                    return make_error(shape.error());
                return desc_with_shape(graph, output, shape.take());
            }
            return desc_from_static_type(graph.value(output).type);
        }
        case OpKind::LayoutTransform: {
            const auto& layout = static_cast<const ir::kernel_ir::LayoutTransformOp&>(op);
            auto input = lookup_runtime_desc(views, layout.inputs()[0]);
            if (!input)
                return make_error(input.error());
            switch (layout.transform()) {
                case LayoutTransformKind::Reshape: {
                    auto shape = core::infer_reshape_shape(
                        input->shape,
                        core::Shape(layout.dims()));
                    if (!shape)
                        return make_error(shape.error());
                    return desc_with_shape(graph, output, shape.take());
                }
                case LayoutTransformKind::Transpose: {
                    auto dims = input->shape.dims();
                    if (dims.size() < 2)
                        return make_error("transpose input rank must be >= 2");
                    std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
                    return desc_with_shape(graph, output, core::Shape(std::move(dims)));
                }
                case LayoutTransformKind::Permute: {
                    std::vector<int64_t> dims;
                    dims.reserve(layout.dims().size());
                    for (int64_t axis : layout.dims())
                        dims.push_back(input->shape.dim(static_cast<int>(axis)));
                    return desc_with_shape(graph, output, core::Shape(std::move(dims)));
                }
                case LayoutTransformKind::Contiguous:
                    return desc_with_shape(graph, output, input->shape);
            }
            return make_error("unsupported layout transform");
        }
        case OpKind::MatMulKernel:
            return resolve_matmul_desc(
                graph,
                static_cast<const ir::kernel_ir::MatMulKernelOp&>(op),
                output,
                views);
        case OpKind::GatherKernel: {
            auto ids = lookup_runtime_desc(views, op.inputs()[0]);
            if (!ids)
                return make_error(ids.error());
            auto table = lookup_runtime_desc(views, op.inputs()[1]);
            if (!table)
                return make_error(table.error());
            if (table->shape.rank() != 1 && table->shape.rank() != 2)
                return make_error("gather table must have rank 1 or rank 2");
            auto dims = ids->shape.dims();
            if (table->shape.rank() == 2)
                dims.push_back(table->shape.dim(1));
            return desc_with_shape(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::SoftmaxKernel:
        case OpKind::NormKernel:
        case OpKind::RoPEKernel: {
            auto input = lookup_runtime_desc(views, op.inputs()[0]);
            if (!input)
                return make_error(input.error());
            return desc_with_shape(graph, output, input->shape);
        }
        case OpKind::SlidingQueryKeyScoreKernel: {
            auto q = lookup_runtime_desc(views, op.inputs()[0]);
            if (!q)
                return make_error(q.error());
            auto k = lookup_runtime_desc(views, op.inputs()[1]);
            if (!k)
                return make_error(k.error());
            int rank = q->shape.rank();
            std::vector<int64_t> outDims;
            if (rank == 4)
                outDims.push_back(q->shape.dim(0));
            outDims.push_back(q->shape.dim(rank - 3));
            outDims.push_back(q->shape.dim(rank - 2));
            outDims.push_back(k->shape.dim(rank - 2));
            return desc_with_shape(graph, output, core::Shape(std::move(outDims)));
        }
        case OpKind::AttentionKernel: {
            auto q = lookup_runtime_desc(views, op.inputs()[0]);
            if (!q)
                return make_error(q.error());
            auto v = lookup_runtime_desc(views, op.inputs()[2]);
            if (!v)
                return make_error(v.error());
            auto dims = q->shape.dims();
            if (dims.empty())
                return make_error("attention query rank must be >= 1");
            dims.back() = v->shape.dim(v->shape.rank() - 1);
            return desc_with_shape(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::LinearKernel: {
            auto x = lookup_runtime_desc(views, op.inputs()[0]);
            if (!x)
                return make_error(x.error());
            auto weight = lookup_runtime_desc(views, op.inputs()[1]);
            if (!weight)
                return make_error(weight.error());
            auto dims = x->shape.dims();
            dims.back() = weight->shape.dim(0);
            return desc_with_shape(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::TopKKernel:
            return resolve_topk_desc(
                graph,
                static_cast<const ir::kernel_ir::TopKKernelOp&>(op),
                output,
                views);
        case OpKind::MoeGatherKernel:
            return resolve_moe_gather_desc(
                graph,
                static_cast<const ir::kernel_ir::MoeGatherKernelOp&>(op),
                output,
                views);
        case OpKind::MoeMatMulKernel:
            return resolve_moe_matmul_desc(
                graph,
                static_cast<const ir::kernel_ir::MoeMatMulKernelOp&>(op),
                output,
                views);
        case OpKind::MoeScatterSumKernel:
        {
            auto reference = lookup_runtime_desc(views, op.inputs()[3]);
            if (!reference)
                return make_error(reference.error());
            return desc_with_shape(graph, output, reference->shape);
        }
        case OpKind::ReductionKernel:
            return resolve_reduction_desc(
                graph,
                static_cast<const ir::kernel_ir::ReductionKernelOp&>(op),
                output,
                views);
    }
    return make_error("unknown KernelIR op kind");
}

Result<void> dealloc_value(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state,
        ValueId value) {
    auto pagedIt = state.pagedTensors.find(value);
    if (pagedIt != state.pagedTensors.end()) {
        state.pagedTensors.erase(pagedIt);
        state.views.erase(value);
        state.bufferDevices.erase(value);
        return {};
    }

    auto buffer = lookup_runtime_buffer(state.buffers, value);
    if (!buffer)
        return make_error(buffer.error());
    auto deviceIt = state.bufferDevices.find(value);
    if (deviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(value));
    auto device = lookup_device(devices, deviceIt->second);
    if (!device)
        return make_error(device.error());

    auto bufferId = buffer.take();
    bool shared = false;
    for (const auto& item : state.buffers) {
        if (item.first == value || item.second != bufferId)
            continue;
        auto otherDevice = state.bufferDevices.find(item.first);
        if (otherDevice != state.bufferDevices.end() &&
            otherDevice->second == deviceIt->second) {
            shared = true;
            break;
        }
    }

    if (!shared && state.borrowedBuffers.find(value) == state.borrowedBuffers.end()) {
        auto dealloc = (*device)->dealloc(bufferId);
        if (!dealloc)
            return make_error(dealloc.error());
    }
    state.buffers.erase(value);
    state.views.erase(value);
    state.bufferDevices.erase(value);
    state.borrowedBuffers.erase(value);
    return {};
}

Result<void> finish_op_lifetimes(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state,
        const Op& finishedOp) {
    for (auto input : finishedOp.inputs()) {
        if (state.remainingUses[input] == 0)
            return make_error("KernelIR value use count underflow");
        state.remainingUses[input]--;
        if (state.remainingUses[input] == 0 && !state.isOutput[input]) {
            auto dealloc = dealloc_value(devices, state, input);
            if (!dealloc)
                return make_error(dealloc.error());
        }
    }

    for (auto output : finishedOp.outputs()) {
        if (state.remainingUses[output] == 0 && !state.isOutput[output]) {
            auto dealloc = dealloc_value(devices, state, output);
            if (!dealloc)
                return make_error(dealloc.error());
        }
    }

    return {};
}

RuntimeState initialize_runtime_state(const Graph& graph) {
    RuntimeState state;
    state.remainingUses.resize(graph.values().size(), 0);
    state.isOutput.resize(graph.values().size(), false);
    for (const auto& value : graph.values())
        state.remainingUses[value.id] = value.uses.size();
    for (auto output : graph.outputs()) {
        if (output < state.isOutput.size()) {
            state.isOutput[output] = true;
            const auto& value = graph.value(output);
            if (value.type.kind == ir::kernel_ir::ValueKind::TensorTuple &&
                value.def.op != ir::kernel_ir::kInvalidOpId) {
                const auto& def = graph.op(value.def.op);
                for (auto input : def.inputs()) {
                    if (input < state.isOutput.size())
                        state.isOutput[input] = true;
                }
            }
        }
    }
    return state;
}

Result<TensorBufferPtr> resolve_input_buffer(
        const InputOp& inputOp,
        std::span<const RunInput> inputs,
        const TensorMap& weights) {
    switch (inputOp.source().kind) {
        case InputSourceKind::Argument: {
            auto index = inputOp.source().index;
            if (index < 0 || static_cast<size_t>(index) >= inputs.size())
                return make_error("input index out of range: " + std::to_string(index));
            const auto& input = inputs[static_cast<size_t>(index)];
            const TensorBufferPtr* host = nullptr;
            if (inputOp.source().tupleElement >= 0) {
                auto* tuple = std::get_if<RunTensorTuple>(&input);
                if (!tuple)
                    return make_error("input arg " + std::to_string(index) +
                                      " must be a tensor tuple");
                auto element = inputOp.source().tupleElement;
                if (static_cast<size_t>(element) >= tuple->elements.size())
                    return make_error("input tuple element out of range");
                host = std::get_if<TensorBufferPtr>(
                    &tuple->elements[static_cast<size_t>(element)]);
                if (!host)
                    return make_error("paged tuple input elements are not supported by Engine::runValues yet");
            } else {
                host = std::get_if<TensorBufferPtr>(&input);
                if (!host)
                    return make_error("input arg " + std::to_string(index) +
                                      " must be a tensor");
            }
            if (!host || !*host)
                return make_error("null input buffer at index: " + std::to_string(index));
            return *host;
        }
        case InputSourceKind::Weight: {
            auto it = weights.find(inputOp.source().name);
            if (it == weights.end())
                return make_error("missing weight buffer: " + inputOp.source().name);
            if (!it->second)
                return make_error("null weight buffer: " + inputOp.source().name);
            return it->second;
        }
        case InputSourceKind::External:
            return make_error("external KernelIR inputs are not supported by Engine::run");
    }
    return make_error("unknown KernelIR input source");
}

Result<void> bind_input_op(
        Device& defaultDevice,
        DeviceId defaultDeviceId,
        const Graph& graph,
        const InputOp& inputOp,
        std::span<const RunInput> inputs,
        const TensorMap& weights,
        RuntimeState& state) {
    auto output = inputOp.outputs()[0];
    const auto& outputType = graph.value(output).type;
    if (outputType.kind == ir::kernel_ir::ValueKind::PagedTensor) {
        if (inputOp.source().kind != InputSourceKind::Argument)
            return make_error("paged tensor inputs must come from arguments");
        auto index = inputOp.source().index;
        if (index < 0 || static_cast<size_t>(index) >= inputs.size())
            return make_error("input index out of range: " + std::to_string(index));

        const DevicePagedTensorView* paged = nullptr;
        const auto& input = inputs[static_cast<size_t>(index)];
        if (inputOp.source().tupleElement >= 0) {
            auto* tuple = std::get_if<RunTensorTuple>(&input);
            if (!tuple)
                return make_error("input arg " + std::to_string(index) +
                                  " must be a tensor tuple");
            auto element = inputOp.source().tupleElement;
            if (static_cast<size_t>(element) >= tuple->elements.size())
                return make_error("input tuple element out of range");
            paged = std::get_if<DevicePagedTensorView>(
                &tuple->elements[static_cast<size_t>(element)]);
        } else {
            paged = std::get_if<DevicePagedTensorView>(&input);
        }
        if (!paged)
            return make_error("input arg " + std::to_string(index) +
                              " must be a paged tensor");

        auto verify = verify_desc_matches_type(
            paged->meta.logicalDesc,
            outputType,
            "value %" + std::to_string(output));
        if (!verify)
            return make_error(verify.error());
        if (paged->meta.growDim != outputType.paged.growDim ||
            paged->meta.pageSize != outputType.paged.pageSize) {
            return make_error("paged tensor input metadata mismatch for value %" +
                              std::to_string(output));
        }

        TensorViewDesc view;
        view.desc = paged->meta.logicalDesc;
        state.pagedTensors[output] = *paged;
        state.views[output] = std::move(view);
        state.bufferDevices[output] = defaultDeviceId;
        return {};
    }

    auto host = resolve_input_buffer(inputOp, inputs, weights);
    if (!host)
        return make_error(host.error());

    auto verify = verify_desc_matches_type(
        (*host)->desc(),
        graph.value(output).type,
        "value %" + std::to_string(output));
    if (!verify)
        return make_error(verify.error());

    auto loaded = defaultDevice.load(**host);
    if (!loaded)
        return make_error(loaded.error());
    auto view = defaultDevice.defaultView((*host)->desc());
    if (!view)
        return make_error(view.error());

    state.buffers[output] = loaded.take();
    state.views[output] = view.take();
    state.bufferDevices[output] = defaultDeviceId;
    return {};
}

Result<void> bind_input_op(
        Device& defaultDevice,
        DeviceId defaultDeviceId,
        const Graph& graph,
        const InputOp& inputOp,
        std::span<const RunInput> inputs,
        const DeviceWeightMap& weights,
        RuntimeState& state) {
    if (inputOp.source().kind != InputSourceKind::Weight) {
        TensorMap emptyWeights;
        return bind_input_op(
            defaultDevice,
            defaultDeviceId,
            graph,
            inputOp,
            inputs,
            emptyWeights,
            state);
    }

    auto output = inputOp.outputs()[0];
    auto deviceIt = weights.weightsByDevice.find(defaultDeviceId);
    if (deviceIt == weights.weightsByDevice.end())
        return make_error("missing device weights for device: " + std::to_string(defaultDeviceId));
    auto weightIt = deviceIt->second.tensors.find(inputOp.source().name);
    if (weightIt == deviceIt->second.tensors.end())
        return make_error("missing device weight buffer: " + inputOp.source().name);

    auto verify = verify_desc_matches_type(
        weightIt->second.view.desc,
        graph.value(output).type,
        "value %" + std::to_string(output));
    if (!verify)
        return make_error(verify.error());

    state.buffers[output] = weightIt->second.buffer;
    state.views[output] = weightIt->second.view;
    state.bufferDevices[output] = defaultDeviceId;
    state.borrowedBuffers.insert(output);
    return {};
}

Result<void> transfer_op(
        std::vector<std::unique_ptr<Device>>& devices,
        DeviceWiseCopier& copier,
        RuntimeState& state,
        const DeviceTransferOp& transfer) {
    auto sourceDevice = lookup_device(devices, transfer.sourceDevice());
    if (!sourceDevice)
        return make_error(sourceDevice.error());
    auto targetDevice = lookup_device(devices, transfer.targetDevice());
    if (!targetDevice)
        return make_error(targetDevice.error());

    auto input = transfer.inputs()[0];
    auto output = transfer.outputs()[0];
    auto inputBuffer = lookup_runtime_buffer(state.buffers, input);
    if (!inputBuffer)
        return make_error(inputBuffer.error());
    auto inputDeviceIt = state.bufferDevices.find(input);
    if (inputDeviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(input));
    if (inputDeviceIt->second != transfer.sourceDevice())
        return make_error("device transfer source does not match runtime value device");

    auto inputView = lookup_runtime_view(state.views, input);
    if (!inputView)
        return make_error(inputView.error());
    auto loaded = copier.copy(
        **sourceDevice,
        DeviceTensorView{inputBuffer.take(), inputView.take()},
        **targetDevice);
    if (!loaded)
        return make_error(loaded.error());

    auto loadedView = loaded.take();
    state.buffers[output] = loadedView.buffer;
    state.views[output] = std::move(loadedView.view);
    state.bufferDevices[output] = transfer.targetDevice();
    return {};
}

Result<void> paged_append_op(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state,
        const PagedAppendOp& append) {
    auto cacheValue = append.cache();
    auto chunkValue = append.chunk();

    auto cacheIt = state.pagedTensors.find(cacheValue);
    if (cacheIt == state.pagedTensors.end())
        return make_error("missing runtime paged tensor for value: " + std::to_string(cacheValue));

    auto cacheDeviceIt = state.bufferDevices.find(cacheValue);
    if (cacheDeviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(cacheValue));
    auto chunkDeviceIt = state.bufferDevices.find(chunkValue);
    if (chunkDeviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(chunkValue));
    if (cacheDeviceIt->second != chunkDeviceIt->second)
        return make_error("paged append cache and chunk must be on the same device");
    if (cacheDeviceIt->second != append.device())
        return make_error("paged append input is not on execution device");

    auto device = lookup_device(devices, cacheDeviceIt->second);
    if (!device)
        return make_error(device.error());

    auto chunkBuffer = lookup_runtime_buffer(state.buffers, chunkValue);
    if (!chunkBuffer)
        return make_error(chunkBuffer.error());
    auto chunkView = lookup_runtime_view(state.views, chunkValue);
    if (!chunkView)
        return make_error(chunkView.error());

    auto denseChunk = (*device)->read(DeviceTensorView{chunkBuffer.take(), chunkView.take()});
    if (!denseChunk)
        return make_error(denseChunk.error());
    auto appended = (*device)->appendPaged(cacheIt->second.tensor, **denseChunk);
    if (!appended)
        return make_error(appended.error());
    auto meta = (*device)->pagedMeta(cacheIt->second.tensor);
    if (!meta)
        return make_error(meta.error());

    cacheIt->second.meta = meta.take();
    auto viewIt = state.views.find(cacheValue);
    if (viewIt == state.views.end())
        return make_error("missing runtime view for value: " + std::to_string(cacheValue));
    viewIt->second.desc = cacheIt->second.meta.logicalDesc;
    return {};
}

Result<void> verify_static_view_shape(
        const TensorViewDesc& view,
        const std::string& context) {
    if (view.desc.shape.has_dynamic())
        return make_error(context + " cannot use strides with dynamic shape");
    return {};
}

Result<bool> try_alias_layout_op(
        const Graph& graph,
        Device& device,
        DeviceId opDevice,
        const ir::kernel_ir::LayoutTransformOp& layout,
        RuntimeState& state) {
    auto input = layout.inputs()[0];
    auto output = layout.outputs()[0];
    auto inputBuffer = lookup_runtime_buffer(state.buffers, input);
    if (!inputBuffer)
        return make_error(inputBuffer.error());
    auto inputView = lookup_runtime_view(state.views, input);
    if (!inputView)
        return make_error(inputView.error());
    auto inputDeviceIt = state.bufferDevices.find(input);
    if (inputDeviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(input));
    if (inputDeviceIt->second != opDevice)
        return make_error("layout transform input is not on execution device");

    auto staticInput = verify_static_view_shape(*inputView, "layout transform input");
    if (!staticInput)
        return make_error(staticInput.error());

    if (layout.transform() == LayoutTransformKind::Reshape) {
        auto isDefault = device.isDefaultView(*inputView);
        if (!isDefault)
            return make_error(isDefault.error());
        if (!*isDefault)
            return false;

        auto desc = resolve_output_desc(graph, layout, output, state.views);
        if (!desc)
            return make_error(desc.error());
        auto view = device.defaultView(desc.take());
        if (!view)
            return make_error(view.error());

        state.buffers[output] = inputBuffer.take();
        state.views[output] = view.take();
        state.bufferDevices[output] = opDevice;
        return true;
    }

    if (layout.transform() == LayoutTransformKind::Transpose ||
        layout.transform() == LayoutTransformKind::Permute) {
        auto desc = resolve_output_desc(graph, layout, output, state.views);
        if (!desc)
            return make_error(desc.error());
        auto descValue = desc.take();
        if (descValue.shape.has_dynamic())
            return make_error("layout transform output cannot use strides with dynamic shape");

        TensorViewDesc outputView;
        outputView.desc = std::move(descValue);
        outputView.storageOffset = inputView->storageOffset;

        if (layout.transform() == LayoutTransformKind::Transpose) {
            outputView.strides = inputView->strides;
            if (outputView.strides.size() < 2)
                return make_error("transpose view input rank must be >= 2");
            std::swap(
                outputView.strides[outputView.strides.size() - 1],
                outputView.strides[outputView.strides.size() - 2]);
        } else {
            outputView.strides.reserve(layout.dims().size());
            for (auto axis : layout.dims()) {
                if (axis < 0 || static_cast<size_t>(axis) >= inputView->strides.size())
                    return make_error("permute view axis out of range");
                outputView.strides.push_back(inputView->strides[static_cast<size_t>(axis)]);
            }
        }

        state.buffers[output] = inputBuffer.take();
        state.views[output] = std::move(outputView);
        state.bufferDevices[output] = opDevice;
        return true;
    }

    if (layout.transform() == LayoutTransformKind::Contiguous) {
        auto isDefault = device.isDefaultView(*inputView);
        if (!isDefault)
            return make_error(isDefault.error());
        if (!*isDefault)
            return false;

        auto desc = resolve_output_desc(graph, layout, output, state.views);
        if (!desc)
            return make_error(desc.error());
        auto view = device.defaultView(desc.take());
        if (!view)
            return make_error(view.error());

        state.buffers[output] = inputBuffer.take();
        state.views[output] = view.take();
        state.bufferDevices[output] = opDevice;
        return true;
    }

    return false;
}

Result<void> allocate_kernel_outputs(
        const Graph& graph,
        Device& device,
        DeviceId opDevice,
        const Op& op,
        RuntimeState& state) {
    for (auto output : op.outputs()) {
        auto desc = resolve_output_desc(graph, op, output, state.views);
        if (!desc)
            return make_error(desc.error());
        auto descValue = desc.take();
        auto buffer = device.alloc(descValue);
        if (!buffer)
            return make_error(buffer.error());
        auto view = device.defaultView(std::move(descValue));
        if (!view)
            return make_error(view.error());
        state.buffers[output] = buffer.take();
        state.views[output] = view.take();
        state.bufferDevices[output] = opDevice;
    }
    return {};
}

Result<std::vector<device::DeviceRunValue>> collect_input_views(
        const RuntimeState& state,
        const Op& op,
        DeviceId opDevice) {
    std::vector<device::DeviceRunValue> inputs;
    inputs.reserve(op.inputs().size());
    for (auto input : op.inputs()) {
        auto inputDeviceIt = state.bufferDevices.find(input);
        if (inputDeviceIt == state.bufferDevices.end())
            return make_error("missing runtime device for value: " + std::to_string(input));
        if (inputDeviceIt->second != opDevice)
            return make_error("KernelIR op input is not on execution device");

        auto pagedIt = state.pagedTensors.find(input);
        if (pagedIt != state.pagedTensors.end()) {
            inputs.push_back(pagedIt->second);
            continue;
        }

        auto buffer = lookup_runtime_buffer(state.buffers, input);
        if (!buffer)
            return make_error(buffer.error());
        auto view = lookup_runtime_view(state.views, input);
        if (!view)
            return make_error(view.error());
        inputs.push_back(DeviceTensorView{buffer.take(), view.take()});
    }
    return inputs;
}

Result<std::vector<device::DeviceRunValue>> collect_output_views(
        const RuntimeState& state,
        const Op& op) {
    std::vector<device::DeviceRunValue> outputs;
    outputs.reserve(op.outputs().size());
    for (auto output : op.outputs()) {
        auto buffer = lookup_runtime_buffer(state.buffers, output);
        if (!buffer)
            return make_error(buffer.error());
        auto view = lookup_runtime_view(state.views, output);
        if (!view)
            return make_error(view.error());
        outputs.push_back(DeviceTensorView{buffer.take(), view.take()});
    }
    return outputs;
}

Result<void> run_kernel_op(
        const Graph& graph,
        Device& device,
        DeviceCompiledGraphId deviceGraph,
        DeviceId opDevice,
        const Op& op,
        size_t opIndex,
        RuntimeState& state,
        const EngineRunOptions* options) {
    auto allocateStart = Clock::now();
    auto allocated = allocate_kernel_outputs(graph, device, opDevice, op, state);
    auto allocateEnd = Clock::now();
    profile_stage(
        options,
        "kernel.allocate_outputs",
        allocateStart,
        allocateEnd,
        opIndex,
        op.id(),
        op.kind());
    if (!allocated)
        return make_error(allocated.error());

    auto collectInputStart = Clock::now();
    auto inputViews = collect_input_views(state, op, opDevice);
    auto collectInputEnd = Clock::now();
    profile_stage(
        options,
        "kernel.collect_inputs",
        collectInputStart,
        collectInputEnd,
        opIndex,
        op.id(),
        op.kind());
    if (!inputViews)
        return make_error(inputViews.error());
    auto collectOutputStart = Clock::now();
    auto outputViews = collect_output_views(state, op);
    auto collectOutputEnd = Clock::now();
    profile_stage(
        options,
        "kernel.collect_outputs",
        collectOutputStart,
        collectOutputEnd,
        opIndex,
        op.id(),
        op.kind());
    if (!outputViews)
        return make_error(outputViews.error());

    profile_device_run_boundary(
        options,
        EngineDeviceRunBoundaryEvent::Boundary::Begin,
        opIndex,
        op.id(),
        opDevice,
        deviceGraph,
        op.kind(),
        inputViews->size(),
        outputViews->size());
    auto start = Clock::now();
    auto runResult = device.run(deviceGraph, op.id(), *inputViews, *outputViews);
    auto end = Clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    profile_device_run_boundary(
        options,
        EngineDeviceRunBoundaryEvent::Boundary::End,
        opIndex,
        op.id(),
        opDevice,
        deviceGraph,
        op.kind(),
        inputViews->size(),
        outputViews->size(),
        elapsed);
    profile_stage(
        options,
        "kernel.device_run",
        start,
        end,
        opIndex,
        op.id(),
        op.kind());
    if (options && options->profileKernel) {
        options->profileKernel({
            opIndex,
            op.id(),
            opDevice,
            deviceGraph,
            op.kind(),
            inputViews->size(),
            outputViews->size(),
            elapsed,
        });
    }
    if (!runResult)
        return make_error(runResult.error());
    return {};
}

Result<RunOutput> read_output_value(
        std::vector<std::unique_ptr<Device>>& devices,
        const Graph& graph,
        RuntimeState& state,
        ValueId value,
        std::vector<ValueId>& uniqueOutputs) {
    if (graph.value(value).type.kind == ir::kernel_ir::ValueKind::TensorTuple) {
        auto it = state.tensorTuples.find(value);
        if (it == state.tensorTuples.end())
            return make_error("missing runtime tensor tuple for value: " + std::to_string(value));
        RunTensorTuple tuple;
        tuple.elements.reserve(it->second.size());
        for (auto element : it->second) {
            auto read = read_output_value(devices, graph, state, element, uniqueOutputs);
            if (!read)
                return make_error(read.error());
            auto* tensor = std::get_if<TensorBufferPtr>(&*read);
            if (!tensor)
                return make_error("nested or paged tuple outputs are not fully supported yet");
            tuple.elements.push_back(*tensor);
        }
        return RunOutput{std::move(tuple)};
    }

    auto buffer = lookup_runtime_buffer(state.buffers, value);
    if (!buffer)
        return make_error(buffer.error());
    auto view = lookup_runtime_view(state.views, value);
    if (!view)
        return make_error(view.error());
    auto outputDeviceIt = state.bufferDevices.find(value);
    if (outputDeviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(value));
    auto outputDevice = lookup_device(devices, outputDeviceIt->second);
    if (!outputDevice)
        return make_error(outputDevice.error());
    auto output = (*outputDevice)->read(DeviceTensorView{buffer.take(), view.take()});
    if (!output)
        return make_error(output.error());

    bool seen = false;
    for (auto existing : uniqueOutputs)
        seen = seen || existing == value;
    if (!seen)
        uniqueOutputs.push_back(value);

    return RunOutput{output.take()};
}

Result<std::vector<RunOutput>> read_graph_outputs(
        std::vector<std::unique_ptr<Device>>& devices,
        const Graph& graph,
        RuntimeState& state,
        std::vector<ValueId>& uniqueOutputs) {
    std::vector<RunOutput> outputs;
    outputs.reserve(graph.outputs().size());
    for (auto value : graph.outputs()) {
        auto output = read_output_value(devices, graph, state, value, uniqueOutputs);
        if (!output)
            return make_error(output.error());
        outputs.push_back(output.take());
        if (graph.value(value).type.kind == ir::kernel_ir::ValueKind::TensorTuple)
            continue;
        bool seen = false;
        for (auto existing : uniqueOutputs)
            seen = seen || existing == value;
        if (!seen)
            uniqueOutputs.push_back(value);
    }
    return outputs;
}

Result<void> dealloc_values(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state,
        const std::vector<ValueId>& values) {
    for (auto value : values) {
        auto dealloc = dealloc_value(devices, state, value);
        if (!dealloc)
            return make_error(dealloc.error());
    }
    return {};
}

Result<void> dealloc_remaining_buffers(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state) {
    for (auto it = state.buffers.begin(); it != state.buffers.end();) {
        auto value = it->first;
        auto deviceIt = state.bufferDevices.find(value);
        if (deviceIt == state.bufferDevices.end())
            return make_error("missing runtime device for value: " + std::to_string(value));
        auto leftoverDevice = lookup_device(devices, deviceIt->second);
        if (!leftoverDevice)
            return make_error(leftoverDevice.error());
        if (state.borrowedBuffers.find(value) == state.borrowedBuffers.end()) {
            auto dealloc = (*leftoverDevice)->dealloc(it->second);
            if (!dealloc)
                return make_error(dealloc.error());
        }
        it = state.buffers.erase(it);
        state.views.erase(value);
        state.bufferDevices.erase(value);
        state.borrowedBuffers.erase(value);
    }
    return {};
}

} // namespace

Engine::Engine(
        std::vector<std::unique_ptr<Device>> devices,
        std::unique_ptr<DeviceWiseCopier> copier)
    : devices_(std::move(devices)),
      copier_(std::move(copier)) {
    if (!copier_)
        copier_ = std::make_unique<HostBounceDeviceWiseCopier>();
}

Result<std::unique_ptr<CompiledKernelGraph>> Engine::compile(
        const ir::mid_ir::Graph& graph,
        const EngineCompileOptions* options) {
    if (devices_.empty())
        return make_error("engine has no devices");

    ir::kernel_ir::LoweringOptions loweringOptions;
    if (options)
        loweringOptions.fusor = options->fusor;

    auto lowered = ir::kernel_ir::lowerMidIRToKernelIR(graph, loweringOptions);
    if (!lowered)
        return make_error(lowered.error());

    auto compiled = std::make_unique<CompiledKernelGraph>();
    compiled->graph = lowered.take();
    compiled->defaultDevice = 0;
    compiled->device = compiled->defaultDevice;

    auto verify = compiled->graph->verify();
    if (!verify)
        return make_error(verify.error());

    std::unordered_set<DeviceId> executableDevices;
    for (const auto& opPtr : compiled->graph->ops()) {
        const auto& op = *opPtr;
        if (op.kind() == OpKind::Input ||
            op.kind() == OpKind::TensorTupleCreate ||
            op.kind() == OpKind::PagedAppend ||
            op.kind() == OpKind::DeviceTransfer)
            continue;
        executableDevices.insert(op.device());
    }

    for (auto deviceId : executableDevices) {
        auto device = lookup_device(devices_, deviceId);
        if (!device)
            return make_error(device.error());
        auto deviceGraph = (*device)->compile(*compiled->graph);
        if (!deviceGraph)
            return make_error(deviceGraph.error());
        compiled->deviceGraphs[deviceId] = deviceGraph.take();
    }

    if (auto it = compiled->deviceGraphs.find(compiled->defaultDevice);
        it != compiled->deviceGraphs.end()) {
        compiled->deviceGraph = it->second;
    }
    return compiled;
}

Result<std::unique_ptr<DeviceWeightMap>> Engine::loadWeights(
        const CompiledKernelGraph& compiled,
        const TensorMap& weights) {
    if (devices_.empty())
        return make_error("engine has no devices");
    if (!compiled.graph)
        return make_error("compiled KernelIR graph is null");

    auto graphDefaultDevice = default_runtime_device(compiled);
    auto defaultDeviceResult = lookup_device(devices_, graphDefaultDevice);
    if (!defaultDeviceResult)
        return make_error(defaultDeviceResult.error());
    auto& defaultDevice = **defaultDeviceResult;
    const auto& graph = *compiled.graph;

    auto loadedWeights = std::make_unique<DeviceWeightMap>();
    auto& deviceWeights = loadedWeights->weightsByDevice[graphDefaultDevice];

    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (op.kind() != OpKind::Input)
            continue;
        const auto& input = static_cast<const InputOp&>(op);
        if (input.source().kind != InputSourceKind::Weight)
            continue;
        const auto& name = input.source().name;
        if (deviceWeights.tensors.find(name) != deviceWeights.tensors.end())
            continue;

        auto hostIt = weights.find(name);
        if (hostIt == weights.end())
            return make_error("missing weight buffer: " + name);
        if (!hostIt->second)
            return make_error("null weight buffer: " + name);

        auto output = input.outputs()[0];
        auto verify = verify_desc_matches_type(
            hostIt->second->desc(),
            graph.value(output).type,
            "value %" + std::to_string(output));
        if (!verify)
            return make_error(verify.error());

        auto loaded = defaultDevice.load(*hostIt->second);
        if (!loaded)
            return make_error(loaded.error());
        auto view = defaultDevice.defaultView(hostIt->second->desc());
        if (!view) {
            auto dealloc = defaultDevice.dealloc(loaded.take());
            if (!dealloc)
                return make_error(dealloc.error());
            return make_error(view.error());
        }

        deviceWeights.tensors[name] = DeviceTensorView{
            loaded.take(),
            view.take(),
        };
    }

    return loadedWeights;
}

Result<void> Engine::deallocWeights(DeviceWeightMap& weights) {
    for (auto& [deviceId, deviceWeights] : weights.weightsByDevice) {
        auto device = lookup_device(devices_, deviceId);
        if (!device)
            return make_error(device.error());
        for (auto& [_, tensor] : deviceWeights.tensors) {
            if (tensor.buffer == 0)
                continue;
            auto dealloc = (*device)->dealloc(tensor.buffer);
            if (!dealloc)
                return make_error(dealloc.error());
            tensor.buffer = 0;
        }
        deviceWeights.tensors.clear();
    }
    weights.weightsByDevice.clear();
    return {};
}

Result<std::vector<TensorBufferPtr>> Engine::run(
        const CompiledKernelGraph& compiled,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options) {
    std::vector<RunInput> valueInputs;
    valueInputs.reserve(inputs.size());
    for (const auto& input : inputs)
        valueInputs.push_back(input);

    auto outputs = runValues(compiled, valueInputs, weights, options);
    if (!outputs)
        return make_error(outputs.error());

    std::vector<TensorBufferPtr> tensorOutputs;
    tensorOutputs.reserve(outputs->size());
    for (auto& output : *outputs) {
        auto* tensor = std::get_if<TensorBufferPtr>(&output);
        if (!tensor)
            return make_error("Engine::run cannot return tensor tuple outputs; use runValues");
        tensorOutputs.push_back(*tensor);
    }
    return tensorOutputs;
}

Result<std::vector<TensorBufferPtr>> Engine::run(
        const CompiledKernelGraph& compiled,
        std::span<TensorBufferPtr const> inputs,
        const DeviceWeightMap& weights,
        const EngineRunOptions* options) {
    std::vector<RunInput> valueInputs;
    valueInputs.reserve(inputs.size());
    for (const auto& input : inputs)
        valueInputs.push_back(input);

    auto outputs = runValues(compiled, valueInputs, weights, options);
    if (!outputs)
        return make_error(outputs.error());

    std::vector<TensorBufferPtr> tensorOutputs;
    tensorOutputs.reserve(outputs->size());
    for (auto& output : *outputs) {
        auto* tensor = std::get_if<TensorBufferPtr>(&output);
        if (!tensor)
            return make_error("Engine::run cannot return tensor tuple outputs; use runValues");
        tensorOutputs.push_back(*tensor);
    }
    return tensorOutputs;
}

Result<std::vector<RunOutput>> Engine::runValues(
        const CompiledKernelGraph& compiled,
        std::span<const RunInput> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options) {
    return runValuesImpl(compiled, inputs, &weights, nullptr, options);
}

Result<std::vector<RunOutput>> Engine::runValues(
        const CompiledKernelGraph& compiled,
        std::span<const RunInput> inputs,
        const DeviceWeightMap& weights,
        const EngineRunOptions* options) {
    return runValuesImpl(compiled, inputs, nullptr, &weights, options);
}

Result<std::vector<RunOutput>> Engine::runValuesImpl(
        const CompiledKernelGraph& compiled,
        std::span<const RunInput> inputs,
        const TensorMap* hostWeights,
        const DeviceWeightMap* deviceWeights,
        const EngineRunOptions* options) {
    auto runValuesStart = Clock::now();
    if (devices_.empty())
        return make_error("engine has no devices");
    if (!compiled.graph)
        return make_error("compiled KernelIR graph is null");

    auto graphDefaultDevice = default_runtime_device(compiled);
    auto defaultDeviceResult = lookup_device(devices_, graphDefaultDevice);
    if (!defaultDeviceResult)
        return make_error(defaultDeviceResult.error());
    auto& defaultDevice = **defaultDeviceResult;
    const auto& graph = *compiled.graph;

    auto state = initialize_runtime_state(graph);

    for (size_t opIndex = 0; opIndex < graph.ops().size(); opIndex++) {
        const auto& op = *graph.ops()[opIndex];

        if (op.kind() == OpKind::Input) {
            auto opStart = Clock::now();
            Result<void> bound;
            if (deviceWeights) {
                bound = bind_input_op(
                    defaultDevice,
                    graphDefaultDevice,
                    graph,
                    static_cast<const InputOp&>(op),
                    inputs,
                    *deviceWeights,
                    state);
            } else if (hostWeights) {
                bound = bind_input_op(
                    defaultDevice,
                    graphDefaultDevice,
                    graph,
                    static_cast<const InputOp&>(op),
                    inputs,
                    *hostWeights,
                    state);
            } else {
                bound = make_error("Engine::runValues missing weights");
            }
            auto opEnd = Clock::now();
            profile_stage(
                options,
                "op.input_bind",
                opStart,
                opEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!bound)
                return make_error(bound.error());
            auto finishStart = Clock::now();
            auto finish = finish_op_lifetimes(devices_, state, op);
            auto finishEnd = Clock::now();
            profile_stage(
                options,
                "op.finish_lifetimes",
                finishStart,
                finishEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!finish)
                return make_error(finish.error());
            continue;
        }

        if (op.kind() == OpKind::TensorTupleCreate) {
            auto opStart = Clock::now();
            state.tensorTuples[op.outputs()[0]] =
                std::vector<ValueId>(op.inputs().begin(), op.inputs().end());
            auto opEnd = Clock::now();
            profile_stage(
                options,
                "op.tensor_tuple_create",
                opStart,
                opEnd,
                opIndex,
                op.id(),
                op.kind());
            continue;
        }

        if (op.kind() == OpKind::PagedAppend) {
            auto opStart = Clock::now();
            auto appended = paged_append_op(
                devices_,
                state,
                static_cast<const PagedAppendOp&>(op));
            auto opEnd = Clock::now();
            profile_stage(
                options,
                "op.paged_append",
                opStart,
                opEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!appended)
                return make_error(appended.error());
            auto finishStart = Clock::now();
            auto finish = finish_op_lifetimes(devices_, state, op);
            auto finishEnd = Clock::now();
            profile_stage(
                options,
                "op.finish_lifetimes",
                finishStart,
                finishEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!finish)
                return make_error(finish.error());
            continue;
        }

        if (op.kind() == OpKind::DeviceTransfer) {
            auto opStart = Clock::now();
            auto transferred = transfer_op(
                devices_,
                *copier_,
                state,
                static_cast<const DeviceTransferOp&>(op));
            auto opEnd = Clock::now();
            profile_stage(
                options,
                "op.device_transfer",
                opStart,
                opEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!transferred)
                return make_error(transferred.error());
            auto finishStart = Clock::now();
            auto finish = finish_op_lifetimes(devices_, state, op);
            auto finishEnd = Clock::now();
            profile_stage(
                options,
                "op.finish_lifetimes",
                finishStart,
                finishEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!finish)
                return make_error(finish.error());
            continue;
        }

        auto opDevice = runtime_op_device(compiled, op);
        auto opDeviceResult = lookup_device(devices_, opDevice);
        if (!opDeviceResult)
            return make_error(opDeviceResult.error());
        auto& device = **opDeviceResult;
        auto deviceGraph = lookup_device_graph(compiled, opDevice);
        if (!deviceGraph)
            return make_error(deviceGraph.error());

        if (op.kind() == OpKind::LayoutTransform) {
            auto aliasStart = Clock::now();
            auto handled = try_alias_layout_op(
                graph,
                device,
                opDevice,
                static_cast<const ir::kernel_ir::LayoutTransformOp&>(op),
                state);
            auto aliasEnd = Clock::now();
            profile_stage(
                options,
                "op.layout_alias_check",
                aliasStart,
                aliasEnd,
                opIndex,
                op.id(),
                op.kind());
            if (!handled)
                return make_error(handled.error());
            if (*handled) {
                auto finishStart = Clock::now();
                auto finish = finish_op_lifetimes(devices_, state, op);
                auto finishEnd = Clock::now();
                profile_stage(
                    options,
                    "op.finish_lifetimes",
                    finishStart,
                    finishEnd,
                    opIndex,
                    op.id(),
                    op.kind());
                if (!finish)
                    return make_error(finish.error());
                continue;
            }
        }

        auto runResult = run_kernel_op(
            graph,
            device,
            *deviceGraph,
            opDevice,
            op,
            opIndex,
            state,
            options);
        if (!runResult)
            return make_error(runResult.error());

        auto finishStart = Clock::now();
        auto finish = finish_op_lifetimes(devices_, state, op);
        auto finishEnd = Clock::now();
        profile_stage(
            options,
            "op.finish_lifetimes",
            finishStart,
            finishEnd,
            opIndex,
            op.id(),
            op.kind());
        if (!finish)
            return make_error(finish.error());
    }

    std::vector<ValueId> uniqueOutputs;
    auto readOutputsStart = Clock::now();
    auto outputs = read_graph_outputs(devices_, graph, state, uniqueOutputs);
    auto readOutputsEnd = Clock::now();
    profile_stage(
        options,
        "run_values.read_outputs",
        readOutputsStart,
        readOutputsEnd);
    if (!outputs)
        return make_error(outputs.error());

    auto deallocOutputsStart = Clock::now();
    auto deallocOutputs = dealloc_values(devices_, state, uniqueOutputs);
    auto deallocOutputsEnd = Clock::now();
    profile_stage(
        options,
        "run_values.dealloc_outputs",
        deallocOutputsStart,
        deallocOutputsEnd);
    if (!deallocOutputs)
        return make_error(deallocOutputs.error());

    auto deallocRemainingStart = Clock::now();
    auto deallocRemaining = dealloc_remaining_buffers(devices_, state);
    auto deallocRemainingEnd = Clock::now();
    profile_stage(
        options,
        "run_values.dealloc_remaining",
        deallocRemainingStart,
        deallocRemainingEnd);
    if (!deallocRemaining)
        return make_error(deallocRemaining.error());

    profile_stage(
        options,
        "run_values.total",
        runValuesStart,
        Clock::now());
    return outputs.take();
}

} // namespace sandy::engine
