#include "Engine.h"

#include "DeviceWiseCopier.h"
#include "MidIRToKernelIR.h"
#include "ShapeUtil.h"

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sandy::engine {

namespace {

using ir::kernel_ir::Graph;
using ir::kernel_ir::DeviceTransferOp;
using ir::kernel_ir::DeviceId;
using ir::kernel_ir::InputOp;
using ir::kernel_ir::InputSourceKind;
using ir::kernel_ir::LayoutTransformKind;
using ir::kernel_ir::Op;
using ir::kernel_ir::OpKind;
using ir::kernel_ir::ValueId;
using ir::kernel_ir::ValueType;

struct RuntimeState {
    std::unordered_map<ValueId, DeviceBufferId> buffers;
    std::unordered_map<ValueId, TensorViewDesc> views;
    std::unordered_map<ValueId, uint32_t> bufferDevices;
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
            auto dims = ids->shape.dims();
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
        case OpKind::CustomKernel: {
            const auto& custom = static_cast<const ir::kernel_ir::CustomKernelOp&>(op);
            if (custom.customName() == "linear") {
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
            return desc_from_static_type(graph.value(output).type);
        }
        case OpKind::ReductionKernel:
            return desc_from_static_type(graph.value(output).type);
    }
    return make_error("unknown KernelIR op kind");
}

Result<void> dealloc_value(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state,
        ValueId value) {
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

    if (!shared) {
        auto dealloc = (*device)->dealloc(bufferId);
        if (!dealloc)
            return make_error(dealloc.error());
    }
    state.buffers.erase(value);
    state.views.erase(value);
    state.bufferDevices.erase(value);
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
        if (output < state.isOutput.size())
            state.isOutput[output] = true;
    }
    return state;
}

Result<TensorBufferPtr> resolve_input_buffer(
        const InputOp& inputOp,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights) {
    switch (inputOp.source().kind) {
        case InputSourceKind::Argument: {
            auto index = inputOp.source().index;
            if (index < 0 || static_cast<size_t>(index) >= inputs.size())
                return make_error("input index out of range: " + std::to_string(index));
            auto host = inputs[static_cast<size_t>(index)];
            if (!host)
                return make_error("null input buffer at index: " + std::to_string(index));
            return host;
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
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights,
        RuntimeState& state) {
    auto output = inputOp.outputs()[0];
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
            return make_error("reshape view requires contiguous input");

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

Result<std::vector<DeviceTensorView>> collect_input_views(
        const RuntimeState& state,
        const Op& op,
        DeviceId opDevice) {
    std::vector<DeviceTensorView> inputs;
    inputs.reserve(op.inputs().size());
    for (auto input : op.inputs()) {
        auto buffer = lookup_runtime_buffer(state.buffers, input);
        if (!buffer)
            return make_error(buffer.error());
        auto view = lookup_runtime_view(state.views, input);
        if (!view)
            return make_error(view.error());
        auto inputDeviceIt = state.bufferDevices.find(input);
        if (inputDeviceIt == state.bufferDevices.end())
            return make_error("missing runtime device for value: " + std::to_string(input));
        if (inputDeviceIt->second != opDevice)
            return make_error("KernelIR op input is not on execution device");
        inputs.push_back(DeviceTensorView{buffer.take(), view.take()});
    }
    return inputs;
}

Result<std::vector<DeviceTensorView>> collect_output_views(
        const RuntimeState& state,
        const Op& op) {
    std::vector<DeviceTensorView> outputs;
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
    auto allocated = allocate_kernel_outputs(graph, device, opDevice, op, state);
    if (!allocated)
        return make_error(allocated.error());

    auto inputViews = collect_input_views(state, op, opDevice);
    if (!inputViews)
        return make_error(inputViews.error());
    auto outputViews = collect_output_views(state, op);
    if (!outputViews)
        return make_error(outputViews.error());

    auto start = std::chrono::steady_clock::now();
    auto runResult = device.run(deviceGraph, op.id(), *inputViews, *outputViews);
    auto end = std::chrono::steady_clock::now();
    if (options && options->profileKernel) {
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
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

Result<std::vector<TensorBufferPtr>> read_graph_outputs(
        std::vector<std::unique_ptr<Device>>& devices,
        const Graph& graph,
        RuntimeState& state,
        std::vector<ValueId>& uniqueOutputs) {
    std::vector<TensorBufferPtr> outputs;
    outputs.reserve(graph.outputs().size());
    for (auto value : graph.outputs()) {
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
        outputs.push_back(output.take());

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
        auto dealloc = (*leftoverDevice)->dealloc(it->second);
        if (!dealloc)
            return make_error(dealloc.error());
        it = state.buffers.erase(it);
        state.views.erase(value);
        state.bufferDevices.erase(value);
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

Result<std::unique_ptr<CompiledKernelGraph>> Engine::compile(const ir::mid_ir::Graph& graph) {
    if (devices_.empty())
        return make_error("engine has no devices");

    auto lowered = ir::kernel_ir::lowerMidIRToKernelIR(graph);
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
        if (op.kind() == OpKind::Input || op.kind() == OpKind::DeviceTransfer)
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

Result<std::vector<TensorBufferPtr>> Engine::run(
        const CompiledKernelGraph& compiled,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options) {
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
            auto bound = bind_input_op(
                defaultDevice,
                graphDefaultDevice,
                graph,
                static_cast<const InputOp&>(op),
                inputs,
                weights,
                state);
            if (!bound)
                return make_error(bound.error());
            auto finish = finish_op_lifetimes(devices_, state, op);
            if (!finish)
                return make_error(finish.error());
            continue;
        }

        if (op.kind() == OpKind::DeviceTransfer) {
            auto transferred = transfer_op(
                devices_,
                *copier_,
                state,
                static_cast<const DeviceTransferOp&>(op));
            if (!transferred)
                return make_error(transferred.error());
            auto finish = finish_op_lifetimes(devices_, state, op);
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
            auto handled = try_alias_layout_op(
                graph,
                device,
                opDevice,
                static_cast<const ir::kernel_ir::LayoutTransformOp&>(op),
                state);
            if (!handled)
                return make_error(handled.error());
            if (*handled) {
                auto finish = finish_op_lifetimes(devices_, state, op);
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

        auto finish = finish_op_lifetimes(devices_, state, op);
        if (!finish)
            return make_error(finish.error());
    }

    std::vector<ValueId> uniqueOutputs;
    auto outputs = read_graph_outputs(devices_, graph, state, uniqueOutputs);
    if (!outputs)
        return make_error(outputs.error());

    auto deallocOutputs = dealloc_values(devices_, state, uniqueOutputs);
    if (!deallocOutputs)
        return make_error(deallocOutputs.error());

    auto deallocRemaining = dealloc_remaining_buffers(devices_, state);
    if (!deallocRemaining)
        return make_error(deallocRemaining.error());

    return outputs.take();
}

} // namespace sandy::engine
