#include "Engine.h"

#include "MidIRToKernelIR.h"
#include "ShapeUtil.h"

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sandy::engine {

namespace {

using ir::kernel_ir::Graph;
using ir::kernel_ir::DeviceTransferOp;
using ir::kernel_ir::InputOp;
using ir::kernel_ir::InputSourceKind;
using ir::kernel_ir::LayoutTransformKind;
using ir::kernel_ir::Op;
using ir::kernel_ir::OpKind;
using ir::kernel_ir::ValueId;
using ir::kernel_ir::ValueType;

Result<Device*> lookup_device(
        std::vector<std::unique_ptr<Device>>& devices,
        uint32_t device) {
    if (device >= devices.size())
        return make_error("invalid device id: " + std::to_string(device));
    if (!devices[device])
        return make_error("null device: " + std::to_string(device));
    return devices[device].get();
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
        const std::unordered_map<ValueId, core::TensorDesc>& descs,
        ValueId value) {
    auto it = descs.find(value);
    if (it == descs.end())
        return make_error("missing runtime descriptor for value: " + std::to_string(value));
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
        const std::unordered_map<ValueId, core::TensorDesc>& descs) {
    auto lhsDesc = lookup_runtime_desc(descs, op.inputs()[0]);
    if (!lhsDesc)
        return make_error(lhsDesc.error());
    auto rhsDesc = lookup_runtime_desc(descs, op.inputs()[1]);
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
        const std::unordered_map<ValueId, core::TensorDesc>& descs) {
    switch (op.kind()) {
        case OpKind::Input:
            return make_error("input op output descriptor is bound externally");
        case OpKind::DeviceTransfer: {
            auto input = lookup_runtime_desc(descs, op.inputs()[0]);
            if (!input)
                return make_error(input.error());
            return desc_with_shape(graph, output, input->shape);
        }
        case OpKind::ElementwiseKernel: {
            auto inputs = op.inputs();
            if (inputs.empty())
                return desc_from_static_type(graph.value(output).type);
            if (inputs.size() == 1) {
                auto desc = lookup_runtime_desc(descs, inputs[0]);
                if (!desc)
                    return make_error(desc.error());
                return desc_with_shape(graph, output, desc->shape);
            }
            if (inputs.size() == 2) {
                auto lhs = lookup_runtime_desc(descs, inputs[0]);
                if (!lhs)
                    return make_error(lhs.error());
                auto rhs = lookup_runtime_desc(descs, inputs[1]);
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
            auto input = lookup_runtime_desc(descs, layout.inputs()[0]);
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
                descs);
        case OpKind::GatherKernel: {
            auto ids = lookup_runtime_desc(descs, op.inputs()[0]);
            if (!ids)
                return make_error(ids.error());
            auto table = lookup_runtime_desc(descs, op.inputs()[1]);
            if (!table)
                return make_error(table.error());
            auto dims = ids->shape.dims();
            dims.push_back(table->shape.dim(1));
            return desc_with_shape(graph, output, core::Shape(std::move(dims)));
        }
        case OpKind::SoftmaxKernel:
        case OpKind::NormKernel:
        case OpKind::RoPEKernel: {
            auto input = lookup_runtime_desc(descs, op.inputs()[0]);
            if (!input)
                return make_error(input.error());
            return desc_with_shape(graph, output, input->shape);
        }
        case OpKind::SlidingQueryKeyScoreKernel: {
            auto q = lookup_runtime_desc(descs, op.inputs()[0]);
            if (!q)
                return make_error(q.error());
            auto k = lookup_runtime_desc(descs, op.inputs()[1]);
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
                auto x = lookup_runtime_desc(descs, op.inputs()[0]);
                if (!x)
                    return make_error(x.error());
                auto weight = lookup_runtime_desc(descs, op.inputs()[1]);
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
        ValueId value,
        std::unordered_map<ValueId, DeviceBufferId>& buffers,
        std::unordered_map<ValueId, core::TensorDesc>& descs,
        std::unordered_map<ValueId, uint32_t>& bufferDevices) {
    auto buffer = lookup_runtime_buffer(buffers, value);
    if (!buffer)
        return make_error(buffer.error());
    auto deviceIt = bufferDevices.find(value);
    if (deviceIt == bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(value));
    auto device = lookup_device(devices, deviceIt->second);
    if (!device)
        return make_error(device.error());
    auto dealloc = (*device)->dealloc(buffer.take());
    if (!dealloc)
        return make_error(dealloc.error());
    buffers.erase(value);
    descs.erase(value);
    bufferDevices.erase(value);
    return {};
}

} // namespace

Engine::Engine(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)) {}

Result<std::unique_ptr<CompiledKernelGraph>> Engine::compile(const ir::mid_ir::Graph& graph) {
    if (devices_.empty())
        return make_error("engine has no devices");

    auto lowered = ir::kernel_ir::lowerMidIRToKernelIR(graph);
    if (!lowered)
        return make_error(lowered.error());

    auto compiled = std::make_unique<CompiledKernelGraph>();
    compiled->graph = lowered.take();
    compiled->device = 0;

    auto device = lookup_device(devices_, compiled->device);
    if (!device)
        return make_error(device.error());
    auto deviceGraph = (*device)->compile(*compiled->graph);
    if (!deviceGraph)
        return make_error(deviceGraph.error());
    compiled->deviceGraph = deviceGraph.take();
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

    auto deviceResult = lookup_device(devices_, compiled.device);
    if (!deviceResult)
        return make_error(deviceResult.error());
    auto& device = **deviceResult;
    const auto& graph = *compiled.graph;

    std::unordered_map<ValueId, DeviceBufferId> buffers;
    std::unordered_map<ValueId, core::TensorDesc> descs;
    std::unordered_map<ValueId, uint32_t> bufferDevices;
    std::vector<size_t> remainingUses(graph.values().size(), 0);
    std::vector<bool> isOutput(graph.values().size(), false);
    for (const auto& value : graph.values())
        remainingUses[value.id] = value.uses.size();
    for (auto output : graph.outputs()) {
        if (output < isOutput.size())
            isOutput[output] = true;
    }

    auto finish_op_lifetimes = [&](const Op& finishedOp) -> Result<void> {
        for (auto input : finishedOp.inputs()) {
            if (remainingUses[input] == 0)
                return make_error("KernelIR value use count underflow");
            remainingUses[input]--;
            if (remainingUses[input] == 0 && !isOutput[input]) {
                auto dealloc = dealloc_value(
                    devices_,
                    input,
                    buffers,
                    descs,
                    bufferDevices);
                if (!dealloc)
                    return make_error(dealloc.error());
            }
        }

        for (auto output : finishedOp.outputs()) {
            if (remainingUses[output] == 0 && !isOutput[output]) {
                auto dealloc = dealloc_value(
                    devices_,
                    output,
                    buffers,
                    descs,
                    bufferDevices);
                if (!dealloc)
                    return make_error(dealloc.error());
            }
        }

        return {};
    };

    for (size_t opIndex = 0; opIndex < graph.ops().size(); opIndex++) {
        const auto& op = *graph.ops()[opIndex];

        if (op.kind() == OpKind::Input) {
            const auto& inputOp = static_cast<const InputOp&>(op);
            auto output = inputOp.outputs()[0];
            TensorBufferPtr host;

            switch (inputOp.source().kind) {
                case InputSourceKind::Argument: {
                    auto index = inputOp.source().index;
                    if (index < 0 || static_cast<size_t>(index) >= inputs.size())
                        return make_error("input index out of range: " + std::to_string(index));
                    host = inputs[static_cast<size_t>(index)];
                    if (!host)
                        return make_error("null input buffer at index: " + std::to_string(index));
                    break;
                }
                case InputSourceKind::Weight: {
                    auto it = weights.find(inputOp.source().name);
                    if (it == weights.end())
                        return make_error("missing weight buffer: " + inputOp.source().name);
                    host = it->second;
                    if (!host)
                        return make_error("null weight buffer: " + inputOp.source().name);
                    break;
                }
                case InputSourceKind::External:
                    return make_error("external KernelIR inputs are not supported by Engine::run");
            }

            auto verify = verify_desc_matches_type(
                host->desc(),
                graph.value(output).type,
                "value %" + std::to_string(output));
            if (!verify)
                return make_error(verify.error());

            auto loaded = device.load(*host);
            if (!loaded)
                return make_error(loaded.error());
            buffers[output] = loaded.take();
            descs[output] = host->desc();
            bufferDevices[output] = compiled.device;

            auto finish = finish_op_lifetimes(op);
            if (!finish)
                return make_error(finish.error());
            continue;
        }

        if (op.kind() == OpKind::DeviceTransfer) {
            const auto& transfer = static_cast<const DeviceTransferOp&>(op);
            auto sourceDevice = lookup_device(devices_, transfer.sourceDevice());
            if (!sourceDevice)
                return make_error(sourceDevice.error());
            auto targetDevice = lookup_device(devices_, transfer.targetDevice());
            if (!targetDevice)
                return make_error(targetDevice.error());

            auto input = transfer.inputs()[0];
            auto output = transfer.outputs()[0];
            auto inputBuffer = lookup_runtime_buffer(buffers, input);
            if (!inputBuffer)
                return make_error(inputBuffer.error());
            auto inputDeviceIt = bufferDevices.find(input);
            if (inputDeviceIt == bufferDevices.end())
                return make_error("missing runtime device for value: " + std::to_string(input));
            if (inputDeviceIt->second != transfer.sourceDevice()) {
                return make_error("device transfer source does not match runtime value device");
            }

            auto host = (*sourceDevice)->read(inputBuffer.take());
            if (!host)
                return make_error(host.error());
            auto loaded = (*targetDevice)->load(**host);
            if (!loaded)
                return make_error(loaded.error());
            buffers[output] = loaded.take();
            descs[output] = (*host)->desc();
            bufferDevices[output] = transfer.targetDevice();

            auto finish = finish_op_lifetimes(op);
            if (!finish)
                return make_error(finish.error());
            continue;
        }

        for (auto output : op.outputs()) {
            auto desc = resolve_output_desc(graph, op, output, descs);
            if (!desc)
                return make_error(desc.error());
            auto descValue = desc.take();
            auto buffer = device.alloc(descValue);
            if (!buffer)
                return make_error(buffer.error());
            buffers[output] = buffer.take();
            descs[output] = std::move(descValue);
            bufferDevices[output] = compiled.device;
        }

        std::vector<DeviceBufferId> inputBuffers;
        inputBuffers.reserve(op.inputs().size());
        for (auto input : op.inputs()) {
            auto buffer = lookup_runtime_buffer(buffers, input);
            if (!buffer)
                return make_error(buffer.error());
            auto inputDeviceIt = bufferDevices.find(input);
            if (inputDeviceIt == bufferDevices.end())
                return make_error("missing runtime device for value: " + std::to_string(input));
            if (inputDeviceIt->second != compiled.device) {
                return make_error("KernelIR op input is not on execution device");
            }
            inputBuffers.push_back(buffer.take());
        }

        std::vector<DeviceBufferId> outputBuffers;
        outputBuffers.reserve(op.outputs().size());
        for (auto output : op.outputs()) {
            auto buffer = lookup_runtime_buffer(buffers, output);
            if (!buffer)
                return make_error(buffer.error());
            outputBuffers.push_back(buffer.take());
        }

        auto start = std::chrono::steady_clock::now();
        auto runResult = device.run(compiled.deviceGraph, op.id(), inputBuffers, outputBuffers);
        auto end = std::chrono::steady_clock::now();
        if (options && options->profileKernel) {
            auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            options->profileKernel({
                opIndex,
                op.id(),
                compiled.device,
                compiled.deviceGraph,
                op.kind(),
                inputBuffers.size(),
                outputBuffers.size(),
                elapsed,
            });
        }
        if (!runResult)
            return make_error(runResult.error());

        auto finish = finish_op_lifetimes(op);
        if (!finish)
            return make_error(finish.error());
    }

    std::vector<TensorBufferPtr> outputs;
    outputs.reserve(graph.outputs().size());
    std::vector<ValueId> uniqueOutputs;
    for (auto value : graph.outputs()) {
        auto buffer = lookup_runtime_buffer(buffers, value);
        if (!buffer)
            return make_error(buffer.error());
        auto outputDeviceIt = bufferDevices.find(value);
        if (outputDeviceIt == bufferDevices.end())
            return make_error("missing runtime device for value: " + std::to_string(value));
        auto outputDevice = lookup_device(devices_, outputDeviceIt->second);
        if (!outputDevice)
            return make_error(outputDevice.error());
        auto output = (*outputDevice)->read(buffer.take());
        if (!output)
            return make_error(output.error());
        outputs.push_back(output.take());

        bool seen = false;
        for (auto existing : uniqueOutputs)
            seen = seen || existing == value;
        if (!seen)
            uniqueOutputs.push_back(value);
    }

    for (auto value : uniqueOutputs) {
        auto dealloc = dealloc_value(
            devices_,
            value,
            buffers,
            descs,
            bufferDevices);
        if (!dealloc)
            return make_error(dealloc.error());
    }

    for (auto it = buffers.begin(); it != buffers.end();) {
        auto value = it->first;
        auto deviceIt = bufferDevices.find(value);
        if (deviceIt == bufferDevices.end())
            return make_error("missing runtime device for value: " + std::to_string(value));
        auto leftoverDevice = lookup_device(devices_, deviceIt->second);
        if (!leftoverDevice)
            return make_error(leftoverDevice.error());
        auto dealloc = (*leftoverDevice)->dealloc(it->second);
        if (!dealloc)
            return make_error(dealloc.error());
        it = buffers.erase(it);
        descs.erase(value);
        bufferDevices.erase(value);
    }

    return outputs;
}

} // namespace sandy::engine
