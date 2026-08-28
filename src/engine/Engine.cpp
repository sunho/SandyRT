#include "Engine.h"

#include "ExecutionPlan.h"
#include "DeviceWiseCopier.h"
#include "InvocationCacheKey.h"
#include "MidIRToKernelIR.h"
#include "RuntimeScratchPlan.h"
#include "RuntimeTensorDesc.h"

#include <absl/container/flat_hash_map.h>

#include <atomic>

#include <algorithm>
#include <chrono>
#include <limits>
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

std::atomic<CompiledProgramId> nextCompiledProgramId{1};

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
    struct BufferRef {
        size_t valueCount = 0;
        bool owned = false;
    };

    absl::flat_hash_map<ValueId, DeviceBufferId> buffers;
    absl::flat_hash_map<ValueId, DevicePagedTensorView> pagedTensors;
    absl::flat_hash_map<ValueId, TensorViewDesc> views;
    absl::flat_hash_map<ValueId, uint32_t> bufferDevices;
    std::unordered_map<ValueId, std::vector<ValueId>> tensorTuples;
    absl::flat_hash_map<DeviceId, absl::flat_hash_map<DeviceBufferId, BufferRef>> bufferRefs;
    std::vector<std::pair<DeviceId, DeviceBufferId>> retiredOwnedBuffers;
    std::unordered_map<DeviceId, DeviceBufferId> scratchBuffers;
    RuntimeTensorDescs tensorDescs;

    void addBuffer(
            ValueId value,
            DeviceBufferId buffer,
            TensorViewDesc view,
            DeviceId device,
            bool owned) {
        auto existing = buffers.find(value);
        if (existing != buffers.end()) {
            auto oldDevice = bufferDevices.at(value);
            auto oldBuffer = existing->second;
            if (oldDevice == device && oldBuffer == buffer) {
                views[value] = std::move(view);
                bufferRefs[device][buffer].owned |= owned;
                return;
            }

            auto refsByDevice = bufferRefs.find(oldDevice);
            auto oldRef = refsByDevice->second.find(oldBuffer);
            oldRef->second.valueCount--;
            if (oldRef->second.valueCount == 0) {
                if (oldRef->second.owned)
                    retiredOwnedBuffers.emplace_back(oldDevice, oldBuffer);
                refsByDevice->second.erase(oldRef);
                if (refsByDevice->second.empty())
                    bufferRefs.erase(refsByDevice);
            }
        }

        buffers[value] = buffer;
        views[value] = std::move(view);
        bufferDevices[value] = device;
        auto& ref = bufferRefs[device][buffer];
        ref.valueCount++;
        ref.owned = ref.owned || owned;
    }
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
        const absl::flat_hash_map<ValueId, DeviceBufferId>& buffers,
        ValueId value) {
    auto it = buffers.find(value);
    if (it == buffers.end())
        return make_error("missing runtime buffer for value: " + std::to_string(value));
    return it->second;
}

Result<TensorViewDesc> lookup_runtime_view(
        const absl::flat_hash_map<ValueId, TensorViewDesc>& views,
        ValueId value) {
    auto it = views.find(value);
    if (it == views.end())
        return make_error("missing runtime view for value: " + std::to_string(value));
    return it->second;
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

    auto bufferIt = state.buffers.find(value);
    if (bufferIt == state.buffers.end())
        return make_error("missing runtime buffer for value: " + std::to_string(value));
    auto deviceIt = state.bufferDevices.find(value);
    if (deviceIt == state.bufferDevices.end())
        return make_error("missing runtime device for value: " + std::to_string(value));

    auto refsByDevice = state.bufferRefs.find(deviceIt->second);
    if (refsByDevice == state.bufferRefs.end())
        return make_error("missing runtime buffer refs for value: " + std::to_string(value));
    auto refIt = refsByDevice->second.find(bufferIt->second);
    if (refIt == refsByDevice->second.end() || refIt->second.valueCount == 0)
        return make_error("missing runtime buffer ref for value: " + std::to_string(value));

    refIt->second.valueCount--;
    if (refIt->second.valueCount == 0) {
        auto owned = refIt->second.owned;
        auto buffer = bufferIt->second;
        refsByDevice->second.erase(refIt);
        if (refsByDevice->second.empty())
            state.bufferRefs.erase(refsByDevice);
        if (owned) {
            auto device = lookup_device(devices, deviceIt->second);
            if (!device)
                return make_error(device.error());
            auto dealloc = (*device)->dealloc(buffer);
            if (!dealloc)
                return make_error(dealloc.error());
        }
    }
    state.buffers.erase(value);
    state.views.erase(value);
    state.bufferDevices.erase(value);
    return {};
}

RuntimeState initialize_runtime_state(
        RuntimeTensorDescs tensorDescs,
        RuntimeScratchPlan scratch) {
    RuntimeState state;
    state.tensorDescs = std::move(tensorDescs);
    state.scratchBuffers = std::move(scratch.buffers);
    for (auto& [value, tensor] : scratch.views) {
        state.addBuffer(
            value,
            tensor.buffer,
            std::move(tensor.view),
            scratch.devices.at(value),
            false);
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

Result<core::TensorDesc> argument_input_desc(
        const InputOp& inputOp,
        std::span<const RunInput> inputs) {
    auto index = inputOp.source().index;
    if (index < 0 || static_cast<size_t>(index) >= inputs.size())
        return make_error("input index out of range: " + std::to_string(index));

    const auto* value = &inputs[static_cast<size_t>(index)];
    if (inputOp.source().tupleElement >= 0) {
        auto* tuple = std::get_if<RunTensorTuple>(value);
        if (!tuple)
            return make_error("input arg " + std::to_string(index) + " must be a tensor tuple");
        auto element = static_cast<size_t>(inputOp.source().tupleElement);
        if (element >= tuple->elements.size())
            return make_error("input tuple element out of range");
        const auto& item = tuple->elements[element];
        if (auto* tensor = std::get_if<TensorBufferPtr>(&item)) {
            if (!*tensor) return make_error("null tensor tuple element");
            return (*tensor)->desc();
        }
        if (auto* paged = std::get_if<DevicePagedTensorView>(&item))
            return paged->meta.logicalDesc;
        return make_error("unsupported tensor tuple element");
    }

    if (auto* tensor = std::get_if<TensorBufferPtr>(value)) {
        if (!*tensor) return make_error("null input buffer at index: " + std::to_string(index));
        return (*tensor)->desc();
    }
    if (auto* paged = std::get_if<DevicePagedTensorView>(value))
        return paged->meta.logicalDesc;
    return make_error("input arg " + std::to_string(index) + " must be a tensor");
}

Result<RuntimeTensorDescs> collect_invocation_input_descs(
        const Graph& graph,
        std::span<const RunInput> inputs,
        const TensorMap* hostWeights,
        const DeviceWeightMap* deviceWeights,
        DeviceId defaultDevice) {
    RuntimeTensorDescs inputDescs(graph.values().size());
    for (const auto& opPtr : graph.ops()) {
        if (opPtr->kind() != OpKind::Input)
            continue;
        const auto& input = static_cast<const InputOp&>(*opPtr);
        auto output = input.outputs()[0];
        Result<core::TensorDesc> desc = make_error("unresolved input descriptor");
        if (input.source().kind == InputSourceKind::Argument) {
            desc = argument_input_desc(input, inputs);
        } else if (input.source().kind == InputSourceKind::Weight && hostWeights) {
            auto weight = hostWeights->find(input.source().name);
            if (weight == hostWeights->end() || !weight->second)
                return make_error("missing weight buffer: " + input.source().name);
            desc = weight->second->desc();
        } else if (input.source().kind == InputSourceKind::Weight && deviceWeights) {
            auto device = deviceWeights->weightsByDevice.find(defaultDevice);
            if (device == deviceWeights->weightsByDevice.end())
                return make_error("missing device weights for device: " + std::to_string(defaultDevice));
            auto weight = device->second.tensors.find(input.source().name);
            if (weight == device->second.tensors.end())
                return make_error("missing device weight buffer: " + input.source().name);
            desc = weight->second.view.desc;
        } else if (input.source().kind == InputSourceKind::External) {
            return make_error("external KernelIR inputs are not supported by Engine::runValues");
        }
        if (!desc)
            return make_error(desc.error());
        auto set = inputDescs.set(output, desc.take());
        if (!set)
            return make_error(set.error());
    }
    return inputDescs;
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

        auto verify = verifyRuntimeTensorDesc(
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

    auto verify = verifyRuntimeTensorDesc(
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

    state.addBuffer(
        output,
        loaded.take(),
        view.take(),
        defaultDeviceId,
        true);
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

    auto verify = verifyRuntimeTensorDesc(
        weightIt->second.view.desc,
        graph.value(output).type,
        "value %" + std::to_string(output));
    if (!verify)
        return make_error(verify.error());

    state.addBuffer(
        output,
        weightIt->second.buffer,
        weightIt->second.view,
        defaultDeviceId,
        false);
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
    state.addBuffer(
        output,
        loadedView.buffer,
        std::move(loadedView.view),
        transfer.targetDevice(),
        true);
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

    auto appended = (*device)->appendPaged(
        cacheIt->second.tensor,
        DeviceTensorView{chunkBuffer.take(), chunkView.take()});
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
        Device& device,
        DeviceId opDevice,
        const ir::kernel_ir::LayoutTransformOp& layout,
        RuntimeState& state) {
    if (!layout.aliasesInput())
        return false;

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

    if (layout.transform() == LayoutTransformKind::Slice) {
        if (layout.dims().size() != inputView->strides.size() ||
            layout.indices().size() != inputView->strides.size())
            return make_error("slice selector count must match runtime input rank");

        TensorViewDesc outputView;
        outputView.desc = state.tensorDescs.get(output);
        outputView.storageOffset = inputView->storageOffset;
        if (outputView.storageOffset < 0)
            return make_error("slice input storage offset must be non-negative");
        outputView.strides.reserve(outputView.desc.shape.rank());

        const auto& inputShape = inputView->desc.shape;
        for (size_t axis = 0; axis < layout.dims().size(); ++axis) {
            if (layout.dims()[axis] == 0) {
                outputView.strides.push_back(inputView->strides[axis]);
                continue;
            }
            if (layout.dims()[axis] != 1)
                return make_error("invalid slice selector kind");
            int64_t dim = inputShape.dim(static_cast<int>(axis));
            int64_t index = layout.indices()[axis];
            if (index < 0)
                index += dim;
            if (index < 0 || index >= dim)
                return make_error("slice index out of range for axis " + std::to_string(axis));
            int64_t stride = inputView->strides[axis];
            if (stride < 0)
                return make_error("slice input stride must be non-negative");
            if (index != 0 &&
                stride > (std::numeric_limits<int64_t>::max() -
                          outputView.storageOffset) / index)
                return make_error("slice storage offset overflow");
            outputView.storageOffset += index * stride;
        }

        state.addBuffer(
            output,
            inputBuffer.take(),
            std::move(outputView),
            opDevice,
            false);
        return true;
    }

    if (layout.transform() == LayoutTransformKind::Reshape) {
        auto isDefault = device.isDefaultView(*inputView);
        if (!isDefault)
            return make_error(isDefault.error());
        if (!*isDefault)
            return make_error("runtime buffer violates contiguous KernelIR value requirement");

        auto view = device.defaultView(state.tensorDescs.get(output));
        if (!view)
            return make_error(view.error());

        view->storageOffset = inputView->storageOffset;

        state.addBuffer(
            output,
            inputBuffer.take(),
            view.take(),
            opDevice,
            false);
        return true;
    }

    if (layout.transform() == LayoutTransformKind::Transpose ||
        layout.transform() == LayoutTransformKind::Permute) {
        auto descValue = state.tensorDescs.get(output);
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

        state.addBuffer(
            output,
            inputBuffer.take(),
            std::move(outputView),
            opDevice,
            false);
        return true;
    }

    if (layout.transform() == LayoutTransformKind::Contiguous) {
        auto isDefault = device.isDefaultView(*inputView);
        if (!isDefault)
            return make_error(isDefault.error());
        if (!*isDefault)
            return make_error("runtime buffer violates contiguous KernelIR value requirement");

        auto view = device.defaultView(state.tensorDescs.get(output));
        if (!view)
            return make_error(view.error());

        view->storageOffset = inputView->storageOffset;

        state.addBuffer(
            output,
            inputBuffer.take(),
            view.take(),
            opDevice,
            false);
        return true;
    }

    return false;
}

Result<void> allocate_kernel_outputs(
        Device& device,
        DeviceId opDevice,
        const Op& op,
        RuntimeState& state) {
    for (auto output : op.outputs()) {
        if (state.buffers.contains(output))
            continue;
        auto descValue = state.tensorDescs.get(output);
        auto buffer = device.alloc(descValue);
        if (!buffer)
            return make_error(buffer.error());
        auto view = device.defaultView(std::move(descValue));
        if (!view)
            return make_error(view.error());
        state.addBuffer(
            output,
            buffer.take(),
            view.take(),
            opDevice,
            true);
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
        Device& device,
        DeviceCompiledGraphId deviceGraph,
        DeviceId opDevice,
        const Op& op,
        size_t opIndex,
        RuntimeState& state,
        const EngineRunOptions* options) {
    auto allocateStart = Clock::now();
    auto allocated = allocate_kernel_outputs(device, opDevice, op, state);
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

bool is_alias_value(const Graph& graph, ValueId value) {
    const auto& def = graph.value(value).def;
    if (def.op == ir::kernel_ir::kInvalidOpId)
        return false;
    const auto& op = graph.op(def.op);
    return op.kind() == OpKind::LayoutTransform &&
        static_cast<const ir::kernel_ir::LayoutTransformOp&>(op).aliasesInput();
}

Result<device::DeviceRunValue> runtime_value(
        const RuntimeState& state,
        ValueId value) {
    auto paged = state.pagedTensors.find(value);
    if (paged != state.pagedTensors.end())
        return device::DeviceRunValue{paged->second};
    auto buffer = state.buffers.find(value);
    auto view = state.views.find(value);
    if (buffer == state.buffers.end() || view == state.views.end())
        return make_error("missing executable boundary value %" + std::to_string(value));
    return device::DeviceRunValue{DeviceTensorView{buffer->second, view->second}};
}

Result<void> run_execution_node(
        const CompiledKernelGraph& compiled,
        const CompiledExecutionNode& node,
        Device& device,
        RuntimeState& state,
        const EngineRunOptions* options) {
    const auto& graph = *compiled.graph;
    for (auto value : node.bindings.exports) {
        const auto& type = graph.value(value).type;
        if (type.kind != ir::kernel_ir::ValueKind::Tensor ||
            state.buffers.contains(value) || is_alias_value(graph, value))
            continue;
        auto desc = state.tensorDescs.get(value);
        auto buffer = device.alloc(desc);
        if (!buffer)
            return make_error(buffer.error());
        auto view = device.defaultView(std::move(desc));
        if (!view) {
            (void)device.dealloc(*buffer);
            return make_error(view.error());
        }
        state.addBuffer(value, *buffer, view.take(), node.device, true);
    }

    device::DeviceExecutableRunState runState;
    {
        auto bindDescriptor = [&](ValueId value) {
            if (runState.tensorDescs.contains(value) ||
                graph.value(value).type.kind == ir::kernel_ir::ValueKind::TensorTuple ||
                !state.tensorDescs.has(value))
                return;
            runState.tensorDescs.emplace(value, state.tensorDescs.get(value));
        };
        for (auto value : node.bindings.imports)
            bindDescriptor(value);
        for (auto value : node.bindings.exports)
            bindDescriptor(value);
        for (auto opId : node.bindings.ops) {
            const auto& op = graph.op(opId);
            for (auto value : op.inputs())
                bindDescriptor(value);
            for (auto value : op.outputs())
                bindDescriptor(value);
        }
    }
    auto bind = [&](ValueId value) -> Result<void> {
        if (runState.values.contains(value))
            return {};
        auto bound = runtime_value(state, value);
        if (!bound)
            return make_error(bound.error());
        runState.values.emplace(value, bound.take());
        return {};
    };
    for (auto value : node.bindings.imports) {
        auto result = bind(value);
        if (!result)
            return make_error(result.error());
    }
    for (auto value : node.bindings.exports) {
        if (is_alias_value(graph, value))
            continue;
        auto result = bind(value);
        if (!result)
            return make_error(result.error());
    }

    if (options && options->profileKernel) {
        runState.profileKernel = [&](ir::kernel_ir::OpId opId, double elapsedMs) {
            const auto& op = graph.op(opId);
            options->profileKernel({
                static_cast<size_t>(opId),
                opId,
                node.device,
                node.executable->compiledGraph(),
                op.kind(),
                op.inputs().size(),
                op.outputs().size(),
                elapsedMs,
            });
        };
    }

    const auto firstOpId = node.bindings.ops.front();
    const auto& firstOp = graph.op(firstOpId);
    profile_device_run_boundary(
        options,
        EngineDeviceRunBoundaryEvent::Boundary::Begin,
        static_cast<size_t>(firstOpId),
        firstOpId,
        node.device,
        node.executable->compiledGraph(),
        firstOp.kind(),
        node.bindings.imports.size(),
        node.bindings.exports.size());
    auto start = Clock::now();
    auto executed = device.runExecutable(*node.executable, runState);
    auto end = Clock::now();
    if (!executed)
        return make_error(executed.error());
    profile_device_run_boundary(
        options,
        EngineDeviceRunBoundaryEvent::Boundary::End,
        static_cast<size_t>(firstOpId),
        firstOpId,
        node.device,
        node.executable->compiledGraph(),
        firstOp.kind(),
        node.bindings.imports.size(),
        node.bindings.exports.size(),
        std::chrono::duration<double, std::milli>(end - start).count());
    profile_stage(options, "executable.device_run", start, end);

    auto merge = [&](ValueId value) -> Result<void> {
        auto found = runState.values.find(value);
        if (found == runState.values.end())
            return make_error("device executable did not produce boundary value %" +
                              std::to_string(value));
        if (auto* paged = std::get_if<DevicePagedTensorView>(&found->second)) {
            state.pagedTensors[value] = *paged;
            state.views[value].desc = paged->meta.logicalDesc;
            state.bufferDevices[value] = node.device;
            return {};
        }
        auto* tensor = std::get_if<DeviceTensorView>(&found->second);
        if (!tensor)
            return make_error("unsupported executable boundary value type");
        state.addBuffer(value, tensor->buffer, tensor->view, node.device, false);
        return {};
    };
    for (auto value : node.bindings.exports) {
        auto result = merge(value);
        if (!result)
            return make_error(result.error());
    }
    for (auto value : node.bindings.mutableImports) {
        auto result = merge(value);
        if (!result)
            return make_error(result.error());
    }
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
    std::vector<ValueId> values;
    values.reserve(state.buffers.size());
    for (const auto& [value, _] : state.buffers)
        values.push_back(value);
    for (auto value : values) {
        if (!state.buffers.contains(value))
            continue;
        auto deallocated = dealloc_value(devices, state, value);
        if (!deallocated)
            return make_error(deallocated.error());
    }
    for (const auto& [deviceId, buffer] : state.retiredOwnedBuffers) {
        auto device = lookup_device(devices, deviceId);
        if (!device)
            return make_error(device.error());
        auto deallocated = (*device)->dealloc(buffer);
        if (!deallocated)
            return make_error(deallocated.error());
    }
    state.retiredOwnedBuffers.clear();
    return {};
}

Result<void> dealloc_scratch_buffers(
        std::vector<std::unique_ptr<Device>>& devices,
        RuntimeState& state) {
    for (const auto& [deviceId, buffer] : state.scratchBuffers) {
        auto device = lookup_device(devices, deviceId);
        if (!device)
            return make_error(device.error());
        auto deallocated = (*device)->dealloc(buffer);
        if (!deallocated)
            return make_error(deallocated.error());
    }
    state.scratchBuffers.clear();
    return {};
}

class ScratchBufferGuard {
public:
    ScratchBufferGuard(
            std::vector<std::unique_ptr<Device>>& devices,
            RuntimeState& state)
        : devices_(devices), state_(state) {}

    ~ScratchBufferGuard() {
        (void)dealloc_scratch_buffers(devices_, state_);
    }

    ScratchBufferGuard(const ScratchBufferGuard&) = delete;
    ScratchBufferGuard& operator=(const ScratchBufferGuard&) = delete;

private:
    std::vector<std::unique_ptr<Device>>& devices_;
    RuntimeState& state_;
};

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
    compiled->programId = nextCompiledProgramId.fetch_add(1, std::memory_order_relaxed);
    compiled->graph = lowered.take();
    compiled->defaultDevice = 0;
    compiled->device = compiled->defaultDevice;

    auto verify = compiled->graph->verify();
    if (!verify)
        return make_error(verify.error());

    auto executionPlan = partitionKernelGraph(*compiled->graph);
    if (!executionPlan)
        return make_error(executionPlan.error());
    compiled->executionNodeForOp = std::move(executionPlan->nodeForOp);
    for (auto& node : executionPlan->nodes) {
        auto device = lookup_device(devices_, node.device);
        if (!device)
            return make_error(device.error());
        auto executable = (*device)->compileExecutable(
            *compiled->graph, node.executable);
        if (!executable)
            return make_error(executable.error());
        auto compiledGraphId = (*executable)->compiledGraph();
        compiled->deviceGraphs.try_emplace(node.device, compiledGraphId);
        compiled->executionNodes.push_back(CompiledExecutionNode{
            node.device,
            executable.take(),
            std::move(node.executable),
        });
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
        auto verify = verifyRuntimeTensorDesc(
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

    auto inferDescsStart = Clock::now();
    auto inputDescs = collect_invocation_input_descs(
        graph,
        inputs,
        hostWeights,
        deviceWeights,
        graphDefaultDevice);
    if (!inputDescs)
        return make_error(inputDescs.error());
    auto cacheKey = buildInvocationCacheKey(
        "runtime-invocation-v1",
        compiled.programId,
        graph,
        *inputDescs);
    if (!cacheKey)
        return make_error(cacheKey.error());

    Clock::time_point inferDescsEnd;
    Clock::time_point scratchPlanStart;
    Clock::time_point scratchPlanEnd;
    auto cachedPlan = runtimePlanCache_.getOrCreate(*cacheKey, [&]() -> Result<CachedInvocationPlan> {
        auto tensorDescs = inferRuntimeTensorDescs(graph, std::move(*inputDescs));
        inferDescsEnd = Clock::now();
        if (!tensorDescs)
            return make_error(tensorDescs.error());
        scratchPlanStart = Clock::now();
        Result<RuntimeScratchLayout> scratchLayout = compiled.executionNodes.empty()
            ? planRuntimeScratchLayout(compiled, *tensorDescs, devices_)
            : RuntimeScratchLayout{};
        scratchPlanEnd = Clock::now();
        if (!scratchLayout)
            return make_error(scratchLayout.error());
        return CachedInvocationPlan{tensorDescs.take(), scratchLayout.take()};
    });
    if (inferDescsEnd == Clock::time_point{})
        inferDescsEnd = Clock::now();
    profile_stage(
        options,
        "run_values.infer_tensor_descs",
        inferDescsStart,
        inferDescsEnd);
    if (!cachedPlan)
        return make_error(cachedPlan.error());
    if (scratchPlanStart == Clock::time_point{})
        scratchPlanStart = Clock::now();
    auto scratchPlan = instantiateRuntimeScratch((*cachedPlan)->scratchLayout, devices_);
    scratchPlanEnd = Clock::now();
    profile_stage(
        options,
        "run_values.plan_scratch",
        scratchPlanStart,
        scratchPlanEnd);
    if (!scratchPlan)
        return make_error(scratchPlan.error());

    auto state = initialize_runtime_state((*cachedPlan)->tensorDescs, scratchPlan.take());
    ScratchBufferGuard scratchGuard(devices_, state);

    // Inputs and tensor-tuple declarations only bind runtime state. Bind them
    // before executing device nodes so inputs interleaved in KernelIR do not
    // artificially split otherwise contiguous same-device executables.
    for (size_t opIndex = 0; opIndex < graph.ops().size(); ++opIndex) {
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
        }
    }

    for (size_t opIndex = 0; opIndex < graph.ops().size(); opIndex++) {
        const auto& op = *graph.ops()[opIndex];

        if (op.kind() == OpKind::Input || op.kind() == OpKind::TensorTupleCreate)
            continue;

        if (op.kind() == OpKind::PagedAppend && compiled.executionNodes.empty()) {
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
            continue;
        }

        if (!compiled.executionNodes.empty() &&
            op.id() < compiled.executionNodeForOp.size() &&
            compiled.executionNodeForOp[op.id()] >= 0) {
            auto nodeIndex = static_cast<size_t>(compiled.executionNodeForOp[op.id()]);
            const auto& node = compiled.executionNodes[nodeIndex];
            if (node.bindings.ops.empty() || node.bindings.ops.front() != op.id())
                continue;
            auto nodeDevice = lookup_device(devices_, node.device);
            if (!nodeDevice)
                return make_error(nodeDevice.error());
            auto executed = run_execution_node(
                compiled, node, **nodeDevice, state, options);
            if (!executed)
                return make_error(executed.error());
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
                continue;
            }
        }

        auto runResult = run_kernel_op(
            device,
            *deviceGraph,
            opDevice,
            op,
            opIndex,
            state,
            options);
        if (!runResult)
            return make_error(runResult.error());

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

    auto deallocScratchStart = Clock::now();
    auto deallocScratch = dealloc_scratch_buffers(devices_, state);
    auto deallocScratchEnd = Clock::now();
    profile_stage(
        options,
        "run_values.dealloc_scratch",
        deallocScratchStart,
        deallocScratchEnd);
    if (!deallocScratch)
        return make_error(deallocScratch.error());

    profile_stage(
        options,
        "run_values.total",
        runValuesStart,
        Clock::now());
    return outputs.take();
}

} // namespace sandy::engine
