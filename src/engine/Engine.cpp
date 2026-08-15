#include "Engine.h"

#include "InvocPlanner.h"

#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sandy::engine {

namespace {

class HostTensorBuffer final : public core::TensorBuffer {
public:
    HostTensorBuffer(core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

Result<TensorBufferPtr> with_desc(TensorBufferPtr buffer, core::TensorDesc desc) {
    auto accessResult = buffer->access();
    if (!accessResult)
        return make_error(accessResult.error());
    auto access = accessResult.take();

    auto data = access.data();
    TensorBufferPtr result = std::make_shared<HostTensorBuffer>(
        std::move(desc),
        std::vector<uint8_t>(data.begin(), data.end()));
    return result;
}

Result<Device*> lookup_device(
        std::vector<std::unique_ptr<Device>>& devices,
        InvocDeviceId device) {
    if (device >= devices.size())
        return make_error("invalid device id: " + std::to_string(device));
    if (!devices[device])
        return make_error("null device: " + std::to_string(device));
    return devices[device].get();
}

Result<DeviceBufferId> lookup_runtime_buffer(
        const std::unordered_map<InvocValueId, DeviceBufferId>& buffers,
        InvocValueId value) {
    auto it = buffers.find(value);
    if (it == buffers.end())
        return make_error("missing runtime buffer for value: " + std::to_string(value));
    return it->second;
}

Result<InvocDeviceId> lookup_value_device(
        const std::unordered_map<InvocValueId, InvocDeviceId>& valueDevices,
        InvocValueId value) {
    auto it = valueDevices.find(value);
    if (it == valueDevices.end())
        return make_error("missing device for value: " + std::to_string(value));
    return it->second;
}

} // namespace

Engine::Engine(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)) {}

Result<std::unique_ptr<InvocPlan>> Engine::compile(const ir::mid_ir::Graph& graph) {
    if (devices_.empty())
        return make_error("engine has no devices");

    InvocPlanner planner(0);
    auto draftResult = planner.plan(graph);
    if (!draftResult)
        return make_error(draftResult.error());
    auto draft = draftResult.take();

    auto plan = std::make_unique<InvocPlan>();
    plan->instructions = std::move(draft.instructions);
    plan->outputs = std::move(draft.outputs);

    for (const auto& source : draft.programSources) {
        if (source.device >= devices_.size())
            return make_error("invocation program references invalid device");
        if (!source.op)
            return make_error("invocation program has no MidIR op");

        auto compiled = devices_[source.device]->compile(*source.op);
        if (!compiled)
            return make_error(compiled.error());
        plan->programs.push_back({
            source.id,
            source.device,
            compiled.take(),
        });
    }

    return plan;
}

Result<std::vector<TensorBufferPtr>> Engine::run(
        const InvocPlan& plan,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights) {
    if (devices_.empty())
        return make_error("engine has no devices");

    std::unordered_map<InvocValueId, DeviceBufferId> buffers;
    std::unordered_map<InvocValueId, InvocDeviceId> valueDevices;
    std::unordered_map<InvocProgramId, InvocProgram> programs;
    std::vector<InvocValueId> outputValues;
    std::vector<core::TensorDesc> outputDescs;

    for (const auto& program : plan.programs)
        programs[program.id] = program;

    for (const auto& instruction : plan.instructions) {
        switch (instruction.kind) {
            case InvocInstructionKind::LoadInput: {
                const auto& load = std::get<InvocLoadInput>(instruction.payload);
                auto device = lookup_device(devices_, load.device);
                if (!device)
                    return make_error(device.error());
                if (load.index < 0 || static_cast<size_t>(load.index) >= inputs.size())
                    return make_error("input index out of range: " + std::to_string(load.index));
                auto& input = inputs[static_cast<size_t>(load.index)];
                if (!input)
                    return make_error("null input buffer at index: " + std::to_string(load.index));

                auto buffer = (*device)->load(*input);
                if (!buffer)
                    return make_error(buffer.error());
                buffers[load.value] = buffer.take();
                valueDevices[load.value] = load.device;
                break;
            }
            case InvocInstructionKind::LoadWeight: {
                const auto& load = std::get<InvocLoadWeight>(instruction.payload);
                auto device = lookup_device(devices_, load.device);
                if (!device)
                    return make_error(device.error());
                auto it = weights.find(load.name);
                if (it == weights.end())
                    return make_error("missing weight buffer: " + load.name);
                if (!it->second)
                    return make_error("null weight buffer: " + load.name);

                auto buffer = (*device)->load(*it->second);
                if (!buffer)
                    return make_error(buffer.error());
                buffers[load.value] = buffer.take();
                valueDevices[load.value] = load.device;
                break;
            }
            case InvocInstructionKind::Alloc: {
                const auto& alloc = std::get<InvocAlloc>(instruction.payload);
                auto device = lookup_device(devices_, alloc.device);
                if (!device)
                    return make_error(device.error());

                auto buffer = (*device)->alloc(alloc.desc);
                if (!buffer)
                    return make_error(buffer.error());
                buffers[alloc.value] = buffer.take();
                valueDevices[alloc.value] = alloc.device;
                break;
            }
            case InvocInstructionKind::RunKernel: {
                const auto& run = std::get<InvocRunKernel>(instruction.payload);
                auto programIt = programs.find(run.program);
                if (programIt == programs.end())
                    return make_error("missing program: " + std::to_string(run.program));
                const auto& program = programIt->second;
                if (program.device != run.device)
                    return make_error("program device does not match run instruction");

                auto device = lookup_device(devices_, run.device);
                if (!device)
                    return make_error(device.error());

                std::vector<DeviceBufferId> inputBuffers;
                inputBuffers.reserve(run.inputs.size());
                for (auto value : run.inputs) {
                    auto buffer = lookup_runtime_buffer(buffers, value);
                    if (!buffer)
                        return make_error(buffer.error());
                    inputBuffers.push_back(buffer.take());
                }

                std::vector<DeviceBufferId> outputBuffers;
                outputBuffers.reserve(run.outputs.size());
                for (auto value : run.outputs) {
                    auto buffer = lookup_runtime_buffer(buffers, value);
                    if (!buffer)
                        return make_error(buffer.error());
                    outputBuffers.push_back(buffer.take());
                }

                auto result = (*device)->run(program.deviceProgram, inputBuffers, outputBuffers);
                if (!result)
                    return make_error(result.error());
                break;
            }
            case InvocInstructionKind::Dealloc: {
                const auto& dealloc = std::get<InvocDealloc>(instruction.payload);
                auto device = lookup_device(devices_, dealloc.device);
                if (!device)
                    return make_error(device.error());
                auto buffer = lookup_runtime_buffer(buffers, dealloc.value);
                if (!buffer)
                    return make_error(buffer.error());

                auto result = (*device)->dealloc(buffer.take());
                if (!result)
                    return make_error(result.error());
                buffers.erase(dealloc.value);
                valueDevices.erase(dealloc.value);
                break;
            }
            case InvocInstructionKind::StoreOutputs: {
                const auto& store = std::get<InvocStoreOutputs>(instruction.payload);
                auto device = lookup_device(devices_, store.device);
                if (!device)
                    return make_error(device.error());
                for (auto value : store.values) {
                    auto valueDevice = lookup_value_device(valueDevices, value);
                    if (!valueDevice)
                        return make_error(valueDevice.error());
                    if (valueDevice.take() != store.device)
                        return make_error("store output value is on a different device");
                    auto buffer = lookup_runtime_buffer(buffers, value);
                    if (!buffer)
                        return make_error(buffer.error());
                }
                if (!store.descs.empty() && store.descs.size() != store.values.size())
                    return make_error("store outputs descriptor count does not match value count");
                outputValues = store.values;
                outputDescs = store.descs;
                break;
            }
        }
    }

    if (outputValues.empty())
        outputValues = plan.outputs;

    std::vector<TensorBufferPtr> outputs;
    outputs.reserve(outputValues.size());
    for (size_t index = 0; index < outputValues.size(); index++) {
        auto value = outputValues[index];
        auto valueDevice = lookup_value_device(valueDevices, value);
        if (!valueDevice)
            return make_error(valueDevice.error());
        auto device = lookup_device(devices_, valueDevice.take());
        if (!device)
            return make_error(device.error());
        auto buffer = lookup_runtime_buffer(buffers, value);
        if (!buffer)
            return make_error(buffer.error());

        auto output = (*device)->read(buffer.take());
        if (!output)
            return make_error(output.error());
        auto outputBuffer = output.take();
        if (index < outputDescs.size()) {
            auto describedOutput = with_desc(std::move(outputBuffer), outputDescs[index]);
            if (!describedOutput)
                return make_error(describedOutput.error());
            outputBuffer = describedOutput.take();
        }
        outputs.push_back(std::move(outputBuffer));

        auto dealloc = (*device)->dealloc(buffers[value]);
        if (!dealloc)
            return make_error(dealloc.error());
        buffers.erase(value);
        valueDevices.erase(value);
    }

    return outputs;
}

} // namespace sandy::engine
