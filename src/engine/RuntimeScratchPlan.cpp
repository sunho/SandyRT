#include "RuntimeScratchPlan.h"

#include <optional>

namespace sandy::engine {

namespace {

using ir::kernel_ir::DeviceId;
using ir::kernel_ir::Graph;
using ir::kernel_ir::Op;
using ir::kernel_ir::OpKind;
using ir::kernel_ir::ValueId;

Result<device::Device*> lookupDevice(
        std::vector<std::unique_ptr<device::Device>>& devices,
        DeviceId id) {
    if (id >= devices.size())
        return make_error("invalid device id: " + std::to_string(id));
    if (!devices[id])
        return make_error("null device: " + std::to_string(id));
    return devices[id].get();
}

DeviceId runtimeOpDevice(const CompiledKernelGraph& compiled, const Op& op) {
    return compiled.deviceGraphs.empty() ? compiled.device : op.device();
}

std::vector<bool> excludedValues(const Graph& graph) {
    std::vector<bool> excluded(graph.values().size(), false);
    for (auto output : graph.outputs()) {
        excluded[output] = true;
        const auto& value = graph.value(output);
        if (value.type.kind != ir::kernel_ir::ValueKind::TensorTuple ||
            value.def.op == ir::kernel_ir::kInvalidOpId)
            continue;
        for (auto element : graph.op(value.def.op).inputs())
            excluded[element] = true;
    }

    // Layout transforms may produce views that outlive their input value. The
    // scratch planner tracks value lifetimes rather than backing-buffer alias
    // lifetimes, so keep every possible alias chain on the regular ref-counted
    // allocation path.
    for (const auto& op : graph.ops()) {
        if (op->kind() != OpKind::LayoutTransform)
            continue;
        if (!static_cast<const ir::kernel_ir::LayoutTransformOp&>(*op).aliasesInput())
            continue;
        for (auto input : op->inputs())
            excluded[input] = true;
        for (auto output : op->outputs())
            excluded[output] = true;
    }
    return excluded;
}

bool mayRequireDenseAllocation(const Op& op) {
    if (op.kind() == OpKind::Input ||
        op.kind() == OpKind::DeviceTransfer ||
        op.kind() == OpKind::TensorTupleCreate ||
        op.kind() == OpKind::PagedAppend)
        return false;

    if (op.kind() != OpKind::LayoutTransform)
        return true;

    auto transform =
        static_cast<const ir::kernel_ir::LayoutTransformOp&>(op).transform();
    return transform == ir::kernel_ir::LayoutTransformKind::Reshape ||
           transform == ir::kernel_ir::LayoutTransformKind::Contiguous;
}

} // namespace

Result<RuntimeScratchLayout> planScratchLayout(
        const CompiledKernelGraph& compiled,
        const RuntimeTensorDescs* tensorDescs,
        const std::vector<bool>* includedValues,
        const std::vector<bool>* excludedScratchValues,
        std::vector<std::unique_ptr<device::Device>>& devices) {
    const auto& graph = *compiled.graph;
    auto excluded = excludedValues(graph);
    std::vector<size_t> remainingUses(graph.values().size(), 0);
    std::vector<std::optional<DeviceId>> plannedDevice(graph.values().size());
    for (const auto& value : graph.values())
        remainingUses[value.id] = value.uses.size();

    std::vector<std::unique_ptr<device::DeviceScratchAllocator>> allocators(devices.size());
    std::vector<bool> allocatorChecked(devices.size(), false);
    auto allocatorFor = [&](DeviceId id) -> Result<device::DeviceScratchAllocator*> {
        auto found = lookupDevice(devices, id);
        if (!found)
            return make_error(found.error());
        if (!allocatorChecked[id]) {
            allocators[id] = (*found)->createScratchAllocator();
            allocatorChecked[id] = true;
        }
        return allocators[id].get();
    };

    for (const auto& opPtr : graph.ops()) {
        const auto& op = *opPtr;
        if (mayRequireDenseAllocation(op)) {
            auto opDevice = runtimeOpDevice(compiled, op);
            auto allocator = allocatorFor(opDevice);
            if (!allocator)
                return make_error(allocator.error());
            if (*allocator) {
                for (auto output : op.outputs()) {
                    const auto& value = graph.value(output);
                    if (excluded[output] ||
                        value.type.kind != ir::kernel_ir::ValueKind::Tensor ||
                        (includedValues && !(*includedValues)[output]) ||
                        (excludedScratchValues && output < excludedScratchValues->size() &&
                         (*excludedScratchValues)[output]))
                        continue;
                    core::TensorDesc desc;
                    if (tensorDescs) {
                        desc = tensorDescs->get(output);
                    } else {
                        if (value.type.shape.has_dynamic())
                            continue;
                        desc = core::TensorDesc(value.type.shape, value.type.dtype);
                    }
                    auto allocated = (*allocator)->alloc(output, std::move(desc));
                    if (!allocated)
                        return make_error(allocated.error());
                    plannedDevice[output] = opDevice;
                }
            }
        }

        auto finishValue = [&](ValueId value) -> Result<void> {
            if (remainingUses[value] == 0)
                return make_error("KernelIR value use count underflow during scratch planning");
            remainingUses[value]--;
            if (remainingUses[value] != 0 || !plannedDevice[value])
                return {};
            auto allocator = allocatorFor(*plannedDevice[value]);
            if (!allocator)
                return make_error(allocator.error());
            return (*allocator)->free(value);
        };

        for (auto input : op.inputs()) {
            auto finished = finishValue(input);
            if (!finished)
                return make_error(finished.error());
        }
        for (auto output : op.outputs()) {
            if (remainingUses[output] != 0 || !plannedDevice[output])
                continue;
            auto allocator = allocatorFor(*plannedDevice[output]);
            if (!allocator)
                return make_error(allocator.error());
            auto freed = (*allocator)->free(output);
            if (!freed)
                return make_error(freed.error());
        }
    }

    RuntimeScratchLayout plan;
    for (DeviceId id = 0; id < allocators.size(); ++id) {
        if (!allocators[id])
            continue;
        auto allocation = allocators[id]->finalizeLayout();
        if (!allocation) {
            return make_error(allocation.error());
        }
        if (allocation->bytes != 0)
            plan.devices[id] = std::move(*allocation);
    }
    return plan;
}

Result<CompileTimeScratchPlan> planCompileTimeScratchLayout(
        const CompiledKernelGraph& compiled,
        std::vector<std::unique_ptr<device::Device>>& devices) {
    std::vector<bool> candidates(compiled.graph->values().size(), false);
    auto excluded = excludedValues(*compiled.graph);
    for (const auto& opPtr : compiled.graph->ops()) {
        const auto& op = *opPtr;
        if (!mayRequireDenseAllocation(op))
            continue;
        for (auto output : op.outputs()) {
            const auto& value = compiled.graph->value(output);
            candidates[output] = !excluded[output] &&
                value.type.kind == ir::kernel_ir::ValueKind::Tensor &&
                !value.type.shape.has_dynamic();
        }
    }

    auto layout = planScratchLayout(compiled, nullptr, &candidates, nullptr, devices);
    if (!layout)
        return make_error(layout.error());

    // Only suppress a value from the runtime planner when a device allocator
    // actually placed it in the fixed pool. Devices may opt out of scratch
    // planning by returning no allocator.
    std::vector<bool> fixed(compiled.graph->values().size(), false);
    for (const auto& [_, deviceLayout] : layout->devices) {
        for (const auto& [value, __] : deviceLayout.placements)
            fixed[value] = true;
    }
    return CompileTimeScratchPlan{layout.take(), std::move(fixed)};
}

Result<RuntimeScratchLayout> planRuntimeScratchLayout(
        const CompiledKernelGraph& compiled,
        const RuntimeTensorDescs& tensorDescs,
        std::vector<std::unique_ptr<device::Device>>& devices) {
    auto layout = planScratchLayout(
        compiled,
        &tensorDescs,
        nullptr,
        &compiled.staticScratchValues,
        devices);
    if (!layout)
        return make_error(layout.error());
    return layout.take();
}

Result<RuntimeScratchPlan> instantiateRuntimeScratch(
        const RuntimeScratchLayout& layout,
        std::vector<std::unique_ptr<device::Device>>& devices) {
    RuntimeScratchPlan plan;
    for (const auto& [id, deviceLayout] : layout.devices) {
        auto found = lookupDevice(devices, id);
        if (!found)
            return make_error(found.error());
        auto buffer = (*found)->alloc(core::TensorDesc(
            core::Shape({static_cast<int64_t>(deviceLayout.bytes)}),
            core::DType::U8));
        if (!buffer) {
            for (const auto& [allocatedDevice, allocatedBuffer] : plan.buffers) {
                auto allocated = lookupDevice(devices, allocatedDevice);
                if (allocated)
                    (void)(*allocated)->dealloc(allocatedBuffer);
            }
            return make_error(buffer.error());
        }
        plan.buffers[id] = *buffer;
        for (const auto& [value, placement] : deviceLayout.placements) {
            auto view = (*found)->defaultView(placement.desc);
            if (!view) {
                for (const auto& [allocatedDevice, allocatedBuffer] : plan.buffers) {
                    auto allocated = lookupDevice(devices, allocatedDevice);
                    if (allocated)
                        (void)(*allocated)->dealloc(allocatedBuffer);
                }
                return make_error(view.error());
            }
            auto elementBytes = core::dtype_size(placement.desc.dtype);
            view->storageOffset = static_cast<int64_t>(placement.byteOffset / elementBytes);
            plan.devices[value] = id;
            plan.views[value] = device::DeviceTensorView{*buffer, view.take()};
        }
    }
    return plan;
}

Result<RuntimePlanCache::PlanPtr> RuntimePlanCache::getOrCreate(
        const core::CacheKey& key,
        const Factory& factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) {
        hits_++;
        return found->second;
    }
    auto created = factory();
    if (!created)
        return make_error(created.error());
    auto plan = std::make_shared<const CachedInvocationPlan>(created.take());
    entries_.emplace(key, plan);
    misses_++;
    return plan;
}

RuntimePlanCacheStats RuntimePlanCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return RuntimePlanCacheStats{hits_, misses_, entries_.size()};
}

} // namespace sandy::engine
