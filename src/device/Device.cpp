#include "Device.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sandy::device {

namespace {

using ir::kernel_ir::LayoutTransformKind;
using ir::kernel_ir::LayoutTransformOp;
using ir::kernel_ir::Op;
using ir::kernel_ir::OpKind;
using ir::kernel_ir::ValueId;
using ir::kernel_ir::ValueKind;

bool is_alias(const Op& op) {
    return op.kind() == OpKind::LayoutTransform &&
        static_cast<const LayoutTransformOp&>(op).aliasesInput();
}

bool produces_dense_storage(const Op& op) {
    if (op.kind() == OpKind::Input ||
        op.kind() == OpKind::TensorTupleCreate ||
        op.kind() == OpKind::PagedAppend ||
        op.kind() == OpKind::DeviceTransfer)
        return false;
    return !is_alias(op);
}

Result<DeviceScratchLayout> plan_executable_scratch(
        Device& device,
        const DeviceExecutable& executable,
        const DeviceExecutableRunState* runState,
        bool fixed) {
    const auto& graph = executable.graph();
    const auto& desc = executable.desc();
    const auto valueCount = graph.values().size();
    std::vector<ValueId> root(valueCount);
    std::vector<size_t> lastUse(valueCount, 0);
    for (ValueId value = 0; value < valueCount; ++value)
        root[value] = value;

    for (size_t index = 0; index < desc.ops.size(); ++index) {
        const auto& op = graph.op(desc.ops[index]);
        for (auto output : op.outputs()) {
            lastUse[output] = index;
            if (is_alias(op))
                root[output] = root[op.inputs()[0]];
        }
    }
    for (size_t index = 0; index < desc.ops.size(); ++index) {
        const auto& op = graph.op(desc.ops[index]);
        for (auto input : op.inputs())
            lastUse[root[input]] = std::max(lastUse[root[input]], index);
    }

    std::vector<bool> escapedRoot(valueCount, false);
    for (auto value : desc.exports)
        escapedRoot[root[value]] = true;

    std::vector<bool> candidate(valueCount, false);
    for (auto opId : desc.ops) {
        const auto& op = graph.op(opId);
        if (!produces_dense_storage(op))
            continue;
        for (auto output : op.outputs()) {
            const auto& value = graph.value(output);
            if (value.type.kind != ValueKind::Tensor || escapedRoot[root[output]])
                continue;
            bool hasStaticShape = !value.type.shape.has_dynamic();
            if (fixed != hasStaticShape)
                continue;
            if (!fixed && executable.hasFixedScratch(output))
                continue;
            if (!fixed && (!runState || !runState->tensorDescs.contains(output)))
                return make_error("missing runtime descriptor for dynamic scratch value %" +
                                  std::to_string(output));
            candidate[output] = true;
        }
    }

    auto allocator = device.createScratchAllocator();
    if (!allocator)
        return DeviceScratchLayout{};

    std::vector<std::vector<ValueId>> releaseAt(desc.ops.size());
    for (ValueId value = 0; value < valueCount; ++value) {
        if (candidate[value])
            releaseAt[lastUse[root[value]]].push_back(value);
    }

    for (size_t index = 0; index < desc.ops.size(); ++index) {
        const auto& op = graph.op(desc.ops[index]);
        for (auto output : op.outputs()) {
            if (!candidate[output])
                continue;
            core::TensorDesc tensorDesc = fixed
                ? core::TensorDesc(graph.value(output).type.shape,
                                   graph.value(output).type.dtype)
                : runState->tensorDescs.at(output);
            auto allocated = allocator->alloc(output, std::move(tensorDesc));
            if (!allocated)
                return make_error(allocated.error());
        }
        for (auto value : releaseAt[index]) {
            auto released = allocator->free(value);
            if (!released)
                return make_error(released.error());
        }
    }
    return allocator->finalizeLayout();
}

Result<std::pair<DeviceBufferId, std::unordered_map<ValueId, DeviceTensorView>>>
instantiate_scratch(Device& device, const DeviceScratchLayout& layout) {
    if (layout.bytes == 0)
        return std::make_pair(
            DeviceBufferId{0},
            std::unordered_map<ValueId, DeviceTensorView>{});

    auto buffer = device.alloc(core::TensorDesc(
        core::Shape({static_cast<int64_t>(layout.bytes)}), core::DType::U8));
    if (!buffer)
        return make_error(buffer.error());

    std::unordered_map<ValueId, DeviceTensorView> views;
    for (const auto& [value, placement] : layout.placements) {
        auto view = device.defaultView(placement.desc);
        if (!view) {
            (void)device.dealloc(*buffer);
            return make_error(view.error());
        }
        auto elementBytes = core::dtype_size(placement.desc.dtype);
        if (elementBytes == 0 || placement.byteOffset % elementBytes != 0) {
            (void)device.dealloc(*buffer);
            return make_error("scratch placement has invalid element alignment");
        }
        view->storageOffset = static_cast<int64_t>(placement.byteOffset / elementBytes);
        views[value] = DeviceTensorView{*buffer, view.take()};
    }
    return std::make_pair(*buffer, std::move(views));
}

Result<const core::TensorDesc*> tensor_desc(
        const DeviceExecutableRunState& state,
        ValueId value) {
    auto found = state.tensorDescs.find(value);
    if (found == state.tensorDescs.end())
        return make_error("missing executable tensor descriptor for value %" +
                          std::to_string(value));
    return &found->second;
}

Result<void> execute_alias(
        Device& device,
        const LayoutTransformOp& layout,
        DeviceExecutableRunState& state) {
    auto inputIt = state.values.find(layout.inputs()[0]);
    if (inputIt == state.values.end())
        return make_error("missing alias input value");
    auto* input = std::get_if<DeviceTensorView>(&inputIt->second);
    if (!input)
        return make_error("layout alias input must be a dense tensor");
    auto outputDesc = tensor_desc(state, layout.outputs()[0]);
    if (!outputDesc)
        return make_error(outputDesc.error());

    TensorViewDesc output;
    output.desc = **outputDesc;
    output.storageOffset = input->view.storageOffset;

    if (layout.transform() == LayoutTransformKind::Slice) {
        if (layout.dims().size() != input->view.strides.size() ||
            layout.indices().size() != input->view.strides.size())
            return make_error("slice selector count must match runtime input rank");
        for (size_t axis = 0; axis < layout.dims().size(); ++axis) {
            if (layout.dims()[axis] == 0) {
                output.strides.push_back(input->view.strides[axis]);
                continue;
            }
            if (layout.dims()[axis] != 1)
                return make_error("invalid slice selector kind");
            auto dim = input->view.desc.shape.dim(static_cast<int>(axis));
            auto index = layout.indices()[axis];
            if (index < 0)
                index += dim;
            if (index < 0 || index >= dim)
                return make_error("slice index out of range");
            auto stride = input->view.strides[axis];
            if (stride < 0 || (index != 0 &&
                stride > (std::numeric_limits<int64_t>::max() - output.storageOffset) / index))
                return make_error("slice storage offset overflow");
            output.storageOffset += index * stride;
        }
    } else if (layout.transform() == LayoutTransformKind::Transpose) {
        output.strides = input->view.strides;
        if (output.strides.size() < 2)
            return make_error("transpose view input rank must be >= 2");
        std::swap(output.strides[output.strides.size() - 1],
                  output.strides[output.strides.size() - 2]);
    } else if (layout.transform() == LayoutTransformKind::Permute) {
        for (auto axis : layout.dims()) {
            if (axis < 0 || static_cast<size_t>(axis) >= input->view.strides.size())
                return make_error("permute view axis out of range");
            output.strides.push_back(input->view.strides[static_cast<size_t>(axis)]);
        }
    } else {
        auto contiguous = device.isDefaultView(input->view);
        if (!contiguous)
            return make_error(contiguous.error());
        if (!*contiguous)
            return make_error("runtime buffer violates contiguous KernelIR value requirement");
        auto defaultOutput = device.defaultView(**outputDesc);
        if (!defaultOutput)
            return make_error(defaultOutput.error());
        output = defaultOutput.take();
        output.storageOffset = input->view.storageOffset;
    }

    state.values[layout.outputs()[0]] = DeviceTensorView{input->buffer, std::move(output)};
    return {};
}

struct BufferCleanup {
    Device* device = nullptr;
    std::vector<DeviceBufferId> buffers;
    ~BufferCleanup() {
        if (!device)
            return;
        for (auto buffer : buffers)
            (void)device->dealloc(buffer);
    }
};

} // namespace

DeviceExecutable::DeviceExecutable(
        Device& owner,
        const ir::kernel_ir::Graph& graph,
        DeviceCompiledGraphId compiledGraph,
        DeviceExecutableDesc desc)
    : owner_(&owner),
      graph_(&graph),
      compiledGraph_(compiledGraph),
      desc_(std::move(desc)) {}

DeviceExecutable::~DeviceExecutable() {
    if (!owner_)
        return;
    (void)owner_->destroyCompiledGraph(compiledGraph_);
    if (fixedScratchBuffer_ != 0)
        (void)owner_->dealloc(fixedScratchBuffer_);
}

bool DeviceExecutable::inputsAndOutputsFixed(const ir::kernel_ir::Op& op) const {
    for (auto value : op.inputs()) {
        if (!fixedBindingValues_.contains(value))
            return false;
    }
    for (auto value : op.outputs()) {
        if (!fixedBindingValues_.contains(value))
            return false;
    }
    return true;
}

Result<DeviceExecutablePtr> Device::compileExecutable(
        const ir::kernel_ir::Graph& graph,
        DeviceExecutableDesc desc) {
    auto compiled = compileExecutableGraph(graph, desc.ops);
    if (!compiled)
        return make_error(compiled.error());
    auto executable = DeviceExecutablePtr(
        new DeviceExecutable(*this, graph, *compiled, std::move(desc)));
    auto layout = plan_executable_scratch(*this, *executable, nullptr, true);
    if (!layout)
        return make_error(layout.error());
    auto scratch = instantiate_scratch(*this, *layout);
    if (!scratch)
        return make_error(scratch.error());
    executable->fixedScratchBuffer_ = scratch->first;
    executable->fixedViews_ = std::move(scratch->second);
    for (const auto& [value, _] : executable->fixedViews_)
        executable->fixedBindingValues_.insert(value);
    for (auto value : executable->desc_.stableImports)
        executable->fixedBindingValues_.insert(value);
    for (auto opId : executable->desc_.ops) {
        const auto& op = graph.op(opId);
        if (!is_alias(op) || graph.value(op.outputs()[0]).type.shape.has_dynamic())
            continue;
        if (executable->fixedBindingValues_.contains(op.inputs()[0]))
            executable->fixedBindingValues_.insert(op.outputs()[0]);
    }
    return executable;
}

Result<void> Device::executeCommands(
        DeviceCompiledGraphId graph,
        std::span<const DeviceRunCommand> commands,
        const std::function<void(ir::kernel_ir::OpId, double)>& profileKernel) {
    for (const auto& command : commands) {
        auto start = std::chrono::steady_clock::now();
        auto ran = run(graph, command.op, command.inputs, command.outputs);
        auto end = std::chrono::steady_clock::now();
        if (!ran)
            return make_error(ran.error());
        if (profileKernel) {
            profileKernel(
                command.op,
                std::chrono::duration<double, std::milli>(end - start).count());
        }
    }
    return {};
}

Result<void> Device::runExecutable(
        const DeviceExecutable& executable,
        DeviceExecutableRunState& state) {
    if (executable.owner_ != this)
        return make_error("device executable belongs to another device");

    for (const auto& [value, view] : executable.fixedViews_)
        state.values[value] = view;

    auto dynamicLayout = plan_executable_scratch(*this, executable, &state, false);
    if (!dynamicLayout)
        return make_error(dynamicLayout.error());
    auto dynamicScratch = instantiate_scratch(*this, *dynamicLayout);
    if (!dynamicScratch)
        return make_error(dynamicScratch.error());
    BufferCleanup cleanup{this};
    if (dynamicScratch->first != 0)
        cleanup.buffers.push_back(dynamicScratch->first);
    for (auto& [value, view] : dynamicScratch->second)
        state.values[value] = std::move(view);

    std::unordered_set<ValueId> exports(
        executable.desc_.exports.begin(), executable.desc_.exports.end());
    std::vector<DeviceRunCommand> commands;
    auto flushCommands = [&]() -> Result<void> {
        if (commands.empty())
            return {};
        auto executed = executeCommands(
            executable.compiledGraph_, commands, state.profileKernel);
        commands.clear();
        return executed;
    };
    for (auto opId : executable.desc_.ops) {
        const auto& op = executable.graph_->op(opId);
        if (is_alias(op)) {
            auto aliased = execute_alias(
                *this, static_cast<const LayoutTransformOp&>(op), state);
            if (!aliased)
                return make_error(aliased.error());
            continue;
        }
        if (op.kind() == OpKind::PagedAppend) {
            auto flushed = flushCommands();
            if (!flushed)
                return make_error(flushed.error());
            const auto& append = static_cast<const ir::kernel_ir::PagedAppendOp&>(op);
            auto cacheIt = state.values.find(append.cache());
            auto chunkIt = state.values.find(append.chunk());
            if (cacheIt == state.values.end() || chunkIt == state.values.end())
                return make_error("paged append executable binding is missing");
            auto* cache = std::get_if<DevicePagedTensorView>(&cacheIt->second);
            auto* chunk = std::get_if<DeviceTensorView>(&chunkIt->second);
            if (!cache || !chunk)
                return make_error("paged append executable binding type mismatch");
            auto appended = appendPaged(cache->tensor, *chunk);
            if (!appended)
                return make_error(appended.error());
            auto meta = pagedMeta(cache->tensor);
            if (!meta)
                return make_error(meta.error());
            cache->meta = meta.take();
            continue;
        }

        for (auto output : op.outputs()) {
            if (state.values.contains(output))
                continue;
            if (exports.contains(output))
                return make_error("missing exported output binding for value %" +
                                  std::to_string(output));
            auto desc = tensor_desc(state, output);
            if (!desc)
                return make_error(desc.error());
            auto buffer = alloc(**desc);
            if (!buffer)
                return make_error(buffer.error());
            auto view = defaultView(**desc);
            if (!view) {
                (void)dealloc(*buffer);
                return make_error(view.error());
            }
            cleanup.buffers.push_back(*buffer);
            state.values[output] = DeviceTensorView{*buffer, view.take()};
        }

        DeviceRunCommand command;
        command.op = op.id();
        command.bindingsFixed = executable.inputsAndOutputsFixed(op);
        for (auto input : op.inputs()) {
            auto found = state.values.find(input);
            if (found == state.values.end())
                return make_error("missing executable input value %" + std::to_string(input));
            command.inputs.push_back(found->second);
        }
        for (auto output : op.outputs()) {
            auto found = state.values.find(output);
            if (found == state.values.end())
                return make_error("missing executable output value %" + std::to_string(output));
            command.outputs.push_back(found->second);
        }
        commands.push_back(std::move(command));
    }
    return flushCommands();
}

std::unique_ptr<DeviceScratchAllocator> Device::createScratchAllocator() {
    return nullptr;
}

Result<void> Device::destroyCompiledGraph(DeviceCompiledGraphId) {
    return {};
}

Result<DeviceCompiledGraphId> Device::compileExecutableGraph(
        const ir::kernel_ir::Graph& graph,
        std::span<const ir::kernel_ir::OpId>) {
    return compile(graph);
}

Result<DevicePagedPoolId> Device::createPagedPool(DevicePagedPoolDesc desc) {
    if (desc.pageSize <= 0)
        return make_error("paged tensor pool page_size must be > 0");
    if (!std::has_single_bit(static_cast<uint64_t>(desc.pageSize)))
        return make_error("paged tensor pool page_size must be a power of two");
    return createPagedPoolImpl(std::move(desc));
}

Result<DevicePagedPoolId> Device::createPagedPoolImpl(DevicePagedPoolDesc) {
    return make_error("device does not support paged tensor pools");
}

Result<void> Device::destroyPagedPool(DevicePagedPoolId) {
    return make_error("device does not support paged tensor pools");
}

Result<DevicePagedTensorId> Device::allocPaged(DevicePagedPoolId, core::Shape) {
    return make_error("device does not support paged tensors");
}

Result<void> Device::deallocPaged(DevicePagedTensorId) {
    return make_error("device does not support paged tensors");
}

Result<void> Device::reservePaged(DevicePagedTensorId, int64_t) {
    return make_error("device does not support paged tensors");
}

Result<void> Device::appendPaged(DevicePagedTensorId, core::TensorBuffer&) {
    return make_error("device does not support paged tensors");
}

Result<void> Device::appendPaged(
        DevicePagedTensorId dst,
        DeviceTensorView denseChunk) {
    auto hostChunk = read(std::move(denseChunk));
    if (!hostChunk)
        return make_error(hostChunk.error());
    return appendPaged(dst, **hostChunk);
}

Result<DevicePagedTensorMeta> Device::pagedMeta(DevicePagedTensorId) const {
    return make_error("device does not support paged tensors");
}

Result<TensorBufferPtr> Device::read(DevicePagedTensorView) {
    return make_error("device does not support reading paged tensors");
}

Result<std::vector<int64_t>> Device::defaultStrides(const core::Shape& shape) const {
    if (shape.has_dynamic())
        return make_error("cannot compute default strides for dynamic shape");

    std::vector<int64_t> strides(static_cast<size_t>(shape.rank()), 1);
    int64_t stride = 1;
    for (int i = shape.rank() - 1; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= shape.dim(i);
    }
    return strides;
}

Result<TensorViewDesc> Device::defaultView(core::TensorDesc desc) const {
    auto strides = defaultStrides(desc.shape);
    if (!strides)
        return make_error(strides.error());

    TensorViewDesc view;
    view.desc = std::move(desc);
    view.strides = strides.take();
    view.storageOffset = 0;
    return view;
}

Result<bool> Device::isDefaultView(const TensorViewDesc& view) const {
    auto strides = defaultStrides(view.desc.shape);
    if (!strides)
        return make_error(strides.error());
    if (view.strides.size() != strides->size())
        return false;
    for (size_t axis = 0; axis < view.strides.size(); ++axis) {
        if (view.desc.shape.dim(static_cast<int>(axis)) != 1 &&
            view.strides[axis] != (*strides)[axis])
            return false;
    }
    return true;
}

} // namespace sandy::device
