#include "InvocationCacheKey.h"

namespace sandy::engine {

Result<core::CacheKey> buildInvocationCacheKey(
        std::string_view domain,
        CompiledProgramId program,
        const ir::kernel_ir::Graph& graph,
        const RuntimeTensorDescs& inputDescs) {
    core::CacheKeyBuilder key(domain);
    key.addU64(program);

    size_t inputCount = 0;
    for (const auto& op : graph.ops()) {
        if (op->kind() == ir::kernel_ir::OpKind::Input)
            inputCount++;
    }
    key.addU64(static_cast<uint64_t>(inputCount));

    for (const auto& op : graph.ops()) {
        if (op->kind() != ir::kernel_ir::OpKind::Input)
            continue;
        const auto& input = static_cast<const ir::kernel_ir::InputOp&>(*op);
        auto output = input.outputs()[0];
        auto desc = inputDescs.lookup(output);
        if (!desc)
            return make_error(desc.error());
        const auto& type = graph.value(output).type;

        key.addU32(static_cast<uint32_t>(type.kind));
        key.addI64(input.source().index);
        key.addI64(input.source().tupleElement);
        key.addTensorDesc(**desc);
        if (type.kind == ir::kernel_ir::ValueKind::PagedTensor) {
            key.addI64(type.paged.growDim);
            key.addI64(type.paged.pageSize);
        }
    }
    return std::move(key).finish();
}

} // namespace sandy::engine

