#include "BuiltinLowering.h"

namespace sandy::ir::mid_ir {

void BuiltinLowering::add(const std::string& name, LowerFn fn) {
    lowerings_[name] = std::move(fn);
}

const BuiltinLowering::LowerFn* BuiltinLowering::lookup(const std::string& name) const {
    auto it = lowerings_.find(name);
    if (it == lowerings_.end()) return nullptr;
    return &it->second;
}

BuiltinLowering BuiltinLowering::createDefault() {
    BuiltinLowering bl;

    bl.add("linear", [](Builder& builder,
                         const std::vector<Value*>& operands,
                         const AttrMap&) -> std::vector<Value*> {
        return {builder.createLinear(operands[0], operands[1], operands[2])};
    });

    bl.add("relu", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&) -> std::vector<Value*> {
        return {builder.createReLU(operands[0])};
    });

    bl.add("add", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap&) -> std::vector<Value*> {
        return {builder.createAdd(operands[0], operands[1])};
    });

    bl.add("mul", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap&) -> std::vector<Value*> {
        return {builder.createMul(operands[0], operands[1])};
    });

    bl.add("sqrt", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&) -> std::vector<Value*> {
        return {builder.createSqrt(operands[0])};
    });

    bl.add("matmul", [](Builder& builder,
                         const std::vector<Value*>& operands,
                         const AttrMap&) -> std::vector<Value*> {
        return {builder.createMatMul(operands[0], operands[1])};
    });

    bl.add("transpose", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap&) -> std::vector<Value*> {
        return {builder.createTranspose(operands[0])};
    });

    bl.add("reshape", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs) -> std::vector<Value*> {
        return {builder.createReshape(operands[0], attrs.at("shape").intListVal)};
    });

    bl.add("permute", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs) -> std::vector<Value*> {
        return {builder.createPermute(operands[0], attrs.at("dims").intListVal)};
    });

    bl.add("sliding_query_key_score", [](Builder& builder,
                                          const std::vector<Value*>& operands,
                                          const AttrMap& attrs) -> std::vector<Value*> {
        int64_t window = 0;
        auto it = attrs.find("window");
        if (it != attrs.end() && it->second.kind == AttrValue::Int)
            window = it->second.intVal;
        return {builder.createSlidingQueryKeyScore(operands[0], operands[1], window)};
    });

    bl.add("softmax", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs) -> std::vector<Value*> {
        int64_t dim = -1;
        auto it = attrs.find("dim");
        if (it != attrs.end() && it->second.kind == AttrValue::Int)
            dim = it->second.intVal;
        return {builder.createSoftmax(operands[0], dim)};
    });

    bl.add("embedding", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap&) -> std::vector<Value*> {
        return {builder.createEmbedding(operands[0], operands[1])};
    });

    bl.add("rms_norm", [](Builder& builder,
                           const std::vector<Value*>& operands,
                           const AttrMap& attrs) -> std::vector<Value*> {
        float epsilon = 1.0e-6f;
        auto it = attrs.find("epsilon");
        if (it != attrs.end() && it->second.kind == AttrValue::Float) {
            epsilon = static_cast<float>(it->second.floatVal);
        }
        return {builder.createRMSNorm(operands[0], operands[1], epsilon)};
    });

    return bl;
}

} // namespace sandy::ir::mid_ir
