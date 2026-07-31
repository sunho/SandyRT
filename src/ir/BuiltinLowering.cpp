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
