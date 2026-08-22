#include "BuiltinLowering.h"

namespace sandy::ir::mid_ir {

namespace {

float get_float_attr_or(const AttrMap& attrs, const std::string& name, float fallback) {
    auto it = attrs.find(name);
    if (it == attrs.end() || it->second.kind != AttrValue::Float)
        return fallback;
    return static_cast<float>(it->second.floatVal);
}

Result<float> get_constant_float(Value* value, const std::string& name) {
    if (!value->def || value->def->kind != OpKind::Constant)
        return make_error(name + " must be a constant");
    auto it = value->def->attrs.find("value");
    if (it == value->def->attrs.end() || it->second.kind != AttrValue::Float)
        return make_error(name + " constant is missing float value");
    return static_cast<float>(it->second.floatVal);
}

int64_t get_int_attr_or(const AttrMap& attrs, const std::string& name, int64_t fallback) {
    auto it = attrs.find(name);
    if (it == attrs.end() || it->second.kind != AttrValue::Int)
        return fallback;
    return it->second.intVal;
}

Result<void> expect_num_results(const std::string& name, int actual, int expected) {
    if (actual != expected)
        return make_error(name + " expects " + std::to_string(expected) + " result(s)");
    return {};
}

} // namespace

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
                         const AttrMap& attrs,
                         int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("linear", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Linear, operands, attrs);
    });

    bl.add("relu", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("relu", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::ReLU, operands, attrs);
    });

    bl.add("add", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap& attrs,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("add", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Add, operands, attrs);
    });

    bl.add("mul", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap& attrs,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("mul", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Mul, operands, attrs);
    });

    bl.add("div", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap& attrs,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("div", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Div, operands, attrs);
    });

    bl.add("sqrt", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("sqrt", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Sqrt, operands, attrs);
    });

    bl.add("tanh", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("tanh", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("tanh expects one operand");
        return builder.createOp(OpKind::Tanh, operands, attrs);
    });

    bl.add("gelu", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("gelu", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("gelu expects one operand");

        auto* x = operands[0];
        auto* half = builder.createUntypedConstantF32(0.5f);
        auto* one = builder.createUntypedConstantF32(1.0f);
        auto* cubicCoeff = builder.createUntypedConstantF32(0.044715f);
        auto* sqrtTwoOverPi = builder.createUntypedConstantF32(0.7978845608028654f);

        auto* x2 = builder.createMul(x, x);
        auto* x3 = builder.createMul(x2, x);
        auto* cubic = builder.createMul(cubicCoeff, x3);
        auto* inner = builder.createAdd(x, cubic);
        auto* scaled = builder.createMul(sqrtTwoOverPi, inner);
        auto* tanh = builder.createTanh(scaled);
        auto* onePlusTanh = builder.createAdd(one, tanh);
        auto* halfX = builder.createMul(half, x);
        return std::vector<Value*>{builder.createMul(halfX, onePlusTanh)};
    });

    bl.add("silu", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("silu", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("silu expects one operand");

        auto* x = operands[0];
        auto* half = builder.createUntypedConstantF32(0.5f);
        auto* one = builder.createUntypedConstantF32(1.0f);
        auto* halfX = builder.createMul(half, x);
        auto* tanh = builder.createTanh(halfX);
        auto* onePlusTanh = builder.createAdd(one, tanh);
        return std::vector<Value*>{builder.createMul(halfX, onePlusTanh)};
    });

    bl.add("softcap", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs,
                          int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("softcap", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1 && operands.size() != 2)
            return make_error("softcap expects one or two operands");

        float cap = 0.0f;
        if (operands.size() == 2) {
            auto capResult = get_constant_float(operands[1], "softcap cap");
            if (!capResult) return make_error(capResult.error());
            cap = capResult.take();
        } else {
            cap = get_float_attr_or(attrs, "cap", 0.0f);
        }
        if (cap <= 0.0f)
            return make_error("softcap cap must be > 0");

        auto* capConst = builder.createUntypedConstantF32(cap);
        auto* invCap = builder.createUntypedConstantF32(1.0f / cap);
        auto* scaled = builder.createMul(operands[0], invCap);
        auto* capped = builder.createTanh(scaled);
        return std::vector<Value*>{builder.createMul(capped, capConst)};
    });

    bl.add("matmul", [](Builder& builder,
                         const std::vector<Value*>& operands,
                         const AttrMap& attrs,
                         int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("matmul", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::MatMul, operands, attrs);
    });

    bl.add("transpose", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap&,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("transpose", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createTranspose(operands[0])};
    });

    bl.add("reshape", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs,
                          int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("reshape", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createReshape(operands[0], attrs.at("shape").intListVal)};
    });

    bl.add("permute", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs,
                          int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("permute", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createPermute(operands[0], attrs.at("dims").intListVal)};
    });

    bl.add("slice", [](Builder& builder,
                        const std::vector<Value*>& operands,
                        const AttrMap& attrs,
                        int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("slice", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("slice expects one operand");
        auto kinds = attrs.find("kinds");
        auto indices = attrs.find("indices");
        if (kinds == attrs.end() || kinds->second.kind != AttrValue::IntList ||
            indices == attrs.end() || indices->second.kind != AttrValue::IntList)
            return make_error("slice expects int-list kinds and indices attrs");
        return std::vector<Value*>{builder.createSlice(
            operands[0], kinds->second.intListVal, indices->second.intListVal)};
    });

    bl.add("paged_append", [](Builder& builder,
                               const std::vector<Value*>& operands,
                               const AttrMap&,
                               int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("paged_append", numResults, 0);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 2)
            return make_error("paged_append expects cache and chunk operands");
        builder.createPagedAppend(operands[0], operands[1]);
        return std::vector<Value*>{};
    });

    bl.add("sliding_query_key_score", [](Builder& builder,
                                          const std::vector<Value*>& operands,
                                          const AttrMap& attrs,
                                          int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("sliding_query_key_score", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::SlidingQueryKeyScore, operands, attrs);
    });

    bl.add("softmax", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs,
                          int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("softmax", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Softmax, operands, attrs);
    });

    bl.add("topk", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("topk", numResults, 2);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("topk expects one operand");
        return builder.createOp(OpKind::TopK, operands, attrs, 2);
    });

    bl.add("sum", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap& attrs,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("sum", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("sum expects one operand");
        return builder.createOp(OpKind::Sum, operands, attrs);
    });

    bl.add("attention", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap& attrs,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("attention", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 3 && operands.size() != 4)
            return make_error("attention expects operands (q, k, v[, position_offsets])");
        return builder.createOp(OpKind::Attention, operands, attrs);
    });

    bl.add("embedding", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap& attrs,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("embedding", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::Embedding, operands, attrs);
    });

    bl.add("moe_gather", [](Builder& builder,
                             const std::vector<Value*>& operands,
                             const AttrMap& attrs,
                             int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("moe_gather", numResults, 4);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 3)
            return make_error("moe_gather expects x, topk_ids, topk_weights");
        int64_t numExperts = get_int_attr_or(attrs, "num_experts", 0);
        int64_t topK = get_int_attr_or(attrs, "top_k", 0);
        if (numExperts <= 0 || topK <= 0)
            return make_error("moe_gather num_experts and top_k attrs must be > 0");
        return builder.createMoeGather(operands[0], operands[1], operands[2], numExperts, topK);
    });

    bl.add("moe_matmul", [](Builder& builder,
                             const std::vector<Value*>& operands,
                             const AttrMap& attrs,
                             int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("moe_matmul", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 3)
            return make_error("moe_matmul expects x, expert_offsets, weight");
        return builder.createOp(OpKind::MoeMatMul, operands, attrs);
    });

    bl.add("moe_scatter_sum", [](Builder& builder,
                                  const std::vector<Value*>& operands,
                                  const AttrMap& attrs,
                                  int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("moe_scatter_sum", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 4)
            return make_error("moe_scatter_sum expects packed_out, packed_weights, token_ids, reference");
        return builder.createOp(OpKind::MoeScatterSum, operands, attrs);
    });

    bl.add("rope", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("rope", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1 && operands.size() != 2)
            return make_error("rope expects one or two operands");
        AttrMap normalizedAttrs = attrs;
        auto it = attrs.find("theta");
        if (it != attrs.end()) {
            normalizedAttrs["rope_theta"] = it->second;
            normalizedAttrs.erase("theta");
        }
        return builder.createOp(OpKind::RoPE, operands, normalizedAttrs);
    });

    bl.add("rms_norm", [](Builder& builder,
                           const std::vector<Value*>& operands,
                           const AttrMap& attrs,
                           int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("rms_norm", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return builder.createOp(OpKind::RMSNorm, operands, attrs);
    });

    bl.add("layer_norm", [](Builder& builder,
                             const std::vector<Value*>& operands,
                             const AttrMap& attrs,
                             int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("layer_norm", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 3)
            return make_error("layer_norm expects operands (x, weight, bias)");
        return builder.createOp(OpKind::LayerNorm, operands, attrs);
    });

    return bl;
}

} // namespace sandy::ir::mid_ir
