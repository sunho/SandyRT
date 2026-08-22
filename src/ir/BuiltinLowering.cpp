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
                         const AttrMap&,
                         int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("linear", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createLinear(operands[0], operands[1], operands[2])};
    });

    bl.add("relu", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("relu", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createReLU(operands[0])};
    });

    bl.add("add", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap&,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("add", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createAdd(operands[0], operands[1])};
    });

    bl.add("mul", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap&,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("mul", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createMul(operands[0], operands[1])};
    });

    bl.add("div", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap&,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("div", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createDiv(operands[0], operands[1])};
    });

    bl.add("sqrt", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("sqrt", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createSqrt(operands[0])};
    });

    bl.add("tanh", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap&,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("tanh", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("tanh expects one operand");
        return std::vector<Value*>{builder.createTanh(operands[0])};
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
        auto* half = builder.createConstantF32(0.5f);
        auto* one = builder.createConstantF32(1.0f);
        auto* cubicCoeff = builder.createConstantF32(0.044715f);
        auto* sqrtTwoOverPi = builder.createConstantF32(0.7978845608028654f);

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
        auto* half = builder.createConstantF32(0.5f);
        auto* one = builder.createConstantF32(1.0f);
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

        auto* capConst = builder.createConstantF32(cap);
        auto* invCap = builder.createConstantF32(1.0f / cap);
        auto* scaled = builder.createMul(operands[0], invCap);
        auto* capped = builder.createTanh(scaled);
        return std::vector<Value*>{builder.createMul(capped, capConst)};
    });

    bl.add("matmul", [](Builder& builder,
                         const std::vector<Value*>& operands,
                         const AttrMap&,
                         int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("matmul", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createMatMul(operands[0], operands[1])};
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
        int64_t window = 0;
        auto it = attrs.find("window");
        if (it != attrs.end() && it->second.kind == AttrValue::Int)
            window = it->second.intVal;
        float scale = -1.0f;
        auto scaleIt = attrs.find("scale");
        if (scaleIt != attrs.end()) {
            if (scaleIt->second.kind != AttrValue::Float)
                return make_error("sliding_query_key_score scale attr must be float");
            scale = static_cast<float>(scaleIt->second.floatVal);
        }
        if (operands.size() == 2)
            return std::vector<Value*>{builder.createSlidingQueryKeyScore(operands[0], operands[1], window, scale)};
        if (operands.size() == 3)
            return std::vector<Value*>{builder.createSlidingQueryKeyScore(operands[0], operands[1], operands[2], window, scale)};
        return make_error("sliding_query_key_score expects 2 or 3 operands");
    });

    bl.add("softmax", [](Builder& builder,
                          const std::vector<Value*>& operands,
                          const AttrMap& attrs,
                          int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("softmax", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        int64_t dim = -1;
        auto it = attrs.find("dim");
        if (it != attrs.end() && it->second.kind == AttrValue::Int)
            dim = it->second.intVal;
        return std::vector<Value*>{builder.createSoftmax(operands[0], dim)};
    });

    bl.add("topk", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("topk", numResults, 2);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("topk expects one operand");
        int64_t k = get_int_attr_or(attrs, "k", 0);
        if (k <= 0)
            return make_error("topk k attr must be > 0");
        int64_t dim = get_int_attr_or(attrs, "dim", -1);
        return builder.createTopK(operands[0], k, dim);
    });

    bl.add("sum", [](Builder& builder,
                      const std::vector<Value*>& operands,
                      const AttrMap& attrs,
                      int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("sum", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("sum expects one operand");
        int64_t dim = get_int_attr_or(attrs, "dim", -1);
        bool keepDims = false;
        auto keepDim = attrs.find("keepdim");
        if (keepDim == attrs.end())
            keepDim = attrs.find("keepdims");
        if (keepDim != attrs.end()) {
            if (keepDim->second.kind != AttrValue::Int)
                return make_error("sum keepdim attr must be int");
            keepDims = keepDim->second.intVal != 0;
        }
        return std::vector<Value*>{builder.createSum(operands[0], dim, keepDims)};
    });

    bl.add("attention", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap& attrs,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("attention", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 3 && operands.size() != 4)
            return make_error("attention expects operands (q, k, v[, position_offsets])");
        int64_t window = get_int_attr_or(attrs, "window", 0);
        float scale = get_float_attr_or(attrs, "scale", -1.0f);
        if (operands.size() == 4)
            return std::vector<Value*>{builder.createAttention(operands[0], operands[1], operands[2], operands[3], window, scale)};
        return std::vector<Value*>{builder.createAttention(operands[0], operands[1], operands[2], window, scale)};
    });

    bl.add("embedding", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap&,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("embedding", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createEmbedding(operands[0], operands[1])};
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
        bool transposeRhs = false;
        auto transposeIt = attrs.find("transpose_rhs");
        if (transposeIt != attrs.end()) {
            if (transposeIt->second.kind != AttrValue::Int)
                return make_error("moe_matmul transpose_rhs attr must be int");
            transposeRhs = transposeIt->second.intVal != 0;
        }
        return std::vector<Value*>{builder.createMoeMatMul(operands[0], operands[1], operands[2], transposeRhs)};
    });

    bl.add("moe_scatter_sum", [](Builder& builder,
                                  const std::vector<Value*>& operands,
                                  const AttrMap&,
                                  int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("moe_scatter_sum", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 4)
            return make_error("moe_scatter_sum expects packed_out, packed_weights, token_ids, reference");
        return std::vector<Value*>{builder.createMoeScatterSum(operands[0], operands[1], operands[2], operands[3])};
    });

    bl.add("rope", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("rope", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1 && operands.size() != 2)
            return make_error("rope expects one or two operands");
        float theta = 10000.0f;
        auto it = attrs.find("theta");
        if (it == attrs.end())
            it = attrs.find("rope_theta");
        if (it != attrs.end() && it->second.kind == AttrValue::Float)
            theta = static_cast<float>(it->second.floatVal);
        int64_t rotaryDim = -1;
        auto rotaryIt = attrs.find("rotary_dim");
        if (rotaryIt != attrs.end()) {
            if (rotaryIt->second.kind != AttrValue::Int)
                return make_error("rope rotary_dim attr must be int");
            rotaryDim = rotaryIt->second.intVal;
        }
        bool splitHalf = false;
        auto splitIt = attrs.find("split_half");
        if (splitIt != attrs.end()) {
            if (splitIt->second.kind != AttrValue::Int)
                return make_error("rope split_half attr must be int");
            splitHalf = splitIt->second.intVal != 0;
        }
        if (operands.size() == 2)
            return std::vector<Value*>{builder.createRoPE(operands[0], operands[1], theta, rotaryDim, splitHalf)};
        return std::vector<Value*>{builder.createRoPE(operands[0], theta, rotaryDim, splitHalf)};
    });

    bl.add("rms_norm", [](Builder& builder,
                           const std::vector<Value*>& operands,
                           const AttrMap& attrs,
                           int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("rms_norm", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        float epsilon = 1.0e-6f;
        auto it = attrs.find("epsilon");
        if (it != attrs.end() && it->second.kind == AttrValue::Float) {
            epsilon = static_cast<float>(it->second.floatVal);
        }
        if (operands.size() == 1)
            return std::vector<Value*>{builder.createRMSNorm(operands[0], epsilon)};
        if (operands.size() == 2)
            return std::vector<Value*>{builder.createRMSNorm(operands[0], operands[1], epsilon)};
        return make_error("rms_norm expects operands (x[, weight])");
    });

    bl.add("layer_norm", [](Builder& builder,
                             const std::vector<Value*>& operands,
                             const AttrMap& attrs,
                             int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("layer_norm", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 3)
            return make_error("layer_norm expects operands (x, weight, bias)");
        float epsilon = 1.0e-5f;
        auto it = attrs.find("epsilon");
        if (it != attrs.end() && it->second.kind == AttrValue::Float) {
            epsilon = static_cast<float>(it->second.floatVal);
        }
        return std::vector<Value*>{builder.createLayerNorm(
            operands[0], operands[1], operands[2], epsilon)};
    });

    return bl;
}

} // namespace sandy::ir::mid_ir
