#include "BuiltinLowering.h"

namespace sandy::ir::mid_ir {

namespace {

Result<int64_t> get_int_attr(const AttrMap& attrs, const std::string& name) {
    auto it = attrs.find(name);
    if (it == attrs.end() || it->second.kind != AttrValue::Int)
        return make_error("missing int attr '" + name + "'");
    return it->second.intVal;
}

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
        return std::vector<Value*>{builder.createSlidingQueryKeyScore(operands[0], operands[1], window, scale)};
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

    bl.add("kv_attention", [](Builder& builder,
                               const std::vector<Value*>& operands,
                               const AttrMap& attrs,
                               int numResults) -> Result<std::vector<Value*>> {
        if (numResults != 1 && numResults != 3)
            return make_error("kv_attention expects 1 or 3 result(s)");
        if (operands.size() != 5)
            return make_error("kv_attention expects operands (x, q_weight, k_weight, v_weight, o_weight)");

        auto headsResult = get_int_attr(attrs, "heads");
        if (!headsResult) return make_error(headsResult.error());
        int64_t heads = headsResult.take();

        auto kvHeadsResult = get_int_attr(attrs, "kv_heads");
        if (!kvHeadsResult) return make_error(kvHeadsResult.error());
        int64_t kvHeads = kvHeadsResult.take();

        auto headDimResult = get_int_attr(attrs, "head_dim");
        if (!headDimResult) return make_error(headDimResult.error());
        int64_t headDim = headDimResult.take();

        int64_t window = get_int_attr_or(attrs, "window", 0);
        float ropeTheta = get_float_attr_or(attrs, "rope_theta", 0.0f);
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0)
            return make_error("kv_attention heads, kv_heads, and head_dim must be positive");
        if (heads % kvHeads != 0)
            return make_error("kv_attention heads must be divisible by kv_heads");
        auto* x = operands[0];
        int rank = x->shape.rank();
        if (rank != 2 && rank != 3)
            return make_error("kv_attention input must have rank 2 or rank 3");
        int64_t hidden = x->shape.dim(rank - 1);
        if (hidden < 0)
            return make_error("kv_attention hidden dimension must be static");

        auto* qWeightT = builder.createTranspose(operands[1]);
        auto* kWeightT = builder.createTranspose(operands[2]);
        auto* vWeightT = builder.createTranspose(operands[3]);
        auto* oWeightT = builder.createTranspose(operands[4]);

        auto* qFlat = builder.createMatMul(x, qWeightT);
        auto* kFlat = builder.createMatMul(x, kWeightT);
        auto* vFlat = builder.createMatMul(x, vWeightT);

        Value* q = nullptr;
        Value* k = nullptr;
        Value* v = nullptr;
        Value* contextFlat = nullptr;

        if (rank == 3) {
            int64_t batch = x->shape.dim(0);
            int64_t seq = x->shape.dim(1);
            if (batch < 0 || seq < 0)
                return make_error("kv_attention batch and sequence dimensions must be static");

            q = builder.createPermute(
                builder.createReshape(qFlat, {batch, seq, heads, headDim}),
                {0, 2, 1, 3});
            k = builder.createPermute(
                builder.createReshape(kFlat, {batch, seq, kvHeads, headDim}),
                {0, 2, 1, 3});
            v = builder.createPermute(
                builder.createReshape(vFlat, {batch, seq, kvHeads, headDim}),
                {0, 2, 1, 3});
            if (ropeTheta > 0.0f) {
                q = builder.createRoPE(q, ropeTheta);
                k = builder.createRoPE(k, ropeTheta);
            }

            auto* scores = builder.createSlidingQueryKeyScore(q, k, window);
            auto* probs = builder.createSoftmax(scores, -1);
            auto* context = builder.createMatMul(probs, v);
            auto* contextSeqMajor = builder.createPermute(context, {0, 2, 1, 3});
            contextFlat = builder.createReshape(contextSeqMajor, {batch, seq, heads * headDim});
        } else {
            int64_t seq = x->shape.dim(0);
            if (seq < 0)
                return make_error("kv_attention sequence dimension must be static");

            q = builder.createPermute(
                builder.createReshape(qFlat, {seq, heads, headDim}),
                {1, 0, 2});
            k = builder.createPermute(
                builder.createReshape(kFlat, {seq, kvHeads, headDim}),
                {1, 0, 2});
            v = builder.createPermute(
                builder.createReshape(vFlat, {seq, kvHeads, headDim}),
                {1, 0, 2});
            if (ropeTheta > 0.0f) {
                q = builder.createRoPE(q, ropeTheta);
                k = builder.createRoPE(k, ropeTheta);
            }

            auto* scores = builder.createSlidingQueryKeyScore(q, k, window);
            auto* probs = builder.createSoftmax(scores, -1);
            auto* context = builder.createMatMul(probs, v);
            auto* contextSeqMajor = builder.createPermute(context, {1, 0, 2});
            contextFlat = builder.createReshape(contextSeqMajor, {seq, heads * headDim});
        }

        auto* out = builder.createMatMul(contextFlat, oWeightT);
        if (numResults == 1)
            return std::vector<Value*>{out};
        return std::vector<Value*>{out, k, v};
    });

    bl.add("attention", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap& attrs,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("attention", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 5)
            return make_error("attention expects operands (x, k, v, q_weight, o_weight)");

        auto headsResult = get_int_attr(attrs, "heads");
        if (!headsResult) return make_error(headsResult.error());
        int64_t heads = headsResult.take();

        auto kvHeadsResult = get_int_attr(attrs, "kv_heads");
        if (!kvHeadsResult) return make_error(kvHeadsResult.error());
        int64_t kvHeads = kvHeadsResult.take();

        auto headDimResult = get_int_attr(attrs, "head_dim");
        if (!headDimResult) return make_error(headDimResult.error());
        int64_t headDim = headDimResult.take();

        int64_t window = get_int_attr_or(attrs, "window", 0);
        float ropeTheta = get_float_attr_or(attrs, "rope_theta", 0.0f);
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0)
            return make_error("attention heads, kv_heads, and head_dim must be positive");
        if (heads % kvHeads != 0)
            return make_error("attention heads must be divisible by kv_heads");
        auto* x = operands[0];
        auto* k = operands[1];
        auto* v = operands[2];
        int rank = x->shape.rank();
        if (rank != 2 && rank != 3)
            return make_error("attention input must have rank 2 or rank 3");

        auto* qWeightT = builder.createTranspose(operands[3]);
        auto* oWeightT = builder.createTranspose(operands[4]);
        auto* qFlat = builder.createMatMul(x, qWeightT);

        Value* q = nullptr;
        Value* contextFlat = nullptr;

        if (rank == 3) {
            int64_t batch = x->shape.dim(0);
            int64_t seq = x->shape.dim(1);
            if (batch < 0 || seq < 0)
                return make_error("attention batch and sequence dimensions must be static");
            if (k->shape.rank() != 4 || v->shape.rank() != 4)
                return make_error("attention k and v must have rank 4 for batched input");

            q = builder.createPermute(
                builder.createReshape(qFlat, {batch, seq, heads, headDim}),
                {0, 2, 1, 3});
            if (ropeTheta > 0.0f)
                q = builder.createRoPE(q, ropeTheta);

            auto* scores = builder.createSlidingQueryKeyScore(q, k, window);
            auto* probs = builder.createSoftmax(scores, -1);
            auto* context = builder.createMatMul(probs, v);
            auto* contextSeqMajor = builder.createPermute(context, {0, 2, 1, 3});
            contextFlat = builder.createReshape(contextSeqMajor, {batch, seq, heads * headDim});
        } else {
            int64_t seq = x->shape.dim(0);
            if (seq < 0)
                return make_error("attention sequence dimension must be static");
            if (k->shape.rank() != 3 || v->shape.rank() != 3)
                return make_error("attention k and v must have rank 3 for unbatched input");

            q = builder.createPermute(
                builder.createReshape(qFlat, {seq, heads, headDim}),
                {1, 0, 2});
            if (ropeTheta > 0.0f)
                q = builder.createRoPE(q, ropeTheta);

            auto* scores = builder.createSlidingQueryKeyScore(q, k, window);
            auto* probs = builder.createSoftmax(scores, -1);
            auto* context = builder.createMatMul(probs, v);
            auto* contextSeqMajor = builder.createPermute(context, {1, 0, 2});
            contextFlat = builder.createReshape(contextSeqMajor, {seq, heads * headDim});
        }

        return std::vector<Value*>{builder.createMatMul(contextFlat, oWeightT)};
    });

    bl.add("embedding", [](Builder& builder,
                            const std::vector<Value*>& operands,
                            const AttrMap&,
                            int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("embedding", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        return std::vector<Value*>{builder.createEmbedding(operands[0], operands[1])};
    });

    bl.add("rope", [](Builder& builder,
                       const std::vector<Value*>& operands,
                       const AttrMap& attrs,
                       int numResults) -> Result<std::vector<Value*>> {
        auto resultCount = expect_num_results("rope", numResults, 1);
        if (!resultCount) return make_error(resultCount.error());
        if (operands.size() != 1)
            return make_error("rope expects one operand");
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
