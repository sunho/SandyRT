#pragma once

#include "MidIR.h"
#include "Result.h"
#include "Tensor.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sandy::ir::kernel_ir {

using ValueId = uint32_t;
using OpId = uint32_t;

static constexpr OpId kInvalidOpId = std::numeric_limits<OpId>::max();

enum class ValueKind {
    Tensor,
    Scalar,
};

struct ValueType {
    ValueKind kind = ValueKind::Tensor;
    core::DType dtype = core::DType::F32;
    core::Shape shape;
};

struct Use {
    OpId op = kInvalidOpId;
    uint32_t operand = 0;
};

struct Def {
    OpId op = kInvalidOpId;
    uint32_t result = 0;
};

struct Value {
    ValueId id = 0;
    ValueType type;

    Def def;
    std::vector<Use> uses;

    std::string debugName;
};

enum class OpKind {
    Input,
    LayoutTransform,
    ElementwiseKernel,
    ReductionKernel,
    MatMulKernel,
    GatherKernel,
    SoftmaxKernel,
    NormKernel,
    RoPEKernel,
    SlidingQueryKeyScoreKernel,
    CustomKernel,
};

const char* op_kind_name(OpKind kind);

class Graph;

class Op {
public:
    Op(OpId id, OpKind kind) : id_(id), kind_(kind) {}
    virtual ~Op() = default;

    OpId id() const { return id_; }
    OpKind kind() const { return kind_; }

    virtual std::span<const ValueId> inputs() const = 0;
    virtual std::span<const ValueId> outputs() const = 0;

    virtual const char* name() const = 0;
    virtual Result<void> verify(const Graph& graph) const = 0;

private:
    OpId id_;
    OpKind kind_;
};

class Graph {
public:
    ValueId addValue(ValueType type, std::string debugName = "");

    template <class OpT, class... Args>
    OpT* addOp(Args&&... args) {
        auto id = nextOpId_++;
        auto op = std::make_unique<OpT>(id, std::forward<Args>(args)...);
        auto* raw = op.get();

        auto inputs = raw->inputs();
        for (uint32_t i = 0; i < inputs.size(); ++i) {
            if (hasValue(inputs[i])) {
                value(inputs[i]).uses.push_back(Use{id, i});
            }
        }

        auto outputs = raw->outputs();
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            if (hasValue(outputs[i])) {
                value(outputs[i]).def = Def{id, i};
            }
        }

        ops_.push_back(std::move(op));
        return raw;
    }

    bool hasValue(ValueId id) const;
    bool hasOp(OpId id) const;

    const Value& value(ValueId id) const;
    Value& value(ValueId id);

    const Op& op(OpId id) const;
    Op& op(OpId id);

    const std::vector<Value>& values() const { return values_; }
    const std::vector<std::unique_ptr<Op>>& ops() const { return ops_; }

    const std::vector<ValueId>& outputs() const { return outputs_; }
    void setOutputs(std::vector<ValueId> outputs);

    Result<void> verify() const;
    void dump() const;

private:
    std::vector<Value> values_;
    std::vector<std::unique_ptr<Op>> ops_;
    std::vector<ValueId> outputs_;

    ValueId nextValueId_ = 0;
    OpId nextOpId_ = 0;
};

enum class InputSourceKind {
    Argument,
    Weight,
    External,
};

struct InputSource {
    InputSourceKind kind = InputSourceKind::Argument;
    int64_t index = -1;
    std::string name;
};

class InputOp final : public Op {
public:
    InputOp(OpId id, InputSource source, ValueId output);

    const InputSource& source() const { return source_; }

    std::span<const ValueId> inputs() const override { return {}; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "input"; }
    Result<void> verify(const Graph& graph) const override;

private:
    InputSource source_;
    std::array<ValueId, 1> outputs_;
};

enum class LayoutTransformKind {
    Reshape,
    Transpose,
    Permute,
    Contiguous,
};

class LayoutTransformOp final : public Op {
public:
    LayoutTransformOp(
        OpId id,
        LayoutTransformKind transform,
        ValueId input,
        ValueId output,
        std::vector<int64_t> dims);

    LayoutTransformKind transform() const { return transform_; }
    const std::vector<int64_t>& dims() const { return dims_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "layout_transform"; }
    Result<void> verify(const Graph& graph) const override;

private:
    LayoutTransformKind transform_;
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    std::vector<int64_t> dims_;
};

enum class BroadcastMode {
    None,
    RightAligned,
};

enum class ScalarOp {
    Load,
    Constant,

    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,

    Neg,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Tanh,
    ReLU,

    Cast,
};

using ScalarId = uint32_t;

struct ElementwiseInput {
    ValueId value = 0;
    BroadcastMode broadcast = BroadcastMode::None;
};

struct ScalarNode {
    ScalarId id = 0;
    ScalarOp op = ScalarOp::Constant;
    core::DType dtype = core::DType::F32;

    uint32_t inputIndex = 0;
    double constant = 0.0;

    std::vector<ScalarId> operands;
};

struct ElementwiseStore {
    ValueId output = 0;
    ScalarId value = 0;
};

class ElementwiseKernelOp final : public Op {
public:
    ElementwiseKernelOp(
        OpId id,
        std::vector<ElementwiseInput> elementwiseInputs,
        std::vector<ValueId> outputs,
        ValueId iterationValue,
        std::vector<ScalarNode> scalars,
        std::vector<ElementwiseStore> stores);

    const std::vector<ElementwiseInput>& elementwiseInputs() const {
        return elementwiseInputs_;
    }
    ValueId iterationValue() const { return iterationValue_; }
    const std::vector<ScalarNode>& scalars() const { return scalars_; }
    const std::vector<ElementwiseStore>& stores() const { return stores_; }

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override;

    const char* name() const override { return "elementwise_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::vector<ElementwiseInput> elementwiseInputs_;
    std::vector<ValueId> inputs_;
    std::vector<ValueId> outputs_;
    ValueId iterationValue_;
    std::vector<ScalarNode> scalars_;
    std::vector<ElementwiseStore> stores_;
};

enum class ReduceOp {
    Sum,
    Max,
    Min,
    Prod,
    Mean,
};

class ReductionKernelOp final : public Op {
public:
    ReductionKernelOp(
        OpId id,
        ReduceOp reduce,
        ValueId input,
        ValueId output,
        std::vector<int64_t> axes,
        bool keepDims);

    ReduceOp reduce() const { return reduce_; }
    const std::vector<int64_t>& axes() const { return axes_; }
    bool keepDims() const { return keepDims_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "reduction_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    ReduceOp reduce_;
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    std::vector<int64_t> axes_;
    bool keepDims_ = false;
};

class MatMulKernelOp final : public Op {
public:
    MatMulKernelOp(
        OpId id,
        ValueId lhs,
        ValueId rhs,
        ValueId output,
        bool transposeLhs,
        bool transposeRhs);

    bool transposeLhs() const { return transposeLhs_; }
    bool transposeRhs() const { return transposeRhs_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "matmul_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 2> inputs_;
    std::array<ValueId, 1> outputs_;
    bool transposeLhs_ = false;
    bool transposeRhs_ = false;
};

class GatherKernelOp final : public Op {
public:
    GatherKernelOp(OpId id, ValueId ids, ValueId table, ValueId output);

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "gather_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 2> inputs_;
    std::array<ValueId, 1> outputs_;
};

class SoftmaxKernelOp final : public Op {
public:
    SoftmaxKernelOp(OpId id, ValueId input, ValueId output, int64_t axis);

    int64_t axis() const { return axis_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "softmax_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    int64_t axis_ = -1;
};

enum class NormKind {
    RMSNorm,
    LayerNorm,
};

class NormKernelOp final : public Op {
public:
    NormKernelOp(
        OpId id,
        NormKind norm,
        std::vector<ValueId> inputs,
        ValueId output,
        double epsilon);

    NormKind norm() const { return norm_; }
    double epsilon() const { return epsilon_; }

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "norm_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    NormKind norm_;
    std::vector<ValueId> inputs_;
    std::array<ValueId, 1> outputs_;
    double epsilon_ = 0.0;
};

class RoPEKernelOp final : public Op {
public:
    RoPEKernelOp(
        OpId id,
        ValueId input,
        ValueId output,
        double theta,
        int64_t rotaryDim,
        bool splitHalf);

    double theta() const { return theta_; }
    int64_t rotaryDim() const { return rotaryDim_; }
    bool splitHalf() const { return splitHalf_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "rope_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    double theta_ = 10000.0;
    int64_t rotaryDim_ = -1;
    bool splitHalf_ = false;
};

class SlidingQueryKeyScoreKernelOp final : public Op {
public:
    SlidingQueryKeyScoreKernelOp(
        OpId id,
        ValueId query,
        ValueId key,
        ValueId output,
        int64_t window,
        double scale);

    int64_t window() const { return window_; }
    double scale() const { return scale_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "sliding_query_key_score_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 2> inputs_;
    std::array<ValueId, 1> outputs_;
    int64_t window_ = 0;
    double scale_ = -1.0;
};

class CustomKernelOp final : public Op {
public:
    CustomKernelOp(
        OpId id,
        std::string customName,
        std::vector<ValueId> inputs,
        std::vector<ValueId> outputs,
        mid_ir::AttrMap attrs);

    const std::string& customName() const { return customName_; }
    const mid_ir::AttrMap& attrs() const { return attrs_; }

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override;

    const char* name() const override { return "custom_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::string customName_;
    std::vector<ValueId> inputs_;
    std::vector<ValueId> outputs_;
    mid_ir::AttrMap attrs_;
};

} // namespace sandy::ir::kernel_ir
