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
using DeviceId = uint32_t;

static constexpr OpId kInvalidOpId = std::numeric_limits<OpId>::max();

enum class ValueKind {
    Tensor,
    PagedTensor,
    TensorTuple,
    Scalar,
};

struct PagedTensorMeta {
    int64_t growDim = -1;
    int64_t pageSize = -1;
};

struct ValueType {
    ValueKind kind = ValueKind::Tensor;
    core::DType dtype = core::DType::F32;
    core::Shape shape;
    PagedTensorMeta paged;
    std::vector<ValueType> elements;
};

struct Use {
    OpId op = kInvalidOpId;
    uint32_t operand = 0;
};

enum class ValueLayout {
    Contiguous,
    Strided,
};

struct Def {
    OpId op = kInvalidOpId;
    uint32_t result = 0;
};

struct Value {
    ValueId id = 0;
    ValueType type;
    ValueLayout layout = ValueLayout::Contiguous;
    DeviceId device = 0;

    Def def;
    std::vector<Use> uses;

    std::string debugName;
};

enum class OpKind {
    Input,
    TensorTupleCreate,
    DeviceTransfer,
    PagedAppend,
    LayoutTransform,
    ElementwiseKernel,
    ReductionKernel,
    LinearKernel,
    MatMulKernel,
    GatherKernel,
    SoftmaxKernel,
    TopKKernel,
    NormKernel,
    RoPEKernel,
    SlidingQueryKeyScoreKernel,
    AttentionKernel,
    MoeGatherKernel,
    MoeMatMulKernel,
    MoeScatterSumKernel,
};

const char* op_kind_name(OpKind kind);

class Graph;

class Op {
public:
    Op(OpId id, OpKind kind, DeviceId device = 0)
        : id_(id), kind_(kind), device_(device) {}
    virtual ~Op() = default;

    OpId id() const { return id_; }
    OpKind kind() const { return kind_; }
    DeviceId device() const { return device_; }

    virtual std::span<const ValueId> inputs() const = 0;
    virtual std::span<const ValueId> outputs() const = 0;

    virtual const char* name() const = 0;
    virtual bool has_side_effects() const { return false; }
    virtual Result<void> verify(const Graph& graph) const = 0;

private:
    OpId id_;
    OpKind kind_;
    DeviceId device_ = 0;
};

class Graph {
public:
    ValueId addValue(ValueType type, std::string debugName = "", DeviceId device = 0);
    ValueId addValue(ValueType type, DeviceId device);

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
    int64_t tupleElement = -1;
};

class InputOp final : public Op {
public:
    InputOp(OpId id, InputSource source, ValueId output, DeviceId device = 0);

    const InputSource& source() const { return source_; }

    std::span<const ValueId> inputs() const override { return {}; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "input"; }
    Result<void> verify(const Graph& graph) const override;

private:
    InputSource source_;
    std::array<ValueId, 1> outputs_;
};

class TensorTupleCreateOp final : public Op {
public:
    TensorTupleCreateOp(
        OpId id,
        std::vector<ValueId> inputs,
        ValueId output,
        DeviceId device = 0);

    std::span<const ValueId> inputs() const override {
        return {inputs_.data(), inputs_.size()};
    }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "tensor_tuple_create"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::vector<ValueId> inputs_;
    std::array<ValueId, 1> outputs_;
};

class DeviceTransferOp final : public Op {
public:
    DeviceTransferOp(
        OpId id,
        DeviceId sourceDevice,
        DeviceId targetDevice,
        ValueId input,
        ValueId output);

    DeviceId sourceDevice() const { return sourceDevice_; }
    DeviceId targetDevice() const { return targetDevice_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "device_transfer"; }
    Result<void> verify(const Graph& graph) const override;

private:
    DeviceId sourceDevice_ = 0;
    DeviceId targetDevice_ = 0;
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
};

class PagedAppendOp final : public Op {
public:
    PagedAppendOp(
        OpId id,
        ValueId cache,
        ValueId chunk,
        DeviceId device = 0);

    ValueId cache() const { return inputs_[0]; }
    ValueId chunk() const { return inputs_[1]; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return {}; }

    const char* name() const override { return "paged_append"; }
    bool has_side_effects() const override { return true; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 2> inputs_;
};

enum class LayoutTransformKind {
    Reshape,
    Transpose,
    Permute,
    Slice,
    Contiguous,
};

enum class LayoutTransformMode {
    Alias,
    Materialize,
};

class LayoutTransformOp final : public Op {
public:
    LayoutTransformOp(
        OpId id,
        LayoutTransformKind transform,
        ValueId input,
        ValueId output,
        std::vector<int64_t> dims,
        LayoutTransformMode mode = LayoutTransformMode::Materialize,
        DeviceId device = 0);
    LayoutTransformOp(
        OpId id,
        LayoutTransformKind transform,
        ValueId input,
        ValueId output,
        std::vector<int64_t> dims,
        std::vector<int64_t> indices,
        LayoutTransformMode mode = LayoutTransformMode::Materialize,
        DeviceId device = 0);

    LayoutTransformKind transform() const { return transform_; }
    const std::vector<int64_t>& dims() const { return dims_; }
    const std::vector<int64_t>& indices() const { return indices_; }
    LayoutTransformMode mode() const { return mode_; }
    bool aliasesInput() const { return mode_ == LayoutTransformMode::Alias; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "layout_transform"; }
    Result<void> verify(const Graph& graph) const override;

private:
    LayoutTransformKind transform_;
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    std::vector<int64_t> dims_;
    std::vector<int64_t> indices_;
    LayoutTransformMode mode_ = LayoutTransformMode::Materialize;
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

class ElementwiseKernelOp final : public Op {
public:
    ElementwiseKernelOp(
        OpId id,
        std::vector<ElementwiseInput> elementwiseInputs,
        ValueId output,
        ScalarId result,
        std::vector<ScalarNode> scalars,
        DeviceId device = 0);

    const std::vector<ElementwiseInput>& elementwiseInputs() const {
        return elementwiseInputs_;
    }
    ValueId output() const { return output_; }
    ScalarId result() const { return result_; }
    const std::vector<ScalarNode>& scalars() const { return scalars_; }

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "elementwise_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::vector<ElementwiseInput> elementwiseInputs_;
    std::vector<ValueId> inputs_;
    std::array<ValueId, 1> outputs_;
    ValueId output_;
    ScalarId result_;
    std::vector<ScalarNode> scalars_;
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
        bool keepDims,
        DeviceId device = 0);

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

class LinearKernelOp final : public Op {
public:
    LinearKernelOp(
        OpId id,
        ValueId input,
        ValueId weight,
        ValueId bias,
        ValueId output,
        DeviceId device = 0);

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "linear_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 3> inputs_;
    std::array<ValueId, 1> outputs_;
};

class MatMulKernelOp final : public Op {
public:
    MatMulKernelOp(
        OpId id,
        ValueId lhs,
        ValueId rhs,
        ValueId output,
        bool transposeLhs,
        bool transposeRhs,
        DeviceId device = 0);

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
    GatherKernelOp(
        OpId id,
        ValueId ids,
        ValueId table,
        ValueId output,
        DeviceId device = 0);

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
    SoftmaxKernelOp(
        OpId id,
        ValueId input,
        ValueId output,
        int64_t axis,
        DeviceId device = 0);

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

class TopKKernelOp final : public Op {
public:
    TopKKernelOp(
        OpId id,
        ValueId input,
        ValueId values,
        ValueId indices,
        int64_t k,
        int64_t axis,
        DeviceId device = 0);

    int64_t k() const { return k_; }
    int64_t axis() const { return axis_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "topk_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 2> outputs_;
    int64_t k_ = 0;
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
        double epsilon,
        DeviceId device = 0);

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
        bool splitHalf,
        DeviceId device = 0);
    RoPEKernelOp(
        OpId id,
        std::vector<ValueId> inputs,
        ValueId output,
        double theta,
        int64_t rotaryDim,
        bool splitHalf,
        DeviceId device = 0);

    double theta() const { return theta_; }
    int64_t rotaryDim() const { return rotaryDim_; }
    bool splitHalf() const { return splitHalf_; }

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "rope_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::vector<ValueId> inputs_;
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
        double scale,
        DeviceId device = 0);
    SlidingQueryKeyScoreKernelOp(
        OpId id,
        ValueId query,
        ValueId key,
        ValueId positionIds,
        ValueId output,
        int64_t window,
        double scale,
        DeviceId device = 0);

    int64_t window() const { return window_; }
    double scale() const { return scale_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "sliding_query_key_score_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::vector<ValueId> inputs_;
    std::array<ValueId, 1> outputs_;
    int64_t window_ = 0;
    double scale_ = -1.0;
};

class AttentionKernelOp final : public Op {
public:
    AttentionKernelOp(
        OpId id,
        ValueId query,
        ValueId key,
        ValueId value,
        ValueId output,
        int64_t window,
        double scale,
        DeviceId device = 0);
    AttentionKernelOp(
        OpId id,
        ValueId query,
        ValueId key,
        ValueId value,
        ValueId positionOffsets,
        ValueId output,
        int64_t window,
        double scale,
        DeviceId device = 0);

    int64_t window() const { return window_; }
    double scale() const { return scale_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "attention_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::vector<ValueId> inputs_;
    std::array<ValueId, 1> outputs_;
    int64_t window_ = 0;
    double scale_ = -1.0;
};

class MoeGatherKernelOp final : public Op {
public:
    MoeGatherKernelOp(
        OpId id,
        ValueId input,
        ValueId topkIds,
        ValueId topkWeights,
        ValueId packedInput,
        ValueId packedWeights,
        ValueId tokenIds,
        ValueId expertOffsets,
        int64_t numExperts,
        int64_t topK,
        DeviceId device = 0);

    int64_t numExperts() const { return numExperts_; }
    int64_t topK() const { return topK_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "moe_gather_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 3> inputs_;
    std::array<ValueId, 4> outputs_;
    int64_t numExperts_ = 0;
    int64_t topK_ = 0;
};

class MoeMatMulKernelOp final : public Op {
public:
    MoeMatMulKernelOp(
        OpId id,
        ValueId input,
        ValueId expertOffsets,
        ValueId weight,
        ValueId output,
        bool transposeRhs,
        DeviceId device = 0);

    bool transposeRhs() const { return transposeRhs_; }

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "moe_matmul_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 3> inputs_;
    std::array<ValueId, 1> outputs_;
    bool transposeRhs_ = false;
};

class MoeScatterSumKernelOp final : public Op {
public:
    MoeScatterSumKernelOp(
        OpId id,
        ValueId packedOutput,
        ValueId packedWeights,
        ValueId tokenIds,
        ValueId reference,
        ValueId output,
        DeviceId device = 0);

    std::span<const ValueId> inputs() const override { return inputs_; }
    std::span<const ValueId> outputs() const override { return outputs_; }

    const char* name() const override { return "moe_scatter_sum_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 4> inputs_;
    std::array<ValueId, 1> outputs_;
};

} // namespace sandy::ir::kernel_ir
