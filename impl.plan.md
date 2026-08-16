# KernelIR Direct Execution Plan

## Context

The current engine path compiles and executes MidIR through an `InvocPlan`.

Current shape:

```text
MidIR Graph
  -> InvocPlanner
  -> InvocPlan
  -> Engine interprets InvocPlan
  -> Device compiles/runs individual MidIR ops
```

Target shape:

```text
MidIR Graph
  -> KernelIR Graph
  -> Engine compiles KernelIR Graph
  -> Engine executes compiled KernelIR Graph directly
  -> Device runs compiled KernelIR op by op
```

The major architectural change is removing the invocation IR. Allocation, deallocation, input loading, output collection, runtime shape resolution, and view handling become direct responsibilities of the engine runtime executor.

KernelIR is a tensor graph, but it is not another high-level mathematical IR. KernelIR ops represent executable kernel boundaries or runtime boundary/view nodes. Fusion has already been decided before or during MidIR-to-KernelIR lowering.

For this first implementation, keep lowering simple:

```text
one MidIR op -> one KernelIR op
```

No KernelIR fusion yet.

## Goals

- Add a new KernelIR abstraction.
- Represent KernelIR as a graph of logical tensor values and concrete op classes.
- Make KernelIR inputs explicit through `InputOp`.
- Represent graph outputs through `Graph::outputs`.
- Keep KernelIR logical only: no strides, no byte sizes, no physical offsets, no device placement.
- Lower MidIR to KernelIR with the simple one-MidIR-op-to-one-KernelIR-op rule.
- Remove `InvocPlan` and `InvocPlanner`.
- Make `Engine` execute compiled KernelIR graphs directly.
- Make temporary buffers deallocate immediately after their last use.
- Change device compilation from per-op MidIR compilation to whole-KernelIR-graph compilation.
- Change device execution to run one compiled KernelIR op by op id.
- Make `CpuDevice` emulate KernelIR.
- Keep MidIR emulation available for debugging.

## Non-goals

- No KernelIR fusion in this pass.
- No explicit device placement in KernelIR.
- No multi-device execution.
- No cross-device copies.
- No TensorState or KV-cache state primitive yet.
- No symbolic stride algebra.
- No KernelIR physical layout model.
- No paged tensor state.
- No copy-on-write state handling.

## Design rule

KernelIR owns logical execution structure.

Engine runtime owns physical execution state.

Device owns backend-specific compiled representation and low-level execution.

```text
KernelIR:
  values, ops, input sources, output values, logical shapes/dtypes

Engine runtime:
  input/weight binding, allocation, views, strides, actual runtime shapes,
  last-use tracking, deallocation, output reads

Device:
  compile KernelIR graph, run compiled op id with concrete buffers
```

## KernelIR abstraction

Create:

```text
src/ir/KernelIR.h
src/ir/KernelIR.cpp
src/ir/MidIRToKernelIR.h
src/ir/MidIRToKernelIR.cpp
```

Update:

```text
src/ir/CMakeLists.txt
```

### IDs

```cpp
namespace sandy::ir::kernel_ir {

using ValueId = uint32_t;
using OpId = uint32_t;

static constexpr OpId kInvalidOpId = UINT32_MAX;

}
```

### Value type

KernelIR values are logical tensor or scalar values. They are not physical buffers.

```cpp
enum class ValueKind {
    Tensor,
    Scalar,
};

struct ValueType {
    ValueKind kind = ValueKind::Tensor;
    core::DType dtype = core::DType::F32;

    // Only meaningful for tensor values.
    // May contain core::Shape::kDynamic == -1.
    core::Shape shape;
};
```

Dynamic dims are allowed in KernelIR. KernelIR does not resolve actual runtime size.

```text
tensor<-1x-1x768xf32>
```

means:

```text
rank = 3
dtype = f32
dim 2 must be 768
dims 0 and 1 are runtime-known
```

### Value

```cpp
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
```

All tensor inputs, weights, intermediate tensors, constants materialized as tensors, and final tensors are `Value`s.

Graph outputs are explicitly stored in `Graph::outputs`.

### Op kind

KernelIR should only contain these ops for now:

```cpp
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
```

No `TensorState`.

No `Output` op. Outputs are a graph field.

### Op base class

Use concrete classes because op payloads are complex. Avoid a single large `std::variant` payload.

```cpp
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
```

This gives each op class proper typed fields while still allowing generic graph traversal through `inputs()` and `outputs()`.

### Graph

```cpp
class Graph {
public:
    ValueId addValue(ValueType type, std::string debugName = "");

    template <class OpT, class... Args>
    OpT* addOp(Args&&... args);

    const Value& value(ValueId id) const;
    Value& value(ValueId id);

    const std::vector<std::unique_ptr<Op>>& ops() const;

    const std::vector<ValueId>& outputs() const;
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
```

`Graph::addOp` must update:

- each output value's `def`
- each input value's `uses`

The graph should preserve topological op order. MidIR lowering can emit ops in MidIR order.

## Input op

Inputs are explicit KernelIR ops.

Examples:

```text
%x = Input(0)
%y = Input(1)
%w = Weight("layer.weight")
```

Use one `InputOp` with a typed source.

```cpp
enum class InputSourceKind {
    Argument,
    Weight,
    External,
};

struct InputSource {
    InputSourceKind kind = InputSourceKind::Argument;

    // For Argument.
    int64_t index = -1;

    // For Weight / External.
    std::string name;
};

class InputOp final : public Op {
public:
    InputOp(OpId id, InputSource source, ValueId output);

    const InputSource& source() const { return source_; }

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override;

    const char* name() const override { return "input"; }
    Result<void> verify(const Graph& graph) const override;

private:
    InputSource source_;
    std::array<ValueId, 1> outputs_;
};
```

Lowering:

- MidIR `Input(index)` -> KernelIR `InputOp { Argument, index }`
- MidIR `Weight(name)` -> KernelIR `InputOp { Weight, name }`

These ops are not compiled as executable kernels. Engine handles them during runtime binding.

## LayoutTransform op

KernelIR does not store strides. Layout transform records logical intent.

```cpp
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

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override;

    const char* name() const override { return "layout_transform"; }
    Result<void> verify(const Graph& graph) const override;

private:
    LayoutTransformKind transform_;
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    std::vector<int64_t> dims_;
};
```

Engine runtime may implement this as:

- a zero-copy view, when possible
- a materializing kernel/copy, when a backend requires contiguous data

For the first implementation, CPU can either:

- use view metadata internally, or
- materialize into a new buffer using existing `core::reshape`, `core::transpose`, and `core::permute`

Choose the simpler implementation first.

## ElementwiseKernel op

Elementwise fusion is represented inside a single op by scalar nodes.

```cpp
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

    // For Load.
    uint32_t inputIndex = 0;

    // For Constant.
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

    const std::vector<ElementwiseInput>& elementwiseInputs() const;
    ValueId iterationValue() const;
    const std::vector<ScalarNode>& scalars() const;
    const std::vector<ElementwiseStore>& stores() const;

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
```

Example scalar body:

```text
out = max(add(sub(x, y), z), 0)

v0 = load input0
v1 = load input1
v2 = sub v0, v1
v3 = load input2
v4 = add v2, v3
v5 = constant 0
v6 = max v4, v5
store output0, v6
```

For first lowering:

- MidIR `Add` -> one `ElementwiseKernelOp` with `Add`
- MidIR `Mul` -> one `ElementwiseKernelOp` with `Mul`
- MidIR `ReLU` -> one `ElementwiseKernelOp` with `ReLU`
- MidIR `Sqrt` -> one `ElementwiseKernelOp` with `Sqrt`
- MidIR `Tanh` -> one `ElementwiseKernelOp` with `Tanh`
- MidIR `Constant` -> one `ElementwiseKernelOp` that fills output with a scalar constant

Broadcasting is marked on inputs using `BroadcastMode::RightAligned`. Actual broadcast indexing is resolved by the device/runtime from concrete shapes.

## ReductionKernel op

```cpp
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

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override;

    const char* name() const override { return "reduction_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    ReduceOp reduce_;
    std::array<ValueId, 1> inputs_;
    std::array<ValueId, 1> outputs_;
    std::vector<int64_t> axes_;
    bool keepDims_ = false;
};
```

Current MidIR does not expose generic reduction ops, but this is needed for KernelIR completeness and future lowering.

## MatMulKernel op

```cpp
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

    std::span<const ValueId> inputs() const override;
    std::span<const ValueId> outputs() const override;

    const char* name() const override { return "matmul_kernel"; }
    Result<void> verify(const Graph& graph) const override;

private:
    std::array<ValueId, 2> inputs_;
    std::array<ValueId, 1> outputs_;
    bool transposeLhs_ = false;
    bool transposeRhs_ = false;
};
```

Lowering:

- MidIR `MatMul` -> `MatMulKernelOp`
- MidIR `Linear` -> `MatMulKernelOp` plus `ElementwiseKernelOp` for bias add, or a `CustomKernelOp` named `linear` if preserving old CPU behavior is easier initially

Prefer the two-op lowering if tests remain manageable:

```text
linear(x, w, b)
  -> matmul_kernel(x, w, transpose_rhs=true)
  -> elementwise_kernel(add matmul_result, b)
```

This technically breaks the one-MidIR-op-to-one-KernelIR-op rule for `Linear`, so if strict one-to-one is required, use:

```text
Linear -> CustomKernelOp("linear")
```

For this implementation plan, use `CustomKernelOp("linear")` first to preserve the stated rule.

## Remaining concrete ops

Implement each as a concrete class.

### GatherKernelOp

For MidIR `Embedding`.

```text
inputs: ids, table
outputs: output
```

### SoftmaxKernelOp

For MidIR `Softmax`.

```text
inputs: x
outputs: output
fields: axis
```

### NormKernelOp

For MidIR `RMSNorm` and `LayerNorm`.

```cpp
enum class NormKind {
    RMSNorm,
    LayerNorm,
};
```

```text
RMSNorm inputs: x, optional weight
LayerNorm inputs: x, weight, bias
outputs: output
fields: kind, epsilon
```

### RoPEKernelOp

For MidIR `RoPE`.

```text
inputs: x
outputs: output
fields: theta, rotary_dim, split_half
```

### SlidingQueryKeyScoreKernelOp

For MidIR `SlidingQueryKeyScore`.

```text
inputs: q, k
outputs: output
fields: window, scale
```

### CustomKernelOp

Escape hatch for one-to-one lowering and backend-specific kernels.

```cpp
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
```

Use `CustomKernelOp` initially for MidIR ops whose existing CPU implementation is easier to preserve directly.

Good initial custom names:

```text
linear
```

Avoid overusing custom ops once direct KernelIR classes exist.

## MidIR to KernelIR lowering

Create:

```cpp
namespace sandy::ir::kernel_ir {

class MidIRToKernelIRLowering {
public:
    Result<std::unique_ptr<Graph>> lower(const mid_ir::Graph& graph);
};

}
```

### Lowering state

```cpp
std::unordered_map<const mid_ir::Value*, kernel_ir::ValueId> valueMap;
```

For every MidIR result, create a KernelIR value with:

```cpp
ValueType {
    ValueKind::Tensor,
    midValue->dtype,
    midValue->shape
}
```

### Lowering rules

| MidIR op | KernelIR op |
|---|---|
| `Input` | `InputOp { Argument(index) }` |
| `Weight` | `InputOp { Weight(name) }` |
| `Constant` | `ElementwiseKernelOp` constant fill |
| `Add` | `ElementwiseKernelOp` |
| `Mul` | `ElementwiseKernelOp` |
| `ReLU` | `ElementwiseKernelOp` |
| `Sqrt` | `ElementwiseKernelOp` |
| `Tanh` | `ElementwiseKernelOp` |
| `MatMul` | `MatMulKernelOp` |
| `Linear` | `CustomKernelOp("linear")` initially |
| `Transpose` | `LayoutTransformOp(Transpose)` |
| `Reshape` | `LayoutTransformOp(Reshape)` |
| `Permute` | `LayoutTransformOp(Permute)` |
| `SlidingQueryKeyScore` | `SlidingQueryKeyScoreKernelOp` |
| `Softmax` | `SoftmaxKernelOp` |
| `Embedding` | `GatherKernelOp` |
| `RoPE` | `RoPEKernelOp` |
| `RMSNorm` | `NormKernelOp(RMSNorm)` |
| `LayerNorm` | `NormKernelOp(LayerNorm)` |

After lowering all ops:

```cpp
kernelGraph->setOutputs(mapped MidIR graph.outputs());
```

### Broadcasting

For binary elementwise ops:

```text
lhs.broadcast = RightAligned if lhs.shape != output.shape
rhs.broadcast = RightAligned if rhs.shape != output.shape
```

It is also acceptable to mark both inputs `RightAligned`; runtime broadcast verification will accept equal shapes.

KernelIR does not compute broadcast strides.

### Constants

MidIR `Constant` creates a tensor result. Lower it as an elementwise fill:

```text
constant fill over output shape
```

Scalar node:

```text
v0 = constant attr["value"]
store output, v0
```

For rank-0 tensors, iteration shape is scalar. CPU runtime must handle `numel == 1`.

## Remove InvocPlan

Delete or stop using:

```text
src/engine/InvocPlan.h
src/engine/InvocPlanner.h
src/engine/InvocPlanner.cpp
test/engine/InvocPlannerTest.cpp
```

Update:

```text
src/engine/CMakeLists.txt
test/engine/CMakeLists.txt
```

Remove references to:

```text
InvocPlan
InvocPlanDraft
InvocProgram
InvocInstruction
InvocPlanner
InvocValueId
InvocProgramId
```

Replace with direct KernelIR compiled graph/runtime executor types.

## New Engine API

`Engine::compile` should lower MidIR to KernelIR and compile the KernelIR graph.

```cpp
class Engine {
public:
    explicit Engine(std::vector<std::unique_ptr<Device>> devices);

    Result<std::unique_ptr<CompiledKernelGraph>> compile(
        const ir::mid_ir::Graph& graph);

    Result<std::vector<core::TensorBufferPtr>> run(
        const CompiledKernelGraph& compiled,
        std::span<core::TensorBufferPtr const> inputs,
        const TensorMap& weights);

private:
    std::vector<std::unique_ptr<Device>> devices_;
};
```

`CompiledKernelGraph` owns the lowered KernelIR graph and the device-side compiled handle.

```cpp
struct CompiledKernelGraph {
    std::unique_ptr<ir::kernel_ir::Graph> graph;
    DeviceCompiledGraphId deviceGraph = 0;
};
```

For now, use device `0` only.

## Device API

Change the device compile model from individual MidIR op compilation to whole KernelIR graph compilation.

```cpp
using DeviceBufferId = uint32_t;
using DeviceCompiledGraphId = uint32_t;

struct DeviceBufferView {
    DeviceBufferId buffer = 0;
    core::TensorDesc desc;

    // Runtime-resolved physical layout.
    // KernelIR does not contain this.
    std::vector<int64_t> strides;
    int64_t offset = 0;
};

class Device {
public:
    virtual ~Device() = default;

    virtual Result<DeviceCompiledGraphId> compile(
        const ir::kernel_ir::Graph& graph) = 0;

    virtual Result<DeviceBufferId> alloc(core::TensorDesc desc) = 0;
    virtual Result<void> dealloc(DeviceBufferId buffer) = 0;

    virtual Result<DeviceBufferId> load(core::TensorBuffer& src) = 0;

    virtual Result<void> run(
        DeviceCompiledGraphId graph,
        ir::kernel_ir::OpId op,
        std::span<const DeviceBufferView> inputs,
        std::span<const DeviceBufferView> outputs) = 0;

    virtual Result<core::TensorBufferPtr> read(DeviceBufferView src) = 0;
};
```

The device receives the compiled graph handle plus the KernelIR op id. This lets the device compile/cache whatever it wants for the whole graph while still executing one op at a time under engine control.

For CPU, `compile(graph)` may initially only store or clone metadata and return a handle. `run(handle, opId, ...)` dispatches to a KernelIR interpreter.

## Runtime executor without InvocPlan

Engine runtime walks `compiled.graph->ops()` in order.

Runtime state:

```cpp
struct RuntimeValue {
    DeviceBufferId buffer = 0;
    core::TensorDesc desc;

    std::vector<int64_t> strides;
    int64_t offset = 0;

    bool ownsBuffer = true;
    bool isView = false;
    DeviceBufferId baseBuffer = 0;
};

using RuntimeValueMap =
    std::unordered_map<ir::kernel_ir::ValueId, RuntimeValue>;
```

### Execution flow

For each op:

```text
InputOp:
  Argument -> load inputs[index]
  Weight   -> load weights[name]
  External -> error for now unless external bindings are added

LayoutTransformOp:
  create view or materialize output

Executable kernel op:
  allocate output buffers
  build DeviceBufferView arrays for inputs and outputs
  device.run(compiled.deviceGraph, op.id(), inputViews, outputViews)

After each op:
  decrement remaining use counts for input values
  deallocate any non-output temporary whose remaining use count becomes zero
```

### Use-count based deallocation

Before execution, compute:

```cpp
std::unordered_map<ValueId, int64_t> remainingUses;
std::unordered_set<ValueId> graphOutputs;
```

Initialize:

```text
remainingUses[value] = value.uses.size()
graphOutputs = graph.outputs()
```

After an op executes, for each input value:

```text
remainingUses[input]--
if remainingUses[input] == 0 and input is not graph output:
    dealloc runtime buffer if owned
```

Important cases:

- Do not deallocate graph outputs before reading them.
- Do not deallocate values backed by borrowed external memory unless the device contract says `load()` owns the loaded device buffer.
- If a value is a view, dealloc only the owner/base buffer when all aliases are dead.

For first implementation, it is acceptable to materialize layout transforms instead of aliasing views. That makes ownership simple:

```text
every runtime value owns exactly one device buffer
```

Then add view aliasing later.

### Output collection

After all ops run:

```text
for value in graph.outputs:
  read RuntimeValue as TensorBufferPtr
```

Then deallocate remaining owned device buffers, including outputs after read.

## Runtime shape and allocation

KernelIR values may contain dynamic dims. Allocation must use actual runtime descs.

For first implementation, output shape resolution can mostly use KernelIR logical value types because existing MidIR inference often produces static shapes. Where `-1` appears, resolve conservatively from inputs.

Recommended helpers:

```cpp
Result<core::TensorDesc> resolve_actual_desc(
    const ir::kernel_ir::Graph& graph,
    const ir::kernel_ir::Op& op,
    ir::kernel_ir::ValueId output,
    const RuntimeValueMap& values);
```

Initial rules:

- If output logical shape has no dynamic dims, use it directly.
- For elementwise outputs with dynamic dims, use `iterationValue` actual shape.
- For layout transforms, compute actual output shape from actual input shape and transform attrs.
- For matmul outputs, compute actual output shape from actual input shapes and transpose flags.
- For softmax/norm/rope, output actual shape equals input actual shape.
- For gather/embedding, compute from ids/table actual shapes.
- For custom `linear`, compute from input and weight actual shapes.

### Contiguous strides

KernelIR has no strides.

Engine/device runtime computes contiguous strides when allocating or loading:

```cpp
std::vector<int64_t> contiguous_strides(core::Shape actualShape);
```

For this first implementation, layout transforms can materialize contiguous outputs. That avoids needing view strides immediately.

## CPU device changes

Current `CpuDevice` compiles and runs MidIR ops. Change it to compile and run KernelIR.

### Compile

```cpp
Result<DeviceCompiledGraphId> CpuDevice::compile(
    const ir::kernel_ir::Graph& graph);
```

Initial CPU compile behavior:

- Store a reference-safe copy or metadata snapshot of the KernelIR graph.
- Return a graph handle.
- No actual code generation required.

### Run

```cpp
Result<void> CpuDevice::run(
    DeviceCompiledGraphId graphId,
    ir::kernel_ir::OpId opId,
    std::span<const DeviceBufferView> inputs,
    std::span<const DeviceBufferView> outputs);
```

CPU run behavior:

- Look up compiled graph by `graphId`.
- Find op by `opId`.
- Dispatch on `kernel_ir::OpKind`.
- Convert `DeviceBufferView` to `core::TensorRef` / `core::MutableTensorRef`.
- Execute using existing `core::` tensor kernels where possible.

### CPU KernelIR emulation mapping

| KernelIR op | CPU implementation |
|---|---|
| `ElementwiseKernel` | scalar interpreter over output numel with runtime broadcasting |
| `MatMulKernel` | `core::matmul` |
| `CustomKernel("linear")` | `core::linear` |
| `LayoutTransform` | `core::reshape`, `core::transpose`, `core::permute` if materialized |
| `SoftmaxKernel` | `core::softmax` |
| `GatherKernel` | `core::embedding` |
| `RoPEKernel` | `core::rope` |
| `NormKernel(RMSNorm)` | `core::rms_norm` |
| `NormKernel(LayerNorm)` | `core::layer_norm` |
| `SlidingQueryKeyScoreKernel` | `core::sliding_query_key_score` |

### Elementwise CPU interpreter

Implement enough scalar ops for current MidIR:

```text
Load
Constant
Add
Mul
Sqrt
Tanh
ReLU
```

Add `Sub`, `Div`, `Max`, etc. as straightforward follow-ups if already declared.

Broadcasting:

- Use actual input/output shapes.
- Apply right-aligned broadcasting.
- Compute input element index from output multidimensional index.
- For broadcasted dimensions, use index `0`.

No KernelIR strides are needed.

## Keep MidIR emulation for debugging

Do not delete old MidIR execution logic outright if it is useful for debugging.

Move or preserve MidIR op emulation behind a clearly named debug helper, for example:

```text
src/engine/debug/MidIRInterpreter.h
src/engine/debug/MidIRInterpreter.cpp
```

or:

```text
src/engine/MidIRDebugInterpreter.h
src/engine/MidIRDebugInterpreter.cpp
```

The production `CpuDevice` should run KernelIR, but tests/debug tools can still compare:

```text
MidIR debug interpreter result
vs
KernelIR CPU result
```

This is useful while validating lowering.

## Test plan

### KernelIR graph tests

Add:

```text
test/ir/KernelIRTest.cpp
```

Cover:

- create input values and `InputOp`
- set graph outputs
- use-def chains update correctly
- verifier catches missing defs
- verifier catches invalid output ids
- layout transform op stores dims
- elementwise op stores scalar body and broadcast modes

### MidIR to KernelIR lowering tests

Add:

```text
test/ir/MidIRToKernelIRTest.cpp
```

Cover:

- `Input(index)` lowers to `InputOp(Argument(index))`
- `Weight(name)` lowers to `InputOp(Weight(name))`
- `Add` lowers to `ElementwiseKernelOp`
- broadcast add marks broadcast intent
- `ReLU`, `Sqrt`, `Tanh`, `Mul` lower to elementwise kernels
- `MatMul` lowers to `MatMulKernelOp`
- `Linear` lowers to `CustomKernelOp("linear")`
- `Reshape`, `Transpose`, `Permute` lower to `LayoutTransformOp`
- graph outputs are preserved

### Engine tests

Update:

```text
test/engine/EngineCompileTest.cpp
test/engine/EngineTest.cpp
test/engine/CpuDeviceTest.cpp
```

Remove or replace:

```text
test/engine/InvocPlannerTest.cpp
```

Cover:

- `Engine::compile(MidIR)` produces `CompiledKernelGraph`
- CPU device receives one compiled KernelIR graph
- engine loads inputs and weights
- engine allocates outputs/intermediates
- engine runs ops in graph order
- temporaries are deallocated after last use
- graph outputs are read correctly
- existing numerical tests still pass

### Deallocation tests

Add a fake device test that records:

```text
load
alloc
run
read
dealloc
```

Verify:

```text
x = Input(0)
w = Weight("w")
y = Add(x, w)
z = Mul(y, y)
output z
```

Expected behavior:

- `y` is deallocated after `Mul`
- `x` and `w` are deallocated after their last use
- `z` is read before deallocation

## Suggested implementation sequence

1. Add `KernelIR.h/.cpp` with `Graph`, `Value`, `Op`, `InputOp`, and skeleton concrete op classes.
2. Add KernelIR graph verifier and dump support.
3. Add `MidIRToKernelIR.h/.cpp`.
4. Implement lowering for `Input`, `Weight`, and simple unary/binary elementwise ops.
5. Add KernelIR and lowering unit tests.
6. Change `Device.h` to use `kernel_ir::Graph`, `DeviceCompiledGraphId`, and `run(graphHandle, opId, views...)`.
7. Add `CompiledKernelGraph` type to engine.
8. Rewrite `Engine::compile` to lower MidIR to KernelIR and call `device.compile(kernelGraph)`. (do rest of mid ir to kernel ir impl here)
9. Rewrite `Engine::run` to execute KernelIR directly.
10. Implement use-count deallocation in `Engine::run`.
11. Port `CpuDevice` to emulate KernelIR.
12. Preserve old MidIR CPU emulation in a debug interpreter file/helper.
13. Remove `InvocPlan` and `InvocPlanner` from build files.
14. Update engine/device tests.
15. Run the full test suite.

## Migration notes

Expect these files to change substantially:

```text
src/engine/Device.h
src/engine/Engine.h
src/engine/Engine.cpp
src/engine/CpuDevice.h
src/engine/CpuDevice.cpp
src/engine/CMakeLists.txt
src/ir/CMakeLists.txt
test/engine/EngineCompileTest.cpp
test/engine/CpuDeviceTest.cpp
test/engine/EngineTest.cpp
test/CMakeLists.txt
```

Expect these files to be removed or retired:

```text
src/engine/InvocPlan.h
src/engine/InvocPlanner.h
src/engine/InvocPlanner.cpp
test/engine/InvocPlannerTest.cpp
```

Do not remove MidIR itself. MidIR remains the semantic model graph and source of type inference.

Do not remove MidIR CPU emulation if it is useful for debugging. Move it out of production `CpuDevice`.

## Final prompt

okay let's not add device for now. give the final implementation plan for:
1. create the abstraction for the kernel ir
2. implement mid ir to kernel ir lowering. keep it simple. follow one mid ir op = one kernel ir op rule.
3. remove invoc plan. and implement the kernel ir running directly from the engine. you must deallocate the temporary buffer as soon as it's not needed by following the use graph. for cpu side remove the compile thingy; change it to compile (kernel ir graph) -> (compiled kernel ir graph handle) then when we run it we run it by device.run(compiled ir graph handle, op id, input buffers, outputs buffers) cpu device side will now emulate the kernel ir instead. (keep mid ir emulation somewhere tho we might use that for debugging)

write this to impl.plan.md and give detailed context. keep this final prompt as the last section tho.

finally do the do the shape calculation at all. rely on mid ir shapes and use them as they are.
