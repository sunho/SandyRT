# Invocation Plan Refactor

## Goals

- Replace the current backend/program abstraction with a single `Engine` that owns device instances.
- Introduce `InvocPlan` as the reusable compiled invocation artifact.
- Make `InvocPlan` own compiled device programs directly. Do not add a separate `Executable`.
- Keep fusion as a MidIR-side decision. Invocation planning only handles buffer planning and execution order.
- Compile every executable MidIR op into one device program.
- Treat reshape/view-like ops as zero-copy aliases in the invocation planner.
- Refactor MidIR inputs from named inputs to positional inputs.
- Keep device placement out of MidIR for this step. All ops use default device `0`.

## Non-Goals

- Do not implement multi-device placement yet.
- Do not implement cross-device copies yet.
- Do not implement fusion in `InvocPlanner`.
- Do not make buffers carry tensor shape. Tensor shape comes from MidIR value metadata and instruction descriptors.
- Do not emit `StoreOutput`; outputs are explicit `InvocPlan::outputs`.

## Target Public API

`Engine` owns the devices and compiles graphs into `InvocPlan`.

```cpp
class Engine {
public:
    explicit Engine(std::vector<std::unique_ptr<Device>> devices);

    Result<std::unique_ptr<InvocPlan>> compile(const ir::mid_ir::Graph& graph);

    Result<std::vector<core::TensorBufferPtr>> run(
        const InvocPlan& plan,
        std::span<core::TensorBufferPtr const> inputs,
        const TensorMap& weights);

private:
    std::vector<std::unique_ptr<Device>> devices_;
};
```

`compile()` produces a reusable `InvocPlan`. `run()` interprets that plan with concrete input and weight buffers.

## Device API

`Device` is the only runtime/backend boundary.

```cpp
using DeviceBufferId = uint32_t;
using DeviceProgramId = uint32_t;

class Device {
public:
    virtual ~Device() = default;

    virtual Result<DeviceProgramId> compile(const ir::mid_ir::Op& op) = 0;

    virtual Result<DeviceBufferId> alloc(core::TensorDesc desc) = 0;
    virtual Result<void> dealloc(DeviceBufferId buffer) = 0;

    virtual Result<DeviceBufferId> load(core::TensorBuffer& src) = 0;

    virtual Result<void> run(
        DeviceProgramId program,
        std::span<const DeviceBufferId> inputs,
        std::span<const DeviceBufferId> outputs) = 0;

    virtual Result<core::TensorBufferPtr> read(DeviceBufferId src) = 0;
};
```

Notes:

- `load()` creates a device buffer from a host `TensorBuffer`.
- `alloc()` creates an empty output/intermediate device buffer from a tensor descriptor.
- `run()` receives already allocated output buffers.
- `read()` copies a device buffer back to a host `TensorBuffer`.
- There is no copy API yet because all planning uses device `0`.

## Invocation IR

Invocation IDs are logical plan IDs. They are not device handles.

```cpp
using InvocDeviceId = uint32_t;
using InvocValueId = uint32_t;
using InvocProgramId = uint32_t;
```

Instruction kinds:

```cpp
enum class InvocInstructionKind {
    Alloc,
    Dealloc,
    LoadInput,
    LoadWeight,
    RunKernel,
};
```

Instruction payloads:

```cpp
struct InvocAlloc {
    InvocDeviceId device;
    InvocValueId value;
    core::TensorDesc desc;
};

struct InvocDealloc {
    InvocDeviceId device;
    InvocValueId value;
};

struct InvocLoadInput {
    InvocDeviceId device;
    int64_t index;
    InvocValueId value;
};

struct InvocLoadWeight {
    InvocDeviceId device;
    std::string name;
    InvocValueId value;
};

struct InvocRunKernel {
    InvocDeviceId device;
    InvocProgramId program;
    std::vector<InvocValueId> inputs;
    std::vector<InvocValueId> outputs;
};
```

Represent the variant with whichever local style is preferred:

- `std::variant<...>` if acceptable.
- One struct with all fields and kind-specific helpers if that better matches the repo style.

Compiled programs live in `InvocPlan`.

```cpp
struct InvocProgram {
    InvocProgramId id;
    InvocDeviceId device;
    DeviceProgramId deviceProgram;
};

struct InvocPlan {
    std::vector<InvocProgram> programs;
    std::vector<InvocInstruction> instructions;
    std::vector<InvocValueId> outputs;
};
```

During planning, there is an internal draft program source:

```cpp
struct InvocProgramSource {
    InvocProgramId id;
    InvocDeviceId device;
    const ir::mid_ir::Op* op;
};
```

This source form should not be the final reusable artifact. `Engine::compile()` consumes it and stores compiled `DeviceProgramId`s in `InvocPlan`.

## MidIR Input Refactor

Change MidIR input ops from string names to numeric indices.

Current shape:

```cpp
builder.createInput("x", shape, dtype);
```

Target shape:

```cpp
builder.createInput(0, shape, dtype);
builder.createInput(1, shape, dtype);
```

MidIR `Input` attributes:

- remove or stop requiring `name`
- add required integer attr `index`

Builder API:

```cpp
Value* createInput(int64_t index, core::Shape shape, core::DType dtype);
```

Update all call sites:

- tests
- compiler materialization if it creates input ops
- CPU runner
- examples if they use input names directly

Weights remain named:

```cpp
Value* createWeight(const std::string& name, core::Shape shape, core::DType dtype);
```

## Planner Draft

`InvocPlanner` builds an internal uncompiled draft.

```cpp
class InvocPlanner {
public:
    explicit InvocPlanner(InvocDeviceId defaultDevice = 0);

    Result<InvocPlanDraft> plan(const ir::mid_ir::Graph& graph);

private:
    InvocDeviceId defaultDevice_;
};
```

Draft shape:

```cpp
struct InvocPlanDraft {
    std::vector<InvocProgramSource> programSources;
    std::vector<InvocInstruction> instructions;
    std::vector<InvocValueId> outputs;
};
```

The draft is an implementation detail used by `Engine::compile()`.

## Planner Algorithm

Walk `graph.entry()->ops` in order.

Maintain:

```cpp
std::unordered_map<const ir::mid_ir::Value*, InvocValueId> valueIds;
std::unordered_set<InvocValueId> materializedValues;
std::unordered_set<InvocValueId> outputValues;
```

Also compute last use for each logical `InvocValueId`.

### Input

For `Input(index=N)`:

1. Create a new `InvocValueId`.
2. Map the MidIR result value to that ID.
3. Emit:

```cpp
load_input(defaultDevice, N, value)
```

The loaded input buffer is a materialized value and should be deallocated after its last use unless it is also a graph output.

### Weight

For `Weight(name)`:

1. Create a new `InvocValueId`.
2. Map the MidIR result value to that ID.
3. Emit:

```cpp
load_weight(defaultDevice, name, value)
```

The loaded weight buffer is a materialized value and should be deallocated after its last use unless it is also a graph output.

### Reshape

For `Reshape`:

1. Lookup the operand value ID.
2. Map the reshape result to the same value ID.
3. Emit no `Alloc`.
4. Emit no `RunKernel`.
5. Emit no independent `Dealloc`.

This means reshape is a zero-copy alias at the invocation level.

The CPU device must reject reshape if asked to compile it, because reshape should not reach device programs.

### Compute Ops

For each non-input, non-weight, non-reshape executable op:

1. Lookup operand value IDs.
2. Create result value IDs.
3. For each result, emit:

```cpp
alloc(defaultDevice, resultValue, TensorDesc(result.shape, result.dtype))
```

4. Create one program source:

```cpp
programSources.push_back({programId, defaultDevice, op});
```

5. Emit:

```cpp
run_kernel(defaultDevice, programId, operandValues, resultValues)
```

Every executable MidIR op gets exactly one program. Any future fusion happens before this planner sees the graph.

### Outputs

After walking ops:

1. Map each `graph.outputs()` value through `valueIds`.
2. Store those IDs in `draft.outputs`.
3. Mark output IDs so planner does not emit normal dealloc before output read.

No `StoreOutput` instruction is emitted.

### Dealloc

Use last-use information to emit `dealloc` after a materialized value's last use.

Rules:

- Only dealloc values that own a real device buffer.
- Do not dealloc reshape aliases independently.
- Do not dealloc graph output values in the instruction stream before read.
- `Engine::run()` reads outputs after all instructions, then deallocs output buffers.

Simple first implementation:

- It is acceptable to emit all non-output deallocs at the end of the instruction stream.
- Later improvement can move deallocs immediately after last use.

## Engine Compile

`Engine::compile(graph)`:

1. Validate `devices_` is not empty.
2. Run `InvocPlanner(0).plan(graph)`.
3. Create an `InvocPlan`.
4. Move/copy draft instructions and outputs into the final plan.
5. For each `InvocProgramSource`:

```cpp
auto compiled = devices_[source.device]->compile(*source.op);
```

6. Append:

```cpp
InvocProgram{source.id, source.device, compiledDeviceProgram}
```

7. Return `std::unique_ptr<InvocPlan>`.

Compile should fail if:

- there are no devices
- a program references an invalid device id
- device compilation fails
- planner emits an invalid op, e.g. reshape as a program

## Engine Run

`Engine::run(plan, inputs, weights)` interprets `plan.instructions`.

Runtime state:

```cpp
std::unordered_map<InvocValueId, DeviceBufferId> buffers;
std::unordered_map<InvocProgramId, InvocProgram> programs;
```

Instruction handling:

### LoadInput

1. Validate device id.
2. Validate input index is in range.
3. Validate input pointer is non-null.
4. Call:

```cpp
devices_[device]->load(*inputs[index])
```

5. Store returned `DeviceBufferId` in `buffers[value]`.

### LoadWeight

1. Validate device id.
2. Lookup `weights[name]`.
3. Validate pointer is non-null.
4. Call:

```cpp
devices_[device]->load(*weight)
```

5. Store returned `DeviceBufferId` in `buffers[value]`.

### Alloc

1. Validate device id.
2. Call:

```cpp
devices_[device]->alloc(desc)
```

3. Store returned `DeviceBufferId` in `buffers[value]`.

### RunKernel

1. Lookup compiled `InvocProgram`.
2. Validate instruction device matches program device.
3. Translate all input/output `InvocValueId`s to `DeviceBufferId`s.
4. Call:

```cpp
devices_[device]->run(program.deviceProgram, inputBuffers, outputBuffers)
```

### Dealloc

1. Lookup `DeviceBufferId`.
2. Call:

```cpp
devices_[device]->dealloc(buffer)
```

3. Remove value from runtime buffer map.

### Output Read

After all instructions:

1. For each `plan.outputs` value:
   - lookup the buffer
   - determine its device
   - call `devices_[device]->read(buffer)`
   - append to output vector
2. Dealloc output buffers after successful read.
3. Return positional output vector.

For the first version, because all values use device `0`, output device lookup can be simple. If needed, add a runtime map:

```cpp
std::unordered_map<InvocValueId, InvocDeviceId> valueDevices;
```

## CPU Device

Replace `CpuInterpreterBackend` with `CpuDevice`.

`CpuDevice` owns:

- CPU device buffers
- CPU device programs

CPU buffer:

```cpp
struct CpuDeviceBuffer {
    core::TensorDesc desc;
    std::vector<uint8_t> data;
};
```

CPU program:

```cpp
struct CpuDeviceProgram {
    ir::mid_ir::OpKind kind;
    ir::mid_ir::AttrMap attrs;
    std::vector<core::TensorDesc> inputDescs;
    std::vector<core::TensorDesc> outputDescs;
};
```

The program should copy enough op metadata during `compile(op)` so it is not dependent on temporary op state beyond the graph lifetime.

CPU device behavior:

- `compile(op)` stores one op's metadata.
- `compile(Reshape)` returns error.
- `alloc(desc)` creates a zero-filled mutable CPU buffer.
- `load(tensor)` accesses the host tensor and copies bytes into a CPU buffer.
- `run(program, inputs, outputs)` dispatches to existing CPU eval logic for exactly one op.
- `read(buffer)` returns a host tensor buffer copy.

Reuse as much as possible from `CpuInterpreterBackend.cpp`:

- tensor ref helpers
- contiguous stride helpers
- attr helpers
- `eval_linear`
- `eval_relu`
- `eval_add`
- `eval_mul`
- `eval_sqrt`
- `eval_tanh`
- `eval_matmul`
- `eval_transpose`
- `eval_permute`
- `eval_sliding_query_key_score`
- `eval_softmax`
- `eval_embedding`
- `eval_rope`
- `eval_rms_norm`
- `eval_layer_norm`

The existing graph interpreter loop should go away. Execution is now instruction-by-instruction in `Engine`.

## File Layout

Suggested engine files:

- `src/engine/Device.h`
- `src/engine/InvocPlan.h`
- `src/engine/InvocPlanner.h`
- `src/engine/InvocPlanner.cpp`
- `src/engine/Engine.h`
- `src/engine/Engine.cpp`
- `src/engine/CpuDevice.h`
- `src/engine/CpuDevice.cpp`

Remove or retire:

- `src/engine/Backend.h`
- `src/engine/CpuInterpreterBackend.h`
- `src/engine/CpuInterpreterBackend.cpp`

Update:

- `src/engine/CMakeLists.txt`
- tests that include `CpuInterpreterBackend.h`

## Migration Steps

1. Add `Device.h` and `InvocPlan.h` with basic types.
2. Add `InvocPlanner` that emits plans for existing MidIR, initially still handling named input if needed.
3. Refactor MidIR input API to positional `Input(index)`.
4. Update compiler/tests/call sites for positional inputs.
5. Replace `Backend` API in `Engine` with device-owned API.
6. Implement `CpuDevice` by moving CPU backend logic.
7. Make `Engine::compile()` return `InvocPlan`.
8. Make `Engine::run()` interpret `InvocPlan`.
9. Delete or stop building old backend classes.
10. Tighten reshape behavior: planner aliases it, CPU device rejects it.
11. Add tests and port existing engine tests.

## Tests

Add or update tests for:

- MidIR builder creates `Input(index=0)` with integer attr.
- Planner emits `LoadInput(0, value)` for input 0.
- Planner emits `LoadWeight(name, value)` for weights.
- Planner emits one `InvocProgram` and one `RunKernel` per executable compute op.
- Planner emits no `Alloc`, no `InvocProgram`, and no `RunKernel` for reshape.
- Planner maps reshape output to the operand value ID.
- Planner exposes graph outputs through `InvocPlan::outputs`.
- Engine compile compiles programs through owned devices.
- Engine run interprets alloc/load/run/dealloc instructions.
- Engine run accepts positional input array and named weight map.
- Engine run returns positional output array.
- CPU device rejects reshape compile.
- Existing CPU numerical tests still pass through `Engine + CpuDevice`.

## First Version Constraints

- One device only: `devices_[0]`.
- All `InvocProgram::device` and instruction device ids are `0`.
- No cross-device copy instruction.
- No fusion.
- No delayed input loading optimization. Eager load at input/weight op order is fine.
- Dealloc at end is acceptable before implementing precise last-use dealloc.

## Step-By-Step Implementation Plan

### 1. Add invocation and device type shells

Add the new headers first without changing behavior:

- `src/engine/Device.h`
- `src/engine/InvocPlan.h`
- `src/engine/InvocPlanner.h`
- empty/minimal `src/engine/InvocPlanner.cpp`

Define:

- `DeviceBufferId`
- `DeviceProgramId`
- `InvocDeviceId`
- `InvocValueId`
- `InvocProgramId`
- `Device`
- `InvocInstruction`
- `InvocProgram`
- `InvocPlan`

Update `src/engine/CMakeLists.txt` so these files build, but keep the old backend path compiling during this step.

Verification:

- project still builds
- no runtime behavior changes

### 2. Refactor MidIR input ops to positional indices

Change `Builder::createInput` from:

```cpp
Value* createInput(const std::string& name, core::Shape shape, core::DType dtype);
```

to:

```cpp
Value* createInput(int64_t index, core::Shape shape, core::DType dtype);
```

Update MidIR input attrs:

- remove input name requirement
- add integer `index`

Update all call sites and tests that create input ops.

For compiler/materializer code that currently receives named inputs, assign stable indices at the graph boundary. Keep that mapping local to materialization or compiler code; MidIR itself should only see `index`.

Verification:

- `ir` tests pass
- compiler materialization tests pass
- no invocation behavior introduced yet

### 3. Implement `InvocPlanner` without lifetime optimization

Implement `InvocPlanner::plan(graph)` returning an internal draft:

```cpp
struct InvocPlanDraft {
    std::vector<InvocProgramSource> programSources;
    std::vector<InvocInstruction> instructions;
    std::vector<InvocValueId> outputs;
};
```

Start with simple allocation/lifetime behavior:

- emit `LoadInput` at input op position
- emit `LoadWeight` at weight op position
- alias `Reshape`
- emit `Alloc` and `RunKernel` for every executable compute op
- emit non-output `Dealloc` instructions at the end

Do not compile device programs in the planner.

Verification:

- add `InvocPlannerTest`
- assert instruction order for a simple `Input -> Linear -> Output`
- assert `Reshape` emits no program and no run instruction
- assert every non-view compute op gets one program source

### 4. Add a fake/test device

Before moving the CPU backend, add a small fake device in tests to validate `Engine` orchestration.

The fake device should:

- record compile calls
- record alloc/load/run/dealloc/read calls
- return deterministic fake ids
- produce dummy host buffers from `read`

This keeps engine tests independent from CPU numerical execution during the refactor.

Verification:

- fake device unit tests pass
- no CPU backend changes yet

### 5. Refactor `Engine` to own devices and compile `InvocPlan`

Change `Engine` constructor to:

```cpp
explicit Engine(std::vector<std::unique_ptr<Device>> devices);
```

Change compile API to:

```cpp
Result<std::unique_ptr<InvocPlan>> compile(const ir::mid_ir::Graph& graph);
```

Implementation:

1. validate device list is non-empty
2. call `InvocPlanner(0).plan(graph)`
3. compile each draft program source through `devices_[source.device]`
4. store compiled `DeviceProgramId` in `InvocPlan::programs`
5. move draft instructions and outputs into `InvocPlan`

At this step, either keep old `Engine::create_plan/run` temporarily for compatibility or update all call sites at once.

Verification:

- fake device confirms one compile call per executable op
- `InvocPlan` owns compiled program ids
- no `Executable` type exists

### 6. Implement `Engine::run(const InvocPlan&, ...)`

Add:

```cpp
Result<std::vector<core::TensorBufferPtr>> run(
    const InvocPlan& plan,
    std::span<core::TensorBufferPtr const> inputs,
    const TensorMap& weights);
```

Interpret instructions with:

```cpp
std::unordered_map<InvocValueId, DeviceBufferId> buffers;
std::unordered_map<InvocValueId, InvocDeviceId> valueDevices;
```

Runtime behavior:

- `LoadInput`: validate index and call `device.load`
- `LoadWeight`: lookup weight by name and call `device.load`
- `Alloc`: call `device.alloc`
- `RunKernel`: lookup compiled program and buffer ids, then call `device.run`
- `Dealloc`: call `device.dealloc` and erase runtime state
- after instructions, read `plan.outputs` into a vector
- dealloc output buffers after successful reads

Verification:

- fake device confirms exact instruction interpretation
- missing input index returns an error
- missing weight returns an error
- invalid program id returns an error

### 7. Create `CpuDevice` skeleton

Add:

- `src/engine/CpuDevice.h`
- `src/engine/CpuDevice.cpp`

Implement only structural behavior first:

- program table
- buffer table
- `compile(op)` copies op kind, attrs, input descs, and output descs
- `alloc(desc)` creates a CPU buffer
- `dealloc(id)` releases or marks buffer slot empty
- `load(tensor)` copies host tensor bytes
- `read(id)` returns a host tensor copy
- `run(...)` returns "unsupported" initially

Verification:

- `CpuDevice` buffer lifecycle tests pass
- `compile(Reshape)` returns error

### 8. Move CPU op execution into `CpuDevice`

Move reusable helpers from `CpuInterpreterBackend.cpp` into `CpuDevice.cpp`.

Adapt execution from graph-level maps to direct program inputs/outputs:

- program inputs are positional device buffer ids
- program outputs are positional preallocated device buffer ids
- op attrs are stored in the CPU program
- output descs come from the compiled op metadata

For each supported op, dispatch through existing tensor calc functions.

Important:

- do not execute `Input`
- do not execute `Weight`
- do not execute `Reshape`
- `run()` should validate input/output arity against the compiled program

Verification:

- add direct `CpuDevice` tests for at least `Linear`, `Add`, and one unary op
- verify CPU results match previous backend tests

### 9. Port engine integration tests to `CpuDevice`

Update tests that use:

```cpp
CpuInterpreterBackend
Engine::create_plan
BackendRunResult
```

to use:

```cpp
CpuDevice
Engine::compile
InvocPlan
std::vector<core::TensorBufferPtr>
```

Update input setup from keyed input map to positional input vector/span.

Weights remain `TensorMap`.

Verification:

- existing numerical engine tests pass through `Engine + CpuDevice`
- output order matches `graph.outputs()`

### 10. Remove old backend abstraction

Once tests are ported:

- remove `Backend.h`
- remove `CpuInterpreterBackend.h`
- remove `CpuInterpreterBackend.cpp`
- remove `Plan`/`SimplePlan` from `Engine`
- remove old `BackendBufferMap`/`BackendRunResult` APIs
- update includes and CMake

Verification:

- `rg "Backend|CpuInterpreterBackend|create_plan|BackendRunResult"` only finds intentional historical references, if any
- full test suite builds and runs

### 11. Tighten planner lifetime handling

After the basic execution path works, improve dealloc placement.

Compute last use of materialized `InvocValueId`s and emit `Dealloc` immediately after last use when possible.

Rules:

- output values are not deallocated in the instruction stream
- aliases do not dealloc independently
- values reused by aliases must remain alive until the aliased output/consumer last use

Verification:

- planner tests assert dealloc appears after last consumer
- reshape alias test verifies the source buffer is not deallocated before the alias output is read

### 12. Final cleanup and consistency pass

Clean up naming and docs:

- make all public names use `InvocPlan`, not `Executable`
- keep `InvocPlan` as the compiled artifact
- ensure comments say fusion is MidIR-side
- ensure reshape handling is documented in planner and CPU device tests

Verification:

- full test suite
- `rg "Executable"` returns no implementation references
- `rg "StoreOutput"` returns no implementation references
- `rg "createInput\\(\""` finds no string-based MidIR input creation
