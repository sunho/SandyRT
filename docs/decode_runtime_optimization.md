# Decode runtime optimization design

## Objective

Reduce decode-side CPU overhead in two stages:

1. Move decisions that depend only on KernelIR types and shapes out of the
   invocation path, beginning with layout aliasing and scratch placement.
2. Replace per-op engine dispatch with multi-op device execution units that can
   eventually be recorded and replayed as CUDA Graphs.

This change implements only the KernelIR layout contract and compile-time
static scratch pool. CUDA Graph capture and the execution-unit refactor remain
follow-up work.

## KernelIR layout contract

MidIR remains layout-agnostic. During MidIR-to-KernelIR lowering, each KernelIR
tensor value is classified as `Contiguous` or `Strided`, and each layout op is
assigned an explicit `Alias` or `Materialize` mode.

- Kernel outputs and graph inputs are contiguous by default.
- Transpose, permute, and slice alias their input and conservatively produce a
  strided value.
- Reshape aliases a contiguous input. A reshape of a strided input is compiled
  as a materializing layout kernel and produces a contiguous value.
- An aliasing reshape/contiguous op is rejected by KernelIR verification unless
  its input and output satisfy the contiguous contract.

The engine no longer decides at invocation time whether a layout op aliases or
materializes. It follows the recorded KernelIR mode. Constructing the concrete
view still happens while binding runtime buffers because strides, offsets, and
buffer handles are runtime objects. The remaining default-view check is contract
validation; failure is an error rather than a fallback to a kernel launch.

During engine execution, alias-mode layout ops are handled as view bindings
without a device kernel launch. Materialize-mode layout ops execute normally.
Backends retain their layout implementations so direct device-level execution
and tests remain supported.

## Static and dynamic scratch

Scratch now has two independent plans and backing pools.

### Static scratch

At `Engine::compile`, the scratch planner simulates KernelIR lifetimes using the
compile-time tensor descriptors. Eligible dense tensor results whose shapes are
fully static are placed by the device scratch allocator. Only values actually
placed by a device allocator are marked static.

The resulting layout is stored on the compiled program. The engine eagerly
creates one backing allocation and manages it as a per-program lease pool. A
concurrent invocation gets a distinct lease; a completed invocation returns its
allocation for reuse. This keeps fixed scratch allocation and placement out of
the steady-state invocation path without introducing cross-request buffer races.

### Dynamic scratch

Values with runtime-dependent shapes are excluded from the static plan. The
existing runtime descriptor inference and lifetime simulation plan only these
remaining values. They are instantiated from a separate dynamic scratch pool
for the invocation and released through the existing runtime cleanup path.

Graph outputs and alias chains remain outside scratch planning because their
backing-buffer lifetimes are not represented by the current value-level lifetime
model. Alias-aware lifetime tracking can relax that restriction later.

## Multi-op execution and CUDA Graph direction

CUDA Graph capture requires an engine abstraction above one-op-at-a-time
`Device::run`. The intended next layer is a compiled device execution unit:

1. Partition each device's KernelIR schedule into consecutive execution units.
   Engine-owned operations such as cross-device transfers or paged-cache
   mutation are explicit boundaries.
2. Compile each unit with a binding schema describing its external inputs,
   outputs, static scratch slots, dynamic scratch slots, and alias/view updates.
3. Execute the unit through one device call. An eager unit loops over its kernels
   internally; a capturable unit records or replays a CUDA Graph.

Kernel compilation should emit execution metadata rather than leaving capture
eligibility to the engine. Kernels initially default to eager. Elementwise
kernels are the first capture candidates when the shared
`inputs_and_outputs_fixed` analysis proves that all bindings required by the
unit have stable shapes and compatible storage. This metadata can later expand
to matmul and other kernel families.

A captured graph executable must be tied to the scratch lease whose device
pointers it recorded. External input/output changes can be handled either by
updating CUDA kernel-node parameters before replay or by binding stable staging
slots; the former is the preferred first implementation because it avoids
copies. The capture cache key must include the compiled unit, concrete dynamic
shape signature, and scratch-pool slot. Cache misses execute/capture once;
subsequent decode steps replay the graph.

The engine should ultimately orchestrate execution units and value ownership,
not inspect individual kernel kinds. KernelIR compilation decides alias versus
materialize and eager versus capturable; the device backend owns launches,
capture, graph-node parameter updates, and replay.

## Follow-up sequence

1. Introduce `CompiledExecutionUnit` and a batched device run interface while
   preserving eager behavior.
2. Move alias-view binding records into the compiled unit's binding program so
   the main engine loop no longer handles layout kinds.
3. Add kernel execution metadata and `inputs_and_outputs_fixed` analysis.
4. Capture the first fixed elementwise unit and cache graph executables per
   scratch lease and invocation shape.
5. Expand capture eligibility and benchmark 1024-token prefill plus 128-token
   decode after a warm-up request.
