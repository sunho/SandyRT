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

While compiling a `DeviceExecutable`, the device simulates the lifetimes of its
internal KernelIR values using compile-time tensor descriptors. Eligible dense
results whose shapes are fully static are placed by the device scratch allocator.
The executable allocates one fixed backing buffer and owns its layout and views.
Destroying the executable releases that buffer. Concurrent execution of one
executable is outside the current synchronous scope.

### Dynamic scratch

Values with runtime-dependent shapes are excluded from the fixed plan. During
`Device::runExecutable`, the device directly simulates its command list using
concrete tensor descriptors, allocates one dynamic backing buffer, and releases
it before synchronous execution returns.

Values exported from an executable remain outside scratch planning. The engine
allocates standalone buffers for them and tracks their cross-node lifetime.
Exported aliases recursively export their backing values so no exported view can
refer to private executable scratch.

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

## Synchronous device-executable implementation plan

The first implementation deliberately excludes asynchronous submission,
events, queues, concurrent use of one executable, and executable-instance
pools. A compiled executable is run synchronously and owns exactly one fixed
scratch allocation.

1. Add an opaque multi-op `DeviceExecutable`, an executable description with
   ordered operation IDs, and runtime import/export bindings.
2. Partition KernelIR into consecutive same-device executable nodes. Inputs,
   tensor-tuple construction, and cross-device transfers remain engine nodes.
   Paged append is included in the device executable.
3. Mark values consumed outside their producer executable, graph outputs, and
   values consumed by engine nodes as exports. Mark values produced outside an
   executable as imports. Paged tensors mutated by paged append are mutable
   imports.
4. Allocate exported dense tensors as standalone engine-owned buffers. Internal
   values are never allocated by the engine.
5. During device-executable compilation, plan all fully static internal dense
   values with the device scratch allocator, allocate one fixed scratch buffer,
   and retain it until the executable is destroyed.
6. During device-executable execution, plan only the remaining dynamic internal
   values using concrete invocation descriptors. Allocate that dynamic scratch
   for the synchronous run and release it before returning.
7. Execute alias-mode layout operations as device-side view-binding commands.
   Materialize-mode layout operations remain kernels. An exported alias also
   exports its backing value so it cannot refer to private scratch.
8. Execute paged append as a device command. The paged tensor remains externally
   owned mutable state; the command updates its device metadata before later
   commands consume it.
9. Switch compiled engine programs to execute device nodes rather than dispatch
   individual kernels. Preserve the legacy per-op path only for manually built
   test programs during migration.
10. Once eager multi-op parity is established, add CUDA Graph capture inside the
    CUDA executable for eligible fixed-scratch command sequences. An executable
    containing dynamic scratch or paged append initially remains eager.

The fixed scratch allocation, its placement layout, and future CUDA Graph
objects are private members of the device executable. Destroying the executable
releases them. Dynamic scratch has no extra compiled representation: runtime
planning directly simulates the executable's command list.
