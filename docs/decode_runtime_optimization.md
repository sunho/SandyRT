# Decode runtime optimization design

## Objective

Reduce decode-side CPU overhead in two stages:

1. Move decisions that depend only on KernelIR types and shapes out of the
   invocation path, beginning with layout aliasing and scratch placement.
2. Replace per-op engine dispatch with multi-op device execution units and
   record fixed-binding CUDA regions for replay.

The runtime now has the KernelIR layout contract, compile-time static scratch,
and synchronous multi-op device executables. The first CUDA Graph layer records
only command regions whose complete binding set has stable addresses.

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

## CUDA Graph fixed-binding implementation

Binding stability is generic device information; CUDA recordability is backend
policy. `DeviceExecutable` classifies values as fixed bindings when they are in
the executable-owned fixed scratch allocation, are static weight imports, or
are static aliases of another fixed binding. It exposes only
`inputsAndOutputsFixed(op)`. It does not mention CUDA Graphs or exclude CUDA
kernel kinds.

Each resolved `DeviceRunCommand` carries the generic `bindingsFixed` result. The
generic device runner batches commands and retains eager behavior as its default.
`CudaDevice` alone partitions that command stream into recordable and eager
regions. The initial CUDA policy excludes commands with non-fixed bindings and
CUDA implementations that perform host-dependent synchronization. In particular,
grouped `MoeMatMulKernel` is eager; ordinary `MatMulKernel` is recorded through
its normal cuBLAS calls. Current MoE gather/scatter and RoPE implementations are
also eager because they copy validation or routing state to the host and
synchronize the stream.

The CUDA stream and cuBLAS handle are initialized with `CudaDevice`, before any
capture. Creating the handle outside capture does not exclude cuBLAS work:
`cublasGemmEx` calls issued between `cudaStreamBeginCapture` and
`cudaStreamEndCapture` are recorded normally.

On the first invocation of a fixed region, CUDA begins stream capture, invokes
the existing per-kernel launch functions to record the region, ends capture,
instantiates `cudaGraphExec_t`, and launches it once for that invocation. There
is no preliminary eager execution of the same request. Later invocations with
the same binding signature call only `cudaGraphLaunch`. A signature contains
the ordered op IDs and every fixed buffer/view identity, including scratch and
weight buffer IDs, dtype, shape, strides, and storage offset.

Capture failure disables that exact region/signature and falls back to eager
execution without retrying every decode step. Per-kernel profiling also uses
the eager path because replay has no individual host launch boundary.

Fixed scratch is safe to capture because its allocation lives until the device
executable is destroyed. CUDA graph objects are destroyed before that scratch
buffer. Dynamic scratch remains allocated per invocation and any command that
touches it is eager. New request inputs and engine exports currently use fresh
allocations, so their boundary commands are eager; they can write into or read
from fixed scratch around a captured interior region. Stable engine input/export
slots are a later extension and will allow those boundary commands to join the
graph without kernel-node parameter updates.

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
10. Batch resolved kernel commands so CUDA can capture consecutive fixed-binding
    regions. Dynamic scratch and paged append split those regions but do not force
    unrelated fixed-scratch commands in the executable to remain eager.

The fixed scratch allocation and placement layout are private members of the
device executable. CUDA Graph objects are cached by the CUDA compiled graph and
destroyed before the executable releases fixed scratch. Dynamic scratch has no
extra compiled representation: runtime planning directly simulates the
executable's command list.
