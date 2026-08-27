# CUDA JIT Elementwise and Invocation Cache Plan

## Goal

Replace the CUDA elementwise scalar interpreter with a cached, straight-line
NVRTC evaluator while preserving the existing runtime tensor loader and storer.
At the same time, cache runtime descriptor inference and scratch placement by
compiled program plus exact input signatures, including Paged-KV length.

Every implementation step below starts by reading this document and ends in a
separate commit. Completed steps are marked in their commit.

## Invariants

- Build and benchmark with `-O2` and `CMAKE_CUDA_ARCHITECTURES=native`.
- A compiled program has a stable monotonic handle; global caches do not use raw
  pointers as identities.
- Tensor signatures include kind, dtype, every concrete dimension, tuple order,
  and paged grow dimension/page size.
- Scratch cache entries contain immutable descriptor/placement calculations,
  never invocation-owned device buffer handles.
- Scratch and JIT caches use the same canonical key serialization and hashing
  implementation, with distinct domain tags.
- Hand-written JIT CUDA lives in real `.cu`/`.cuh` files for syntax highlighting
  and is embedded into the binary at build time.
- The initial JIT specializes only scalar evaluation. Runtime load/store policies
  preserve contiguous, strided, broadcast, BF16/F32, and paged behavior.
- The current interpreter remains available as an explicit fallback until JIT
  parity is established.

## Step 1: Canonical cache keys and program handles

Status: complete

- Add a shared `CacheKey`/`CacheKeyBuilder` implementation with stable byte
  encoding, domain separation, hash combining, shapes, dtypes, and paged tensor
  metadata.
- Assign each `CompiledKernelGraph` a monotonic `CompiledProgramId`.
- Build invocation signatures in KernelIR input order, including tuple elements.
- Add unit tests for equality, shape/dtype changes, Paged-KV length changes,
  domain separation, and program separation.

## Step 2: Split and cache scratch layouts

Status: complete

- Separate deterministic scratch placement from device-buffer instantiation.
- Cache inferred `RuntimeTensorDescs` plus immutable `RuntimeScratchLayout` by the
  full invocation signature and device/scratch ABI.
- Instantiate fresh buffers/views from a cached layout for each invocation.
- Make cache lookup and same-key computation thread-safe.
- Add hit/miss tests, including a miss when Paged-KV length changes.

## Step 3: Generic CUDA JIT and highlighted embedded sources

Status: pending

- Add reusable NVRTC compiler, module/function RAII, diagnostics, architecture
  selection, and thread-safe in-flight compile caching.
- Link `CUDA::nvrtc` and `CUDA::cuda_driver`.
- Store ABI, tensor access, policy template, and kernel entry point as checked-in
  `.cuh`/`.cu` files.
- Embed those source files during the build and supply them to NVRTC as virtual
  headers with meaningful diagnostic names.
- Include template/header content hashes, architecture, compiler options, and JIT
  ABI in keys produced by the shared cache-key builder.

## Step 4: Straight-line elementwise evaluator JIT

Status: pending

- Emit one `GeneratedElementwiseEvaluator::eval` body per scalar DAG.
- Instantiate the highlighted policy kernel as
  `ElementwiseKernel<RuntimeLoader, GeneratedEvaluator, RuntimeStorer>`.
- Compile/cache the evaluator and retain its loaded function in the compiled CUDA
  operation.
- Launch it on the existing CUDA stream, with interpreter fallback on configured
  JIT failure.
- Do not include runtime pointers in cache keys.

## Step 5: Correctness, cache, and failure coverage

Status: pending

- Test every scalar operation and fused chains.
- Test F32/BF16, constants, broadcast, strided tensors, paged tensors, refreshed
  paged metadata, and zero-sized output.
- Test JIT reuse, ABI/template invalidation, concurrent same-key lookup, useful
  NVRTC diagnostics, and interpreter/JIT parity.
- Run the complete CUDA device test suite and document any unrelated failure.

## Step 6: Warmed benchmark and profile comparison

Status: pending

- Rebuild the server with `-O2` and native CUDA architecture.
- Warm up with a 1024-token prefill.
- Measure 1024-token prefill plus 128-token decode with server profiling and
  Nsight Systems.
- Compare elementwise GPU time, scratch planning, descriptor inference, wall
  times, throughput, compilation time, hit/miss counts, and module count against
  the `cudaMemcpy2DAsync` baseline.
- Record results in this document and commit the completed analysis.

## Baseline

- Prefill wall: 231.671 ms; elementwise GPU: 89.628 ms.
- Decode wall: 6005.908 ms; elementwise GPU: 244.306 ms.
- Prefill scratch planning: 2.992 ms.
- Decode scratch planning: 224.690 ms total, 1.769 ms/evaluation.
- Decode descriptor inference: 112.557 ms total, 0.886 ms/evaluation.
