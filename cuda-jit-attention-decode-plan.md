# CUDA JIT Decode Attention Plan

## Goal

JIT only the decode attention path and change its consecutive-key traversal to
resolve paged K/V pointers once per physical page rather than once per scalar
element. Keep prefill behavior unchanged. Split decode and prefill attention
sources first so later work can evolve and benchmark them independently.

Every implementation step starts by rereading this document and ends in a
separate commit. Status and validation results are recorded before committing.

## Invariants

- Build with `Release`, `-O2 -DNDEBUG`, and
  `CMAKE_CUDA_ARCHITECTURES=native`.
- The existing compiled `CudaAttentionKernels.cu` remains intact as the fallback.
- JIT decode and any future JIT prefill use separate highlighted `.cu` source
  templates inside `cuda/jit/templates`. This task adds and compiles only the
  decode JIT source; the prefill source remains a separate future module.
- The compiled decode implementation remains available behind an explicit
  fallback flag until JIT parity is established.
- Dtype and tensor access kind are compile-time policies. Runtime pointers,
  batch size, current KV length, page count, split count, and scratch sizes do
  not enter generated source and therefore do not fragment the kernel cache.
- The complete effective CUDA compilation unit remains the CUDA kernel cache
  key. Runtime scratch-plan keys remain separate and continue to include the
  dynamic paged-KV shape.
- Paged decode iterates consecutive keys page-by-page. Each warp loads the K and
  V page-table entries uniformly in all lanes once for each page segment it
  visits, then advances typed row pointers within that page. It does not use an
  explicit warp shuffle to broadcast page pointers.
- The page-major JIT fast path is selected only when K and V are both paged and
  have compatible page size, shift, and grow dimension. Contiguous, mixed, or
  incompatible paging layouts retain the compiled decode fallback in this
  iteration.
- Page size/shift remain runtime metadata in the first implementation so one
  compiled kernel can serve different page sizes. We can specialize page size
  later only if profiling justifies another source variant.
- Preserve causal/sliding-window behavior, rank-3/rank-4 tensors, optional
  position offsets, BF16/F32, contiguous K/V, arbitrary split boundaries, and
  partial first/last pages.

## Step 1: Record the plan

Status: complete

- Commit this document before source changes.

## Step 2: Split decode and prefill JIT modules

Status: pending

- Add a shared highlighted JIT attention ABI/config boundary.
- Add `CudaJitAttentionDecodeKernel.cu` for the decode partial/reduce entries.
- Reserve `CudaJitAttentionPrefillKernel.cu` as a distinct source module; do not
  route or compile it in this decode-only task.
- Leave the compiled fallback source unchanged.
- Commit the JIT module boundary together with embedding/build plumbing.

## Step 3: Page-major typed decode access

Status: pending

- Add reusable typed row/page helpers to the highlighted JIT tensor-access
  source.
- Implement page-segment iteration that handles a split beginning or ending in
  the middle of a page.
- Retain a contiguous-access specialization with direct consecutive rows.
- Add focused tests spanning page boundaries, split boundaries, sliding-window
  boundaries, BF16/F32, and contiguous/paged K/V.
- Commit the access primitive and its tests separately.

## Step 4: Cached JIT decode partial and reduce kernels

Status: pending

- Use the highlighted decode ABI/config/kernel sources established in Step 2;
  do not migrate prefill in this task.
- Specialize head dimension, dtype, Q/K/V/output access, rank, position-offset
  presence/dtype, query-heads-per-KV-head, and window/scale policy.
- Keep `tk`, page count, page pointers, split size, and number of splits in the
  runtime ABI so all decode positions reuse the same module.
- Compile the partial and reduce entries together per decode source variant and
  cache them using the complete generated source.
- Add an enabled-by-default decode JIT flag and explicit fallback-on-error flag,
  mirroring the other migrated CUDA families.
- Test JIT/fallback parity, cache reuse across changing KV lengths, and useful
  compile diagnostics, then commit.

## Step 5: Validation and profile comparison

Status: pending

- Run focused attention tests and the complete CUDA device suite, documenting
  any pre-existing unrelated failure.
- Rebuild the server with `-O2` and native CUDA architecture.
- Warm with a separate 1024-token prefill request, then measure 1024-token
  prefill plus 128-token decode under server profiling and Nsight Systems.
- Compare decode partial/reduce GPU time, complete attention GPU time, host
  `device.run`, cache behavior, end-to-end decode wall time, and throughput
  against the typed tensor-access profile.
- Record results here and commit the completed analysis.
