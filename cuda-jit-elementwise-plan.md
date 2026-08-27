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
- The CUDA JIT cache hashes the complete effective compilation unit: main
  source, generated and embedded headers, entry point, options, ABI, target
  architecture, and NVRTC version. A kernel family controls reuse by deciding
  which runtime properties appear in its generated source.
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

Status: complete

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

Status: complete

- Emit one `GeneratedElementwiseEvaluator::eval` body per scalar DAG.
- Instantiate the highlighted policy kernel as
  `ElementwiseKernel<RuntimeLoader, GeneratedEvaluator, RuntimeStorer>`.
- Compile/cache the evaluator and retain its loaded function in the compiled CUDA
  operation.
- Launch it on the existing CUDA stream, with interpreter fallback on configured
  JIT failure.
- Do not include runtime pointers in cache keys.
- Elementwise emits only the scalar DAG. Shapes, tensor value IDs, broadcast,
  dtype/access policy, and pointers stay in runtime parameters, so they do not
  split the full-source JIT cache.

## Step 5: Correctness, cache, and failure coverage

Status: complete

- Test every scalar operation and fused chains.
- Test F32/BF16, constants, broadcast, strided tensors, paged tensors, refreshed
  paged metadata, and zero-sized output.
- Test JIT reuse, ABI/template invalidation, concurrent same-key lookup, useful
  NVRTC diagnostics, and interpreter/JIT parity.
- Run the complete CUDA device test suite and document any unrelated failure.

Validation: 12/12 focused JIT tests pass. The complete CUDA device suite is
62/63 passing. The sole failure is the pre-existing, unrelated
`CudaDeviceTest.RunGatherReportsOutOfRangeId`: the asynchronous gather launch
returns success where the test expects synchronous device-side bounds reporting.

## Step 6: Warmed benchmark and profile comparison

Status: complete

- Rebuild the server with `-O2` and native CUDA architecture.
- Warm up with a 1024-token prefill.
- Measure 1024-token prefill plus 128-token decode with server profiling and
  Nsight Systems.
- Compare elementwise GPU time, scratch planning, descriptor inference, wall
  times, throughput, compilation time, hit/miss counts, and module count against
  the `cudaMemcpy2DAsync` baseline.
- Record results in this document and commit the completed analysis.

### Configuration and artifacts

- Release flags were verified in `build-server-cuda-o2-native/CMakeCache.txt`:
  C++ and CUDA both use `-O2 -DNDEBUG`, and
  `CMAKE_CUDA_ARCHITECTURES=native`.
- The worker was run with server `--profile` inside Nsight Systems 2026.1.3.
- A separate 1024-token/1-token request warmed prefill before the measured
  1024-token/128-token request.
- The report, SQLite export, and request logs are under
  `artifacts/gemma4_a4b_bench_1024p_128d_o2_native_warm_jit/`.

### Cache and compilation behavior

- The eval-token graph populated 15 JIT modules: 15 misses and 1,102 hits.
- The prefill graph reused those same modules: cumulative totals became 15
  misses and 2,219 hits, still 15 modules. This confirms that identical scalar
  DAGs reuse code across eval and prefill shapes; runtime pointers and shapes do
  not enter the scalar JIT key.
- Normal, non-Nsight JIT compile/module-load time was 240.307 ms total. Nsight
  instrumentation inflated the same one-time work to 3,446.693 ms.
- After warmup, the runtime plan cache had 0 hits, 1 miss, and 1 entry. After the
  measured request it had 1 hit, 128 misses, and 128 entries. The repeated
  1024-token prefill shape hit; all 127 decode evaluations missed because each
  Paged-KV length is distinct, as required for scratch correctness.

### End-to-end server profile under Nsight

| Measurement | Baseline | JIT | Change |
|---|---:|---:|---:|
| Prefill wall | 231.671 ms | 149.241 ms | -35.6% |
| Prefill throughput | 4,420.054 tok/s | 6,861.368 tok/s | +55.2% |
| Decode wall, 128 output tokens | 6,005.887 ms | 6,042.613 ms | +0.6% |
| Decode throughput | 21.312 tok/s | 21.183 tok/s | -0.6% |

Nsight adds unusually high overhead to the driver-API JIT launches. A
supplementary run without Nsight, after the same warmup, measured 138.030 ms
prefill (7,418.656 tok/s) and 4,644.523 ms decode (27.559 tok/s). This is not
used as the baseline comparison because the old capture was made under Nsight.

### GPU time by stage: measured prefill

These are durations from `CUPTI_ACTIVITY_KIND_KERNEL`, not host launch time.
The launch counts are identical between profiles.

| GPU stage | Launches | Baseline | JIT | Change |
|---|---:|---:|---:|---:|
| Elementwise | 1,117 | 89.628 ms | 8.957 ms | -90.0% |
| Attention | 30 | 55.653 ms | 55.643 ms | -0.0% |
| cuBLAS matmul, dense + MoE | 361 | 24.268 ms | 24.216 ms | -0.2% |
| Norm | 331 | 7.070 ms | 7.078 ms | +0.1% |
| MoE scatter sum | 60 | 4.233 ms | 4.236 ms | +0.1% |
| MoE gather | 120 | 2.876 ms | 2.871 ms | -0.2% |
| RoPE | 60 | 0.956 ms | 0.950 ms | -0.6% |
| TopK | 32 | 0.674 ms | 0.672 ms | -0.3% |
| Layout transform | 30 | 0.637 ms | 0.637 ms | 0.0% |
| Softmax | 30 | 0.158 ms | 0.156 ms | -1.3% |
| Gather | 31 | 0.060 ms | 0.060 ms | 0.0% |
| Reduction | 30 | 0.052 ms | 0.052 ms | 0.0% |
| **All kernels** | **2,232** | **186.266 ms** | **105.528 ms** | **-43.3%** |

Prefill memcpy remained unchanged: 2,640 copies and 1.482 ms baseline versus
2,640 copies and 1.478 ms with JIT. The elementwise improvement is therefore a
kernel-compute reduction, not a copy-count artifact.

### GPU time by stage: matched decode window

The old CUPTI capture stopped recording before all 127 decode evaluations. To
avoid comparing unequal work, this table uses the first 122 complete decode
evaluations present in both reports. Both sides contain exactly 269,254 kernel
launches, with identical per-stage launch counts.

| GPU stage | Launches | Baseline | JIT | Change |
|---|---:|---:|---:|---:|
| Attention | 7,320 | 788.522 ms | 788.295 ms | -0.0% |
| cuBLAS matmul, dense + MoE | 40,992 | 750.479 ms | 747.350 ms | -0.4% |
| Norm | 40,382 | 252.222 ms | 252.774 ms | +0.2% |
| Elementwise | 136,274 | 243.583 ms | 162.202 ms | -33.4% |
| TopK | 3,904 | 46.715 ms | 46.486 ms | -0.5% |
| RoPE | 7,320 | 33.211 ms | 32.930 ms | -0.8% |
| MoE gather | 14,640 | 18.891 ms | 19.096 ms | +1.1% |
| Softmax | 3,660 | 11.614 ms | 11.453 ms | -1.4% |
| MoE scatter sum | 7,320 | 9.854 ms | 9.893 ms | +0.4% |
| Gather | 3,782 | 6.031 ms | 6.145 ms | +1.9% |
| Reduction | 3,660 | 6.103 ms | 5.995 ms | -1.8% |
| **All kernels** | **269,254** | **2,167.227 ms** | **2,082.619 ms** | **-3.9%** |

Decode memcpy was also stable for the matched window: 88,564 copies and
29.564 ms baseline versus 88,564 copies and 29.604 ms with JIT. Paged-KV append
therefore did not regress at the GPU-copy level.

### Host-observed device-run time by operation

These server-profile intervals measure host-observed `device.run` boundaries.
They include launch overhead and any asynchronous work absorbed by later CUDA
calls, so they are not exclusive CPU compute time. They are nevertheless useful
for locating host launch pressure.

| Prefill operation | Baseline | JIT | Change |
|---|---:|---:|---:|
| MoE gather | 83.516 ms | 59.030 ms | -29.3% |
| MoE matmul | 62.598 ms | 15.092 ms | -75.9% |
| RoPE | 12.725 ms | 6.782 ms | -46.7% |
| MoE scatter sum | 9.371 ms | 9.207 ms | -1.7% |
| Elementwise | 9.279 ms | 8.236 ms | -11.2% |
| Dense matmul | 5.415 ms | 5.500 ms | +1.6% |
| Norm | 2.879 ms | 3.062 ms | +6.4% |
| Attention | 0.820 ms | 0.792 ms | -3.5% |
| Layout transform | 0.395 ms | 0.386 ms | -2.3% |
| Softmax | 0.361 ms | 0.274 ms | -24.1% |
| Gather | 0.318 ms | 0.308 ms | -3.1% |
| TopK | 0.312 ms | 0.326 ms | +4.3% |
| Reduction | 0.263 ms | 0.273 ms | +3.9% |

| Decode operation, 127 evals | Baseline | JIT | Change |
|---|---:|---:|---:|
| Elementwise | 1,173.417 ms | 1,052.644 ms | -10.3% |
| MoE matmul | 808.697 ms | 842.078 ms | +4.1% |
| Dense matmul | 505.746 ms | 540.815 ms | +6.9% |
| MoE gather | 410.472 ms | 452.714 ms | +10.3% |
| Norm | 352.516 ms | 383.229 ms | +8.7% |
| RoPE | 219.859 ms | 240.540 ms | +9.4% |
| MoE scatter sum | 190.418 ms | 200.689 ms | +5.4% |
| Attention | 94.519 ms | 106.207 ms | +12.4% |
| Gather | 37.233 ms | 40.699 ms | +9.3% |
| TopK | 36.089 ms | 38.965 ms | +8.0% |
| Softmax | 33.460 ms | 34.211 ms | +2.2% |
| Reduction | 32.460 ms | 34.321 ms | +5.7% |

### Engine planning and lifetime stages

| Phase/stage | Baseline | JIT/cache | Change |
|---|---:|---:|---:|
| Prefill total `run_values` | 230.653 ms | 148.474 ms | -35.6% |
| Prefill scratch planning | 2.992 ms | 1.042 ms | -65.2% |
| Prefill descriptor inference | 1.209 ms | 0.687 ms | -43.2% |
| Prefill remaining-buffer deallocation | 1.603 ms | 1.411 ms | -12.0% |
| Prefill output read | 1.274 ms | 1.029 ms | -19.2% |
| Decode total `run_values` | 5,834.045 ms | 5,946.907 ms | +1.9% |
| Decode scratch planning | 224.690 ms | 217.758 ms | -3.1% |
| Decode descriptor inference | 112.557 ms | 136.633 ms | +21.4% |
| Decode remaining-buffer deallocation | 201.977 ms | 148.489 ms | -26.5% |
| Decode output read | 134.017 ms | 130.348 ms | -2.7% |

The scratch cache helps repeated exact signatures, demonstrated by the warmed
prefill hit. It intentionally cannot reuse decode layouts across changing
Paged-KV lengths, so decode planning remains about 1.715 ms/evaluation. The
descriptor-inference increase is run-to-run host variance rather than JIT work;
the JIT does not execute in that stage.

### Conclusion

The straight-line evaluator removes the scalar switch/loop cost and achieves
the intended GPU reduction, especially for wide prefill tensors. Decode remains
dominated by a very large number of tiny launches: 136,274 elementwise launches
in the matched window and 269,254 launches overall. No CUDA Graph capture is
active (`graphId` is NULL for every kernel in this profile). CUDA Graphs or a
broader fusion/store specialization are therefore the next mechanisms likely
to improve decode wall time; further scalar arithmetic specialization alone
has limited headroom.

## Baseline

- Prefill wall: 231.671 ms; elementwise GPU: 89.628 ms.
- Decode wall: 6005.908 ms; elementwise GPU: 244.306 ms.
- Prefill scratch planning: 2.992 ms.
- Decode scratch planning: 224.690 ms total, 1.769 ms/evaluation.
- Decode descriptor inference: 112.557 ms total, 0.886 ms/evaluation.
