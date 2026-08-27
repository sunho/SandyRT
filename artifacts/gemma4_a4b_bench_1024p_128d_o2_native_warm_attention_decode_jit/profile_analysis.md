# Gemma 4 A4B page-major JIT decode-attention profile

Date: 2026-08-27

## Configuration

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Build: CMake `Release`; C++ and CUDA `-O2 -DNDEBUG`
- CUDA architecture: `CMAKE_CUDA_ARCHITECTURES=native`
- Profiler: Nsight Systems 2026.1.3, CUDA/NVTX trace
- Server-side profiling: enabled
- Warmup: separate 1024-token prefill plus 1-token output request
- Measurement: 1024-token prefill plus 128-token output request
- Warmup input: `[1] * 1024`; measured input: `[2] * 1024`
- Baseline: the immediately preceding typed tensor-access JIT profile in
  `../gemma4_a4b_bench_1024p_128d_o2_native_warm_tensor_jit/`
- Source state: the implementation adds the cached decode-attention JIT
  dispatch on top of commit `e3a3541`.

The GPU comparison uses the first 122 complete decode evaluations in both
captures. Each evaluation has 2,207 kernels. The measured prefill is the second
2,232-kernel prefill segment, after the separate warmup request.

## Result

The page-major JIT makes the attention partial kernel 88.9% faster and total
decode attention GPU execution 82.3% faster. Summed decode GPU kernel time falls
33.4%. The representative no-Nsight end-to-end decode improves 3.9% in latency
and 4.1% in throughput because CPU launch/orchestration is now the bottleneck.
Prefill is unchanged, as intended.

## End-to-end

| Mode | Measurement | Typed baseline | Page-major JIT | Change |
|---|---|---:|---:|---:|
| No Nsight | Prefill wall | 131.774 ms | 133.274 ms | +1.1% |
| No Nsight | Prefill throughput | 7,770.881 tok/s | 7,683.394 tok/s | -1.1% |
| No Nsight | Decode wall | 4,365.471 ms | 4,194.467 ms | -3.9% |
| No Nsight | Decode throughput | 29.321 tok/s | 30.516 tok/s | +4.1% |
| Nsight | Prefill wall | 145.312 ms | 145.694 ms | +0.3% |
| Nsight | Prefill throughput | 7,046.911 tok/s | 7,028.450 tok/s | -0.3% |
| Nsight | Decode wall | 5,955.659 ms | 5,948.698 ms | -0.1% |
| Nsight | Decode throughput | 21.492 tok/s | 21.517 tok/s | +0.1% |

Nsight's per-launch tracing makes the host launch path dominate the measured
wall clock. The no-Nsight numbers are the representative application result;
the Nsight GPU durations are used for kernel attribution.

## Decode attention GPU detail: first 122 complete evaluations

| Component | Launches | Typed baseline | Page-major JIT | Change |
|---|---:|---:|---:|---:|
| Partial | 3,660 | 727.388 ms | 80.415 ms | -88.9% |
| Reduce | 3,660 | 58.881 ms | 58.710 ms | -0.3% |
| **Attention total** | **7,320** | **786.269 ms** | **139.125 ms** | **-82.3%** |

The partial launch geometry remains one warp per `(batch, qHead, split)`. The
JIT kernel resolves K and V page-table pointers once per page segment in every
lane, then advances typed row pointers inside the page. There is no explicit
shuffle/broadcast. The reduction path intentionally retains the same
algorithm, explaining its flat result.

The old compiled partial kernels used 86 registers/thread at head dimension
256 and 102 at head dimension 512. The combined JIT entry reports 40 and 55
registers/thread for its two source variants. This specialization benefit is
additional to eliminating repeated paged-address calculation.

## Nsight GPU time: measured prefill

| Stage | Kernels | Typed baseline | Page-major JIT | Change |
|---|---:|---:|---:|---:|
| Attention | 30 | 55.641 ms | 55.686 ms | +0.1% |
| Matmul | 361 | 24.360 ms | 24.389 ms | +0.1% |
| Elementwise | 1,117 | 6.732 ms | 6.737 ms | +0.1% |
| Norm | 331 | 5.021 ms | 5.017 ms | -0.1% |
| MoE scatter | 60 | 4.236 ms | 4.236 ms | +0.0% |
| MoE gather | 120 | 2.853 ms | 2.866 ms | +0.5% |
| TopK | 32 | 0.671 ms | 0.672 ms | +0.1% |
| RoPE | 60 | 0.646 ms | 0.646 ms | +0.0% |
| Layout | 30 | 0.576 ms | 0.575 ms | -0.2% |
| Softmax | 30 | 0.103 ms | 0.103 ms | +0.0% |
| Reduction | 30 | 0.046 ms | 0.045 ms | -2.2% |
| Gather | 31 | 0.043 ms | 0.044 ms | +2.3% |
| **Total** | **2,232** | **100.929 ms** | **101.017 ms** | **+0.1%** |

## Nsight GPU time: first 122 complete decode evaluations

| Stage | Kernels | Typed baseline | Page-major JIT | Change |
|---|---:|---:|---:|---:|
| Matmul | 40,992 | 747.719 ms | 746.328 ms | -0.2% |
| Norm | 40,382 | 184.953 ms | 185.118 ms | +0.1% |
| Attention | 7,320 | 786.269 ms | 139.125 ms | -82.3% |
| Elementwise | 136,274 | 117.405 ms | 117.333 ms | -0.1% |
| TopK | 3,904 | 46.733 ms | 46.588 ms | -0.3% |
| MoE gather | 14,640 | 18.911 ms | 18.933 ms | +0.1% |
| RoPE | 7,320 | 12.123 ms | 12.123 ms | +0.0% |
| MoE scatter | 7,320 | 9.955 ms | 10.041 ms | +0.9% |
| Softmax | 3,660 | 7.044 ms | 7.174 ms | +1.8% |
| Reduction | 3,660 | 5.244 ms | 5.071 ms | -3.3% |
| Gather | 3,782 | 4.364 ms | 4.490 ms | +2.9% |
| **Total** | **269,254** | **1,940.723 ms** | **1,292.325 ms** | **-33.4%** |

Attention accounts for 647.144 ms of the 648.398 ms reduction in summed GPU
kernel duration. The remaining stages are effectively unchanged.

## Server host-observed `device.run`: measured prefill

These are CPU-side dispatch durations from server profiling, not CUDA kernel
durations.

| Stage | Calls | Typed baseline | Page-major JIT | Change |
|---|---:|---:|---:|---:|
| MoE gather | 30 | 56.952 ms | 56.967 ms | +0.0% |
| MoE matmul | 90 | 13.591 ms | 13.746 ms | +1.1% |
| MoE scatter | 30 | 9.076 ms | 9.133 ms | +0.6% |
| Elementwise | 1,117 | 8.744 ms | 8.781 ms | +0.4% |
| RoPE | 60 | 6.119 ms | 6.228 ms | +1.8% |
| Dense matmul | 241 | 5.481 ms | 5.428 ms | -1.0% |
| Norm | 331 | 2.722 ms | 2.702 ms | -0.7% |
| Attention | 30 | 0.763 ms | 0.748 ms | -2.0% |
| Layout | 30 | 0.347 ms | 0.312 ms | -10.1% |
| TopK | 31 | 0.318 ms | 0.314 ms | -1.3% |
| Gather | 31 | 0.274 ms | 0.265 ms | -3.3% |
| Softmax | 30 | 0.248 ms | 0.232 ms | -6.5% |
| Reduction | 30 | 0.221 ms | 0.217 ms | -1.8% |
| **Summed `device.run`** | **2,081** | **104.856 ms** | **105.073 ms** | **+0.2%** |

## Server host-observed `device.run`: all 127 decode evaluations

| Stage | Calls | Typed baseline | Page-major JIT | Change |
|---|---:|---:|---:|---:|
| Elementwise | 141,859 | 1,098.064 ms | 1,100.213 ms | +0.2% |
| MoE matmul | 11,430 | 830.123 ms | 828.661 ms | -0.2% |
| Dense matmul | 30,607 | 534.276 ms | 528.481 ms | -1.1% |
| MoE gather | 3,810 | 428.868 ms | 401.824 ms | -6.3% |
| Norm | 42,037 | 331.296 ms | 334.775 ms | +1.1% |
| RoPE | 7,620 | 223.695 ms | 222.893 ms | -0.4% |
| MoE scatter | 3,810 | 195.589 ms | 197.573 ms | +1.0% |
| Attention | 3,810 | 107.033 ms | 89.524 ms | -16.4% |
| TopK | 3,937 | 39.434 ms | 40.638 ms | +3.1% |
| Gather | 3,937 | 33.607 ms | 33.696 ms | +0.3% |
| Reduction | 3,810 | 28.082 ms | 28.219 ms | +0.5% |
| Softmax | 3,810 | 27.837 ms | 28.169 ms | +1.2% |
| **Summed `device.run`** | **260,477** | **3,877.904 ms** | **3,834.667 ms** | **-1.1%** |

The 647 ms summed GPU attention reduction turns into only 17.5 ms less
attention dispatch time and 171 ms no-Nsight end-to-end improvement. There are
260,477 host `device.run` calls for 127 decode evaluations. Host dispatch and
graph orchestration overlap or serialize independently of summed GPU kernel
duration, so attention is no longer the primary wall-clock limiter.

## Engine CPU stages

| Phase | Stage | Typed baseline | Page-major JIT | Change |
|---|---|---:|---:|---:|
| Prefill | `run_values.total` | 144.567 ms | 144.907 ms | +0.2% |
| Prefill | Paged-KV append | 22.016 ms | 22.469 ms | +2.1% |
| Prefill | Output allocation | 4.345 ms | 4.313 ms | -0.7% |
| Prefill | Remaining deallocation | 1.442 ms | 1.490 ms | +3.3% |
| Prefill | Input binding | 1.259 ms | 1.120 ms | -11.0% |
| Prefill | Scratch planning | 1.028 ms | 1.009 ms | -1.8% |
| Prefill | Output readback | 1.017 ms | 1.019 ms | +0.2% |
| Prefill | Collect inputs | 0.989 ms | 0.910 ms | -8.0% |
| Prefill | Descriptor inference | 0.823 ms | 0.665 ms | -19.2% |
| Prefill | Collect outputs | 0.747 ms | 0.702 ms | -6.0% |
| Prefill | Layout alias check | 0.211 ms | 0.266 ms | +26.1% |
| Decode | `run_values.total` | 5,860.257 ms | 5,843.057 ms | -0.3% |
| Decode | Scratch planning | 215.371 ms | 228.499 ms | +6.1% |
| Decode | Remaining deallocation | 156.081 ms | 159.508 ms | +2.2% |
| Decode | Input binding | 154.419 ms | 151.308 ms | -2.0% |
| Decode | Descriptor inference | 142.139 ms | 145.390 ms | +2.3% |
| Decode | Output readback | 129.647 ms | 129.658 ms | +0.0% |
| Decode | Output allocation | 117.293 ms | 117.137 ms | -0.1% |
| Decode | Collect inputs | 101.851 ms | 99.879 ms | -1.9% |
| Decode | Paged-KV append | 94.220 ms | 94.278 ms | +0.1% |
| Decode | Collect outputs | 69.333 ms | 73.480 ms | +6.0% |
| Decode | Layout alias check | 25.828 ms | 26.519 ms | +2.7% |

Paged-KV append is unchanged because this change optimizes attention reads,
not cache writes. Scratch planning and remaining deallocation remain around
1.80 ms and 1.26 ms per decode evaluation under Nsight, respectively.

## JIT cache and validation

- Graph 1: 1,844 hits, 25 misses/modules.
- Graph 2: 3,684 cumulative hits, still 25 misses/modules.
- The baseline had 23 modules, so the two new modules are exactly the local
  head-dimension-256 and global head-dimension-512 decode-attention variants.
- Dynamic KV length, page count, split count, and scratch size stay in the
  runtime ABI and do not create new CUDA modules.
- Focused compile/page-boundary/window/cache-reuse tests pass.
- Full CUDA device suite: 70/71 pass. The sole failure is the pre-existing,
  unrelated `CudaDeviceTest.RunGatherReportsOutOfRangeId` asynchronous gather
  bounds-reporting test.

## Next bottleneck

CUDA Graph capture is now more attractive than further work on the attention
partial loop. The decode performs about 2,051 host device operations and 2,207
GPU kernels per token. Elementwise dispatch alone consumes 1.100 seconds of
host time across the request, followed by MoE matmul at 0.829 seconds and dense
matmul at 0.528 seconds. Capturing a stable per-token launch graph, or otherwise
batching/fusing launches, targets the wall-clock bottleneck exposed by this JIT
optimization.
