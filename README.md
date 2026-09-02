# SandyRT

SandyRT is an LLM inference runtime built from scratch in C++20 and CUDA. Models are written in **Sandy**, a small Go-flavored tensor DSL, compiled through a multi-level IR, and run on hand-written CUDA kernels.

## Demo: Gemma 4 MoE (26B-A4B)

<!-- TODO: attach gemma4 moe demo video here -->

## Features

- **One model, one flat file.** Attention, MoE routing, KV caches, and sampling are plain tensor ops in a single Sandy file, with no class hierarchy or config indirection.
- **Compiler and kernels from scratch.** Sandy lowers through multi-level IRs into hand-written CUDA kernels JIT-compiled by NVRTC. cuBLAS is used only for dense matmuls.
- **Optimized CUDA backend.**
    - 15,200 tok/s prefill and 128.6 tok/s decode for the 26B MoE (BF16) on a single RTX PRO 6000.
    - CUDA Graph replay cuts decode CUDA API calls ~99% (57,936 → 839 per 128 tokens).
    - JIT paged decode attention resolves page tables once per page, not per element — 82% less attention GPU time.
    - NVRTC-fused elementwise chains run as one straight-line kernel per chain instead of one launch per op — +55% prefill throughput.

## The Sandy language

Sandy is a small DSL for writing model forward passes. It looks like Go, but every function is a tensor computation the compiler traces into a graph. Weights are referenced by name (`weight_scope` + `@weight`) and resolved against a safetensors checkpoint. A sandy file is fully explicit about its computation without module/class indirections, which makes it easy to read — for humans and machines alike. This makes it a good playground to try hand customizations and optimizations.

### Paged KV caches

```go
func decode_attention(x Tensor, position_id Tensor, k_cache PagedTensor, v_cache PagedTensor) Tensor {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __rope(__rms_norm(q, @q_norm.weight), position_id, rope_theta=10000.0, split_half=1)
        k = __rope(__rms_norm(k, @k_norm.weight), position_id, rope_theta=10000.0, split_half=1)

        __paged_append(k_cache, k)
        __paged_append(v_cache, v)

        ctx := __attention(q, k_cache, v_cache, position_id, window=1024, scale=1.0)
        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}
```

### Config constants and per-layer weight scopes

```go
config const NUM_LAYERS int
config const TOP_K int

func main(input_id Tensor, k_cache [NUM_LAYERS]PagedTensor, v_cache [NUM_LAYERS]PagedTensor) (Tensor, Tensor) {
    x := __embedding(input_id, @embed_tokens.weight)

    for i := range(NUM_LAYERS) {
        weight_scope "layers.{i}" {
            x = __add(x, attention(x, k_cache[i], v_cache[i]))
            x = __add(x, mlp(x))
        }
    }

    logits := __matmul(__rms_norm(x, @norm.weight), __transpose(@embed_tokens.weight))
    return __topk(logits, k=TOP_K, dim=-1)
}
```

### MoE ops

```go
func moe_ffn(x Tensor) Tensor {
    weight_scope "router" {
        logits := __matmul(x, __transpose(@proj.weight))
        topk_weights, topk_ids := __topk(__softmax(logits, dim=-1), k=8, dim=-1)
    }

    weight_scope "experts" {
        packed_x, packed_weights, token_ids, expert_offsets := __moe_gather(
            x, topk_ids, topk_weights, num_experts=128, top_k=8)

        gate := __moe_matmul(packed_x, expert_offsets, __transpose(@gate_proj.weight))
        up := __moe_matmul(packed_x, expert_offsets, __transpose(@up_proj.weight))
        down := __moe_matmul(__mul(__gelu(gate), up), expert_offsets, __transpose(@down_proj.weight))

        out := __moe_scatter_sum(down, packed_weights, token_ids, x)
    }
    return out
}
```

## Running Gemma 4 MoE

```bash
./scripts/setup_gemma4_moe.sh          # venv + CUDA build in build-cuda/
source .venv/bin/activate

hf auth login                          # accept Google's model terms first
python scripts/run_gemma4a4b26b.py --download --prepare-only

python scripts/run_gemma4a4b26b.py \
  --eval-token \
  --max-answer-tokens 32 \
  "Explain mixture-of-experts models briefly."
```

The prepare step downloads the ~52 GB BF16 checkpoint and converts it (keep ~110 GB free). Details in [`GEMMA4_MOE.md`](GEMMA4_MOE.md).

## Building

```bash
cmake -B build -DSANDY_ENABLE_CUDA=ON      # CPU-only: omit the flag
cmake --build build -j
```

Requires CMake ≥ 3.20, a C++20 compiler, and the CUDA toolkit for the GPU backend. Add `-DSANDY_ENABLE_SERVER=ON` for the gRPC server, which pairs with an OpenAI-compatible chat frontend (`scripts/chat_openai_server.py`).
