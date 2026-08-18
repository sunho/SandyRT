func gemma_dense_mlp(x Tensor) Tensor {
    weight_scope "mlp" {
        gate := __matmul(x, __transpose(@gate_proj.weight))
        up := __matmul(x, __transpose(@up_proj.weight))
        x = __mul(__gelu(gate), up)
        x = __matmul(x, __transpose(@down_proj.weight))
    }
    return x
}

func gemma_router_topk(x Tensor) (Tensor, Tensor) {
    weight_scope "router" {
        r := __rms_norm(x)
        r = __mul(r, @scale)
        r = __mul(r, 0.018844459036110227)

        logits := __matmul(r, __transpose(@proj.weight))
        probs := __softmax(logits, dim=-1)

        topk_weights, topk_ids := __topk(probs, k=8, dim=-1)
        denom := __sum(topk_weights, dim=-1, keepdim=1)
        topk_weights = __div(topk_weights, denom)

        expert_scale := __embedding(topk_ids, @per_expert_scale.weight)
        topk_weights = __mul(topk_weights, expert_scale)
    }
    return topk_ids, topk_weights
}

func gemma_moe(x Tensor) Tensor {
    topk_ids, topk_weights := gemma_router_topk(x)

    weight_scope "experts" {
        packed_x, packed_weights, token_ids, expert_offsets := __moe_gather(
            x,
            topk_ids,
            topk_weights,
            num_experts=128,
            top_k=8)

        gate := __moe_matmul(packed_x, expert_offsets, @gate_proj.weight, transpose_rhs=1)
        up := __moe_matmul(packed_x, expert_offsets, @up_proj.weight, transpose_rhs=1)
        hidden := __mul(__gelu(gate), up)

        packed_out := __moe_matmul(hidden, expert_offsets, @down_proj.weight, transpose_rhs=1)
        out := __moe_scatter_sum(packed_out, packed_weights, token_ids, x)
    }
    return out
}

func gemma_feed_forward(x Tensor) Tensor {
    residual := x

    dense := __rms_norm(residual, @pre_feedforward_layernorm.weight)
    dense = gemma_dense_mlp(dense)
    dense = __rms_norm(dense, @post_feedforward_layernorm_1.weight)

    routed := __rms_norm(residual, @pre_feedforward_layernorm_2.weight)
    routed = gemma_moe(routed)
    routed = __rms_norm(routed, @post_feedforward_layernorm_2.weight)

    h := __add(dense, routed)
    h = __rms_norm(h, @post_feedforward_layernorm.weight)
    x = __add(residual, h)
    x = __mul(x, @skip_scale)

    return x
}

func gemma_local_kv_attention(x Tensor) (Tensor, Tensor, Tensor) {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __reshape(q, shape=[-1, -1, 16, 256])
        k = __reshape(k, shape=[-1, -1, 8, 256])
        v = __reshape(v, shape=[-1, -1, 8, 256])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        q = __rms_norm(q, @q_norm.weight)
        k = __rms_norm(k, @k_norm.weight)
        v = __rms_norm(v)

        q = __rope(q, rope_theta=10000.0, split_half=1)
        k = __rope(k, rope_theta=10000.0, split_half=1)

        ctx := __attention(q, k, v, window=1024, scale=1.0)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, -1, 4096])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out, k, v
}

func gemma_global_kv_attention(x Tensor) (Tensor, Tensor, Tensor) {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __reshape(q, shape=[-1, -1, 16, 512])
        k = __reshape(k, shape=[-1, -1, 2, 512])
        v = __reshape(v, shape=[-1, -1, 2, 512])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        q = __rms_norm(q, @q_norm.weight)
        k = __rms_norm(k, @k_norm.weight)
        v = __rms_norm(v)

        q = __rope(q, rope_theta=1000000.0, rotary_dim=128, split_half=1)
        k = __rope(k, rope_theta=1000000.0, rotary_dim=128, split_half=1)

        ctx := __attention(q, k, v, window=0, scale=1.0)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, -1, 8192])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out, k, v
}

func gemma_local_attention(x Tensor, k Tensor, v Tensor) Tensor {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        q = __reshape(q, shape=[-1, -1, 16, 256])
        q = __permute(q, dims=[0, 2, 1, 3])
        q = __rms_norm(q, @q_norm.weight)
        q = __rope(q, rope_theta=10000.0, split_half=1)

        ctx := __attention(q, k, v, window=1024, scale=1.0)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, -1, 4096])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}

func gemma_global_attention(x Tensor, k Tensor, v Tensor) Tensor {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        q = __reshape(q, shape=[-1, -1, 16, 512])
        q = __permute(q, dims=[0, 2, 1, 3])
        q = __rms_norm(q, @q_norm.weight)
        q = __rope(q, rope_theta=1000000.0, rotary_dim=128, split_half=1)

        ctx := __attention(q, k, v, window=0, scale=1.0)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, -1, 8192])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}

func gemma_local_kv_layer(x Tensor, i int) (Tensor, Tensor, Tensor) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h, k, v := gemma_local_kv_attention(h)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        x = gemma_feed_forward(x)
    }
    return x, k, v
}

func gemma_global_kv_layer(x Tensor, i int) (Tensor, Tensor, Tensor) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h, k, v := gemma_global_kv_attention(h)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        x = gemma_feed_forward(x)
    }
    return x, k, v
}

func gemma_local_layer(x Tensor, i int, k Tensor, v Tensor) Tensor {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = gemma_local_attention(h, k, v)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        x = gemma_feed_forward(x)
    }
    return x
}

func gemma_global_layer(x Tensor, i int, k Tensor, v Tensor) Tensor {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = gemma_global_attention(h, k, v)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        x = gemma_feed_forward(x)
    }
    return x
}

func gemma4a4b26b_model(input_ids Tensor) Tensor {
    weight_scope "language_model.model" {
        x := __embedding(input_ids, @embed_tokens.weight)
        x = __mul(x, 53.0659966456864)

        var sliding_k Tensor
        var sliding_v Tensor
        var full_k Tensor
        var full_v Tensor

        for i := range(30) {
            if i % 6 == 5 {
                x, full_k, full_v = gemma_global_kv_layer(x, i)
            } else {
                x, sliding_k, sliding_v = gemma_local_kv_layer(x, i)
            }
        }

        x = __rms_norm(x, @norm.weight)
        logits := __matmul(x, __transpose(@embed_tokens.weight))
        logits = __softcap(logits, 30.0)
    }
    return logits
}
