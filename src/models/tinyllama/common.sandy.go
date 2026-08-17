func tinyllama_mlp(x Tensor) Tensor {
    weight_scope "mlp" {
        gate := __matmul(x, __transpose(@gate_proj.weight))
        up := __matmul(x, __transpose(@up_proj.weight))
        x = __mul(__silu(gate), up)
        x = __matmul(x, __transpose(@down_proj.weight))
    }
    return x
}

func tinyllama_eval_attention(x Tensor, position_id Tensor, k_cache Tensor, v_cache Tensor) Tensor {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __reshape(q, shape=[-1, -1, 32, 64])
        k = __reshape(k, shape=[-1, -1, 4, 64])
        v = __reshape(v, shape=[-1, -1, 4, 64])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        q = __rope(q, position_id, rope_theta=10000.0, split_half=1)
        k = __rope(k, position_id, rope_theta=10000.0, split_half=1)

        __paged_append(k_cache, k)
        __paged_append(v_cache, v)

        ctx := __attention(q, k_cache, v_cache, position_id, window=0)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, -1, 2048])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}

func tinyllama_eval_block(x Tensor, position_id Tensor, i int, k_cache Tensor, v_cache Tensor) Tensor {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight, epsilon=0.00001)
        h = tinyllama_eval_attention(h, position_id, k_cache, v_cache)
        x = __add(x, h)

        h = __rms_norm(x, @post_attention_layernorm.weight, epsilon=0.00001)
        h = tinyllama_mlp(h)
        x = __add(x, h)
    }
    return x
}
