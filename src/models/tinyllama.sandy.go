func tinyllama_mlp(x Tensor) Tensor {
    weight_scope "mlp" {
        gate := __matmul(x, __transpose(@gate_proj.weight))
        up := __matmul(x, __transpose(@up_proj.weight))
        x = __mul(__silu(gate), up)
        x = __matmul(x, __transpose(@down_proj.weight))
    }
    return x
}

func tinyllama_attention(x Tensor) Tensor {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __reshape(q, shape=[-1, 32, 32, 64])
        k = __reshape(k, shape=[-1, 32, 4, 64])
        v = __reshape(v, shape=[-1, 32, 4, 64])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        q = __rope(q, rope_theta=10000.0, split_half=1)
        k = __rope(k, rope_theta=10000.0, split_half=1)

        ctx := __attention(q, k, v, window=0)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, 32, 2048])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}

func tinyllama_block(x Tensor, i int) Tensor {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight, epsilon=0.00001)
        h = tinyllama_attention(h)
        x = __add(x, h)

        h = __rms_norm(x, @post_attention_layernorm.weight, epsilon=0.00001)
        h = tinyllama_mlp(h)
        x = __add(x, h)
    }
    return x
}

func main(input_ids Tensor[[1, 32], i64]) Tensor {
    weight_scope "model" {
        x := __embedding(input_ids, @embed_tokens.weight)

        for i := range(22) {
            x = tinyllama_block(x, i)
        }

        x = __rms_norm(x, @norm.weight, epsilon=0.00001)
        logits := __matmul(x, __transpose(@lm_head.weight))
    }
    return logits
}
