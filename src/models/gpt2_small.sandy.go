func gpt2_attention(x Node, i int) Node {
    weight_scope "h.{i}.attn" {
        q := __linear(x, @q_proj.weight, @q_proj.bias)
        k := __linear(x, @k_proj.weight, @k_proj.bias)
        v := __linear(x, @v_proj.weight, @v_proj.bias)

        q = __reshape(q, shape=[1, 16, 12, 64])
        k = __reshape(k, shape=[1, 16, 12, 64])
        v = __reshape(v, shape=[1, 16, 12, 64])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        scores := __sliding_query_key_score(q, k, window=0)
        probs := __softmax(scores, dim=-1)
        ctx := __matmul(probs, v)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[1, 16, 768])

        out := __linear(ctx, @c_proj.weight, @c_proj.bias)
    }
    return out
}

func gpt2_mlp(x Node, i int) Node {
    weight_scope "h.{i}.mlp" {
        x = __linear(x, @c_fc.weight, @c_fc.bias)
        x = __gelu(x)
        x = __linear(x, @c_proj.weight, @c_proj.bias)
    }
    return x
}

func gpt2_block(x Node, i int) Node {
    weight_scope "h.{i}" {
        h := __layer_norm(x, @ln_1.weight, @ln_1.bias, epsilon=0.00001)
        h = gpt2_attention(h, i)
        x = __add(x, h)

        h = __layer_norm(x, @ln_2.weight, @ln_2.bias, epsilon=0.00001)
        h = gpt2_mlp(h, i)
        x = __add(x, h)
    }
    return x
}

func main(input_ids Node, position_ids Node) Node {
    x := __embedding(input_ids, @wte.weight)
    pos := __embedding(position_ids, @wpe.weight)
    x = __add(x, pos)

    for i := range(12) {
        x = gpt2_block(x, i)
    }

    x = __layer_norm(x, @ln_f.weight, @ln_f.bias, epsilon=0.00001)
    logits := __matmul(x, __transpose(@wte.weight))
    return logits
}
