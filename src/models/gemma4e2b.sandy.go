func gemma_gated_mlp(x Node) Node {
    weight_scope "mlp" {
        gate := __matmul(x, __transpose(@gate_proj.weight))
        up := __matmul(x, __transpose(@up_proj.weight))
        x = __mul(__gelu(gate), up)
        x = __matmul(x, __transpose(@down_proj.weight))
    }
    return x
}

func gemma_per_layer_input(input_embed Node, input_ids Node, i int) Node {
    weight_scope "per_layer_inputs.{i}" {
        model_input := __matmul(input_embed, @model_projection.weight)
        model_input = __mul(model_input, 0.0255126953125)
        model_input = __rms_norm(model_input, @projection_norm.weight)

        token_input := __embedding(input_ids, @embedding.weight)
        token_input = __mul(token_input, __sqrt(256))

        out := __add(model_input, token_input)
        out = __mul(out, 0.70703125)
    }
    return out
}

func gemma_apply_per_layer_input(x Node, per_layer_input Node) Node {
    h := __matmul(x, @per_layer_input_gate.weight)
    h = __mul(__gelu(h), per_layer_input)
    h = __matmul(h, @per_layer_projection.weight)
    h = __rms_norm(h, @post_per_layer_input_norm.weight)
    x = __add(x, h)
    return x
}

func gemma_local_kv_attention(x Node) (Node, Node, Node) {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __reshape(q, shape=[-1, 16, 8, 256])
        k = __reshape(k, shape=[-1, 16, 1, 256])
        v = __reshape(v, shape=[-1, 16, 1, 256])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        q = __rms_norm(q, @q_norm.weight)
        k = __rms_norm(k, @k_norm.weight)
        v = __rms_norm(v)

        q = __rope(q, rope_theta=10000.0, split_half=1)
        k = __rope(k, rope_theta=10000.0, split_half=1)

        scores := __sliding_query_key_score(q, k, window=512, scale=1.0)
        probs := __softmax(scores, dim=-1)
        ctx := __matmul(probs, v)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, 16, 2048])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out, k, v
}

func gemma_global_kv_attention(x Node) (Node, Node, Node) {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        k := __matmul(x, __transpose(@k_proj.weight))
        v := __matmul(x, __transpose(@v_proj.weight))

        q = __reshape(q, shape=[-1, 16, 8, 512])
        k = __reshape(k, shape=[-1, 16, 1, 512])
        v = __reshape(v, shape=[-1, 16, 1, 512])

        q = __permute(q, dims=[0, 2, 1, 3])
        k = __permute(k, dims=[0, 2, 1, 3])
        v = __permute(v, dims=[0, 2, 1, 3])

        q = __rms_norm(q, @q_norm.weight)
        k = __rms_norm(k, @k_norm.weight)
        v = __rms_norm(v)

        q = __rope(q, rope_theta=1000000.0, rotary_dim=128, split_half=1)
        k = __rope(k, rope_theta=1000000.0, rotary_dim=128, split_half=1)

        scores := __sliding_query_key_score(q, k, window=0, scale=1.0)
        probs := __softmax(scores, dim=-1)
        ctx := __matmul(probs, v)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, 16, 4096])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out, k, v
}

func gemma_local_attention(x Node, k Node, v Node) Node {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        q = __reshape(q, shape=[-1, 16, 8, 256])
        q = __permute(q, dims=[0, 2, 1, 3])
        q = __rms_norm(q, @q_norm.weight)
        q = __rope(q, rope_theta=10000.0, split_half=1)

        scores := __sliding_query_key_score(q, k, window=512, scale=1.0)
        probs := __softmax(scores, dim=-1)
        ctx := __matmul(probs, v)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, 16, 2048])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}

func gemma_global_attention(x Node, k Node, v Node) Node {
    weight_scope "self_attn" {
        q := __matmul(x, __transpose(@q_proj.weight))
        q = __reshape(q, shape=[-1, 16, 8, 512])
        q = __permute(q, dims=[0, 2, 1, 3])
        q = __rms_norm(q, @q_norm.weight)
        q = __rope(q, rope_theta=1000000.0, rotary_dim=128, split_half=1)

        scores := __sliding_query_key_score(q, k, window=0, scale=1.0)
        probs := __softmax(scores, dim=-1)
        ctx := __matmul(probs, v)

        ctx = __permute(ctx, dims=[0, 2, 1, 3])
        ctx = __reshape(ctx, shape=[-1, 16, 4096])

        out := __matmul(ctx, __transpose(@o_proj.weight))
    }
    return out
}

func gemma_local_kv_layer(x Node, per_layer_input Node, i int) (Node, Node, Node) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h, k, v := gemma_local_kv_attention(h)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = gemma_gated_mlp(h)
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)

        x = gemma_apply_per_layer_input(x, per_layer_input)
        x = __mul(x, @skip_scale)
    }
    return x, k, v
}

func gemma_global_kv_layer(x Node, per_layer_input Node, i int) (Node, Node, Node) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h, k, v := gemma_global_kv_attention(h)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = gemma_gated_mlp(h)
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)

        x = gemma_apply_per_layer_input(x, per_layer_input)
        x = __mul(x, @skip_scale)
    }
    return x, k, v
}

func gemma_local_layer(x Node, per_layer_input Node, i int, k Node, v Node) Node {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = gemma_local_attention(h, k, v)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = gemma_gated_mlp(h)
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)

        x = gemma_apply_per_layer_input(x, per_layer_input)
        x = __mul(x, @skip_scale)
    }
    return x
}

func gemma_global_layer(x Node, per_layer_input Node, i int, k Node, v Node) Node {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = gemma_global_attention(h, k, v)
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = gemma_gated_mlp(h)
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)

        x = gemma_apply_per_layer_input(x, per_layer_input)
        x = __mul(x, @skip_scale)
    }
    return x
}

func main(input_ids Node) Node {
    weight_scope "language_model.model" {
        x := __embedding(input_ids, @embed_tokens.weight)
        x = __mul(x, 39.25)
        input_embed := x

        var sliding_k Node
        var sliding_v Node
        var full_k Node
        var full_v Node

        for i := range(15) {
            per_layer_input := gemma_per_layer_input(input_embed, input_ids, i)
            if i % 5 == 4 {
                x, full_k, full_v = gemma_global_kv_layer(x, per_layer_input, i)
            } else {
                x, sliding_k, sliding_v = gemma_local_kv_layer(x, per_layer_input, i)
            }
        }

        for i := range(15, 35) {
            per_layer_input := gemma_per_layer_input(input_embed, input_ids, i)
            if i % 5 == 4 {
                x = gemma_global_layer(x, per_layer_input, i, full_k, full_v)
            } else {
                x = gemma_local_layer(x, per_layer_input, i, sliding_k, sliding_v)
            }
        }

        x = __rms_norm(x, @norm.weight)
        logits := __matmul(x, __transpose(@embed_tokens.weight))
        logits = __softcap(logits, 30.0)
    }
    return logits
}
