func gemma_gated_mlp(x Node) Node {
    weight_scope "mlp" {
        gate := __matmul(x, __transpose(@gate_proj.weight))
        up := __matmul(x, __transpose(@up_proj.weight))
        x = __mul(__gelu(gate), up)
        x = __matmul(x, __transpose(@down_proj.weight))
    }
    return x
}

func gemma_kv_layer(x Node, i int, window int, head_dim int, rope_theta float) (Node, Node, Node) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h, k, v := __kv_attention(h,
            @self_attn.q_proj.weight,
            @self_attn.k_proj.weight,
            @self_attn.v_proj.weight,
            @self_attn.o_proj.weight,
            heads=8, kv_heads=1,
            head_dim=head_dim, window=window, rope_theta=rope_theta,
        )
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = gemma_gated_mlp(h)
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)
    }
    return x, k, v
}

func gemma_layer(x Node, i int, k Node, v Node, window int, head_dim int, rope_theta float) Node {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = __attention(h, k, v,
            @self_attn.q_proj.weight,
            @self_attn.o_proj.weight,
            heads=8, kv_heads=1,
            head_dim=head_dim, window=window, rope_theta=rope_theta,
        )
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = gemma_gated_mlp(h)
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)
    }
    return x
}

func main(input_ids Node) Node {
    weight_scope "language_model.model" {
        x := __embedding(input_ids, @embed_tokens.weight)
        x = __mul(x, __sqrt(1536))

        var sliding_k Node
        var sliding_v Node
        var full_k Node
        var full_v Node

        for i := range(15) {
            if i % 5 == 4 {
                x, full_k, full_v = gemma_kv_layer(x, i, 0, 512, 1000000.0)
            } else {
                x, sliding_k, sliding_v = gemma_kv_layer(x, i, 512, 256, 10000.0)
            }
        }

        for i := range(15, 35) {
            if i % 5 == 4 {
                x = gemma_layer(x, i, full_k, full_v, 0, 512, 1000000.0)
            } else {
                x = gemma_layer(x, i, sliding_k, sliding_v, 512, 256, 10000.0)
            }
        }

        x = __rms_norm(x, @norm.weight)
        logits := __matmul(x, __transpose(@embed_tokens.weight))
        logits = __softcap(logits, 30.0)
    }
    return logits
}
