func gemma_kv_layer(x Node, i int, window int, head_dim int, rope_theta float) (Node, Node) {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h, kv := __kv_attention(h,
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
        h = __gated_mlp(h,
            @mlp.gate_proj.weight,
            @mlp.up_proj.weight,
            @mlp.down_proj.weight,
            act="gelu",
        )
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)
    }
    return x, kv
}

func gemma_layer(x Node, i int, kv Node, window int, head_dim int, rope_theta float) Node {
    weight_scope "layers.{i}" {
        h := __rms_norm(x, @input_layernorm.weight)
        h = __attention(h, kv,
            @self_attn.q_proj.weight,
            @self_attn.o_proj.weight,
            heads=8, kv_heads=1,
            head_dim=head_dim, window=window, rope_theta=rope_theta,
        )
        h = __rms_norm(h, @post_attention_layernorm.weight)
        x = __add(x, h)

        h = __rms_norm(x, @pre_feedforward_layernorm.weight)
        h = __gated_mlp(h,
            @mlp.gate_proj.weight,
            @mlp.up_proj.weight,
            @mlp.down_proj.weight,
            act="gelu",
        )
        h = __rms_norm(h, @post_feedforward_layernorm.weight)
        x = __add(x, h)
    }
    return x
}

func main(input_ids Node) Node {
    weight_scope "language_model.model" {
        x := __embedding(input_ids, @embed_tokens.weight)
        x = __mul(x, __sqrt(1536))

        var sliding_kv Node
        var full_kv Node

        for i := range(15) {
            if i % 5 == 4 {
                x, full_kv = gemma_kv_layer(x, i, 0, 512, 1000000.0)
            } else {
                x, sliding_kv = gemma_kv_layer(x, i, 512, 256, 10000.0)
            }
        }

        for i := range(15, 35) {
            if i % 5 == 4 {
                x = gemma_layer(x, i, full_kv, 0, 512, 1000000.0)
            } else {
                x = gemma_layer(x, i, sliding_kv, 512, 256, 10000.0)
            }
        }

        x = __rms_norm(x, @norm.weight)
        logits := __matmul(x, __transpose(@embed_tokens.weight))
        logits = __softcap(logits, 30.0)
    }
    return logits
}
