import "common.sandy.go"

func main(input_id Tensor[[1], i64], position_id Tensor[[1], i64], local_k_cache [20]PagedTensor[[1, 2, -1, 256], bf16, page_size=16], local_v_cache [20]PagedTensor[[1, 2, -1, 256], bf16, page_size=16], global_k_cache [4]PagedTensor[[1, 2, -1, 512], bf16, page_size=16], global_v_cache [4]PagedTensor[[1, 2, -1, 512], bf16, page_size=16]) ([]Tensor, []Tensor, Tensor) {
    weight_scope "language_model.model" {
        x := __embedding(input_id, @embed_tokens.weight)
        x = __mul(x, 50.5)
        input_embed := x

        var next_k []Tensor
        var next_v []Tensor

        var sliding_k Tensor
        var sliding_v Tensor
        var full_k Tensor
        var full_v Tensor

        local_i := 0
        global_i := 0

        for i := range(24) {
            per_layer_input := gemma_per_layer_input(input_embed, input_id, i)
            if i % 6 == 5 {
                x, full_k, full_v = gemma_global_eval_kv_layer(x, per_layer_input, position_id, i, global_k_cache[global_i], global_v_cache[global_i])
                next_k = append(next_k, full_k)
                next_v = append(next_v, full_v)
                global_i = global_i + 1
            } else {
                x, sliding_k, sliding_v = gemma_local_eval_kv_layer(x, per_layer_input, position_id, i, local_k_cache[local_i], local_v_cache[local_i])
                next_k = append(next_k, sliding_k)
                next_v = append(next_v, sliding_v)
                local_i = local_i + 1
            }
        }

        cached_sliding_k := local_k_cache[19]
        cached_sliding_v := local_v_cache[19]
        cached_full_k := global_k_cache[3]
        cached_full_v := global_v_cache[3]

        for i := range(24, 42) {
            per_layer_input := gemma_per_layer_input(input_embed, input_id, i)
            if i % 6 == 5 {
                x = gemma_global_eval_layer(x, per_layer_input, position_id, i, cached_full_k, cached_full_v)
            } else {
                x = gemma_local_eval_layer(x, per_layer_input, position_id, i, cached_sliding_k, cached_sliding_v)
            }
        }

        x = __rms_norm(x, @norm.weight)
        logits := __matmul(x, __transpose(@embed_tokens.weight))
        logits = __softcap(logits, 30.0)

        return next_k, next_v, logits
    }
}
