import "common.sandy.go"

config const TOP_K int

func main(input_id Tensor[[1, -1], i64], position_id Tensor[[1], i64], local_k_cache [25]PagedTensor[[1, 8, -1, 256], bf16, page_size=32], local_v_cache [25]PagedTensor[[1, 8, -1, 256], bf16, page_size=32], global_k_cache [5]PagedTensor[[1, 2, -1, 512], bf16, page_size=32], global_v_cache [5]PagedTensor[[1, 2, -1, 512], bf16, page_size=32]) (Tensor, Tensor) {
    weight_scope "language_model.model" {
        x := __embedding(input_id, @embed_tokens.weight)
        x = __mul(x, 53.0)

        local_i := 0
        global_i := 0

        for i := range(30) {
            if i % 6 == 5 {
                x = gemma_global_eval_kv_layer(x, position_id, i, global_k_cache[global_i], global_v_cache[global_i])
                global_i = global_i + 1
            } else {
                x = gemma_local_eval_kv_layer(x, position_id, i, local_k_cache[local_i], local_v_cache[local_i])
                local_i = local_i + 1
            }
        }

        x = __rms_norm(x, @norm.weight)
        x = x[:, -1, :]
        logits := __matmul(x, __transpose(@embed_tokens.weight))
        logits = __softcap(logits, 30.0)
        topk_values, topk_ids := __topk(logits, k=TOP_K, dim=-1)

        return topk_values, topk_ids
    }
}
