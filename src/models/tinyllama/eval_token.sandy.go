import "common.sandy.go"

func main(input_id Tensor[[1, 1], i64], position_id Tensor[[1], i64], k_cache [22]PagedTensor[[1, 4, -1, 64], bf16, page_size=16], v_cache [22]PagedTensor[[1, 4, -1, 64], bf16, page_size=16]) Tensor {
    weight_scope "model" {
        x := __embedding(input_id, @embed_tokens.weight)

        for i := range(22) {
            x = tinyllama_eval_block(x, position_id, i, k_cache[i], v_cache[i])
        }

        x = __rms_norm(x, @norm.weight, epsilon=0.00001)
        logits := __matmul(x, __transpose(@lm_head.weight))
    }
    return logits
}
