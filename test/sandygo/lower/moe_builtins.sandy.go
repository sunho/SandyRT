// RUN: %sandygo_lower %s | FileCheck %s
// INPUT: x f32 [2, 4]
// WEIGHT: expert.weight f32 [4, 4, 4]

func main(x Tensor) Tensor {
    probs := __softmax(x, dim=-1)
    values, ids := __topk(probs, k=2, dim=-1)
    denom := __sum(values, dim=-1, keepdim=1)
    values = __div(values, denom)
    packed_x, packed_weights, token_ids, offsets := __moe_gather(
        x,
        ids,
        values,
        num_experts=4,
        top_k=2)
    projected := __moe_matmul(packed_x, offsets, @expert.weight, transpose_rhs=1)
    return __moe_scatter_sum(projected, packed_weights, token_ids, x)
}

// CHECK: softmax
// CHECK: topk
// CHECK: sum
// CHECK: div
// CHECK: moe_gather
// CHECK: moe_matmul
// CHECK: moe_scatter_sum
