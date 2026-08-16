// RUN: %sandygo_lower %s | FileCheck %s

func main(x [2]Tensor[[3], bf16]) []Tensor {
    var out []Tensor
    out = append(out, x[0])
    out = append(out, __relu(x[1]))
    return out
}

// CHECK-NOT: tensor_tuple_get
// CHECK: %0 = input({{(?:tuple_element=0, index=0|index=0, tuple_element=0)}}) : bf16[3]
// CHECK-NEXT: %1 = input({{(?:tuple_element=1, index=0|index=0, tuple_element=1)}}) : bf16[3]
// CHECK-NEXT: %2 = relu(%1) : bf16[3]
// CHECK-NEXT: %3 = tensor_tuple_create(%0, %2) : tensor_tuple<2>
// CHECK-NEXT: return %3
