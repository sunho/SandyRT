// RUN: %sandygo_lower %s | FileCheck %s

func main(k [2]Tensor[[128], bf16]) Tensor {
    return k[0]
}

// CHECK-NOT: tensor_tuple_get
// CHECK: %0 = input({{(?:tuple_element=0, index=0|index=0, tuple_element=0)}}) : bf16[128]
// CHECK-NEXT: return %0
