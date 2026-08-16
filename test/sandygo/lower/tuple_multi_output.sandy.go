// RUN: %sandygo_lower %s | FileCheck %s

func main(k [2]Tensor[[128], bf16], v [2]Tensor[[256], f32]) ([]Tensor, []Tensor) {
    var newK []Tensor
    var newV []Tensor
    newK = append(newK, k[0])
    newK = append(newK, k[1])
    newV = append(newV, v[0])
    newV = append(newV, v[1])
    return newK, newV
}

// CHECK-NOT: tensor_tuple_get
// CHECK: %0 = input({{(?:tuple_element=0, index=0|index=0, tuple_element=0)}}) : bf16[128]
// CHECK-NEXT: %1 = input({{(?:tuple_element=1, index=0|index=0, tuple_element=1)}}) : bf16[128]
// CHECK-NEXT: %2 = input({{(?:tuple_element=0, index=1|index=1, tuple_element=0)}}) : f32[256]
// CHECK-NEXT: %3 = input({{(?:tuple_element=1, index=1|index=1, tuple_element=1)}}) : f32[256]
// CHECK-NEXT: %4 = tensor_tuple_create(%0, %1) : tensor_tuple<2>
// CHECK-NEXT: %5 = tensor_tuple_create(%2, %3) : tensor_tuple<2>
// CHECK-NEXT: return %4, %5
