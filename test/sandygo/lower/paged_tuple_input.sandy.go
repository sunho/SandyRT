// RUN: %sandygo_lower %s | FileCheck %s

func main(k [2]PagedTensor[[128], bf16, page_size=16]) []Tensor {
    var out []Tensor
    out = append(out, k[0])
    out = append(out, k[1])
    return out
}

// CHECK-NOT: tensor_tuple_get
// CHECK: %0 = paged_tensor_input({{.*(?:tuple_element=0.*index=0|index=0.*tuple_element=0).*}}) : bf16[128]
// CHECK-NEXT: %1 = paged_tensor_input({{.*(?:tuple_element=1.*index=0|index=0.*tuple_element=1).*}}) : bf16[128]
// CHECK-NEXT: %2 = tensor_tuple_create(%0, %1) : tensor_tuple<2>
// CHECK-NEXT: return %2
