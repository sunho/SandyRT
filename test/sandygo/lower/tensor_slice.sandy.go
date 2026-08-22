// RUN: %sandygo_lower %s | FileCheck %s

func main(x Tensor[[1, -1, 4], f32]) Tensor {
    return x[:, -1, :]
}

// CHECK: %0 = input(index=0) : f32[1, ?, 4]
// CHECK-NEXT: %1 = slice(%0, indices=[0, -1, 0], kinds=[0, 1, 0]) : f32[1, 4]
// CHECK-NEXT: return %1
