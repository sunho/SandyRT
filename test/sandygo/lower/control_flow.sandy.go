// RUN: %sandygo_lower %s | FileCheck %s

func block(x Tensor, i int) Tensor {
    if i == 1 {
        return __relu(x)
    }
    return __tanh(x)
}

func main(x Tensor[[4], f32]) Tensor {
    for i := range(3) {
        x = block(x, i)
    }
    return x
}

// CHECK: %0 = input(index=0) : f32[4]
// CHECK-NEXT: %1 = tanh(%0) : f32[4]
// CHECK-NEXT: %2 = relu(%1) : f32[4]
// CHECK-NEXT: %3 = tanh(%2) : f32[4]
// CHECK-NEXT: return %3
