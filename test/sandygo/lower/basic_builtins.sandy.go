// RUN: %sandygo_lower %s | FileCheck %s

func main(x Tensor[[2, 3], f32]) Tensor {
    y := __relu(x)
    y = __add(y, x)
    y = __mul(y, 2.0)
    y = __sqrt(y)
    y = __tanh(y)
    return y
}

// CHECK: %0 = input(index=0) : f32[2, 3]
// CHECK-NEXT: %1 = relu(%0) : f32[2, 3]
// CHECK-NEXT: %2 = add(%1, %0) : f32[2, 3]
// CHECK-NEXT: %3 = constant(value=2) : f32[]
// CHECK-NEXT: %4 = mul(%2, %3) : f32[2, 3]
// CHECK-NEXT: %5 = sqrt(%4) : f32[2, 3]
// CHECK-NEXT: %6 = tanh(%5) : f32[2, 3]
// CHECK-NEXT: return %6
