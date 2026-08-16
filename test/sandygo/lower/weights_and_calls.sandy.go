// RUN: %sandygo_lower %s | FileCheck %s
// WEIGHT: fc1.weight f32 [4, 3]
// WEIGHT: fc1.bias f32 [4]
// WEIGHT: fc2.weight f32 [2, 4]
// WEIGHT: fc2.bias f32 [2]

func layer1(x Tensor) Tensor {
    weight_scope "fc1" {
        x = __linear(x, @weight, @bias)
        return __relu(x)
    }
}

func layer2(x Tensor) Tensor {
    weight_scope "fc2" {
        return __linear(x, @weight, @bias)
    }
}

func main(x Tensor[[5, 3], f32]) Tensor {
    x = layer1(x)
    x = layer2(x)
    return x
}

// CHECK: %0 = input(index=0) : f32[5, 3]
// CHECK-NEXT: %1 = weight(name="fc1.weight") : f32[4, 3]
// CHECK-NEXT: %2 = weight(name="fc1.bias") : f32[4]
// CHECK-NEXT: %3 = linear(%0, %1, %2) : f32[5, 4]
// CHECK-NEXT: %4 = relu(%3) : f32[5, 4]
// CHECK-NEXT: %5 = weight(name="fc2.weight") : f32[2, 4]
// CHECK-NEXT: %6 = weight(name="fc2.bias") : f32[2]
// CHECK-NEXT: %7 = linear(%4, %5, %6) : f32[5, 2]
// CHECK-NEXT: return %7
