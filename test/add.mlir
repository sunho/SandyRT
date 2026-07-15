func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
