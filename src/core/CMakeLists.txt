add_library(CoreLib
  Tensor.cpp
  TensorBuffer.cpp
  TensorCalc.cpp
  ShapeUtil.cpp
)

target_include_directories(CoreLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(CoreLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
