add_library(CoreLib
  Allocator.cpp
  CacheKey.cpp
  Tensor.cpp
  TensorBuffer.cpp
  TensorCalc.cpp
  ShapeUtil.cpp
)

target_include_directories(CoreLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

option(SANDY_MATMUL_FAST "Enable Accelerate-backed matmul fast path where available" OFF)
option(SANDY_MATMUL_CUBLAS "Enable cuBLAS-backed matmul fast path where available" OFF)

if(SANDY_MATMUL_FAST)
  target_compile_definitions(CoreLib PUBLIC SANDY_MATMUL_FAST=1)
  if(APPLE)
    target_link_libraries(CoreLib PUBLIC "-framework Accelerate")
  endif()
endif()

if(SANDY_MATMUL_CUBLAS)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(CoreLib PUBLIC SANDY_MATMUL_CUBLAS=1)
  target_link_libraries(CoreLib PUBLIC CUDA::cudart CUDA::cublas)
endif()

set_target_properties(CoreLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
