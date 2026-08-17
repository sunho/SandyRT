add_library(DeviceLib
  Device.cpp
  DeviceWiseCopier.cpp
  cpu/CpuDevice.cpp
  cpu/MidIRCpuInterpreter.cpp
)

if(SANDY_ENABLE_CUDA)
  target_sources(DeviceLib
    PRIVATE
    cuda/CudaRuntime.cpp
    cuda/CudaDevice.cpp
    cuda/CudaPagedTensor.cpp
    cuda/CudaElementwiseKernels.cu
    cuda/CudaGatherKernels.cu
    cuda/CudaKernels.cu
    cuda/CudaMatMulKernels.cu
    cuda/CudaNormKernels.cu
    cuda/CudaSoftmaxKernels.cu
  )
endif()

target_include_directories(DeviceLib
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_link_libraries(DeviceLib
  PUBLIC
  CoreLib
  IRLib
)

if(SANDY_ENABLE_CUDA)
  target_link_libraries(DeviceLib
    PUBLIC
    CUDA::cudart
    CUDA::cublas
  )
endif()

set_target_properties(DeviceLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

if(SANDY_ENABLE_CUDA)
  set_target_properties(DeviceLib PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  set_source_files_properties(
    cuda/CudaElementwiseKernels.cu
    cuda/CudaGatherKernels.cu
    cuda/CudaKernels.cu
    cuda/CudaMatMulKernels.cu
    cuda/CudaNormKernels.cu
    cuda/CudaSoftmaxKernels.cu
    PROPERTIES LANGUAGE CUDA
  )
endif()
