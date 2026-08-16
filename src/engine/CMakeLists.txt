add_library(EngineLib
  Engine.cpp
  DeviceWiseCopier.cpp
  CpuDevice.cpp
  MidIRCpuInterpreter.cpp
)

if(SANDY_ENABLE_CUDA)
  target_sources(EngineLib
    PRIVATE
    CudaRuntime.cpp
    CudaDevice.cpp
    CudaKernels.cu
  )
endif()

target_include_directories(EngineLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(EngineLib
  PUBLIC
  CoreLib
  IRLib
)

if(SANDY_ENABLE_CUDA)
  target_link_libraries(EngineLib
    PUBLIC
    CUDA::cudart
  )
endif()

set_target_properties(EngineLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

if(SANDY_ENABLE_CUDA)
  set_target_properties(EngineLib PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  set_source_files_properties(CudaKernels.cu PROPERTIES
    LANGUAGE CUDA
  )
endif()
