add_executable(sandy_grpc_worker
  HostTensorBuffer.cpp
  Model.cpp
  Sampler.cpp
  SandyInferenceService.cpp
  SandyWorkerMain.cpp
  Session.cpp
)

target_include_directories(sandy_grpc_worker
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sandy_grpc_worker
  PRIVATE
  SandyInferenceProto
  SandyCompilerLib
  EngineLib
  WeightLib
  ${SANDY_GRPC_GRPCPP}
)

if(SANDY_ENABLE_CUDA)
  target_compile_definitions(sandy_grpc_worker
    PRIVATE
    SANDY_SERVER_ENABLE_CUDA
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
      OUTPUT_VARIABLE SANDY_ACTIVE_LIBSTDCXX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS "${SANDY_ACTIVE_LIBSTDCXX}")
      get_filename_component(SANDY_ACTIVE_LIBSTDCXX_DIR "${SANDY_ACTIVE_LIBSTDCXX}" DIRECTORY)
      target_link_options(sandy_grpc_worker
        BEFORE PRIVATE
        "-L${SANDY_ACTIVE_LIBSTDCXX_DIR}"
      )
    endif()
  endif()
endif()

set_target_properties(sandy_grpc_worker PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
