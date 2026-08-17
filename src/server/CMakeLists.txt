add_executable(sandy_grpc_worker
  GemmaModel.cpp
  GemmaSession.cpp
  HostTensorBuffer.cpp
  Sampler.cpp
  SandyInferenceService.cpp
  SandyWorkerMain.cpp
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

set_target_properties(sandy_grpc_worker PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
