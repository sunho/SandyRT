get_filename_component(SANDY_INFERENCE_PROTO
  "${CMAKE_CURRENT_SOURCE_DIR}/sandy_inference.proto"
  ABSOLUTE)
get_filename_component(SANDY_INFERENCE_PROTO_PATH
  "${SANDY_INFERENCE_PROTO}"
  PATH)

set(SANDY_INFERENCE_PROTO_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/sandy_inference.pb.cc")
set(SANDY_INFERENCE_PROTO_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/sandy_inference.pb.h")
set(SANDY_INFERENCE_GRPC_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/sandy_inference.grpc.pb.cc")
set(SANDY_INFERENCE_GRPC_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/sandy_inference.grpc.pb.h")

add_custom_command(
  OUTPUT
    "${SANDY_INFERENCE_PROTO_SRCS}"
    "${SANDY_INFERENCE_PROTO_HDRS}"
    "${SANDY_INFERENCE_GRPC_SRCS}"
    "${SANDY_INFERENCE_GRPC_HDRS}"
  COMMAND ${SANDY_PROTOBUF_PROTOC}
  ARGS
    --experimental_allow_proto3_optional
    --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${SANDY_INFERENCE_PROTO_PATH}"
    --plugin=protoc-gen-grpc="${SANDY_GRPC_CPP_PLUGIN}"
    "${SANDY_INFERENCE_PROTO}"
  DEPENDS "${SANDY_INFERENCE_PROTO}")

add_library(SandyInferenceProto
  ${SANDY_INFERENCE_GRPC_SRCS}
  ${SANDY_INFERENCE_GRPC_HDRS}
  ${SANDY_INFERENCE_PROTO_SRCS}
  ${SANDY_INFERENCE_PROTO_HDRS}
)

target_include_directories(SandyInferenceProto
  PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${SANDY_PROTOBUF_INCLUDE_DIRS}
)

target_link_libraries(SandyInferenceProto
  PUBLIC
  ${SANDY_GRPC_GRPCPP}
  ${SANDY_PROTOBUF_LIBPROTOBUF}
)

set_target_properties(SandyInferenceProto PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
