cmake_minimum_required(VERSION 3.20)
project(SandyRT LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SANDY_ENABLE_CUDA "Build Sandy CUDA backend" OFF)
option(SANDY_ENABLE_SERVER "Build Sandy gRPC inference server" OFF)
option(SANDY_USE_BUNDLED_GRPC "Use external/grpc for the server build" OFF)

if(SANDY_ENABLE_CUDA)
  enable_language(CUDA)
  get_filename_component(SANDY_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(SANDY_CUDA_TOOLKIT_ROOT "${SANDY_CUDA_BIN_DIR}" DIRECTORY)
  if(EXISTS "${SANDY_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib/libcudart.so")
    set(CUDAToolkit_ROOT "${SANDY_CUDA_TOOLKIT_ROOT}" CACHE PATH "" FORCE)
    set(CUDA_CUDART
        "${SANDY_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib/libcudart.so"
        CACHE FILEPATH "" FORCE)
    set(CUDA_cudart_LIBRARY
        "${SANDY_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib/libcudart.so"
        CACHE FILEPATH "" FORCE)
    set(CUDA_cublas_LIBRARY
        "${SANDY_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib/libcublas.so"
        CACHE FILEPATH "" FORCE)
    set(CUDA_cublasLt_LIBRARY
        "${SANDY_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib/libcublasLt.so"
        CACHE FILEPATH "" FORCE)
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(external/json)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL OFF)
set(ABSL_BUILD_TESTING OFF)
add_subdirectory(external/abseil-cpp EXCLUDE_FROM_ALL)
add_subdirectory(src/core)
add_subdirectory(src/ir)
add_subdirectory(src/sandygo)
add_subdirectory(src/weight)
add_subdirectory(src/compiler)
add_subdirectory(src/device)
add_subdirectory(src/engine)
add_subdirectory(src/cli)

if(SANDY_ENABLE_SERVER)
  find_package(Threads REQUIRED)

  if(SANDY_USE_BUNDLED_GRPC)
    set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_CODEGEN ON CACHE BOOL "" FORCE)
    set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
    set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
    set(gRPC_ABSL_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_CARES_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_PROTOBUF_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_RE2_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_SSL_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_ZLIB_PROVIDER "module" CACHE STRING "" FORCE)
    add_subdirectory(external/grpc EXCLUDE_FROM_ALL)

    set(SANDY_PROTOBUF_LIBPROTOBUF libprotobuf)
    set(SANDY_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
    set(SANDY_GRPC_GRPCPP grpc++)
    set(SANDY_GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)
  else()
    find_package(Protobuf CONFIG QUIET)
    if(NOT Protobuf_FOUND)
      find_package(Protobuf REQUIRED)
    endif()

    if(TARGET protobuf::libprotobuf)
      set(SANDY_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
    else()
      set(SANDY_PROTOBUF_LIBPROTOBUF ${Protobuf_LIBRARIES})
      set(SANDY_PROTOBUF_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS})
    endif()

    if(TARGET protobuf::protoc)
      set(SANDY_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
    elseif(Protobuf_PROTOC_EXECUTABLE)
      set(SANDY_PROTOBUF_PROTOC ${Protobuf_PROTOC_EXECUTABLE})
    else()
      message(FATAL_ERROR "Could not find protoc for installed Protobuf")
    endif()

    find_package(gRPC CONFIG QUIET)
    if(gRPC_FOUND)
      set(SANDY_GRPC_GRPCPP gRPC::grpc++)
      set(SANDY_GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    else()
      find_package(PkgConfig REQUIRED)
      pkg_check_modules(GRPC REQUIRED IMPORTED_TARGET grpc++)
      find_program(SANDY_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
      set(SANDY_GRPC_GRPCPP PkgConfig::GRPC)
    endif()
  endif()

  add_subdirectory(proto)
  add_subdirectory(src/server)
endif()

option(SANDY_BUILD_TESTS "Build Sandy test binaries" ON)

if(SANDY_BUILD_TESTS)
  # GoogleTest
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/googletest/CMakeLists.txt")
    add_subdirectory(external/googletest)
  else()
    find_package(GTest CONFIG QUIET)
    if(NOT TARGET GTest::gtest_main)
      find_package(GTest MODULE QUIET)
    endif()
    if(NOT TARGET GTest::gtest_main)
      include(FetchContent)
      FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
      )
      FetchContent_MakeAvailable(googletest)
    endif()
  endif()

  enable_testing()
  add_subdirectory(test)
else()
  add_executable(cpu_runner
    test/engine/CpuRunner.cpp
  )

  target_link_libraries(cpu_runner
    PRIVATE
    SandyRTLib
    EngineLib
    WeightLib
  )

  set_target_properties(cpu_runner PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  add_executable(multi_gemma4_runner
    test/engine/MultiGemmaRunner.cpp
  )

  target_link_libraries(multi_gemma4_runner
    PRIVATE
    SandyRTLib
    EngineLib
    WeightLib
  )

  set_target_properties(multi_gemma4_runner PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )
endif()
