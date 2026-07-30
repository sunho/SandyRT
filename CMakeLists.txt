cmake_minimum_required(VERSION 3.20)
project(SandyCompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM configuration
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
set(LLVM_INSTALL_UTILS ON CACHE BOOL "" FORCE)
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_PCH OFF CACHE BOOL "" FORCE)
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON CACHE BOOL "" FORCE)

add_subdirectory(external/llvm-project/llvm EXCLUDE_FROM_ALL)

# LLVM/MLIR cmake modules (source tree)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/mlir/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set up include paths that StableHLO (embedded) and our code need
set(LLVM_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm/include"
  "${CMAKE_BINARY_DIR}/external/llvm-project/llvm/include"
)
set(MLIR_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/mlir/include"
  "${CMAKE_BINARY_DIR}/external/llvm-project/llvm/tools/mlir/include"
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# StableHLO (embedded mode — we provide LLVM/MLIR include dirs above)
set(STABLEHLO_BUILD_EMBEDDED ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/stablehlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/stablehlo)
add_subdirectory(external/stablehlo)

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
if(NOT LLVM_ENABLE_EH)
  add_compile_options(-fno-exceptions)
endif()

# Override LLVM's C++17 with C++20 for our targets
set(CMAKE_CXX_STANDARD 20 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/json)
add_subdirectory(src/ir)
add_subdirectory(src/sandygo)
add_subdirectory(src/weight)
add_subdirectory(src/compiler)
add_subdirectory(src/cli)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(test)
