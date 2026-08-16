cmake_minimum_required(VERSION 3.20)
project(SandyCompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SANDY_ENABLE_CUDA "Build Sandy CUDA backend" OFF)

if(SANDY_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(external/json)
add_subdirectory(src/core)
add_subdirectory(src/ir)
add_subdirectory(src/sandygo)
add_subdirectory(src/weight)
add_subdirectory(src/compiler)
add_subdirectory(src/device)
add_subdirectory(src/engine)
add_subdirectory(src/cli)

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
    SandyCompilerLib
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
    SandyCompilerLib
    EngineLib
    WeightLib
  )

  set_target_properties(multi_gemma4_runner PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )
endif()
