add_executable(sandygo_tests
  sandygo/LexerTest.cpp
  sandygo/ParserTest.cpp
)

target_link_libraries(sandygo_tests
  PRIVATE
  SandyGoLib
  GTest::gtest_main
)

set_target_properties(sandygo_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

include(GoogleTest)
gtest_discover_tests(sandygo_tests)

add_executable(dump_ast
  sandygo/DumpAST.cpp
)

target_link_libraries(dump_ast PRIVATE SandyGoLib)

set_target_properties(dump_ast PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(dump_ir
  sandygo/DumpIR.cpp
)

target_link_libraries(dump_ir PRIVATE SandyGoLib)

set_target_properties(dump_ir PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(dump_mid_ir
  sandygo/DumpMidIR.cpp
)

target_link_libraries(dump_mid_ir PRIVATE SandyCompilerLib)

set_target_properties(dump_mid_ir PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(SANDYGO_LOWER_CHECK_COMMAND
  ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/sandygo/sandygo_lower_check.py
  --dump-mid-ir $<TARGET_FILE:dump_mid_ir>
  --test-dir ${CMAKE_CURRENT_SOURCE_DIR}/sandygo/lower
  --cwd ${PROJECT_SOURCE_DIR}
)

add_custom_target(sandygo_lower_check
  COMMAND ${SANDYGO_LOWER_CHECK_COMMAND}
  DEPENDS dump_mid_ir
  USES_TERMINAL
)

add_test(NAME sandygo_lower_check
  COMMAND ${SANDYGO_LOWER_CHECK_COMMAND}
)

add_executable(load_safetensors
  weight/LoadSafeTensorsTest.cpp
)

target_link_libraries(load_safetensors PRIVATE WeightLib)

set_target_properties(load_safetensors PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(mid_ir_tests
  ir/KernelIRTest.cpp
  ir/MidIRTest.cpp
)

target_link_libraries(mid_ir_tests
  PRIVATE
  IRLib
  GTest::gtest_main
)

set_target_properties(mid_ir_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(mid_ir_tests)

add_executable(materialize_tests
  compiler/MaterializeTest.cpp
)

target_link_libraries(materialize_tests
  PRIVATE
  SandyCompilerLib
  GTest::gtest_main
)

target_compile_definitions(materialize_tests
  PRIVATE
  SANDY_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

set_target_properties(materialize_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(materialize_tests)

add_executable(compiler_tests
  compiler/CompilerTest.cpp
)

target_link_libraries(compiler_tests
  PRIVATE
  SandyCompilerLib
  GTest::gtest_main
)

set_target_properties(compiler_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(compiler_tests)

add_executable(engine_tests
  engine/EngineTest.cpp
  engine/EngineCompileTest.cpp
  engine/CpuDeviceTest.cpp
)

target_link_libraries(engine_tests
  PRIVATE
  EngineLib
  GTest::gtest_main
)

set_target_properties(engine_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(engine_tests)

if(SANDY_ENABLE_CUDA)
  add_executable(cuda_device_tests
    engine/CudaDeviceTest.cpp
  )

  target_link_libraries(cuda_device_tests
    PRIVATE
    EngineLib
    GTest::gtest_main
  )

  set_target_properties(cuda_device_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  gtest_discover_tests(cuda_device_tests)
endif()

add_executable(cpu_runner
  engine/CpuRunner.cpp
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
  engine/MultiGemmaRunner.cpp
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
