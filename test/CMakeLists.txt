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

add_executable(load_safetensors
  weight/LoadSafeTensorsTest.cpp
)

target_link_libraries(load_safetensors PRIVATE WeightLib)

set_target_properties(load_safetensors PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(mid_ir_tests
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

set_target_properties(materialize_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(materialize_tests)
