add_library(SandyCompilerLib
  Compiler.cpp
)

target_include_directories(SandyCompilerLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SandyCompilerLib
  PUBLIC
  IRLib
  SandyGoLib
  WeightLib
)

set_target_properties(SandyCompilerLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
