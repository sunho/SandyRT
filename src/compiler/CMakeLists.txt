add_library(SandyRTLib
  Compiler.cpp
)

target_include_directories(SandyRTLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SandyRTLib
  PUBLIC
  IRLib
  SandyGoLib
  WeightLib
)

set_target_properties(SandyRTLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
