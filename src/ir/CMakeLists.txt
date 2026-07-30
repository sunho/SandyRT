add_library(IRLib
  HighIR.cpp
  Shape.cpp
)

target_include_directories(IRLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(IRLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
