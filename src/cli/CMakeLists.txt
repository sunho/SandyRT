add_executable(sandy main.cpp)

target_link_libraries(sandy
  PRIVATE
  SandyCompilerLib
  WeightLib
)

set_target_properties(sandy PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
