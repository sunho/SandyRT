add_library(EngineLib
  Engine.cpp
  RuntimeScratchPlan.cpp
  RuntimeTensorDesc.cpp
)

target_include_directories(EngineLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(EngineLib
  PUBLIC
  CoreLib
  IRLib
  DeviceLib
)

set_target_properties(EngineLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
