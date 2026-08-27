add_library(EngineLib
  Engine.cpp
  InvocationCacheKey.cpp
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
  absl::flat_hash_map
)

set_target_properties(EngineLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
