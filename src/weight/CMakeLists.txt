add_library(WeightLib
  SafeTensorWeights.cpp
)

target_include_directories(WeightLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(WeightLib
  PUBLIC CoreLib
  PRIVATE nlohmann_json
)

set_target_properties(WeightLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
