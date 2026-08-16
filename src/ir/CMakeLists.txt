add_library(IRLib
  BuiltinLowering.cpp
  HighIR.cpp
  KernelIR.cpp
  MidIR.cpp
  MidIRMaterializer.cpp
  MidIRPass.cpp
  MidIRToKernelIR.cpp
)

target_include_directories(IRLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../weight
)

target_link_libraries(IRLib
  PUBLIC CoreLib
)

set_target_properties(IRLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
