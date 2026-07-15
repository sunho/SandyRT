add_library(SandyCompilerLib
  Compiler.cpp
)

target_include_directories(SandyCompilerLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SandyCompilerLib
  PUBLIC
  StablehloRegister
  StablehloOps
  MLIRFuncDialect
  MLIRParser
  MLIRIR
  MLIRSupport
)
