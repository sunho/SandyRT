add_library(SandyGoLib
  Lexer.cpp
  Parser.cpp
  Interpreter.cpp
)

target_include_directories(SandyGoLib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SandyGoLib PUBLIC IRLib)

set_target_properties(SandyGoLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
