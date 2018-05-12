cmake_minimum_required(VERSION 3.8)
#define the project
project(neuralnet)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wunused-parameter
  -Wc++11-extra-semi
  )
endif()

#add definition for `neuralnet` binary
add_executable(neuralnet src/neuralnet.tuts.cpp)


