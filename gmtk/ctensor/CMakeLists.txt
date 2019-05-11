cmake_minimum_required(VERSION 3.9)
project(ctensor)

set(CMAKE_CXX_STANDARD 11)

INCLUDE_DIRECTORIES($ENV{ANACONDA_HOME}/lib/python3.6/site-packages/torch/lib/include)
LINK_DIRECTORIES($ENV{ANACONDA_HOME}/lib/python3.6/site-packages/torch/lib $ENV{ANACONDA_HOME}/lib)

add_library(ctensor SHARED dense_tensor.cpp dense_tensor.h)
target_link_libraries(ctensor ATen)
add_executable(debug main.cpp dense_tensor.cpp dense_tensor.h)
target_link_libraries(debug ATen)