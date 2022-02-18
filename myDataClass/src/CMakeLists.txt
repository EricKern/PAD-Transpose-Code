# pad c++ library

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

add_library(pad INTERFACE)

target_compile_features(pad INTERFACE cxx_std_20)

target_include_directories(
  pad
  INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)
target_link_libraries(
  pad
  INTERFACE TBB::tbb Threads::Threads OpenMP::OpenMP_CXX
)