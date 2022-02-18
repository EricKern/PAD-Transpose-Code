cmake_minimum_required(VERSION 3.20)
project(myProj)

find_package(benchmark REQUIRED)

## PAD Library Interface
add_subdirectory(src)

### mp-media
## memory performance benchmark
add_executable(dataV2Driver dataV2Driver.cpp)
target_compile_features(dataV2Driver PRIVATE cxx_std_20)
target_link_libraries(dataV2Driver PRIVATE benchmark::benchmark pad)
target_compile_options(
  dataV2Driver
  PRIVATE
    -Wall
    -march=core-avx2
    -mprefer-vector-width=256
    -static
    -O3
)
