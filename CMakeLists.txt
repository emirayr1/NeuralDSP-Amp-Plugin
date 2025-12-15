cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(neuralAmp)

# Ensure x64 build
set(CMAKE_GENERATOR_PLATFORM x64)

set(CMAKE_PREFIX_PATH "../../../LIBTORCH/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Fix for MSVC intrinsics error and warnings
if(MSVC)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_compile_options(/bigobj)
  # Disable size_t conversion warnings common with LibTorch
  add_compile_options(/wd4267 /wd4244)
endif()

add_executable(neuralAmp src/main.cpp)
target_link_libraries(neuralAmp "${TORCH_LIBRARIES}")
set_property(TARGET neuralAmp PROPERTY CXX_STANDARD 17)
