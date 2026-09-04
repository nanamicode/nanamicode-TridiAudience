cmake_minimum_required(VERSION 3.22)
project(tridi_gaze_wrapper LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED NCNN_ROOT)
  message(FATAL_ERROR "NCNN_ROOT is required")
endif()

add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES
  IMPORTED_LOCATION "${NCNN_ROOT}/armeabi-v7a/lib/libncnn.a"
  INTERFACE_INCLUDE_DIRECTORIES "${NCNN_ROOT}/armeabi-v7a/include/ncnn"
)

add_library(tridi_vision SHARED gaze_neural_wrapper.cpp)
target_include_directories(tridi_vision PRIVATE "${NCNN_ROOT}/armeabi-v7a/include/ncnn")
target_compile_options(tridi_vision PRIVATE -O3 -ffast-math -fvisibility=hidden -fopenmp -mfpu=neon)
target_link_options(tridi_vision PRIVATE -fopenmp)
target_link_libraries(tridi_vision PRIVATE ncnn android log dl)
