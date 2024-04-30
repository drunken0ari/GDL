cmake_minimum_required(VERSION 3.27)
project(GlassPlaneDetectionAndLocalization)

set( CMAKE_BUILD_TYPE Debug )

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(GlassPlaneDetectionAndLocalization main.cpp Util.cpp GPDL.cpp)

target_link_libraries(GlassPlaneDetectionAndLocalization ${OpenCV_LIBS})