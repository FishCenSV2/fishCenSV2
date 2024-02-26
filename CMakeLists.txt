cmake_minimum_required(VERSION 3.0.0)
project(fishCenSV2 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)
include(CTest)
enable_testing()

add_executable(fishCenSV2 main.cpp ${CMAKE_SOURCE_DIR}/libs/yolo/yolov8.cpp 
                                   ${CMAKE_SOURCE_DIR}/libs/yolo/yolov8.hpp
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/BYTETracker.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/BYTETracker.h
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/kalmanFilter.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/kalmanFilter.h
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/lapjv.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/lapjv.h
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/STrack.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/STrack.h
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/utils.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/dataType.h)

target_link_libraries(fishCenSV2 ${OpenCV_LIBS}
                                        cudart
                                        nvinfer
                                        OpenMP::OpenMP_CXX)

target_include_directories(fishCenSV2 PRIVATE ${CMAKE_SOURCE_DIR}/libs/yolo)
target_include_directories(fishCenSV2 PRIVATE ${CMAKE_SOURCE_DIR}/libs/bytetrack)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
target_compile_options(fishCenSV2 PRIVATE -O2 -fopenmp -march=armv8-a+crypto -mcpu=cortex-a57+crypto)