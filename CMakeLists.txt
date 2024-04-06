cmake_minimum_required(VERSION 3.0.0)
project(fishCenSV2 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(BOOST_ROOT /usr/local/include/boost_1_84_0)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)

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
                                   ${CMAKE_SOURCE_DIR}/libs/bytetrack/dataType.h
                                   ${CMAKE_SOURCE_DIR}/libs/client/client.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/client/client.hpp
                                   ${CMAKE_SOURCE_DIR}/libs/server/server.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/server/server.hpp
                                   ${CMAKE_SOURCE_DIR}/libs/server/session.cpp
                                   ${CMAKE_SOURCE_DIR}/libs/server/session.hpp)

target_link_libraries(fishCenSV2 ${OpenCV_LIBS}
                                 ${Boost_LIBRARIES}   
                                        cudart
                                        nvinfer
                                        OpenMP::OpenMP_CXX)

target_include_directories(fishCenSV2 PRIVATE ${CMAKE_SOURCE_DIR}/libs/yolo)
target_include_directories(fishCenSV2 PRIVATE ${CMAKE_SOURCE_DIR}/libs/bytetrack)
target_include_directories(fishCenSV2 PRIVATE ${CMAKE_SOURCE_DIR}/libs/client)
target_include_directories(fishCenSV2 PRIVATE ${CMAKE_SOURCE_DIR}/libs/server)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
target_compile_options(fishCenSV2 PRIVATE -O2 -fopenmp -march=armv8-a+crypto -mcpu=cortex-a57+crypto)