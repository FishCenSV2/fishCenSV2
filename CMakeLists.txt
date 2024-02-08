cmake_minimum_required(VERSION 3.0.0)
project(fishCenSV2 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)
include(CTest)
enable_testing()

add_executable(fishCenSV2 main.cpp)
target_link_libraries(fishCenSV2 ${OpenCV_LIBS}
                                        cudart
                                        nvinfer)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
