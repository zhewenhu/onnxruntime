cmake_minimum_required(VERSION 3.10)
project (multithreshold)
add_definitions(-std=c++11)


set(TEST_SOURCE multithreshold_test.cc)
set(HEADER multithreshold.h)
set(SOURCE multithreshold.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories(/home/zhewen/onnxruntime/include)

#Include path to header files for Custom Op Test
#include_directories(/home/zhewen/onnx/onnxruntime-linux-x64-1.8.1/include)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/zhewen/onnx/onnxruntime-linux-x64-1.8.1/lib)

if(${ONNXRUNTIME_LIBRARY})
    message(FATAL_ERROR "lib not found")
else()
    message(STATUS "lib found as ${ONNXRUNTIME_LIBRARY}")
endif()

target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
