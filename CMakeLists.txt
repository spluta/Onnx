
set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required (VERSION 3.5)
project (Onnx)

set(APP_NAME Onnx)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include_directories(${SC_PATH}/include/plugin_interface)
include_directories(${SC_PATH}/include/common)
include_directories(${SC_PATH}/external_libraries/libsndfile/)

set(CMAKE_SHARED_MODULE_PREFIX "")
if(APPLE OR WIN32)
set(CMAKE_SHARED_MODULE_SUFFIX ".scx")
endif()


# add the plugin we are making
add_library(${APP_NAME} MODULE OnnxUGen.cpp)

# OnnxRuntime
if(APPLE)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    else()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
   endif()
elseif(WIN32)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
endif()

find_package(ONNXRUNTIME REQUIRED)
target_link_libraries(${APP_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(${APP_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

# Copy onnxruntime.dll on Windows
if(WIN32)
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll"
            )
endif()

target_link_libraries(${APP_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate/build/src/libsamplerate.a)

# cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH='/Users/spluta1/Documents/SC/SuperColliderBuild/SuperCollider' ..
# cmake --build . --config Release