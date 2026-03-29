cmake_minimum_required(VERSION 3.27.0)
project(SmartSpectraAttentionDemo CXX)

# Prefer relative paths in generated make rules when supported by the generator.
set(CMAKE_USE_RELATIVE_PATHS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SmartSpectra REQUIRED)
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "Path to ONNX Runtime installation root")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES
        onnxruntime_cxx_api.h
        onnxruntime/core/session/onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        /usr/local/include
        /usr/include
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
)

if(NOT ONNXRUNTIME_LIBRARY)
    find_file(ONNXRUNTIME_LIBRARY
        NAMES
            libonnxruntime.so
            libonnxruntime.so.1.17.0
            libonnxruntime.so.1.18.1
        PATHS
            ${ONNXRUNTIME_ROOT}/lib
            /usr/local/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or install ONNX Runtime headers/libs.")
endif()

add_executable(create_demo_video
    src/attention_demo/create_demo_video.cpp
    src/attention_demo/demo_video_app.cpp
    src/attention_demo/attention_estimator.cpp
    src/attention_demo/status_flags.cpp
    src/attention_demo/overlay_renderer.cpp
)

target_include_directories(create_demo_video PRIVATE
    include
    emotion_detector
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(create_demo_video
    SmartSpectra::Container
    SmartSpectra::Gui
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)
