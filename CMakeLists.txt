cmake_minimum_required(VERSION 3.27.0)
project(SmartSpectraAttentionDemo CXX)

# Prefer relative paths in generated make rules when supported by the generator.
set(CMAKE_USE_RELATIVE_PATHS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SmartSpectra REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(create_demo_video
    src/attention_demo/create_demo_video.cpp
    src/attention_demo/demo_video_app.cpp
    src/attention_demo/attention_estimator.cpp
    src/attention_demo/status_flags.cpp
    src/attention_demo/overlay_renderer.cpp
)

target_include_directories(create_demo_video PRIVATE
    include
)

target_link_libraries(create_demo_video
    SmartSpectra::Container
    SmartSpectra::Gui
    ${OpenCV_LIBS}
)
