cmake_minimum_required(VERSION 3.27.0)
project(SmartSpectraHelloVitals CXX)
 
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
# Find required packages
find_package(SmartSpectra REQUIRED)
find_package(OpenCV REQUIRED)
 
# Create executable
add_executable(hello_vitals hello_vitals.cpp)
add_executable(attention_tracker attention_tracker.cpp)
 
# Link libraries
target_link_libraries(hello_vitals
    SmartSpectra::Container
    SmartSpectra::Gui
    ${OpenCV_LIBS}
)

target_link_libraries(attention_tracker
    SmartSpectra::Container
    SmartSpectra::Gui
    ${OpenCV_LIBS}
)