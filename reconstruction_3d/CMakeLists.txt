cmake_minimum_required(VERSION 2.6)
project(reconstruction_3d)

find_package(OpenCV 3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")



add_executable(reconstruction_3d main.cpp)
target_link_libraries(reconstruction_3d ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

install(TARGETS reconstruction_3d RUNTIME DESTINATION bin)
