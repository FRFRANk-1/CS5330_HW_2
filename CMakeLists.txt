cmake_minimum_required(VERSION 3.0.0)
project(HW_2 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( Opencv REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(HW_2 main.cpp DirectoryReader.cpp)

target_link_libraries( HW_2 ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
