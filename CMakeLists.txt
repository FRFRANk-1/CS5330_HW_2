cmake_minimum_required(VERSION 3.5.0)
project(HW_2 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( Opencv REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(HW_2 main.cpp csv_util.cpp process_Img_funs.cpp)
target_compile_features(HW_2 PUBLIC cxx_std_17) 

target_link_libraries( HW_2 ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
