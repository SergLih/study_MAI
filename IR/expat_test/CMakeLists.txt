cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project(expat_test)

find_package(Catch2 REQUIRED)
add_executable(tests test.cpp tstring.cpp index.cpp exprtree.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2)

#link_libraries(stdc++fs)
#include(FindEXPAT)
#find_package(EXPAT REQUIRED)
#include_directories(${EXPAT_INCLUDE_DIRS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
#add_custom_target(expat_test COMMAND make -C /home/sergey/MAI/maga/Inf_search/expat_test
#                    CLION_EXE_DIR=/home/sergey/MAI/maga/Inf_search/expat_test/cmake-build-debug)
add_executable(${PROJECT_NAME} main.cpp tstring.cpp index.cpp exprtree.cpp)
#target_link_libraries(${PROJECT_NAME} ${EXPAT_LIBRARIES})