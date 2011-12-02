cmake_minimum_required(VERSION 2.6)

set (CMAKE_BUILD_TYPE DEBUG)
#set (CMAKE_BUILD_TYPE RELWITHDEBINFO)
#set (CMAKE_BUILD_TYPE RELEASE)

set(CMAKE_CXX_FLAGS " -O0 -g -std=gnu++0x -msse -msse2 -msse3 -Wall -Wextra -Winit-self -Wpointer-arith -Wno-sign-compare -Wcast-qual -Wno-unused-parameter -Wno-uninitialized -Wno-missing-field-initializers -Wnon-virtual-dtor -Woverloaded-virtual -Wsign-promo ${CMAKE_CXX_FLAGS}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(ut ut.cpp)
