cmake_minimum_required(VERSION 3.24)
project(cuckoo_hashmap)

set(CMAKE_CXX_STANDARD 11)

add_executable(test_hashmap test_hashmap.cpp cuckoo_hashmap.hpp)
