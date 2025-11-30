cmake_minimum_required(VERSION 3.15)
project(diyrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(diyrt
    src/main.cpp
)

target_include_directories(diyrt PRIVATE include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(diyrt PRIVATE -O3 -march=native -ffast-math)
endif()
