cmake_minimum_required(VERSION 3.15)
project(diyrenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS: Help CMake find Homebrew's libomp
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "Found Homebrew libomp at ${HOMEBREW_LIBOMP_PREFIX}")
    endif()
endif()

# Find OpenMP for multi-threading
find_package(OpenMP)

# =============================================================================
# pybind11 モジュール
# =============================================================================

# pybind11 を探す
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    # pybind11 が見つからない場合は FetchContent で取得
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Python モジュールをビルド
pybind11_add_module(diyrenderer
    src/pybind_module.cpp
    src/node_evaluator.cpp
)

target_include_directories(diyrenderer PRIVATE include)

# 最適化フラグ
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(diyrenderer PRIVATE 
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -ftree-vectorize
    )
endif()

# OpenMP をリンク
if(OpenMP_CXX_FOUND)
    target_link_libraries(diyrenderer PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - multi-threaded rendering enabled")
else()
    message(STATUS "OpenMP not found - single-threaded rendering")
endif()

message(STATUS "Building pybind11 module: diyrenderer")
