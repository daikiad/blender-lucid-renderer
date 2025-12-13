cmake_minimum_required(VERSION 3.25)

# =============================================================================
# Compiler Setup (macOS)
# =============================================================================
# Use GCC 15 on macOS because AppleClang has issues with mp-units C++20 templates
if(APPLE AND NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(GXX_15 g++-15 PATHS /opt/homebrew/bin /usr/local/bin)
    find_program(GCC_15 gcc-15 PATHS /opt/homebrew/bin /usr/local/bin)
    if(GXX_15 AND GCC_15)
        set(CMAKE_CXX_COMPILER ${GXX_15})
        set(CMAKE_C_COMPILER ${GCC_15})
        message(STATUS "Using GCC 15: ${GXX_15}")
    else()
        message(WARNING "GCC 15 not found. Install with: brew install gcc@15")
    endif()
endif()

project(diyrenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Python Setup (for Blender compatibility)
# =============================================================================
# Use UV's Python 3.11 (Blender uses Python 3.11)
if(APPLE AND NOT DEFINED Python_EXECUTABLE)
    file(GLOB UV_PYTHON_PATHS "$ENV{HOME}/.local/share/uv/python/cpython-3.11*-macos-aarch64-none/bin/python3.11")
    if(UV_PYTHON_PATHS)
        list(GET UV_PYTHON_PATHS 0 UV_PYTHON)
        set(Python_EXECUTABLE "${UV_PYTHON}")
        set(PYTHON_EXECUTABLE "${UV_PYTHON}")
        set(PYBIND11_PYTHON_VERSION 3.11)
        message(STATUS "Using UV Python 3.11: ${UV_PYTHON}")
    else()
        message(WARNING "UV Python 3.11 not found. Install with: uv python install 3.11")
    endif()
endif()

# =============================================================================
# Dependencies (via Conan)
# =============================================================================
find_package(OpenMP)
find_package(mp-units REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(pybind11 REQUIRED)
find_package(GTest REQUIRED)

# =============================================================================
# Main Target: diyrenderer (Python module)
# =============================================================================
pybind11_add_module(diyrenderer
    src/pybind_module.cpp
    src/node_evaluator.cpp
)

target_include_directories(diyrenderer PRIVATE include)
target_link_libraries(diyrenderer PRIVATE 
    mp-units::mp-units 
    nlohmann_json::nlohmann_json
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(diyrenderer PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
endif()

# Performance: -O3 + native + ffast-math + LTO for cross-file optimization
target_compile_options(diyrenderer PRIVATE -O3 -march=native -ffast-math -flto)
target_link_options(diyrenderer PRIVATE -flto)

# =============================================================================
# Test Target: test_render_units (Google Test)
# =============================================================================
enable_testing()
add_executable(test_render_units test/render_units_test.cpp)
target_include_directories(test_render_units PRIVATE include)
target_link_libraries(test_render_units PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_render_units PRIVATE -O2)
include(GoogleTest)
gtest_discover_tests(test_render_units)
