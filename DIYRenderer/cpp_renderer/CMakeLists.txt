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
# Dependencies
# =============================================================================
find_package(OpenMP)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# mp-units (compile-time dimensional analysis)
set(MP_UNITS_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
set(MP_UNITS_API_CONTRACTS NONE CACHE STRING "" FORCE)
add_subdirectory(extern/mp-units/src)

# =============================================================================
# Main Target: diyrenderer (Python module)
# =============================================================================
pybind11_add_module(diyrenderer
    src/pybind_module.cpp
    src/node_evaluator.cpp
)

target_include_directories(diyrenderer PRIVATE include)
target_link_libraries(diyrenderer PRIVATE mp-units::mp-units)

if(OpenMP_CXX_FOUND)
    target_link_libraries(diyrenderer PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
endif()

target_compile_options(diyrenderer PRIVATE -O3 -march=native -ffast-math)

# =============================================================================
# Test Target: test_units
# =============================================================================
add_executable(test_units test/test_units.cpp)
target_include_directories(test_units PRIVATE include)
target_link_libraries(test_units PRIVATE mp-units::mp-units)
target_compile_options(test_units PRIVATE -O2)
