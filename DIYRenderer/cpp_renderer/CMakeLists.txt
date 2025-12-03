cmake_minimum_required(VERSION 3.15)
project(diyrt LANGUAGES CXX)

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

add_executable(diyrt
    src/main.cpp
    src/node_evaluator.cpp
)

target_include_directories(diyrt PRIVATE include)

# Optimization flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(diyrt PRIVATE 
        -O3                 # Maximum optimization
        -march=native       # Optimize for current CPU
        -ffast-math         # Fast floating point (allows reordering)
        -funroll-loops      # Unroll small loops
        -ftree-vectorize    # Auto-vectorization
    )
endif()

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(diyrt PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - multi-threaded rendering enabled")
else()
    message(STATUS "OpenMP not found - single-threaded rendering")
endif()
