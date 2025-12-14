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
# Coverage Option
# =============================================================================
option(COVERAGE "Enable code coverage measurement (gcov)" OFF)

if(COVERAGE)
    message(STATUS "Code coverage enabled")
    set(COVERAGE_FLAGS "--coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_FLAGS}")
endif()

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
# Test Targets (Google Test)
# =============================================================================
enable_testing()
include(GoogleTest)

# --- test_render_units: Unit type system tests ---
add_executable(test_render_units test/render_units_test.cpp)
target_include_directories(test_render_units PRIVATE include)
target_link_libraries(test_render_units PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_render_units PRIVATE -O2)
gtest_discover_tests(test_render_units)

# --- test_geometry: Ray-primitive intersection tests ---
add_executable(test_geometry test/test_geometry.cpp)
target_include_directories(test_geometry PRIVATE include)
target_link_libraries(test_geometry PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_geometry PRIVATE -O2)
gtest_discover_tests(test_geometry)

# --- test_bsdf: BSDF and material function tests ---
add_executable(test_bsdf test/test_bsdf.cpp)
target_include_directories(test_bsdf PRIVATE include)
target_link_libraries(test_bsdf PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_bsdf PRIVATE -O2)
gtest_discover_tests(test_bsdf)

# --- test_light: Light source and sampling tests ---
add_executable(test_light test/test_light.cpp)
target_include_directories(test_light PRIVATE include)
target_link_libraries(test_light PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_light PRIVATE -O2)
gtest_discover_tests(test_light)

# --- test_random: RNG quality tests ---
add_executable(test_random test/test_random.cpp)
target_include_directories(test_random PRIVATE include)
target_link_libraries(test_random PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_random PRIVATE -O2)
gtest_discover_tests(test_random)

# --- test_integrator: MIS and path tracer tests ---
add_executable(test_integrator test/test_integrator.cpp)
target_include_directories(test_integrator PRIVATE include)
target_link_libraries(test_integrator PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_integrator PRIVATE -O2)
gtest_discover_tests(test_integrator)

# --- test_integrator_e2e: End-to-end integrator comparison tests ---
add_executable(test_integrator_e2e test/test_integrator_e2e.cpp)
target_include_directories(test_integrator_e2e PRIVATE include)
target_link_libraries(test_integrator_e2e PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
# Use -O3 for faster rendering in E2E tests
target_compile_options(test_integrator_e2e PRIVATE -O3 -march=native)
gtest_discover_tests(test_integrator_e2e)

# --- test_scene_lights: SceneLights and light index mapping tests ---
add_executable(test_scene_lights 
    test/test_scene_lights.cpp
    src/node_evaluator.cpp
)
target_include_directories(test_scene_lights PRIVATE include)
target_link_libraries(test_scene_lights PRIVATE 
    mp-units::mp-units 
    nlohmann_json::nlohmann_json
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_scene_lights PRIVATE -O2)
gtest_discover_tests(test_scene_lights)

# --- test_path_tracer: Path tracer convergence/regression tests ---
add_executable(test_path_tracer 
    test/test_path_tracer.cpp
    src/node_evaluator.cpp
)
target_include_directories(test_path_tracer PRIVATE include)
target_link_libraries(test_path_tracer PRIVATE 
    mp-units::mp-units 
    nlohmann_json::nlohmann_json
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_path_tracer PRIVATE -O3 -march=native)
gtest_discover_tests(test_path_tracer)

# =============================================================================
# Coverage Target
# =============================================================================
if(COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)
    find_program(GCOV_15 gcov-15 PATHS /opt/homebrew/bin /usr/local/bin)
    
    if(LCOV AND GENHTML)
        # Set gcov tool option for GCC 15 compatibility
        if(GCOV_15)
            set(GCOV_TOOL_OPT "--gcov-tool" "${GCOV_15}")
            message(STATUS "Using gcov-15 for coverage: ${GCOV_15}")
        else()
            set(GCOV_TOOL_OPT "")
        endif()
        
        add_custom_target(coverage
            COMMENT "Generating coverage report..."
            # Clean previous coverage data
            COMMAND ${LCOV} ${GCOV_TOOL_OPT} --directory . --zerocounters
            # Run all tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            # Capture coverage data
            COMMAND ${LCOV} ${GCOV_TOOL_OPT} --ignore-errors gcov,gcov,inconsistent,inconsistent --directory . --capture --output-file coverage.info
            # Remove system headers and test files from coverage
            COMMAND ${LCOV} ${GCOV_TOOL_OPT} --ignore-errors unused --remove coverage.info 
                '/usr/*' 
                '*/test/*' 
                '*/.conan2/*'
                '*/mp-units/*'
                '*/nlohmann/*'
                '*/gtest/*'
                '*/opt/homebrew/*'
                --output-file coverage.info
            # Generate HTML report
            COMMAND ${GENHTML} coverage.info --output-directory coverage_report --ignore-errors source,inconsistent
            COMMAND ${CMAKE_COMMAND} -E echo "Coverage report: ${CMAKE_BINARY_DIR}/coverage_report/index.html"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    else()
        message(WARNING "lcov/genhtml not found. Install with: brew install lcov")
    endif()
endif()