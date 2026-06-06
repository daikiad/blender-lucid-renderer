cmake_minimum_required(VERSION 3.25)

# Compiler is selected by the Conan profile (Apple Clang on macOS, libc++).
project(lucidrenderer LANGUAGES CXX)

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
# Dawn (WebGPU) GPU Backend Option
# =============================================================================
option(LUCID_USE_DAWN "Build with WebGPU/Dawn GPU backend" ON)

if(LUCID_USE_DAWN)
    # Dawn's exported targets reference Threads::Threads
    find_package(Threads REQUIRED)
    find_package(Dawn REQUIRED)
    message(STATUS "Dawn found: WebGPU/GPU backend enabled")
endif()

# =============================================================================
# Python Setup (for Blender compatibility)
# =============================================================================
# Use UV's Python 3.13 (Blender 5.1 uses Python 3.13)
if(APPLE AND NOT DEFINED Python_EXECUTABLE)
    file(GLOB UV_PYTHON_PATHS "$ENV{HOME}/.local/share/uv/python/cpython-3.13*-macos-aarch64-none/bin/python3.13")
    if(UV_PYTHON_PATHS)
        list(GET UV_PYTHON_PATHS 0 UV_PYTHON)
        set(Python_EXECUTABLE "${UV_PYTHON}")
        set(PYTHON_EXECUTABLE "${UV_PYTHON}")
        set(PYBIND11_PYTHON_VERSION 3.13)
        message(STATUS "Using UV Python 3.13: ${UV_PYTHON}")
    else()
        message(WARNING "UV Python 3.13 not found. Install with: uv python install 3.13")
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
# Main Target: lucidrenderer (Python module)
# =============================================================================
set(LUCID_SOURCES
    src/pybind_module.cpp
    src/node_evaluator.cpp
)
if(LUCID_USE_DAWN)
    list(APPEND LUCID_SOURCES
        src/gpu/dawn_context.cpp
        src/gpu/shader_module.cpp
    )
endif()

pybind11_add_module(lucidrenderer ${LUCID_SOURCES})

target_include_directories(lucidrenderer PRIVATE include)
target_link_libraries(lucidrenderer PRIVATE
    mp-units::mp-units
    nlohmann_json::nlohmann_json
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lucidrenderer PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
endif()

# Performance: -O3 + native + ffast-math + LTO for cross-file optimization
target_compile_options(lucidrenderer PRIVATE -O3 -march=native -ffast-math -flto)
target_link_options(lucidrenderer PRIVATE -flto)

if(LUCID_USE_DAWN)
    target_link_libraries(lucidrenderer PRIVATE dawn::webgpu_dawn)
    target_compile_definitions(lucidrenderer PRIVATE
        LUCID_HAS_DAWN=1
        LUCID_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    )
    # Make the Dawn dylib loadable from inside Blender's process:
    # @loader_path resolves relative to the .so, and Dawn_DIR/../../ is the
    # install prefix's lib directory.
    set_target_properties(lucidrenderer PROPERTIES
        INSTALL_RPATH "@loader_path;${Dawn_DIR}/../../"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

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
# Diagnostics Tests (Path Variance Analyzer)
# =============================================================================

# --- test_path_types: Path type and classification tests ---
add_executable(test_path_types test/diagnostics/test_path_types.cpp)
target_include_directories(test_path_types PRIVATE include)
target_link_libraries(test_path_types PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_path_types PRIVATE -O2)
gtest_discover_tests(test_path_types)

# --- test_path_stats: Path statistics and grouping tests ---
add_executable(test_path_stats test/diagnostics/test_path_stats.cpp)
target_include_directories(test_path_stats PRIVATE include)
target_link_libraries(test_path_stats PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_path_stats PRIVATE -O2)
gtest_discover_tests(test_path_stats)

# --- test_path_stats_config: Configuration and memory estimation tests ---
add_executable(test_path_stats_config test/diagnostics/test_path_stats_config.cpp)
target_include_directories(test_path_stats_config PRIVATE include)
target_link_libraries(test_path_stats_config PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_path_stats_config PRIVATE -O2)
gtest_discover_tests(test_path_stats_config)

# --- test_pixel_path_data: Per-pixel Top-N group management tests ---
add_executable(test_pixel_path_data test/diagnostics/test_pixel_path_data.cpp)
target_include_directories(test_pixel_path_data PRIVATE include)
target_link_libraries(test_pixel_path_data PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_pixel_path_data PRIVATE -O2)
gtest_discover_tests(test_pixel_path_data)

# --- test_diagnostic_film: Full image diagnostic data tests ---
add_executable(test_diagnostic_film test/diagnostics/test_diagnostic_film.cpp)
target_include_directories(test_diagnostic_film PRIVATE include)
target_link_libraries(test_diagnostic_film PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_diagnostic_film PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(test_diagnostic_film PRIVATE -O2)
gtest_discover_tests(test_diagnostic_film)

# --- test_diagnostic_integrator: Path tracer integration tests ---
add_executable(test_diagnostic_integrator test/diagnostics/test_diagnostic_integrator.cpp)
target_include_directories(test_diagnostic_integrator PRIVATE include)
target_link_libraries(test_diagnostic_integrator PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_diagnostic_integrator PRIVATE -O2)
gtest_discover_tests(test_diagnostic_integrator)

# --- test_diagnostic_export: Python binding export tests ---
add_executable(test_diagnostic_export test/diagnostics/test_diagnostic_export.cpp)
target_include_directories(test_diagnostic_export PRIVATE include)
target_link_libraries(test_diagnostic_export PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_diagnostic_export PRIVATE -O2)
gtest_discover_tests(test_diagnostic_export)

# --- test_diagnostic_pybind: Pybind11 binding interface tests ---
add_executable(test_diagnostic_pybind test/diagnostics/test_diagnostic_pybind.cpp)
target_include_directories(test_diagnostic_pybind PRIVATE include)
target_link_libraries(test_diagnostic_pybind PRIVATE 
    mp-units::mp-units 
    GTest::gtest 
    GTest::gtest_main
)
target_compile_options(test_diagnostic_pybind PRIVATE -O2)
gtest_discover_tests(test_diagnostic_pybind)

# --- test_pyrenderer_diagnostics: PyRenderer diagnostic integration tests ---
add_executable(test_pyrenderer_diagnostics 
    test/test_pyrenderer_diagnostics.cpp
    src/node_evaluator.cpp
)
target_include_directories(test_pyrenderer_diagnostics PRIVATE include)
target_link_libraries(test_pyrenderer_diagnostics PRIVATE 
    mp-units::mp-units 
    nlohmann_json::nlohmann_json
    GTest::gtest 
    GTest::gtest_main
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_pyrenderer_diagnostics PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(test_pyrenderer_diagnostics PRIVATE -O2)
gtest_discover_tests(test_pyrenderer_diagnostics)

# --- test_path_tracer_diagnostics: Path tracer + diagnostics integration ---
add_executable(test_path_tracer_diagnostics 
    test/diagnostics/test_path_tracer_diagnostics.cpp
    src/node_evaluator.cpp
)
target_include_directories(test_path_tracer_diagnostics PRIVATE include test)
target_link_libraries(test_path_tracer_diagnostics PRIVATE 
    mp-units::mp-units 
    nlohmann_json::nlohmann_json
    GTest::gtest 
    GTest::gtest_main
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_path_tracer_diagnostics PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(test_path_tracer_diagnostics PRIVATE -O2)
gtest_discover_tests(test_path_tracer_diagnostics)

# =============================================================================
# GPU Tests (only when Dawn is enabled)
# =============================================================================
if(LUCID_USE_DAWN)
    add_executable(test_gpu_dawn_context
        test/test_gpu_dawn_context.cpp
        src/gpu/dawn_context.cpp
        src/gpu/shader_module.cpp
    )
    target_include_directories(test_gpu_dawn_context PRIVATE include)
    target_link_libraries(test_gpu_dawn_context PRIVATE
        dawn::webgpu_dawn
        GTest::gtest
        GTest::gtest_main
    )
    target_compile_definitions(test_gpu_dawn_context PRIVATE
        LUCID_HAS_DAWN=1
        LUCID_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    )
    target_compile_options(test_gpu_dawn_context PRIVATE -O2)
    gtest_discover_tests(test_gpu_dawn_context)
endif()

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