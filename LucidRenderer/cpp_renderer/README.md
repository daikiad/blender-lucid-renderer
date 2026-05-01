# Lucid Renderer - C++ Backend

A physically-based renderer implemented in C++20 with type-safe units, designed as a Blender add-on backend.

## Features

- **Type-safe units** using [mp-units](https://mpusz.github.io/mp-units/) (Position, Direction, Length, Area, etc.)
- **Path tracing** with Multiple Importance Sampling (MIS)
- **GGX microfacet BSDF** with Fresnel, including transmission/refraction
- **Light types**: Point, Sun, Area, Mesh, Spot
- **Python binding** via pybind11 for Blender integration
- **OpenMP parallelization** for tile-based rendering

## Requirements

- **macOS** (Apple Silicon tested)
- **GCC 15** (required for mp-units C++20 support; AppleClang has template issues)
- **Conan 2.x** (package manager)
- **CMake 3.25+**
- **Python 3.11** (for Blender compatibility)

## Installation

### 1. Install Prerequisites

```bash
# Install GCC 15
brew install gcc@15

# Install Conan 2
pip install conan

# Install Python 3.11 via uv (recommended for Blender compatibility)
uv python install 3.11
```

### 2. Configure Conan Profile

Create or update your Conan profile for GCC 15:

```bash
# Create default profile if it doesn't exist
conan profile detect

# Edit the profile to use GCC 15
conan profile path default
# Then edit the file to set:
#   [settings]
#   compiler=gcc
#   compiler.version=15
#   compiler.libcxx=libstdc++11
```

Or use the provided profile:
```bash
conan install . --profile=conan_gcc15_profile --build=missing --output-folder=build_pybind
```

### 3. Build

```bash
cd LucidRenderer/cpp_renderer

# Install dependencies with Conan
conan install . --build=missing --output-folder=build_pybind

# Configure with CMake (using Conan toolchain)
cmake -B build_pybind --preset conan-release

# Build
cmake --build build_pybind -j
```

Output: `build_pybind/lucidrenderer.cpython-311-darwin.so`

## Testing

The project includes comprehensive tests using Google Test:

```bash
# Run all tests
ctest --test-dir build_pybind --output-on-failure

# Run specific test suite
./build_pybind/test_bsdf --gtest_filter="FresnelTest.*"

# Run with verbose output
ctest --test-dir build_pybind -V
```

### Test Suites

| Test File | Description | Tests |
|-----------|-------------|-------|
| `test_render_units` | Type-safe unit system | 34 |
| `test_geometry` | Ray-primitive intersections (Triangle, AABB, Sphere, etc.) | 40 |
| `test_bsdf` | Fresnel, GGX NDF/G1/G2, VNDF sampling | 32 |
| `test_light` | Light sources, triangle sampling, PDF | 29 |
| `test_random` | RNG quality, uniformity, correlation | 19 |

**Total: 154 tests**

## Project Structure

```
cpp_renderer/
├── include/
│   ├── units/          # Type-safe units (Position, Direction, Length, etc.)
│   ├── core/           # Ray, HitRecord
│   ├── geometry/       # Intersection functions
│   ├── bsdf/           # Fresnel, GGX, material sampling
│   ├── light/          # Light sources and sampling
│   ├── math/           # RNG, sampling utilities
│   ├── integrator/     # Path tracer with MIS
│   └── renderer.hpp    # Main renderer class
├── src/
│   ├── pybind_module.cpp   # Python bindings
│   └── node_evaluator.cpp  # Blender node evaluation
├── test/
│   ├── test_geometry.cpp
│   ├── test_bsdf.cpp
│   ├── test_light.cpp
│   ├── test_random.cpp
│   └── render_units_test.cpp
├── CMakeLists.txt
├── conanfile.txt
└── README.md
```

## Dependencies (via Conan)

| Package | Version | Purpose |
|---------|---------|---------|
| mp-units | 2.4.0 | Type-safe physical units |
| nlohmann_json | 3.11.3 | JSON scene parsing |
| pybind11 | 2.13.6 | Python bindings |
| gtest | 1.15.0 | Unit testing |

## Usage in Blender

The compiled module integrates with the Blender add-on:

```python
import lucidrenderer

# Create renderer instance
renderer = lucidrenderer.PyRenderer()

# Set scene data (from Blender export)
renderer.set_scene_json(scene_json_string)

# Render a tile
pixels = renderer.render_tile(
    tile_x, tile_y, tile_width, tile_height,
    full_width, full_height,
    samples_per_pixel,
    seed_offset,
    pass_type  # 0=combined, 1=albedo, 2=normal
)
```

## Key Implementation Notes

### Fresnel Dielectric (eta convention)

```cpp
// eta = n_incident / n_transmitted
// Air → Glass: eta = 1.0 / 1.5 = 0.667
// Glass → Air: eta = 1.5 / 1.0 = 1.5 (TIR possible)
float F = fresnelDielectric(cosThetaI, eta);
```

### GGX NDF

```cpp
// D = α² / (π * (cos²θ * (α² - 1) + 1)²)
// where α² = roughness⁴ (Blender/Cycles convention)
// Note: GGX_EPSILON in denominator affects roughness < 0.1
float D = ggxD(NdotH, roughness);
```

### Position Type (quantity_point)

```cpp
// Position is an affine space type (mp-units quantity_point)
// To get coordinates:
Vec3f v = displacement_from_origin(position).numerical_value_in(m);
float x = v.x;
```

## Troubleshooting

### Build Errors with AppleClang

mp-units requires advanced C++20 features. Use GCC 15:
```bash
brew install gcc@15
# CMakeLists.txt automatically detects and uses it
```

### Python Version Mismatch

Ensure Python 3.11 for Blender compatibility:
```bash
uv python install 3.11
# CMakeLists.txt automatically finds UV's Python
```

### Conan Cache Issues

```bash
conan remove "*" -c  # Clear cache
conan install . --build=missing --output-folder=build_pybind
```

## License

Part of the blender-lucid-renderer-addon project.
