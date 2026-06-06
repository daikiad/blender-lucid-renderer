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
- **Apple Clang** (the `clang++` that ships with Xcode / Command Line Tools — Clang 17+ recommended)
- **Conan 2.x** (package manager)
- **CMake 3.25+**
- **Python 3.13** (for Blender 5.1 compatibility)

## Installation

### 1. Install Prerequisites

```bash
# Apple Clang ships with Xcode Command Line Tools
xcode-select --install   # only if not already installed

# Install Conan 2
pip install conan

# Install Python 3.13 via uv (recommended for Blender 5.1 compatibility)
uv python install 3.13
```

### 2. Build Lucid (CPU only)

The repository ships a Conan profile (`conan_profile`) pinned to
Apple Clang + libc++ + C++20.

```bash
cd LucidRenderer/cpp_renderer

# Install dependencies with Conan
conan install . --build=missing --output-folder=build_pybind \
    -pr:h=./conan_profile -pr:b=./conan_profile

# If you want the GPU backend, build Dawn first (see Step 3) and:
export Dawn_DIR=$HOME/.local/dawn/lib/cmake/Dawn

# Configure with CMake
cmake --preset conan-release

# Build
cmake --build --preset conan-release
```

Output: `build_pybind/lucidrenderer.cpython-313-darwin.so`

To skip the GPU backend entirely (CPU path tracer + diagnostics only):
```bash
cmake --preset conan-release -DLUCID_USE_DAWN=OFF
```

### 3. (Optional) Build Dawn for the GPU backend

Lucid optionally accelerates rendering on the GPU via Google's
[Dawn](https://dawn.googlesource.com/dawn) implementation of WebGPU.
The CPU path tracer (with mp-units typed code and the diagnostics
system) is unaffected and continues to work standalone.

Build Dawn with Apple Clang so its C++ stdlib (libc++) matches the
Lucid build:

```bash
git clone https://dawn.googlesource.com/dawn ~/src/dawn
cd ~/src/dawn

cmake -B out/Release \
    -DCMAKE_BUILD_TYPE=Release \
    -DDAWN_BUILD_SAMPLES=OFF \
    -DDAWN_BUILD_TESTS=OFF \
    -DDAWN_FETCH_DEPENDENCIES=ON \
    -DDAWN_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local/dawn

cmake --build out/Release --target install -j
```

Initial build is roughly 10-30 minutes and produces ~1-2 GB of artifacts.
Subsequent builds are incremental.

Before configuring Lucid:
```bash
export Dawn_DIR=$HOME/.local/dawn/lib/cmake/Dawn
```

#### Verify the GPU backend works

After building Lucid:
```python
import lucidrenderer
assert lucidrenderer.dawn_enabled
print(lucidrenderer.gpu_adapter_info())   # e.g. "apple / Apple M4 Pro / Metal"

out = lucidrenderer.gpu_run_double_test(64)
assert out == [i * 2.0 for i in range(64)]
```

#### Loading Dawn inside Blender

If the Dawn dylib is not found when Blender imports `lucidrenderer`, set:
```bash
export DYLD_LIBRARY_PATH=$HOME/.local/dawn/lib:$DYLD_LIBRARY_PATH
```
before launching Blender, or symlink the dylib next to Lucid's `.so`.

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

### Build Errors with mp-units

mp-units requires advanced C++20 features. Apple Clang 17+ works; older
versions may fail on template instantiation depth. Update Xcode / Command
Line Tools if you hit template errors in `mp-units` headers.

### Python Version Mismatch

Ensure Python 3.13 for Blender 5.1 compatibility:
```bash
uv python install 3.13
# CMakeLists.txt automatically finds UV's Python
```

### Conan Cache Issues

```bash
conan remove "*" -c  # Clear cache
conan install . --build=missing --output-folder=build_pybind
```

## License

Part of the blender-lucid-renderer-addon project.
