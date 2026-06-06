# Lucid

A custom path-tracing renderer add-on for Blender. The Python side integrates with Blender; the rendering core is C++ wrapped via pybind11. The selling point isn't general-purpose rendering — it's **per-pixel light path inspection and analysis**.

## What it does

- **Path tracing** — BSDF / NEE / MIS sampling strategies, Principled BSDF and Emission node materials, Point / Sun / Area / Mesh / Spot lights, progressive sampling
- **Path diagnostics** — hover over a pixel in the Image Editor to list every light path that reached it; group by sampling strategy with mean / variance contribution; draw paths as line overlays in 2D and 3D views; export selected paths as Curve objects in the 3D Viewport
- **Type-safe rendering math (experimental)** — the C++ core leans on [mp-units](https://mpusz.github.io/mp-units/) to give the units in the rendering equation real types. See [§ Type-safe rendering math](#type-safe-rendering-math-experimental) below.
- **Debug passes** — normal / albedo / emission previews
- **Multi-instance** — multiple viewports, F12 renders, and material previews run concurrently with their own session (ADR 003)
- **Event-driven diff updates** — uses `depsgraph.id_type_updated()` so geometry / material / light changes only rebuild what they need

## Architecture

```
[Blender]
   ↓  Python add-on (LucidRenderer/)
   ↓    engine.py / viewport.py / render_session.py / scene_sync.py
   ↓    diagnostics.py / hover_diagnostics.py / path_visualizer.py
   ↓
[pybind11]   ←  lucidrenderer.cpython-311-darwin.so
   ↓
[C++ core] (LucidRenderer/cpp_renderer/)
   path tracer / BSDF / lights / BVH / DiagnosticFilm
```

For details, see:
- [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) — module layout and reading guide
- [`docs/PATH_DIAGNOSTICS.md`](docs/PATH_DIAGNOSTICS.md) — path diagnostics design
- [`docs/adr/`](docs/adr/) — design decisions (001: baseline architecture / 002: refactor comparison / 003: multi-instance)
- [`LucidRenderer/cpp_renderer/README.md`](LucidRenderer/cpp_renderer/README.md) — C++ backend internals

## Type-safe rendering math (experimental)

Most renderers carry positions, directions, radiance, BSDF values, and PDFs around as bare `Vec3f` / `float`. Lucid instead encodes them in the type system using the C++ ISO units library [mp-units](https://mpusz.github.io/mp-units/), so dimensional and semantic mistakes get caught at compile time. This is the part that's the most fun-and-painful, and it's why the build pins to GCC 15 (mp-units' templates choke on AppleClang).

What's actually typed:

- **Affine-correct positions.** `Position` is a `quantity_point<isq::displacement[m], Vec3f>` — an affine point, not a vector. `Position - Position` returns a `Displacement`, `Position + Displacement` returns a `Position`, but `Position + Position` doesn't compile. Vector-vs-point confusion in transforms goes away.
- **Real units on geometric quantities.** `Length [m]`, `Area [m²]`, `Volume [m³]`, `Angle [rad]`, `SolidAngle [sr]`, `Displacement [m]` (vector-valued), `Velocity [m/s]` — all `quantity<...>` with proper SI dimensions.
- **PDFs in their actual units.** `PdfW = quantity<1/sr>` for solid-angle densities, `PdfA = quantity<1/m²>` for area densities. Mixing them up — e.g. forgetting the area-to-solid-angle conversion in MIS — fails to compile instead of producing subtly wrong weights.
- **RGB tagged by physical role.** The same `RGB3f` struct is parameterized by a `quantity_spec` tag: `RGB<Attenuation>` for material reflectances in [0,1], `RGB<Throughput>` for path weights in [0,∞), `RGB<Radiance>` in W/(sr·m²), `RGB<BSDF>` in 1/sr. Multiplying a radiance by another radiance, or accidentally treating a throughput as an attenuation, won't typecheck.
- **Custom `quantity_spec` for vector area.** mp-units' built-in `isq::area` is scalar; the cross product of two displacements is an oriented area (a vector), so the project declares its own `QUANTITY_SPEC(oriented_area, isq::area, quantity_character::vector)` and uses that for surface normals scaled by area.

It's not yet wired through every code path — it's an experiment in seeing how far the rendering equation can be lifted into the type system before ergonomics break down — but the core geometry, BSDF, and light sampling layers are all on it. See [`LucidRenderer/cpp_renderer/include/units/render_units.hpp`](LucidRenderer/cpp_renderer/include/units/render_units.hpp) for the type definitions.

## Setup

### 1. Build the C++ core

`mp-units 2.4.0` exercises C++20 templates that AppleClang doesn't handle, so the build uses **Homebrew GCC 15**. A matching Conan profile is included in the repo.

```bash
brew install gcc@15
pip install conan      # or: uv pip install conan

cd LucidRenderer/cpp_renderer
conan install . --output-folder=build_pybind --build=missing \
  -pr:h=./conan_gcc15_profile -pr:b=./conan_gcc15_profile
cmake --preset conan-release
cmake --build --preset conan-release
```

Output: `build_pybind/lucidrenderer.cpython-311-darwin.so`

### 2. Register the add-on with Blender

Symlink `LucidRenderer/` into Blender's addons directory:

```bash
ln -s "$(pwd)/LucidRenderer" \
      "$HOME/Library/Application Support/Blender/5.1/scripts/addons/LucidRenderer"
```

Launch Blender → Edit > Preferences > Add-ons → enable `Lucid Renderer (Minimal Example)` → in the Render Engine dropdown, pick `Lucid (Minimal)`.

## Using it

- **Viewport rendering** — switch the shading mode to Rendered and Lucid takes over
- **F12 render** — works as usual
- **Path diagnostics** — in Properties > Render > Diagnostics, enable `Enable Path Recording`. After a render, hover over a pixel in the Image Editor to see the list of light paths that reached it, grouped by sampling strategy.
- **Raw path storage** (experimental) — toggle `Store All Paths` in the Diagnostics panel to retain every per-sample path so paths can be regrouped and analyzed on demand. Memory cost is high (~6.5 GB for FHD at 32 SPP).

## Dependencies

| | Version |
|---|---|
| Blender | 5.1 |
| Python | 3.13 (must match Blender's bundled Python) |
| GCC | 15 (Homebrew) |
| Conan | 2.x |
| CMake | 3.30+ |
| mp-units | 2.4.0 |
| nlohmann_json | 3.11.3 |
| pybind11 | 2.13.6 |
| GoogleTest | 1.15.0 |

## Status

Personal project. Developed and tested on macOS (Apple Silicon). Linux and Windows are not verified.
