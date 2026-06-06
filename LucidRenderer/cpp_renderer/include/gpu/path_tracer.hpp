/**
 * path_tracer.hpp - GPU path tracer (Phase 2a: diffuse + emissive, brute-force)
 * ============================================================================
 *
 * Mirrors the CPU `traceSimple()` path: BSDF-sampling only, no NEE/MIS.
 * Lambertian diffuse materials, emissive triangle hits terminate the path,
 * environment color on miss. Russian roulette from depth 3.
 *
 * The CPU mp-units typed renderer remains the reference; this is a separate
 * brute-force implementation in WGSL. BVH on GPU is a later phase.
 *
 * Per-render: one dispatch produces `samples` light paths per pixel and
 * returns a flat RGBA float buffer (Y-flipped, pixel-value space — same
 * convention as `PyRenderer::render_tile()`'s output).
 */

#pragma once

#ifdef LUCID_HAS_DAWN

#include <webgpu/webgpu_cpp.h>
#include <cstdint>
#include <optional>
#include <vector>

#include "units/render_units.hpp"

// Forward declarations
struct Scene;

namespace lucid::gpu {

class DawnContext;

// ----------------------------------------------------------------------------
// PackedPathScene - flat triangle buffer with per-triangle material
// ----------------------------------------------------------------------------
// Layout per triangle (32 floats / 128 bytes, std430-aligned):
//   v0.xyz, _pad         (vec4)
//   v1.xyz, _pad         (vec4)
//   v2.xyz, _pad         (vec4)
//   n0.xyz, _pad         (vec4)
//   n1.xyz, _pad         (vec4)
//   n2.xyz, smooth_flag  (vec4)   // 1.0 if smooth normals, else 0.0
//   albedo.rgb,   _pad   (vec4)
//   emission.rgb, _pad   (vec4)
// `triangles.size() == triangle_count * 32`.
//
// Environment color / strength live in PathTracerParamsGpu, not here.
//
// `point_lights` is 8 floats per light (std430 alignment):
//   pos.xyz,   _pad      (vec4)
//   color.xyz, intensity (vec4)   // intensity ≈ Blender Light.energy in W
struct PackedPathScene {
    std::vector<float> triangles;
    uint32_t triangle_count = 0;
    std::vector<float> point_lights;
    uint32_t point_light_count = 0;
};

PackedPathScene pack_scene_for_path_tracer(const Scene& scene);

// ----------------------------------------------------------------------------
// PathTracerParamsGpu - uniform layout (128 bytes, std140-aligned)
// ----------------------------------------------------------------------------
struct PathTracerParamsGpu {
    // Camera (mirrors CameraParamsGpu)
    float    pos[3];       float _pad0;
    float    fwd[3];       float _pad1;
    float    right[3];     float _pad2;
    float    up[3];        float fov_scale;
    float    aspect;
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t full_w;
    uint32_t full_h;
    uint32_t _pad3;

    // Path-trace controls
    uint32_t samples;          // Number of paths per pixel this dispatch.
    uint32_t sample_offset;    // For viewport accumulation: starting sample idx.
    uint32_t max_bounces;      // Max bounce depth.
    uint32_t frame_seed;       // Per-frame seed for the PCG (mixed with pixel idx).

    // Environment color (RGB) + strength (alpha slot)
    float    env_color[3];     float env_strength;

    // Lights
    uint32_t point_light_count;
    uint32_t _pad4;
    uint32_t _pad5;
    uint32_t _pad6;
};
static_assert(sizeof(PathTracerParamsGpu) == 144,
              "PathTracerParamsGpu must match the WGSL uniform layout (144 bytes)");

PathTracerParamsGpu make_path_tracer_params(
    const render::Position& pos,
    const render::Direction& fwd,
    const render::Direction& right,
    const render::Direction& up,
    float fov_rad,
    float aspect,
    uint32_t tile_x, uint32_t tile_y,
    uint32_t tile_w, uint32_t tile_h,
    uint32_t full_w, uint32_t full_h,
    uint32_t samples, uint32_t sample_offset, uint32_t max_bounces,
    uint32_t frame_seed,
    const float env_color[3], float env_strength,
    uint32_t point_light_count);

// ----------------------------------------------------------------------------
// PathTracer - cached pipeline + storage buffers
// ----------------------------------------------------------------------------
class PathTracer {
public:
    static std::optional<PathTracer> create(DawnContext& ctx);

    // Run a path-trace dispatch. Returns a flat RGBA float buffer of size
    // tile_w * tile_h * 4, Y-flipped to match the CPU pybind output. Each
    // RGB value is the path-traced pixel color (Radiance × CameraSensitivity);
    // alpha is always 1.0. Throws std::runtime_error on dispatch failure.
    std::vector<float> render(DawnContext& ctx,
                              const PackedPathScene& scene,
                              const PathTracerParamsGpu& params);

    PathTracer(PathTracer&&) noexcept = default;
    PathTracer& operator=(PathTracer&&) noexcept = default;
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

private:
    PathTracer(wgpu::ShaderModule, wgpu::ComputePipeline, wgpu::BindGroupLayout);

    wgpu::ShaderModule    shader_;
    wgpu::ComputePipeline pipeline_;
    wgpu::BindGroupLayout layout_;

    // Cached storage / staging buffers; grown on demand.
    wgpu::Buffer params_buf_;
    wgpu::Buffer tri_buf_;          uint64_t tri_buf_capacity_ = 0;
    wgpu::Buffer point_lights_buf_; uint64_t point_lights_buf_capacity_ = 0;
    wgpu::Buffer out_buf_;          uint64_t out_buf_capacity_ = 0;
    wgpu::Buffer stage_buf_;        uint64_t stage_buf_capacity_ = 0;
};

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
