/**
 * debug_renderer.hpp - GPU normal-debug renderer (Phase 1b)
 * =========================================================
 *
 * Brute-force GPU implementation of the CPU `traceNormal` debug path: shoots
 * one ray per pixel against every triangle in the scene and writes a flat
 * RGBA buffer (the same layout that `PyRenderer::render_debug` returns).
 *
 * This is intentionally a separate code path from the CPU mp-units typed
 * renderer; the CPU side is the typed reference, this side is the speed lane.
 * BVH on GPU is a later phase — for now, scenes above ~10k triangles will
 * crawl.
 *
 * Lifetime:
 *   - `DebugRenderer::create()` builds the compute pipeline, shader module,
 *     and bind-group layout once. Re-use across many renders.
 *   - GPU storage buffers are cached on the `DebugRenderer` and grown only
 *     when capacity is insufficient.
 */

#pragma once

#ifdef LUCID_HAS_DAWN

#include <webgpu/webgpu_cpp.h>
#include <cstdint>
#include <optional>
#include <vector>

#include "units/render_units.hpp"

// Forward declarations from core/
struct Scene;

namespace lucid::gpu {

class DawnContext;

// ----------------------------------------------------------------------------
// PackedScene - flat triangle buffer for GPU consumption
// ----------------------------------------------------------------------------
// Layout per triangle (24 floats / 96 bytes, std430-aligned):
//   v0.xyz, _pad   (vec4)
//   v1.xyz, _pad   (vec4)
//   v2.xyz, _pad   (vec4)
//   n0.xyz, _pad   (vec4)
//   n1.xyz, _pad   (vec4)
//   n2.xyz, smooth_flag (1.0 if smooth normals exist, 0.0 if face normal only)
// `triangles.size() == triangle_count * 24`.
struct PackedScene {
    std::vector<float> triangles;
    uint32_t triangle_count = 0;
};

PackedScene pack_scene_for_debug(const Scene& scene);

// ----------------------------------------------------------------------------
// CameraParamsGpu - uniform layout (64 bytes, std140-aligned)
// ----------------------------------------------------------------------------
// Mirrors the per-pixel ray-gen done in pybind_renderer.hpp::render_debug.
struct CameraParamsGpu {
    float    pos[3];       float _pad0;
    float    fwd[3];       float _pad1;
    float    right[3];     float _pad2;
    float    up[3];        float fov_scale;   // tan(fov / 2)
    float    aspect;
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t full_w;
    uint32_t full_h;
    uint32_t _pad3;
};
static_assert(sizeof(CameraParamsGpu) == 96,
              "CameraParamsGpu must match the WGSL uniform layout (96 bytes)");

// Build CameraParamsGpu from a CPU-side `render::Camera` and a tile spec.
CameraParamsGpu make_camera_params(const render::Position& pos,
                                   const render::Direction& fwd,
                                   const render::Direction& right,
                                   const render::Direction& up,
                                   float fov_rad,
                                   float aspect,
                                   uint32_t tile_x, uint32_t tile_y,
                                   uint32_t tile_w, uint32_t tile_h,
                                   uint32_t full_w, uint32_t full_h);

// ----------------------------------------------------------------------------
// DebugRenderer - cached pipeline + storage buffers
// ----------------------------------------------------------------------------
class DebugRenderer {
public:
    // Builds pipeline / shader / layout against `ctx`'s device. Returns nullopt
    // if shader compilation fails (error logged to stderr).
    static std::optional<DebugRenderer> create(DawnContext& ctx);

    // Render a tile. Returns a flat RGBA buffer of size tile_w * tile_h * 4.
    // The shader does the Y-flip so the output matches CPU `render_debug`.
    // Throws std::runtime_error on dispatch / readback failure.
    std::vector<float> render_normal(DawnContext& ctx,
                                     const PackedScene& scene,
                                     const CameraParamsGpu& cam);

    DebugRenderer(DebugRenderer&&) noexcept = default;
    DebugRenderer& operator=(DebugRenderer&&) noexcept = default;
    DebugRenderer(const DebugRenderer&) = delete;
    DebugRenderer& operator=(const DebugRenderer&) = delete;

private:
    DebugRenderer(wgpu::ShaderModule, wgpu::ComputePipeline, wgpu::BindGroupLayout);

    wgpu::ShaderModule     shader_;
    wgpu::ComputePipeline  pipeline_;
    wgpu::BindGroupLayout  layout_;

    // Cached buffers; grown on demand.
    wgpu::Buffer cam_buf_;
    wgpu::Buffer tri_buf_;    uint64_t tri_buf_capacity_ = 0;
    wgpu::Buffer out_buf_;    uint64_t out_buf_capacity_ = 0;
    wgpu::Buffer stage_buf_;  uint64_t stage_buf_capacity_ = 0;
};

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
