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
#include <memory>
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
// Layout per triangle (40 floats / 160 bytes, std430-aligned):
//   v0.xyz, _pad                (vec4)
//   v1.xyz, _pad                (vec4)
//   v2.xyz, _pad                (vec4)
//   n0.xyz, _pad                (vec4)
//   n1.xyz, _pad                (vec4)
//   n2.xyz, smooth_flag         (vec4)   // 1.0 if smooth normals, else 0.0
//   albedo.rgb,    metallic     (vec4)
//   emission.rgb,  roughness    (vec4)   // roughness clamped to MIN_ROUGHNESS
//   transmission, ior, _, _     (vec4)
//   _pad, _pad, _pad, _pad      (vec4)   // reserved (future per-fragment UV)
// `triangles.size() == triangle_count * 40`.
//
// Environment color / strength live in PathTracerParamsGpu, not here.
//
// `lights` is the unified light buffer. Every entry — POINT, SUN, SPOT, AREA
// or EMISSIVE_MESH triangle — is the same 128-byte `GpuLight`. The shader
// branches on the `type` discriminant; unused fields are zeroed by the packer.
// `selection_pdf` is precomputed = area / total_area; the CDF in `light_cdf`
// implements area-weighted importance sampling for `select_light(u)`.
//
// Light type discriminants (mirrored in WGSL):
enum class GpuLightType : uint32_t {
    Point        = 0,   // delta (radius==0) or sphere (radius>0)
    Sun          = 1,   // delta direction
    Spot         = 2,   // delta position + cone
    AreaRect     = 3,
    AreaDisk     = 4,
    AreaEllipse  = 5,
    EmissiveMesh = 6,   // triangle (v0/v1/v2 carried in primary/normal/right/up slots)
};

struct GpuLight {
    uint32_t type;            float    area;          float    selection_pdf; uint32_t _p0;
    float    position[3];     float    _p1;
    float    emission[3];     float    _p2;
    float    normal[3];       float    radius;
    float    right[3];        float    sizeX;
    float    up[3];           float    sizeY;
    float    v1[3];           float    spotAngle;
    float    v2[3];           float    spotBlend;
};
static_assert(sizeof(GpuLight) == 128,
              "GpuLight must be 128 bytes / 8 vec4s for std430");
//
// BVH (Phase 1c): single global BVH over the flat triangle list. Each node is
// 32 bytes / 2 vec4s (std430). The convention mirrors the CPU BVH: `triCount > 0`
// means leaf, with `triStart` in the `left` slot and `triCount` in the `right`
// slot. Internal nodes have `triCount == 0` and store child node indices.
// Triangles are reordered during packing so that leaves can index the triangle
// buffer directly — no separate triIndices indirection.
struct GpuBvhNode {
    float bmin[3];   int32_t left;             // child idx OR triStart (when leaf)
    float bmax[3];   int32_t right_or_count;   // child idx OR triCount (when leaf)
};
static_assert(sizeof(GpuBvhNode) == 32, "GpuBvhNode must be 32 bytes for std430");

struct PackedPathScene {
    std::vector<float> triangles;
    uint32_t triangle_count = 0;
    // Unified light buffer (replaces the Phase 2a POINT-only `point_lights`).
    // 32 floats / 128 bytes per entry (GpuLight). Includes every native light
    // AND every emissive triangle, in that order; the first
    // `emissive_mesh_light_count` entries are EMISSIVE_MESH triangles.
    std::vector<GpuLight> lights;
    uint32_t light_count = 0;
    // Area-weighted CDF over `lights`, used by select_light(u) for importance
    // sampling. `light_cdf.size() == light_count`. Values in [0,1] monotonic.
    std::vector<float> light_cdf;
    std::vector<GpuBvhNode> bvh_nodes;
    uint32_t bvh_node_count = 0;
    // Monotonic id assigned at pack time. PathTracer compares against the id
    // it last uploaded so the static buffers (tri / lights / bvh) skip
    // WriteBuffer on subsequent dispatches with the same scene. 0 = unset
    // (always re-upload, e.g. for tests that construct PackedPathScene by hand).
    uint64_t cache_id = 0;
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

    // Lights / BVH counts
    uint32_t light_count;       // unified light buffer (POINT/SUN/SPOT/AREA/EMISSIVE)
    uint32_t bvh_node_count;    // 0 → shader skips traversal (empty scene)
    // Integrator selection: 0=simple (BSDF + POINT delta-NEE only; Phase 2a),
    //                       1=nee    (NEE for all light types, no MIS),
    //                       2=mis    (NEE + MIS power heuristic, default).
    uint32_t algorithm;
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
    uint32_t light_count,
    uint32_t bvh_node_count,
    uint32_t algorithm);

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
    //
    // The sync render() and async start/poll/stop API are mutually exclusive;
    // do not interleave calls.
    std::vector<float> render(DawnContext& ctx,
                              const PackedPathScene& scene,
                              const PathTracerParamsGpu& params);

    // ----- Async accumulator API ----------------------------------------------
    //
    // Begins (or restarts) a background path-trace session that keeps firing
    // 1-sample dispatches into an on-GPU accumulator. The Blender viewport
    // calls `poll_async` from the draw thread to fetch the latest snapshot;
    // the GPU work continues independently of the polling rate.
    //
    // Lifetime: the caller owns `scene` for the duration of the async session
    // and must `stop_async()` before invalidating it. `start_async` blocks
    // briefly (uploads buffers, kicks worker) but returns once the session
    // is live. Calling start_async() while a session is running silently
    // stops the old one first.
    void start_async(DawnContext& ctx,
                     const PackedPathScene& scene,
                     const PathTracerParamsGpu& base_params);

    // Stops the worker, joins the thread. No-op if not running. Safe to call
    // from the destructor; in fact the destructor always calls this.
    void stop_async();

    // Non-blocking reset: keep the running worker alive but on its next loop
    // iteration apply `new_base_params`, wipe the accumulator, reset the
    // sample counter, and resume. Intended for camera moves — the common
    // hot path — where the alternative (stop_async + start_async) would
    // join the worker thread (~5-30 ms, occasionally up to the MapAsync
    // timeout) and reallocate buffers, stalling Blender's UI every frame.
    // Caller must ensure dimensions and scene haven't changed; for those
    // cases use stop+start.
    void reset_async(const PathTracerParamsGpu& new_base_params);

    bool is_async_running() const noexcept;
    uint32_t async_samples_completed() const noexcept;
    // Monotonic counter incremented every time the worker publishes a fresh
    // snapshot. Cheap atomic read — Blender uses it to skip the heavy
    // poll_async / pixel-conversion path when nothing's changed.
    uint32_t async_snapshot_revision() const noexcept;

    struct AsyncSnapshot {
        uint32_t samples = 0;          // 0 if no snapshot taken yet
        uint32_t width   = 0;
        uint32_t height  = 0;
        // RGBA, row-major, Y-flipped (same as render()), already divided by
        // `samples`. Empty when samples == 0.
        std::vector<float> pixels;
    };
    // Non-blocking. Returns the latest CPU-side snapshot the worker has
    // produced. If samples == 0, no snapshot exists yet; the caller should
    // keep its existing texture.
    AsyncSnapshot poll_async() const;
    // ---------------------------------------------------------------------------

    // Defined in the .cpp where AsyncState is complete (unique_ptr<AsyncState>
    // needs the destructor in scope to generate ops).
    ~PathTracer();
    PathTracer(PathTracer&&) noexcept;
    PathTracer& operator=(PathTracer&&) noexcept;
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

private:
    PathTracer(wgpu::ShaderModule, wgpu::ComputePipeline, wgpu::BindGroupLayout);

    struct AsyncState;
    std::unique_ptr<AsyncState> async_state_;   // Heap-stable; worker captures it.

    wgpu::ShaderModule    shader_;
    wgpu::ComputePipeline pipeline_;
    wgpu::BindGroupLayout layout_;

    // Cached storage / staging buffers; grown on demand.
    wgpu::Buffer params_buf_;
    wgpu::Buffer tri_buf_;          uint64_t tri_buf_capacity_ = 0;
    wgpu::Buffer lights_buf_;       uint64_t lights_buf_capacity_ = 0;
    wgpu::Buffer light_cdf_buf_;    uint64_t light_cdf_buf_capacity_ = 0;
    wgpu::Buffer bvh_buf_;          uint64_t bvh_buf_capacity_ = 0;
    wgpu::Buffer out_buf_;          uint64_t out_buf_capacity_ = 0;
    wgpu::Buffer stage_buf_;        uint64_t stage_buf_capacity_ = 0;

    // Last successfully uploaded PackedPathScene::cache_id; 0 means stale.
    uint64_t last_scene_cache_id_ = 0;
};

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
