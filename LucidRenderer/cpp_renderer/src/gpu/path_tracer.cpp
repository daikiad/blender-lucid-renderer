/**
 * path_tracer.cpp - GPU path tracer impl (Phase 2a)
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/path_tracer.hpp"
#include "gpu/dawn_context.hpp"
#include "gpu/shader_module.hpp"
#include "gpu/sync.hpp"

#include "core/scene.hpp"
#include "light/light.hpp"  // Scene contains std::vector<Light>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

// Forward decls — implementations live in src/node_evaluator.cpp. Mirror the
// CPU `getMaterialParams()` semantics so all six material sockets are
// constant-folded at uv=(0,0). UV-driven node-tree eval (procedural textures)
// remains out of scope on GPU; matches Phase 2a's albedo+emission pattern.
extern render::AttenuationRGB getAlbedoFromNodeTree(const NodeTree& tree,
                                                    const render::Vec2f& uv);
extern render::RGB3f          getEmissionFromNodeTree(const NodeTree& tree,
                                                       const render::Vec2f& uv);
extern float                  getMetallicFromNodeTree(const NodeTree& tree,
                                                       const render::Vec2f& uv);
extern float                  getRoughnessFromNodeTree(const NodeTree& tree,
                                                        const render::Vec2f& uv);
extern float                  getTransmissionFromNodeTree(const NodeTree& tree,
                                                           const render::Vec2f& uv);
extern float                  getIORFromNodeTree(const NodeTree& tree,
                                                  const render::Vec2f& uv);

namespace lucid::gpu {

namespace {

constexpr uint64_t kMinStorageBytes = 1024;

inline render::Vec3f position_to_vec3(const render::Position& p) {
    return render::displacement_from_origin(p)
        .numerical_value_in(render::si::metre);
}

inline void pack_vec3(std::vector<float>& dst, render::Vec3f v, float pad) {
    dst.push_back(v.x);
    dst.push_back(v.y);
    dst.push_back(v.z);
    dst.push_back(pad);
}

// ----------------------------------------------------------------------------
// Flat BVH builder (median split on longest axis, leaves at <= 4 triangles)
// ----------------------------------------------------------------------------
// Mirrors the CPU `BVH::build` semantics so traversal logic is identical:
//   - leaf when triCount > 0; left = triStart, right = triCount
//   - internal when triCount == 0; left/right = child node indices
// Triangles are reordered to match leaf order so leaves can index the packed
// triangle array directly (no separate triIndices buffer).

struct TriAabb {
    float bmin[3];
    float bmax[3];
    float centroid[3];
};

inline void aabb_union(float (&dst_min)[3], float (&dst_max)[3],
                       const float src_min[3], const float src_max[3]) {
    dst_min[0] = std::min(dst_min[0], src_min[0]);
    dst_min[1] = std::min(dst_min[1], src_min[1]);
    dst_min[2] = std::min(dst_min[2], src_min[2]);
    dst_max[0] = std::max(dst_max[0], src_max[0]);
    dst_max[1] = std::max(dst_max[1], src_max[1]);
    dst_max[2] = std::max(dst_max[2], src_max[2]);
}

// Encoding (32 bytes, std430):
//   leaf:     left = triStart (>= 0), right_or_count = triCount (>= 1)
//   internal: left = left_child (>= 0), right_or_count = -(right_child + 1) (<= -1)
// Detection: `right_or_count > 0` is a leaf; `< 0` is internal. A 0 value never
// occurs because leaves always have count >= 1.
int32_t build_bvh_recursive(std::vector<GpuBvhNode>& nodes,
                            std::vector<int32_t>& tri_order,
                            const std::vector<TriAabb>& tri_aabbs,
                            int32_t start, int32_t end) {
    const int32_t node_idx = static_cast<int32_t>(nodes.size());
    nodes.emplace_back();

    // Compute bounds for this node.
    float bmin[3] = { 1e30f,  1e30f,  1e30f};
    float bmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int32_t i = start; i < end; ++i) {
        const TriAabb& a = tri_aabbs[tri_order[i]];
        aabb_union(bmin, bmax, a.bmin, a.bmax);
    }
    {
        auto& node = nodes[node_idx];
        node.bmin[0] = bmin[0]; node.bmin[1] = bmin[1]; node.bmin[2] = bmin[2];
        node.bmax[0] = bmax[0]; node.bmax[1] = bmax[1]; node.bmax[2] = bmax[2];
    }

    const int32_t count = end - start;
    if (count <= 4) {
        nodes[node_idx].left           = start;
        nodes[node_idx].right_or_count = count;   // > 0 → leaf
        return node_idx;
    }

    // Longest-axis median split.
    const float ex = bmax[0] - bmin[0];
    const float ey = bmax[1] - bmin[1];
    const float ez = bmax[2] - bmin[2];
    int axis = 0;
    if (ey > ex)                          axis = 1;
    if (ez > (axis == 0 ? ex : ey))       axis = 2;

    const int32_t mid = (start + end) / 2;
    std::nth_element(tri_order.begin() + start,
                     tri_order.begin() + mid,
                     tri_order.begin() + end,
                     [&](int32_t a, int32_t b) {
                         return tri_aabbs[a].centroid[axis]
                              < tri_aabbs[b].centroid[axis];
                     });

    const int32_t left  = build_bvh_recursive(nodes, tri_order, tri_aabbs, start, mid);
    const int32_t right = build_bvh_recursive(nodes, tri_order, tri_aabbs, mid, end);
    nodes[node_idx].left           = left;
    nodes[node_idx].right_or_count = -(right + 1);   // < 0 → internal
    return node_idx;
}

// Build a global BVH over `tri_aabbs`. Returns the permutation `tri_order`
// such that `tri_order[i]` gives the original triangle index that should land
// at packed position `i`.
void build_global_bvh(std::vector<GpuBvhNode>& nodes,
                      std::vector<int32_t>& tri_order,
                      const std::vector<TriAabb>& tri_aabbs) {
    const int32_t n = static_cast<int32_t>(tri_aabbs.size());
    tri_order.resize(n);
    for (int32_t i = 0; i < n; ++i) tri_order[i] = i;
    if (n == 0) return;
    nodes.reserve(2 * static_cast<size_t>(n));
    build_bvh_recursive(nodes, tri_order, tri_aabbs, 0, n);
}

}  // namespace

// ----------------------------------------------------------------------------
// PackedPathScene / params packing
// ----------------------------------------------------------------------------

PackedPathScene pack_scene_for_path_tracer(const Scene& scene) {
    PackedPathScene out;
    // Monotonic id starting at 1 (0 = "unset / always re-upload"). Each pack
    // call gets a fresh id, even if the resulting buffers happen to look
    // identical — comparisons in PathTracer::render are by id, not content.
    static std::atomic<uint64_t> next_cache_id{1};
    out.cache_id = next_cache_id.fetch_add(1, std::memory_order_relaxed);

    uint32_t total = 0;
    for (const auto& m : scene.meshes) {
        total += static_cast<uint32_t>(m.triangles.size());
    }
    out.triangle_count = total;
    out.triangles.reserve(static_cast<size_t>(total) * 40);

    // ---- Point lights (NEE) ----
    // Pack each POINT light as 12 floats / 48 B (std430):
    //   pos.xyz,      radius      (vec4)
    //   emission.xyz, area        (vec4)
    //   _pad, _pad, _pad, _pad    (vec4, reserved for future light_kind)
    //
    // emission is already premultiplied on the CPU side ([pybind_renderer.hpp:1278-1296]):
    //   radius > 0 (sphere): color * energy / (π * area)   — Lambertian sphere radiance
    //   radius = 0 (delta):  color * energy / (4π)         — point intensity I [W/sr]
    // The shader does NOT need raw `energy`; it branches on radius and applies
    // the correct measure-conversion (sphere area-pdf vs delta 1/d²).
    for (const auto& light : scene.nativeLights) {
        if (light.type != LightType::POINT) continue;
        const auto p = position_to_vec3(light.position);
        const float radius = light.radius.numerical_value_in(mp_units::si::metre);
        const float area   = light.area.numerical_value_in(
            mp_units::square(mp_units::si::metre));
        out.point_lights.push_back(p.x);
        out.point_lights.push_back(p.y);
        out.point_lights.push_back(p.z);
        out.point_lights.push_back(radius);
        const float er = light.emission.r.numerical_value_in(render::radiance_unit);
        const float eg = light.emission.g.numerical_value_in(render::radiance_unit);
        const float eb = light.emission.b.numerical_value_in(render::radiance_unit);
        out.point_lights.push_back(er);
        out.point_lights.push_back(eg);
        out.point_lights.push_back(eb);
        out.point_lights.push_back(area);
        out.point_lights.push_back(0.0f);
        out.point_lights.push_back(0.0f);
        out.point_lights.push_back(0.0f);
        out.point_lights.push_back(0.0f);
        ++out.point_light_count;
    }

    // Pack into a temp buffer first; BVH build below picks an ordering and we
    // then copy into out.triangles in that order.
    std::vector<float>   tmp_triangles;       // 40 floats per tri (unordered)
    std::vector<TriAabb> tri_aabbs;
    tmp_triangles.reserve(static_cast<size_t>(total) * 40);
    tri_aabbs.reserve(total);

    // Mirrors CPU `bsdf::MIN_ROUGHNESS` ([bsdf/ggx.hpp:24]). Clamped at pack
    // time so the shader can assume `roughness >= MIN_ROUGHNESS` everywhere.
    constexpr float kMinRoughness = 0.01f;

    for (const auto& mesh : scene.meshes) {
        // Evaluate all six Principled BSDF sockets once per mesh, mirroring
        // CPU `getMaterialParams()` ([integrator/path_tracer.hpp:43-63]) but
        // at uv=(0,0). Per-fragment UV-driven node-tree evaluation (procedural
        // textures) is out of scope; the same uv=(0,0) contract already used
        // for albedo+emission in Phase 2a is extended to the other four.
        render::AttenuationRGB albedo;
        render::RGB3f          emission_rgb;
        float                  metallic;
        float                  roughness;
        float                  transmission;
        float                  ior;
        const bool use_nodes = mesh.material.useNodes && mesh.material.nodeTree.valid;
        if (use_nodes) {
            const render::Vec2f uv{0.0f, 0.0f};
            albedo       = getAlbedoFromNodeTree(mesh.material.nodeTree, uv);
            emission_rgb = getEmissionFromNodeTree(mesh.material.nodeTree, uv);
            metallic     = getMetallicFromNodeTree(mesh.material.nodeTree, uv);
            roughness    = getRoughnessFromNodeTree(mesh.material.nodeTree, uv);
            transmission = getTransmissionFromNodeTree(mesh.material.nodeTree, uv);
            ior          = getIORFromNodeTree(mesh.material.nodeTree, uv);
        } else {
            albedo = mesh.material.albedo;
            emission_rgb.r = mesh.material.emission.r.numerical_value_in(render::radiance_unit);
            emission_rgb.g = mesh.material.emission.g.numerical_value_in(render::radiance_unit);
            emission_rgb.b = mesh.material.emission.b.numerical_value_in(render::radiance_unit);
            metallic     = mesh.material.metallic;
            roughness    = mesh.material.roughness;
            transmission = mesh.material.transmission;
            ior          = mesh.material.ior;
        }
        roughness = std::max(roughness, kMinRoughness);

        const float albedo_r   = albedo.r;
        const float albedo_g   = albedo.g;
        const float albedo_b   = albedo.b;
        const float emission_r = emission_rgb.r;
        const float emission_g = emission_rgb.g;
        const float emission_b = emission_rgb.b;

        for (const auto& tri : mesh.triangles) {
            const auto v0 = position_to_vec3(mesh.vertices[tri.i0]);
            const auto v1 = position_to_vec3(mesh.vertices[tri.i1]);
            const auto v2 = position_to_vec3(mesh.vertices[tri.i2]);

            pack_vec3(tmp_triangles, v0, 0.0f);
            pack_vec3(tmp_triangles, v1, 0.0f);
            pack_vec3(tmp_triangles, v2, 0.0f);

            const float smooth_flag = tri.smooth ? 1.0f : 0.0f;
            pack_vec3(tmp_triangles, tri.n0.vec(), 0.0f);
            pack_vec3(tmp_triangles, tri.n1.vec(), 0.0f);
            pack_vec3(tmp_triangles, tri.n2.vec(), smooth_flag);

            // Material packed into 3 vec4s + 1 reserved vec4 (12+4 = 16 floats).
            // Layout: (albedo.rgb, metallic), (emission.rgb, roughness),
            //         (transmission, ior, _, _), (_, _, _, _ reserved).
            tmp_triangles.push_back(albedo_r);
            tmp_triangles.push_back(albedo_g);
            tmp_triangles.push_back(albedo_b);
            tmp_triangles.push_back(metallic);
            tmp_triangles.push_back(emission_r);
            tmp_triangles.push_back(emission_g);
            tmp_triangles.push_back(emission_b);
            tmp_triangles.push_back(roughness);
            tmp_triangles.push_back(transmission);
            tmp_triangles.push_back(ior);
            tmp_triangles.push_back(0.0f);
            tmp_triangles.push_back(0.0f);
            // Reserved vec4 (e.g. future per-fragment UV data, normal-map tangent).
            tmp_triangles.push_back(0.0f);
            tmp_triangles.push_back(0.0f);
            tmp_triangles.push_back(0.0f);
            tmp_triangles.push_back(0.0f);

            TriAabb a;
            a.bmin[0] = std::min(std::min(v0.x, v1.x), v2.x);
            a.bmin[1] = std::min(std::min(v0.y, v1.y), v2.y);
            a.bmin[2] = std::min(std::min(v0.z, v1.z), v2.z);
            a.bmax[0] = std::max(std::max(v0.x, v1.x), v2.x);
            a.bmax[1] = std::max(std::max(v0.y, v1.y), v2.y);
            a.bmax[2] = std::max(std::max(v0.z, v1.z), v2.z);
            a.centroid[0] = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
            a.centroid[1] = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
            a.centroid[2] = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);
            tri_aabbs.push_back(a);
        }
    }

    // ---- Build BVH and reorder triangles to match leaf order ----
    std::vector<int32_t> tri_order;
    build_global_bvh(out.bvh_nodes, tri_order, tri_aabbs);
    out.bvh_node_count = static_cast<uint32_t>(out.bvh_nodes.size());

    out.triangles.resize(tmp_triangles.size());
    for (size_t new_idx = 0; new_idx < tri_order.size(); ++new_idx) {
        const size_t old_idx = static_cast<size_t>(tri_order[new_idx]);
        std::memcpy(&out.triangles[new_idx * 40],
                    &tmp_triangles[old_idx * 40],
                    40 * sizeof(float));
    }
    return out;
}

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
    uint32_t point_light_count,
    uint32_t bvh_node_count) {
    PathTracerParamsGpu p{};
    const auto pc = position_to_vec3(pos);
    p.pos[0] = pc.x; p.pos[1] = pc.y; p.pos[2] = pc.z;
    const auto f = fwd.vec();
    p.fwd[0] = f.x; p.fwd[1] = f.y; p.fwd[2] = f.z;
    const auto r = right.vec();
    p.right[0] = r.x; p.right[1] = r.y; p.right[2] = r.z;
    const auto u = up.vec();
    p.up[0] = u.x; p.up[1] = u.y; p.up[2] = u.z;
    p.fov_scale = std::tan(fov_rad * 0.5f);
    p.aspect    = aspect;
    p.tile_x = tile_x; p.tile_y = tile_y;
    p.tile_w = tile_w; p.tile_h = tile_h;
    p.full_w = full_w; p.full_h = full_h;
    p.samples       = samples;
    p.sample_offset = sample_offset;
    p.max_bounces   = max_bounces;
    p.frame_seed    = frame_seed;
    p.env_color[0]  = env_color[0];
    p.env_color[1]  = env_color[1];
    p.env_color[2]  = env_color[2];
    p.env_strength  = env_strength;
    p.point_light_count = point_light_count;
    p.bvh_node_count    = bvh_node_count;
    return p;
}

// ----------------------------------------------------------------------------
// PathTracer
// ----------------------------------------------------------------------------

PathTracer::PathTracer(wgpu::ShaderModule s,
                       wgpu::ComputePipeline p,
                       wgpu::BindGroupLayout l)
    : shader_(std::move(s)), pipeline_(std::move(p)), layout_(std::move(l)) {}

std::optional<PathTracer> PathTracer::create(DawnContext& ctx) {
    wgpu::ShaderModule shader;
    try {
        shader = load_wgsl(ctx.instance(), ctx.device(), "path_trace.wgsl");
    } catch (const ShaderCompilationError& e) {
        std::cerr << "[PathTracer] Shader compile failed: " << e.what() << std::endl;
        return std::nullopt;
    }

    wgpu::BindGroupLayoutEntry entries[5] = {};
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[0].buffer.minBindingSize = sizeof(PathTracerParamsGpu);

    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Compute;
    entries[2].buffer.type = wgpu::BufferBindingType::Storage;

    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Compute;
    entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    entries[4].binding = 4;
    entries[4].visibility = wgpu::ShaderStage::Compute;
    entries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutDescriptor bgl{};
    bgl.entryCount = 5;
    bgl.entries    = entries;
    wgpu::BindGroupLayout layout = ctx.device().CreateBindGroupLayout(&bgl);

    wgpu::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayoutCount = 1;
    pl.bindGroupLayouts     = &layout;
    wgpu::PipelineLayout pipeline_layout = ctx.device().CreatePipelineLayout(&pl);

    wgpu::ComputePipelineDescriptor cp{};
    cp.layout             = pipeline_layout;
    cp.compute.module     = shader;
    cp.compute.entryPoint = wgpu::StringView{"main", 4};
    wgpu::ComputePipeline pipeline = ctx.device().CreateComputePipeline(&cp);

    return PathTracer(std::move(shader), std::move(pipeline), std::move(layout));
}

std::vector<float> PathTracer::render(DawnContext& ctx,
                                      const PackedPathScene& scene,
                                      const PathTracerParamsGpu& params) {
    if (params.tile_w == 0 || params.tile_h == 0) return {};

    const uint64_t pixel_count = uint64_t{params.tile_w} * uint64_t{params.tile_h};
    const uint64_t out_bytes   = pixel_count * 4 * sizeof(float);

    // ---- Params uniform buffer (constant size) ----
    if (!params_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = sizeof(PathTracerParamsGpu);
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        params_buf_ = ctx.device().CreateBuffer(&d);
    }
    ctx.queue().WriteBuffer(params_buf_, 0, &params, sizeof(params));

    // ---- Static buffers (tri / lights / bvh): skip WriteBuffer when the
    //      packed scene's cache_id matches what we last uploaded. If any
    //      buffer had to grow we force re-upload regardless. cache_id == 0
    //      means "unset" (e.g. hand-built test scenes) → always re-upload.
    const bool scene_id_matches =
        scene.cache_id != 0 && scene.cache_id == last_scene_cache_id_;

    // ---- Triangle storage (grow on demand) ----
    const uint64_t tri_bytes = static_cast<uint64_t>(scene.triangles.size())
                               * sizeof(float);
    const uint64_t tri_alloc = std::max(tri_bytes, kMinStorageBytes);
    bool tri_buf_new = false;
    if (tri_alloc > tri_buf_capacity_ || !tri_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = tri_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        tri_buf_          = ctx.device().CreateBuffer(&d);
        tri_buf_capacity_ = tri_alloc;
        tri_buf_new       = true;
    }
    if (tri_bytes > 0 && (tri_buf_new || !scene_id_matches)) {
        ctx.queue().WriteBuffer(tri_buf_, 0, scene.triangles.data(), tri_bytes);
    }

    // ---- Point-light storage (grow on demand) ----
    const uint64_t pl_bytes = static_cast<uint64_t>(scene.point_lights.size())
                              * sizeof(float);
    const uint64_t pl_alloc = std::max(pl_bytes, kMinStorageBytes);
    bool pl_buf_new = false;
    if (pl_alloc > point_lights_buf_capacity_ || !point_lights_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = pl_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        point_lights_buf_          = ctx.device().CreateBuffer(&d);
        point_lights_buf_capacity_ = pl_alloc;
        pl_buf_new                 = true;
    }
    if (pl_bytes > 0 && (pl_buf_new || !scene_id_matches)) {
        ctx.queue().WriteBuffer(point_lights_buf_, 0,
                                scene.point_lights.data(), pl_bytes);
    }

    // ---- BVH node storage (grow on demand) ----
    const uint64_t bvh_bytes = static_cast<uint64_t>(scene.bvh_nodes.size())
                               * sizeof(GpuBvhNode);
    const uint64_t bvh_alloc = std::max(bvh_bytes, kMinStorageBytes);
    bool bvh_buf_new = false;
    if (bvh_alloc > bvh_buf_capacity_ || !bvh_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = bvh_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        bvh_buf_          = ctx.device().CreateBuffer(&d);
        bvh_buf_capacity_ = bvh_alloc;
        bvh_buf_new       = true;
    }
    if (bvh_bytes > 0 && (bvh_buf_new || !scene_id_matches)) {
        ctx.queue().WriteBuffer(bvh_buf_, 0, scene.bvh_nodes.data(), bvh_bytes);
    }

    // All three uploads (or skips) succeeded — remember the id so we can skip
    // next time. Setting it to 0 is the canonical "stale" marker if the test
    // happened to pass a cache_id == 0 scene.
    last_scene_cache_id_ = scene.cache_id;

    // ---- Output storage (grow on demand) ----
    if (out_bytes > out_buf_capacity_ || !out_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = std::max(out_bytes, kMinStorageBytes);
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        out_buf_          = ctx.device().CreateBuffer(&d);
        out_buf_capacity_ = d.size;
    }

    // ---- Staging (MapRead) (grow on demand) ----
    if (out_bytes > stage_buf_capacity_ || !stage_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = std::max(out_bytes, kMinStorageBytes);
        d.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        stage_buf_          = ctx.device().CreateBuffer(&d);
        stage_buf_capacity_ = d.size;
    }

    // ---- Bind group (rebuilt per call) ----
    wgpu::BindGroupEntry bg_entries[5] = {};
    bg_entries[0].binding = 0;
    bg_entries[0].buffer  = params_buf_;
    bg_entries[0].offset  = 0;
    bg_entries[0].size    = sizeof(PathTracerParamsGpu);
    bg_entries[1].binding = 1;
    bg_entries[1].buffer  = tri_buf_;
    bg_entries[1].offset  = 0;
    bg_entries[1].size    = tri_buf_capacity_;
    bg_entries[2].binding = 2;
    bg_entries[2].buffer  = out_buf_;
    bg_entries[2].offset  = 0;
    bg_entries[2].size    = out_bytes;
    bg_entries[3].binding = 3;
    bg_entries[3].buffer  = point_lights_buf_;
    bg_entries[3].offset  = 0;
    bg_entries[3].size    = point_lights_buf_capacity_;
    bg_entries[4].binding = 4;
    bg_entries[4].buffer  = bvh_buf_;
    bg_entries[4].offset  = 0;
    bg_entries[4].size    = bvh_buf_capacity_;

    wgpu::BindGroupDescriptor bg{};
    bg.layout     = layout_;
    bg.entryCount = 5;
    bg.entries    = bg_entries;
    wgpu::BindGroup bind_group = ctx.device().CreateBindGroup(&bg);

    // ---- Dispatch ----
    wgpu::CommandEncoder encoder = ctx.device().CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline_);
        pass.SetBindGroup(0, bind_group);
        const uint32_t wg_x = (params.tile_w + 7) / 8;
        const uint32_t wg_y = (params.tile_h + 7) / 8;
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(out_buf_, 0, stage_buf_, 0, out_bytes);
    wgpu::CommandBuffer cmd = encoder.Finish();
    ctx.queue().Submit(1, &cmd);

    // ---- Map + readback (timeout-bounded) ----
    std::string map_err;
    wgpu::Future map_future = stage_buf_.MapAsync(
        wgpu::MapMode::Read, 0, out_bytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [&map_err](wgpu::MapAsyncStatus status, wgpu::StringView msg) {
            if (status != wgpu::MapAsyncStatus::Success) {
                map_err.assign(msg.data, msg.length);
            }
        });
    if (!wait_for_with_timeout(ctx.instance(), map_future)) {
        throw std::runtime_error("PathTracer::render: MapAsync timed out (>5s)");
    }
    if (!map_err.empty()) {
        throw std::runtime_error("PathTracer::render: MapAsync failed: " + map_err);
    }

    const float* mapped = static_cast<const float*>(
        stage_buf_.GetConstMappedRange(0, out_bytes));
    if (!mapped) {
        throw std::runtime_error("PathTracer::render: GetConstMappedRange returned null");
    }
    std::vector<float> result(mapped, mapped + pixel_count * 4);
    stage_buf_.Unmap();
    return result;
}

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
