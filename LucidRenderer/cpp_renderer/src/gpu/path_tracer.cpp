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
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

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

    // ---- Native lights: POINT / SUN / SPOT / AREA into the unified buffer ----
    // emission for native lights is already pre-multiplied on the CPU JSON
    // parser side ([pybind_renderer.hpp:1261-1340]) so the shader consumes
    // light.emission directly and does NOT need to divide by 4π again.
    auto emit_native = [&out](const Light& light) {
        GpuLight g{};
        const auto p = render::displacement_from_origin(light.position)
                          .numerical_value_in(render::si::metre);
        g.position[0] = p.x; g.position[1] = p.y; g.position[2] = p.z;
        g.emission[0] = light.emission.r.numerical_value_in(render::radiance_unit);
        g.emission[1] = light.emission.g.numerical_value_in(render::radiance_unit);
        g.emission[2] = light.emission.b.numerical_value_in(render::radiance_unit);
        const auto n = light.normal.vec();
        g.normal[0] = n.x; g.normal[1] = n.y; g.normal[2] = n.z;
        g.area   = light.area.numerical_value_in(
            mp_units::square(mp_units::si::metre));
        g.radius = light.radius.numerical_value_in(mp_units::si::metre);
        switch (light.type) {
            case LightType::POINT:
                g.type = static_cast<uint32_t>(GpuLightType::Point);
                break;
            case LightType::SUN:
                g.type = static_cast<uint32_t>(GpuLightType::Sun);
                break;
            case LightType::SPOT: {
                g.type = static_cast<uint32_t>(GpuLightType::Spot);
                g.spotAngle = light.spotAngle.numerical_value_in(mp_units::si::radian);
                g.spotBlend = light.spotBlend;
                break;
            }
            case LightType::AREA: {
                switch (light.shape) {
                    case AreaLightShape::DISK:
                        g.type = static_cast<uint32_t>(GpuLightType::AreaDisk); break;
                    case AreaLightShape::ELLIPSE:
                        g.type = static_cast<uint32_t>(GpuLightType::AreaEllipse); break;
                    default:  // SQUARE, RECTANGLE
                        g.type = static_cast<uint32_t>(GpuLightType::AreaRect); break;
                }
                const auto r = light.right.vec();
                const auto u = light.up.vec();
                g.right[0] = r.x; g.right[1] = r.y; g.right[2] = r.z;
                g.up[0]    = u.x; g.up[1]    = u.y; g.up[2]    = u.z;
                g.sizeX    = light.sizeX.numerical_value_in(mp_units::si::metre);
                g.sizeY    = light.sizeY.numerical_value_in(mp_units::si::metre);
                break;
            }
            default:
                return;  // EMISSIVE_MESH (shouldn't be in nativeLights but skip if so)
        }
        out.lights.push_back(g);
    };
    for (const auto& light : scene.nativeLights) emit_native(light);
    const uint32_t native_light_count = static_cast<uint32_t>(out.lights.size());

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

        // A mesh is "emissive" if any channel of its constant-folded emission
        // is above the same threshold the shader uses to detect an emissive
        // hit. All triangles in such a mesh are added to the light buffer as
        // EMISSIVE_MESH entries; we cache the light_idx for each into the
        // triangle's reserved vec4 slot (i32 reinterpreted as f32 via bitcast)
        // so the integrator can compute MIS weights against pdf_light when a
        // BSDF sample lands on the surface.
        const bool mesh_is_emissive =
            emission_r > 1e-6f || emission_g > 1e-6f || emission_b > 1e-6f;

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

            // Reserved vec4: slot[0] = light_idx (-1 if not emissive). Stored
            // as f32 bits so the std430 alignment / Triangle struct doesn't
            // need a mixed-type layout; shader reads via bitcast<i32>(f32).
            int32_t light_idx = -1;
            if (mesh_is_emissive) {
                // Triangle area: 0.5 * |cross(v1-v0, v2-v0)|.
                const float e1x = v1.x - v0.x;
                const float e1y = v1.y - v0.y;
                const float e1z = v1.z - v0.z;
                const float e2x = v2.x - v0.x;
                const float e2y = v2.y - v0.y;
                const float e2z = v2.z - v0.z;
                const float cx = e1y * e2z - e1z * e2y;
                const float cy = e1z * e2x - e1x * e2z;
                const float cz = e1x * e2y - e1y * e2x;
                const float tri_area = 0.5f * std::sqrt(cx*cx + cy*cy + cz*cz);
                if (tri_area > 1e-12f) {
                    GpuLight g{};
                    g.type        = static_cast<uint32_t>(GpuLightType::EmissiveMesh);
                    g.area        = tri_area;
                    g.position[0] = v0.x; g.position[1] = v0.y; g.position[2] = v0.z;
                    g.emission[0] = emission_r;
                    g.emission[1] = emission_g;
                    g.emission[2] = emission_b;
                    const auto fn = tri.faceNormal.vec();
                    g.normal[0] = fn.x; g.normal[1] = fn.y; g.normal[2] = fn.z;
                    g.v1[0] = v1.x; g.v1[1] = v1.y; g.v1[2] = v1.z;
                    g.v2[0] = v2.x; g.v2[1] = v2.y; g.v2[2] = v2.z;
                    light_idx = static_cast<int32_t>(out.lights.size());
                    out.lights.push_back(g);
                }
            }
            tmp_triangles.push_back(std::bit_cast<float>(light_idx));
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

    // ---- Light selection: area-weighted CDF ----
    // CPU sceneLights ([scene_lights.hpp:114-121]) uses a CDF over light areas
    // for importance sampling. SUN/SPOT have area=1 (symbolic) so they get
    // their fair share; emissive triangles and area lights weight by physical
    // surface area.
    out.light_count = static_cast<uint32_t>(out.lights.size());
    out.light_cdf.resize(out.light_count);
    (void)native_light_count;  // silence "unused" — useful for debugging
    if (out.light_count > 0) {
        float total_area = 0.0f;
        for (const auto& l : out.lights) total_area += l.area;
        if (total_area < 1e-12f) {
            // Degenerate; assign uniform selection.
            const float inv_n = 1.0f / static_cast<float>(out.light_count);
            float cum = 0.0f;
            for (size_t i = 0; i < out.lights.size(); ++i) {
                out.lights[i].selection_pdf = inv_n;
                cum += inv_n;
                out.light_cdf[i] = cum;
            }
        } else {
            const float inv_total = 1.0f / total_area;
            float cum = 0.0f;
            for (size_t i = 0; i < out.lights.size(); ++i) {
                const float p = out.lights[i].area * inv_total;
                out.lights[i].selection_pdf = p;
                cum += p;
                out.light_cdf[i] = cum;
            }
            // Clamp the tail to exactly 1.0 — guards binary search at u≈1.
            if (!out.light_cdf.empty()) out.light_cdf.back() = 1.0f;
        }
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
    uint32_t light_count,
    uint32_t bvh_node_count,
    uint32_t algorithm) {
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
    p.light_count    = light_count;
    p.bvh_node_count = bvh_node_count;
    p.algorithm      = algorithm;
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

    wgpu::BindGroupLayoutEntry entries[6] = {};
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

    // Binding 3: unified light buffer (was POINT-only; now POINT/SUN/SPOT/AREA/EMISSIVE_MESH)
    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Compute;
    entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    entries[4].binding = 4;
    entries[4].visibility = wgpu::ShaderStage::Compute;
    entries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // Binding 5: light_cdf (f32 per light, area-weighted CDF for select_light)
    entries[5].binding = 5;
    entries[5].visibility = wgpu::ShaderStage::Compute;
    entries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutDescriptor bgl{};
    bgl.entryCount = 6;
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

    // ---- Unified light buffer (grow on demand) ----
    const uint64_t lt_bytes = static_cast<uint64_t>(scene.lights.size())
                              * sizeof(GpuLight);
    const uint64_t lt_alloc = std::max(lt_bytes, kMinStorageBytes);
    bool lt_buf_new = false;
    if (lt_alloc > lights_buf_capacity_ || !lights_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = lt_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        lights_buf_          = ctx.device().CreateBuffer(&d);
        lights_buf_capacity_ = lt_alloc;
        lt_buf_new           = true;
    }
    if (lt_bytes > 0 && (lt_buf_new || !scene_id_matches)) {
        ctx.queue().WriteBuffer(lights_buf_, 0,
                                scene.lights.data(), lt_bytes);
    }

    // ---- Light selection CDF (grow on demand) ----
    const uint64_t cdf_bytes = static_cast<uint64_t>(scene.light_cdf.size())
                               * sizeof(float);
    const uint64_t cdf_alloc = std::max(cdf_bytes, kMinStorageBytes);
    bool cdf_buf_new = false;
    if (cdf_alloc > light_cdf_buf_capacity_ || !light_cdf_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = cdf_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        light_cdf_buf_          = ctx.device().CreateBuffer(&d);
        light_cdf_buf_capacity_ = cdf_alloc;
        cdf_buf_new             = true;
    }
    if (cdf_bytes > 0 && (cdf_buf_new || !scene_id_matches)) {
        ctx.queue().WriteBuffer(light_cdf_buf_, 0,
                                scene.light_cdf.data(), cdf_bytes);
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
    // `CopyDst` lets us ClearBuffer this before each dispatch — the shader is
    // now additive (mirrors the async accumulator path), so the sync API has
    // to wipe the previous contribution to keep its "render N samples in one
    // call" contract.
    if (out_bytes > out_buf_capacity_ || !out_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = std::max(out_bytes, kMinStorageBytes);
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc
                | wgpu::BufferUsage::CopyDst;
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
    wgpu::BindGroupEntry bg_entries[6] = {};
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
    bg_entries[3].buffer  = lights_buf_;
    bg_entries[3].offset  = 0;
    bg_entries[3].size    = lights_buf_capacity_;
    bg_entries[4].binding = 4;
    bg_entries[4].buffer  = bvh_buf_;
    bg_entries[4].offset  = 0;
    bg_entries[4].size    = bvh_buf_capacity_;
    bg_entries[5].binding = 5;
    bg_entries[5].buffer  = light_cdf_buf_;
    bg_entries[5].offset  = 0;
    bg_entries[5].size    = light_cdf_buf_capacity_;

    wgpu::BindGroupDescriptor bg{};
    bg.layout     = layout_;
    bg.entryCount = 6;
    bg.entries    = bg_entries;
    wgpu::BindGroup bind_group = ctx.device().CreateBindGroup(&bg);

    // ---- Dispatch ----
    wgpu::CommandEncoder encoder = ctx.device().CreateCommandEncoder();
    // Zero out_buf_ first — shader is additive, so without a wipe each render()
    // call would keep adding on top of whatever the last call left behind.
    encoder.ClearBuffer(out_buf_, 0, out_bytes);
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

// ============================================================================
// Async accumulator session
// ============================================================================
//
// A worker thread continuously submits 1-sample dispatches into a persistent
// on-GPU accumulator buffer. Periodically it reads back the accumulator and
// normalises (sum / sample_count) into a CPU-side snapshot, protected by a
// mutex. The Blender viewport polls this snapshot with `poll_async` from the
// draw thread; the GPU work continues independently of the polling rate.
//
// Lifetime: AsyncState is heap-allocated and owned by `PathTracer::async_state_`
// via unique_ptr — its address is stable even if PathTracer is moved during
// create()→optional. The worker captures `this` (the AsyncState pointer).

struct PathTracer::AsyncState {
    // ---- Refcount-copied handles from PathTracer at start_async time ----
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout layout;
    wgpu::Buffer          params_buf;
    wgpu::Buffer          tri_buf;          uint64_t tri_buf_size = 0;
    wgpu::Buffer          lights_buf;       uint64_t lights_buf_size = 0;
    wgpu::Buffer          light_cdf_buf;    uint64_t light_cdf_buf_size = 0;
    wgpu::Buffer          bvh_buf;          uint64_t bvh_buf_size = 0;

    // ---- Owned by this session ----
    wgpu::Buffer          accum_buf;        uint64_t accum_buf_capacity = 0;
    wgpu::Buffer          stage_buf;        uint64_t stage_buf_capacity = 0;
    wgpu::BindGroup       bind_group;       // cached, references the above

    // ---- Session params ----
    DawnContext*          ctx = nullptr;
    // base_params is read by the worker every dispatch (camera + render bounds
    // live here) and replaced by reset_async on the main thread. Guarded by
    // base_params_mu so the worker always sees a consistent set of fields.
    mutable std::mutex    base_params_mu;
    PathTracerParamsGpu   base_params{};
    uint32_t              width  = 0;
    uint32_t              height = 0;
    uint64_t              accum_bytes = 0;  // width*height*4*sizeof(float)

    // ---- Worker thread / synchronisation ----
    std::thread           worker;
    std::atomic<bool>     stop_requested{false};
    std::atomic<uint32_t> samples_completed{0};
    // Signal flag set by reset_async; worker consumes at the top of each loop
    // iteration. acquire-release pairs with the base_params write so the
    // worker sees the new params once it observes the flag.
    std::atomic<bool>     reset_pending{false};

    // ---- Snapshot for Blender poll ----
    mutable std::mutex    snapshot_mu;
    std::vector<float>    snapshot_pixels;        // RGBA averaged
    uint32_t              snapshot_samples = 0;
    uint32_t              snapshot_w = 0, snapshot_h = 0;
    // Monotonic revision — bumped on each successful take_snapshot. Lets the
    // viewport skip the heavy poll path when nothing's changed since last frame.
    std::atomic<uint32_t> snapshot_revision{0};

    void worker_loop();
    bool submit_one_sample();
    bool take_snapshot();
};

void PathTracer::AsyncState::worker_loop() {
    // Build the bind group once — it references buffers that don't change.
    {
        wgpu::BindGroupLayoutEntry dummy{};
        (void)dummy;
        wgpu::BindGroupEntry bg_entries[6] = {};
        bg_entries[0].binding = 0;
        bg_entries[0].buffer  = params_buf;
        bg_entries[0].offset  = 0;
        bg_entries[0].size    = sizeof(PathTracerParamsGpu);
        bg_entries[1].binding = 1;
        bg_entries[1].buffer  = tri_buf;
        bg_entries[1].offset  = 0;
        bg_entries[1].size    = tri_buf_size;
        bg_entries[2].binding = 2;
        bg_entries[2].buffer  = accum_buf;
        bg_entries[2].offset  = 0;
        bg_entries[2].size    = accum_bytes;
        bg_entries[3].binding = 3;
        bg_entries[3].buffer  = lights_buf;
        bg_entries[3].offset  = 0;
        bg_entries[3].size    = lights_buf_size;
        bg_entries[4].binding = 4;
        bg_entries[4].buffer  = bvh_buf;
        bg_entries[4].offset  = 0;
        bg_entries[4].size    = bvh_buf_size;
        bg_entries[5].binding = 5;
        bg_entries[5].buffer  = light_cdf_buf;
        bg_entries[5].offset  = 0;
        bg_entries[5].size    = light_cdf_buf_size;

        wgpu::BindGroupDescriptor bg{};
        bg.layout     = layout;
        bg.entryCount = 6;
        bg.entries    = bg_entries;
        bind_group    = ctx->device().CreateBindGroup(&bg);
    }

    // Throttle parameter: snapshot every K dispatches. With ~300 samples/sec
    // on a 1000-tri scene, K=16 gives ~20Hz snapshot updates — plenty for
    // the viewport's redraw cadence.
    constexpr uint32_t kSnapshotEveryNSamples = 16;
    uint32_t since_last_snapshot = 0;

    while (!stop_requested.load(std::memory_order_relaxed)) {
        // Check for an in-place reset signalled by reset_async. acquire so we
        // observe the new base_params (release-stored by the main thread).
        if (reset_pending.exchange(false, std::memory_order_acquire)) {
            // Wipe the accumulator and reset the sample counter. base_params
            // was already updated by the main thread before setting the flag.
            samples_completed.store(0, std::memory_order_release);
            {
                wgpu::CommandEncoder enc = ctx->device().CreateCommandEncoder();
                enc.ClearBuffer(accum_buf, 0, accum_bytes);
                wgpu::CommandBuffer cmd = enc.Finish();
                ctx->queue().Submit(1, &cmd);
            }
            // Mark the snapshot as "not yet a new frame" so poll_async returns
            // samples=0 until the first post-reset snapshot lands. The viewport
            // keeps drawing its previous texture during the gap.
            {
                std::lock_guard<std::mutex> lk(snapshot_mu);
                snapshot_samples = 0;
            }
            since_last_snapshot = 0;
        }

        if (!submit_one_sample()) {
            // Dispatch failed; back off briefly and retry.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        ++since_last_snapshot;
        if (since_last_snapshot >= kSnapshotEveryNSamples) {
            take_snapshot();
            since_last_snapshot = 0;
        }
    }

    // Final snapshot on shutdown — gives the caller one last consistent state.
    take_snapshot();
}

bool PathTracer::AsyncState::submit_one_sample() {
    const uint32_t sample_offset = samples_completed.load(std::memory_order_relaxed);

    // Snapshot base_params under the lock — reset_async may be racing to
    // overwrite it with a new camera. Mutex is taken for the field copy only;
    // the actual WriteBuffer and Submit run lock-free.
    PathTracerParamsGpu p;
    {
        std::lock_guard<std::mutex> lk(base_params_mu);
        p = base_params;
    }
    // Fresh frame_seed per sample so PCG streams don't repeat across the
    // running accumulator. Mirrors the pattern from PyRenderer::render_tile_gpu.
    p.samples       = 1;
    p.sample_offset = sample_offset;
    p.frame_seed    = sample_offset * 0x9e3779b1u + 0xdeadbeefu;
    ctx->queue().WriteBuffer(params_buf, 0, &p, sizeof(p));

    wgpu::CommandEncoder encoder = ctx->device().CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        const uint32_t wg_x = (width  + 7) / 8;
        const uint32_t wg_y = (height + 7) / 8;
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
        pass.End();
    }
    wgpu::CommandBuffer cmd = encoder.Finish();
    ctx->queue().Submit(1, &cmd);
    samples_completed.fetch_add(1, std::memory_order_release);
    return true;
}

bool PathTracer::AsyncState::take_snapshot() {
    // Copy accumulator → stage, map, copy out, normalise.
    wgpu::CommandEncoder encoder = ctx->device().CreateCommandEncoder();
    encoder.CopyBufferToBuffer(accum_buf, 0, stage_buf, 0, accum_bytes);
    wgpu::CommandBuffer cmd = encoder.Finish();
    ctx->queue().Submit(1, &cmd);

    // Capture the error string into a heap-owned slot so a callback that
    // fires AFTER wait_for_with_timeout returns (timeout case) doesn't write
    // through a dangling reference. The shared_ptr keeps the slot alive as
    // long as the callback can possibly run.
    auto err_slot = std::make_shared<std::string>();
    wgpu::Future future = stage_buf.MapAsync(
        wgpu::MapMode::Read, 0, accum_bytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [err_slot](wgpu::MapAsyncStatus status, wgpu::StringView msg) {
            if (status != wgpu::MapAsyncStatus::Success) {
                err_slot->assign(msg.data, msg.length);
            }
        });
    const bool ok = wait_for_with_timeout(ctx->instance(), future);
    if (!ok || !err_slot->empty()) {
        if (!ok) {
            std::cerr << "[PathTracer] async snapshot MapAsync timed out; "
                      << "recreating stage buffer\n";
        } else {
            std::cerr << "[PathTracer] async snapshot MapAsync failed: "
                      << *err_slot << "; recreating stage buffer\n";
        }
        // Dawn keeps the buffer in a "pending map" state until the callback
        // resolves. On timeout we don't know when (or if) the callback fires,
        // and any subsequent CopyBufferToBuffer/MapAsync on the same handle
        // errors out (`already has an outstanding map pending`). Drop the
        // handle and let Dawn release the old buffer when the pending callback
        // eventually resolves through the shared_ptr.
        wgpu::BufferDescriptor d{};
        d.size  = std::max<uint64_t>(accum_bytes, kMinStorageBytes);
        d.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        stage_buf          = ctx->device().CreateBuffer(&d);
        stage_buf_capacity = d.size;
        return false;
    }

    const float* mapped = static_cast<const float*>(
        stage_buf.GetConstMappedRange(0, accum_bytes));
    if (!mapped) {
        std::cerr << "[PathTracer] async snapshot GetConstMappedRange returned null\n";
        return false;
    }

    const uint32_t s = samples_completed.load(std::memory_order_acquire);
    std::vector<float> pix(static_cast<size_t>(width) * height * 4);
    if (s == 0) {
        // No samples yet → return zeros.
        std::fill(pix.begin(), pix.end(), 0.0f);
    } else {
        const float inv_s = 1.0f / static_cast<float>(s);
        const size_t n_pixels = static_cast<size_t>(width) * height;
        for (size_t i = 0; i < n_pixels; ++i) {
            pix[i * 4 + 0] = mapped[i * 4 + 0] * inv_s;
            pix[i * 4 + 1] = mapped[i * 4 + 1] * inv_s;
            pix[i * 4 + 2] = mapped[i * 4 + 2] * inv_s;
            pix[i * 4 + 3] = 1.0f;
        }
    }
    stage_buf.Unmap();

    {
        std::lock_guard<std::mutex> lk(snapshot_mu);
        snapshot_pixels  = std::move(pix);
        snapshot_samples = s;
        snapshot_w       = width;
        snapshot_h       = height;
    }
    snapshot_revision.fetch_add(1, std::memory_order_release);
    return true;
}

uint32_t PathTracer::async_snapshot_revision() const noexcept {
    if (!async_state_) return 0;
    return async_state_->snapshot_revision.load(std::memory_order_acquire);
}

PathTracer::~PathTracer() {
    stop_async();
}

// Move ctor/assign live here (not in the header) because they require the full
// definition of AsyncState for the unique_ptr deleter.
PathTracer::PathTracer(PathTracer&&) noexcept = default;
PathTracer& PathTracer::operator=(PathTracer&&) noexcept = default;

void PathTracer::start_async(DawnContext& ctx,
                             const PackedPathScene& scene,
                             const PathTracerParamsGpu& base_params) {
    stop_async();

    // Sync render() shares params_buf_ / tri_buf_ / etc. — make sure those are
    // sized + uploaded for the new scene. Reuse the existing render() upload
    // path but skip the dispatch by allocating-and-writing only.
    if (!params_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = sizeof(PathTracerParamsGpu);
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        params_buf_ = ctx.device().CreateBuffer(&d);
    }
    const uint64_t tri_bytes = static_cast<uint64_t>(scene.triangles.size())
                               * sizeof(float);
    const uint64_t tri_alloc = std::max(tri_bytes, kMinStorageBytes);
    if (tri_alloc > tri_buf_capacity_ || !tri_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = tri_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        tri_buf_          = ctx.device().CreateBuffer(&d);
        tri_buf_capacity_ = tri_alloc;
    }
    if (tri_bytes > 0) {
        ctx.queue().WriteBuffer(tri_buf_, 0, scene.triangles.data(), tri_bytes);
    }
    const uint64_t lt_bytes = static_cast<uint64_t>(scene.lights.size())
                              * sizeof(GpuLight);
    const uint64_t lt_alloc = std::max(lt_bytes, kMinStorageBytes);
    if (lt_alloc > lights_buf_capacity_ || !lights_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = lt_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        lights_buf_          = ctx.device().CreateBuffer(&d);
        lights_buf_capacity_ = lt_alloc;
    }
    if (lt_bytes > 0) {
        ctx.queue().WriteBuffer(lights_buf_, 0, scene.lights.data(), lt_bytes);
    }
    const uint64_t cdf_bytes = static_cast<uint64_t>(scene.light_cdf.size())
                               * sizeof(float);
    const uint64_t cdf_alloc = std::max(cdf_bytes, kMinStorageBytes);
    if (cdf_alloc > light_cdf_buf_capacity_ || !light_cdf_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = cdf_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        light_cdf_buf_          = ctx.device().CreateBuffer(&d);
        light_cdf_buf_capacity_ = cdf_alloc;
    }
    if (cdf_bytes > 0) {
        ctx.queue().WriteBuffer(light_cdf_buf_, 0,
                                scene.light_cdf.data(), cdf_bytes);
    }
    const uint64_t bvh_bytes = static_cast<uint64_t>(scene.bvh_nodes.size())
                               * sizeof(GpuBvhNode);
    const uint64_t bvh_alloc = std::max(bvh_bytes, kMinStorageBytes);
    if (bvh_alloc > bvh_buf_capacity_ || !bvh_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = bvh_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        bvh_buf_          = ctx.device().CreateBuffer(&d);
        bvh_buf_capacity_ = bvh_alloc;
    }
    if (bvh_bytes > 0) {
        ctx.queue().WriteBuffer(bvh_buf_, 0, scene.bvh_nodes.data(), bvh_bytes);
    }
    last_scene_cache_id_ = scene.cache_id;

    auto st = std::make_unique<AsyncState>();
    st->ctx              = &ctx;
    st->pipeline         = pipeline_;
    st->layout           = layout_;
    st->params_buf       = params_buf_;
    st->tri_buf          = tri_buf_;
    st->tri_buf_size     = tri_buf_capacity_;
    st->lights_buf       = lights_buf_;
    st->lights_buf_size  = lights_buf_capacity_;
    st->light_cdf_buf    = light_cdf_buf_;
    st->light_cdf_buf_size = light_cdf_buf_capacity_;
    st->bvh_buf          = bvh_buf_;
    st->bvh_buf_size     = bvh_buf_capacity_;
    st->base_params      = base_params;
    st->width            = base_params.tile_w;
    st->height           = base_params.tile_h;
    st->accum_bytes      = uint64_t{st->width} * st->height * 4 * sizeof(float);

    // Allocate session-owned buffers.
    {
        wgpu::BufferDescriptor d{};
        d.size  = std::max<uint64_t>(st->accum_bytes, kMinStorageBytes);
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc
                | wgpu::BufferUsage::CopyDst;
        st->accum_buf          = ctx.device().CreateBuffer(&d);
        st->accum_buf_capacity = d.size;
    }
    {
        wgpu::BufferDescriptor d{};
        d.size  = std::max<uint64_t>(st->accum_bytes, kMinStorageBytes);
        d.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        st->stage_buf          = ctx.device().CreateBuffer(&d);
        st->stage_buf_capacity = d.size;
    }

    // Zero the accumulator before the worker starts.
    {
        wgpu::CommandEncoder enc = ctx.device().CreateCommandEncoder();
        enc.ClearBuffer(st->accum_buf, 0, st->accum_bytes);
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue().Submit(1, &cmd);
    }

    AsyncState* raw = st.get();
    async_state_ = std::move(st);
    async_state_->worker = std::thread([raw]() { raw->worker_loop(); });
}

void PathTracer::stop_async() {
    if (!async_state_) return;
    async_state_->stop_requested.store(true, std::memory_order_relaxed);
    if (async_state_->worker.joinable()) {
        async_state_->worker.join();
    }
    async_state_.reset();
}

void PathTracer::reset_async(const PathTracerParamsGpu& new_base_params) {
    if (!async_state_) return;
    // Dimensions / scene-content changes can't be handled in-place — they
    // require buffer reallocation or fresh static-buffer uploads, which the
    // worker can't do mid-stride. Caller is expected to route those through
    // stop_async + start_async; here we just guard against the obvious case.
    if (new_base_params.tile_w != async_state_->width
        || new_base_params.tile_h != async_state_->height) {
        std::cerr << "[PathTracer] reset_async called with mismatched dimensions ("
                  << new_base_params.tile_w << "x" << new_base_params.tile_h
                  << " vs session " << async_state_->width << "x"
                  << async_state_->height << "); falling back to stop+start\n";
        return;
    }
    {
        std::lock_guard<std::mutex> lk(async_state_->base_params_mu);
        async_state_->base_params = new_base_params;
    }
    // release: pairs with the worker's acquire on the same flag.
    async_state_->reset_pending.store(true, std::memory_order_release);
}

bool PathTracer::is_async_running() const noexcept {
    return async_state_ != nullptr;
}

uint32_t PathTracer::async_samples_completed() const noexcept {
    if (!async_state_) return 0;
    return async_state_->samples_completed.load(std::memory_order_acquire);
}

PathTracer::AsyncSnapshot PathTracer::poll_async() const {
    AsyncSnapshot s{};
    if (!async_state_) return s;
    std::lock_guard<std::mutex> lk(async_state_->snapshot_mu);
    s.samples = async_state_->snapshot_samples;
    s.width   = async_state_->snapshot_w;
    s.height  = async_state_->snapshot_h;
    s.pixels  = async_state_->snapshot_pixels;
    return s;
}

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
