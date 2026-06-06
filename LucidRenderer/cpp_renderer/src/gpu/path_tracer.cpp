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
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

// Forward decls — implementations live in src/node_evaluator.cpp. We mirror the
// CPU `getEmission()` / albedo handling so Blender materials using an Emission
// node (typical Cornell box ceiling) actually emit on the GPU side too.
extern render::AttenuationRGB getAlbedoFromNodeTree(const NodeTree& tree,
                                                    const render::Vec2f& uv);
extern render::RGB3f          getEmissionFromNodeTree(const NodeTree& tree,
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

}  // namespace

// ----------------------------------------------------------------------------
// PackedPathScene / params packing
// ----------------------------------------------------------------------------

PackedPathScene pack_scene_for_path_tracer(const Scene& scene) {
    PackedPathScene out;
    uint32_t total = 0;
    for (const auto& m : scene.meshes) {
        total += static_cast<uint32_t>(m.triangles.size());
    }
    out.triangle_count = total;
    out.triangles.reserve(static_cast<size_t>(total) * 32);

    // ---- Point lights (NEE) ----
    // Pack only LightType::POINT into a flat buffer of 8 floats per light:
    //   pos.xyz, _pad, color.xyz, intensity
    // intensity = Blender Light.energy (W). color comes from light.emission per
    // channel (un-normalized; shader divides by d² and multiplies by intensity).
    for (const auto& light : scene.nativeLights) {
        if (light.type != LightType::POINT) continue;
        const auto p = position_to_vec3(light.position);
        out.point_lights.push_back(p.x);
        out.point_lights.push_back(p.y);
        out.point_lights.push_back(p.z);
        out.point_lights.push_back(0.0f);
        const float er = light.emission.r.numerical_value_in(render::radiance_unit);
        const float eg = light.emission.g.numerical_value_in(render::radiance_unit);
        const float eb = light.emission.b.numerical_value_in(render::radiance_unit);
        const float energy = light.energy.numerical_value_in(mp_units::si::watt);
        out.point_lights.push_back(er);
        out.point_lights.push_back(eg);
        out.point_lights.push_back(eb);
        out.point_lights.push_back(energy);
        ++out.point_light_count;
    }

    for (const auto& mesh : scene.meshes) {
        // Evaluate albedo + emission once per mesh. If the material uses a node
        // tree (Blender's typical Cornell-box "Emission" node setup), evaluate
        // it at uv=(0,0) to get a flat constant. CPU `getMaterialParams()` /
        // `getEmission()` do the same. Phase 2a doesn't support per-fragment
        // UV-driven textures, just per-mesh constants from the node graph.
        render::AttenuationRGB albedo;
        render::RGB3f          emission_rgb;
        const bool use_nodes = mesh.material.useNodes && mesh.material.nodeTree.valid;
        if (use_nodes) {
            albedo       = getAlbedoFromNodeTree(mesh.material.nodeTree,
                                                  render::Vec2f{0.0f, 0.0f});
            emission_rgb = getEmissionFromNodeTree(mesh.material.nodeTree,
                                                    render::Vec2f{0.0f, 0.0f});
        } else {
            albedo = mesh.material.albedo;
            emission_rgb.r = mesh.material.emission.r.numerical_value_in(render::radiance_unit);
            emission_rgb.g = mesh.material.emission.g.numerical_value_in(render::radiance_unit);
            emission_rgb.b = mesh.material.emission.b.numerical_value_in(render::radiance_unit);
        }
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

            pack_vec3(out.triangles, v0, 0.0f);
            pack_vec3(out.triangles, v1, 0.0f);
            pack_vec3(out.triangles, v2, 0.0f);

            const float smooth_flag = tri.smooth ? 1.0f : 0.0f;
            pack_vec3(out.triangles, tri.n0.vec(), 0.0f);
            pack_vec3(out.triangles, tri.n1.vec(), 0.0f);
            pack_vec3(out.triangles, tri.n2.vec(), smooth_flag);

            // Albedo (RGB + pad)
            out.triangles.push_back(albedo_r);
            out.triangles.push_back(albedo_g);
            out.triangles.push_back(albedo_b);
            out.triangles.push_back(0.0f);
            // Emission (RGB + pad)
            out.triangles.push_back(emission_r);
            out.triangles.push_back(emission_g);
            out.triangles.push_back(emission_b);
            out.triangles.push_back(0.0f);
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
    uint32_t point_light_count) {
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

    wgpu::BindGroupLayoutEntry entries[4] = {};
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

    wgpu::BindGroupLayoutDescriptor bgl{};
    bgl.entryCount = 4;
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

    // ---- Triangle storage (grow on demand) ----
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

    // ---- Point-light storage (grow on demand) ----
    const uint64_t pl_bytes = static_cast<uint64_t>(scene.point_lights.size())
                              * sizeof(float);
    const uint64_t pl_alloc = std::max(pl_bytes, kMinStorageBytes);
    if (pl_alloc > point_lights_buf_capacity_ || !point_lights_buf_) {
        wgpu::BufferDescriptor d{};
        d.size  = pl_alloc;
        d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        point_lights_buf_          = ctx.device().CreateBuffer(&d);
        point_lights_buf_capacity_ = pl_alloc;
    }
    if (pl_bytes > 0) {
        ctx.queue().WriteBuffer(point_lights_buf_, 0,
                                scene.point_lights.data(), pl_bytes);
    }

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
    wgpu::BindGroupEntry bg_entries[4] = {};
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

    wgpu::BindGroupDescriptor bg{};
    bg.layout     = layout_;
    bg.entryCount = 4;
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
