/**
 * debug_renderer.cpp - GPU normal-debug renderer impl
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/debug_renderer.hpp"
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

namespace lucid::gpu {

namespace {

// Minimum allocation size for storage / staging buffers. Avoids creating
// zero-size buffers (Dawn validation errors) and reduces re-alloc churn for
// small renders.
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
// PackedScene / Camera packing
// ----------------------------------------------------------------------------

PackedScene pack_scene_for_debug(const Scene& scene) {
    PackedScene out;
    uint32_t total = 0;
    for (const auto& m : scene.meshes) {
        total += static_cast<uint32_t>(m.triangles.size());
    }
    out.triangle_count = total;
    out.triangles.reserve(static_cast<size_t>(total) * 24);

    for (const auto& mesh : scene.meshes) {
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
        }
    }
    return out;
}

CameraParamsGpu make_camera_params(const render::Position& pos,
                                   const render::Direction& fwd,
                                   const render::Direction& right,
                                   const render::Direction& up,
                                   float fov_rad,
                                   float aspect,
                                   uint32_t tile_x, uint32_t tile_y,
                                   uint32_t tile_w, uint32_t tile_h,
                                   uint32_t full_w, uint32_t full_h) {
    CameraParamsGpu cam{};
    const auto p = position_to_vec3(pos);
    cam.pos[0] = p.x; cam.pos[1] = p.y; cam.pos[2] = p.z;
    const auto f = fwd.vec();
    cam.fwd[0] = f.x; cam.fwd[1] = f.y; cam.fwd[2] = f.z;
    const auto r = right.vec();
    cam.right[0] = r.x; cam.right[1] = r.y; cam.right[2] = r.z;
    const auto u = up.vec();
    cam.up[0] = u.x; cam.up[1] = u.y; cam.up[2] = u.z;

    cam.fov_scale = std::tan(fov_rad * 0.5f);
    cam.aspect    = aspect;
    cam.tile_x    = tile_x; cam.tile_y = tile_y;
    cam.tile_w    = tile_w; cam.tile_h = tile_h;
    cam.full_w    = full_w; cam.full_h = full_h;
    return cam;
}

// ----------------------------------------------------------------------------
// DebugRenderer
// ----------------------------------------------------------------------------

DebugRenderer::DebugRenderer(wgpu::ShaderModule s,
                             wgpu::ComputePipeline p,
                             wgpu::BindGroupLayout l)
    : shader_(std::move(s)), pipeline_(std::move(p)), layout_(std::move(l)) {}

std::optional<DebugRenderer> DebugRenderer::create(DawnContext& ctx) {
    wgpu::ShaderModule shader;
    try {
        shader = load_wgsl(ctx.instance(), ctx.device(), "debug_normal.wgsl");
    } catch (const ShaderCompilationError& e) {
        std::cerr << "[DebugRenderer] Shader compile failed: " << e.what() << std::endl;
        return std::nullopt;
    }

    // Bind group layout: uniform Camera, read-only Triangle storage, RW output
    wgpu::BindGroupLayoutEntry entries[3] = {};
    entries[0].binding             = 0;
    entries[0].visibility          = wgpu::ShaderStage::Compute;
    entries[0].buffer.type         = wgpu::BufferBindingType::Uniform;
    entries[0].buffer.minBindingSize = sizeof(CameraParamsGpu);

    entries[1].binding             = 1;
    entries[1].visibility          = wgpu::ShaderStage::Compute;
    entries[1].buffer.type         = wgpu::BufferBindingType::ReadOnlyStorage;

    entries[2].binding             = 2;
    entries[2].visibility          = wgpu::ShaderStage::Compute;
    entries[2].buffer.type         = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor bgl_desc{};
    bgl_desc.entryCount = 3;
    bgl_desc.entries    = entries;
    wgpu::BindGroupLayout layout = ctx.device().CreateBindGroupLayout(&bgl_desc);

    wgpu::PipelineLayoutDescriptor pl_desc{};
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts     = &layout;
    wgpu::PipelineLayout pipeline_layout = ctx.device().CreatePipelineLayout(&pl_desc);

    wgpu::ComputePipelineDescriptor cp_desc{};
    cp_desc.layout               = pipeline_layout;
    cp_desc.compute.module       = shader;
    cp_desc.compute.entryPoint   = wgpu::StringView{"main", 4};
    wgpu::ComputePipeline pipeline = ctx.device().CreateComputePipeline(&cp_desc);

    return DebugRenderer(std::move(shader), std::move(pipeline), std::move(layout));
}

std::vector<float> DebugRenderer::render_normal(DawnContext& ctx,
                                                const PackedScene& scene,
                                                const CameraParamsGpu& cam) {
    if (cam.tile_w == 0 || cam.tile_h == 0) {
        return {};
    }
    const uint64_t pixel_count = uint64_t{cam.tile_w} * uint64_t{cam.tile_h};
    const uint64_t out_bytes   = pixel_count * 4 * sizeof(float);

    // ---- Camera uniform buffer ----
    if (!cam_buf_) {
        wgpu::BufferDescriptor desc{};
        desc.size  = sizeof(CameraParamsGpu);
        desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        cam_buf_   = ctx.device().CreateBuffer(&desc);
    }
    ctx.queue().WriteBuffer(cam_buf_, 0, &cam, sizeof(cam));

    // ---- Triangle storage buffer ----
    const uint64_t tri_bytes = static_cast<uint64_t>(scene.triangles.size())
                               * sizeof(float);
    const uint64_t tri_alloc = std::max(tri_bytes, kMinStorageBytes);
    if (tri_alloc > tri_buf_capacity_ || !tri_buf_) {
        wgpu::BufferDescriptor desc{};
        desc.size  = tri_alloc;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        tri_buf_              = ctx.device().CreateBuffer(&desc);
        tri_buf_capacity_     = tri_alloc;
    }
    if (tri_bytes > 0) {
        ctx.queue().WriteBuffer(tri_buf_, 0, scene.triangles.data(), tri_bytes);
    }

    // ---- Output storage buffer ----
    if (out_bytes > out_buf_capacity_ || !out_buf_) {
        wgpu::BufferDescriptor desc{};
        desc.size  = std::max(out_bytes, kMinStorageBytes);
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        out_buf_              = ctx.device().CreateBuffer(&desc);
        out_buf_capacity_     = desc.size;
    }

    // ---- Staging buffer (MapRead) ----
    if (out_bytes > stage_buf_capacity_ || !stage_buf_) {
        wgpu::BufferDescriptor desc{};
        desc.size  = std::max(out_bytes, kMinStorageBytes);
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        stage_buf_            = ctx.device().CreateBuffer(&desc);
        stage_buf_capacity_   = desc.size;
    }

    // ---- Bind group (cheap, per call) ----
    wgpu::BindGroupEntry bg_entries[3] = {};
    bg_entries[0].binding = 0;
    bg_entries[0].buffer  = cam_buf_;
    bg_entries[0].offset  = 0;
    bg_entries[0].size    = sizeof(CameraParamsGpu);
    bg_entries[1].binding = 1;
    bg_entries[1].buffer  = tri_buf_;
    bg_entries[1].offset  = 0;
    bg_entries[1].size    = tri_buf_capacity_;
    bg_entries[2].binding = 2;
    bg_entries[2].buffer  = out_buf_;
    bg_entries[2].offset  = 0;
    bg_entries[2].size    = out_bytes;

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout     = layout_;
    bg_desc.entryCount = 3;
    bg_desc.entries    = bg_entries;
    wgpu::BindGroup bg = ctx.device().CreateBindGroup(&bg_desc);

    // ---- Encode + submit ----
    wgpu::CommandEncoder encoder = ctx.device().CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline_);
        pass.SetBindGroup(0, bg);
        const uint32_t wg_x = (cam.tile_w + 7) / 8;
        const uint32_t wg_y = (cam.tile_h + 7) / 8;
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(out_buf_, 0, stage_buf_, 0, out_bytes);
    wgpu::CommandBuffer cmd = encoder.Finish();
    ctx.queue().Submit(1, &cmd);

    // ---- Map + readback ----
    std::string map_err;
    wgpu::Future map_future = stage_buf_.MapAsync(
        wgpu::MapMode::Read, 0, out_bytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [&map_err](wgpu::MapAsyncStatus status, wgpu::StringView msg) {
            if (status != wgpu::MapAsyncStatus::Success) {
                map_err.assign(msg.data, msg.length);
            }
        });
    wait_for(ctx.instance(), map_future);
    if (!map_err.empty()) {
        throw std::runtime_error("DebugRenderer::render_normal: MapAsync failed: "
                                 + map_err);
    }

    const float* mapped = static_cast<const float*>(
        stage_buf_.GetConstMappedRange(0, out_bytes));
    if (!mapped) {
        throw std::runtime_error("DebugRenderer::render_normal: GetConstMappedRange returned null");
    }
    std::vector<float> result(mapped, mapped + pixel_count * 4);
    stage_buf_.Unmap();

    return result;
}

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
