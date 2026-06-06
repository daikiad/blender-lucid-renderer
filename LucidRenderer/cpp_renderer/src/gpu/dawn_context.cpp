/**
 * dawn_context.cpp - Dawn (WebGPU) initialization and the Phase 1a smoke test
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/dawn_context.hpp"
#include "gpu/shader_module.hpp"
#include "gpu/sync.hpp"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace lucid::gpu {

namespace {

std::string sv_to_string(wgpu::StringView sv) {
    if (sv.data == nullptr || sv.length == 0) return {};
    return std::string(sv.data, sv.length);
}

const char* backend_name(wgpu::BackendType bt) {
    switch (bt) {
        case wgpu::BackendType::Metal:    return "Metal";
        case wgpu::BackendType::Vulkan:   return "Vulkan";
        case wgpu::BackendType::D3D12:    return "D3D12";
        case wgpu::BackendType::D3D11:    return "D3D11";
        case wgpu::BackendType::OpenGL:   return "OpenGL";
        case wgpu::BackendType::OpenGLES: return "OpenGLES";
        case wgpu::BackendType::WebGPU:   return "WebGPU";
        case wgpu::BackendType::Null:     return "Null";
        default:                          return "Unknown";
    }
}

void on_uncaptured_error(const wgpu::Device&,
                         wgpu::ErrorType type,
                         wgpu::StringView message) {
    std::cerr << "[Dawn] Uncaptured device error (type=" << static_cast<int>(type)
              << "): " << sv_to_string(message) << std::endl;
}

}  // namespace

DawnContext::DawnContext(wgpu::Instance i,
                         wgpu::Adapter a,
                         wgpu::Device d,
                         wgpu::Queue q)
    : instance_(std::move(i)),
      adapter_(std::move(a)),
      device_(std::move(d)),
      queue_(std::move(q)) {}

std::optional<DawnContext> DawnContext::create() {
    // ---- 1. Instance ----
    // TimedWaitAny must be requested so we can WaitAny on futures with a finite timeout.
    static const wgpu::InstanceFeatureName kInstanceFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny,
    };
    wgpu::InstanceDescriptor inst_desc{};
    inst_desc.requiredFeatureCount = 1;
    inst_desc.requiredFeatures = kInstanceFeatures;
    wgpu::Instance instance = wgpu::CreateInstance(&inst_desc);
    if (!instance) {
        std::cerr << "[Dawn] CreateInstance failed" << std::endl;
        return std::nullopt;
    }

    // ---- 2. Adapter ----
    wgpu::RequestAdapterOptions ad_opts{};
    ad_opts.powerPreference = wgpu::PowerPreference::HighPerformance;
#ifdef __APPLE__
    // Explicit Metal request: makes adapter_info() deterministic and surfaces
    // failures early if Dawn was built without Metal support.
    ad_opts.backendType = wgpu::BackendType::Metal;
#endif

    wgpu::Adapter adapter;
    std::string adapter_err;
    wgpu::Future ad_future = instance.RequestAdapter(
        &ad_opts,
        wgpu::CallbackMode::WaitAnyOnly,
        [&adapter, &adapter_err](wgpu::RequestAdapterStatus status,
                                 wgpu::Adapter a,
                                 wgpu::StringView msg) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                adapter = std::move(a);
            } else {
                adapter_err = sv_to_string(msg);
            }
        });
    wait_for(instance, ad_future);
    if (!adapter) {
        std::cerr << "[Dawn] RequestAdapter failed: " << adapter_err << std::endl;
        return std::nullopt;
    }

    // ---- 3. Device ----
    wgpu::DeviceDescriptor dev_desc{};
    dev_desc.SetUncapturedErrorCallback(on_uncaptured_error);

    wgpu::Device device;
    std::string device_err;
    wgpu::Future dev_future = adapter.RequestDevice(
        &dev_desc,
        wgpu::CallbackMode::WaitAnyOnly,
        [&device, &device_err](wgpu::RequestDeviceStatus status,
                               wgpu::Device d,
                               wgpu::StringView msg) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                device = std::move(d);
            } else {
                device_err = sv_to_string(msg);
            }
        });
    wait_for(instance, dev_future);
    if (!device) {
        std::cerr << "[Dawn] RequestDevice failed: " << device_err << std::endl;
        return std::nullopt;
    }

    // ---- 4. Queue ----
    wgpu::Queue queue = device.GetQueue();

    return DawnContext(std::move(instance),
                       std::move(adapter),
                       std::move(device),
                       std::move(queue));
}

std::string DawnContext::adapter_info() const {
    wgpu::AdapterInfo info{};
    adapter_.GetInfo(&info);
    return sv_to_string(info.vendor) + " / " +
           sv_to_string(info.device) + " / " +
           backend_name(info.backendType);
}

// ----------------------------------------------------------------------------
// run_double_test - Phase 1a end-to-end smoke
// ----------------------------------------------------------------------------

std::vector<float> run_double_test(size_t n) {
    auto ctx_opt = DawnContext::create();
    if (!ctx_opt) {
        throw std::runtime_error("run_double_test: failed to create Dawn context");
    }
    const DawnContext& ctx = *ctx_opt;

    // Input: [0, 1, ..., n-1]
    std::vector<float> input(n);
    std::iota(input.begin(), input.end(), 0.0f);

    const uint64_t buf_size = sizeof(float) * n;

    // Input storage buffer (host-written, GPU-read).
    wgpu::BufferDescriptor in_desc{};
    in_desc.size  = buf_size;
    in_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer in_buf = ctx.device().CreateBuffer(&in_desc);
    ctx.queue().WriteBuffer(in_buf, 0, input.data(), buf_size);

    // Output storage buffer (GPU-written, copied to staging for readback).
    wgpu::BufferDescriptor out_desc{};
    out_desc.size  = buf_size;
    out_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer out_buf = ctx.device().CreateBuffer(&out_desc);

    // Staging buffer for mapping back to host.
    wgpu::BufferDescriptor stage_desc{};
    stage_desc.size  = buf_size;
    stage_desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer stage_buf = ctx.device().CreateBuffer(&stage_desc);

    // Shader + pipeline.
    wgpu::ShaderModule mod = load_wgsl(ctx.instance(), ctx.device(), "double_test.wgsl");

    wgpu::ComputePipelineDescriptor pipe_desc{};
    pipe_desc.compute.module     = mod;
    pipe_desc.compute.entryPoint = wgpu::StringView{"main", 4};
    wgpu::ComputePipeline pipeline = ctx.device().CreateComputePipeline(&pipe_desc);

    // Bind group: binding 0 = input, binding 1 = output.
    wgpu::BindGroupEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].buffer  = in_buf;
    entries[0].offset  = 0;
    entries[0].size    = buf_size;
    entries[1].binding = 1;
    entries[1].buffer  = out_buf;
    entries[1].offset  = 0;
    entries[1].size    = buf_size;

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout     = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 2;
    bg_desc.entries    = entries;
    wgpu::BindGroup bg = ctx.device().CreateBindGroup(&bg_desc);

    // Encode + submit.
    wgpu::CommandEncoder encoder = ctx.device().CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bg);
        const uint32_t workgroups = static_cast<uint32_t>((n + 63) / 64);
        pass.DispatchWorkgroups(workgroups, 1, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(out_buf, 0, stage_buf, 0, buf_size);
    wgpu::CommandBuffer cmd = encoder.Finish();
    ctx.queue().Submit(1, &cmd);

    // Map staging buffer and read back.
    std::string map_err;
    wgpu::Future map_future = stage_buf.MapAsync(
        wgpu::MapMode::Read, 0, buf_size,
        wgpu::CallbackMode::WaitAnyOnly,
        [&map_err](wgpu::MapAsyncStatus status, wgpu::StringView msg) {
            if (status != wgpu::MapAsyncStatus::Success) {
                map_err = sv_to_string(msg);
            }
        });
    wait_for(ctx.instance(), map_future);
    if (!map_err.empty()) {
        throw std::runtime_error("run_double_test: MapAsync failed: " + map_err);
    }

    const float* mapped = static_cast<const float*>(
        stage_buf.GetConstMappedRange(0, buf_size));
    if (!mapped) {
        throw std::runtime_error("run_double_test: GetConstMappedRange returned null");
    }
    std::vector<float> output(mapped, mapped + n);
    stage_buf.Unmap();

    return output;
}

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
