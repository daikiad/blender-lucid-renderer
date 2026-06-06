/**
 * dawn_context.hpp - RAII wrapper around Dawn (WebGPU) Instance/Adapter/Device/Queue
 * ===================================================================================
 *
 * DawnContext::create() performs the full async initialization synchronously
 * (Future-based wait under the hood). On macOS, the Metal backend is requested
 * explicitly so adapter_info() is deterministic across runs and so failures
 * surface early if Dawn was built without Metal support.
 *
 * The context is move-only. Phase 1a uses it only to power run_double_test();
 * later phases will pass the underlying Device/Queue into pipeline-building
 * helpers.
 */

#pragma once

#ifdef LUCID_HAS_DAWN

#include <webgpu/webgpu_cpp.h>
#include <optional>
#include <string>
#include <vector>

namespace lucid::gpu {

class DawnContext {
public:
    // Initializes Instance -> Adapter -> Device -> Queue. Returns nullopt
    // on any failure (logged to stderr). Safe to call once per process or
    // per render session.
    static std::optional<DawnContext> create();

    [[nodiscard]] std::string adapter_info() const;

    [[nodiscard]] const wgpu::Instance& instance() const { return instance_; }
    [[nodiscard]] const wgpu::Adapter&  adapter()  const { return adapter_; }
    [[nodiscard]] const wgpu::Device&   device()   const { return device_; }
    [[nodiscard]] const wgpu::Queue&    queue()    const { return queue_; }

    DawnContext(DawnContext&&) noexcept = default;
    DawnContext& operator=(DawnContext&&) noexcept = default;
    DawnContext(const DawnContext&) = delete;
    DawnContext& operator=(const DawnContext&) = delete;

private:
    DawnContext(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue);

    wgpu::Instance instance_;
    wgpu::Adapter  adapter_;
    wgpu::Device   device_;
    wgpu::Queue    queue_;
};

// Phase 1a end-to-end smoke test:
// runs a 64-thread compute shader that doubles each element of an N-element
// input buffer and returns the result. Used both by GTest and by the pybind
// `gpu_run_double_test()` entry point. Throws std::runtime_error on failure.
std::vector<float> run_double_test(size_t n);

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
