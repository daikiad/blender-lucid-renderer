/**
 * sync.hpp - Synchronous wait wrapper for WebGPU Future-based async APIs
 * ======================================================================
 *
 * Dawn's RequestAdapter, RequestDevice, MapAsync, GetCompilationInfo, etc.
 * return wgpu::Future objects. wait_for() blocks until the future resolves
 * by calling Instance::WaitAny() with no timeout.
 *
 * Use this only in non-realtime paths (initialization, tests). The path
 * tracer hot loop should never call wait_for(); it dispatches and then
 * accumulates the result via GPU-resident pipelines.
 */

#pragma once

#ifdef LUCID_HAS_DAWN

#include <webgpu/webgpu_cpp.h>
#include <limits>

namespace lucid::gpu {

// Block until `future` resolves. UINT64_MAX = infinite wait.
inline void wait_for(const wgpu::Instance& instance, wgpu::Future future) {
    wgpu::FutureWaitInfo info{future};
    instance.WaitAny(1, &info, std::numeric_limits<uint64_t>::max());
}

// Block until `future` resolves, with a timeout. Returns true on success,
// false on timeout. Use this instead of wait_for() inside per-frame paths
// (viewport rendering) so a stalled GPU doesn't hang the calling thread.
inline bool wait_for_with_timeout(const wgpu::Instance& instance,
                                  wgpu::Future future,
                                  uint64_t timeout_ns = 5'000'000'000ull) {
    wgpu::FutureWaitInfo info{future};
    const auto status = instance.WaitAny(1, &info, timeout_ns);
    return status == wgpu::WaitStatus::Success && info.completed;
}

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
