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

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
