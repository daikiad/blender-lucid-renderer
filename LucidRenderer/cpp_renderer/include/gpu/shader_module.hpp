/**
 * shader_module.hpp - WGSL shader loader from disk
 * =================================================
 *
 * Loads a `.wgsl` file and creates a wgpu::ShaderModule. Compilation errors
 * are surfaced as ShaderCompilationError exceptions (synchronous; uses
 * wait_for under the hood to drive Dawn's async GetCompilationInfo).
 *
 * Path resolution: absolute paths are used as-is; relative paths are resolved
 * against the LUCID_SHADER_DIR macro (set by CMake to the repo's shaders/).
 */

#pragma once

#ifdef LUCID_HAS_DAWN

#include <webgpu/webgpu_cpp.h>
#include <filesystem>
#include <stdexcept>

namespace lucid::gpu {

class ShaderCompilationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Load WGSL source from disk and compile it into a ShaderModule.
//   - `instance` is needed to drive the async GetCompilationInfo wait.
//   - `path` is resolved against LUCID_SHADER_DIR if relative.
// Throws ShaderCompilationError if the file is missing or WGSL compilation fails.
wgpu::ShaderModule load_wgsl(const wgpu::Instance& instance,
                             const wgpu::Device& device,
                             const std::filesystem::path& path);

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
