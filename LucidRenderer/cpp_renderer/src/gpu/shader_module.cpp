/**
 * shader_module.cpp - WGSL loader implementation
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/shader_module.hpp"
#include "gpu/sync.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace lucid::gpu {

namespace fs = std::filesystem;

namespace {

fs::path resolve(const fs::path& path) {
    if (path.is_absolute()) return path;
#ifdef LUCID_SHADER_DIR
    return fs::path(LUCID_SHADER_DIR) / path;
#else
    return path;
#endif
}

std::string read_file(const fs::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw ShaderCompilationError("Cannot open WGSL file: " + path.string());
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

wgpu::ShaderModule load_wgsl(const wgpu::Instance& instance,
                             const wgpu::Device& device,
                             const fs::path& path) {
    const fs::path abs = resolve(path);
    const std::string src = read_file(abs);

    // Create ShaderModule via the WGSL source chained descriptor.
    wgpu::ShaderSourceWGSL wgsl_desc{};
    wgsl_desc.code = wgpu::StringView{src.data(), src.length()};

    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl_desc;
    const std::string label = abs.filename().string();
    desc.label = wgpu::StringView{label.data(), label.length()};

    wgpu::ShaderModule mod = device.CreateShaderModule(&desc);

    // Surface any compilation errors via the async GetCompilationInfo callback.
    std::string error_msg;
    wgpu::Future future = mod.GetCompilationInfo(
        wgpu::CallbackMode::WaitAnyOnly,
        [&error_msg](wgpu::CompilationInfoRequestStatus /*status*/,
                     wgpu::CompilationInfo const* info) {
            if (!info) return;
            for (size_t i = 0; i < info->messageCount; ++i) {
                const auto& m = info->messages[i];
                if (m.type == wgpu::CompilationMessageType::Error) {
                    error_msg.append(m.message.data, m.message.length);
                    error_msg.push_back('\n');
                }
            }
        });
    wait_for(instance, future);

    if (!error_msg.empty()) {
        throw ShaderCompilationError("WGSL compilation failed for " +
                                     abs.string() + ":\n" + error_msg);
    }

    return mod;
}

}  // namespace lucid::gpu

#endif  // LUCID_HAS_DAWN
