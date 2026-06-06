/**
 * test_gpu_dawn_context.cpp - GTest for DawnContext + WGSL loader smoke
 *
 * Verifies:
 *  1. DawnContext::create() succeeds and reports a Metal backend on macOS
 *  2. The trivial double_test.wgsl pipeline produces bit-exact correct output
 *  3. Loading a missing WGSL file raises ShaderCompilationError
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/dawn_context.hpp"
#include "gpu/shader_module.hpp"

#include <gtest/gtest.h>
#include <string>

using lucid::gpu::DawnContext;
using lucid::gpu::run_double_test;
using lucid::gpu::load_wgsl;
using lucid::gpu::ShaderCompilationError;

TEST(DawnContextTest, Create) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value())
        << "DawnContext::create() returned nullopt - is the GPU adapter available?";

    const std::string info = ctx->adapter_info();
    EXPECT_FALSE(info.empty()) << "adapter_info() returned empty string";

#ifdef __APPLE__
    EXPECT_NE(info.find("Metal"), std::string::npos)
        << "Expected Metal backend on macOS, got: " << info;
#endif
}

TEST(DawnContextTest, DoubleCompute) {
    constexpr size_t N = 64;
    const auto out = run_double_test(N);

    ASSERT_EQ(out.size(), N);
    // Input is [0, 1, ..., N-1]; output should be [0, 2, 4, ..., 2*(N-1)].
    // Multiply-by-2 is bit-exact in f32, so EXPECT_EQ is safe.
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(out[i], static_cast<float>(i) * 2.0f) << "at index " << i;
    }
}

TEST(ShaderLoaderTest, MissingFile) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());

    EXPECT_THROW(
        load_wgsl(ctx->instance(), ctx->device(), "this_file_does_not_exist.wgsl"),
        ShaderCompilationError);
}

#else  // LUCID_HAS_DAWN

#include <gtest/gtest.h>

TEST(DawnContextTest, SkippedDawnDisabled) {
    GTEST_SKIP() << "Built without LUCID_HAS_DAWN";
}

#endif  // LUCID_HAS_DAWN
