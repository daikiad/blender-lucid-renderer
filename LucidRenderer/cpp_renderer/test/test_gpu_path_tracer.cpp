/**
 * test_gpu_path_tracer.cpp - GTests for the GPU path tracer (Phase 2a)
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/dawn_context.hpp"
#include "gpu/path_tracer.hpp"
#include "core/scene.hpp"
#include "light/light.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using lucid::gpu::DawnContext;
using lucid::gpu::PathTracer;
using lucid::gpu::PackedPathScene;
using lucid::gpu::PathTracerParamsGpu;
using lucid::gpu::pack_scene_for_path_tracer;
using lucid::gpu::make_path_tracer_params;

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

auto default_camera_params(uint32_t w, uint32_t h,
                           uint32_t samples = 1,
                           uint32_t sample_offset = 0,
                           uint32_t max_bounces = 4,
                           const float env_color[3] = nullptr,
                           float env_strength = 0.0f) {
    auto pos   = render::make_position(0.0f, 0.0f, 5.0f);
    auto fwd   = render::direction_from_unit_vector({0.0f, 0.0f, -1.0f});
    auto right = render::direction_from_unit_vector({1.0f, 0.0f, 0.0f});
    auto up    = render::direction_from_unit_vector({0.0f, 1.0f, 0.0f});
    const float zero_env[3] = {0.0f, 0.0f, 0.0f};
    return make_path_tracer_params(
        pos, fwd, right, up,
        60.0f * kPi / 180.0f, 1.0f,
        0, 0, w, h, w, h,
        samples, sample_offset, max_bounces,
        /*frame_seed=*/12345u,
        env_color ? env_color : zero_env,
        env_strength,
        /*point_light_count=*/0u,
        /*bvh_node_count=*/0u);
}

// Helper: a single emissive +Z triangle in front of the camera at z = -2.
Mesh make_emissive_z_triangle(float emission_r, float emission_g, float emission_b) {
    Mesh m;
    m.vertices = {
        render::make_position(-1.5f, -1.5f, -2.0f),
        render::make_position(+1.5f, -1.5f, -2.0f),
        render::make_position( 0.0f, +2.0f, -2.0f),
    };
    Triangle t;
    t.i0 = 0; t.i1 = 1; t.i2 = 2;
    const auto n = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    t.faceNormal = n; t.n0 = n; t.n1 = n; t.n2 = n;
    t.smooth = false; t.hasUV = false;
    m.triangles.push_back(t);
    // Material with given emission, no albedo.
    m.material.albedo   = render::make_attenuation_rgb(0.0f, 0.0f, 0.0f);
    m.material.emission = render::make_radiance_rgb(emission_r, emission_g, emission_b);
    return m;
}

}  // namespace

TEST(PathTracerTest, EmptyScene_EnvOnly) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    Scene scene;
    auto packed = pack_scene_for_path_tracer(scene);
    EXPECT_EQ(packed.triangle_count, 0u);

    const uint32_t W = 32, H = 32;
    const float env[3] = {0.5f, 0.6f, 0.7f};
    auto params = default_camera_params(W, H,
                                        /*samples=*/1, /*offset=*/0, /*max_bounces=*/2,
                                        env, /*env_strength=*/1.0f);
    auto out = pt->render(*ctx, packed, params);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    // Every pixel should hit env and accumulate (0.5, 0.6, 0.7) * 1 sample.
    for (size_t i = 0; i < out.size(); i += 4) {
        EXPECT_NEAR(out[i + 0], 0.5f, 1e-4f) << "r at pixel " << (i / 4);
        EXPECT_NEAR(out[i + 1], 0.6f, 1e-4f) << "g at pixel " << (i / 4);
        EXPECT_NEAR(out[i + 2], 0.7f, 1e-4f) << "b at pixel " << (i / 4);
        EXPECT_NEAR(out[i + 3], 1.0f, 1e-5f) << "a at pixel " << (i / 4);
    }
}

TEST(PathTracerTest, SingleEmissiveTriangle) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    // One emissive +Z triangle (emission red = 2.0)
    Scene scene;
    scene.meshes.push_back(make_emissive_z_triangle(2.0f, 0.0f, 0.0f));
    auto packed = pack_scene_for_path_tracer(scene);
    ASSERT_EQ(packed.triangle_count, 1u);
    ASSERT_EQ(packed.triangles.size(), 32u);

    const uint32_t W = 32, H = 32;
    const uint32_t samples = 4;
    auto params = default_camera_params(W, H,
                                        samples, /*offset=*/0, /*max_bounces=*/2);
    params.bvh_node_count = packed.bvh_node_count;
    auto out = pt->render(*ctx, packed, params);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    // Center pixel: ray hits the triangle's face, emission accumulates over `samples`.
    const uint32_t cx = W / 2;
    const uint32_t cy = H / 2;
    const size_t idx = (cy * W + cx) * 4;
    const float r = out[idx + 0];
    EXPECT_GT(r, 1.0f * static_cast<float>(samples)) << "expected ~ emission * samples";
    EXPECT_LT(r, 3.0f * static_cast<float>(samples)) << "shouldn't blow up";
    EXPECT_NEAR(out[idx + 1], 0.0f, 1e-5f);
    EXPECT_NEAR(out[idx + 2], 0.0f, 1e-5f);
    EXPECT_NEAR(out[idx + 3], 1.0f, 1e-5f);
}

TEST(PathTracerTest, SingleEmissiveTriangle_NoNaN) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    Scene scene;
    scene.meshes.push_back(make_emissive_z_triangle(1.0f, 1.0f, 1.0f));
    auto packed = pack_scene_for_path_tracer(scene);

    const uint32_t W = 64, H = 64;
    auto params = default_camera_params(W, H,
                                        /*samples=*/16, /*offset=*/0, /*max_bounces=*/16);
    params.bvh_node_count = packed.bvh_node_count;
    auto out = pt->render(*ctx, packed, params);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    // All pixels finite and non-negative (no NaN/Inf leakage from RR or normalize).
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "non-finite at index " << i;
        EXPECT_GE(out[i], 0.0f) << "negative value at index " << i;
    }
}

#else  // LUCID_HAS_DAWN

#include <gtest/gtest.h>
TEST(PathTracerTest, SkippedDawnDisabled) {
    GTEST_SKIP() << "Built without LUCID_HAS_DAWN";
}

#endif  // LUCID_HAS_DAWN
