/**
 * test_gpu_path_tracer.cpp - GTests for the GPU path tracer (Phase 2a)
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/dawn_context.hpp"
#include "gpu/path_tracer.hpp"
#include "core/scene.hpp"
#include "light/light.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <numbers>
#include <thread>

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

// Helper: a +Z quad at z = z_plane with given Principled material params,
// large enough to cover the center pixel of a default 32x32 render.
Mesh make_principled_quad(float z_plane,
                          float albedo_r, float albedo_g, float albedo_b,
                          float metallic, float roughness,
                          float transmission, float ior) {
    Mesh m;
    m.vertices = {
        render::make_position(-1.5f, -1.5f, z_plane),
        render::make_position(+1.5f, -1.5f, z_plane),
        render::make_position(+1.5f, +1.5f, z_plane),
        render::make_position(-1.5f, +1.5f, z_plane),
    };
    const auto n = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    Triangle t1, t2;
    t1.i0 = 0; t1.i1 = 1; t1.i2 = 2;
    t2.i0 = 0; t2.i1 = 2; t2.i2 = 3;
    t1.faceNormal = n; t1.n0 = n; t1.n1 = n; t1.n2 = n; t1.smooth = false; t1.hasUV = false;
    t2.faceNormal = n; t2.n0 = n; t2.n1 = n; t2.n2 = n; t2.smooth = false; t2.hasUV = false;
    m.triangles = {t1, t2};
    m.material.albedo       = render::make_attenuation_rgb(albedo_r, albedo_g, albedo_b);
    m.material.emission     = render::make_radiance_rgb(0.0f, 0.0f, 0.0f);
    m.material.metallic     = metallic;
    m.material.roughness    = roughness;
    m.material.transmission = transmission;
    m.material.ior          = ior;
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
    ASSERT_EQ(packed.triangles.size(), 40u);

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
    // The .w channel now carries the sample count for the additive shader
    // (single-dispatch sync render still adds `samples` once because the
    // buffer is wiped beforehand).
    EXPECT_NEAR(out[idx + 3], static_cast<float>(samples), 1e-5f);
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

// ============================================================================
// Material parity tests — Phase 2-Mat (Principled BSDF + correct POINT lights)
// ============================================================================

TEST(PathTracerTest, MaterialParity_TriangleStride) {
    // Stride bump 32 → 40 floats per tri must be reflected in the packed
    // buffer; 1 triangle should occupy exactly 40 floats.
    Scene scene;
    scene.meshes.push_back(make_emissive_z_triangle(1.0f, 0.0f, 0.0f));
    auto packed = pack_scene_for_path_tracer(scene);
    ASSERT_EQ(packed.triangle_count, 1u);
    EXPECT_EQ(packed.triangles.size(), 40u);
}

TEST(PathTracerTest, MaterialParity_PointLightStride) {
    // Each POINT light occupies 12 floats (3 vec4s: pos/radius, emission/area,
    // reserved).
    Scene scene;
    Light l;
    l.type = LightType::POINT;
    l.position = render::make_position(0.0f, 1.0f, 0.0f);
    l.emission = render::make_radiance_rgb(1.0f, 1.0f, 1.0f);
    l.energy   = 5.0f * mp_units::si::watt;
    l.radius   = 0.0f * mp_units::si::metre;
    l.area     = 1.0f * mp_units::square(mp_units::si::metre);
    scene.nativeLights.push_back(l);
    auto packed = pack_scene_for_path_tracer(scene);
    ASSERT_EQ(packed.point_light_count, 1u);
    EXPECT_EQ(packed.point_lights.size(), 12u);
}

TEST(PathTracerTest, MaterialParity_GlassNoNaN) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    // Glass quad in front of an emissive triangle.
    Scene scene;
    scene.meshes.push_back(make_principled_quad(
        /*z=*/-1.5f,
        /*albedo=*/0.9f, 0.9f, 0.9f,
        /*metallic=*/0.0f, /*roughness=*/0.05f,
        /*transmission=*/1.0f, /*ior=*/1.45f));
    scene.meshes.push_back(make_emissive_z_triangle(2.0f, 2.0f, 2.0f));
    auto packed = pack_scene_for_path_tracer(scene);

    const uint32_t W = 32, H = 32;
    auto params = default_camera_params(W, H,
                                        /*samples=*/16, /*offset=*/0, /*max_bounces=*/8);
    params.bvh_node_count = packed.bvh_node_count;
    auto out = pt->render(*ctx, packed, params);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "non-finite at " << i;
        EXPECT_GE(out[i], 0.0f) << "negative at " << i;
    }
}

TEST(PathTracerTest, MaterialParity_MetallicNoNaN) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    // Gold-like metallic quad lit by an emissive triangle behind the camera.
    Scene scene;
    scene.meshes.push_back(make_principled_quad(
        /*z=*/-1.5f,
        /*albedo=*/0.95f, 0.78f, 0.5f,
        /*metallic=*/1.0f, /*roughness=*/0.1f,
        /*transmission=*/0.0f, /*ior=*/1.45f));
    scene.meshes.push_back(make_emissive_z_triangle(3.0f, 3.0f, 3.0f));
    auto packed = pack_scene_for_path_tracer(scene);

    const uint32_t W = 32, H = 32;
    auto params = default_camera_params(W, H,
                                        /*samples=*/16, /*offset=*/0, /*max_bounces=*/8);
    params.bvh_node_count = packed.bvh_node_count;
    auto out = pt->render(*ctx, packed, params);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "non-finite at " << i;
        EXPECT_GE(out[i], 0.0f) << "negative at " << i;
    }
}

TEST(PathTracerTest, MaterialParity_PointLightFluxConservation) {
    // Lambert quad at z=-1 lit by a delta POINT light directly above it (1m
    // away). Camera at z=5 looks along -Z; the center pixel ray hits the quad
    // at (0,0,-1).
    //
    // Per-sample contribution (Lambert, normal incidence):
    //   f       = albedo · (1 - F_dielectric(1, 0.04)) / π
    //           ≈ 0.7 · 0.96 / π  ≈ 0.214
    //   L       = light.emission / d²
    //           = (energy / 4π) / 1.0  ≈ 0.796
    //   cos_θ   = 1
    //   per_sample = f · L · cos_θ  ≈ 0.170
    //   sum     = per_sample · samples = 0.170 · 64 ≈ 10.9
    //
    // The shader still adds a tiny GGX specular contribution on top (roughness=1
    // gives a wide, near-uniform lobe at F0=0.04) so the actual value lands a
    // few % above this floor. Tolerance is wide enough to absorb MC noise at
    // samples=64.
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    Scene scene;
    scene.meshes.push_back(make_principled_quad(
        /*z=*/-1.0f,
        /*albedo=*/0.7f, 0.7f, 0.7f,
        /*metallic=*/0.0f, /*roughness=*/1.0f,
        /*transmission=*/0.0f, /*ior=*/1.45f));
    Light l;
    l.type = LightType::POINT;
    // 1m directly above the quad center along +Z (between camera and quad).
    l.position = render::make_position(0.0f, 0.0f, 0.0f);
    // CPU-side premultiplication: emission = color * energy / (4π).
    const float energy = 10.0f;
    const float intensity_per_ch = 1.0f * energy / (4.0f * kPi);   // ≈ 0.796
    l.emission = render::make_radiance_rgb(intensity_per_ch,
                                            intensity_per_ch,
                                            intensity_per_ch);
    l.energy   = energy * mp_units::si::watt;
    l.radius   = 0.0f * mp_units::si::metre;
    l.area     = 1.0f * mp_units::square(mp_units::si::metre);
    scene.nativeLights.push_back(l);
    auto packed = pack_scene_for_path_tracer(scene);

    const uint32_t W = 32, H = 32;
    const uint32_t samples = 64;
    auto params = default_camera_params(W, H, samples, /*offset=*/0, /*max_bounces=*/3);
    params.bvh_node_count    = packed.bvh_node_count;
    params.point_light_count = packed.point_light_count;
    auto out = pt->render(*ctx, packed, params);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    const uint32_t cx = W / 2;
    const uint32_t cy = H / 2;
    const size_t   idx = (cy * W + cx) * 4;
    const float    r = out[idx + 0];
    EXPECT_TRUE(std::isfinite(r));
    EXPECT_GT(r, 0.0f);
    // Closed-form expectation with full BSDF eval (Lambert · (1-F_dielectric)).
    const float albedo   = 0.7f;
    const float F0       = 0.04f;
    const float f_lambert= albedo * (1.0f - F0) / kPi;            // ≈ 0.214
    const float d2       = 1.0f;
    const float expected_sum = f_lambert * intensity_per_ch * static_cast<float>(samples) / d2;
    // ±50% envelope absorbs MC noise + small GGX specular contribution.
    EXPECT_GT(r, expected_sum * 0.5f) << "GPU POINT light contribution looks too dim";
    EXPECT_LT(r, expected_sum * 1.5f) << "GPU POINT light contribution looks too bright";
}

// ============================================================================
// Async accumulator smoke test
// ============================================================================

TEST(PathTracerTest, AsyncAccumulator_SmokeTest) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto pt = PathTracer::create(*ctx);
    ASSERT_TRUE(pt.has_value());

    Scene scene;
    scene.meshes.push_back(make_emissive_z_triangle(1.0f, 1.0f, 1.0f));
    auto packed = pack_scene_for_path_tracer(scene);

    const uint32_t W = 32, H = 32;
    auto params = default_camera_params(W, H, /*samples=*/1, /*offset=*/0, /*max_bounces=*/4);
    params.bvh_node_count = packed.bvh_node_count;

    EXPECT_FALSE(pt->is_async_running());
    pt->start_async(*ctx, packed, params);
    EXPECT_TRUE(pt->is_async_running());

    // Give the worker a chance to accumulate some samples.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const uint32_t samples_after_200ms = pt->async_samples_completed();
    EXPECT_GT(samples_after_200ms, 0u) << "worker should have produced at least some samples";

    auto snap = pt->poll_async();
    // Snapshot might be 0 if no full 16-sample chunk hit yet — give it a bit more time.
    if (snap.samples == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        snap = pt->poll_async();
    }
    EXPECT_GT(snap.samples, 0u);
    ASSERT_EQ(snap.pixels.size(), static_cast<size_t>(W) * H * 4);
    EXPECT_EQ(snap.width, W);
    EXPECT_EQ(snap.height, H);

    // Center pixel ray should hit the emissive triangle; averaged radiance > 0.
    const size_t idx = ((H / 2) * W + W / 2) * 4;
    EXPECT_GT(snap.pixels[idx + 0], 0.5f) << "averaged center R should reflect emission";
    EXPECT_NEAR(snap.pixels[idx + 3], 1.0f, 1e-5f) << "alpha should be 1.0 (normalised)";
    for (size_t i = 0; i < snap.pixels.size(); ++i) {
        EXPECT_TRUE(std::isfinite(snap.pixels[i])) << "non-finite at " << i;
    }

    pt->stop_async();
    EXPECT_FALSE(pt->is_async_running());
}

#else  // LUCID_HAS_DAWN

#include <gtest/gtest.h>
TEST(PathTracerTest, SkippedDawnDisabled) {
    GTEST_SKIP() << "Built without LUCID_HAS_DAWN";
}

#endif  // LUCID_HAS_DAWN
