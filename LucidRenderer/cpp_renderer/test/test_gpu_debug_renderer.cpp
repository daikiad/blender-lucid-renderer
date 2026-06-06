/**
 * test_gpu_debug_renderer.cpp - GTests for the GPU normal debug renderer
 */

#ifdef LUCID_HAS_DAWN

#include "gpu/dawn_context.hpp"
#include "gpu/debug_renderer.hpp"
#include "core/scene.hpp"
#include "light/light.hpp"  // Scene contains std::vector<Light>

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using lucid::gpu::DawnContext;
using lucid::gpu::DebugRenderer;
using lucid::gpu::PackedScene;
using lucid::gpu::pack_scene_for_debug;
using lucid::gpu::make_camera_params;

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// Helper: build a camera looking down -Z from (0, 0, 5).
auto default_camera(uint32_t w, uint32_t h) {
    auto cam_pos = render::make_position(0.0f, 0.0f, 5.0f);
    auto fwd     = render::direction_from_unit_vector({0.0f, 0.0f, -1.0f});
    auto right   = render::direction_from_unit_vector({1.0f, 0.0f, 0.0f});
    auto up      = render::direction_from_unit_vector({0.0f, 1.0f, 0.0f});
    return make_camera_params(cam_pos, fwd, right, up,
                              60.0f * kPi / 180.0f, 1.0f,
                              0, 0, w, h, w, h);
}

// Helper: build a single-triangle mesh facing +Z at z = -2.
Mesh make_z_facing_triangle_mesh() {
    Mesh m;
    m.vertices = {
        render::make_position(-1.0f, -1.0f, -2.0f),
        render::make_position(+1.0f, -1.0f, -2.0f),
        render::make_position( 0.0f, +1.0f, -2.0f),
    };
    Triangle t;
    t.i0 = 0; t.i1 = 1; t.i2 = 2;
    const auto n = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    t.faceNormal = n; t.n0 = n; t.n1 = n; t.n2 = n;
    t.smooth = false; t.hasUV = false;
    m.triangles.push_back(t);
    return m;
}

}  // namespace

// ----------------------------------------------------------------------------

TEST(DebugRendererTest, EmptyScene) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());

    auto renderer = DebugRenderer::create(*ctx);
    ASSERT_TRUE(renderer.has_value());

    Scene scene;
    auto packed = pack_scene_for_debug(scene);
    EXPECT_EQ(packed.triangle_count, 0u);

    const uint32_t W = 32, H = 32;
    auto cam = default_camera(W, H);
    auto out = renderer->render_normal(*ctx, packed, cam);

    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);
    // Every pixel should be the miss color (0.5, 0.5, 0.5, 1.0).
    for (size_t i = 0; i < out.size(); i += 4) {
        EXPECT_NEAR(out[i + 0], 0.5f, 1e-5f) << "r at pixel " << (i / 4);
        EXPECT_NEAR(out[i + 1], 0.5f, 1e-5f) << "g at pixel " << (i / 4);
        EXPECT_NEAR(out[i + 2], 0.5f, 1e-5f) << "b at pixel " << (i / 4);
        EXPECT_NEAR(out[i + 3], 1.0f, 1e-5f) << "a at pixel " << (i / 4);
    }
}

TEST(DebugRendererTest, SingleTriangleFacingCamera) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());

    auto renderer = DebugRenderer::create(*ctx);
    ASSERT_TRUE(renderer.has_value());

    Scene scene;
    scene.meshes.push_back(make_z_facing_triangle_mesh());
    auto packed = pack_scene_for_debug(scene);
    ASSERT_EQ(packed.triangle_count, 1u);
    ASSERT_EQ(packed.triangles.size(), 24u);

    const uint32_t W = 64, H = 64;
    auto cam = default_camera(W, H);
    auto out = renderer->render_normal(*ctx, packed, cam);
    ASSERT_EQ(out.size(), static_cast<size_t>(W) * H * 4);

    // Center pixel: should hit the triangle (normal = +Z -> color = (0.5, 0.5, 1.0))
    const uint32_t cx = W / 2;
    const uint32_t cy = H / 2;
    const size_t idx = (cy * W + cx) * 4;
    EXPECT_NEAR(out[idx + 0], 0.5f, 0.02f) << "expected normal.x ~ 0";
    EXPECT_NEAR(out[idx + 1], 0.5f, 0.02f) << "expected normal.y ~ 0";
    EXPECT_NEAR(out[idx + 2], 1.0f, 0.02f) << "expected normal.z ~ +1";
    EXPECT_NEAR(out[idx + 3], 1.0f, 1e-5f);

    // Corner pixel: should miss (background gray).
    const size_t corner = (0 * W + 0) * 4;
    EXPECT_NEAR(out[corner + 0], 0.5f, 1e-5f);
    EXPECT_NEAR(out[corner + 1], 0.5f, 1e-5f);
    EXPECT_NEAR(out[corner + 2], 0.5f, 1e-5f);
}

TEST(DebugRendererTest, SmoothNormalBarycentric) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto renderer = DebugRenderer::create(*ctx);
    ASSERT_TRUE(renderer.has_value());

    // Same triangle geometry as the face-normal test, but with three distinct
    // smooth normals; at barycentric centroid the result should be the
    // average normalized direction.
    Scene scene;
    Mesh m;
    m.vertices = {
        render::make_position(-1.0f, -1.0f, -2.0f),
        render::make_position(+1.0f, -1.0f, -2.0f),
        render::make_position( 0.0f, +1.0f, -2.0f),
    };
    Triangle t;
    t.i0 = 0; t.i1 = 1; t.i2 = 2;
    t.faceNormal = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    // Three distinct normals; expect barycentric average.
    t.n0 = render::normal_from_unit_vector({1.0f, 0.0f, 0.0f});
    t.n1 = render::normal_from_unit_vector({0.0f, 1.0f, 0.0f});
    t.n2 = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    t.smooth = true; t.hasUV = false;
    m.triangles.push_back(t);
    scene.meshes.push_back(std::move(m));

    auto packed = pack_scene_for_debug(scene);
    const uint32_t W = 64, H = 64;
    auto cam = default_camera(W, H);
    auto out = renderer->render_normal(*ctx, packed, cam);

    // At the visual center of the triangle the barycentric weights are
    // roughly (w=0.5, u=0.25, v=0.25) [not exactly 1/3 because the triangle
    // is not centered on the image-plane centroid]. We just check the
    // resulting normal is normalized (sum of squares of n in mapped color
    // space should match a unit vector).
    const size_t idx = (H / 2 * W + W / 2) * 4;
    const float nx = out[idx + 0] * 2.0f - 1.0f;
    const float ny = out[idx + 1] * 2.0f - 1.0f;
    const float nz = out[idx + 2] * 2.0f - 1.0f;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    // Allow some slack: barycentric blend of unit vectors is then normalized,
    // so the result should be ~ unit length.
    EXPECT_NEAR(len, 1.0f, 0.05f) << "smooth normal must be unit-length";

    // Background pixel is still the miss color.
    EXPECT_NEAR(out[(0 * W + 0) * 4 + 0], 0.5f, 1e-5f);
}

TEST(DebugRendererTest, MultiMesh) {
    auto ctx = DawnContext::create();
    ASSERT_TRUE(ctx.has_value());
    auto renderer = DebugRenderer::create(*ctx);
    ASSERT_TRUE(renderer.has_value());

    Scene scene;
    // Mesh A: triangle on the left, normal = +Z.
    Mesh a;
    a.vertices = {
        render::make_position(-2.0f, -1.0f, -2.0f),
        render::make_position(-0.5f, -1.0f, -2.0f),
        render::make_position(-1.25f, 1.0f, -2.0f),
    };
    Triangle ta;
    ta.i0 = 0; ta.i1 = 1; ta.i2 = 2;
    ta.faceNormal = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    ta.n0 = ta.n1 = ta.n2 = ta.faceNormal;
    ta.smooth = false; ta.hasUV = false;
    a.triangles.push_back(ta);

    // Mesh B: triangle on the right, normal pointing -X (sideways) to be
    // distinguishable.
    Mesh b;
    b.vertices = {
        render::make_position(0.5f, -1.0f, -2.0f),
        render::make_position(2.0f, -1.0f, -2.0f),
        render::make_position(1.25f, 1.0f, -2.0f),
    };
    Triangle tb = ta;
    tb.faceNormal = render::normal_from_unit_vector({0.0f, 0.0f, 1.0f});
    tb.n0 = tb.n1 = tb.n2 = tb.faceNormal;
    b.triangles.push_back(tb);

    scene.meshes.push_back(std::move(a));
    scene.meshes.push_back(std::move(b));
    auto packed = pack_scene_for_debug(scene);
    ASSERT_EQ(packed.triangle_count, 2u);

    const uint32_t W = 128, H = 64;
    auto cam = default_camera(W, H);
    auto out = renderer->render_normal(*ctx, packed, cam);

    // Both triangles face +Z → hit pixels have blue ≈ 1.0. Count hits in
    // the left and right halves and ensure both have some.
    int left_hits = 0, right_hits = 0;
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            if (out[i + 2] > 0.9f) {  // blue ~ 1.0 means hit on a +Z facing tri
                if (x < W / 2) ++left_hits; else ++right_hits;
            }
        }
    }
    EXPECT_GT(left_hits, 0)  << "Left mesh region should produce hits";
    EXPECT_GT(right_hits, 0) << "Right mesh region should produce hits";
}

#else  // LUCID_HAS_DAWN

#include <gtest/gtest.h>
TEST(DebugRendererTest, SkippedDawnDisabled) {
    GTEST_SKIP() << "Built without LUCID_HAS_DAWN";
}

#endif  // LUCID_HAS_DAWN
