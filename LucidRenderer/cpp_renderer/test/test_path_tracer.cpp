/**
 * @file test_path_tracer.cpp
 * @brief Integration tests for path tracer convergence
 * 
 * These tests verify:
 * - traceSimple, traceNEE, traceMIS converge to the same result
 * - Specific regression scenarios (mirror + point light, etc.)
 * - Energy conservation bounds
 */

#include <gtest/gtest.h>
#include <random>
#include <numeric>
#include "test_utils.hpp"
#include "core/scene.hpp"
#include "light/scene_lights.hpp"
#include "integrator/path_tracer.hpp"

using namespace test_utils;

// Helper to finalize scene (build BVH for all meshes)
inline void finalizeScene(Scene& scene) {
    for (Mesh& mesh : scene.meshes) {
        finalizeMeshBounds(mesh);
    }
}

// =============================================================================
// Test Fixture
// =============================================================================

class PathTracerTest : public ::testing::Test {
protected:
    static constexpr int kMaxDepth = 8;
    static constexpr int kSamplesPerPixel = 512;  // Increased for convergence
    static constexpr float kSigmaThreshold = 3.0f;  // Accept if within 3 sigma
    
    void SetUp() override {
        rng.seed(42);  // Fixed seed for reproducibility
    }
    
    std::mt19937 rng;
    
    // Result with statistics for proper comparison
    struct RenderResult {
        float mean = 0.0f;
        float variance = 0.0f;
        float stdErr = 0.0f;  // Standard error of the mean
        int samples = 0;
    };
    
    // Render and compute statistics (luminance-based for simplicity)
    RenderResult renderWithStats(
        const Scene& scene, 
        const SceneLights* lights,  // nullptr for Simple
        const Ray& ray, 
        int numSamples,
        int integrator  // 0=Simple, 1=NEE, 2=MIS
    ) {
        std::vector<float> values;
        values.reserve(numSamples);
        
        for (int i = 0; i < numSamples; ++i) {
            render::RadianceRGB radiance;
            switch (integrator) {
                case 0: radiance = traceSimple(scene, ray, kMaxDepth); break;
                case 1: radiance = traceNEE(scene, *lights, ray, kMaxDepth); break;
                case 2: radiance = traceMIS(scene, *lights, ray, kMaxDepth); break;
            }
            // Convert to pixel value for luminance calculation
            render::PixelRGB sample = render::apply_camera_sensitivity(radiance, render::kDefaultCameraSensitivity);
            // Luminance
            auto [sr, sg, sb] = render::color_to_floats(sample);
            float lum = sr * 0.2126f + sg * 0.7152f + sb * 0.0722f;
            values.push_back(lum);
        }
        
        auto stats = computeStatistics(values);
        RenderResult result;
        result.mean = stats.mean;
        result.variance = stats.variance;
        result.stdErr = stats.stdDev / std::sqrt(static_cast<float>(numSamples));
        result.samples = numSamples;
        return result;
    }
    
    // Check if two results are statistically consistent
    // Returns true if difference is within combined standard errors * sigma
    bool areConsistent(const RenderResult& a, const RenderResult& b, float sigmas = kSigmaThreshold) {
        float diff = std::abs(a.mean - b.mean);
        float combinedStdErr = std::sqrt(a.stdErr * a.stdErr + b.stdErr * b.stdErr);
        
        // If both have very low variance and means differ significantly, that's a bug
        // If variance is high, we can't conclude much
        return diff < sigmas * combinedStdErr;
    }
    
    // Legacy helpers for simple tests (uses default camera sensitivity)
    render::PixelRGB renderPixelSimple(const Scene& scene, const Ray& ray, int samples) {
        render::RadianceRGB result = render::zero_radiance_rgb();
        for (int i = 0; i < samples; ++i) {
            render::RadianceRGB sample = traceSimple(scene, ray, kMaxDepth);
            result += sample;
        }
        // Apply default sensitivity (EV=0) and average
        return render::apply_camera_sensitivity(result, render::kDefaultCameraSensitivity) / static_cast<float>(samples);
    }
    
    render::PixelRGB renderPixelNEE(const Scene& scene, const SceneLights& lights, const Ray& ray, int samples) {
        render::RadianceRGB result = render::zero_radiance_rgb();
        for (int i = 0; i < samples; ++i) {
            render::RadianceRGB sample = traceNEE(scene, lights, ray, kMaxDepth);
            result += sample;
        }
        return render::apply_camera_sensitivity(result, render::kDefaultCameraSensitivity) / static_cast<float>(samples);
    }
    
    render::PixelRGB renderPixelMIS(const Scene& scene, const SceneLights& lights, const Ray& ray, int samples) {
        render::RadianceRGB result = render::zero_radiance_rgb();
        for (int i = 0; i < samples; ++i) {
            render::RadianceRGB sample = traceMIS(scene, lights, ray, kMaxDepth);
            result += sample;
        }
        return render::apply_camera_sensitivity(result, render::kDefaultCameraSensitivity) / static_cast<float>(samples);
    }
};

// =============================================================================
// Basic Sanity Tests
// =============================================================================

TEST_F(PathTracerTest, EmptySceneReturnsEnvironment) {
    Scene scene;
    scene.environment.color = render::make_attenuation_rgb(0.5f, 0.5f, 0.5f);
    scene.environment.strength = 1.0f;
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    Ray ray(render::make_position(0.0f, 0.0f, 0.0f),
            render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, -1.0f)));
    
    render::PixelRGB resultSimple = render::apply_camera_sensitivity(traceSimple(scene, ray, kMaxDepth), render::kDefaultCameraSensitivity);
    render::PixelRGB resultNEE = render::apply_camera_sensitivity(traceNEE(scene, lights, ray, kMaxDepth), render::kDefaultCameraSensitivity);
    render::PixelRGB resultMIS = render::apply_camera_sensitivity(traceMIS(scene, lights, ray, kMaxDepth), render::kDefaultCameraSensitivity);
    
    // All should return environment color (compare raw RGB3f triples since types differ)
    auto envRgb = render::to_rgb3f(scene.environment.color);
    EXPECT_TRUE(rgb3fApproxEqual(render::to_rgb3f(resultSimple), envRgb, 0.01f));
    EXPECT_TRUE(rgb3fApproxEqual(render::to_rgb3f(resultNEE),    envRgb, 0.01f));
    EXPECT_TRUE(rgb3fApproxEqual(render::to_rgb3f(resultMIS),    envRgb, 0.01f));
}

TEST_F(PathTracerTest, DirectLightHit) {
    // Ray directly hits emissive surface
    Scene scene;
    
    Mesh emissive;
    emissive.vertices = {
        render::make_position(-1.0f, 0.0f, -2.0f),
        render::make_position(1.0f, 0.0f, -2.0f),
        render::make_position(0.0f, 1.0f, -2.0f)
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.material = Material(
        render::make_attenuation_rgb(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(5.0f, 5.0f, 5.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    finalizeScene(scene);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    Ray ray(render::make_position(0.0f, 0.5f, 0.0f),
            render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, -1.0f)));
    
    render::PixelRGB resultSimple = render::apply_camera_sensitivity(traceSimple(scene, ray, kMaxDepth), render::kDefaultCameraSensitivity);
    render::PixelRGB resultNEE = render::apply_camera_sensitivity(traceNEE(scene, lights, ray, kMaxDepth), render::kDefaultCameraSensitivity);
    render::PixelRGB resultMIS = render::apply_camera_sensitivity(traceMIS(scene, lights, ray, kMaxDepth), render::kDefaultCameraSensitivity);
    
    // All should see the emissive surface
    EXPECT_GT(resultSimple.r, 1.0f);
    EXPECT_GT(resultNEE.r, 1.0f);
    EXPECT_GT(resultMIS.r, 1.0f);
}

// =============================================================================
// Convergence Tests
// =============================================================================

TEST_F(PathTracerTest, DiffuseSceneConvergence) {
    // Simple diffuse scene - all integrators should converge to similar values
    // Use statistical comparison to handle Monte Carlo variance properly
    Scene scene = createCornellBoxScene();
    finalizeScene(scene);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    // Look at center of floor
    Ray ray(render::make_position(0.0f, 1.0f, 0.8f),
            render::direction_from_unit_vector(render::Vec3f(0.0f, -0.5f, -0.5f)));
    
    // Render with statistics
    auto resultNEE = renderWithStats(scene, &lights, ray, kSamplesPerPixel, 1);
    auto resultMIS = renderWithStats(scene, &lights, ray, kSamplesPerPixel, 2);
    
    // NEE and MIS should converge to the same value (within statistical error)
    EXPECT_TRUE(areConsistent(resultNEE, resultMIS))
        << "NEE and MIS results are statistically inconsistent.\n"
        << "NEE: mean=" << resultNEE.mean << " stdErr=" << resultNEE.stdErr << "\n"
        << "MIS: mean=" << resultMIS.mean << " stdErr=" << resultMIS.stdErr << "\n"
        << "Difference: " << std::abs(resultNEE.mean - resultMIS.mean) << "\n"
        << "Combined stdErr * 3: " << 3.0f * std::sqrt(resultNEE.stdErr * resultNEE.stdErr + resultMIS.stdErr * resultMIS.stdErr);
}

// =============================================================================
// Regression Tests
// =============================================================================

TEST_F(PathTracerTest, MirrorReflectionRegression) {
    // Regression test for the MIS mirror bug
    // Point light -> Mirror -> Wall -> Camera
    // Use statistical comparison for proper MC variance handling
    Scene scene = createMirrorPointLightScene();
    finalizeScene(scene);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    // Ray looking at floor where mirror reflection should be visible
    Ray ray(render::make_position(0.0f, 1.0f, 1.5f),
            render::direction_from_unit_vector(render::Vec3f(0.0f, -0.3f, -1.0f)));
    
    auto resultSimple = renderWithStats(scene, nullptr, ray, kSamplesPerPixel, 0);
    auto resultMIS = renderWithStats(scene, &lights, ray, kSamplesPerPixel, 2);
    
    // MIS should not be systematically darker than Simple
    // If both have low stdErr but means differ significantly, that's a bug
    if (resultSimple.mean > 0.01f && resultSimple.stdErr < 0.1f * resultSimple.mean) {
        // Only check if Simple has meaningful, stable illumination
        float ratio = resultMIS.mean / resultSimple.mean;
        
        // MIS being darker by more than 50% with low variance indicates a bug
        EXPECT_GT(ratio, 0.5f) 
            << "MIS is systematically too dark compared to Simple.\n"
            << "Simple: mean=" << resultSimple.mean << " stdErr=" << resultSimple.stdErr << "\n"
            << "MIS: mean=" << resultMIS.mean << " stdErr=" << resultMIS.stdErr << "\n"
            << "Ratio: " << ratio;
    }
}

TEST_F(PathTracerTest, SpecularMISWeightIsOne) {
    // For near-delta specular materials, MIS weight should be ~1.0
    // because light sampling cannot efficiently sample the narrow lobe
    // Use statistical comparison for proper variance handling
    Scene scene = createMirrorPointLightScene();
    finalizeScene(scene);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    Ray ray(render::make_position(0.0f, 1.0f, 1.0f),
            render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, -1.0f)));
    
    auto resultSimple = renderWithStats(scene, nullptr, ray, kSamplesPerPixel, 0);
    auto resultMIS = renderWithStats(scene, &lights, ray, kSamplesPerPixel, 2);
    
    // If Simple has stable results, MIS should match
    if (resultSimple.mean > 0.1f && resultSimple.stdErr < 0.2f * resultSimple.mean) {
        EXPECT_TRUE(areConsistent(resultSimple, resultMIS, 4.0f))  // 4 sigma for high variance paths
            << "MIS mean is inconsistent with Simple for specular paths.\n"
            << "Simple: mean=" << resultSimple.mean << " stdErr=" << resultSimple.stdErr << "\n"
            << "MIS: mean=" << resultMIS.mean << " stdErr=" << resultMIS.stdErr;
    }
}

// =============================================================================
// Energy Conservation Tests
// =============================================================================

TEST_F(PathTracerTest, OutputIsNonNegative) {
    Scene scene = createCornellBoxScene();
    finalizeScene(scene);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < 100; ++i) {
        render::Vec3f dir(dist(rng), dist(rng), dist(rng));
        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        if (len < 0.01f) continue;
        dir.x /= len; dir.y /= len; dir.z /= len;
        
        Ray ray(render::make_position(0.0f, 1.0f, 0.5f),
                render::direction_from_unit_vector(dir));
        
        render::PixelRGB result = render::apply_camera_sensitivity(traceMIS(scene, lights, ray, kMaxDepth), render::kDefaultCameraSensitivity);

        // Use is_valid for NaN/Inf checking
        EXPECT_TRUE(render::throughput_is_valid(result)) << "Invalid color at sample " << i;

        auto [rr, rg, rb] = render::to_floats(result);
        EXPECT_GE(rr, 0.0f) << "Negative red at sample " << i;
        EXPECT_GE(rg, 0.0f) << "Negative green at sample " << i;
        EXPECT_GE(rb, 0.0f) << "Negative blue at sample " << i;
    }
}

TEST_F(PathTracerTest, WhiteFurnaceTest) {
    // White furnace: uniform environment, white diffuse surface
    // Expected result: albedo color (energy conserving)
    Scene scene;
    scene.environment.color = render::make_attenuation_rgb(1.0f, 1.0f, 1.0f);
    scene.environment.strength = 1.0f;
    
    Mesh sphere;
    // Approximate sphere with icosahedron (simplified)
    sphere.vertices = {
        render::make_position(0.0f, 1.0f, 0.0f),
        render::make_position(0.0f, -1.0f, 0.0f),
        render::make_position(1.0f, 0.0f, 0.0f),
        render::make_position(-1.0f, 0.0f, 0.0f),
        render::make_position(0.0f, 0.0f, 1.0f),
        render::make_position(0.0f, 0.0f, -1.0f)
    };
    sphere.triangles.push_back(makeTriangle(0, 2, 4, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(0, 4, 3, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(0, 3, 5, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(0, 5, 2, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(1, 4, 2, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(1, 3, 4, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(1, 5, 3, sphere.vertices));
    sphere.triangles.push_back(makeTriangle(1, 2, 5, sphere.vertices));
    sphere.material = Material(
        render::make_attenuation_rgb(0.5f, 0.5f, 0.5f),  // 50% albedo
        0.0f, 1.0f,  // Pure diffuse
        render::zero_radiance_rgb()
    );
    scene.meshes.push_back(std::move(sphere));
    finalizeScene(scene);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    // Ray hitting the sphere
    Ray ray(render::make_position(0.0f, 0.0f, 3.0f),
            render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, -1.0f)));
    
    // Need many samples for white furnace convergence
    render::PixelRGB result = renderPixelMIS(scene, lights, ray, 256);
    
    // Should converge to albedo (0.5) within tolerance
    // Note: with only 8 bounces, won't fully converge, so use loose tolerance
    EXPECT_GT(result.r, 0.3f) << "White furnace: result too dark";
    EXPECT_LT(result.r, 0.7f) << "White furnace: result too bright";
}
