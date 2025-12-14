/**
 * @file test_scene_lights.cpp
 * @brief Tests for SceneLights construction and light index mapping
 * 
 * These tests verify:
 * - SceneLights builds correctly from scenes with emissive meshes and native lights
 * - findNativeLightIndex() correctly maps scene.nativeLights indices to sceneLights.lights indices
 * - CDF construction for importance sampling
 * - Light selection probability calculations
 */

#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "light/scene_lights.hpp"
#include "core/scene.hpp"

using namespace test_utils;

class SceneLightsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST_F(SceneLightsTest, EmptySceneProducesEmptyLights) {
    Scene scene;
    SceneLights lights;
    lights.buildFromScene(scene);
    
    EXPECT_TRUE(lights.lights.empty());
    EXPECT_FALSE(lights.hasLights());
    EXPECT_EQ(lights.nativeLightStartIndex, 0);
}

TEST_F(SceneLightsTest, SingleEmissiveMesh) {
    Scene scene;
    
    Mesh emissive;
    emissive.vertices = {
        render::make_position(0.0f, 1.0f, 0.0f),
        render::make_position(1.0f, 1.0f, 0.0f),
        render::make_position(0.5f, 1.0f, 1.0f)
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.material = Material(
        render::ColorRGB(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(10.0f, 10.0f, 10.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    EXPECT_EQ(lights.lights.size(), 1);
    EXPECT_TRUE(lights.hasLights());
    EXPECT_EQ(lights.nativeLightStartIndex, 1);  // Native lights start after emissive mesh
    EXPECT_EQ(lights.lights[0].type, LightType::EMISSIVE_MESH);
}

TEST_F(SceneLightsTest, SingleNativePointLight) {
    Scene scene;
    
    // Add a non-emissive mesh
    Mesh floor;
    floor.vertices = {
        render::make_position(-1.0f, 0.0f, -1.0f),
        render::make_position(1.0f, 0.0f, -1.0f),
        render::make_position(0.0f, 0.0f, 1.0f)
    };
    floor.triangles.push_back(makeTriangle(0, 1, 2, floor.vertices));
    floor.material = Material(
        render::ColorRGB(0.8f, 0.8f, 0.8f),
        0.0f, 0.5f,
        render::zero_radiance_rgb()
    );
    scene.meshes.push_back(std::move(floor));
    
    // Add point light
    Light pointLight;
    pointLight.type = LightType::POINT;
    pointLight.position = render::make_position(0.0f, 2.0f, 0.0f);
    pointLight.normal = render::normal_from_unit_vector(render::Vec3f(0.0f, -1.0f, 0.0f));
    pointLight.radius = 0.1f * mp_units::si::metre;
    pointLight.area = 0.1256f * mp_units::square(mp_units::si::metre);  // 4πr²
    pointLight.emission = render::make_radiance_rgb(100.0f, 100.0f, 100.0f);
    scene.nativeLights.push_back(pointLight);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    EXPECT_EQ(lights.lights.size(), 1);
    EXPECT_TRUE(lights.hasLights());
    EXPECT_EQ(lights.nativeLightStartIndex, 0);  // No emissive meshes
    EXPECT_EQ(lights.lights[0].type, LightType::POINT);
}

// =============================================================================
// Index Mapping Tests (Regression test for MIS bug)
// =============================================================================

TEST_F(SceneLightsTest, FindNativeLightIndexWithNoEmissiveMeshes) {
    Scene scene;
    
    // Two native lights, no emissive meshes
    Light light1, light2;
    light1.type = LightType::POINT;
    light1.position = render::make_position(0.0f, 2.0f, 0.0f);
    light1.area = 0.1f * mp_units::square(mp_units::si::metre);
    light1.emission = render::make_radiance_rgb(50.0f, 50.0f, 50.0f);
    
    light2.type = LightType::POINT;
    light2.position = render::make_position(1.0f, 2.0f, 0.0f);
    light2.area = 0.1f * mp_units::square(mp_units::si::metre);
    light2.emission = render::make_radiance_rgb(100.0f, 100.0f, 100.0f);
    
    scene.nativeLights.push_back(light1);
    scene.nativeLights.push_back(light2);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    EXPECT_EQ(lights.nativeLightStartIndex, 0);
    EXPECT_EQ(lights.findNativeLightIndex(0), 0);
    EXPECT_EQ(lights.findNativeLightIndex(1), 1);
    EXPECT_EQ(lights.findNativeLightIndex(-1), -1);  // Invalid index
    EXPECT_EQ(lights.findNativeLightIndex(2), -1);   // Out of bounds
}

TEST_F(SceneLightsTest, FindNativeLightIndexWithEmissiveMeshes) {
    // This is the key regression test for the MIS bug
    Scene scene;
    
    // Add emissive mesh with 2 triangles (creates 2 emissive triangle lights)
    Mesh emissive;
    emissive.vertices = {
        render::make_position(-0.5f, 2.0f, -0.5f),
        render::make_position(0.5f, 2.0f, -0.5f),
        render::make_position(0.5f, 2.0f, 0.5f),
        render::make_position(-0.5f, 2.0f, 0.5f)
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.triangles.push_back(makeTriangle(0, 2, 3, emissive.vertices));
    emissive.material = Material(
        render::ColorRGB(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(10.0f, 10.0f, 10.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    
    // Add native point light
    Light pointLight;
    pointLight.type = LightType::POINT;
    pointLight.position = render::make_position(0.0f, 3.0f, 0.0f);
    pointLight.area = 0.1f * mp_units::square(mp_units::si::metre);
    pointLight.emission = render::make_radiance_rgb(100.0f, 100.0f, 100.0f);
    scene.nativeLights.push_back(pointLight);
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    // Should have 2 emissive triangles + 1 point light = 3 total
    EXPECT_EQ(lights.lights.size(), 3);
    EXPECT_EQ(lights.nativeLightStartIndex, 2);  // Native lights start at index 2
    
    // lights[0], lights[1] = emissive triangles
    EXPECT_EQ(lights.lights[0].type, LightType::EMISSIVE_MESH);
    EXPECT_EQ(lights.lights[1].type, LightType::EMISSIVE_MESH);
    
    // lights[2] = native point light
    EXPECT_EQ(lights.lights[2].type, LightType::POINT);
    
    // findNativeLightIndex should correctly map
    // scene.nativeLights[0] -> sceneLights.lights[2]
    EXPECT_EQ(lights.findNativeLightIndex(0), 2);
    EXPECT_EQ(lights.findNativeLightIndex(1), -1);  // Only 1 native light
}

// =============================================================================
// CDF and Selection Tests
// =============================================================================

TEST_F(SceneLightsTest, CDFIsNormalized) {
    Scene scene;
    
    // Two emissive triangles with different areas
    Mesh emissive;
    emissive.vertices = {
        render::make_position(0.0f, 1.0f, 0.0f),
        render::make_position(2.0f, 1.0f, 0.0f),  // Large triangle
        render::make_position(1.0f, 1.0f, 2.0f),
        render::make_position(1.0f, 1.0f, 0.1f)   // Small triangle
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.triangles.push_back(makeTriangle(0, 3, 1, emissive.vertices));
    emissive.material = Material(
        render::ColorRGB(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(10.0f, 10.0f, 10.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    // CDF should end at 1.0
    ASSERT_FALSE(lights.cdf.empty());
    EXPECT_NEAR(lights.cdf.back(), 1.0f, 1e-5f);
}

TEST_F(SceneLightsTest, SelectLightReturnsValidIndex) {
    Scene scene;
    
    Mesh emissive;
    emissive.vertices = {
        render::make_position(0.0f, 1.0f, 0.0f),
        render::make_position(1.0f, 1.0f, 0.0f),
        render::make_position(0.5f, 1.0f, 1.0f)
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.material = Material(
        render::ColorRGB(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(10.0f, 10.0f, 10.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    float prob;
    for (int i = 0; i < 100; ++i) {
        float u = static_cast<float>(i) / 100.0f;
        int idx = lights.selectLight(u, prob);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int>(lights.lights.size()));
        EXPECT_GT(prob, 0.0f);
        EXPECT_LE(prob, 1.0f);
    }
}

TEST_F(SceneLightsTest, GetPdfForLightSumsToOne) {
    Scene scene;
    
    // Add multiple lights with different areas
    Mesh emissive;
    emissive.vertices = {
        render::make_position(0.0f, 1.0f, 0.0f),
        render::make_position(1.0f, 1.0f, 0.0f),
        render::make_position(0.5f, 1.0f, 0.5f),
        render::make_position(0.0f, 1.0f, 1.0f)
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.triangles.push_back(makeTriangle(0, 2, 3, emissive.vertices));
    emissive.material = Material(
        render::ColorRGB(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(10.0f, 10.0f, 10.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    float totalProb = 0.0f;
    for (size_t i = 0; i < lights.lights.size(); ++i) {
        totalProb += lights.getPdfForLight(static_cast<int>(i));
    }
    EXPECT_NEAR(totalProb, 1.0f, 1e-5f);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(SceneLightsTest, NonEmissiveMeshesIgnored) {
    Scene scene;
    
    // Add non-emissive mesh
    Mesh floor;
    floor.vertices = {
        render::make_position(-1.0f, 0.0f, -1.0f),
        render::make_position(1.0f, 0.0f, -1.0f),
        render::make_position(0.0f, 0.0f, 1.0f)
    };
    floor.triangles.push_back(makeTriangle(0, 1, 2, floor.vertices));
    floor.material = Material(
        render::ColorRGB(0.8f, 0.8f, 0.8f),
        0.0f, 0.5f,
        render::zero_radiance_rgb()  // No emission
    );
    scene.meshes.push_back(std::move(floor));
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    EXPECT_TRUE(lights.lights.empty());
    EXPECT_FALSE(lights.hasLights());
}

TEST_F(SceneLightsTest, DegenerateTrianglesIgnored) {
    Scene scene;
    
    // Degenerate triangle (zero area) - use collinear points
    Mesh emissive;
    emissive.vertices = {
        render::make_position(0.0f, 1.0f, 0.0f),
        render::make_position(1.0f, 1.0f, 0.0f),  // Collinear
        render::make_position(2.0f, 1.0f, 0.0f)   // Collinear (zero area)
    };
    emissive.triangles.push_back(makeTriangle(0, 1, 2, emissive.vertices));
    emissive.material = Material(
        render::ColorRGB(1.0f, 1.0f, 1.0f),
        0.0f, 0.5f,
        render::make_radiance_rgb(10.0f, 10.0f, 10.0f)
    );
    scene.meshes.push_back(std::move(emissive));
    
    SceneLights lights;
    lights.buildFromScene(scene);
    
    // Degenerate triangle should be skipped
    EXPECT_TRUE(lights.lights.empty());
}
