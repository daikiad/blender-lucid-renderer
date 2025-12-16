#pragma once

/**
 * @file test_utils.hpp
 * @brief Shared test utilities for DIY Renderer tests
 */

#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>

#include "units/render_units.hpp"
#include "core/scene.hpp"
#include "geometry/intersection.hpp"
#include "light/light.hpp"

namespace test_utils {

// =============================================================================
// Constants
// =============================================================================

constexpr float kEpsilon = 1e-5f;
constexpr float kLooseEpsilon = 1e-3f;
constexpr float kRelativeEpsilon = 0.05f;  // 5% relative error for convergence tests

// =============================================================================
// Float Comparison Helpers
// =============================================================================

inline bool approxEqual(float a, float b, float eps = kEpsilon) {
    return std::abs(a - b) < eps;
}

inline bool relativeEqual(float a, float b, float relEps = kRelativeEpsilon) {
    float maxVal = std::max(std::abs(a), std::abs(b));
    if (maxVal < kEpsilon) return true;  // Both near zero
    return std::abs(a - b) / maxVal < relEps;
}

// Attenuation RGB comparison (for albedo, coefficients [0,1])
inline bool attenuationApproxEqual(const render::AttenuationRGB& a, const render::AttenuationRGB& b, float eps = kEpsilon) {
    auto [ar, ag, ab] = render::to_floats(a);
    auto [br, bg, bb] = render::to_floats(b);
    return approxEqual(ar, br, eps) && 
           approxEqual(ag, bg, eps) && 
           approxEqual(ab, bb, eps);
}

inline bool attenuationRelativeEqual(const render::AttenuationRGB& a, const render::AttenuationRGB& b, float relEps = kRelativeEpsilon) {
    auto [ar, ag, ab] = render::to_floats(a);
    auto [br, bg, bb] = render::to_floats(b);
    return relativeEqual(ar, br, relEps) && 
           relativeEqual(ag, bg, relEps) && 
           relativeEqual(ab, bb, relEps);
}

// Throughput RGB comparison (for path weights [0,∞))
inline bool throughputApproxEqual(const render::ThroughputRGB& a, const render::ThroughputRGB& b, float eps = kEpsilon) {
    auto [ar, ag, ab] = render::to_floats(a);
    auto [br, bg, bb] = render::to_floats(b);
    return approxEqual(ar, br, eps) && 
           approxEqual(ag, bg, eps) && 
           approxEqual(ab, bb, eps);
}

inline bool throughputRelativeEqual(const render::ThroughputRGB& a, const render::ThroughputRGB& b, float relEps = kRelativeEpsilon) {
    auto [ar, ag, ab] = render::to_floats(a);
    auto [br, bg, bb] = render::to_floats(b);
    return relativeEqual(ar, br, relEps) && 
           relativeEqual(ag, bg, relEps) && 
           relativeEqual(ab, bb, relEps);
}

// RGB3f comparison (generic dimensionless RGB)
inline bool rgb3fApproxEqual(const render::RGB3f& a, const render::RGB3f& b, float eps = kEpsilon) {
    auto [ar, ag, ab] = render::to_floats(a);
    auto [br, bg, bb] = render::to_floats(b);
    return approxEqual(ar, br, eps) && 
           approxEqual(ag, bg, eps) && 
           approxEqual(ab, bb, eps);
}

inline bool rgb3fRelativeEqual(const render::RGB3f& a, const render::RGB3f& b, float relEps = kRelativeEpsilon) {
    auto [ar, ag, ab] = render::to_floats(a);
    auto [br, bg, bb] = render::to_floats(b);
    return relativeEqual(ar, br, relEps) && 
           relativeEqual(ag, bg, relEps) && 
           relativeEqual(ab, bb, relEps);
}

// =============================================================================
// Triangle Helper
// =============================================================================

inline Triangle makeTriangle(int i0, int i1, int i2, 
                              const std::vector<render::Position>& vertices) {
    Triangle tri;
    tri.i0 = i0;
    tri.i1 = i1;
    tri.i2 = i2;
    tri.smooth = false;
    tri.hasUV = false;
    
    // Compute face normal
    const render::Position& a = vertices[i0];
    const render::Position& b = vertices[i1];
    const render::Position& c = vertices[i2];
    render::Displacement ab = b - a;
    render::Displacement ac = c - a;
    // Use disp_to_vec to extract Vec3f from Displacement for cross product
    auto cross = render::cross(render::disp_to_vec(ab), render::disp_to_vec(ac));
    tri.faceNormal = render::make_normal_or_default(
        cross,
        render::normal_from_unit_vector(render::Vec3f(0.0f, 1.0f, 0.0f))
    );
    tri.n0 = tri.n1 = tri.n2 = tri.faceNormal;
    
    // Compute centroid
    tri.centroid = a + (ab + ac) / 3.0f;
    
    return tri;
}

// =============================================================================
// Test Scene Builders
// =============================================================================

/**
 * Create a minimal scene with a single triangle
 */
inline Scene createSingleTriangleScene() {
    Scene scene;
    
    Mesh mesh;
    mesh.vertices = {
        render::make_position(0.0f, 0.0f, 0.0f),
        render::make_position(1.0f, 0.0f, 0.0f),
        render::make_position(0.5f, 1.0f, 0.0f)
    };
    mesh.triangles.push_back(makeTriangle(0, 1, 2, mesh.vertices));
    mesh.material = Material(
        render::make_color_rgb(0.8f, 0.8f, 0.8f),  // albedo
        0.0f,  // metallic
        0.5f,  // roughness
        render::zero_radiance_rgb()  // emission
    );
    
    scene.meshes.push_back(std::move(mesh));
    return scene;
}

/**
 * Create a Cornell Box-like scene
 */
inline Scene createCornellBoxScene() {
    Scene scene;
    
    // Floor (white)
    {
        Mesh floor;
        floor.vertices = {
            render::make_position(-1.0f, 0.0f, -1.0f),
            render::make_position( 1.0f, 0.0f, -1.0f),
            render::make_position( 1.0f, 0.0f,  1.0f),
            render::make_position(-1.0f, 0.0f,  1.0f)
        };
        floor.triangles.push_back(makeTriangle(0, 1, 2, floor.vertices));
        floor.triangles.push_back(makeTriangle(0, 2, 3, floor.vertices));
        floor.material = Material(
            render::make_color_rgb(0.73f, 0.73f, 0.73f),
            0.0f, 0.5f, render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(floor));
    }
    
    // Ceiling (white)
    {
        Mesh ceiling;
        ceiling.vertices = {
            render::make_position(-1.0f, 2.0f, -1.0f),
            render::make_position( 1.0f, 2.0f, -1.0f),
            render::make_position( 1.0f, 2.0f,  1.0f),
            render::make_position(-1.0f, 2.0f,  1.0f)
        };
        ceiling.triangles.push_back(makeTriangle(0, 2, 1, ceiling.vertices));
        ceiling.triangles.push_back(makeTriangle(0, 3, 2, ceiling.vertices));
        ceiling.material = Material(
            render::make_color_rgb(0.73f, 0.73f, 0.73f),
            0.0f, 0.5f, render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(ceiling));
    }
    
    // Back wall (white)
    {
        Mesh backWall;
        backWall.vertices = {
            render::make_position(-1.0f, 0.0f, -1.0f),
            render::make_position( 1.0f, 0.0f, -1.0f),
            render::make_position( 1.0f, 2.0f, -1.0f),
            render::make_position(-1.0f, 2.0f, -1.0f)
        };
        backWall.triangles.push_back(makeTriangle(0, 1, 2, backWall.vertices));
        backWall.triangles.push_back(makeTriangle(0, 2, 3, backWall.vertices));
        backWall.material = Material(
            render::make_color_rgb(0.73f, 0.73f, 0.73f),
            0.0f, 0.5f, render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(backWall));
    }
    
    // Left wall (red)
    {
        Mesh leftWall;
        leftWall.vertices = {
            render::make_position(-1.0f, 0.0f, -1.0f),
            render::make_position(-1.0f, 0.0f,  1.0f),
            render::make_position(-1.0f, 2.0f,  1.0f),
            render::make_position(-1.0f, 2.0f, -1.0f)
        };
        leftWall.triangles.push_back(makeTriangle(0, 1, 2, leftWall.vertices));
        leftWall.triangles.push_back(makeTriangle(0, 2, 3, leftWall.vertices));
        leftWall.material = Material(
            render::make_color_rgb(0.65f, 0.05f, 0.05f),
            0.0f, 0.5f, render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(leftWall));
    }
    
    // Right wall (green)
    {
        Mesh rightWall;
        rightWall.vertices = {
            render::make_position(1.0f, 0.0f, -1.0f),
            render::make_position(1.0f, 0.0f,  1.0f),
            render::make_position(1.0f, 2.0f,  1.0f),
            render::make_position(1.0f, 2.0f, -1.0f)
        };
        rightWall.triangles.push_back(makeTriangle(0, 2, 1, rightWall.vertices));
        rightWall.triangles.push_back(makeTriangle(0, 3, 2, rightWall.vertices));
        rightWall.material = Material(
            render::make_color_rgb(0.12f, 0.45f, 0.15f),
            0.0f, 0.5f, render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(rightWall));
    }
    
    // Emissive light on ceiling
    {
        Mesh light;
        light.vertices = {
            render::make_position(-0.25f, 1.99f, -0.25f),
            render::make_position( 0.25f, 1.99f, -0.25f),
            render::make_position( 0.25f, 1.99f,  0.25f),
            render::make_position(-0.25f, 1.99f,  0.25f)
        };
        light.triangles.push_back(makeTriangle(0, 2, 1, light.vertices));
        light.triangles.push_back(makeTriangle(0, 3, 2, light.vertices));
        light.material = Material(
            render::make_color_rgb(1.0f, 1.0f, 1.0f),
            0.0f, 0.5f,
            render::make_radiance_rgb(15.0f, 15.0f, 15.0f)
        );
        scene.meshes.push_back(std::move(light));
    }
    
    return scene;
}

/**
 * Create a scene with a mirror and point light (for MIS testing)
 */
inline Scene createMirrorPointLightScene() {
    Scene scene;
    
    // Floor (diffuse)
    {
        Mesh floor;
        floor.vertices = {
            render::make_position(-2.0f, 0.0f, -2.0f),
            render::make_position( 2.0f, 0.0f, -2.0f),
            render::make_position( 2.0f, 0.0f,  2.0f),
            render::make_position(-2.0f, 0.0f,  2.0f)
        };
        floor.triangles.push_back(makeTriangle(0, 1, 2, floor.vertices));
        floor.triangles.push_back(makeTriangle(0, 2, 3, floor.vertices));
        floor.material = Material(
            render::make_color_rgb(0.8f, 0.8f, 0.8f),
            0.0f, 0.5f, render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(floor));
    }
    
    // Mirror (metallic, low roughness)
    {
        Mesh mirror;
        mirror.vertices = {
            render::make_position(-0.5f, 0.5f, -1.0f),
            render::make_position( 0.5f, 0.5f, -1.0f),
            render::make_position( 0.5f, 1.5f, -1.0f),
            render::make_position(-0.5f, 1.5f, -1.0f)
        };
        mirror.triangles.push_back(makeTriangle(0, 1, 2, mirror.vertices));
        mirror.triangles.push_back(makeTriangle(0, 2, 3, mirror.vertices));
        mirror.material = Material(
            render::make_color_rgb(0.95f, 0.95f, 0.95f),
            1.0f,   // metallic
            0.0f,   // roughness (will be clamped to MIN_ROUGHNESS)
            render::zero_radiance_rgb()
        );
        scene.meshes.push_back(std::move(mirror));
    }
    
    // Point light
    Light pointLight;
    pointLight.type = LightType::POINT;
    pointLight.position = render::make_position(0.0f, 2.0f, 0.0f);
    pointLight.normal = render::normal_from_unit_vector(render::Vec3f(0.0f, -1.0f, 0.0f));
    pointLight.radius = 0.1f * mp_units::si::metre;
    pointLight.area = 4.0f * 3.14159f * 0.01f * mp_units::square(mp_units::si::metre);
    pointLight.energy = 100.0f * mp_units::si::watt;
    pointLight.emission = render::make_radiance_rgb(100.0f, 100.0f, 100.0f);
    scene.nativeLights.push_back(pointLight);
    
    return scene;
}

// =============================================================================
// Statistics Helpers
// =============================================================================

struct Statistics {
    float mean = 0.0f;
    float variance = 0.0f;
    float stdDev = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
};

inline Statistics computeStatistics(const std::vector<float>& values) {
    Statistics stats;
    if (values.empty()) return stats;
    
    float sum = 0.0f;
    stats.min = values[0];
    stats.max = values[0];
    
    for (float v : values) {
        sum += v;
        stats.min = std::min(stats.min, v);
        stats.max = std::max(stats.max, v);
    }
    stats.mean = sum / static_cast<float>(values.size());
    
    float varSum = 0.0f;
    for (float v : values) {
        float diff = v - stats.mean;
        varSum += diff * diff;
    }
    stats.variance = varSum / static_cast<float>(values.size());
    stats.stdDev = std::sqrt(stats.variance);
    
    return stats;
}

// =============================================================================
// File Helpers
// =============================================================================

inline std::string getTestScenesDir() {
    // Relative to build directory
    return "../test_scenes/";
}

inline std::string readFileContents(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}  // namespace test_utils
