/**
 * test_light.cpp - Light Sampling Tests
 *
 * Comprehensive tests for light sources and sampling:
 * - Light creation and initialization
 * - Triangle area calculation
 * - Triangle point sampling
 * - Light sampling (various types)
 * - PDF calculation
 */

#include "light/light.hpp"
#include "core/ray.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace render;
using namespace mp_units::si::unit_symbols;

constexpr float kEps = 1e-4f;

// ============================================================================
// Triangle Area Tests
// ============================================================================

class TriangleAreaTest : public ::testing::Test {
protected:
    // Unit right triangle: vertices at (0,0,0), (1,0,0), (0,1,0)
    Position v0 = make_position(0.0f, 0.0f, 0.0f);
    Position v1 = make_position(1.0f, 0.0f, 0.0f);
    Position v2 = make_position(0.0f, 1.0f, 0.0f);
};

TEST_F(TriangleAreaTest, UnitRightTriangle) {
    // Area = 0.5 * base * height = 0.5 * 1 * 1 = 0.5 m²
    Area area = triangleArea(v0, v1, v2);
    EXPECT_NEAR(area.numerical_value_in(m2), 0.5f, kEps);
}

TEST_F(TriangleAreaTest, ScaledTriangle) {
    // Scale vertices by 2x -> area scales by 4x
    Position sv0 = make_position(0.0f, 0.0f, 0.0f);
    Position sv1 = make_position(2.0f, 0.0f, 0.0f);
    Position sv2 = make_position(0.0f, 2.0f, 0.0f);
    
    Area area = triangleArea(sv0, sv1, sv2);
    EXPECT_NEAR(area.numerical_value_in(m2), 2.0f, kEps);  // 0.5 * 4 = 2
}

TEST_F(TriangleAreaTest, DegenerateTriangle_Collinear) {
    // Collinear vertices -> zero area
    Position dv0 = make_position(0.0f, 0.0f, 0.0f);
    Position dv1 = make_position(1.0f, 0.0f, 0.0f);
    Position dv2 = make_position(2.0f, 0.0f, 0.0f);
    
    Area area = triangleArea(dv0, dv1, dv2);
    EXPECT_NEAR(area.numerical_value_in(m2), 0.0f, kEps);
}

TEST_F(TriangleAreaTest, EquilateralTriangle) {
    // Equilateral triangle with side 1: area = sqrt(3)/4 ≈ 0.433
    float h = std::sqrt(3.0f) / 2.0f;
    Position ev0 = make_position(0.0f, 0.0f, 0.0f);
    Position ev1 = make_position(1.0f, 0.0f, 0.0f);
    Position ev2 = make_position(0.5f, h, 0.0f);
    
    Area area = triangleArea(ev0, ev1, ev2);
    float expected = std::sqrt(3.0f) / 4.0f;
    EXPECT_NEAR(area.numerical_value_in(m2), expected, kEps);
}

TEST_F(TriangleAreaTest, TriangleIn3D) {
    // Triangle not in XY plane
    Position p0 = make_position(0.0f, 0.0f, 0.0f);
    Position p1 = make_position(1.0f, 0.0f, 1.0f);
    Position p2 = make_position(0.0f, 1.0f, 1.0f);
    
    Area area = triangleArea(p0, p1, p2);
    // Should be non-zero and calculable
    EXPECT_GT(area.numerical_value_in(m2), 0.0f);
}

// ============================================================================
// Triangle Point Sampling Tests
// ============================================================================

class TrianglePointSamplingTest : public ::testing::Test {
protected:
    Position v0 = make_position(0.0f, 0.0f, 0.0f);
    Position v1 = make_position(1.0f, 0.0f, 0.0f);
    Position v2 = make_position(0.0f, 1.0f, 0.0f);
};

TEST_F(TrianglePointSamplingTest, CornerSampling_V0) {
    // u1=0, u2=0 should give vertex v0
    Position p = sampleTrianglePoint(v0, v1, v2, 0.0f, 0.0f);
    Vec3f pv = displacement_from_origin(p).numerical_value_in(m);
    EXPECT_NEAR(pv.x, 0.0f, kEps);
    EXPECT_NEAR(pv.y, 0.0f, kEps);
}

TEST_F(TrianglePointSamplingTest, CornerSampling_V1) {
    // u1=1, u2=1 should give vertex v1
    Position p = sampleTrianglePoint(v0, v1, v2, 1.0f, 1.0f);
    Vec3f pv = displacement_from_origin(p).numerical_value_in(m);
    EXPECT_NEAR(pv.x, 1.0f, kEps);
    EXPECT_NEAR(pv.y, 0.0f, kEps);
}

TEST_F(TrianglePointSamplingTest, CornerSampling_V2) {
    // u1=1, u2=0 should give vertex v2
    Position p = sampleTrianglePoint(v0, v1, v2, 1.0f, 0.0f);
    Vec3f pv = displacement_from_origin(p).numerical_value_in(m);
    EXPECT_NEAR(pv.x, 0.0f, kEps);
    EXPECT_NEAR(pv.y, 1.0f, kEps);
}

TEST_F(TrianglePointSamplingTest, PointInsideTriangle) {
    // Random (0.25, 0.25) should be inside triangle
    Position p = sampleTrianglePoint(v0, v1, v2, 0.25f, 0.25f);
    Vec3f pv = displacement_from_origin(p).numerical_value_in(m);
    
    float x = pv.x;
    float y = pv.y;
    
    // Check barycentric constraints: x >= 0, y >= 0, x + y <= 1
    EXPECT_GE(x, 0.0f);
    EXPECT_GE(y, 0.0f);
    EXPECT_LE(x + y, 1.0f + kEps);
}

TEST_F(TrianglePointSamplingTest, AllSamplesInsideTriangle) {
    // Statistical test: many samples should all be inside
    uint32_t seed = 12345;
    for (int i = 0; i < 100; ++i) {
        float u1 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        float u2 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        
        Position p = sampleTrianglePoint(v0, v1, v2, u1, u2);
        Vec3f pv = displacement_from_origin(p).numerical_value_in(m);
        
        float x = pv.x;
        float y = pv.y;
        
        EXPECT_GE(x, -kEps);
        EXPECT_GE(y, -kEps);
        EXPECT_LE(x + y, 1.0f + kEps);
    }
}

// ============================================================================
// Light Initialization Tests
// ============================================================================

TEST(LightInitializationTest, DefaultConstruction) {
    Light light;
    EXPECT_EQ(light.type, LightType::EMISSIVE_MESH);
    EXPECT_EQ(light.meshIndex, -1);
    EXPECT_EQ(light.triangleIndex, -1);
}

TEST(LightInitializationTest, PointLightSetup) {
    Light light;
    light.type = LightType::POINT;
    light.position = make_position(1.0f, 2.0f, 3.0f);
    light.energy = 100.0f * W;
    
    EXPECT_EQ(light.type, LightType::POINT);
    Vec3f pos = displacement_from_origin(light.position).numerical_value_in(m);
    EXPECT_NEAR(pos.x, 1.0f, kEps);
    EXPECT_NEAR(light.energy.numerical_value_in(W), 100.0f, kEps);
}

TEST(LightInitializationTest, AreaLightSetup) {
    Light light;
    light.type = LightType::AREA;
    light.position = make_position(0.0f, 0.0f, 0.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    light.sizeX = 2.0f * m;
    light.sizeY = 2.0f * m;
    light.area = light.sizeX * light.sizeY;
    
    EXPECT_NEAR(light.area.numerical_value_in(m2), 4.0f, kEps);
}

TEST(LightInitializationTest, EmissiveMeshSetup) {
    Light light;
    light.type = LightType::EMISSIVE_MESH;
    light.v0 = make_position(0.0f, 0.0f, 0.0f);
    light.v1 = make_position(1.0f, 0.0f, 0.0f);
    light.v2 = make_position(0.0f, 1.0f, 0.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    light.area = triangleArea(light.v0, light.v1, light.v2);
    light.meshIndex = 0;
    light.triangleIndex = 5;
    
    EXPECT_NEAR(light.area.numerical_value_in(m2), 0.5f, kEps);
    EXPECT_EQ(light.meshIndex, 0);
    EXPECT_EQ(light.triangleIndex, 5);
}

// ============================================================================
// Light Sample Tests
// ============================================================================

TEST(LightSampleTest, DefaultConstruction) {
    LightSample sample;
    EXPECT_FALSE(sample.isValid());  // Default PDF is 0
}

TEST(LightSampleTest, ValidSample) {
    LightSample sample;
    sample.pdf = 1.0f * per_sr;
    sample.distance = 1.0f * m;
    EXPECT_TRUE(sample.isValid());
}

TEST(LightSampleTest, InvalidSample_ZeroPDF) {
    LightSample sample;
    sample.pdf = 0.0f * per_sr;
    sample.distance = 1.0f * m;
    EXPECT_FALSE(sample.isValid());
}

TEST(LightSampleTest, InvalidSample_ZeroDistance) {
    LightSample sample;
    sample.pdf = 1.0f * per_sr;
    sample.distance = 0.0f * m;
    EXPECT_FALSE(sample.isValid());
}

// ============================================================================
// Light Sampling Integration Tests
// ============================================================================

class LightSamplingTest : public ::testing::Test {
protected:
    Position shadingPoint = make_position(0.0f, 0.0f, 0.0f);
};

TEST_F(LightSamplingTest, PointLight_BasicSampling) {
    Light light;
    light.type = LightType::POINT;
    light.position = make_position(0.0f, 0.0f, 5.0f);
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    light.radius = 0.0f * m;
    
    LightSample sample = sampleLight(light, shadingPoint, 0.5f, 0.5f);
    
    EXPECT_TRUE(sample.isValid());
    EXPECT_NEAR(sample.distance.numerical_value_in(m), 5.0f, kEps);
}

TEST_F(LightSamplingTest, SunLight_BasicSampling) {
    Light light;
    light.type = LightType::SUN;
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));  // Sun shining down
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    LightSample sample = sampleLight(light, shadingPoint, 0.5f, 0.5f);
    
    EXPECT_TRUE(sample.isValid());
    // Direction should point up toward sun
    EXPECT_GT(sample.direction.z(), 0.0f);
}

TEST_F(LightSamplingTest, AreaLight_BasicSampling) {
    Light light;
    light.type = LightType::AREA;
    light.position = make_position(0.0f, 0.0f, 3.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));  // Facing down
    light.right = direction_from_unit_vector(Vec3f(1.0f, 0.0f, 0.0f));
    light.up = direction_from_unit_vector(Vec3f(0.0f, 1.0f, 0.0f));
    light.sizeX = 2.0f * m;
    light.sizeY = 2.0f * m;
    light.area = light.sizeX * light.sizeY;
    light.shape = AreaLightShape::SQUARE;
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    LightSample sample = sampleLight(light, shadingPoint, 0.5f, 0.5f);
    
    EXPECT_TRUE(sample.isValid());
    // Distance should be around 3m (center of area light)
    EXPECT_NEAR(sample.distance.numerical_value_in(m), 3.0f, 0.5f);
}

TEST_F(LightSamplingTest, EmissiveMesh_BasicSampling) {
    Light light;
    light.type = LightType::EMISSIVE_MESH;
    light.v0 = make_position(-1.0f, -1.0f, 5.0f);
    light.v1 = make_position(1.0f, -1.0f, 5.0f);
    light.v2 = make_position(0.0f, 1.0f, 5.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));
    light.area = triangleArea(light.v0, light.v1, light.v2);
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    LightSample sample = sampleLight(light, shadingPoint, 0.5f, 0.5f);
    
    EXPECT_TRUE(sample.isValid());
    // Should sample somewhere on the triangle at z=5
    Vec3f pos = displacement_from_origin(sample.position).numerical_value_in(m);
    EXPECT_NEAR(pos.z, 5.0f, kEps);
}

TEST_F(LightSamplingTest, SpotLight_BasicSampling) {
    Light light;
    light.type = LightType::SPOT;
    light.position = make_position(0.0f, 0.0f, 5.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));  // Pointing down
    light.spotAngle = 0.5f * rad;  // ~28.6 degree cone
    light.spotBlend = 0.15f;
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    LightSample sample = sampleLight(light, shadingPoint, 0.5f, 0.5f);
    
    EXPECT_TRUE(sample.isValid());
    // Should have non-zero emission (shading point is inside cone)
    EXPECT_GT(sample.emission.r.numerical_value_in(W / sr / m2), 0.0f);
}

TEST_F(LightSamplingTest, SpotLight_OutsideCone) {
    Light light;
    light.type = LightType::SPOT;
    light.position = make_position(0.0f, 0.0f, 5.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));  // Pointing UP (away from shading point)
    light.spotAngle = 0.5f * rad;
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    LightSample sample = sampleLight(light, shadingPoint, 0.5f, 0.5f);
    
    // Shading point should be outside cone
    EXPECT_NEAR(sample.emission.r.numerical_value_in(W / sr / m2), 0.0f, kEps);
}

// ============================================================================
// PDF Calculation Tests
// ============================================================================

TEST(LightPDFTest, AreaLight_PDFCalculation) {
    Light light;
    light.type = LightType::AREA;
    light.position = make_position(0.0f, 0.0f, 3.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));
    light.area = 4.0f * m2;
    
    Position shadingPoint = make_position(0.0f, 0.0f, 0.0f);
    Position lightPoint = make_position(0.0f, 0.0f, 3.0f);
    
    PdfW pdf = pdfLightSample(light, shadingPoint, lightPoint, light.normal);
    
    // PDF = d² / (A × cosθ) = 9 / (4 × 1) = 2.25 / sr
    EXPECT_NEAR(pdf.numerical_value_in(per_sr), 2.25f, kEps);
}

TEST(LightPDFTest, GrazingAngle_ZeroPDF) {
    Light light;
    light.type = LightType::AREA;
    light.position = make_position(0.0f, 0.0f, 0.0f);
    light.normal = normal_from_unit_vector(Vec3f(1.0f, 0.0f, 0.0f));  // Normal perpendicular to view direction
    light.area = 4.0f * m2;
    
    Position shadingPoint = make_position(0.0f, 0.0f, -3.0f);
    Position lightPoint = make_position(0.0f, 0.0f, 0.0f);
    
    PdfW pdf = pdfLightSample(light, shadingPoint, lightPoint, light.normal);
    
    // cosθ ≈ 0 → PDF should be 0
    EXPECT_NEAR(pdf.numerical_value_in(per_sr), 0.0f, kEps);
}

TEST(LightPDFTest, ZeroArea_ZeroPDF) {
    Light light;
    light.type = LightType::AREA;
    light.area = 0.0f * m2;
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    
    Position shadingPoint = make_position(0.0f, 0.0f, 0.0f);
    Position lightPoint = make_position(0.0f, 0.0f, 3.0f);
    
    PdfW pdf = pdfLightSample(light, shadingPoint, lightPoint, light.normal);
    
    EXPECT_NEAR(pdf.numerical_value_in(per_sr), 0.0f, kEps);
}

// ============================================================================
// Statistical Sampling Tests
// ============================================================================

TEST(LightStatisticalTest, AreaLight_UniformCoverage) {
    // Sample area light many times and verify coverage
    Light light;
    light.type = LightType::AREA;
    light.position = make_position(0.0f, 0.0f, 5.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));
    light.right = direction_from_unit_vector(Vec3f(1.0f, 0.0f, 0.0f));
    light.up = direction_from_unit_vector(Vec3f(0.0f, 1.0f, 0.0f));
    light.sizeX = 2.0f * m;
    light.sizeY = 2.0f * m;
    light.area = light.sizeX * light.sizeY;
    light.shape = AreaLightShape::SQUARE;
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    Position shadingPoint = make_position(0.0f, 0.0f, 0.0f);
    
    // Count samples in quadrants
    int quadrants[4] = {0, 0, 0, 0};
    uint32_t seed = 12345;
    const int numSamples = 1000;
    
    for (int i = 0; i < numSamples; ++i) {
        float u1 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        float u2 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        
        LightSample sample = sampleLight(light, shadingPoint, u1, u2);
        
        if (sample.isValid()) {
            Vec3f pos = displacement_from_origin(sample.position).numerical_value_in(m);
            float x = pos.x;
            float y = pos.y;
            
            int qx = x >= 0 ? 1 : 0;
            int qy = y >= 0 ? 2 : 0;
            quadrants[qx + qy]++;
        }
    }
    
    // Each quadrant should have roughly 25% of samples (±10% tolerance)
    int expectedPerQuadrant = numSamples / 4;
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(quadrants[i], expectedPerQuadrant * 0.15);
        EXPECT_LT(quadrants[i], expectedPerQuadrant * 1.85);
    }
}

TEST(LightStatisticalTest, DiskLight_WithinRadius) {
    Light light;
    light.type = LightType::AREA;
    light.position = make_position(0.0f, 0.0f, 5.0f);
    light.normal = normal_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));
    light.right = direction_from_unit_vector(Vec3f(1.0f, 0.0f, 0.0f));
    light.up = direction_from_unit_vector(Vec3f(0.0f, 1.0f, 0.0f));
    light.sizeX = 2.0f * m;  // radius = 1
    light.sizeY = 2.0f * m;
    float radiusVal = 1.0f;
    light.area = std::numbers::pi_v<float> * radiusVal * radiusVal * m2;
    light.shape = AreaLightShape::DISK;
    light.emission = make_radiance_rgb(1.0f, 1.0f, 1.0f);
    
    Position shadingPoint = make_position(0.0f, 0.0f, 0.0f);
    
    uint32_t seed = 54321;
    
    for (int i = 0; i < 100; ++i) {
        float u1 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        float u2 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        
        LightSample sample = sampleLight(light, shadingPoint, u1, u2);
        
        if (sample.isValid()) {
            Vec3f pos = displacement_from_origin(sample.position).numerical_value_in(m);
            float x = pos.x;
            float y = pos.y;
            float r2 = x * x + y * y;
            
            // All samples should be within disk radius
            EXPECT_LE(r2, radiusVal * radiusVal + kEps);
        }
    }
}
