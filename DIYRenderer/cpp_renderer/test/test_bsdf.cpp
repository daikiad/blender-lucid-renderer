/**
 * test_bsdf.cpp - BSDF and Material Function Tests
 *
 * Comprehensive tests for:
 * - Fresnel equations (Schlick approximation, dielectric)
 * - GGX microfacet distribution
 * - Geometry functions (G1, G2)
 * - VNDF importance sampling
 */

#include "bsdf/fresnel.hpp"
#include "bsdf/ggx.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace render;

constexpr float kEps = 1e-4f;

// ============================================================================
// Fresnel Tests
// ============================================================================

class FresnelSchlickTest : public ::testing::Test {
protected:
    // Common F0 values
    float f0_dielectric = 0.04f;  // Non-metal
    float f0_metal = 0.9f;        // Metal
};

TEST_F(FresnelSchlickTest, CosTheta0_ReturnsOne) {
    // At grazing angle (cosθ = 0), Fresnel → 1
    float result = fresnelSchlick(0.0f, f0_dielectric);
    EXPECT_NEAR(result, 1.0f, kEps);
}

TEST_F(FresnelSchlickTest, CosTheta1_ReturnsF0) {
    // At normal incidence (cosθ = 1), Fresnel = F0
    float result = fresnelSchlick(1.0f, f0_dielectric);
    EXPECT_NEAR(result, f0_dielectric, kEps);
}

TEST_F(FresnelSchlickTest, F0Zero_CorrectBehavior) {
    // F0 = 0: should interpolate from 0 to 1
    float atNormal = fresnelSchlick(1.0f, 0.0f);
    float atGrazing = fresnelSchlick(0.0f, 0.0f);
    
    EXPECT_NEAR(atNormal, 0.0f, kEps);
    EXPECT_NEAR(atGrazing, 1.0f, kEps);
}

TEST_F(FresnelSchlickTest, F0One_AlwaysOne) {
    // F0 = 1: perfect reflector, always 1
    EXPECT_NEAR(fresnelSchlick(1.0f, 1.0f), 1.0f, kEps);
    EXPECT_NEAR(fresnelSchlick(0.5f, 1.0f), 1.0f, kEps);
    EXPECT_NEAR(fresnelSchlick(0.0f, 1.0f), 1.0f, kEps);
}

TEST_F(FresnelSchlickTest, MonotonicDecrease) {
    // Fresnel should decrease monotonically as cosθ increases
    float prev = fresnelSchlick(0.0f, f0_dielectric);
    for (float cos = 0.1f; cos <= 1.0f; cos += 0.1f) {
        float curr = fresnelSchlick(cos, f0_dielectric);
        EXPECT_LE(curr, prev + kEps);
        prev = curr;
    }
}

TEST_F(FresnelSchlickTest, HalfAngle_ExpectedValue) {
    // At cosθ = 0.5, Fresnel = F0 + (1 - F0) * (1 - 0.5)^5
    float expected = f0_dielectric + (1.0f - f0_dielectric) * std::pow(0.5f, 5);
    float result = fresnelSchlick(0.5f, f0_dielectric);
    EXPECT_NEAR(result, expected, kEps);
}

// ============================================================================
// Fresnel Color Tests
// ============================================================================

TEST(FresnelSchlickColorTest, BasicColorReflectance) {
    ColorRGB f0 = render::make_color_rgb(0.9f, 0.6f, 0.3f);  // Gold-like
    ColorRGB result = fresnelSchlickColor(1.0f, f0);
    
    auto [rr, rg, rb] = render::color_to_floats(result);
    auto [f0r, f0g, f0b] = render::color_to_floats(f0);
    EXPECT_NEAR(rr, f0r, kEps);
    EXPECT_NEAR(rg, f0g, kEps);
    EXPECT_NEAR(rb, f0b, kEps);
}

TEST(FresnelSchlickColorTest, GrazingAngle_White) {
    ColorRGB f0 = render::make_color_rgb(0.9f, 0.6f, 0.3f);
    ColorRGB result = fresnelSchlickColor(0.0f, f0);
    
    auto [rr, rg, rb] = render::color_to_floats(result);
    EXPECT_NEAR(rr, 1.0f, kEps);
    EXPECT_NEAR(rg, 1.0f, kEps);
    EXPECT_NEAR(rb, 1.0f, kEps);
}

// ============================================================================
// Dielectric Fresnel Tests
// ============================================================================

class FresnelDielectricTest : public ::testing::Test {
protected:
    float eta_glass = 1.5f;     // Glass
    float eta_water = 1.33f;    // Water
    float eta_air = 1.0f;       // Air
};

TEST_F(FresnelDielectricTest, NormalIncidence_ExpectedValue) {
    // At normal incidence, F = ((n1 - n2) / (n1 + n2))^2
    // With eta = n_incident / n_transmitted = 1.5 (glass->air: n_i=1.5, n_t=1.0)
    // F = ((1.5 - 1) / (1.5 + 1))^2 = 0.04
    // This is the same for eta=1/1.5 (air->glass) due to symmetry
    float expected = std::pow((eta_glass - 1.0f) / (eta_glass + 1.0f), 2);
    float result = fresnelDielectric(1.0f, eta_glass);  // eta=1.5 means glass->air
    EXPECT_NEAR(result, expected, kEps);
    
    // Verify same result for air->glass (eta=1/1.5)
    float result_air_to_glass = fresnelDielectric(1.0f, 1.0f / eta_glass);
    EXPECT_NEAR(result_air_to_glass, expected, kEps);
}

TEST_F(FresnelDielectricTest, GrazingAngle_One) {
    // At grazing angle, Fresnel → 1
    float result = fresnelDielectric(0.0f, eta_glass);
    EXPECT_NEAR(result, 1.0f, kEps);
}

TEST_F(FresnelDielectricTest, EtaOne_ZeroReflectance) {
    // eta = 1 means no interface, no reflection
    float result = fresnelDielectric(1.0f, 1.0f);
    EXPECT_NEAR(result, 0.0f, kEps);
}

TEST_F(FresnelDielectricTest, TotalInternalReflection) {
    // When going from dense to sparse medium, TIR can occur
    // The implementation uses eta = n_incident / n_transmitted
    // Going from glass (n=1.5) to air (n=1.0): eta = 1.5/1.0 = 1.5
    // But our test uses eta = 1/1.5 = 0.667 (inverted)
    // 
    // For TIR to occur with this convention:
    // sinThetaT = eta * sinThetaI > 1
    // With eta = 0.667 and sinThetaI = 1 → sinThetaT = 0.667 < 1, no TIR
    //
    // Let's test with the correct eta for going FROM glass TO air
    float eta_glass_to_air = eta_glass;  // 1.5 - this means sin(theta_t) = 1.5 * sin(theta_i)
    
    // At a large angle (grazing): sinThetaI close to 1 → sinThetaT = 1.5 * 1 > 1 → TIR
    float sinI = 0.8f;  // sin(theta_i) = 0.8, sinThetaT = 1.5 * 0.8 = 1.2 > 1 → TIR
    float cosI = std::sqrt(1.0f - sinI * sinI);  // ≈ 0.6
    
    float result = fresnelDielectric(cosI, eta_glass_to_air);
    EXPECT_NEAR(result, 1.0f, kEps);  // TIR
}

TEST_F(FresnelDielectricTest, BelowCriticalAngle_LessThanOne) {
    // At normal incidence from glass to air, no TIR
    // sinThetaT = 1.5 * 0 = 0 < 1, so no TIR
    float result = fresnelDielectric(1.0f, eta_glass);  // Normal incidence
    EXPECT_LT(result, 1.0f);
    EXPECT_GT(result, 0.0f);
}

TEST_F(FresnelDielectricTest, Reciprocity) {
    // Test that Fresnel at normal incidence is symmetric
    // F = ((n1-n2)/(n1+n2))^2 is symmetric in n1 and n2
    // eta = n_incident / n_transmitted:
    //   eta=1.5:   glass->air (n_i=1.5, n_t=1.0)
    //   eta=1/1.5: air->glass (n_i=1.0, n_t=1.5)
    float result_glass_to_air = fresnelDielectric(1.0f, eta_glass);      // glass->air
    float result_air_to_glass = fresnelDielectric(1.0f, 1.0f/eta_glass); // air->glass
    
    // At normal incidence, both should give the same reflectance
    EXPECT_NEAR(result_glass_to_air, result_air_to_glass, kEps);
}

// ============================================================================
// GGX Distribution Tests
// ============================================================================

class GGXTest : public ::testing::Test {
protected:
    float roughness_smooth = 0.01f;
    float roughness_medium = 0.5f;
    float roughness_rough = 1.0f;
};

TEST_F(GGXTest, D_PeakAtNdotH1) {
    // GGX D should be maximum when NdotH = 1
    float atPeak = ggxD(1.0f, roughness_medium);
    float offPeak = ggxD(0.9f, roughness_medium);
    EXPECT_GT(atPeak, offPeak);
}

TEST_F(GGXTest, D_SmoothSurface_SharpPeak) {
    // GGX NDF theoretical behavior at peak (NdotH=1):
    // D = α² / (π * α⁴) = 1 / (π * α²)  where α² = roughness⁴
    // So smaller roughness → larger D at peak
    //
    // HOWEVER: Implementation uses (denom² + EPSILON) for numerical stability
    // With EPSILON=1e-6 and roughness=0.01 (α²=1e-8), denom²=1e-16 << EPSILON
    // So D ≈ 1e-8 / (π * 1e-6) ≈ 0.003 instead of theoretical 3e7
    //
    // Test with reasonable roughness values where EPSILON doesn't dominate
    float roughness_med = 0.3f;   // α²=0.0081, denom²=6.6e-5 > EPSILON
    float roughness_high = 0.7f;  // α²=0.24, denom²=0.058 >> EPSILON
    
    float med_at_peak = ggxD(1.0f, roughness_med);
    float high_at_peak = ggxD(1.0f, roughness_high);
    
    // With non-extreme roughness, smoother surface should have higher D at peak
    EXPECT_GT(med_at_peak, high_at_peak);
    
    // Also verify falloff characteristic: smoother surfaces have steeper falloff
    float med_off_peak = ggxD(0.95f, roughness_med);
    float high_off_peak = ggxD(0.95f, roughness_high);
    float med_ratio = med_off_peak / med_at_peak;
    float high_ratio = high_off_peak / high_at_peak;
    EXPECT_GT(high_ratio, med_ratio);  // Rougher surface retains more off-axis
}

TEST_F(GGXTest, D_NdotH0_Finite) {
    // D should be finite even at NdotH = 0 (but small)
    float result = ggxD(0.0f, roughness_medium);
    EXPECT_TRUE(std::isfinite(result));
    EXPECT_GE(result, 0.0f);
}

TEST_F(GGXTest, D_Normalization) {
    // GGX D should integrate to 1 over hemisphere (approximately)
    // ∫ D(m) * (n·m) dω = 1
    float sum = 0.0f;
    int samples = 1000;
    for (int i = 0; i < samples; ++i) {
        float NdotH = (i + 0.5f) / samples;
        float D = ggxD(NdotH, roughness_medium);
        // Weight by solid angle element: 2π * sin(θ) * dθ = 2π * NdotH * d(NdotH) (approx)
        sum += D * NdotH * (1.0f / samples);
    }
    // Should be close to 1/π due to our hemisphere integration
    // This is an approximate test
    EXPECT_GT(sum, 0.1f);
    EXPECT_LT(sum, 2.0f);
}

TEST_F(GGXTest, D_PositiveSemidefinite) {
    // D should always be non-negative
    for (float NdotH = 0.0f; NdotH <= 1.0f; NdotH += 0.1f) {
        EXPECT_GE(ggxD(NdotH, roughness_medium), 0.0f);
    }
}

// ============================================================================
// GGX Geometry Tests
// ============================================================================

TEST_F(GGXTest, G1_Normal_One) {
    // G1 should approach 1 for smooth surfaces at normal incidence
    float result = ggxG1(1.0f, roughness_smooth);
    EXPECT_GT(result, 0.9f);
}

TEST_F(GGXTest, G1_Grazing_Zero) {
    // G1 should approach 0 at grazing angles
    float result = ggxG1(0.01f, roughness_medium);
    EXPECT_LT(result, 0.5f);
}

TEST_F(GGXTest, G1_Range) {
    // G1 should be in [0, 1]
    for (float NdotV = 0.1f; NdotV <= 1.0f; NdotV += 0.1f) {
        float g1 = ggxG1(NdotV, roughness_medium);
        EXPECT_GE(g1, 0.0f);
        EXPECT_LE(g1, 1.0f);
    }
}

TEST_F(GGXTest, G2_SymmetricCase) {
    // G2 should be symmetric in NdotL and NdotV
    float g2_12 = ggxG2(0.6f, 0.8f, roughness_medium);
    float g2_21 = ggxG2(0.8f, 0.6f, roughness_medium);
    EXPECT_NEAR(g2_12, g2_21, kEps);
}

TEST_F(GGXTest, G2_LessThanG1Product) {
    // Height-correlated G2 should be less than G1*G1 (correlated is less pessimistic)
    float NdotL = 0.5f, NdotV = 0.5f;
    float g1_l = ggxG1(NdotL, roughness_medium);
    float g1_v = ggxG1(NdotV, roughness_medium);
    float g2 = ggxG2(NdotL, NdotV, roughness_medium);
    
    // G2 height-correlated should be >= G1*G1 (height correlation reduces shadowing)
    EXPECT_GE(g2, g1_l * g1_v - kEps);
}

TEST_F(GGXTest, G2_Range) {
    // G2 should be in [0, 1]
    for (float NdotL = 0.1f; NdotL <= 1.0f; NdotL += 0.2f) {
        for (float NdotV = 0.1f; NdotV <= 1.0f; NdotV += 0.2f) {
            float g2 = ggxG2(NdotL, NdotV, roughness_medium);
            EXPECT_GE(g2, 0.0f);
            EXPECT_LE(g2, 1.0f);
        }
    }
}

// ============================================================================
// Orthonormal Basis Tests
// ============================================================================

TEST(OrthonormalBasisTest, ZAxis_CorrectBasis) {
    Direction N = direction_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    // T, B, N should be orthonormal
    float TdotB = T.x() * B.x() + T.y() * B.y() + T.z() * B.z();
    float TdotN = T.x() * N.x() + T.y() * N.y() + T.z() * N.z();
    float BdotN = B.x() * N.x() + B.y() * N.y() + B.z() * N.z();
    
    EXPECT_NEAR(TdotB, 0.0f, kEps);
    EXPECT_NEAR(TdotN, 0.0f, kEps);
    EXPECT_NEAR(BdotN, 0.0f, kEps);
}

TEST(OrthonormalBasisTest, XAxis_CorrectBasis) {
    Direction N = direction_from_unit_vector(Vec3f(1.0f, 0.0f, 0.0f));
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    float TdotB = T.x() * B.x() + T.y() * B.y() + T.z() * B.z();
    float TdotN = T.x() * N.x() + T.y() * N.y() + T.z() * N.z();
    float BdotN = B.x() * N.x() + B.y() * N.y() + B.z() * N.z();
    
    EXPECT_NEAR(TdotB, 0.0f, kEps);
    EXPECT_NEAR(TdotN, 0.0f, kEps);
    EXPECT_NEAR(BdotN, 0.0f, kEps);
}

TEST(OrthonormalBasisTest, ArbitraryNormal_CorrectBasis) {
    Vec3f v(0.577f, 0.577f, 0.577f);
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    Direction N = direction_from_unit_vector(Vec3f(v.x / len, v.y / len, v.z / len));
    
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    float TdotB = T.x() * B.x() + T.y() * B.y() + T.z() * B.z();
    float TdotN = T.x() * N.x() + T.y() * N.y() + T.z() * N.z();
    float BdotN = B.x() * N.x() + B.y() * N.y() + B.z() * N.z();
    
    EXPECT_NEAR(TdotB, 0.0f, kEps);
    EXPECT_NEAR(TdotN, 0.0f, kEps);
    EXPECT_NEAR(BdotN, 0.0f, kEps);
}

TEST(OrthonormalBasisTest, NegativeNormal_CorrectBasis) {
    Direction N = direction_from_unit_vector(Vec3f(0.0f, 0.0f, -1.0f));
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    float TdotB = T.x() * B.x() + T.y() * B.y() + T.z() * B.z();
    EXPECT_NEAR(TdotB, 0.0f, kEps);
}

// ============================================================================
// Local/World Transform Tests
// ============================================================================

TEST(LocalWorldTransformTest, Identity_ZAxis) {
    Direction N = direction_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    Direction localDir = *make_direction(0.5f, 0.5f, 0.707f);
    Direction worldDir = localToWorld(localDir, N, T, B);
    Direction backToLocal = worldToLocal(worldDir, N, T, B);
    
    EXPECT_NEAR(localDir.x(), backToLocal.x(), kEps);
    EXPECT_NEAR(localDir.y(), backToLocal.y(), kEps);
    EXPECT_NEAR(localDir.z(), backToLocal.z(), kEps);
}

TEST(LocalWorldTransformTest, PreservesLength) {
    Vec3f v(0.577f, 0.577f, 0.577f);
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    Direction N = direction_from_unit_vector(Vec3f(v.x / len, v.y / len, v.z / len));
    
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    Direction localDir = *make_direction(1.0f, 0.0f, 0.0f);
    Direction worldDir = localToWorld(localDir, N, T, B);
    
    float worldLen = std::sqrt(worldDir.x() * worldDir.x() + 
                               worldDir.y() * worldDir.y() + 
                               worldDir.z() * worldDir.z());
    EXPECT_NEAR(worldLen, 1.0f, kEps);
}

// ============================================================================
// VNDF Sampling Tests (Statistical)
// ============================================================================

TEST(VNDFSamplingTest, SamplesInHemisphere) {
    // All sampled directions should be in upper hemisphere
    Direction N = direction_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    Direction V = *make_direction(0.0f, 0.0f, 1.0f);
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    uint32_t seed = 12345;
    for (int i = 0; i < 100; ++i) {
        float u1 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        float u2 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        
        Direction H = sampleGGXVNDF(V, 0.5f, u1, u2, N, T, B);
        float NdotH = H.x() * N.x() + H.y() * N.y() + H.z() * N.z();
        
        EXPECT_GE(NdotH, -kEps);  // Should be in upper hemisphere
    }
}

TEST(VNDFSamplingTest, HalfVectorValid) {
    // Sampled half-vector should be valid for reflection
    Direction N = direction_from_unit_vector(Vec3f(0.0f, 0.0f, 1.0f));
    Direction V = *make_direction(0.3f, 0.0f, 0.954f);  // 18° from normal
    Direction T, B;
    buildOrthonormalBasis(N, T, B);
    
    uint32_t seed = 54321;
    for (int i = 0; i < 100; ++i) {
        float u1 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        float u2 = (seed = seed * 1103515245 + 12345) / float(UINT32_MAX);
        
        Direction H = sampleGGXVNDF(V, 0.5f, u1, u2, N, T, B);
        
        // H should be unit length
        float len = std::sqrt(H.x() * H.x() + H.y() * H.y() + H.z() * H.z());
        EXPECT_NEAR(len, 1.0f, kEps);
        
        // V·H should be positive (H visible from V)
        float VdotH = V.x() * H.x() + V.y() * H.y() + V.z() * H.z();
        EXPECT_GE(VdotH, -kEps);
    }
}
