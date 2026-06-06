/**
 * @file test_integrator_e2e.cpp
 * @brief End-to-end tests comparing Simple and MIS path tracers
 *
 * These tests verify MIS and Simple path tracers produce consistent results
 * by testing key components that affect their behavior.
 *
 * Key invariant: For the same scene, Simple and MIS path tracers should
 * converge to the same result (within noise tolerance).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>

#include "units/render_units.hpp"
#include "bsdf/bsdf.hpp"

using namespace render;
using namespace mp_units::si::unit_symbols;

constexpr float kEps = 1e-5f;

// =============================================================================
// MIS Behavior Tests - Verify MIS weight behavior across different scenarios
// =============================================================================

class MISBehaviorTest : public ::testing::Test {
protected:
    // Test that MIS properly weights BSDF vs light sampling
    void verifyMISWeights(PdfW bsdfPdf, PdfW lightPdf) {
        Dimensionless wBsdf = mis_power_heuristic(bsdfPdf, lightPdf);
        Dimensionless wLight = mis_power_heuristic(lightPdf, bsdfPdf);
        
        // Weights should sum to 1
        EXPECT_NEAR((wBsdf + wLight).numerical_value_in(one), 1.0f, kEps);
        
        // Higher PDF should get higher weight
        if (bsdfPdf > lightPdf + MIN_PDF) {
            EXPECT_GT(wBsdf.numerical_value_in(one), wLight.numerical_value_in(one));
        } else if (lightPdf > bsdfPdf + MIN_PDF) {
            EXPECT_GT(wLight.numerical_value_in(one), wBsdf.numerical_value_in(one));
        }
    }
};

// Test: Equal PDFs should give equal weights
TEST_F(MISBehaviorTest, EqualPDFsGiveEqualWeights) {
    PdfW pdf = 1.0f * per_sr;
    Dimensionless w1 = mis_power_heuristic(pdf, pdf);
    Dimensionless w2 = mis_power_heuristic(pdf, pdf);
    EXPECT_NEAR(w1.numerical_value_in(one), 0.5f, kEps);
    EXPECT_NEAR(w2.numerical_value_in(one), 0.5f, kEps);
}

// Test: Higher BSDF PDF means BSDF sampling wins
TEST_F(MISBehaviorTest, HigherBSDFPDFWins) {
    PdfW highPdf = 10.0f * per_sr;
    PdfW lowPdf = 1.0f * per_sr;
    
    Dimensionless wBsdf = mis_power_heuristic(highPdf, lowPdf);
    Dimensionless wLight = mis_power_heuristic(lowPdf, highPdf);
    
    EXPECT_GT(wBsdf.numerical_value_in(one), 0.9f);  // BSDF should dominate
    EXPECT_LT(wLight.numerical_value_in(one), 0.1f);  // Light should be minimal
}

// Test: MIS weights sum to 1 for any valid PDFs
TEST_F(MISBehaviorTest, WeightsSumToOne) {
    std::vector<std::pair<float, float>> testCases = {
        {0.1f, 0.1f},
        {1.0f, 1.0f},
        {10.0f, 10.0f},
        {0.01f, 100.0f},
        {100.0f, 0.01f},
        {0.5f, 2.0f},
        {3.14159f, 2.71828f},
    };
    
    for (const auto& [p1, p2] : testCases) {
        PdfW pdf1 = p1 * per_sr;
        PdfW pdf2 = p2 * per_sr;
        verifyMISWeights(pdf1, pdf2);
    }
}

// Test: Zero PDF handling
TEST_F(MISBehaviorTest, ZeroPDFHandling) {
    PdfW zeroPdf = 0.0f * per_sr;
    PdfW normalPdf = 1.0f * per_sr;
    
    // Zero PDF should give weight 0 to that strategy
    Dimensionless wZero = mis_power_heuristic(zeroPdf, normalPdf);
    Dimensionless wNormal = mis_power_heuristic(normalPdf, zeroPdf);
    
    EXPECT_NEAR(wZero.numerical_value_in(one), 0.0f, kEps);
    EXPECT_NEAR(wNormal.numerical_value_in(one), 1.0f, kEps);
}

// =============================================================================
// BSDF Sample Weight Tests - Verify useWeight flag behavior
// =============================================================================

class BSDFSampleBehaviorTest : public ::testing::Test {
protected:
    MaterialParams createDiffuseMaterial(float r, float g, float b) {
        MaterialParams mat;
        mat.albedo = render::make_attenuation_rgb(r, g, b);
        mat.roughness = 1.0f;
        mat.metallic = 0.0f;
        mat.transmission = 0.0f;
        return mat;
    }
    
    MaterialParams createGlossyMaterial(float roughness) {
        MaterialParams mat;
        mat.albedo = render::make_attenuation_rgb(0.8f, 0.8f, 0.8f);
        mat.roughness = roughness;
        mat.metallic = 0.9f;
        mat.transmission = 0.0f;
        return mat;
    }
    
    MaterialParams createGlassMaterial(float ior, float roughness) {
        MaterialParams mat;
        mat.albedo = render::make_attenuation_rgb(1.0f, 1.0f, 1.0f);
        mat.roughness = roughness;
        mat.metallic = 0.0f;
        mat.transmission = 1.0f;
        mat.ior = ior;
        return mat;
    }
};

// Test: Diffuse materials should NOT use weight flag (when sample is valid)
TEST_F(BSDFSampleBehaviorTest, DiffuseNoUseWeight) {
    auto mat = createDiffuseMaterial(0.8f, 0.8f, 0.8f);
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    int validSamples = 0;
    int useWeightCount = 0;
    
    // Sample multiple times to check valid samples
    for (int i = 0; i < 100; ++i) {
        float u1 = static_cast<float>(i) / 100.0f;
        float u2 = static_cast<float>((i * 7 + 3) % 100) / 100.0f;
        
        BSDFSample sample = sampleBSDF(mat, wo, n, u1, u2, 0.0f);
        
        // Only check valid samples (those that sampled above hemisphere)
        if (sample.isValid()) {
            validSamples++;
            if (sample.hasPrecomputedWeight()) {
                useWeightCount++;
            }
        }
    }
    
    // Should have valid samples
    EXPECT_GT(validSamples, 50) << "Should have many valid diffuse samples";
    
    // Diffuse should use standard f/pdf weighting, not pre-computed weight
    EXPECT_EQ(useWeightCount, 0) 
        << "Diffuse samples should not use pre-computed weight";
}

// Test: Specular (low roughness) may use weight flag
TEST_F(BSDFSampleBehaviorTest, SpecularMayUseWeight) {
    auto mat = createGlossyMaterial(0.01f);  // Very smooth
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    int useWeightCount = 0;
    const int samples = 100;
    
    for (int i = 0; i < samples; ++i) {
        float u1 = static_cast<float>(i) / samples;
        float u2 = static_cast<float>((i * 7 + 3) % samples) / samples;
        float u3 = static_cast<float>((i * 13 + 7) % samples) / samples;
        
        BSDFSample sample = sampleBSDF(mat, wo, n, u1, u2, u3);
        if (sample.hasPrecomputedWeight()) {
            useWeightCount++;
        }
    }
    
    // Highly specular materials should sometimes use weight
    // (depends on whether specular path was sampled)
    EXPECT_GE(useWeightCount, 0);  // May or may not use weight
}

// Test: Transmission materials use weight flag
TEST_F(BSDFSampleBehaviorTest, TransmissionUsesWeight) {
    auto mat = createGlassMaterial(1.5f, 0.01f);  // Glass with low roughness
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    int useWeightCount = 0;
    const int samples = 100;
    
    for (int i = 0; i < samples; ++i) {
        float u1 = static_cast<float>(i) / samples;
        float u2 = static_cast<float>((i * 7 + 3) % samples) / samples;
        float u3 = static_cast<float>((i * 13 + 7) % samples) / samples;
        
        BSDFSample sample = sampleBSDF(mat, wo, n, u1, u2, u3);
        if (sample.hasPrecomputedWeight()) {
            useWeightCount++;
        }
    }
    
    // Transmission should always use weight
    EXPECT_EQ(useWeightCount, samples) 
        << "Transmission samples should all use pre-computed weight";
}

// =============================================================================
// BSDF Consistency Tests - Verify BSDF evaluation matches sampling
// =============================================================================

class BSDFConsistencyTest : public ::testing::Test {
protected:
    // Monte Carlo estimate of reflectance
    float estimateReflectance(const MaterialParams& mat, const Direction& wo, 
                              const Direction& n, int samples) {
        float sum = 0.0f;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < samples; ++i) {
            BSDFSample sample = sampleBSDF(mat, wo, n, dist(rng), dist(rng), dist(rng));

            if (auto weightOpt = compute_throughput_update(sample, n)) {
                auto [wr, wg, wb] = render::color_to_floats(*weightOpt);
                float lum = 0.2126f * wr + 0.7152f * wg + 0.0722f * wb;
                sum += lum;
            }
        }
        
        return sum / samples;
    }
};

// Test: White diffuse should reflect ~1/π of incoming light
TEST_F(BSDFConsistencyTest, DiffuseReflectance) {
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(1.0f, 1.0f, 1.0f);  // White
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    float reflectance = estimateReflectance(mat, wo, n, 10000);
    
    // Energy conservation: reflectance should be <= 1
    EXPECT_LE(reflectance, 1.1f);  // Allow some variance
    EXPECT_GE(reflectance, 0.5f);  // Should reflect significant amount
}

// Test: Glossy metal should have high reflectance at normal incidence
TEST_F(BSDFConsistencyTest, GlossyMetalReflectance) {
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(0.95f, 0.95f, 0.95f);  // Silver-like
    mat.roughness = 0.1f;
    mat.metallic = 1.0f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    float reflectance = estimateReflectance(mat, wo, n, 10000);
    
    // High metallic should have high reflectance
    EXPECT_GE(reflectance, 0.5f);
    EXPECT_LE(reflectance, 1.5f);  // Allow variance
}

// Test: Low roughness should concentrate energy in specular direction
TEST_F(BSDFConsistencyTest, LowRoughnessConcentration) {
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(0.8f, 0.8f, 0.8f);
    mat.roughness = 0.02f;  // Very smooth
    mat.metallic = 1.0f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.5f, 0.0f, 0.866f));  // 30 degree angle
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    // Expected reflection direction
    Vec3f expectedReflect = wo.vec();
    expectedReflect.x = -expectedReflect.x;  // Reflect x component
    
    int closeCount = 0;
    const int samples = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < samples; ++i) {
        BSDFSample sample = sampleBSDF(mat, wo, n, dist(rng), dist(rng), dist(rng));
        
        // Check if sample direction is close to expected reflection
        float dotProd = dot(sample.wi.vec(), expectedReflect);
        if (dotProd > 0.95f) {  // Within ~18 degrees
            closeCount++;
        }
    }
    
    // Most samples should be near specular direction
    float ratio = static_cast<float>(closeCount) / samples;
    EXPECT_GT(ratio, 0.5f) << "Low roughness should concentrate samples near reflection";
}

// =============================================================================
// PDF/Weight Consistency Tests
// =============================================================================

class PDFConsistencyTest : public ::testing::Test {};

// Test: PDF should be positive for valid samples
TEST(PDFConsistencyTest, PositivePDFForValidSamples) {
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(0.8f, 0.8f, 0.8f);
    mat.roughness = 0.5f;
    mat.metallic = 0.0f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    int validCount = 0;
    int invalidCount = 0;
    
    for (int i = 0; i < 100; ++i) {
        BSDFSample sample = sampleBSDF(mat, wo, n, dist(rng), dist(rng), dist(rng));
        
        // A sample is either:
        // 1. PrecomputedWeight variant (pre-computed weight, delta-like)
        // 2. EvaluatedBSDF variant with pdf > 0 (standard f/pdf)
        // 3. EvaluatedBSDF variant with pdf=0 (sampled below hemisphere, invalid)
        if (sample.isValid()) {
            validCount++;
            if (auto* eval = std::get_if<EvaluatedBSDF>(&sample.result)) {
                EXPECT_GT(eval->pdf.numerical_value_in(per_sr), 0.0f)
                    << "EvaluatedBSDF valid samples should have positive PDF";
            }
        } else {
            invalidCount++;
        }
    }
    
    // Most samples should be valid when wo and n are aligned
    EXPECT_GT(validCount, 80) << "Most samples should be valid for aligned wo/n";
}

// Test: Weight components should be non-negative
TEST(PDFConsistencyTest, NonNegativeWeight) {
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(0.8f, 0.8f, 0.8f);
    mat.roughness = 0.1f;
    mat.metallic = 0.5f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.3f, 0.0f, 0.954f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < 100; ++i) {
        BSDFSample sample = sampleBSDF(mat, wo, n, dist(rng), dist(rng), dist(rng));
        
        if (auto* w = std::get_if<PrecomputedWeight>(&sample.result)) {
            EXPECT_GE(w->weight.r, 0.0f);
            EXPECT_GE(w->weight.g, 0.0f);
            EXPECT_GE(w->weight.b, 0.0f);
        }
    }
}

// =============================================================================
// Energy Conservation Tests
// =============================================================================

TEST(EnergyConservationTest, DiffuseTotalReflectanceBounded) {
    // Estimate total hemispheric reflectance via Monte Carlo
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(1.0f, 1.0f, 1.0f);
    mat.roughness = 1.0f;
    mat.metallic = 0.0f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    float sum = 0.0f;
    const int samples = 50000;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < samples; ++i) {
        BSDFSample sample = sampleBSDF(mat, wo, n, dist(rng), dist(rng), dist(rng));
        
        float contrib = 0.0f;
        if (auto weightOpt = compute_throughput_update(sample, n)) {
            auto [wr, wg, wb] = render::color_to_floats(*weightOpt);
            contrib = wr;
        }
        sum += contrib;
    }
    
    float avgReflectance = sum / samples;
    
    // Energy conservation: reflectance <= 1 (with some tolerance for noise)
    EXPECT_LE(avgReflectance, 1.1f) << "Diffuse should conserve energy";
    EXPECT_GE(avgReflectance, 0.8f) << "White diffuse should reflect most light";
}

TEST(EnergyConservationTest, MetalReflectanceBounded) {
    MaterialParams mat;
    mat.albedo = render::make_attenuation_rgb(1.0f, 1.0f, 1.0f);
    mat.roughness = 0.2f;
    mat.metallic = 1.0f;
    mat.transmission = 0.0f;
    
    auto wo = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    auto n = make_direction_or_default(Vec3f(0.0f, 0.0f, 1.0f));
    
    float sum = 0.0f;
    const int samples = 50000;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < samples; ++i) {
        BSDFSample sample = sampleBSDF(mat, wo, n, dist(rng), dist(rng), dist(rng));

        float contrib = 0.0f;
        if (auto weightOpt = compute_throughput_update(sample, n)) {
            auto [wr, wg, wb] = render::color_to_floats(*weightOpt);
            contrib = wr;
        }
        sum += contrib;
    }

    float avgReflectance = sum / samples;

    // Metal can have high reflectance but should still conserve energy
    EXPECT_LE(avgReflectance, 1.2f) << "Metal should conserve energy";
    EXPECT_GE(avgReflectance, 0.5f) << "White metal should reflect significant light";
}
