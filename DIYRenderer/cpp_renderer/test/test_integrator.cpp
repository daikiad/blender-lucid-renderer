/**
 * test_integrator.cpp - Path Tracer and MIS Tests
 *
 * Tests for:
 * - MIS weight calculation correctness
 * - MIS weight properties (symmetry, sum to 1)
 * - Basic integrator sanity checks
 *
 * These tests help prevent bugs like incorrect argument ordering
 * in MIS weight calculations.
 */

#include "units/render_units.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace render;
using namespace mp_units::si::unit_symbols;

constexpr float kEps = 1e-5f;

// ============================================================================
// MIS Weight Tests
// ============================================================================

class MISWeightTest : public ::testing::Test {
protected:
    // Typical PDF values in path tracing
    PdfW pdf_low = 0.1f * per_sr;
    PdfW pdf_medium = 0.5f * per_sr;
    PdfW pdf_high = 2.0f * per_sr;
};

TEST_F(MISWeightTest, WeightForFirstArgument) {
    // mis_power_heuristic(pf, pg) should return weight for pf
    // When pf >> pg, weight should be close to 1
    Dimensionless weight = mis_power_heuristic(pdf_high, pdf_low);
    EXPECT_GT(weight.numerical_value_in(one), 0.9f);
    
    // When pf << pg, weight should be close to 0
    Dimensionless weight2 = mis_power_heuristic(pdf_low, pdf_high);
    EXPECT_LT(weight2.numerical_value_in(one), 0.1f);
}

TEST_F(MISWeightTest, WeightsSumToOne) {
    // w_f + w_g should equal 1 (fundamental MIS property)
    // This test would have caught the argument order bug
    Dimensionless wf = mis_power_heuristic(pdf_medium, pdf_high);
    Dimensionless wg = mis_power_heuristic(pdf_high, pdf_medium);
    EXPECT_NEAR((wf + wg).numerical_value_in(one), 1.0f, kEps);
    
    // Test with various combinations
    Dimensionless wf2 = mis_power_heuristic(pdf_low, pdf_high);
    Dimensionless wg2 = mis_power_heuristic(pdf_high, pdf_low);
    EXPECT_NEAR((wf2 + wg2).numerical_value_in(one), 1.0f, kEps);
    
    Dimensionless wf3 = mis_power_heuristic(pdf_low, pdf_medium);
    Dimensionless wg3 = mis_power_heuristic(pdf_medium, pdf_low);
    EXPECT_NEAR((wf3 + wg3).numerical_value_in(one), 1.0f, kEps);
}

TEST_F(MISWeightTest, EqualPdfsGiveHalfWeight) {
    // When both PDFs are equal, each strategy gets 0.5 weight
    Dimensionless weight = mis_power_heuristic(pdf_medium, pdf_medium);
    EXPECT_NEAR(weight.numerical_value_in(one), 0.5f, kEps);
}

TEST_F(MISWeightTest, WeightInValidRange) {
    // MIS weight should always be in [0, 1]
    std::vector<PdfW> pdfs = {0.01f * per_sr, 0.1f * per_sr, 0.5f * per_sr, 
                               1.0f * per_sr, 2.0f * per_sr, 10.0f * per_sr};
    
    for (auto pf : pdfs) {
        for (auto pg : pdfs) {
            Dimensionless w = mis_power_heuristic(pf, pg);
            float w_val = w.numerical_value_in(one);
            EXPECT_GE(w_val, 0.0f) << "pf=" << pf.numerical_value_in(per_sr) 
                               << ", pg=" << pg.numerical_value_in(per_sr);
            EXPECT_LE(w_val, 1.0f) << "pf=" << pf.numerical_value_in(per_sr) 
                               << ", pg=" << pg.numerical_value_in(per_sr);
        }
    }
}

TEST_F(MISWeightTest, PowerHeuristicFormula) {
    // Verify the formula: w_f = pf² / (pf² + pg²)
    float pf_val = pdf_medium.numerical_value_in(per_sr);
    float pg_val = pdf_high.numerical_value_in(per_sr);
    
    float expected = (pf_val * pf_val) / (pf_val * pf_val + pg_val * pg_val);
    Dimensionless actual = mis_power_heuristic(pdf_medium, pdf_high);
    
    EXPECT_NEAR(actual.numerical_value_in(one), expected, kEps);
}

TEST_F(MISWeightTest, ZeroPdfHandling) {
    // When one PDF is zero or very small, handle gracefully
    PdfW zero_pdf = 0.0f * per_sr;
    PdfW tiny_pdf = 1e-10f * per_sr;
    
    // Should not crash or produce NaN
    Dimensionless w1 = mis_power_heuristic(pdf_medium, zero_pdf);
    Dimensionless w2 = mis_power_heuristic(zero_pdf, pdf_medium);
    Dimensionless w3 = mis_power_heuristic(pdf_medium, tiny_pdf);
    
    EXPECT_TRUE(std::isfinite(w1.numerical_value_in(one)));
    EXPECT_TRUE(std::isfinite(w2.numerical_value_in(one)));
    EXPECT_TRUE(std::isfinite(w3.numerical_value_in(one)));
    
    // With zero alternative, weight should be ~1
    EXPECT_NEAR(w1.numerical_value_in(one), 1.0f, 0.01f);
}

// ============================================================================
// BSDF Sample Weight Tests
// ============================================================================

class BSDFSampleWeightTest : public ::testing::Test {
protected:
    BSDFRGB bsdf_white = make_bsdf_rgb(1.0f, 1.0f, 1.0f);
    BSDFRGB bsdf_gray = make_bsdf_rgb(0.5f, 0.5f, 0.5f);
    PdfW pdf_cosine = 0.5f * per_sr;  // Typical cosine hemisphere PDF
};

TEST_F(BSDFSampleWeightTest, BasicWeight) {
    // weight = (f / pdf) * cos_theta
    float cos_theta = 0.8f;
    ThroughputRGB weight = bsdf_sample_weight(bsdf_gray, cos_theta, pdf_cosine);
    
    // f = 0.5/sr, pdf = 0.5/sr, cos = 0.8
    // weight = (0.5/0.5) * 0.8 = 0.8
    auto [wr, wg, wb] = render::color_to_floats(weight);
    EXPECT_NEAR(wr, 0.8f, kEps);
    EXPECT_NEAR(wg, 0.8f, kEps);
    EXPECT_NEAR(wb, 0.8f, kEps);
}

TEST_F(BSDFSampleWeightTest, ZeroPdfReturnsZero) {
    PdfW zero_pdf = 0.0f * per_sr;
    ThroughputRGB weight = bsdf_sample_weight(bsdf_white, 1.0f, zero_pdf);
    
    auto [wr, wg, wb] = render::color_to_floats(weight);
    EXPECT_NEAR(wr, 0.0f, kEps);
    EXPECT_NEAR(wg, 0.0f, kEps);
    EXPECT_NEAR(wb, 0.0f, kEps);
}

TEST_F(BSDFSampleWeightTest, MinPdfThreshold) {
    // PDF below MIN_PDF should return zero
    PdfW tiny_pdf = MIN_PDF * 0.5f;
    ThroughputRGB weight = bsdf_sample_weight(bsdf_white, 1.0f, tiny_pdf);
    
    auto [wr, wg, wb] = render::color_to_floats(weight);
    EXPECT_NEAR(wr, 0.0f, kEps);
}

// ============================================================================
// PDF Conversion Tests
// ============================================================================

class PDFConversionTest : public ::testing::Test {
protected:
    Length dist_1m = 1.0f * m;
    Length dist_2m = 2.0f * m;
    Area area_1m2 = 1.0f * m2;
};

TEST_F(PDFConversionTest, AreaToPdfW_Basic) {
    // pdf_ω = d² / (A × cos θ)
    // At 1m distance, 1m² area, cos=1: pdf = 1/sr
    float cos_theta = 1.0f;
    PdfW pdf = compute_pdf_w_from_area(dist_1m, area_1m2, cos_theta);
    EXPECT_NEAR(pdf.numerical_value_in(per_sr), 1.0f, kEps);
}

TEST_F(PDFConversionTest, AreaToPdfW_Distance) {
    // Double distance -> 4x PDF (inverse square law)
    float cos_theta = 1.0f;
    PdfW pdf_1m = compute_pdf_w_from_area(dist_1m, area_1m2, cos_theta);
    PdfW pdf_2m = compute_pdf_w_from_area(dist_2m, area_1m2, cos_theta);
    
    EXPECT_NEAR(pdf_2m.numerical_value_in(per_sr), 
                pdf_1m.numerical_value_in(per_sr) * 4.0f, kEps);
}

TEST_F(PDFConversionTest, AreaToPdfW_Angle) {
    // Grazing angle -> higher PDF (1/cos)
    float cos_normal = 1.0f;
    float cos_grazing = 0.5f;
    
    PdfW pdf_normal = compute_pdf_w_from_area(dist_1m, area_1m2, cos_normal);
    PdfW pdf_grazing = compute_pdf_w_from_area(dist_1m, area_1m2, cos_grazing);
    
    EXPECT_NEAR(pdf_grazing.numerical_value_in(per_sr),
                pdf_normal.numerical_value_in(per_sr) * 2.0f, kEps);
}

TEST_F(PDFConversionTest, AreaToPdfW_ZeroHandling) {
    // Zero area or cos should return zero PDF (not infinity)
    Area zero_area = 0.0f * m2;
    PdfW pdf = compute_pdf_w_from_area(dist_1m, zero_area, 1.0f);
    EXPECT_NEAR(pdf.numerical_value_in(per_sr), 0.0f, kEps);
    
    PdfW pdf2 = compute_pdf_w_from_area(dist_1m, area_1m2, 0.0f);
    EXPECT_NEAR(pdf2.numerical_value_in(per_sr), 0.0f, kEps);
}

// ============================================================================
// Radiance/Color Conversion Tests  
// ============================================================================

TEST(RadianceConversionTest, RoundTrip) {
    AttenuationRGB original = render::make_attenuation_rgb(0.5f, 0.7f, 0.3f);
    RadianceRGB radiance = to_radiance(original);
    AttenuationRGB back = render::apply_camera_sensitivity(radiance, render::kDefaultCameraSensitivity);
    
    auto [or_, og, ob] = render::color_to_floats(original);
    auto [br, bg, bb] = render::color_to_floats(back);
    EXPECT_NEAR(or_, br, kEps);
    EXPECT_NEAR(og, bg, kEps);
    EXPECT_NEAR(ob, bb, kEps);
}

TEST(RadianceConversionTest, Multiplication) {
    ThroughputRGB throughput = render::make_throughput_rgb(0.5f, 0.5f, 0.5f);
    RadianceRGB emission = to_radiance(render::make_attenuation_rgb(2.0f, 2.0f, 2.0f));
    
    RadianceRGB result = throughput * emission;
    AttenuationRGB result_color = render::apply_camera_sensitivity(result, render::kDefaultCameraSensitivity);
    
    auto [rr, rg, rb] = render::color_to_floats(result_color);
    EXPECT_NEAR(rr, 1.0f, kEps);
    EXPECT_NEAR(rg, 1.0f, kEps);
    EXPECT_NEAR(rb, 1.0f, kEps);
}

// ============================================================================
// Integration Test: MIS Weight Consistency
// ============================================================================

TEST(MISConsistencyTest, LightAndBSDFWeightsSum) {
    // Simulate typical MIS scenario:
    // - Light sampling PDF
    // - BSDF sampling PDF
    // - Their weights should sum to 1
    
    // Typical area light: small solid angle
    PdfW light_pdf = 0.2f * per_sr;
    // Typical diffuse BSDF: cosine weighted
    PdfW bsdf_pdf = 0.8f * per_sr;
    
    Dimensionless light_weight = mis_power_heuristic(light_pdf, bsdf_pdf);
    Dimensionless bsdf_weight = mis_power_heuristic(bsdf_pdf, light_pdf);
    
    EXPECT_NEAR((light_weight + bsdf_weight).numerical_value_in(one), 1.0f, kEps);
    
    // BSDF has higher PDF, so should have higher weight
    EXPECT_GT(bsdf_weight.numerical_value_in(one), light_weight.numerical_value_in(one));
}

TEST(MISConsistencyTest, UsagePatternCorrectness) {
    // This test documents the correct usage pattern
    PdfW sampled_pdf = 0.5f * per_sr;      // PDF of the sampling strategy we used
    PdfW alternative_pdf = 0.3f * per_sr;  // PDF of the alternative strategy
    
    // CORRECT: First argument is the strategy we sampled with
    Dimensionless weight = mis_power_heuristic(sampled_pdf, alternative_pdf);
    
    // The weight should reflect that sampled_pdf > alternative_pdf
    // So we should get a weight > 0.5
    EXPECT_GT(weight.numerical_value_in(one), 0.5f);
    
    // Verify formula: sampled² / (sampled² + alt²)
    float s = sampled_pdf.numerical_value_in(per_sr);
    float a = alternative_pdf.numerical_value_in(per_sr);
    float expected = (s*s) / (s*s + a*a);
    EXPECT_NEAR(weight.numerical_value_in(one), expected, kEps);
}
