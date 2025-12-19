/**
 * test_path_stats.cpp - TDD tests for path statistics and grouping
 * 
 * Test-first approach for variance tracking and path aggregation.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "diagnostics/path_stats.hpp"

using namespace render;
using namespace render::diagnostics;

// ============================================================================
// PathStatistics Tests
// ============================================================================

TEST(PathStatisticsTest, Size) {
    // Should be reasonably small for per-group storage
    // Target: ~48 bytes (3 RGB * 4 floats * 4 bytes)
    EXPECT_LE(sizeof(PathStatistics), 64);
}

TEST(PathStatisticsTest, Initial) {
    PathStatistics stats;
    
    EXPECT_EQ(stats.count, 0);
    EXPECT_EQ(stats.mean.r, 0.0f);
    EXPECT_EQ(stats.mean.g, 0.0f);
    EXPECT_EQ(stats.mean.b, 0.0f);
    EXPECT_EQ(stats.variance.r, 0.0f);
    EXPECT_EQ(stats.variance.g, 0.0f);
    EXPECT_EQ(stats.variance.b, 0.0f);
}

TEST(PathStatisticsTest, SingleSample) {
    PathStatistics stats;
    stats.add_sample(RGB3f{1.0f, 2.0f, 3.0f});
    
    EXPECT_EQ(stats.count, 1);
    EXPECT_FLOAT_EQ(stats.mean.r, 1.0f);
    EXPECT_FLOAT_EQ(stats.mean.g, 2.0f);
    EXPECT_FLOAT_EQ(stats.mean.b, 3.0f);
    // Variance undefined for n=1, should be 0
    EXPECT_FLOAT_EQ(stats.variance.r, 0.0f);
}

TEST(PathStatisticsTest, MultipleSamples) {
    PathStatistics stats;
    
    // Add samples: 1, 2, 3 (mean = 2, var = 1)
    stats.add_sample(RGB3f{1.0f, 1.0f, 1.0f});
    stats.add_sample(RGB3f{2.0f, 2.0f, 2.0f});
    stats.add_sample(RGB3f{3.0f, 3.0f, 3.0f});
    
    EXPECT_EQ(stats.count, 3);
    EXPECT_FLOAT_EQ(stats.mean.r, 2.0f);
    EXPECT_FLOAT_EQ(stats.mean.g, 2.0f);
    EXPECT_FLOAT_EQ(stats.mean.b, 2.0f);
    
    // Sample variance = sum((xi - mean)^2) / (n-1)
    // = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = (1 + 0 + 1) / 2 = 1
    EXPECT_FLOAT_EQ(stats.variance.r, 1.0f);
}

TEST(PathStatisticsTest, WelfordStability) {
    // Test numerical stability with large offset values
    PathStatistics stats;
    
    const float offset = 1e6f;
    stats.add_sample(RGB3f{offset + 1.0f, 0.0f, 0.0f});
    stats.add_sample(RGB3f{offset + 2.0f, 0.0f, 0.0f});
    stats.add_sample(RGB3f{offset + 3.0f, 0.0f, 0.0f});
    
    EXPECT_NEAR(stats.mean.r, offset + 2.0f, 0.01f);
    EXPECT_NEAR(stats.variance.r, 1.0f, 0.01f);  // Variance should still be 1
}

TEST(PathStatisticsTest, TotalVariance) {
    PathStatistics stats;
    stats.add_sample(RGB3f{1.0f, 2.0f, 3.0f});
    stats.add_sample(RGB3f{3.0f, 4.0f, 5.0f});
    
    // Mean = (2, 3, 4), Var = (2, 2, 2)
    // Total variance = 2 + 2 + 2 = 6
    EXPECT_FLOAT_EQ(stats.total_variance(), 6.0f);
}

TEST(PathStatisticsTest, Luminance) {
    PathStatistics stats;
    stats.add_sample(RGB3f{0.5f, 0.7f, 0.3f});
    
    // Y = 0.2126*R + 0.7152*G + 0.0722*B
    // Y = 0.2126*0.5 + 0.7152*0.7 + 0.0722*0.3
    float expected = 0.2126f * 0.5f + 0.7152f * 0.7f + 0.0722f * 0.3f;
    EXPECT_FLOAT_EQ(stats.mean_luminance(), expected);
}

// ============================================================================
// PathGroup Tests  
// ============================================================================

TEST(PathGroupTest, Size) {
    // Target: ~568 bytes with geometry arrays for path visualization
    // hash(8) + stats(48) + depth(4) + vertices(8*16=128) + positions(12*16=192) + normals(12*16=192) = 572
    EXPECT_LE(sizeof(PathGroup), 600);
}

TEST(PathGroupTest, Initialize) {
    PathTrace trace;
    trace.add_vertex(0, 1, BsdfType::Emission);
    trace.add_vertex(1, 2, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 0.5f, 0.25f};
    
    PathGroup group;
    group.init_from(trace);
    
    EXPECT_EQ(group.path_hash, trace.hash());
    EXPECT_EQ(group.depth, 2);
    EXPECT_EQ(group.vertices[0], trace.vertices[0]);
    EXPECT_EQ(group.vertices[1], trace.vertices[1]);
    EXPECT_EQ(group.stats.count, 1);
    EXPECT_FLOAT_EQ(group.stats.mean.r, 1.0f);
}

TEST(PathGroupTest, AddMatchingSample) {
    PathTrace trace1;
    trace1.add_vertex(0, 1, BsdfType::Emission);
    trace1.add_vertex(1, 2, BsdfType::Diffuse);
    trace1.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    
    PathGroup group;
    group.init_from(trace1);
    
    // Second sample with same path structure
    PathTrace trace2;
    trace2.add_vertex(0, 1, BsdfType::Emission);
    trace2.add_vertex(1, 2, BsdfType::Diffuse);
    trace2.contribution = RGB3f{3.0f, 3.0f, 3.0f};
    
    EXPECT_TRUE(group.matches(trace2));
    group.add_sample(trace2);
    
    EXPECT_EQ(group.stats.count, 2);
    EXPECT_FLOAT_EQ(group.stats.mean.r, 2.0f);
}

TEST(PathGroupTest, MatchesDifferentPath) {
    PathTrace trace1;
    trace1.add_vertex(0, 1, BsdfType::Emission);
    trace1.add_vertex(1, 2, BsdfType::Diffuse);
    
    PathGroup group;
    group.init_from(trace1);
    
    // Different path structure
    PathTrace trace2;
    trace2.add_vertex(0, 1, BsdfType::Emission);
    trace2.add_vertex(1, 2, BsdfType::Glossy);  // Different BSDF
    
    EXPECT_FALSE(group.matches(trace2));
}

TEST(PathGroupTest, PathSignature) {
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Emission);
    trace.add_vertex(1, 1, BsdfType::Diffuse);
    trace.add_vertex(2, 2, BsdfType::Glossy);
    
    PathGroup group;
    group.init_from(trace);
    
    // Should produce "L D G" style signature
    std::string sig = group.signature();
    EXPECT_EQ(sig, "LDG");
}

// ============================================================================
// BasicPathStats Tests (per-coarse-type aggregated stats)
// ============================================================================

TEST(BasicPathStatsTest, CoarseTypeNames) {
    // 8 combinations of (direct/indirect, delta/no-delta, nee/bsdf)
    EXPECT_STREQ(coarse_type_name(0b000), "Direct-NoDelta-BSDF");
    EXPECT_STREQ(coarse_type_name(0b001), "Direct-NoDelta-NEE");
    EXPECT_STREQ(coarse_type_name(0b010), "Direct-Delta-BSDF");
    EXPECT_STREQ(coarse_type_name(0b100), "Indirect-NoDelta-BSDF");
    EXPECT_STREQ(coarse_type_name(0b111), "Indirect-Delta-NEE");
}
