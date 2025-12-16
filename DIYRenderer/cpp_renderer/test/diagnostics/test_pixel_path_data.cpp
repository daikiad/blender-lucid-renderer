/**
 * test_pixel_path_data.cpp - TDD tests for per-pixel path data management
 * 
 * Tests Top-N group management and variance-based eviction.
 */

#include <gtest/gtest.h>
#include "diagnostics/pixel_path_data.hpp"

using namespace render;
using namespace render::diagnostics;

// ============================================================================
// PixelPathData Basic Tests
// ============================================================================

TEST(PixelPathDataTest, Construction) {
    PixelPathData<16, 4> data;
    
    EXPECT_EQ(data.group_count(), 0);
    EXPECT_EQ(data.total_samples(), 0);
    EXPECT_EQ(data.overflow_count(), 0);
}

TEST(PixelPathDataTest, AddFirstPath) {
    PixelPathData<16, 4> data;
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Emission);
    trace.add_vertex(1, 1, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 0.5f, 0.25f};
    
    data.record_path(trace);
    
    EXPECT_EQ(data.group_count(), 1);
    EXPECT_EQ(data.total_samples(), 1);
}

TEST(PixelPathDataTest, AddMatchingPath) {
    PixelPathData<16, 4> data;
    
    // First path
    PathTrace trace1;
    trace1.add_vertex(0, 0, BsdfType::Emission);
    trace1.add_vertex(1, 1, BsdfType::Diffuse);
    trace1.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    data.record_path(trace1);
    
    // Same path type, different contribution
    PathTrace trace2;
    trace2.add_vertex(0, 0, BsdfType::Emission);
    trace2.add_vertex(1, 1, BsdfType::Diffuse);
    trace2.contribution = RGB3f{3.0f, 3.0f, 3.0f};
    data.record_path(trace2);
    
    // Should still be 1 group with 2 samples
    EXPECT_EQ(data.group_count(), 1);
    EXPECT_EQ(data.total_samples(), 2);
    
    // Mean should be (1+3)/2 = 2
    const auto* group = data.top_group(0);
    ASSERT_NE(group, nullptr);
    EXPECT_FLOAT_EQ(group->stats.mean.r, 2.0f);
}

TEST(PixelPathDataTest, AddDifferentPaths) {
    PixelPathData<16, 4> data;
    
    // Diffuse path
    PathTrace diffuse;
    diffuse.add_vertex(0, 0, BsdfType::Emission);
    diffuse.add_vertex(1, 1, BsdfType::Diffuse);
    diffuse.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    data.record_path(diffuse);
    
    // Glossy path
    PathTrace glossy;
    glossy.add_vertex(0, 0, BsdfType::Emission);
    glossy.add_vertex(2, 2, BsdfType::Glossy);
    glossy.contribution = RGB3f{0.5f, 0.5f, 0.5f};
    data.record_path(glossy);
    
    EXPECT_EQ(data.group_count(), 2);
    EXPECT_EQ(data.total_samples(), 2);
}

// ============================================================================
// Top-N Eviction Tests
// ============================================================================

TEST(PixelPathDataTest, EvictionWhenFull) {
    // Small capacity for testing eviction
    PixelPathData<4, 2> data;
    
    // Add 4 different path types
    for (int i = 0; i < 4; ++i) {
        PathTrace trace;
        trace.add_vertex(i, i, BsdfType::Diffuse);
        trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
        data.record_path(trace);
    }
    EXPECT_EQ(data.group_count(), 4);
    
    // Add 5th path - should trigger eviction
    PathTrace new_path;
    new_path.add_vertex(99, 99, BsdfType::Glossy);
    new_path.contribution = RGB3f{10.0f, 10.0f, 10.0f};  // High contribution
    data.record_path(new_path);
    
    // Still max 4 groups
    EXPECT_EQ(data.group_count(), 4);
    EXPECT_GT(data.overflow_count(), 0);
}

TEST(PixelPathDataTest, KeepHighVarianceGroups) {
    PixelPathData<2, 2> data;
    
    // Low variance path (consistent)
    for (int i = 0; i < 10; ++i) {
        PathTrace low_var;
        low_var.add_vertex(0, 0, BsdfType::Diffuse);
        low_var.contribution = RGB3f{1.0f, 1.0f, 1.0f};
        data.record_path(low_var);
    }
    
    // High variance path (inconsistent)
    PathTrace high_var1;
    high_var1.add_vertex(1, 1, BsdfType::Glossy);
    high_var1.contribution = RGB3f{0.1f, 0.1f, 0.1f};
    data.record_path(high_var1);
    
    PathTrace high_var2;
    high_var2.add_vertex(1, 1, BsdfType::Glossy);
    high_var2.contribution = RGB3f{10.0f, 10.0f, 10.0f};
    data.record_path(high_var2);
    
    // Now add a third type that will trigger eviction
    PathTrace third;
    third.add_vertex(2, 2, BsdfType::Mirror);
    third.contribution = RGB3f{5.0f, 5.0f, 5.0f};
    data.record_path(third);
    
    // The high-variance group should be kept over the low-variance one
    bool found_high_var = false;
    for (size_t i = 0; i < data.group_count(); ++i) {
        const auto* g = data.top_group(i);
        if (g && g->vertices[0].bsdf_type == BsdfType::Glossy) {
            found_high_var = true;
            break;
        }
    }
    EXPECT_TRUE(found_high_var);
}

// ============================================================================
// Coarse Stats Tests
// ============================================================================

TEST(PixelPathDataTest, CoarseStatsAccumulation) {
    PixelPathData<16, 4> data;
    
    // Direct diffuse path (coarse type 0)
    PathTrace direct;
    direct.add_vertex(0, 0, BsdfType::Emission);
    direct.add_vertex(1, 1, BsdfType::Diffuse);
    direct.contribution = RGB3f{1.0f, 2.0f, 3.0f};
    data.record_path(direct);
    
    // Check coarse stats updated
    const auto& coarse = data.coarse_stats(direct.coarse_type());
    EXPECT_EQ(coarse.count, 1);
    EXPECT_FLOAT_EQ(coarse.mean.r, 1.0f);
}

// ============================================================================
// Outlier Detection Tests  
// ============================================================================

TEST(PixelPathDataTest, OutlierDetection) {
    PixelPathData<16, 4> data;
    
    // Build baseline with consistent values
    for (int i = 0; i < 20; ++i) {
        PathTrace normal;
        normal.add_vertex(0, 0, BsdfType::Emission);
        normal.add_vertex(1, 1, BsdfType::Diffuse);
        normal.contribution = RGB3f{1.0f, 1.0f, 1.0f};
        data.record_path(normal);
    }
    
    // Add outlier (way above mean)
    PathTrace outlier;
    outlier.add_vertex(0, 0, BsdfType::Emission);
    outlier.add_vertex(1, 1, BsdfType::Diffuse);
    outlier.contribution = RGB3f{100.0f, 100.0f, 100.0f};
    data.record_path(outlier);
    
    // Should be marked as outlier
    EXPECT_GT(data.outlier_count(), 0);
}

// ============================================================================
// Sorting Tests
// ============================================================================

TEST(PixelPathDataTest, SortByVariance) {
    PixelPathData<16, 4> data;
    
    // Path 1: Low variance
    for (int i = 0; i < 5; ++i) {
        PathTrace trace;
        trace.add_vertex(0, 0, BsdfType::Diffuse);
        trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
        data.record_path(trace);
    }
    
    // Path 2: High variance
    PathTrace high1;
    high1.add_vertex(1, 1, BsdfType::Glossy);
    high1.contribution = RGB3f{0.1f, 0.1f, 0.1f};
    data.record_path(high1);
    
    PathTrace high2;
    high2.add_vertex(1, 1, BsdfType::Glossy);
    high2.contribution = RGB3f{10.0f, 10.0f, 10.0f};
    data.record_path(high2);
    
    // Sort by variance descending
    data.sort_by_variance();
    
    // First group should have higher variance
    const auto* first = data.top_group(0);
    const auto* second = data.top_group(1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_GE(first->total_variance(), second->total_variance());
}

TEST(PixelPathDataTest, SortByMeanContribution) {
    PixelPathData<16, 4> data;
    
    // Low contribution path
    PathTrace low;
    low.add_vertex(0, 0, BsdfType::Diffuse);
    low.contribution = RGB3f{0.1f, 0.1f, 0.1f};
    data.record_path(low);
    
    // High contribution path
    PathTrace high;
    high.add_vertex(1, 1, BsdfType::Glossy);
    high.contribution = RGB3f{10.0f, 10.0f, 10.0f};
    data.record_path(high);
    
    data.sort_by_mean();
    
    const auto* first = data.top_group(0);
    const auto* second = data.top_group(1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_GE(first->mean_contribution(), second->mean_contribution());
}

// ============================================================================
// Total Statistics Tests
// ============================================================================

TEST(PixelPathDataTest, TotalMeanVariance) {
    PixelPathData<16, 4> data;
    
    // Add multiple paths
    for (int i = 0; i < 10; ++i) {
        PathTrace trace;
        trace.add_vertex(0, 0, BsdfType::Diffuse);
        trace.contribution = RGB3f{static_cast<float>(i), 
                                   static_cast<float>(i), 
                                   static_cast<float>(i)};
        data.record_path(trace);
    }
    
    // Mean should be (0+1+...+9)/10 = 4.5
    RGB3f mean = data.total_mean();
    EXPECT_NEAR(mean.r, 4.5f, 0.01f);
    
    // Total variance should be positive
    float var = data.total_variance();
    EXPECT_GT(var, 0.0f);
}
