/**
 * test_path_stats_config.cpp - TDD tests for path recording configuration
 * 
 * Tests for memory estimation and preset configurations.
 */

#include <gtest/gtest.h>
#include "diagnostics/path_stats_config.hpp"

using namespace render::diagnostics;

// ============================================================================
// PathRecordingConfig Tests
// ============================================================================

TEST(PathRecordingConfigTest, DefaultValues) {
    PathRecordingConfig config;
    
    EXPECT_EQ(config.subsample_factor, 2);
    EXPECT_EQ(config.max_groups_per_pixel, 16);
    EXPECT_EQ(config.max_outliers_per_pixel, 4);
    EXPECT_EQ(config.max_depth, 16);
    EXPECT_GT(config.variance_threshold, 0.0f);
    EXPECT_GT(config.outlier_threshold, 1.0f);
}

TEST(PathRecordingConfigTest, Presets) {
    auto minimal = PathRecordingConfig::minimal();
    EXPECT_EQ(minimal.subsample_factor, 4);
    EXPECT_EQ(minimal.max_groups_per_pixel, 8);
    
    auto standard = PathRecordingConfig::standard();
    EXPECT_EQ(standard.subsample_factor, 2);
    EXPECT_EQ(standard.max_groups_per_pixel, 16);
    
    auto detailed = PathRecordingConfig::detailed();
    EXPECT_EQ(detailed.subsample_factor, 1);
    EXPECT_EQ(detailed.max_groups_per_pixel, 32);
}

// ============================================================================
// Memory Estimation Tests
// ============================================================================

TEST(PathRecordingConfigTest, MemoryEstimate1080p) {
    // 1920x1080 = 2,073,600 pixels
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    
    // Minimal preset: subsample=4 → 518,400 sampled pixels
    auto minimal = PathRecordingConfig::minimal();
    size_t mem_minimal = minimal.estimate_memory_bytes(width, height);
    
    // Should be around 200-500MB
    EXPECT_GT(mem_minimal, 100 * 1024 * 1024);   // > 100 MB
    EXPECT_LT(mem_minimal, 1024UL * 1024 * 1024);   // < 1 GB
    
    // Standard preset: subsample=2 → 518,400 sampled pixels
    auto standard = PathRecordingConfig::standard();
    size_t mem_standard = standard.estimate_memory_bytes(width, height);
    
    // Should be around 1-4GB
    EXPECT_GT(mem_standard, 500 * 1024 * 1024);  // > 500 MB
    EXPECT_LT(mem_standard, 5UL * 1024 * 1024 * 1024);  // < 5 GB
    
    // Detailed preset: subsample=1 → 2,073,600 sampled pixels
    auto detailed = PathRecordingConfig::detailed();
    size_t mem_detailed = detailed.estimate_memory_bytes(width, height);
    
    // Should be around 10-40GB (depends on PathGroup size with geometry)
    EXPECT_GT(mem_detailed, 3UL * 1024 * 1024 * 1024);  // > 3 GB
    EXPECT_LT(mem_detailed, 50UL * 1024 * 1024 * 1024); // < 50 GB
}

TEST(PathRecordingConfigTest, MemoryScalesWithResolution) {
    auto config = PathRecordingConfig::standard();
    
    size_t mem_720p = config.estimate_memory_bytes(1280, 720);
    size_t mem_1080p = config.estimate_memory_bytes(1920, 1080);
    size_t mem_4k = config.estimate_memory_bytes(3840, 2160);
    
    // 1080p should be ~2.25x of 720p
    double ratio_720_1080 = static_cast<double>(mem_1080p) / static_cast<double>(mem_720p);
    EXPECT_NEAR(ratio_720_1080, 2.25, 0.5);
    
    // 4K should be ~4x of 1080p
    double ratio_1080_4k = static_cast<double>(mem_4k) / static_cast<double>(mem_1080p);
    EXPECT_NEAR(ratio_1080_4k, 4.0, 0.5);
}

TEST(PathRecordingConfigTest, SampledPixelCount) {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    
    PathRecordingConfig config;
    config.subsample_factor = 1;
    EXPECT_EQ(config.sampled_pixel_count(width, height), width * height);
    
    config.subsample_factor = 2;
    // With factor 2, we sample every other pixel in each dimension
    EXPECT_EQ(config.sampled_pixel_count(width, height), (width/2) * (height/2));
    
    config.subsample_factor = 4;
    EXPECT_EQ(config.sampled_pixel_count(width, height), (width/4) * (height/4));
}

TEST(PathRecordingConfigTest, MemoryFormatted) {
    auto config = PathRecordingConfig::standard();
    
    std::string formatted = config.estimate_memory_formatted(1920, 1080);
    
    // Should contain a number and unit
    EXPECT_TRUE(formatted.find("GB") != std::string::npos || 
                formatted.find("MB") != std::string::npos);
}

// ============================================================================
// PixelPathData Size Tests
// ============================================================================

TEST(PathRecordingConfigTest, PerPixelDataSize) {
    auto minimal = PathRecordingConfig::minimal();
    auto standard = PathRecordingConfig::standard();
    auto detailed = PathRecordingConfig::detailed();
    
    // Per-pixel size should increase with more groups
    size_t size_minimal = minimal.per_pixel_data_size();
    size_t size_standard = standard.per_pixel_data_size();
    size_t size_detailed = detailed.per_pixel_data_size();
    
    EXPECT_LT(size_minimal, size_standard);
    EXPECT_LT(size_standard, size_detailed);
    
    // Reasonable bounds
    EXPECT_GT(size_minimal, 100);     // At least 100 bytes per pixel
    EXPECT_LT(size_detailed, 32768);  // At most 32KB per pixel (with geometry)
}
