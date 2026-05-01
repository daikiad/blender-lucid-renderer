/**
 * path_stats_config.hpp - Path recording configuration
 * 
 * Configuration for memory budget and recording precision.
 * Includes memory estimation for UI display.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>

#include "diagnostics/path_stats.hpp"

namespace render::diagnostics {

// ============================================================================
// PathRecordingConfig - Configuration for path statistics recording
// ============================================================================

struct PathRecordingConfig {
    // Spatial subsampling (1 = every pixel, 2 = every 2nd pixel in each dim)
    uint8_t subsample_factor;
    
    // Maximum groups (unique path types) per pixel
    uint8_t max_groups_per_pixel;
    
    // Maximum outlier samples to store per pixel
    uint8_t max_outliers_per_pixel;
    
    // Maximum path depth to track
    uint8_t max_depth;
    
    // Minimum variance to track a group (prune low-variance groups)
    float variance_threshold;
    
    // Threshold for outlier detection (stddev multiplier)
    float outlier_threshold;
    
    // Default constructor (Standard preset)
    PathRecordingConfig()
        : subsample_factor(2)
        , max_groups_per_pixel(16)
        , max_outliers_per_pixel(4)
        , max_depth(16)
        , variance_threshold(0.001f)
        , outlier_threshold(3.0f)
    {}
    
    // ========================================================================
    // Presets
    // ========================================================================
    
    /**
     * Minimal preset - ~200MB at 1080p
     * Good for quick overview
     */
    static PathRecordingConfig minimal() {
        PathRecordingConfig config;
        config.subsample_factor = 4;
        config.max_groups_per_pixel = 8;
        config.max_outliers_per_pixel = 2;
        config.max_depth = 8;
        config.variance_threshold = 0.01f;
        config.outlier_threshold = 5.0f;
        return config;
    }
    
    /**
     * Standard preset - ~1.5GB at 1080p
     * Balanced memory vs detail
     */
    static PathRecordingConfig standard() {
        PathRecordingConfig config;
        config.subsample_factor = 2;
        config.max_groups_per_pixel = 16;
        config.max_outliers_per_pixel = 4;
        config.max_depth = 16;
        config.variance_threshold = 0.001f;
        config.outlier_threshold = 3.0f;
        return config;
    }
    
    /**
     * Detailed preset - ~6GB at 1080p
     * Maximum detail for deep analysis
     */
    static PathRecordingConfig detailed() {
        PathRecordingConfig config;
        config.subsample_factor = 1;
        config.max_groups_per_pixel = 32;
        config.max_outliers_per_pixel = 8;
        config.max_depth = 16;
        config.variance_threshold = 0.0001f;
        config.outlier_threshold = 2.0f;
        return config;
    }
    
    // ========================================================================
    // Memory Estimation
    // ========================================================================
    
    /**
     * Number of pixels that will be sampled
     */
    [[nodiscard]] size_t sampled_pixel_count(size_t width, size_t height) const {
        size_t sampled_w = width / subsample_factor;
        size_t sampled_h = height / subsample_factor;
        return sampled_w * sampled_h;
    }
    
    /**
     * Estimated bytes per sampled pixel
     * 
     * Per pixel:
     * - Basic stats for 8 coarse types: 8 * sizeof(PathStatistics)
     * - Top-N groups: max_groups * sizeof(PathGroup)
     * - Outlier storage: max_outliers * outlier_entry_size
     * - Overhead: ~32 bytes
     */
    [[nodiscard]] size_t per_pixel_data_size() const {
        // PathStatistics size
        constexpr size_t stats_size = sizeof(PathStatistics);  // ~56 bytes
        
        // PathGroup size (approximate with max depth)
        constexpr size_t base_group_size = sizeof(PathGroup);  // ~200 bytes
        
        // Per-pixel breakdown:
        // - 8 coarse type stats
        size_t coarse_stats = NUM_COARSE_TYPES * stats_size;
        
        // - Top-N path groups
        size_t groups = max_groups_per_pixel * base_group_size;
        
        // - Outlier samples (hash + contribution)
        constexpr size_t outlier_entry_size = sizeof(uint64_t) + sizeof(RGB3f);
        size_t outliers = max_outliers_per_pixel * outlier_entry_size;
        
        // - Overhead (counters, flags, etc.)
        constexpr size_t overhead = 32;
        
        return coarse_stats + groups + outliers + overhead;
    }
    
    /**
     * Estimate total memory usage in bytes
     */
    [[nodiscard]] size_t estimate_memory_bytes(size_t width, size_t height) const {
        size_t pixels = sampled_pixel_count(width, height);
        size_t per_pixel = per_pixel_data_size();
        
        // Add global overhead (~1MB for hash tables, etc.)
        constexpr size_t global_overhead = 1024 * 1024;
        
        return pixels * per_pixel + global_overhead;
    }
    
    /**
     * Human-readable memory estimate
     */
    [[nodiscard]] std::string estimate_memory_formatted(size_t width, size_t height) const {
        size_t bytes = estimate_memory_bytes(width, height);
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        
        if (bytes >= 1024UL * 1024 * 1024) {
            oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
        } else if (bytes >= 1024 * 1024) {
            oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
        } else {
            oss << (static_cast<double>(bytes) / 1024.0) << " KB";
        }
        
        return oss.str();
    }
};

}  // namespace render::diagnostics
