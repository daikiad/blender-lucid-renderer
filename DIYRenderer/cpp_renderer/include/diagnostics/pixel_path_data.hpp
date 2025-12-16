/**
 * pixel_path_data.hpp - Per-pixel path statistics storage
 * 
 * Manages Top-N path groups by variance with efficient eviction.
 * Template parameters allow compile-time memory layout control.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "diagnostics/path_stats.hpp"

namespace render::diagnostics {

// ============================================================================
// OutlierSample - Stored outlier path contribution
// ============================================================================

struct OutlierSample {
    uint64_t path_hash;
    RGB3f contribution;
    
    OutlierSample() : path_hash(0), contribution{} {}
    OutlierSample(uint64_t h, const RGB3f& c) : path_hash(h), contribution(c) {}
};

// ============================================================================
// PixelPathData - Per-pixel path statistics with Top-N management
// ============================================================================

template <size_t MaxGroups = 16, size_t MaxOutliers = 4>
class PixelPathData {
public:
    PixelPathData()
        : groups_{}
        , coarse_stats_{}
        , outliers_{}
        , num_groups_(0)
        , num_outliers_(0)
        , total_samples_(0)
        , overflow_count_(0)
        , total_stats_{}
    {}
    
    /**
     * Record a path trace
     */
    void record_path(const PathTrace& trace) {
        ++total_samples_;
        
        // Update total stats
        total_stats_.add_sample(trace.contribution);
        
        // Update coarse stats
        uint8_t coarse = trace.coarse_type();
        coarse_stats_[coarse].add_sample(trace.contribution);
        
        // Check for outlier before adding to group
        if (is_outlier(trace)) {
            add_outlier(trace);
        }
        
        // Try to find matching group
        uint64_t hash = trace.hash();
        for (size_t i = 0; i < num_groups_; ++i) {
            if (groups_[i].path_hash == hash) {
                groups_[i].add_sample(trace);
                return;
            }
        }
        
        // No match - need to add new group
        if (num_groups_ < MaxGroups) {
            // Space available
            groups_[num_groups_].init_from(trace);
            ++num_groups_;
        } else {
            // Full - need to evict lowest variance group
            evict_and_add(trace);
        }
    }
    
    /**
     * Number of active groups
     */
    [[nodiscard]] size_t group_count() const { return num_groups_; }
    
    /**
     * Total samples recorded
     */
    [[nodiscard]] size_t total_samples() const { return total_samples_; }
    
    /**
     * Count of samples lost to overflow
     */
    [[nodiscard]] size_t overflow_count() const { return overflow_count_; }
    
    /**
     * Number of outliers detected
     */
    [[nodiscard]] size_t outlier_count() const { return num_outliers_; }
    
    /**
     * Access group by index (sorted order after sort_by_*)
     */
    [[nodiscard]] const PathGroup* top_group(size_t index) const {
        if (index >= num_groups_) return nullptr;
        return &groups_[index];
    }
    
    /**
     * Access coarse stats by type
     */
    [[nodiscard]] const PathStatistics& coarse_stats(uint8_t type) const {
        return coarse_stats_[type & 0x07];
    }
    
    /**
     * Sort groups by variance (descending) for noise analysis
     */
    void sort_by_variance() {
        std::sort(groups_.begin(), groups_.begin() + num_groups_,
            [](const PathGroup& a, const PathGroup& b) {
                return a.total_variance() > b.total_variance();
            });
    }
    
    /**
     * Sort groups by mean contribution (descending) for contribution analysis
     */
    void sort_by_mean() {
        std::sort(groups_.begin(), groups_.begin() + num_groups_,
            [](const PathGroup& a, const PathGroup& b) {
                return a.mean_contribution() > b.mean_contribution();
            });
    }
    
    /**
     * Total mean across all samples
     */
    [[nodiscard]] RGB3f total_mean() const {
        return total_stats_.mean;
    }
    
    /**
     * Total variance across all samples
     */
    [[nodiscard]] float total_variance() const {
        return total_stats_.total_variance();
    }
    
private:
    std::array<PathGroup, MaxGroups> groups_;
    std::array<PathStatistics, NUM_COARSE_TYPES> coarse_stats_;
    std::array<OutlierSample, MaxOutliers> outliers_;
    
    size_t num_groups_;
    size_t num_outliers_;
    size_t total_samples_;
    size_t overflow_count_;
    
    PathStatistics total_stats_;  // Overall pixel statistics
    
    /**
     * Check if a sample is an outlier (> 3 sigma from mean)
     */
    [[nodiscard]] bool is_outlier(const PathTrace& trace) const {
        if (total_samples_ < 10) return false;  // Need baseline
        
        const float threshold = 3.0f;  // 3 sigma
        
        // Check each channel
        float stddev_r = std::sqrt(total_stats_.variance.r + 1e-10f);
        float stddev_g = std::sqrt(total_stats_.variance.g + 1e-10f);
        float stddev_b = std::sqrt(total_stats_.variance.b + 1e-10f);
        
        float dev_r = std::abs(trace.contribution.r - total_stats_.mean.r);
        float dev_g = std::abs(trace.contribution.g - total_stats_.mean.g);
        float dev_b = std::abs(trace.contribution.b - total_stats_.mean.b);
        
        return (dev_r > threshold * stddev_r) ||
               (dev_g > threshold * stddev_g) ||
               (dev_b > threshold * stddev_b);
    }
    
    /**
     * Add outlier sample (circular buffer)
     */
    void add_outlier(const PathTrace& trace) {
        size_t idx = num_outliers_ % MaxOutliers;
        outliers_[idx] = OutlierSample{trace.hash(), trace.contribution};
        ++num_outliers_;
    }
    
    /**
     * Evict lowest-variance group and add new one
     */
    void evict_and_add(const PathTrace& trace) {
        // Find group with minimum variance (least interesting for noise analysis)
        size_t min_idx = 0;
        float min_var = groups_[0].total_variance();
        
        for (size_t i = 1; i < num_groups_; ++i) {
            float var = groups_[i].total_variance();
            // Prefer evicting groups with fewer samples
            float adjusted_var = var * std::log2(static_cast<float>(groups_[i].stats.count + 1));
            float min_adjusted = min_var * std::log2(static_cast<float>(groups_[min_idx].stats.count + 1));
            
            if (adjusted_var < min_adjusted) {
                min_idx = i;
                min_var = var;
            }
        }
        
        // Check if new path has higher "interest" than minimum
        // For now, always replace (can be refined)
        ++overflow_count_;
        groups_[min_idx].init_from(trace);
    }
};

}  // namespace render::diagnostics
