/**
 * diagnostic_film.hpp - Full image diagnostic data management
 * 
 * Manages per-pixel path statistics for the entire render.
 * Supports export to binary format for Python analysis.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "diagnostics/path_stats_config.hpp"
#include "diagnostics/pixel_path_data.hpp"

namespace render::diagnostics {

// ============================================================================
// GlobalDiagnosticStats - Summary statistics for entire film
// ============================================================================

struct GlobalDiagnosticStats {
    size_t total_samples;
    size_t active_pixels;
    size_t total_groups;
    size_t total_overflow;
    RGB3f mean;
    float total_variance;
};

// ============================================================================
// GroupReference - Reference to a group with its pixel location
// ============================================================================

struct GroupReference {
    size_t pixel_x;
    size_t pixel_y;
    const PathGroup* group;
    
    // For sorting
    [[nodiscard]] float variance() const {
        return group ? group->total_variance() : 0.0f;
    }
    
    [[nodiscard]] float mean() const {
        return group ? group->mean_contribution() : 0.0f;
    }
};

// ============================================================================
// DiagnosticFilm - Main film class for path variance analysis
// ============================================================================

class DiagnosticFilm {
public:
    // Use standard config pixel data (16 groups, 4 outliers)
    using PixelData = PixelPathData<16, 4>;
    
    /**
     * Construct film with given dimensions and config
     */
    DiagnosticFilm(size_t width, size_t height, const PathRecordingConfig& config)
        : width_(width)
        , height_(height)
        , config_(config)
        , sampled_width_(width / config.subsample_factor)
        , sampled_height_(height / config.subsample_factor)
        , pixels_(sampled_width_ * sampled_height_)
        , total_recorded_(0)
    {}
    
    // ========================================================================
    // Dimensions
    // ========================================================================
    
    [[nodiscard]] size_t width() const { return width_; }
    [[nodiscard]] size_t height() const { return height_; }
    [[nodiscard]] size_t sampled_width() const { return sampled_width_; }
    [[nodiscard]] size_t sampled_height() const { return sampled_height_; }
    
    // ========================================================================
    // Recording
    // ========================================================================
    
    /**
     * Record a path at the given pixel coordinates
     * Thread-safe for different pixels (same pixel needs external sync)
     */
    void record_path(size_t x, size_t y, const PathTrace& trace) {
        // Map to sampled coordinates
        size_t sx = x / config_.subsample_factor;
        size_t sy = y / config_.subsample_factor;
        
        if (sx >= sampled_width_ || sy >= sampled_height_) {
            return;  // Out of bounds
        }
        
        size_t idx = sy * sampled_width_ + sx;
        pixels_[idx].record_path(trace);
        ++total_recorded_;
    }
    
    /**
     * Total paths recorded
     */
    [[nodiscard]] size_t total_paths_recorded() const {
        return total_recorded_.load();
    }
    
    /**
     * Access pixel data (read-only)
     */
    [[nodiscard]] const PixelData* pixel_data(size_t x, size_t y) const {
        size_t sx = x / config_.subsample_factor;
        size_t sy = y / config_.subsample_factor;
        
        if (sx >= sampled_width_ || sy >= sampled_height_) {
            return nullptr;
        }
        
        return &pixels_[sy * sampled_width_ + sx];
    }
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    /**
     * Compute global statistics
     */
    [[nodiscard]] GlobalDiagnosticStats global_stats() const {
        GlobalDiagnosticStats stats{};
        PathStatistics total_stats;
        
        for (size_t i = 0; i < pixels_.size(); ++i) {
            const auto& px = pixels_[i];
            if (px.total_samples() > 0) {
                ++stats.active_pixels;
                stats.total_samples += px.total_samples();
                stats.total_groups += px.group_count();
                stats.total_overflow += px.overflow_count();
                
                // Accumulate mean (weighted by samples)
                // Note: This is approximate; proper computation needs running average
            }
        }
        
        return stats;
    }
    
    /**
     * Get top N groups by variance across all pixels
     */
    [[nodiscard]] std::vector<GroupReference> top_groups_by_variance(size_t n) const {
        std::vector<GroupReference> all_groups;
        all_groups.reserve(pixels_.size() * 4);  // Estimate
        
        for (size_t sy = 0; sy < sampled_height_; ++sy) {
            for (size_t sx = 0; sx < sampled_width_; ++sx) {
                const auto& px = pixels_[sy * sampled_width_ + sx];
                for (size_t i = 0; i < px.group_count(); ++i) {
                    const auto* g = px.top_group(i);
                    if (g && g->stats.count >= 2) {  // Need at least 2 samples for variance
                        all_groups.push_back({
                            sx * config_.subsample_factor,
                            sy * config_.subsample_factor,
                            g
                        });
                    }
                }
            }
        }
        
        // Partial sort for top N
        size_t limit = std::min(n, all_groups.size());
        std::partial_sort(all_groups.begin(), all_groups.begin() + limit, all_groups.end(),
            [](const GroupReference& a, const GroupReference& b) {
                return a.variance() > b.variance();
            });
        
        all_groups.resize(limit);
        return all_groups;
    }
    
    /**
     * Get top N groups by mean contribution
     */
    [[nodiscard]] std::vector<GroupReference> top_groups_by_mean(size_t n) const {
        std::vector<GroupReference> all_groups;
        all_groups.reserve(pixels_.size() * 4);
        
        for (size_t sy = 0; sy < sampled_height_; ++sy) {
            for (size_t sx = 0; sx < sampled_width_; ++sx) {
                const auto& px = pixels_[sy * sampled_width_ + sx];
                for (size_t i = 0; i < px.group_count(); ++i) {
                    const auto* g = px.top_group(i);
                    if (g && g->stats.count >= 1) {
                        all_groups.push_back({
                            sx * config_.subsample_factor,
                            sy * config_.subsample_factor,
                            g
                        });
                    }
                }
            }
        }
        
        size_t limit = std::min(n, all_groups.size());
        std::partial_sort(all_groups.begin(), all_groups.begin() + limit, all_groups.end(),
            [](const GroupReference& a, const GroupReference& b) {
                return a.mean() > b.mean();
            });
        
        all_groups.resize(limit);
        return all_groups;
    }
    
    // ========================================================================
    // Export
    // ========================================================================
    
    /**
     * Generate metadata JSON
     */
    [[nodiscard]] std::string metadata_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"width\": " << width_ << ",\n";
        oss << "  \"height\": " << height_ << ",\n";
        oss << "  \"sampled_width\": " << sampled_width_ << ",\n";
        oss << "  \"sampled_height\": " << sampled_height_ << ",\n";
        oss << "  \"subsample_factor\": " << static_cast<int>(config_.subsample_factor) << ",\n";
        oss << "  \"max_groups\": " << static_cast<int>(config_.max_groups_per_pixel) << ",\n";
        oss << "  \"max_outliers\": " << static_cast<int>(config_.max_outliers_per_pixel) << ",\n";
        oss << "  \"total_samples\": " << total_recorded_.load() << "\n";
        oss << "}\n";
        return oss.str();
    }
    
    /**
     * Export to binary format
     * 
     * Format:
     *   Magic: "PVAR" (4 bytes)
     *   Version: uint32 (4 bytes)
     *   Width: uint32 (4 bytes)
     *   Height: uint32 (4 bytes)
     *   Sampled Width: uint32 (4 bytes)
     *   Sampled Height: uint32 (4 bytes)
     *   Config: subsample(1) + max_groups(1) + max_outliers(1) + padding(1)
     *   Per-pixel data...
     */
    void export_binary(std::ostream& out) const {
        // Magic
        out.write("PVAR", 4);
        
        // Version
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Dimensions
        uint32_t w = static_cast<uint32_t>(width_);
        uint32_t h = static_cast<uint32_t>(height_);
        uint32_t sw = static_cast<uint32_t>(sampled_width_);
        uint32_t sh = static_cast<uint32_t>(sampled_height_);
        out.write(reinterpret_cast<const char*>(&w), sizeof(w));
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
        out.write(reinterpret_cast<const char*>(&sw), sizeof(sw));
        out.write(reinterpret_cast<const char*>(&sh), sizeof(sh));
        
        // Config
        out.write(reinterpret_cast<const char*>(&config_.subsample_factor), 1);
        out.write(reinterpret_cast<const char*>(&config_.max_groups_per_pixel), 1);
        out.write(reinterpret_cast<const char*>(&config_.max_outliers_per_pixel), 1);
        char padding = 0;
        out.write(&padding, 1);
        
        // Per-pixel summary (simplified for now)
        for (const auto& px : pixels_) {
            // Write group count and total samples
            uint16_t gc = static_cast<uint16_t>(px.group_count());
            uint32_t ts = static_cast<uint32_t>(px.total_samples());
            out.write(reinterpret_cast<const char*>(&gc), sizeof(gc));
            out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
            
            // Write coarse stats mean
            RGB3f mean = px.total_mean();
            out.write(reinterpret_cast<const char*>(&mean), sizeof(mean));
            
            // Write total variance
            float var = px.total_variance();
            out.write(reinterpret_cast<const char*>(&var), sizeof(var));
        }
    }
    
    // ========================================================================
    // Clear
    // ========================================================================
    
    /**
     * Clear all recorded data
     */
    void clear() {
        for (auto& px : pixels_) {
            px = PixelData{};
        }
        total_recorded_ = 0;
    }
    
private:
    size_t width_;
    size_t height_;
    PathRecordingConfig config_;
    size_t sampled_width_;
    size_t sampled_height_;
    
    std::vector<PixelData> pixels_;
    std::atomic<size_t> total_recorded_;
};

}  // namespace render::diagnostics
