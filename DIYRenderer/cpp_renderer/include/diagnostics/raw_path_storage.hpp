/**
 * raw_path_storage.hpp - Raw path data storage for full path collection
 * 
 * Separates collection phase from analysis phase:
 * 
 * Collection Phase:
 *   PathTracer → RawPathStorage (stores ALL paths per pixel)
 * 
 * Analysis Phase:
 *   RawPathStorage → grouping, statistics, visualization
 * 
 * Memory estimate (optimized):
 *   - RawPath fixed: ~32 bytes
 *   - Per vertex: 16 bytes (position only, no normal)
 *   - Average depth 5: ~112 bytes/path
 *   - FHD 32SPP: ~6.5GB
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "diagnostics/path_types.hpp"
#include "units/render_units.hpp"

namespace render::diagnostics {

// ============================================================================
// RawPath - Single path with minimal data (variable length)
// ============================================================================

/**
 * Compact storage for a single path's data.
 * Positions are stored inline (variable length based on depth).
 */
struct RawPath {
    // Fixed header (32 bytes)
    uint64_t         signature_hash;   // Path type hash for grouping
    RGB3f            contribution;     // Final contribution to pixel
    uint8_t          depth;            // Number of vertices (1-16)
    uint8_t          coarse_type;      // Coarse classification bits
    SamplingStrategy strategy;         // How the path was sampled
    LightSourceType  light_type;       // Light source type
    uint8_t          _pad[4];          // Padding for alignment
    
    // Variable length data follows:
    // - PathVertex vertices[depth]    (8 bytes each)
    // - Vec3f positions[depth]        (12 bytes each)
    // Total per vertex: 20 bytes
    
    std::vector<PathVertex> vertices;
    std::vector<Vec3f>      positions;
    
    RawPath()
        : signature_hash(0)
        , contribution{}
        , depth(0)
        , coarse_type(0)
        , strategy(SamplingStrategy::BSDF)
        , light_type(LightSourceType::Unknown)
        , _pad{}
        , vertices{}
        , positions{}
    {}
    
    /**
     * Initialize from a PathTrace
     */
    void init_from(const PathTrace& trace) {
        signature_hash = trace.hash();
        contribution = trace.contribution;
        depth = static_cast<uint8_t>(trace.depth);
        coarse_type = trace.coarse_type();
        strategy = trace.strategy;
        light_type = trace.light_type;
        
        vertices.resize(depth);
        positions.resize(depth);
        
        for (size_t i = 0; i < depth; ++i) {
            vertices[i] = trace.vertices[i];
            positions[i] = trace.positions[i];
        }
    }
    
    /**
     * Estimated memory usage in bytes
     */
    [[nodiscard]] size_t memory_usage() const {
        return sizeof(RawPath) 
             + vertices.capacity() * sizeof(PathVertex)
             + positions.capacity() * sizeof(Vec3f);
    }
    
    /**
     * Generate path signature string (e.g., "CD.DDL")
     */
    [[nodiscard]] std::string signature_string() const {
        std::string sig = "C";  // Camera
        for (size_t i = 0; i < depth; ++i) {
            sig += bsdf_type_short(vertices[i].bsdf_type);
        }
        return sig;
    }
};

// ============================================================================
// PixelRawPaths - All paths for a single pixel
// ============================================================================

class PixelRawPaths {
public:
    PixelRawPaths() = default;
    
    /**
     * Reserve space for expected number of samples
     */
    void reserve(size_t sample_count) {
        paths_.reserve(sample_count);
    }
    
    /**
     * Add a path from PathTrace
     */
    void add_path(const PathTrace& trace) {
        paths_.emplace_back();
        paths_.back().init_from(trace);
    }
    
    /**
     * Number of paths stored
     */
    [[nodiscard]] size_t count() const { return paths_.size(); }
    
    /**
     * Access path by index
     */
    [[nodiscard]] const RawPath& path(size_t index) const { 
        return paths_[index]; 
    }
    
    /**
     * Access all paths
     */
    [[nodiscard]] const std::vector<RawPath>& paths() const { 
        return paths_; 
    }
    
    /**
     * Clear all paths
     */
    void clear() { 
        paths_.clear(); 
    }
    
    /**
     * Estimated memory usage
     */
    [[nodiscard]] size_t memory_usage() const {
        size_t total = sizeof(PixelRawPaths);
        for (const auto& p : paths_) {
            total += p.memory_usage();
        }
        return total;
    }
    
private:
    std::vector<RawPath> paths_;
};

// ============================================================================
// RawPathStorage - Full image storage for all paths
// ============================================================================

class RawPathStorage {
public:
    RawPathStorage() : width_(0), height_(0), sample_count_(0) {}
    
    /**
     * Initialize storage for image dimensions
     */
    void init(size_t width, size_t height, size_t expected_samples_per_pixel) {
        width_ = width;
        height_ = height;
        sample_count_ = expected_samples_per_pixel;
        
        pixels_.resize(width * height);
        
        // Pre-reserve space for each pixel
        for (auto& pixel : pixels_) {
            pixel.reserve(expected_samples_per_pixel);
        }
    }
    
    /**
     * Record a path for a specific pixel
     */
    void record_path(size_t x, size_t y, const PathTrace& trace) {
        if (x >= width_ || y >= height_) return;
        
        size_t idx = y * width_ + x;
        pixels_[idx].add_path(trace);
    }
    
    /**
     * Get paths for a specific pixel
     */
    [[nodiscard]] const PixelRawPaths& pixel(size_t x, size_t y) const {
        static PixelRawPaths empty;
        if (x >= width_ || y >= height_) return empty;
        return pixels_[y * width_ + x];
    }
    
    /**
     * Get paths for a specific pixel (mutable)
     */
    [[nodiscard]] PixelRawPaths& pixel_mut(size_t x, size_t y) {
        return pixels_[y * width_ + x];
    }
    
    /**
     * Image dimensions
     */
    [[nodiscard]] size_t width() const { return width_; }
    [[nodiscard]] size_t height() const { return height_; }
    [[nodiscard]] size_t sample_count() const { return sample_count_; }
    
    /**
     * Total number of paths stored
     */
    [[nodiscard]] size_t total_path_count() const {
        size_t total = 0;
        for (const auto& pixel : pixels_) {
            total += pixel.count();
        }
        return total;
    }
    
    /**
     * Number of pixels with at least one path
     */
    [[nodiscard]] size_t active_pixel_count() const {
        size_t count = 0;
        for (const auto& pixel : pixels_) {
            if (pixel.count() > 0) ++count;
        }
        return count;
    }
    
    /**
     * Estimated total memory usage in bytes
     */
    [[nodiscard]] size_t memory_usage() const {
        size_t total = sizeof(RawPathStorage);
        for (const auto& pixel : pixels_) {
            total += pixel.memory_usage();
        }
        return total;
    }
    
    /**
     * Estimated memory usage in MB
     */
    [[nodiscard]] double memory_usage_mb() const {
        return memory_usage() / (1024.0 * 1024.0);
    }
    
    /**
     * Clear all data
     */
    void clear() {
        for (auto& pixel : pixels_) {
            pixel.clear();
        }
    }
    
    /**
     * Full reset (deallocate memory)
     */
    void reset() {
        pixels_.clear();
        pixels_.shrink_to_fit();
        width_ = 0;
        height_ = 0;
        sample_count_ = 0;
    }
    
private:
    std::vector<PixelRawPaths> pixels_;
    size_t width_;
    size_t height_;
    size_t sample_count_;
};

// ============================================================================
// PathAnalyzer - Analysis utilities for RawPathStorage
// ============================================================================

/**
 * Result of grouping paths by signature
 */
struct PathGroupResult {
    uint64_t    signature_hash;
    std::string signature_string;
    size_t      sample_count;
    RGB3f       total_contribution;
    RGB3f       mean_contribution;
    double      variance;
    SamplingStrategy strategy;
    LightSourceType  light_type;
    
    // Indices into PixelRawPaths for paths in this group
    std::vector<size_t> path_indices;
};

/**
 * Analyze paths for a single pixel
 */
class PixelPathAnalyzer {
public:
    /**
     * Group paths by signature and compute statistics
     */
    static std::vector<PathGroupResult> analyze(const PixelRawPaths& pixel_paths) {
        std::vector<PathGroupResult> results;
        
        // Map: signature_hash -> result index
        std::unordered_map<uint64_t, size_t> hash_to_index;
        
        const auto& paths = pixel_paths.paths();
        for (size_t i = 0; i < paths.size(); ++i) {
            const auto& path = paths[i];
            
            auto it = hash_to_index.find(path.signature_hash);
            if (it == hash_to_index.end()) {
                // New group
                hash_to_index[path.signature_hash] = results.size();
                
                PathGroupResult group;
                group.signature_hash = path.signature_hash;
                group.signature_string = path.signature_string();
                group.sample_count = 1;
                group.total_contribution = path.contribution;
                group.mean_contribution = path.contribution;
                group.variance = 0.0;
                group.strategy = path.strategy;
                group.light_type = path.light_type;
                group.path_indices.push_back(i);
                
                results.push_back(std::move(group));
            } else {
                // Existing group - Welford's online algorithm
                PathGroupResult& group = results[it->second];
                group.path_indices.push_back(i);
                
                size_t n = ++group.sample_count;
                group.total_contribution = group.total_contribution + path.contribution;
                
                // Update mean and variance (Welford's)
                RGB3f old_mean = group.mean_contribution;
                group.mean_contribution.r = old_mean.r + (path.contribution.r - old_mean.r) / n;
                group.mean_contribution.g = old_mean.g + (path.contribution.g - old_mean.g) / n;
                group.mean_contribution.b = old_mean.b + (path.contribution.b - old_mean.b) / n;
                
                // Variance of luminance
                float lum = path.contribution.luminance();
                float old_lum_mean = old_mean.luminance();
                float new_lum_mean = group.mean_contribution.luminance();
                group.variance += (lum - old_lum_mean) * (lum - new_lum_mean);
            }
        }
        
        // Finalize variance
        for (auto& group : results) {
            if (group.sample_count > 1) {
                group.variance /= (group.sample_count - 1);
            }
        }
        
        // Sort by mean contribution (descending)
        std::sort(results.begin(), results.end(),
            [](const PathGroupResult& a, const PathGroupResult& b) {
                return a.mean_contribution.luminance() > b.mean_contribution.luminance();
            });
        
        return results;
    }
    
    /**
     * Sort groups by variance (descending)
     */
    static void sort_by_variance(std::vector<PathGroupResult>& groups) {
        std::sort(groups.begin(), groups.end(),
            [](const PathGroupResult& a, const PathGroupResult& b) {
                return a.variance > b.variance;
            });
    }
    
    /**
     * Sort groups by mean contribution (descending)
     */
    static void sort_by_mean(std::vector<PathGroupResult>& groups) {
        std::sort(groups.begin(), groups.end(),
            [](const PathGroupResult& a, const PathGroupResult& b) {
                return a.mean_contribution.luminance() > b.mean_contribution.luminance();
            });
    }
};

} // namespace render::diagnostics
