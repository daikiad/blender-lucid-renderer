/**
 * path_stats.hpp - Path statistics and grouping
 * 
 * Welford's online algorithm for numerically stable variance computation.
 * PathGroup aggregates statistics for identical path structures.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "diagnostics/path_types.hpp"

namespace render::diagnostics {

// Number of coarse path types (3-bit classification)
constexpr size_t NUM_COARSE_TYPES = 8;

/**
 * Get human-readable name for coarse path type
 * 
 * Bits: [indirect][delta][nee]
 */
constexpr const char* coarse_type_name(uint8_t type) {
    // clang-format off
    switch (type & 0x07) {
        case 0b000: return "Direct-NoDelta-BSDF";
        case 0b001: return "Direct-NoDelta-NEE";
        case 0b010: return "Direct-Delta-BSDF";
        case 0b011: return "Direct-Delta-NEE";
        case 0b100: return "Indirect-NoDelta-BSDF";
        case 0b101: return "Indirect-NoDelta-NEE";
        case 0b110: return "Indirect-Delta-BSDF";
        case 0b111: return "Indirect-Delta-NEE";
    }
    // clang-format on
    return "Unknown";
}

// ============================================================================
// PathStatistics - Online mean/variance computation (Welford's algorithm)
// ============================================================================

struct PathStatistics {
    uint32_t count;     // Number of samples
    RGB3f mean;         // Running mean
    RGB3f m2;           // Sum of squared differences (for variance)
    RGB3f variance;     // Computed variance (updated on each sample)
    
    PathStatistics()
        : count(0)
        , mean{}
        , m2{}
        , variance{}
    {}
    
    /**
     * Add a sample using Welford's online algorithm
     * Numerically stable for large offsets and many samples
     */
    void add_sample(const RGB3f& value) {
        ++count;
        
        if (count == 1) {
            mean = value;
            m2 = RGB3f{};
            variance = RGB3f{};
        } else {
            // Welford's update
            RGB3f delta{
                value.r - mean.r,
                value.g - mean.g,
                value.b - mean.b
            };
            
            mean.r += delta.r / static_cast<float>(count);
            mean.g += delta.g / static_cast<float>(count);
            mean.b += delta.b / static_cast<float>(count);
            
            RGB3f delta2{
                value.r - mean.r,
                value.g - mean.g,
                value.b - mean.b
            };
            
            m2.r += delta.r * delta2.r;
            m2.g += delta.g * delta2.g;
            m2.b += delta.b * delta2.b;
            
            // Sample variance (n-1 denominator)
            float n_minus_1 = static_cast<float>(count - 1);
            variance.r = m2.r / n_minus_1;
            variance.g = m2.g / n_minus_1;
            variance.b = m2.b / n_minus_1;
        }
    }
    
    /**
     * Total variance across all channels
     */
    [[nodiscard]] float total_variance() const {
        return variance.r + variance.g + variance.b;
    }
    
    /**
     * Mean luminance (ITU-R BT.709)
     */
    [[nodiscard]] float mean_luminance() const {
        return 0.2126f * mean.r + 0.7152f * mean.g + 0.0722f * mean.b;
    }
    
    /**
     * Variance luminance
     */
    [[nodiscard]] float variance_luminance() const {
        // Variance of linear combination: Var(aX + bY + cZ) ≈ a²Var(X) + b²Var(Y) + c²Var(Z)
        // (assuming independence)
        return 0.2126f * 0.2126f * variance.r 
             + 0.7152f * 0.7152f * variance.g 
             + 0.0722f * 0.0722f * variance.b;
    }
};

// ============================================================================
// PathGroup - Aggregated statistics for identical path structures
// ============================================================================

// Maximum depth stored in PathGroup (can be less than MAX_PATH_DEPTH for memory)
constexpr size_t MAX_GROUP_DEPTH = 16;

struct PathGroup {
    uint64_t    path_hash;               // Hash for quick matching
    PathStatistics stats;                // Aggregated statistics
    uint8_t     depth;                   // Path depth
    uint8_t     coarse_type;             // Coarse classification
    LightSourceType light_type;          // Light source type for Heckbert
    uint8_t     _pad[5];                 // Padding for alignment
    PathVertex  vertices[MAX_GROUP_DEPTH];  // Path structure
    
    PathGroup()
        : path_hash(0)
        , stats{}
        , depth(0)
        , coarse_type(0)
        , light_type(LightSourceType::Unknown)
        , _pad{}
        , vertices{}
    {}
    
    /**
     * Initialize from a PathTrace
     */
    void init_from(const PathTrace& trace) {
        path_hash = trace.hash();
        depth = static_cast<uint8_t>(std::min(trace.depth, MAX_GROUP_DEPTH));
        coarse_type = trace.coarse_type();
        light_type = trace.light_type;
        
        for (size_t i = 0; i < depth; ++i) {
            vertices[i] = trace.vertices[i];
        }
        
        stats = PathStatistics{};
        stats.add_sample(trace.contribution);
    }
    
    /**
     * Check if a trace matches this group
     */
    [[nodiscard]] bool matches(const PathTrace& trace) const {
        return path_hash == trace.hash();
    }
    
    /**
     * Add a sample from a matching trace
     */
    void add_sample(const PathTrace& trace) {
        stats.add_sample(trace.contribution);
    }
    
    /**
     * Generate path signature string (e.g., "LDG" for Light-Diffuse-Glossy)
     */
    [[nodiscard]] std::string signature() const {
        std::string sig;
        sig.reserve(depth);
        for (size_t i = 0; i < depth; ++i) {
            sig += bsdf_type_short(vertices[i].bsdf_type);
        }
        return sig;
    }
    
    /**
     * Generate Heckbert notation signature
     * Format: "LXX reflections DSE" where LXX is light type (LDD/LSD/LDE)
     * and DSE is pinhole camera. Space-separated sections.
     * Example: "LDD D D S DSE" = Area light → Diffuse → Diffuse → Specular → Pinhole
     */
    [[nodiscard]] std::string signature_heckbert() const {
        std::string sig;
        
        // Light source (3 chars)
        sig += light_source_heckbert(light_type);
        
        // Reflections (skip light/environment vertices at the end)
        size_t reflection_end = depth;
        while (reflection_end > 0 && 
               (vertices[reflection_end - 1].bsdf_type == BsdfType::Emission ||
                vertices[reflection_end - 1].bsdf_type == BsdfType::Environment)) {
            --reflection_end;
        }
        
        // Add space before reflections if any
        if (reflection_end > 0) {
            sig += ' ';
            for (size_t i = 0; i < reflection_end; ++i) {
                sig += bsdf_type_heckbert(vertices[i].bsdf_type);
                if (i + 1 < reflection_end) {
                    sig += ' ';  // Space between each reflection
                }
            }
        }
        
        // Camera (pinhole = DSE)
        sig += " DSE";
        
        return sig;
    }
    
    /**
     * Get object IDs in path for name lookup
     */
    [[nodiscard]] std::vector<int32_t> object_ids() const {
        std::vector<int32_t> ids;
        ids.reserve(depth);
        for (size_t i = 0; i < depth; ++i) {
            ids.push_back(vertices[i].object_id);
        }
        return ids;
    }
    
    /**
     * Get total variance (for sorting/ranking)
     */
    [[nodiscard]] float total_variance() const {
        return stats.total_variance();
    }
    
    /**
     * Get mean luminance (for sorting/ranking)
     */
    [[nodiscard]] float mean_contribution() const {
        return stats.mean_luminance();
    }
};

}  // namespace render::diagnostics
