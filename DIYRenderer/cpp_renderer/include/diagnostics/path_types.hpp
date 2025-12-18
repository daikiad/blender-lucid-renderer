/**
 * path_types.hpp - Path tracing diagnostic types
 * 
 * Core types for path classification and variance analysis.
 * 
 * Design constraints:
 * - PathVertex must be 8 bytes (cache-friendly, ~2GB for 1080p film)
 * - Stable enum values for serialization
 * - Hash function for path grouping
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "units/render_units.hpp"  // RGB3f

namespace render::diagnostics {

// Maximum vertices in a path (camera to light)
constexpr size_t MAX_PATH_DEPTH = 16;

// ============================================================================
// BsdfType - Surface interaction classification
// ============================================================================

enum class BsdfType : uint8_t {
    Diffuse     = 0,  // Lambert, Oren-Nayar
    Glossy      = 1,  // GGX, Beckmann with roughness
    Mirror      = 2,  // Perfect specular reflection
    Glass       = 3,  // Transmission (refraction)
    Emission    = 4,  // Light source
    Environment = 5,  // Environment map (sky)
};

/**
 * Get human-readable name for BSDF type
 */
constexpr const char* bsdf_type_name(BsdfType type) {
    switch (type) {
        case BsdfType::Diffuse:     return "Diffuse";
        case BsdfType::Glossy:      return "Glossy";
        case BsdfType::Mirror:      return "Mirror";
        case BsdfType::Glass:       return "Glass";
        case BsdfType::Emission:    return "Emission";
        case BsdfType::Environment: return "Environment";
    }
    return "Unknown";
}

/**
 * Get single-character abbreviation for path signature
 * D=Diffuse, G=Glossy, M=Mirror, T=Transmission(Glass), L=Light, E=Environment
 */
constexpr char bsdf_type_short(BsdfType type) {
    switch (type) {
        case BsdfType::Diffuse:     return 'D';
        case BsdfType::Glossy:      return 'G';
        case BsdfType::Mirror:      return 'M';
        case BsdfType::Glass:       return 'T';  // Transmission
        case BsdfType::Emission:    return 'L';  // Light
        case BsdfType::Environment: return 'E';
    }
    return '?';
}

/**
 * Get Heckbert notation character for BSDF type
 * D=Diffuse, S=Specular (Glossy/Mirror/Glass)
 */
constexpr char bsdf_type_heckbert(BsdfType type) {
    switch (type) {
        case BsdfType::Diffuse:     return 'D';
        case BsdfType::Glossy:      return 'S';  // Specular
        case BsdfType::Mirror:      return 'S';  // Specular
        case BsdfType::Glass:       return 'S';  // Specular (transmission)
        case BsdfType::Emission:    return 'E';  // Light end
        case BsdfType::Environment: return 'E';  // Light end
    }
    return '?';
}

// ============================================================================
// LightSourceType - Light source classification for Heckbert notation
// ============================================================================

enum class LightSourceType : uint8_t {
    Unknown     = 0,  // Unclassified
    Point       = 1,  // Point light (LSD)
    Area        = 2,  // Area light (LDD)  
    Directional = 3,  // Directional light (LSD)
    Spot        = 4,  // Spot light (LSD)
    Environment = 5,  // Environment map (LDE)
    Emissive    = 6,  // Emissive mesh (LDD, but use object name)
};

// ============================================================================
// SamplingStrategy - How the path was sampled (Plan E)
// ============================================================================

/**
 * Sampling strategy used to find the light source
 * 
 * This distinguishes HOW a path was constructed, which affects MIS weights.
 * A "completed path" is one that successfully reached a light source.
 */
enum class SamplingStrategy : uint8_t {
    BSDF     = 0,  // BSDF sampling hit light (no MIS)
    NEE      = 1,  // Pure NEE without MIS
    MIS_BSDF = 2,  // MIS: BSDF sampling component (weighted)
    MIS_NEE  = 3,  // MIS: NEE sampling component (weighted)
};

/**
 * Get human-readable name for sampling strategy
 */
constexpr const char* sampling_strategy_name(SamplingStrategy strategy) {
    switch (strategy) {
        case SamplingStrategy::BSDF:     return "BSDF";
        case SamplingStrategy::NEE:      return "NEE";
        case SamplingStrategy::MIS_BSDF: return "MIS_BSDF";
        case SamplingStrategy::MIS_NEE:  return "MIS_NEE";
    }
    return "Unknown";
}

/**
 * Get short name for display
 */
constexpr const char* sampling_strategy_short(SamplingStrategy strategy) {
    switch (strategy) {
        case SamplingStrategy::BSDF:     return "B";
        case SamplingStrategy::NEE:      return "N";
        case SamplingStrategy::MIS_BSDF: return "MB";
        case SamplingStrategy::MIS_NEE:  return "MN";
    }
    return "?";
}

/**
 * Get 3-character Heckbert notation for light source
 * First char: L (light)
 * Second char: positional property (D=diffuse/area, S=specular/point)
 * Third char: directional property (D=diffuse, S=specular/directional, E=environment)
 */
constexpr const char* light_source_heckbert(LightSourceType type) {
    switch (type) {
        case LightSourceType::Point:       return "LSD";  // Point light
        case LightSourceType::Area:        return "LDD";  // Area light
        case LightSourceType::Directional: return "LSD";  // Directional
        case LightSourceType::Spot:        return "LSD";  // Spot light
        case LightSourceType::Environment: return "LDE";  // Environment
        case LightSourceType::Unknown:     return "L??";  // Unknown
    }
    return "L??";
}

/**
 * Convert integer light type to LightSourceType
 * Compatible with LightType enum values (0=EMISSIVE_MESH, 1=POINT, 2=SUN, 3=SPOT, 4=AREA)
 */
constexpr LightSourceType to_light_source_type(int light_type_enum) {
    switch (light_type_enum) {
        case 0: return LightSourceType::Area;        // EMISSIVE_MESH → Area (diffuse)
        case 1: return LightSourceType::Point;       // POINT
        case 2: return LightSourceType::Directional; // SUN
        case 3: return LightSourceType::Spot;        // SPOT
        case 4: return LightSourceType::Area;        // AREA
        default: return LightSourceType::Unknown;
    }
}

// ============================================================================
// PathVertex - Single interaction point (8 bytes)
// ============================================================================

struct PathVertex {
    int32_t  object_id;    // Object index (-1 = environment)
    int16_t  material_id;  // Material index (-1 = none)
    BsdfType bsdf_type;    // Surface type
    uint8_t  flags;        // Bitflags (delta, light_sampled)
    
    // Flag bits
    static constexpr uint8_t FLAG_DELTA         = 0x01;  // Delta distribution (mirror/glass)
    static constexpr uint8_t FLAG_LIGHT_SAMPLED = 0x02;  // NEE (Next Event Estimation)
    
    // Default constructor
    constexpr PathVertex()
        : object_id(-1)
        , material_id(-1)
        , bsdf_type(BsdfType::Diffuse)
        , flags(0)
    {}
    
    // Constructor with values
    constexpr PathVertex(int32_t obj, int16_t mat, BsdfType type, uint8_t f = 0)
        : object_id(obj)
        , material_id(mat)
        , bsdf_type(type)
        , flags(f)
    {}
    
    // Flag accessors
    [[nodiscard]] constexpr bool is_delta() const {
        return (flags & FLAG_DELTA) != 0;
    }
    
    [[nodiscard]] constexpr bool is_light_sampled() const {
        return (flags & FLAG_LIGHT_SAMPLED) != 0;
    }
    
    constexpr void set_delta(bool value) {
        if (value) {
            flags |= FLAG_DELTA;
        } else {
            flags &= ~FLAG_DELTA;
        }
    }
    
    constexpr void set_light_sampled(bool value) {
        if (value) {
            flags |= FLAG_LIGHT_SAMPLED;
        } else {
            flags &= ~FLAG_LIGHT_SAMPLED;
        }
    }
    
    // Equality for path matching
    [[nodiscard]] constexpr bool operator==(const PathVertex& other) const {
        return object_id == other.object_id
            && material_id == other.material_id
            && bsdf_type == other.bsdf_type
            && flags == other.flags;
    }
    
    [[nodiscard]] constexpr bool operator!=(const PathVertex& other) const {
        return !(*this == other);
    }
};

// Verify size constraint
static_assert(sizeof(PathVertex) == 8, "PathVertex must be 8 bytes");

// ============================================================================
// PathTrace - Complete path from camera to light
// ============================================================================

struct PathTrace {
    PathVertex vertices[MAX_PATH_DEPTH];  // Vertex array
    size_t     depth;                     // Number of vertices (0 = empty)
    RGB3f      contribution;              // Final radiance contribution
    LightSourceType light_type;           // Light source type for Heckbert notation
    SamplingStrategy strategy;            // How this path was sampled (Plan E)
    
    // Default constructor
    PathTrace()
        : vertices{}
        , depth(0)
        , contribution{}
        , light_type(LightSourceType::Unknown)
        , strategy(SamplingStrategy::BSDF)
    {}
    
    /**
     * Clear path for reuse
     */
    void clear() {
        depth = 0;
        contribution = RGB3f{};
        light_type = LightSourceType::Unknown;
        strategy = SamplingStrategy::BSDF;
    }
    
    /**
     * Add a vertex to the path
     * Returns false if path is full
     */
    bool add_vertex(int32_t object_id, int16_t material_id, BsdfType type) {
        if (depth >= MAX_PATH_DEPTH) {
            return false;
        }
        vertices[depth++] = PathVertex{object_id, material_id, type};
        return true;
    }
    
    /**
     * Compute hash for path grouping
     * Based on full vertex sequence for exact matching
     */
    [[nodiscard]] uint64_t hash() const {
        // FNV-1a hash
        uint64_t h = 14695981039346656037ULL;
        
        // Include depth in hash
        h ^= depth;
        h *= 1099511628211ULL;
        
        // Hash each vertex
        for (size_t i = 0; i < depth; ++i) {
            const auto& v = vertices[i];
            h ^= static_cast<uint64_t>(v.object_id);
            h *= 1099511628211ULL;
            h ^= static_cast<uint64_t>(v.material_id);
            h *= 1099511628211ULL;
            h ^= static_cast<uint64_t>(v.bsdf_type);
            h *= 1099511628211ULL;
            h ^= static_cast<uint64_t>(v.flags);
            h *= 1099511628211ULL;
        }
        
        return h;
    }
    
    /**
     * Compute coarse path type (8-bit classification)
     * 
     * Bit layout:
     *   bit 0: has_light_sampled (NEE used)
     *   bit 1: has_delta (mirror/glass in path)
     *   bit 2: is_indirect (depth > 2)
     *   bits 3-7: reserved
     */
    [[nodiscard]] uint8_t coarse_type() const {
        uint8_t type = 0;
        
        for (size_t i = 0; i < depth; ++i) {
            if (vertices[i].is_light_sampled()) {
                type |= 0x01;
            }
            if (vertices[i].is_delta()) {
                type |= 0x02;
            }
        }
        
        // Indirect = more than direct lighting (camera -> surface -> light = 2)
        if (depth > 2) {
            type |= 0x04;
        }
        
        return type;
    }
};

}  // namespace render::diagnostics
