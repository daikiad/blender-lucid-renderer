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
    
    // Default constructor
    PathTrace()
        : vertices{}
        , depth(0)
        , contribution{}
    {}
    
    /**
     * Clear path for reuse
     */
    void clear() {
        depth = 0;
        contribution = RGB3f{};
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
