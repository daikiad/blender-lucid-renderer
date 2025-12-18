/**
 * diagnostic_integrator.hpp - Path tracer integration for diagnostics
 * 
 * Provides helpers for recording path diagnostics during rendering:
 * - PathDiagnosticRecorder: Per-thread path construction
 * - DiagnosticPathTracer: Wrapper around DiagnosticFilm with enable/disable
 * - classify_bsdf: Material to BsdfType classification
 */

#pragma once

#include "diagnostics/diagnostic_film.hpp"

namespace render::diagnostics {

// ============================================================================
// classify_bsdf - Classify material into BsdfType (template for any material struct)
// ============================================================================

/**
 * Classify a material's BSDF type based on its parameters
 * Works with any struct that has roughness, metallic, transmission fields
 */
template <typename MaterialT>
[[nodiscard]] inline BsdfType classify_bsdf(const MaterialT& mat) {
    // Transmission takes priority
    if (mat.transmission > 0.5f) {
        return BsdfType::Glass;
    }
    
    // Check for mirror (perfect specular)
    constexpr float MIRROR_THRESHOLD = 0.01f;
    if (mat.roughness < MIRROR_THRESHOLD && mat.metallic > 0.5f) {
        return BsdfType::Mirror;
    }
    
    // Check for glossy (metallic with roughness)
    if (mat.metallic > 0.5f || mat.roughness < 0.5f) {
        return BsdfType::Glossy;
    }
    
    // Default: diffuse
    return BsdfType::Diffuse;
}

/**
 * Check if a BSDF sample is from a delta distribution
 */
template <typename MaterialT>
[[nodiscard]] inline bool is_delta_bsdf(const MaterialT& mat) {
    constexpr float DELTA_THRESHOLD = 0.01f;
    
    if (mat.transmission > 0.5f && mat.roughness < DELTA_THRESHOLD) {
        return true;  // Perfect glass
    }
    if (mat.metallic > 0.5f && mat.roughness < DELTA_THRESHOLD) {
        return true;  // Perfect mirror
    }
    return false;
}

// ============================================================================
// PathDiagnosticRecorder - Per-thread path construction helper
// ============================================================================

/**
 * Records path vertices during tracing
 * 
 * Usage:
 *   PathDiagnosticRecorder recorder;
 *   recorder.begin_path();
 *   // During tracing:
 *   recorder.record_vertex(obj_id, mat_id, type, is_delta, is_nee);
 *   // For NEE:
 *   recorder.record_nee_contribution(light_id, light_type, contribution);
 *   // At path end:
 *   PathTrace trace = recorder.end_path(contribution);
 *   // Get NEE paths:
 *   const auto& nee_paths = recorder.get_nee_paths();
 */
class PathDiagnosticRecorder {
public:
    PathDiagnosticRecorder() : current_path_{}, nee_paths_{} {}
    
    /**
     * Start recording a new path
     */
    void begin_path() {
        current_path_.clear();
        nee_paths_.clear();
    }
    
    /**
     * Record a surface interaction vertex
     */
    void record_vertex(int32_t object_id, int16_t material_id, 
                       BsdfType type, bool is_delta, bool is_light_sampled) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{object_id, material_id, type, 0};
        v.set_delta(is_delta);
        v.set_light_sampled(is_light_sampled);
        
        current_path_.vertices[current_path_.depth++] = v;
    }
    
    /**
     * Record environment map hit (no surface)
     * Sets light type to Environment for Heckbert notation
     */
    void record_environment_hit() {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{-1, -1, BsdfType::Environment, 0};
        current_path_.vertices[current_path_.depth++] = v;
        
        // Environment is also a light source
        current_path_.light_type = LightSourceType::Environment;
    }
    
    /**
     * Record light hit with light source type
     * @param light_id Light index
     * @param light_source_type Type for Heckbert notation (Area, Point, etc.)
     */
    void record_light_hit(int32_t light_id, LightSourceType light_source_type = LightSourceType::Unknown) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        // Use negative IDs for lights to distinguish from mesh objects
        PathVertex v{-(light_id + 2), -1, BsdfType::Emission, 0};
        current_path_.vertices[current_path_.depth++] = v;
        
        // Record light source type for Heckbert notation
        current_path_.light_type = light_source_type;
    }
    
    /**
     * Record emissive mesh hit
     * Emissive meshes use the Emissive light type to show object name
     */
    void record_emissive_hit(int32_t object_id, int16_t material_id) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{object_id, material_id, BsdfType::Emission, 0};
        current_path_.vertices[current_path_.depth++] = v;
        
        // Use Emissive type to distinguish from Blender native lights
        current_path_.light_type = LightSourceType::Emissive;
    }
    
    /**
     * Get current path depth
     */
    [[nodiscard]] size_t current_depth() const {
        return current_path_.depth;
    }
    
    /**
     * Finalize path with contribution and return PathTrace
     */
    [[nodiscard]] PathTrace end_path(const RGB3f& contribution) {
        current_path_.contribution = contribution;
        return current_path_;
    }
    
    /**
     * Create a branched path for NEE (Next Event Estimation)
     * Returns a copy of current path with light hit appended
     * Does not modify the current path being traced
     * 
     * @param light_id Index into scene.nativeLights
     * @param light_source_type Type for Heckbert notation
     * @param contribution NEE contribution at this vertex
     */
    [[nodiscard]] PathTrace create_nee_path(int32_t light_id, 
                                            LightSourceType light_source_type,
                                            const RGB3f& contribution) {
        PathTrace nee_path = current_path_;  // Copy current path
        
        if (nee_path.depth < MAX_PATH_DEPTH) {
            // Add light vertex
            PathVertex v{-(light_id + 2), -1, BsdfType::Emission, 0};
            nee_path.vertices[nee_path.depth++] = v;
        }
        
        nee_path.light_type = light_source_type;
        nee_path.contribution = contribution;
        
        return nee_path;
    }
    
    /**
     * Record NEE contribution as a separate path
     * The NEE path is stored and can be retrieved later
     * 
     * @param light_id Index into scene.nativeLights  
     * @param light_source_type Type for Heckbert notation
     * @param contribution NEE contribution (throughput * light emission * BSDF weight)
     */
    void record_nee_contribution(int32_t light_id,
                                 LightSourceType light_source_type,
                                 const RGB3f& contribution) {
        PathTrace nee_path = create_nee_path(light_id, light_source_type, contribution);
        nee_paths_.push_back(nee_path);
    }
    
    /**
     * Get all recorded NEE paths
     */
    [[nodiscard]] const std::vector<PathTrace>& get_nee_paths() const {
        return nee_paths_;
    }
    
private:
    PathTrace current_path_;
    std::vector<PathTrace> nee_paths_;
};

// ============================================================================
// DiagnosticPathTracer - Wrapper with enable/disable control
// ============================================================================

/**
 * Wraps DiagnosticFilm with runtime enable/disable control
 * 
 * Usage:
 *   DiagnosticPathTracer diag(width, height, config);
 *   diag.set_enabled(true);
 *   
 *   // During rendering:
 *   if (diag.should_record(x, y)) {
 *       PathTrace trace = build_trace(...);
 *       diag.record_path(x, y, trace);
 *   }
 */
class DiagnosticPathTracer {
public:
    DiagnosticPathTracer(size_t width, size_t height, const PathRecordingConfig& config)
        : film_(width, height, config)
        , config_(config)
        , enabled_(true)
    {}
    
    /**
     * Enable/disable recording
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool is_enabled() const { return enabled_; }
    
    /**
     * Check if we should record at this pixel (subsampling)
     */
    [[nodiscard]] bool should_record(size_t x, size_t y) const {
        if (!enabled_) return false;
        
        // Subsample check
        return (x % config_.subsample_factor == 0) && 
               (y % config_.subsample_factor == 0);
    }
    
    /**
     * Record a completed path
     */
    void record_path(size_t x, size_t y, const PathTrace& trace) {
        if (!enabled_) return;
        film_.record_path(x, y, trace);
    }
    
    /**
     * Access the underlying film
     */
    [[nodiscard]] const DiagnosticFilm& film() const { return film_; }
    [[nodiscard]] DiagnosticFilm& film() { return film_; }
    
    /**
     * Clear all recorded data
     */
    void clear() {
        film_.clear();
    }
    
    /**
     * Get configuration
     */
    [[nodiscard]] const PathRecordingConfig& config() const { return config_; }
    
private:
    DiagnosticFilm film_;
    PathRecordingConfig config_;
    bool enabled_;
};

}  // namespace render::diagnostics
