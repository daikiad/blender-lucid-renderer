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
#include "diagnostics/raw_path_storage.hpp"

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
 *
 *   // Plan E - Completed Path API:
 *   recorder.begin_path();
 *   recorder.record_vertex(...);
 *   recorder.record_completed_path(strategy, light_id, light_type, contribution);
 *   const auto& paths = recorder.get_completed_paths();
 */
class PathDiagnosticRecorder {
public:
    PathDiagnosticRecorder() : current_path_{}, nee_paths_{}, completed_paths_{} {}
    
    /**
     * Start recording a new path
     */
    void begin_path() {
        current_path_.clear();
        nee_paths_.clear();
        completed_paths_.clear();
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
     * Record a surface interaction vertex with geometry (position and normal)
     * Used for path visualization feature
     */
    void record_vertex_with_geometry(int32_t object_id, int16_t material_id, 
                                     BsdfType type, bool is_delta, bool is_light_sampled,
                                     const Vec3f& position, const Vec3f& normal) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{object_id, material_id, type, 0};
        v.set_delta(is_delta);
        v.set_light_sampled(is_light_sampled);
        
        size_t idx = current_path_.depth;
        current_path_.vertices[idx] = v;
        current_path_.positions[idx] = position;
        current_path_.normals[idx] = normal;
        ++current_path_.depth;
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
     * Record environment map hit with ray direction for visualization
     * The position is set to a far point along the ray direction
     */
    void record_environment_hit_with_geometry(const Vec3f& ray_direction, const Vec3f& ray_origin) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{-1, -1, BsdfType::Environment, 0};
        
        size_t idx = current_path_.depth;
        current_path_.vertices[idx] = v;
        // Place environment hit at a far distance along ray direction
        constexpr float ENV_DISTANCE = 1000.0f;
        current_path_.positions[idx] = ray_origin + ray_direction * ENV_DISTANCE;
        current_path_.normals[idx] = ray_direction * -1.0f;  // Normal points back toward camera
        ++current_path_.depth;
        
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
     * Record light hit with geometry (position for visualization)
     */
    void record_light_hit_with_geometry(int32_t light_id, 
                                        LightSourceType light_source_type,
                                        const Vec3f& position) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{-(light_id + 2), -1, BsdfType::Emission, 0};
        
        size_t idx = current_path_.depth;
        current_path_.vertices[idx] = v;
        current_path_.positions[idx] = position;
        current_path_.normals[idx] = Vec3f{0, 0, 0};  // Lights don't have surface normals
        ++current_path_.depth;
        
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
     * Record emissive mesh hit with geometry
     */
    void record_emissive_hit_with_geometry(int32_t object_id, int16_t material_id,
                                           const Vec3f& position, const Vec3f& normal) {
        if (current_path_.depth >= MAX_PATH_DEPTH) return;
        
        PathVertex v{object_id, material_id, BsdfType::Emission, 0};
        
        size_t idx = current_path_.depth;
        current_path_.vertices[idx] = v;
        current_path_.positions[idx] = position;
        current_path_.normals[idx] = normal;
        ++current_path_.depth;
        
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
    
    // ========================================================================
    // Plan E: Completed Path API
    // ========================================================================
    
    /**
     * Record a completed path that reached a light source
     * 
     * This is the new API for Plan E. Instead of maintaining a "main path"
     * and separate NEE paths, we record each completed path individually
     * with its sampling strategy and MIS-weighted contribution.
     * 
     * @param strategy How the light was found (BSDF, NEE, MIS_BSDF, MIS_NEE)
     * @param light_id Light index (-1 for environment)
     * @param light_source_type Type of light source
     * @param contribution MIS-weighted contribution of this path
     */
    void record_completed_path(SamplingStrategy strategy,
                               int32_t light_id,
                               LightSourceType light_source_type,
                               const RGB3f& contribution) {
        PathTrace path = current_path_;  // Copy current vertices (including geometry)
        
        // Add light vertex
        if (path.depth < MAX_PATH_DEPTH) {
            if (light_source_type == LightSourceType::Environment) {
                // Environment: use -1 as object_id marker
                PathVertex v{-1, -1, BsdfType::Emission, 0};
                path.vertices[path.depth++] = v;
            } else {
                // Light: use negative ID encoding (-(light_id + 2))
                PathVertex v{-(light_id + 2), -1, BsdfType::Emission, 0};
                path.vertices[path.depth++] = v;
            }
        }
        
        path.light_type = light_source_type;
        path.strategy = strategy;
        path.contribution = contribution;
        
        completed_paths_.push_back(path);
    }
    
    /**
     * Record a completed path with light position for visualization
     * 
     * @param strategy How the light was found (BSDF, NEE, MIS_BSDF, MIS_NEE)
     * @param light_id Light index (-1 for environment)
     * @param light_source_type Type of light source
     * @param contribution MIS-weighted contribution of this path
     * @param light_position World position of the light hit point
     */
    void record_completed_path_with_geometry(SamplingStrategy strategy,
                                             int32_t light_id,
                                             LightSourceType light_source_type,
                                             const RGB3f& contribution,
                                             const Vec3f& light_position) {
        PathTrace path = current_path_;  // Copy current vertices (including geometry)
        
        // Add light vertex with geometry
        if (path.depth < MAX_PATH_DEPTH) {
            size_t idx = path.depth;
            if (light_source_type == LightSourceType::Environment) {
                PathVertex v{-1, -1, BsdfType::Emission, 0};
                path.vertices[idx] = v;
            } else {
                PathVertex v{-(light_id + 2), -1, BsdfType::Emission, 0};
                path.vertices[idx] = v;
            }
            path.positions[idx] = light_position;
            path.normals[idx] = Vec3f{0, 0, 0};  // Lights don't have meaningful normals
            ++path.depth;
        }
        
        path.light_type = light_source_type;
        path.strategy = strategy;
        path.contribution = contribution;
        
        completed_paths_.push_back(path);
    }
    
    /**
     * Get all completed paths recorded for this pixel
     */
    [[nodiscard]] const std::vector<PathTrace>& get_completed_paths() const {
        return completed_paths_;
    }
    
private:
    PathTrace current_path_;
    std::vector<PathTrace> nee_paths_;
    std::vector<PathTrace> completed_paths_;  // Plan E: completed paths
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
        , raw_storage_enabled_(false)
        , raw_storage_()
    {}
    
    /**
     * Enable/disable recording
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool is_enabled() const { return enabled_; }
    
    /**
     * Enable raw path storage for full path collection
     * @param expected_spp Expected samples per pixel for memory pre-allocation
     */
    void enable_raw_storage(size_t expected_spp) {
        raw_storage_.init(film_.width(), film_.height(), expected_spp);
        raw_storage_enabled_ = true;
    }
    
    /**
     * Disable raw path storage and free memory
     */
    void disable_raw_storage() {
        raw_storage_enabled_ = false;
        raw_storage_.reset();
    }
    
    /**
     * Check if raw storage is enabled
     */
    [[nodiscard]] bool is_raw_storage_enabled() const { return raw_storage_enabled_; }
    
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
        
        // Also record to raw storage if enabled
        if (raw_storage_enabled_) {
            raw_storage_.record_path(x, y, trace);
        }
    }
    
    /**
     * Access the underlying film
     */
    [[nodiscard]] const DiagnosticFilm& film() const { return film_; }
    [[nodiscard]] DiagnosticFilm& film() { return film_; }
    
    /**
     * Access raw path storage (for full path analysis)
     */
    [[nodiscard]] const RawPathStorage& raw_storage() const { return raw_storage_; }
    [[nodiscard]] RawPathStorage& raw_storage() { return raw_storage_; }
    
    /**
     * Clear all recorded data
     */
    void clear() {
        film_.clear();
        if (raw_storage_enabled_) {
            raw_storage_.clear();
        }
    }
    
    /**
     * Get configuration
     */
    [[nodiscard]] const PathRecordingConfig& config() const { return config_; }
    
private:
    DiagnosticFilm film_;
    PathRecordingConfig config_;
    bool enabled_;
    bool raw_storage_enabled_;
    RawPathStorage raw_storage_;
};

}  // namespace render::diagnostics
