/**
 * diagnostic_export.hpp - Export structures for Python binding
 * 
 * POD structures suitable for pybind11 exposure.
 * DiagnosticExporter provides query interface over DiagnosticFilm.
 */

#pragma once

#include <string>
#include <vector>
#include <array>
#include <sstream>

#include "diagnostics/diagnostic_film.hpp"

namespace render::diagnostics {

// ============================================================================
// Export Structures (POD for pybind11)
// ============================================================================

/**
 * Exported path group information
 */
struct ExportedGroupInfo {
    size_t pixel_x;
    size_t pixel_y;
    std::string signature;           // e.g., "LDG" (simple notation)
    std::string signature_heckbert;  // e.g., "LDD D D S DSE" (Heckbert notation)
    std::vector<int32_t> object_ids; // Object IDs for name lookup
    std::string object_path;         // e.g., "Light → Plane → Sphere → Camera"
    uint32_t sample_count;
    float mean_luminance;
    float variance_luminance;
    std::array<float, 3> mean_rgb;
    std::array<float, 3> variance_rgb;
    uint8_t depth;
    uint8_t coarse_type;
    std::string coarse_type_name;
    
    // Plan E: Sampling strategy
    SamplingStrategy strategy = SamplingStrategy::BSDF;  // How the path was sampled
    std::string strategy_name;                           // "BSDF", "NEE", "MIS_BSDF", "MIS_NEE"
    
    // Path geometry for visualization (positions and normals at each vertex)
    std::vector<std::array<float, 3>> positions;  // World positions [depth+1 for camera]
    std::vector<std::array<float, 3>> normals;    // Surface normals [depth]
};

/**
 * Exported coarse type statistics
 */
struct ExportedCoarseStats {
    uint8_t type_id;
    std::string name;
    uint32_t sample_count;
    float mean_luminance;
    float variance_luminance;
    std::array<float, 3> mean_rgb;
    std::array<float, 3> variance_rgb;
};

/**
 * Diagnostic suggestion
 */
struct DiagnosticSuggestion {
    std::string category;   // "Settings", "Scene", "Algorithm"
    std::string severity;   // "info", "warning", "critical"
    std::string message;    // Human-readable description
    std::string action;     // Suggested action
};

// ============================================================================
// DiagnosticExporter - Query interface for Python binding
// ============================================================================

class DiagnosticExporter {
public:
    explicit DiagnosticExporter(const DiagnosticFilm& film)
        : film_(film)
    {}
    
    /**
     * Get global statistics
     */
    [[nodiscard]] GlobalDiagnosticStats get_global_stats() const {
        return film_.global_stats();
    }
    
    /**
     * Get top N groups by variance
     */
    [[nodiscard]] std::vector<ExportedGroupInfo> get_top_variance_groups(size_t n) const {
        auto refs = film_.top_groups_by_variance(n);
        return convert_refs(refs);
    }
    
    /**
     * Get top N groups by mean contribution
     */
    [[nodiscard]] std::vector<ExportedGroupInfo> get_top_mean_groups(size_t n) const {
        auto refs = film_.top_groups_by_mean(n);
        return convert_refs(refs);
    }
    
    /**
     * Get coarse type statistics
     */
    [[nodiscard]] std::vector<ExportedCoarseStats> get_coarse_stats() const {
        std::vector<ExportedCoarseStats> result;
        result.reserve(NUM_COARSE_TYPES);
        
        // Aggregate coarse stats across all pixels
        std::array<PathStatistics, NUM_COARSE_TYPES> aggregated{};
        
        for (size_t sy = 0; sy < film_.sampled_height(); ++sy) {
            for (size_t sx = 0; sx < film_.sampled_width(); ++sx) {
                const auto* px = film_.pixel_data(
                    sx * 2,  // Approximate - config not accessible
                    sy * 2
                );
                if (!px) continue;
                
                for (uint8_t t = 0; t < NUM_COARSE_TYPES; ++t) {
                    const auto& cs = px->coarse_stats(t);
                    if (cs.count > 0) {
                        // Simple merge (not perfectly accurate for variance)
                        aggregated[t].count += cs.count;
                        // Weighted average for mean
                        float w = static_cast<float>(cs.count) / 
                                  static_cast<float>(aggregated[t].count);
                        aggregated[t].mean.r = aggregated[t].mean.r * (1.0f - w) + cs.mean.r * w;
                        aggregated[t].mean.g = aggregated[t].mean.g * (1.0f - w) + cs.mean.g * w;
                        aggregated[t].mean.b = aggregated[t].mean.b * (1.0f - w) + cs.mean.b * w;
                        // Keep max variance as estimate
                        aggregated[t].variance.r = std::max(aggregated[t].variance.r, cs.variance.r);
                        aggregated[t].variance.g = std::max(aggregated[t].variance.g, cs.variance.g);
                        aggregated[t].variance.b = std::max(aggregated[t].variance.b, cs.variance.b);
                    }
                }
            }
        }
        
        for (uint8_t t = 0; t < NUM_COARSE_TYPES; ++t) {
            ExportedCoarseStats ecs;
            ecs.type_id = t;
            ecs.name = coarse_type_name(t);
            ecs.sample_count = aggregated[t].count;
            ecs.mean_luminance = aggregated[t].mean_luminance();
            ecs.variance_luminance = aggregated[t].variance_luminance();
            ecs.mean_rgb = {aggregated[t].mean.r, aggregated[t].mean.g, aggregated[t].mean.b};
            ecs.variance_rgb = {aggregated[t].variance.r, aggregated[t].variance.g, aggregated[t].variance.b};
            result.push_back(ecs);
        }
        
        return result;
    }
    
    /**
     * Export as binary buffer
     */
    [[nodiscard]] std::vector<uint8_t> export_binary() const {
        std::ostringstream oss(std::ios::binary);
        film_.export_binary(oss);
        std::string str = oss.str();
        return std::vector<uint8_t>(str.begin(), str.end());
    }
    
    /**
     * Export metadata as JSON string
     */
    [[nodiscard]] std::string export_metadata_json() const {
        return film_.metadata_json();
    }
    
    /**
     * Generate improvement suggestions based on analysis
     */
    [[nodiscard]] std::vector<DiagnosticSuggestion> generate_suggestions() const {
        std::vector<DiagnosticSuggestion> suggestions;
        
        auto top_var = film_.top_groups_by_variance(10);
        if (top_var.empty()) return suggestions;
        
        // Analyze top variance groups
        size_t glossy_high_var = 0;
        size_t diffuse_high_var = 0;
        size_t indirect_high_var = 0;
        
        for (const auto& ref : top_var) {
            if (!ref.group) continue;
            
            // Check path signature
            bool has_glossy = false;
            bool has_indirect = ref.group->depth > 2;
            
            for (size_t i = 0; i < ref.group->depth; ++i) {
                if (ref.group->vertices[i].bsdf_type == BsdfType::Glossy) {
                    has_glossy = true;
                }
            }
            
            if (has_glossy) ++glossy_high_var;
            if (has_indirect) ++indirect_high_var;
            if (!has_glossy && !has_indirect) ++diffuse_high_var;
        }
        
        // Generate suggestions based on patterns
        if (glossy_high_var > 3) {
            suggestions.push_back({
                "Settings",
                "warning",
                "High variance detected in glossy reflections",
                "Consider increasing samples or using adaptive sampling for glossy materials"
            });
        }
        
        if (indirect_high_var > 3) {
            suggestions.push_back({
                "Algorithm",
                "info",
                "Significant variance from indirect illumination",
                "MIS algorithm recommended; consider path guiding for complex scenes"
            });
        }
        
        if (diffuse_high_var > 3) {
            suggestions.push_back({
                "Scene",
                "info",
                "High variance in diffuse lighting",
                "Check for small or distant light sources; consider area lights"
            });
        }
        
        // Check for overall high variance
        auto stats = film_.global_stats();
        if (stats.total_samples > 0 && stats.total_overflow > stats.active_pixels * 0.1) {
            suggestions.push_back({
                "Settings",
                "warning",
                "Many path types exceeded tracking capacity",
                "Consider using 'Detailed' preset for more accurate analysis"
            });
        }
        
        return suggestions;
    }
    
private:
    const DiagnosticFilm& film_;
    
    /**
     * Convert GroupReference to ExportedGroupInfo
     */
    [[nodiscard]] std::vector<ExportedGroupInfo> convert_refs(
            const std::vector<GroupReference>& refs) const {
        std::vector<ExportedGroupInfo> result;
        result.reserve(refs.size());
        
        for (const auto& ref : refs) {
            if (!ref.group) continue;
            
            ExportedGroupInfo info;
            info.pixel_x = ref.pixel_x;
            info.pixel_y = ref.pixel_y;
            info.signature = ref.group->signature();
            info.sample_count = ref.group->stats.count;
            info.mean_luminance = ref.group->stats.mean_luminance();
            info.variance_luminance = ref.group->stats.variance_luminance();
            info.mean_rgb = {ref.group->stats.mean.r, 
                            ref.group->stats.mean.g, 
                            ref.group->stats.mean.b};
            info.variance_rgb = {ref.group->stats.variance.r,
                                ref.group->stats.variance.g,
                                ref.group->stats.variance.b};
            info.depth = ref.group->depth;
            info.coarse_type = ref.group->coarse_type;
            info.coarse_type_name = coarse_type_name(ref.group->coarse_type);
            
            result.push_back(info);
        }
        
        return result;
    }
};

}  // namespace render::diagnostics
