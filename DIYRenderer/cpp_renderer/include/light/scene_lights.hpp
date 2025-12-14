/**
 * scene_lights.hpp - Scene Light Collection and MIS
 * ==================================================
 * 
 * Manages all lights in a scene for efficient sampling:
 * - SceneLights: Collection of all emissive surfaces and native lights
 * - Light selection via CDF (importance sampling by area)
 * - Multiple Importance Sampling (MIS) utilities
 * - Native light intersection
 * 
 * Physical Units:
 * - totalArea: [m²] - total emissive surface area
 * - Light selection probability: dimensionless [0,1]
 * - MIS weights: dimensionless [0,1]
 * - PDF for light sampling: [1/sr] (solid angle measure)
 */

#pragma once
#include "../core/scene.hpp"
#include "../geometry/intersection.hpp"
#include "../units/render_units.hpp"
#include "light.hpp"
#include <vector>
#include <algorithm>

// Forward declaration for node evaluator
render::ColorRGB getEmissionFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);

// ========== MIS Weight Functions ==========
// NOTE: Use render::mis_power_heuristic(PdfW, PdfW) from render_units.hpp for typed MIS.
// The typed version provides compile-time unit safety.

// ========== Scene Lights Collection ==========

// Use render::displacement_from_origin from render_units.hpp for low-level geometry
using render::displacement_from_origin;

/**
 * SceneLights - Collection of all light sources for importance sampling
 * 
 * Uses area-weighted CDF for light selection.
 * Larger area lights are more likely to be sampled.
 */
struct SceneLights {
    std::vector<Light> lights;
    std::vector<float> cdf;                   // CDF for importance sampling (dimensionless)
    
    // Unit-typed properties
    render::Area totalEmissiveArea{0.0f * mp_units::square(mp_units::si::metre)};  // [m²]
    
    // Index where native lights start in lights[] (after emissive mesh triangles)
    int nativeLightStartIndex = 0;
    
    SceneLights() {}
    
    void buildFromScene(const Scene& scene) {
        lights.clear();
        totalEmissiveArea = 0.0f * mp_units::square(mp_units::si::metre);
        
        // Add emissive mesh triangles
        for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
            const Mesh& mesh = scene.meshes[mi];
            
            // Check if mesh has emission using is_emissive (unit-typed)
            render::RadianceRGB emission = mesh.material.emission;
            if (mesh.material.useNodes && mesh.material.nodeTree.valid) {
                render::ColorRGB emissionColor = getEmissionFromNodeTree(mesh.material.nodeTree, render::Vec2f(0.0f, 0.0f));
                emission = render::to_radiance(emissionColor);
            }
            
            if (!render::is_emissive(emission)) continue;
            
            // Add each triangle as a light
            for (size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                
                Light light;
                light.v0 = mesh.vertices[tri.i0];
                light.v1 = mesh.vertices[tri.i1];
                light.v2 = mesh.vertices[tri.i2];
                light.normal = tri.faceNormal;
                light.emission = emission;  // Already RadianceRGB
                light.area = triangleArea(light.v0, light.v1, light.v2);
                light.meshIndex = (int)mi;
                light.triangleIndex = (int)ti;
                // Compute centroid using typed arithmetic (Displacement - ISQ compliant!)
                render::Displacement centroid_pv = (displacement_from_origin(light.v0) + 
                                                       displacement_from_origin(light.v1) + 
                                                       displacement_from_origin(light.v2)) / 3.0f;
                light.position = render::world_origin + centroid_pv;
                
                if (light.area > render::MIN_AREA) {
                    lights.push_back(light);
                    totalEmissiveArea += light.area;
                }
            }
        }
        
        // Add native Blender lights
        // Record where native lights start (after emissive mesh triangles)
        nativeLightStartIndex = static_cast<int>(lights.size());
        
        for (const Light& nativeLight : scene.nativeLights) {
            Light light = nativeLight;
            
            if (light.area < render::MIN_AREA) {
                light.area = 1.0f * mp_units::square(mp_units::si::metre);  // Default for point-like lights
            }
            
            lights.push_back(light);
            totalEmissiveArea += light.area;
        }
        
        // Build CDF for light selection (dimensionless ratios)
        cdf.resize(lights.size());
        render::Area cumulative = 0.0f * mp_units::square(mp_units::si::metre);
        for (size_t i = 0; i < lights.size(); ++i) {
            cumulative += lights[i].area;
            // Extract at storage boundary (cdf is float array for std::lower_bound)
            cdf[i] = render::area_ratio(cumulative, totalEmissiveArea).numerical_value_in(mp_units::one);
        }
    }
    
    /**
     * Select a light based on area-weighted probability
     * @param u Random number in [0,1] (dimensionless)
     * @param selectionProb Output: probability of selecting this light (dimensionless)
     * @return Light index, or -1 if no lights
     */
    int selectLight(float u, float& selectionProb) const {
        if (lights.empty()) {
            selectionProb = 0.0f;
            return -1;
        }
        
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        int idx = (int)(it - cdf.begin());
        idx = std::min(idx, (int)lights.size() - 1);
        
        // Selection probability = area_i / total_area (extract at interface boundary)
        selectionProb = render::area_ratio(lights[idx].area, totalEmissiveArea).numerical_value_in(mp_units::one);
        
        return idx;
    }
    
    /**
     * Get probability of selecting a specific light
     * @return Selection probability (dimensionless)
     */
    float getPdfForLight(int lightIdx) const {
        if (lightIdx < 0 || lightIdx >= (int)lights.size()) return 0.0f;
        // Extract at interface boundary (return type is float)
        return render::area_ratio(lights[lightIdx].area, totalEmissiveArea).numerical_value_in(mp_units::one);
    }
    
    /**
     * Find the index in lights[] for a given scene.nativeLights index
     * @param nativeLightIndex Index into scene.nativeLights (from intersectNativeLights)
     * @return Index into lights[], or -1 if not found
     */
    int findNativeLightIndex(int nativeLightIndex) const {
        int idx = nativeLightStartIndex + nativeLightIndex;
        return (idx >= 0 && idx < static_cast<int>(lights.size())) ? idx : -1;
    }
    
    bool hasLights() const { return !lights.empty(); }
};

// ========== Native Light Intersection ==========

/**
 * Intersect ray with all native lights in the scene
 * @param depth current bounce depth (0 = primary ray from camera)
 * Native lights are invisible to camera (depth 0) but visible via reflections (depth > 0)
 */
inline LightHit intersectNativeLights(const Scene& scene, const Ray& ray, int depth) {
    LightHit result;
    
    // Native lights are invisible to camera rays
    if (depth == 0) {
        return result;
    }
    
    for (size_t i = 0; i < scene.nativeLights.size(); ++i) {
        const Light& light = scene.nativeLights[i];
render::Length t_dist;
        render::Position hitPoint;
        render::Normal hitNormal;
        
        switch (light.type) {
            case LightType::POINT: {
                if (light.radius > render::MIN_LENGTH) {
                    render::Normal temp_normal;
                    if (geometry::intersectSphere(ray, light.position, light.radius, t_dist, temp_normal)) {
                        if (t_dist < result.t) {
                            result.hit = true;
                            result.t = t_dist;
                            result.point = ray.at(t_dist);
                            result.normal = temp_normal;
                            result.emission = light.emission;
                            result.lightIndex = (int)i;
                        }
                    }
                }
                break;
            }
            
            case LightType::AREA: {
                bool hitLight = false;
                
                if (light.shape == AreaLightShape::DISK || light.shape == AreaLightShape::ELLIPSE) {
                    // Ellipse radii = size / 2
                    auto radiusX = light.sizeX * 0.5f;
                    auto radiusY = light.sizeY * 0.5f;
                    hitLight = geometry::intersectEllipse(ray, light.position, light.normal.as_direction(), 
                                                light.right, light.up, 
                                                radiusX, radiusY, t_dist, hitPoint);
                } else {
                    hitLight = geometry::intersectRectangle(ray, light.position, light.normal.as_direction(), 
                                                  light.right, light.up, 
                                                  light.sizeX, light.sizeY, t_dist, hitPoint);
                }
                
                if (hitLight && t_dist < result.t) {
                    float facing = render::dot(light.normal.vec(), ray.direction.vec());
                    if (facing < 0) {  // Front side
                        result.hit = true;
                        result.t = t_dist;
                        result.point = hitPoint;
                        result.normal = light.normal;
                        result.emission = light.emission;
                        result.lightIndex = (int)i;
                    }
                }
                break;
            }
            
            case LightType::SUN:
            case LightType::SPOT:
            default:
                break;
        }
    }
    
    return result;
}

