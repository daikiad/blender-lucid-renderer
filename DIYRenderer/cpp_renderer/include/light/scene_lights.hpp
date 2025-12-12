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
#include "../units/units.hpp"
#include "light.hpp"
#include <vector>
#include <algorithm>

// Forward declaration for node evaluator
diy::Color3 getEmissionFromNodeTree(const NodeTree& tree, const Vec2& uv);

// ========== MIS Weight Functions ==========

/**
 * Power heuristic with β = 2
 * w = pf² / (pf² + pg²)
 * 
 * @param pf PDF of sampling strategy f [1/sr]
 * @param pg PDF of sampling strategy g [1/sr]
 * @return MIS weight (dimensionless, [0,1])
 */
inline float powerHeuristic(float pf, float pg) {
    float f2 = pf * pf;
    float g2 = pg * pg;
    return f2 / (f2 + g2 + 1e-6f);
}

/**
 * Balance heuristic
 * w = pf / (pf + pg)
 * 
 * @param pf PDF of sampling strategy f [1/sr]
 * @param pg PDF of sampling strategy g [1/sr]
 * @return MIS weight (dimensionless, [0,1])
 */
inline float balanceHeuristic(float pf, float pg) {
    return pf / (pf + pg + 1e-6f);
}

// ========== Scene Lights Collection ==========

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
    diy::units::Area totalEmissiveArea{0.0f * diy::units::square_metre};  // [m²]
    
    SceneLights() {}
    
    void buildFromScene(const Scene& scene) {
        lights.clear();
        totalEmissiveArea = 0.0f * diy::units::square_metre;
        float totalAreaAccum = 0.0f;  // Local accumulator
        
        // Add emissive mesh triangles
        for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
            const Mesh& mesh = scene.meshes[mi];
            
            // Check if mesh has emission
            diy::Color3 emission(
                diy::units::to_radiance(mesh.material.emission.x),
                diy::units::to_radiance(mesh.material.emission.y),
                diy::units::to_radiance(mesh.material.emission.z)
            );
            if (mesh.material.useNodes && mesh.material.nodeTree.valid) {
                emission = getEmissionFromNodeTree(mesh.material.nodeTree, Vec2(0.0f, 0.0f));
            }
            
            float emissionStrength = emission.x_raw() + emission.y_raw() + emission.z_raw();
            if (emissionStrength < 1e-6f) continue;
            
            // Add each triangle as a light
            for (size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                
                Light light;
                light.v0 = mesh.vertices[tri.i0];
                light.v1 = mesh.vertices[tri.i1];
                light.v2 = mesh.vertices[tri.i2];
                light.normal = tri.faceNormal;
                light.emission = diy::Radiance3(emission.x_raw(), emission.y_raw(), emission.z_raw());
                light.area = triangleArea(light.v0, light.v1, light.v2);
                light.meshIndex = (int)mi;
                light.triangleIndex = (int)ti;
                light.position = (light.v0 + light.v1 + light.v2) / 3.0f;
                
                if (diy::units::to_square_meters(light.area) > 1e-6f) {
                    lights.push_back(light);
                    totalAreaAccum += diy::units::to_square_meters(light.area);
                }
            }
        }
        
        // Add native Blender lights
        for (const Light& nativeLight : scene.nativeLights) {
            Light light = nativeLight;
            
            if (diy::units::to_square_meters(light.area) < 1e-6f) {
                light.area = 1.0f * diy::units::square_metre;  // Default for point-like lights
            }
            
            lights.push_back(light);
            totalAreaAccum += diy::units::to_square_meters(light.area);
        }
        
        // Store total area
        totalEmissiveArea = diy::units::square_meters(totalAreaAccum);
        
        // Build CDF for light selection
        cdf.resize(lights.size());
        float cumulative = 0.0f;
        for (size_t i = 0; i < lights.size(); ++i) {
            cumulative += diy::units::to_square_meters(lights[i].area);
            cdf[i] = (totalAreaAccum > 0.0f) ? (cumulative / totalAreaAccum) : 0.0f;
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
        
        // Selection probability = area_i / total_area (dimensionless)
        float totalArea = diy::units::to_square_meters(totalEmissiveArea);
        selectionProb = (totalArea > 0.0f) ? diy::units::to_square_meters(lights[idx].area) / totalArea : 0.0f;
        
        return idx;
    }
    
    /**
     * Get probability of selecting a specific light
     * @return Selection probability (dimensionless)
     */
    float getPdfForLight(int lightIdx) const {
        if (lightIdx < 0 || lightIdx >= (int)lights.size()) return 0.0f;
        float totalArea = diy::units::to_square_meters(totalEmissiveArea);
        return (totalArea > 0.0f) ? diy::units::to_square_meters(lights[lightIdx].area) / totalArea : 0.0f;
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
        float t;
        diy::Position3 hitPoint;
        diy::Direction3 hitNormal;
        
        switch (light.type) {
            case LightType::POINT: {
                if (diy::units::to_meters(light.radius) > 1e-6f) {
                    float radius_raw = diy::units::to_meters(light.radius);
                    if (geometry::intersectSphere(ray, light.position, radius_raw, t, hitNormal)) {
                        if (t < diy::units::to_meters(result.t)) {
                            result.hit = true;
                            result.t = t * mp_units::si::metre;
                            result.point = diy::Position3(
                                ray.origin.x_raw() + ray.direction.x_raw() * t,
                                ray.origin.y_raw() + ray.direction.y_raw() * t,
                                ray.origin.z_raw() + ray.direction.z_raw() * t
                            );
                            result.normal = hitNormal;
                            result.emission = light.emission;
                            result.lightIndex = (int)i;
                        }
                    }
                }
                break;
            }
            
            case LightType::AREA: {
                bool hitLight = false;
                float sizeX_raw = diy::units::to_meters(light.sizeX);
                float sizeY_raw = diy::units::to_meters(light.sizeY);
                
                if (light.shape == AreaLightShape::DISK || light.shape == AreaLightShape::ELLIPSE) {
                    float radiusX = sizeX_raw * 0.5f;
                    float radiusY = sizeY_raw * 0.5f;
                    hitLight = geometry::intersectEllipse(ray, light.position, light.normal, 
                                                light.right, light.up, 
                                                radiusX, radiusY, t, hitPoint);
                } else {
                    hitLight = geometry::intersectRectangle(ray, light.position, light.normal, 
                                                  light.right, light.up, 
                                                  sizeX_raw, sizeY_raw, t, hitPoint);
                }
                
                if (hitLight && t < diy::units::to_meters(result.t)) {
                    float facing = diy::dot(light.normal, ray.direction).numerical_value_in(mp_units::one);
                    if (facing < 0) {  // Front side
                        result.hit = true;
                        result.t = t * mp_units::si::metre;
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

