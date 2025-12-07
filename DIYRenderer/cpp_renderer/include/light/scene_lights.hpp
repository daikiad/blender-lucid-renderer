/**
 * scene_lights.hpp - Scene Light Collection and MIS
 * ==================================================
 * 
 * Manages all lights in a scene for efficient sampling:
 * - SceneLights: Collection of all emissive surfaces and native lights
 * - Light selection via CDF (importance sampling by area)
 * - Multiple Importance Sampling (MIS) utilities
 * - Native light intersection
 */

#pragma once
#include "../core/scene.hpp"
#include "../geometry/intersection.hpp"
#include "light.hpp"
#include <vector>
#include <algorithm>

// Forward declaration for node evaluator
Vec3 getEmissionFromNodeTree(const NodeTree& tree, const Vec2& uv);

// ========== MIS Weight Functions ==========

/**
 * Power heuristic with β = 2
 * w = pf² / (pf² + pg²)
 */
inline float powerHeuristic(float pf, float pg) {
    float f2 = pf * pf;
    float g2 = pg * pg;
    return f2 / (f2 + g2 + 1e-6f);
}

/**
 * Balance heuristic
 * w = pf / (pf + pg)
 */
inline float balanceHeuristic(float pf, float pg) {
    return pf / (pf + pg + 1e-6f);
}

// ========== Scene Lights Collection ==========

struct SceneLights {
    std::vector<Light> lights;
    float totalArea;
    std::vector<float> cdf;  // CDF for importance sampling
    
    SceneLights() : totalArea(0.0f) {}
    
    void buildFromScene(const Scene& scene) {
        lights.clear();
        totalArea = 0.0f;
        
        // Add emissive mesh triangles
        for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
            const Mesh& mesh = scene.meshes[mi];
            
            // Check if mesh has emission
            Vec3 emission = mesh.material.emission;
            if (mesh.material.useNodes && mesh.material.nodeTree.valid) {
                emission = getEmissionFromNodeTree(mesh.material.nodeTree, Vec2(0.0f, 0.0f));
            }
            
            float emissionStrength = emission.x + emission.y + emission.z;
            if (emissionStrength < 1e-6f) continue;
            
            // Add each triangle as a light
            for (size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                
                Light light;
                light.v0 = mesh.vertices[tri.i0];
                light.v1 = mesh.vertices[tri.i1];
                light.v2 = mesh.vertices[tri.i2];
                light.normal = tri.faceNormal;
                light.emission = emission;
                light.area = triangleArea(light.v0, light.v1, light.v2);
                light.meshIndex = (int)mi;
                light.triangleIndex = (int)ti;
                light.position = (light.v0 + light.v1 + light.v2) * (1.0f / 3.0f);
                
                if (light.area > 1e-6f) {
                    lights.push_back(light);
                    totalArea += light.area;
                }
            }
        }
        
        // Add native Blender lights
        for (const Light& nativeLight : scene.nativeLights) {
            Light light = nativeLight;
            
            if (light.area < 1e-6f) {
                light.area = 1.0f;  // Default for point-like lights
            }
            
            lights.push_back(light);
            totalArea += light.area;
        }
        
        // Build CDF for light selection
        cdf.resize(lights.size());
        float cumulative = 0.0f;
        for (size_t i = 0; i < lights.size(); ++i) {
            cumulative += lights[i].area;
            cdf[i] = cumulative / totalArea;
        }
    }
    
    /**
     * Select a light based on area-weighted probability
     * @return index and selection probability
     */
    int selectLight(float u, float& selectionProb) const {
        if (lights.empty()) {
            selectionProb = 0.0f;
            return -1;
        }
        
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        int idx = (int)(it - cdf.begin());
        idx = std::min(idx, (int)lights.size() - 1);
        
        selectionProb = lights[idx].area / totalArea;
        
        return idx;
    }
    
    float getPdfForLight(int lightIdx) const {
        if (lightIdx < 0 || lightIdx >= (int)lights.size()) return 0.0f;
        return lights[lightIdx].area / totalArea;
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
        Vec3 hitPoint, hitNormal;
        
        switch (light.type) {
            case LightType::POINT: {
                if (light.radius > 1e-6f) {
                    if (geometry::intersectSphere(ray, light.position, light.radius, t, hitNormal)) {
                        if (t < result.t) {
                            result.hit = true;
                            result.t = t;
                            result.point = ray.o + ray.d * t;
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
                
                if (light.shape == AreaLightShape::DISK || light.shape == AreaLightShape::ELLIPSE) {
                    float radiusX = light.sizeX * 0.5f;
                    float radiusY = light.sizeY * 0.5f;
                    hitLight = geometry::intersectEllipse(ray, light.position, light.normal, 
                                                light.right, light.up, 
                                                radiusX, radiusY, t, hitPoint);
                } else {
                    hitLight = geometry::intersectRectangle(ray, light.position, light.normal, 
                                                  light.right, light.up, 
                                                  light.sizeX, light.sizeY, t, hitPoint);
                }
                
                if (hitLight && t < result.t) {
                    float facing = Vec3::dot(light.normal, ray.d);
                    if (facing < 0) {  // Front side
                        result.hit = true;
                        result.t = t;
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
