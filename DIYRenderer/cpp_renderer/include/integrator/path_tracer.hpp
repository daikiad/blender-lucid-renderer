/**
 * path_tracer.hpp - Path Tracing Integrators
 * ===========================================
 * 
 * Path tracing implementations with different sampling strategies:
 * - traceSimple: BSDF sampling only (no NEE)
 * - traceNEE: Next Event Estimation (light sampling)
 * - traceMIS: Multiple Importance Sampling (combines both)
 */

#pragma once
#include "../math/vec3.hpp"
#include "../math/random.hpp"
#include "../core/scene.hpp"
#include "../core/material.hpp"
#include "../geometry/bvh.hpp"
#include "../bsdf/bsdf.hpp"
#include "../light/light.hpp"
#include "../light/scene_lights.hpp"
#include <cmath>

// Forward declarations for node evaluators
Vec3 getAlbedoFromNodeTree(const NodeTree& tree, const Vec2& uv);
Vec3 getEmissionFromNodeTree(const NodeTree& tree, const Vec2& uv);
float getTransmissionFromNodeTree(const NodeTree& tree, const Vec2& uv);
float getIORFromNodeTree(const NodeTree& tree, const Vec2& uv);
float getMetallicFromNodeTree(const NodeTree& tree, const Vec2& uv);
float getRoughnessFromNodeTree(const NodeTree& tree, const Vec2& uv);

// ========== Helper Functions ==========

inline MaterialParams getMaterialParams(const Hit& hit) {
    MaterialParams mat;
    mat.albedo = hit.material.albedo;
    mat.metallic = hit.material.metallic;
    mat.roughness = hit.material.roughness;
    mat.transmission = hit.material.transmission;
    mat.ior = hit.material.ior;
    
    if (hit.material.useNodes && hit.material.nodeTree.valid) {
        mat.albedo = getAlbedoFromNodeTree(hit.material.nodeTree, hit.uv);
        mat.metallic = getMetallicFromNodeTree(hit.material.nodeTree, hit.uv);
        mat.roughness = getRoughnessFromNodeTree(hit.material.nodeTree, hit.uv);
        mat.transmission = getTransmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        mat.ior = getIORFromNodeTree(hit.material.nodeTree, hit.uv);
    }
    
    // Enforce minimum roughness
    mat.roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    return mat;
}

inline Vec3 getEmission(const Hit& hit) {
    if (hit.material.useNodes && hit.material.nodeTree.valid) {
        return getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
    }
    return hit.material.emission;
}

// ========== Simple Path Tracer (BSDF only) ==========

/**
 * Simple path tracer using BSDF sampling only (no NEE)
 * Most basic implementation - good for testing
 */
inline Vec3 traceSimple(const Scene& scene, const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float tMin = 0.0f;  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        // Check if we hit a native light closer than any mesh
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            result = result + throughput * lightHit.emission;
            break;
        }
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        Vec3 emission = getEmission(hit);
        
        // Add emission
        result = result + throughput * emission;
        
        // Setup normals
        Vec3 n = hit.normal;
        Vec3 wo = currentRay.d * -1.0f;
        wo.normalize();
        bool frontFace = Vec3::dot(wo, n) > 0;
        
        Vec3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Sample BSDF
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < 1e-6f && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float absNdotL = std::abs(Vec3::dot(sampleNormal, bsdfSample.wi));
            if (absNdotL > 1e-6f && bsdfSample.pdf > 1e-6f) {
                float weightX = bsdfSample.f.x * absNdotL / bsdfSample.pdf;
                float weightY = bsdfSample.f.y * absNdotL / bsdfSample.pdf;
                float weightZ = bsdfSample.f.z * absNdotL / bsdfSample.pdf;
                
                const float MAX_WEIGHT = 10.0f;
                weightX = std::min(weightX, MAX_WEIGHT);
                weightY = std::min(weightY, MAX_WEIGHT);
                weightZ = std::min(weightZ, MAX_WEIGHT);
                
                throughput = Vec3(throughput.x * weightX, throughput.y * weightY, throughput.z * weightZ);
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.x, throughput.y, throughput.z});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x) || std::isinf(throughput.x) ||
            std::isnan(throughput.y) || std::isinf(throughput.y) ||
            std::isnan(throughput.z) || std::isinf(throughput.z)) {
            break;
        }
        
        // Setup next ray
        currentRay.o = hit.point;
        currentRay.d = bsdfSample.wi;
        tMin = geometry::RAY_T_MIN;  // Subsequent rays need offset
    }
    
    return result;
}

// ========== NEE Path Tracer ==========

/**
 * Path tracer with Next Event Estimation
 * Uses light sampling for direct illumination
 */
inline Vec3 traceNEE(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float tMin = 0.0f;  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            result = result + throughput * lightHit.emission;
            break;
        }
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        Vec3 emission = getEmission(hit);
        
        // Setup normals
        Vec3 n = hit.normal;
        Vec3 wo = currentRay.d * -1.0f;
        wo.normalize();
        bool frontFace = Vec3::dot(wo, n) > 0;
        
        Vec3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Add emission only on first hit
        float emissionStrength = emission.x + emission.y + emission.z;
        if (emissionStrength > 1e-6f && depth == 0) {
            result = result + throughput * emission;
        }
        
        // Next Event Estimation
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && emissionStrength < 1e-6f && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > 1e-6f) {
                    float NdotL = Vec3::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay{hit.point, ls.direction};
                        Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN, ls.distance);
                        
                        bool inShadow = shadowHit.hit;
                        
                        if (!inShadow) {
                            Vec3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            float pdfLight = ls.pdf * lightSelectProb;
                            
                            Vec3 contrib = Vec3(
                                f.x * ls.emission.x * NdotL / pdfLight,
                                f.y * ls.emission.y * NdotL / pdfLight,
                                f.z * ls.emission.z * NdotL / pdfLight
                            );
                            result = result + throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < 1e-6f && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float NdotL = std::abs(Vec3::dot(sampleNormal, bsdfSample.wi));
            if (NdotL > 1e-6f && bsdfSample.pdf > 1e-6f) {
                throughput = Vec3(
                    throughput.x * bsdfSample.f.x * NdotL / bsdfSample.pdf,
                    throughput.y * bsdfSample.f.y * NdotL / bsdfSample.pdf,
                    throughput.z * bsdfSample.f.z * NdotL / bsdfSample.pdf
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.x, throughput.y, throughput.z});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x) || std::isinf(throughput.x) ||
            std::isnan(throughput.y) || std::isinf(throughput.y) ||
            std::isnan(throughput.z) || std::isinf(throughput.z)) {
            break;
        }
        
        // Setup next ray
        currentRay.o = hit.point;
        currentRay.d = bsdfSample.wi;
        tMin = geometry::RAY_T_MIN;  // Subsequent rays need offset
    }
    
    return result;
}

// ========== MIS Path Tracer ==========

/**
 * Path tracer with Multiple Importance Sampling
 * Combines BSDF and light sampling with proper MIS weights
 */
inline Vec3 traceMIS(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float lastBsdfPdf = 0.0f;
    float tMin = 0.0f;  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            if (lastBsdfPdf < 1e-6f) {
                result = result + throughput * lightHit.emission;
            } else {
                float misWeight = 0.5f;  // Simplified MIS weight
                result = result + throughput * lightHit.emission * misWeight;
            }
            break;
        }
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        Vec3 emission = getEmission(hit);
        
        // Setup normals
        Vec3 n = hit.normal;
        Vec3 wo = currentRay.d * -1.0f;
        wo.normalize();
        bool frontFace = Vec3::dot(wo, n) > 0;
        
        Vec3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Add emission with MIS weight
        float emissionStrength = emission.x + emission.y + emission.z;
        if (emissionStrength > 1e-6f) {
            float emissionWeight = 1.0f;
            
            if (lastBsdfPdf > 1e-6f && sceneLights.hasLights()) {
                float cosLight = std::abs(Vec3::dot(hit.normal, currentRay.d));
                if (cosLight > 1e-6f) {
                    float lightPdf = (hit.t * hit.t) / (sceneLights.totalArea * cosLight);
                    emissionWeight = powerHeuristic(lastBsdfPdf, lightPdf);
                }
            }
            result = result + throughput * emission * emissionWeight;
        }
        
        // Next Event Estimation with MIS
        bool isEmissive = emissionStrength > 1e-6f;
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && !isEmissive && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > 1e-6f) {
                    float NdotL = Vec3::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay{hit.point, ls.direction};
                        Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN, ls.distance);
                        
                        bool inShadow = shadowHit.hit;
                        
                        if (!inShadow) {
                            Vec3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            float pdfLight = ls.pdf * lightSelectProb;
                            float pdfBsdf = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            float misWeight = powerHeuristic(pdfLight, pdfBsdf);
                            
                            Vec3 contrib = Vec3(
                                f.x * ls.emission.x * NdotL * misWeight / pdfLight,
                                f.y * ls.emission.y * NdotL * misWeight / pdfLight,
                                f.z * ls.emission.z * NdotL * misWeight / pdfLight
                            );
                            result = result + throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < 1e-6f && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float NdotL = std::abs(Vec3::dot(sampleNormal, bsdfSample.wi));
            if (NdotL > 1e-6f && bsdfSample.pdf > 1e-6f) {
                throughput = Vec3(
                    throughput.x * bsdfSample.f.x * NdotL / bsdfSample.pdf,
                    throughput.y * bsdfSample.f.y * NdotL / bsdfSample.pdf,
                    throughput.z * bsdfSample.f.z * NdotL / bsdfSample.pdf
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.x, throughput.y, throughput.z});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x) || std::isinf(throughput.x) ||
            std::isnan(throughput.y) || std::isinf(throughput.y) ||
            std::isnan(throughput.z) || std::isinf(throughput.z)) {
            break;
        }
        
        // Setup next ray
        currentRay.o = hit.point;
        currentRay.d = bsdfSample.wi;
        tMin = geometry::RAY_T_MIN;  // Subsequent rays need offset
        
        // Store BSDF PDF for next emission's MIS weight calculation
        lastBsdfPdf = bsdfSample.useWeight ? 0.0f : bsdfSample.pdf;
    }
    
    return result;
}
