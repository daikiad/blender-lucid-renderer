/**
 * path_tracer.hpp - Path Tracing Integrators
 * ===========================================
 * 
 * Path tracing implementations with different sampling strategies:
 * - traceSimple: BSDF sampling only (no NEE)
 * - traceNEE: Next Event Estimation (light sampling)
 * - traceMIS: Multiple Importance Sampling (combines both)
 * 
 * Physical Units in Path Tracing:
 * - Radiance (result): [W/(sr·m²)] - what we're computing
 * - Throughput: dimensionless - accumulated BSDF weights
 * - PDF: [1/sr] - probability density in solid angle
 * - Distance (t): [m] - ray intersection distance
 * - Emission: [W/(sr·m²)] - radiance from emitters
 * 
 * The rendering equation:
 *   L_o(x, ω_o) = L_e(x, ω_o) + ∫ f(x, ω_i, ω_o) L_i(x, ω_i) |ω_i · n| dω_i
 *   [W/(sr·m²)]  [W/(sr·m²)]    [1/sr]     [W/(sr·m²)]    [1]    [sr]
 */

#pragma once
#include "../math/vec3_unit.hpp"
#include "../math/random.hpp"
#include "../core/scene.hpp"
#include "../core/material.hpp"
#include "../geometry/bvh.hpp"
#include "../bsdf/bsdf.hpp"
#include "../light/light.hpp"
#include "../light/scene_lights.hpp"
#include "../units/units.hpp"
#include <cmath>

// Forward declarations for node evaluators
diy::Color3 getAlbedoFromNodeTree(const NodeTree& tree, const Vec2& uv);
diy::Color3 getEmissionFromNodeTree(const NodeTree& tree, const Vec2& uv);
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

inline diy::Color3 getEmission(const Hit& hit) {
    if (hit.material.useNodes && hit.material.nodeTree.valid) {
        return getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
    }
    return diy::Color3(
        diy::units::to_radiance(hit.material.emission.x),
        diy::units::to_radiance(hit.material.emission.y),
        diy::units::to_radiance(hit.material.emission.z)
    );
}

// ========== Simple Path Tracer (BSDF only) ==========

/**
 * Simple path tracer using BSDF sampling only (no NEE)
 * Most basic implementation - good for testing
 */
inline diy::Color3 traceSimple(const Scene& scene, const Ray& ray, int maxDepth) {
    diy::Throughput3 result(0, 0, 0);
    diy::Throughput3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float tMin = 0.0f;  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        // Check if we hit a native light closer than any mesh
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            diy::Color3 le(
                diy::units::to_radiance(lightHit.emission.x),
                diy::units::to_radiance(lightHit.emission.y),
                diy::units::to_radiance(lightHit.emission.z)
            );
            result = result + throughput * le;
            break;
        }
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        diy::Color3 emission = getEmission(hit);
        
        // Add emission
        result = result + throughput * emission;
        
        // Setup normals
        diy::Direction3 n = hit.normal;
        diy::Direction3 wo = currentRay.direction * -1.0f;
        wo.normalize();
        bool frontFace = diy::dot(wo, n).numerical_value_in(mp_units::one) > 0;
        
        diy::Direction3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Sample BSDF
        diy::Direction3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        float pdf_raw = diy::units::to_per_sr(bsdfSample.pdf);
        
        if (pdf_raw < 1e-6f && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float absNdotL = std::abs(diy::dot(sampleNormal, bsdfSample.wi).numerical_value_in(mp_units::one));
            if (absNdotL > 1e-6f && pdf_raw > 1e-6f) {
                float weightX = bsdfSample.f.x_raw() * absNdotL / pdf_raw;
                float weightY = bsdfSample.f.y_raw() * absNdotL / pdf_raw;
                float weightZ = bsdfSample.f.z_raw() * absNdotL / pdf_raw;
                
                const float MAX_WEIGHT = 10.0f;
                weightX = std::min(weightX, MAX_WEIGHT);
                weightY = std::min(weightY, MAX_WEIGHT);
                weightZ = std::min(weightZ, MAX_WEIGHT);
                
                throughput = diy::Throughput3(
                    throughput.x_raw() * weightX, 
                    throughput.y_raw() * weightY, 
                    throughput.z_raw() * weightZ
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.x_raw(), throughput.y_raw(), throughput.z_raw()});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x_raw()) || std::isinf(throughput.x_raw()) ||
            std::isnan(throughput.y_raw()) || std::isinf(throughput.y_raw()) ||
            std::isnan(throughput.z_raw()) || std::isinf(throughput.z_raw())) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN;  // Subsequent rays need offset
    }
    
    return diy::Color3(result.x_raw(), result.y_raw(), result.z_raw());
}

// ========== NEE Path Tracer ==========

/**
 * Path tracer with Next Event Estimation
 * Uses light sampling for direct illumination
 */
inline diy::Color3 traceNEE(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    diy::Throughput3 result(0, 0, 0);
    diy::Throughput3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float tMin = 0.0f;  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            diy::Color3 le(
                diy::units::to_radiance(lightHit.emission.x),
                diy::units::to_radiance(lightHit.emission.y),
                diy::units::to_radiance(lightHit.emission.z)
            );
            result = result + throughput * le;
            break;
        }
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        diy::Color3 emission = getEmission(hit);
        
        // Setup normals
        diy::Direction3 n = hit.normal;
        diy::Direction3 wo = currentRay.direction * -1.0f;
        wo.normalize();
        bool frontFace = diy::dot(wo, n).numerical_value_in(mp_units::one) > 0;
        
        diy::Direction3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Add emission only on first hit
        float emissionStrength = emission.x_raw() + emission.y_raw() + emission.z_raw();
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
                
                float ls_pdf = ls.pdf_raw();
                float ls_distance = ls.distance_raw();
                
                if (ls_pdf > 1e-6f) {
                    float NdotL = diy::dot(shadingNormal, ls.direction).numerical_value_in(mp_units::one);
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN, ls_distance);
                        
                        bool inShadow = shadowHit.hit;
                        
                        if (!inShadow) {
                            diy::BSDF3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            float pdfLight = ls_pdf * lightSelectProb;
                            
                            diy::Color3 ls_emission_color(
                                diy::units::to_radiance(ls.emission.x),
                                diy::units::to_radiance(ls.emission.y),
                                diy::units::to_radiance(ls.emission.z)
                            );
                            
                            diy::Throughput3 contrib(
                                f.x_raw() * ls_emission_color.x_raw() * NdotL / pdfLight,
                                f.y_raw() * ls_emission_color.y_raw() * NdotL / pdfLight,
                                f.z_raw() * ls_emission_color.z_raw() * NdotL / pdfLight
                            );
                            result = result + throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        diy::Direction3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        float pdf_raw = diy::units::to_per_sr(bsdfSample.pdf);
        
        if (pdf_raw < 1e-6f && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float NdotL = std::abs(diy::dot(sampleNormal, bsdfSample.wi).numerical_value_in(mp_units::one));
            if (NdotL > 1e-6f && pdf_raw > 1e-6f) {
                throughput = diy::Throughput3(
                    throughput.x_raw() * bsdfSample.f.x_raw() * NdotL / pdf_raw,
                    throughput.y_raw() * bsdfSample.f.y_raw() * NdotL / pdf_raw,
                    throughput.z_raw() * bsdfSample.f.z_raw() * NdotL / pdf_raw
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.x_raw(), throughput.y_raw(), throughput.z_raw()});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x_raw()) || std::isinf(throughput.x_raw()) ||
            std::isnan(throughput.y_raw()) || std::isinf(throughput.y_raw()) ||
            std::isnan(throughput.z_raw()) || std::isinf(throughput.z_raw())) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN;  // Subsequent rays need offset
    }
    
    return diy::Color3(result.x_raw(), result.y_raw(), result.z_raw());
}

// ========== MIS Path Tracer ==========

/**
 * Path tracer with Multiple Importance Sampling
 * Combines BSDF and light sampling with proper MIS weights
 */
inline diy::Color3 traceMIS(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    diy::Throughput3 result(0, 0, 0);
    diy::Throughput3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float lastBsdfPdf = 0.0f;
    float tMin = 0.0f;  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            diy::Color3 lightEmission(
                diy::units::to_radiance(lightHit.emission.x),
                diy::units::to_radiance(lightHit.emission.y),
                diy::units::to_radiance(lightHit.emission.z)
            );
            if (lastBsdfPdf < 1e-6f) {
                result = result + throughput * lightEmission;
            } else {
                float misWeight = 0.5f;  // Simplified MIS weight
                result = result + throughput * lightEmission * misWeight;
            }
            break;
        }
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        diy::Color3 emission = getEmission(hit);
        
        // Setup normals
        diy::Direction3 n = hit.normal;
        diy::Direction3 wo = currentRay.direction * -1.0f;
        wo.normalize();
        bool frontFace = diy::dot(wo, n).numerical_value_in(mp_units::one) > 0;
        
        diy::Direction3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Add emission with MIS weight
        float emissionStrength = emission.x_raw() + emission.y_raw() + emission.z_raw();
        if (emissionStrength > 1e-6f) {
            float emissionWeight = 1.0f;
            
            if (lastBsdfPdf > 1e-6f && sceneLights.hasLights()) {
                float cosLight = std::abs(diy::dot(hit.normal, currentRay.direction).numerical_value_in(mp_units::one));
                if (cosLight > 1e-6f) {
                    float t_raw = hit.t_meters();
                    float totalArea = diy::units::to_square_meters(sceneLights.totalEmissiveArea);
                    float lightPdf = (totalArea > 0.0f) ? (t_raw * t_raw) / (totalArea * cosLight) : 0.0f;
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
                
                float ls_pdf = ls.pdf_raw();
                float ls_distance = ls.distance_raw();
                
                if (ls_pdf > 1e-6f) {
                    float NdotL = diy::dot(shadingNormal, ls.direction).numerical_value_in(mp_units::one);
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN, ls_distance);
                        
                        bool inShadow = shadowHit.hit;
                        
                        if (!inShadow) {
                            diy::BSDF3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            float pdfLight = ls_pdf * lightSelectProb;
                            diy::units::PdfSolidAngle pdfBsdf_typed = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                            float pdfBsdf_raw = diy::units::to_per_sr(pdfBsdf_typed);
                            
                            float misWeight = powerHeuristic(pdfLight, pdfBsdf_raw);
                            
                            diy::Color3 ls_emission_color(
                                diy::units::to_radiance(ls.emission.x),
                                diy::units::to_radiance(ls.emission.y),
                                diy::units::to_radiance(ls.emission.z)
                            );
                            
                            diy::Throughput3 contrib(
                                f.x_raw() * ls_emission_color.x_raw() * NdotL * misWeight / pdfLight,
                                f.y_raw() * ls_emission_color.y_raw() * NdotL * misWeight / pdfLight,
                                f.z_raw() * ls_emission_color.z_raw() * NdotL * misWeight / pdfLight
                            );
                            result = result + throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        diy::Direction3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        float pdf_raw = diy::units::to_per_sr(bsdfSample.pdf);
        
        if (pdf_raw < 1e-6f && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float NdotL = std::abs(diy::dot(sampleNormal, bsdfSample.wi).numerical_value_in(mp_units::one));
            if (NdotL > 1e-6f && pdf_raw > 1e-6f) {
                throughput = diy::Throughput3(
                    throughput.x_raw() * bsdfSample.f.x_raw() * NdotL / pdf_raw,
                    throughput.y_raw() * bsdfSample.f.y_raw() * NdotL / pdf_raw,
                    throughput.z_raw() * bsdfSample.f.z_raw() * NdotL / pdf_raw
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.x_raw(), throughput.y_raw(), throughput.z_raw()});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x_raw()) || std::isinf(throughput.x_raw()) ||
            std::isnan(throughput.y_raw()) || std::isinf(throughput.y_raw()) ||
            std::isnan(throughput.z_raw()) || std::isinf(throughput.z_raw())) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN;  // Subsequent rays need offset
        
        // Store BSDF PDF for next emission's MIS weight calculation
        lastBsdfPdf = bsdfSample.useWeight ? 0.0f : pdf_raw;
    }
    
    return diy::Color3(result.x_raw(), result.y_raw(), result.z_raw());
}
