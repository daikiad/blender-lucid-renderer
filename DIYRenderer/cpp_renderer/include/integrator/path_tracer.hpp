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
#include "../units/render_units.hpp"
#include "../math/random.hpp"
#include "../core/scene.hpp"
#include "../core/material.hpp"
#include "../geometry/bvh.hpp"
#include "../bsdf/bsdf.hpp"
#include "../light/light.hpp"
#include "../light/scene_lights.hpp"
#include <cmath>

// Forward declarations for node evaluators
render::ColorRGB getAlbedoFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
render::ColorRGB getEmissionFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getTransmissionFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getIORFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getMetallicFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getRoughnessFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);

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

inline render::RadianceRGB getEmission(const Hit& hit) {
    if (hit.material.useNodes && hit.material.nodeTree.valid) {
        return render::to_radiance(getEmissionFromNodeTree(hit.material.nodeTree, hit.uv));
    }
    return hit.material.emission;
}

// ========== Simple Path Tracer (BSDF only) ==========

/**
 * Simple path tracer using BSDF sampling only (no NEE)
 * Most basic implementation - good for testing
 */
inline render::ColorRGB traceSimple(const Scene& scene, const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ColorRGB throughput(1, 1, 1);
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        // Check if we hit a native light closer than any mesh
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            result += throughput * lightHit.emission;  // ColorRGB * RadianceRGB -> RadianceRGB
            break;
        }
        
        if (!hit.hit) {
            result += throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);
        
        // Add emission
        result += throughput * emission;
        
        // Setup normals
        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo.vec(), n.vec()) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Sample BSDF
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < render::MIN_PDF && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float absNdotL = std::abs(render::dot(sampleNormal.vec(), bsdfSample.wi.vec()));
            if (absNdotL > 1e-6f && bsdfSample.pdf > render::MIN_PDF) {
                render::ColorRGB weight = render::bsdf_sample_weight(bsdfSample.f, absNdotL, bsdfSample.pdf);
                
                const float MAX_WEIGHT = 10.0f;
                weight.r = std::min(weight.r, MAX_WEIGHT);
                weight.g = std::min(weight.g, MAX_WEIGHT);
                weight.b = std::min(weight.b, MAX_WEIGHT);
                
                throughput = throughput * weight;
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.r, throughput.g, throughput.b});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.r) || std::isinf(throughput.r) ||
            std::isnan(throughput.g) || std::isinf(throughput.g) ||
            std::isnan(throughput.b) || std::isinf(throughput.b)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;  // Subsequent rays need offset
    }
    
    return render::to_color(result);  // RadianceRGB -> ColorRGB at interface boundary
}

// ========== NEE Path Tracer ==========

/**
 * Path tracer with Next Event Estimation
 * Uses light sampling for direct illumination
 */
inline render::ColorRGB traceNEE(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ColorRGB throughput(1, 1, 1);
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);  // First ray starts from camera
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            result += throughput * lightHit.emission;
            break;
        }
        
        if (!hit.hit) {
            result += throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);
        
        // Setup normals
        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo.vec(), n.vec()) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Add emission only on first hit
        float emissionStrength = render::to_color(emission).r + render::to_color(emission).g + render::to_color(emission).b;
        if (emissionStrength > 1e-6f && depth == 0) {
            result += throughput * emission;
        }
        
        // Next Event Estimation
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && emissionStrength < 1e-6f && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > render::MIN_PDF) {
                    float NdotL = render::dot(shadingNormal.vec(), ls.direction.vec());
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN_TYPED, ls.distance);
                        
                        bool inShadow = shadowHit.hit;
                        
                        if (!inShadow) {
                            render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            render::PdfW pdfLight = ls.pdf * lightSelectProb;
                            
                            // f [1/sr] * L [W/(sr·m²)] * cosθ / pdf = contrib [W/(sr·m²)]
                            render::ColorRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight);
                            render::RadianceRGB contrib = bsdf_weight * ls.emission;
                            result += throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < render::MIN_PDF && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float NdotL = std::abs(render::dot(sampleNormal.vec(), bsdfSample.wi.vec()));
            if (NdotL > 1e-6f && bsdfSample.pdf > render::MIN_PDF) {
                render::ColorRGB weight = render::bsdf_sample_weight(bsdfSample.f, NdotL, bsdfSample.pdf);
                throughput = throughput * weight;
            } else {
                break;
            }
        }
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.r, throughput.g, throughput.b});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.r) || std::isinf(throughput.r) ||
            std::isnan(throughput.g) || std::isinf(throughput.g) ||
            std::isnan(throughput.b) || std::isinf(throughput.b)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;  // Subsequent rays need offset
    }
    
    return render::to_color(result);
}

// ========== MIS Path Tracer ==========

/**
 * Path tracer with Multiple Importance Sampling
 * Combines BSDF and light sampling with proper MIS weights
 */
inline render::ColorRGB traceMIS(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ColorRGB throughput(1, 1, 1);
    Ray currentRay = ray;
    render::PdfW lastBsdfPdf = render::zero_pdf_w();
    render::Length tMin = render::metres(0.0f);  // First ray starts from camera

    auto findLightIndex = [&](int meshIdx, int triIdx) -> int {
        for (size_t i = 0; i < sceneLights.lights.size(); ++i) {
            const Light& l = sceneLights.lights[i];
            if (l.meshIndex == meshIdx && l.triangleIndex == triIdx) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            render::PdfW lightPdf = render::zero_pdf_w();
            if (sceneLights.hasLights() && lightHit.lightIndex >= 0) {
                float selectProb = sceneLights.getPdfForLight(lightHit.lightIndex);
                if (selectProb > 0.0f) {
                    const Light& l = sceneLights.lights[lightHit.lightIndex];
                    lightPdf = pdfLightSample(l, currentRay.origin, lightHit.point, lightHit.normal) * selectProb;
                }
            }
            render::PdfW bsdfPdf = lastBsdfPdf;
            float misWeight = (bsdfPdf > render::MIN_PDF && lightPdf > render::MIN_PDF)
                                ? render::mis_power_heuristic(lightPdf, bsdfPdf)
                                : 1.0f;
            result += throughput * lightHit.emission * misWeight;
            break;
        }
        
        if (!hit.hit) {
            result += throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);
        
        // Setup normals
        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo.vec(), n.vec()) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Add emission with MIS weight
        render::ColorRGB emission_color = render::to_color(emission);
        float emissionStrength = emission_color.r + emission_color.g + emission_color.b;
        if (emissionStrength > 1e-6f) {
            float misWeight = 1.0f;
            if (sceneLights.hasLights() && lastBsdfPdf > render::MIN_PDF) {
                int lightIdx = findLightIndex(hit.meshIdx, hit.triIdx);
                if (lightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(lightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[lightIdx];
                        render::PdfW lightPdf = pdfLightSample(l, currentRay.origin, hit.point, hit.normal) * selectProb;
                        if (lightPdf > render::MIN_PDF) {
                            misWeight = render::mis_power_heuristic(lightPdf, lastBsdfPdf);
                        }
                    }
                }
            }
            result += throughput * emission * misWeight;
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
                
                if (ls.pdf > render::MIN_PDF) {
                    float NdotL = render::dot(shadingNormal.vec(), ls.direction.vec());
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN_TYPED, ls.distance);
                        
                        bool inShadow = shadowHit.hit;
                        
                        if (!inShadow) {
                            render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            render::PdfW pdfLight_typed = ls.pdf * lightSelectProb;
                            render::PdfW pdfBsdf_typed = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            float misWeight = render::mis_power_heuristic(pdfLight_typed, pdfBsdf_typed);
                            
                            // f [1/sr] * L [W/(sr·m²)] * cosθ * misWeight / pdf
                            render::ColorRGB bsdf_color = render::to_color(f);
                            // Use typed bsdf_sample_weight pattern: ColorRGB = f/pdf * cos
                            render::ColorRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight_typed);
                            render::RadianceRGB contrib = (bsdf_weight * misWeight) * ls.emission;
                            result += throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < render::MIN_PDF && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            throughput = throughput * bsdfSample.weight;
        } else {
            float NdotL = std::abs(render::dot(sampleNormal.vec(), bsdfSample.wi.vec()));
            if (NdotL > 1e-6f && bsdfSample.pdf > render::MIN_PDF) {
                render::ColorRGB weight = render::bsdf_sample_weight(bsdfSample.f, NdotL, bsdfSample.pdf);
                throughput = throughput * weight;
            } else {
                break;
            }
        }
        
        // Store PDF for next bounce MIS (0 if using pre-computed weight)
        lastBsdfPdf = bsdfSample.useWeight ? render::zero_pdf_w() : bsdfSample.pdf;
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = std::max({throughput.r, throughput.g, throughput.b});
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.r) || std::isinf(throughput.r) ||
            std::isnan(throughput.g) || std::isinf(throughput.g) ||
            std::isnan(throughput.b) || std::isinf(throughput.b)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;  // Subsequent rays need offset
        
        // Store BSDF PDF for next emission's MIS weight calculation
        // Note: lastBsdfPdf already set in "Store PDF for next bounce MIS" section
    }
    
    return render::to_color(result);
}
