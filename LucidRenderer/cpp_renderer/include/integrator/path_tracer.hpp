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
#include "../volume/phase.hpp"
#include "../volume/transmittance.hpp"
#include <cmath>

// Forward declarations for node evaluators
render::AttenuationRGB getAlbedoFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
render::RGB3f getEmissionFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getTransmissionFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getIORFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getMetallicFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);
float getRoughnessFromNodeTree(const NodeTree& tree, const render::Vec2f& uv);

// ========== Helper Functions ==========

inline MaterialParams getMaterialParams(const Hit& hit) {
    MaterialParams mat;
    mat.albedo = hit.material->albedo;
    mat.metallic = hit.material->metallic;
    mat.roughness = hit.material->roughness;
    mat.transmission = hit.material->transmission;
    mat.ior = hit.material->ior;
    
    if (hit.material->useNodes && hit.material->nodeTree.valid) {
        mat.albedo = getAlbedoFromNodeTree(hit.material->nodeTree, hit.uv);
        mat.metallic = getMetallicFromNodeTree(hit.material->nodeTree, hit.uv);
        mat.roughness = getRoughnessFromNodeTree(hit.material->nodeTree, hit.uv);
        mat.transmission = getTransmissionFromNodeTree(hit.material->nodeTree, hit.uv);
        mat.ior = getIORFromNodeTree(hit.material->nodeTree, hit.uv);
    }
    
    // Enforce minimum roughness
    mat.roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    return mat;
}

inline render::RadianceRGB getEmission(const Hit& hit) {
    if (hit.material->useNodes && hit.material->nodeTree.valid) {
        return render::to_radiance(getEmissionFromNodeTree(hit.material->nodeTree, hit.uv));
    }
    return hit.material->emission;
}

// Find the next *real* surface and native-light hit along `ray`, ignoring
// `volume_boundary_only` meshes (Blender smoke-domain cubes etc.). Returns
// the closer of {real surface, native light} via the `hit` / `lightHit`
// out-params; the returned `t` values are in the **original ray's**
// parameter frame, which is critical for downstream `sample_volume_distance`
// — it slabs the ray against the volume bbox using the original origin, so
// advancing the ray past boundary hits would skip the smoke domain bbox and
// no scatter could ever fire on camera rays.
inline void intersect_skipping_volume_boundaries(
    const Scene& scene, const Ray& ray, render::Length tMin, int depth,
    Hit& hit, LightHit& lightHit) {
    hit.hit = false;
    render::Length search_min = tMin;
    for (int peek = 0; peek < 8; ++peek) {
        Hit h = intersectScene(scene, ray, search_min);
        if (!h.hit) break;
        if (!h.material->volume_boundary_only) { hit = h; break; }
        // Boundary hit — skip past it and keep searching for a real surface.
        search_min = h.t + render::metres(1e-3f);
    }
    lightHit = intersectNativeLights(scene, ray, depth);
}

// Shadow ray test that treats `volume_boundary_only` meshes as transparent.
// Returns true if a *real* surface (not just a volume-boundary mesh) occludes
// the segment [0, t_max] from `ray.origin` along `ray.direction`.
// Used for NEE — without this, a scatter inside the smoke domain always
// reports "shadowed" because the boundary cube blocks the shadow ray on the
// way out, leaving the medium permanently unlit.
inline bool is_shadowed_skipping_volume_boundaries(
    const Scene& scene, const Ray& ray, render::Length t_max) {
    Ray probe = ray;
    float t_offset = 0.0f;
    const float t_max_f = t_max.numerical_value_in(render::si::metre);
    for (int peek = 0; peek < 8; ++peek) {
        const float remaining = t_max_f - t_offset;
        if (remaining <= 0.0f) return false;
        Hit h = intersectScene(scene, probe, geometry::RAY_T_MIN_TYPED, render::metres(remaining));
        if (!h.hit) return false;
        if (!h.material->volume_boundary_only) return true;
        const float skip = h.t.numerical_value_in(render::si::metre) + 1e-3f;
        t_offset += skip;
        probe = Ray(probe.at(render::metres(skip)), probe.direction);
    }
    return false;
}

// ========== Simple Path Tracer (BSDF only) ==========

/**
 * Simple path tracer using BSDF sampling only (no NEE)
 * Most basic implementation - good for testing
 * Returns RadianceRGB - caller applies camera sensitivity for final pixel value
 */
inline render::RadianceRGB traceSimple(const Scene& scene, const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);  // First ray starts from camera

    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit;
        LightHit lightHit;
        intersect_skipping_volume_boundaries(scene, currentRay, tMin, depth, hit, lightHit);

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
        bool frontFace = render::dot(wo, n) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Sample BSDF
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (!bsdfSample.isValid()) {
            break;
        }

        // Update throughput via variant-aware helper
        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (!render::throughput_is_valid(throughput)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;  // Subsequent rays need offset
    }
    
    return result;  // Return RadianceRGB, let caller apply camera sensitivity
}

// ========== Simple Path Tracer + Volume Absorption (Stage A) ==========
//
// Identical to traceSimple, but applies Beer-Lambert volume attenuation to
// `throughput` between each pair of events (camera → surface, or surface →
// next surface). No scattering — the medium can only absorb, never redirect.
//
// Educational copy: keeping it parallel to traceSimple makes the "what does
// volume rendering add?" delta obvious in diff. Once you understand the
// pattern, we'll merge with traceSimple (or just replace it).
inline render::RadianceRGB traceVolumeSimple(const Scene& scene, const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);

    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit;
        LightHit lightHit;
        intersect_skipping_volume_boundaries(scene, currentRay, tMin, depth, hit, lightHit);

        // ------------------------------------------------------------------
        // Volume absorption along the segment from currentRay.origin to the
        // closer of {hit.t, lightHit.t}. On env miss neither hits, so a large
        // fallback works — the volume bbox clips the integral naturally.
        // T = exp(-∫σ_t ds) per channel; throughput *= T.
        // ------------------------------------------------------------------
        render::Length t_seg = render::metres(1e6f);
        if (hit.hit)      t_seg = std::min(t_seg, hit.t);
        if (lightHit.hit) t_seg = std::min(t_seg, lightHit.t);
        const auto T_seg = transmittance_along_segment(scene, currentRay, t_seg);
        throughput = throughput * T_seg;

        // Check if we hit a native light closer than any mesh
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

        result += throughput * emission;

        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo, n) > 0;

        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }

        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        if (!bsdfSample.isValid()) {
            break;
        }

        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;

        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }

        if (!render::throughput_is_valid(throughput)) {
            break;
        }

        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;
    }

    return result;
}

// ========== NEE Path Tracer ==========

/**
 * Path tracer with Next Event Estimation
 * Uses light sampling for direct illumination
 * Returns RadianceRGB - caller applies camera sensitivity for final pixel value
 */
inline render::RadianceRGB traceNEE(const Scene& scene, const SceneLights& sceneLights,
                     const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);  // First ray starts from camera

    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit;
        LightHit lightHit;
        intersect_skipping_volume_boundaries(scene, currentRay, tMin, depth, hit, lightHit);

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
        bool frontFace = render::dot(wo, n) > 0;

        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }

        // Add emission only on first hit
        // Use is_emissive for checking emission strength (unit-typed)
        bool hasEmission = render::is_emissive(emission);
        if (hasEmission && depth == 0) {
            result += throughput * emission;
        }
        
        // Next Event Estimation
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && !hasEmission && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > render::MIN_PDF) {
                    float NdotL = render::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        bool inShadow = is_shadowed_skipping_volume_boundaries(scene, shadowRay, ls.distance);

                        if (!inShadow) {
                            render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            render::PdfW pdfLight = ls.pdf * lightSelectProb;
                            
                            // f [1/sr] * L [W/(sr·m²)] * cosθ / pdf = contrib [W/(sr·m²)]
                            render::ThroughputRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight);
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
        
        if (!bsdfSample.isValid()) {
            break;
        }

        // Update throughput via variant-aware helper
        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (!render::throughput_is_valid(throughput)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;  // Subsequent rays need offset
    }
    
    return result;  // Return RadianceRGB, let caller apply camera sensitivity
}

// ========== MIS Path Tracer ==========

/**
 * Path tracer with Multiple Importance Sampling
 * Combines BSDF and light sampling with proper MIS weights
 * Returns RadianceRGB - caller applies camera sensitivity for final pixel value
 */
inline render::RadianceRGB traceMIS(const Scene& scene, const SceneLights& sceneLights,
                     const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
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
        Hit hit;
        LightHit lightHit;
        intersect_skipping_volume_boundaries(scene, currentRay, tMin, depth, hit, lightHit);

        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            render::PdfW lightPdf = render::zero_pdf_w();
            if (sceneLights.hasLights() && lightHit.lightIndex >= 0) {
                // lightHit.lightIndex is index into scene.nativeLights, not sceneLights.lights
                // Convert to sceneLights.lights index
                int sceneLightIdx = sceneLights.findNativeLightIndex(lightHit.lightIndex);
                if (sceneLightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(sceneLightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[sceneLightIdx];
                        lightPdf = pdfLightSample(l, currentRay.origin, lightHit.point, lightHit.normal) * selectProb;
                    }
                }
            }
            render::PdfW bsdfPdf = lastBsdfPdf;
            // MIS weight for BSDF sampling strategy hitting a light
            // mis_power_heuristic(pf, pg) returns weight for strategy f
            // Here we sampled via BSDF, so pf = bsdfPdf
            render::Dimensionless misWeight = (bsdfPdf > render::MIN_PDF && lightPdf > render::MIN_PDF)
                                ? render::mis_power_heuristic(bsdfPdf, lightPdf)
                                : render::Dimensionless{1.0f};
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
        bool frontFace = render::dot(wo, n) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Add emission with MIS weight
        bool hasEmission = render::is_emissive(emission);
        if (hasEmission) {
            render::Dimensionless misWeight{1.0f};
            if (sceneLights.hasLights() && lastBsdfPdf > render::MIN_PDF) {
                int lightIdx = findLightIndex(hit.meshIdx, hit.triIdx);
                if (lightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(lightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[lightIdx];
                        render::PdfW lightPdf = pdfLightSample(l, currentRay.origin, hit.point, hit.normal) * selectProb;
                        if (lightPdf > render::MIN_PDF) {
                            // MIS weight for BSDF sampling strategy hitting emissive surface
                            // mis_power_heuristic(pf, pg) returns weight for strategy f
                            // Here we sampled via BSDF, so pf = lastBsdfPdf
                            misWeight = render::mis_power_heuristic(lastBsdfPdf, lightPdf);
                        }
                    }
                }
            }
            result += throughput * emission * misWeight;
        }
        
        // Next Event Estimation with MIS
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && !hasEmission && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > render::MIN_PDF) {
                    float NdotL = render::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        bool inShadow = is_shadowed_skipping_volume_boundaries(scene, shadowRay, ls.distance);

                        if (!inShadow) {
                            render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            render::PdfW pdfLight_typed = ls.pdf * lightSelectProb;
                            render::PdfW pdfBsdf_typed = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            render::Dimensionless misWeight = render::mis_power_heuristic(pdfLight_typed, pdfBsdf_typed);
                            
                            // f [1/sr] * L [W/(sr·m²)] * cosθ * misWeight / pdf
                            // Use typed bsdf_sample_weight pattern: ThroughputRGB = f/pdf * cos
                            render::ThroughputRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight_typed);
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
        
        if (!bsdfSample.isValid()) {
            break;
        }

        // Update throughput via variant-aware helper
        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;
        
        // Store PDF for next bounce MIS — PrecomputedWeight samples return zero
        // (delta-like, light-sampling PDF unreliable), making next-bounce MIS weight = 1.0.
        lastBsdfPdf = mis_pdf_for_next_bounce(bsdfSample);
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (!render::throughput_is_valid(throughput)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;  // Subsequent rays need offset
        
        // Store BSDF PDF for next emission's MIS weight calculation
        // Note: lastBsdfPdf already set in "Store PDF for next bounce MIS" section
    }
    
    return result;  // Return RadianceRGB, let caller apply camera sensitivity
}

// ========== MIS Path Tracer + Volume Scattering ==========
//
// `traceVolumeMIS` is the full volume path tracer. Same MIS structure as
// `traceMIS`, but each bounce starts with a delta-tracking distance sample
// inside the medium. If a scatter event is drawn before the next surface,
// we redirect via the Henyey-Greenstein phase function (using the
// material's `anisotropy` field) and do in-medium NEE with MIS between
// the light sampler and the phase function. Otherwise we apply
// extinction-transmittance along the surface segment and shade the surface
// exactly like `traceMIS`.
//
// `lastBsdfPdf` is overloaded: it carries the PDF of whichever sampler
// chose the last direction — BSDF on surface bounces, HG phase pdf on
// volume bounces. The MIS weight at the next emission hit uses it as `pf`.
inline render::RadianceRGB traceVolumeMIS(const Scene& scene, const SceneLights& sceneLights,
                          const Ray& ray, int maxDepth) {
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::PdfW lastBsdfPdf = render::zero_pdf_w();
    render::Length tMin = render::metres(0.0f);

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
        Hit hit;
        LightHit lightHit;
        intersect_skipping_volume_boundaries(scene, currentRay, tMin, depth, hit, lightHit);

        // Segment end: next surface or native light (or "far away" if neither).
        render::Length t_surface = render::metres(1e6f);
        if (hit.hit)      t_surface = std::min(t_surface, hit.t);
        if (lightHit.hit) t_surface = std::min(t_surface, lightHit.t);

        // ---- 1. Distance sample inside the volumes along [0, t_surface] ----
        VolumeScatterEvent vs = sample_volume_distance(scene, currentRay, t_surface);

        if (vs.happened && vs.vol != nullptr) {
            // ===== 2a. Volume scatter event =====
            // Per-channel single-scattering albedo (Cycles convention):
            //   sigma_total = density * grid_sample
            //   sigma_s,c   = sigma_total * color_c
            //   sigma_t,c   = sigma_total * (1 + absorption_color_c)
            //   albedo_c    = sigma_s,c / sigma_t,c = color_c / (1 + absorption_color_c)
            // With abs=0 this reduces to throughput *= color, so the medium's
            // Color directly tints the scattered light. The Beer-Lambert factor
            // along [0, vs.t] is already implicit in the delta-tracking
            // sampling pdf; do NOT multiply by an explicit transmittance here.
            {
                const auto& v = *vs.vol;
                const float a_r = v.color.r / (1.0f + v.absorption_color.r);
                const float a_g = v.color.g / (1.0f + v.absorption_color.g);
                const float a_b = v.color.b / (1.0f + v.absorption_color.b);
                throughput = throughput * render::make_attenuation_rgb(a_r, a_g, a_b);
            }

            const render::Position x = currentRay.at(vs.t);
            const render::Direction wo = -currentRay.direction;
            const float g = vs.vol->anisotropy;

            // ---- 2a.i. In-medium NEE with MIS (phase vs light) ----
            if (sceneLights.hasLights()) {
                float lightSelectProb;
                int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);

                if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                    const Light& light = sceneLights.lights[lightIdx];
                    LightSample ls = sampleLight(light, x, randf(), randf());

                    if (ls.pdf > render::MIN_PDF) {
                        Ray shadowRay(x, ls.direction);
                        bool inShadow = is_shadowed_skipping_volume_boundaries(scene, shadowRay, ls.distance);

                        if (!inShadow) {
                            // Attenuate light through any intervening media.
                            const auto T_shadow = extinction_transmittance_along_segment(scene, shadowRay, ls.distance);

                            const float cos_theta = render::dot(wo, ls.direction);
                            const float phase_eval = volume::hg_eval(g, cos_theta);
                            const float phase_pdf  = volume::hg_pdf (g, cos_theta);
                            // Wrap as PdfW for the typed mis_power_heuristic.
                            const render::PdfW phase_pdf_w = phase_pdf * render::per_sr;
                            const render::PdfW pdfLight_typed = ls.pdf * lightSelectProb;

                            const render::Dimensionless misWeight =
                                render::mis_power_heuristic(pdfLight_typed, phase_pdf_w);

                            // f/pdf as a dimensionless weight; phase function has no cos
                            // factor (it's already normalized over the sphere, not the
                            // hemisphere) so we pass abs_cos_theta=1.
                            const render::BSDFRGB phase_f = render::make_bsdf_rgb(phase_eval, phase_eval, phase_eval);
                            const render::ThroughputRGB phase_weight =
                                render::bsdf_sample_weight(phase_f, 1.0f, pdfLight_typed);

                            const render::RadianceRGB contrib =
                                (phase_weight * misWeight) * (T_shadow * ls.emission);
                            result += throughput * contrib;
                        }
                    }
                }
            }

            // ---- 2a.ii. Sample phase function for next bounce ----
            auto [wi, phase_pdf] = volume::hg_sample(g, wo, randf(), randf());
            // HG: phase_eval / phase_pdf == 1, so throughput already updated above.
            lastBsdfPdf = phase_pdf * render::per_sr;
            currentRay  = Ray(x, wi);
            // Inside the medium, no geometry offset; epsilon-step out so we
            // don't pin-prick at exactly vs.t (matches surface convention).
            tMin = geometry::RAY_T_MIN_TYPED;

            // Russian Roulette (treat scatter as a bounce)
            if (depth >= 3) {
                float maxThroughput = render::throughput_max_component(throughput);
                float rrProb = std::min(maxThroughput, 0.95f);
                if (randf() > rrProb) break;
                throughput = throughput * (1.0f / rrProb);
            }
            if (!render::throughput_is_valid(throughput)) break;
            continue;
        }

        // ===== 2b. Surface event =====
        // No scatter happened in [0, t_surface]. With delta tracking, the
        // probability of "no real collision before t_surface" is exactly
        // T(t_surface), so the estimator's T/pdf ratio cancels to 1.
        // Do NOT multiply by extinction transmittance here.

        // ---- Native light hit (BSDF strategy hit a light directly) ----
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            render::PdfW lightPdf = render::zero_pdf_w();
            if (sceneLights.hasLights() && lightHit.lightIndex >= 0) {
                int sceneLightIdx = sceneLights.findNativeLightIndex(lightHit.lightIndex);
                if (sceneLightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(sceneLightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[sceneLightIdx];
                        lightPdf = pdfLightSample(l, currentRay.origin, lightHit.point, lightHit.normal) * selectProb;
                    }
                }
            }
            render::PdfW bsdfPdf = lastBsdfPdf;
            render::Dimensionless misWeight = (bsdfPdf > render::MIN_PDF && lightPdf > render::MIN_PDF)
                                ? render::mis_power_heuristic(bsdfPdf, lightPdf)
                                : render::Dimensionless{1.0f};
            result += throughput * lightHit.emission * misWeight;
            break;
        }

        if (!hit.hit) {
            result += throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
            break;
        }

        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);

        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo, n) > 0;

        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }

        // ---- Emissive surface w/ MIS weight ----
        bool hasEmission = render::is_emissive(emission);
        if (hasEmission) {
            render::Dimensionless misWeight{1.0f};
            if (sceneLights.hasLights() && lastBsdfPdf > render::MIN_PDF) {
                int lightIdx = findLightIndex(hit.meshIdx, hit.triIdx);
                if (lightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(lightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[lightIdx];
                        render::PdfW lightPdf = pdfLightSample(l, currentRay.origin, hit.point, hit.normal) * selectProb;
                        if (lightPdf > render::MIN_PDF) {
                            misWeight = render::mis_power_heuristic(lastBsdfPdf, lightPdf);
                        }
                    }
                }
            }
            result += throughput * emission * misWeight;
        }

        // ---- Surface NEE with MIS (same as traceMIS) ----
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && !hasEmission && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);

            if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());

                if (ls.pdf > render::MIN_PDF) {
                    float NdotL = render::dot(shadingNormal, ls.direction);

                    if (NdotL > 1e-6f) {
                        Ray shadowRay(hit.point, ls.direction);
                        bool inShadow = is_shadowed_skipping_volume_boundaries(scene, shadowRay, ls.distance);

                        if (!inShadow) {
                            // Attenuate through volumes between surface and light.
                            const auto T_shadow = extinction_transmittance_along_segment(scene, shadowRay, ls.distance);

                            render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);

                            render::PdfW pdfLight_typed = ls.pdf * lightSelectProb;
                            render::PdfW pdfBsdf_typed  = pdfBSDF(mat, wo, ls.direction, shadingNormal);

                            render::Dimensionless misWeight =
                                render::mis_power_heuristic(pdfLight_typed, pdfBsdf_typed);

                            render::ThroughputRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight_typed);
                            render::RadianceRGB contrib = (bsdf_weight * misWeight) * (T_shadow * ls.emission);
                            result += throughput * contrib;
                        }
                    }
                }
            }
        }

        // ---- BSDF sampling for next bounce ----
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());

        if (!bsdfSample.isValid()) {
            break;
        }

        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;

        lastBsdfPdf = mis_pdf_for_next_bounce(bsdfSample);

        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }

        if (!render::throughput_is_valid(throughput)) {
            break;
        }

        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;
    }

    return result;
}

// ========== Diagnostic Path Tracer Variants ==========
// These variants record path data for variance analysis

#include "diagnostics/diagnostic_integrator.hpp"

// Helper to convert Position to Vec3f for diagnostics
inline render::Vec3f position_to_vec3f(const render::Position& pos) {
    render::Displacement disp = render::displacement_from_origin(pos);
    return disp.numerical_value_in(render::si::metre);
}

// Helper to convert Normal to Vec3f for diagnostics
inline render::Vec3f normal_to_vec3f(const render::Normal& n) {
    return n.vec();
}

// Helper to calculate ray hit point position as Vec3f
inline render::Vec3f ray_hit_position(const Ray& ray, render::Length t) {
    render::Displacement origin_disp = render::displacement_from_origin(ray.origin);
    render::Vec3f origin_vec = origin_disp.numerical_value_in(render::si::metre);
    float t_val = t.numerical_value_in(render::si::metre);
    return render::Vec3f{
        origin_vec.x + ray.direction.vec().x * t_val,
        origin_vec.y + ray.direction.vec().y * t_val,
        origin_vec.z + ray.direction.vec().z * t_val
    };
}

// Helper to calculate far environment hit position
inline render::Vec3f env_hit_position(const Ray& ray, float distance = 1000.0f) {
    render::Displacement origin_disp = render::displacement_from_origin(ray.origin);
    render::Vec3f origin_vec = origin_disp.numerical_value_in(render::si::metre);
    return render::Vec3f{
        origin_vec.x + ray.direction.vec().x * distance,
        origin_vec.y + ray.direction.vec().y * distance,
        origin_vec.z + ray.direction.vec().z * distance
    };
}

/**
 * Simple path tracer with diagnostic recording
 * Records each vertex for path analysis
 */
inline render::RadianceRGB traceSimpleWithDiagnostics(
    const Scene& scene, 
    const Ray& ray, 
    int maxDepth,
    render::diagnostics::PathDiagnosticRecorder& recorder) 
{
    using namespace render::diagnostics;
    
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        // Check if we hit a native light closer than any mesh
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            // Get light type for Heckbert notation
            LightSourceType lightSourceType = LightSourceType::Unknown;
            if (lightHit.lightIndex >= 0 && 
                lightHit.lightIndex < static_cast<int>(scene.nativeLights.size())) {
                lightSourceType = to_light_source_type(
                    static_cast<int>(scene.nativeLights[lightHit.lightIndex].type));
            }
            
            // Calculate light position for visualization
            render::Vec3f lightPos = ray_hit_position(currentRay, lightHit.t);
            
            // Calculate contribution
            render::RadianceRGB pathContrib = throughput * lightHit.emission;
            render::RGB3f contrib = render::RGB3f(
                pathContrib.r.numerical_value_in(render::radiance_unit),
                pathContrib.g.numerical_value_in(render::radiance_unit),
                pathContrib.b.numerical_value_in(render::radiance_unit)
            );
            
            // Plan E: Record as completed path with BSDF strategy and geometry
            recorder.record_completed_path_with_geometry(SamplingStrategy::BSDF, 
                                           lightHit.lightIndex, 
                                           lightSourceType, 
                                           contrib,
                                           lightPos);
            
            result += pathContrib;
            break;
        }
        
        if (!hit.hit) {
            // Calculate environment contribution
            render::RadianceRGB envContrib = throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
            render::RGB3f contrib = render::RGB3f(
                envContrib.r.numerical_value_in(render::radiance_unit),
                envContrib.g.numerical_value_in(render::radiance_unit),
                envContrib.b.numerical_value_in(render::radiance_unit)
            );
            
            // Calculate far point for environment visualization
            render::Vec3f envPos = env_hit_position(currentRay);
            
            // Plan E: Record as completed path hitting environment with geometry
            recorder.record_completed_path_with_geometry(SamplingStrategy::BSDF,
                                           -1,  // No light index for environment
                                           LightSourceType::Environment,
                                           contrib,
                                           envPos);
            
            result += envContrib;
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);
        
        // Record vertex with geometry for path visualization
        BsdfType bsdfType = classify_bsdf(mat);
        bool isDelta = is_delta_bsdf(mat);
        
        // Convert hit position and normal to Vec3f for diagnostics
        render::Vec3f hitPos = position_to_vec3f(hit.point);
        render::Vec3f hitNorm = normal_to_vec3f(hit.normal);
        
        if (render::is_emissive(emission)) {
            // Emissive mesh hit - this is a completed path!
            render::RadianceRGB emissionContrib = throughput * emission;
            render::RGB3f contrib = render::RGB3f(
                emissionContrib.r.numerical_value_in(render::radiance_unit),
                emissionContrib.g.numerical_value_in(render::radiance_unit),
                emissionContrib.b.numerical_value_in(render::radiance_unit)
            );
            
            // Record emissive hit as completed path with geometry
            recorder.record_completed_path_with_geometry(SamplingStrategy::BSDF,
                                           hit.meshIdx,  // Use mesh index for emissive
                                           LightSourceType::Emissive,
                                           contrib,
                                           hitPos);
            
            result += emissionContrib;
            break;  // Path terminates at emissive surface
        } else {
            recorder.record_vertex_with_geometry(hit.meshIdx, 0, 
                                   bsdfType, isDelta, false,
                                   hitPos, hitNorm);
        }
        
        // Setup normals
        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo, n) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Sample BSDF
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (!bsdfSample.isValid()) {
            break;
        }

        // Update throughput via variant-aware helper
        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (!render::throughput_is_valid(throughput)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;
    }
    
    return result;
}

/**
 * NEE path tracer with diagnostic recording
 */
inline render::RadianceRGB traceNEEWithDiagnostics(
    const Scene& scene, 
    const SceneLights& sceneLights, 
    const Ray& ray, 
    int maxDepth,
    render::diagnostics::PathDiagnosticRecorder& recorder) 
{
    using namespace render::diagnostics;
    
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::Length tMin = render::metres(0.0f);
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, tMin);
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        if (lightHit.hit && (!hit.hit || lightHit.t < hit.t)) {
            // Direct light hit (depth == 0 only in pure NEE)
            // After first bounce, NEE relies on direct light sampling, not BSDF hits
            if (depth == 0) {
                LightSourceType lightSourceType = LightSourceType::Unknown;
                if (lightHit.lightIndex >= 0 && 
                    lightHit.lightIndex < static_cast<int>(scene.nativeLights.size())) {
                    lightSourceType = to_light_source_type(
                        static_cast<int>(scene.nativeLights[lightHit.lightIndex].type));
                }
                
                // Calculate light position for visualization
                render::Vec3f lightPos = ray_hit_position(currentRay, lightHit.t);
                
                render::RadianceRGB pathContrib = throughput * lightHit.emission;
                render::RGB3f contrib = render::RGB3f(
                    pathContrib.r.numerical_value_in(render::radiance_unit),
                    pathContrib.g.numerical_value_in(render::radiance_unit),
                    pathContrib.b.numerical_value_in(render::radiance_unit)
                );
                
                // Direct view of light uses BSDF strategy (camera ray hit light)
                recorder.record_completed_path_with_geometry(SamplingStrategy::BSDF,
                                               lightHit.lightIndex,
                                               lightSourceType,
                                               contrib,
                                               lightPos);
                
                result += pathContrib;
            }
            // Ignore BSDF light hits after first bounce in pure NEE
            break;
        }
        
        if (!hit.hit) {
            // Environment hit
            if (depth == 0) {
                // Direct view of environment uses BSDF strategy
                render::RadianceRGB envContrib = throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
                render::RGB3f contrib = render::RGB3f(
                    envContrib.r.numerical_value_in(render::radiance_unit),
                    envContrib.g.numerical_value_in(render::radiance_unit),
                    envContrib.b.numerical_value_in(render::radiance_unit)
                );
                
                // Calculate far point for environment visualization
                render::Vec3f envPos = env_hit_position(currentRay);
                
                recorder.record_completed_path_with_geometry(SamplingStrategy::BSDF,
                                               -1,
                                               LightSourceType::Environment,
                                               contrib,
                                               envPos);
                
                result += envContrib;
            }
            // Ignore environment hits after first bounce in pure NEE
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);
        
        // Record vertex
        BsdfType bsdfType = classify_bsdf(mat);
        bool isDelta = is_delta_bsdf(mat);
        bool hasEmission = render::is_emissive(emission);
        
        // Setup normals
        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo, n) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Add emission only on first hit (direct view of emissive)
        if (hasEmission && depth == 0) {
            // Calculate emissive hit position for visualization
            render::Vec3f emissivePos = position_to_vec3f(hit.point);
            
            render::RadianceRGB emissionContrib = throughput * emission;
            render::RGB3f contrib = render::RGB3f(
                emissionContrib.r.numerical_value_in(render::radiance_unit),
                emissionContrib.g.numerical_value_in(render::radiance_unit),
                emissionContrib.b.numerical_value_in(render::radiance_unit)
            );
            // Direct view uses BSDF strategy (camera ray hit emissive)
            recorder.record_completed_path_with_geometry(SamplingStrategy::BSDF,
                                           hit.meshIdx,
                                           LightSourceType::Emissive,
                                           contrib,
                                           emissivePos);
            result += emissionContrib;
        } else if (!hasEmission) {
            // Convert hit position and normal to Vec3f for diagnostics
            render::Vec3f hitPos = position_to_vec3f(hit.point);
            render::Vec3f hitNorm = normal_to_vec3f(hit.normal);
            
            // Record vertex first (before NEE) with geometry
            recorder.record_vertex_with_geometry(hit.meshIdx, 0, 
                                   bsdfType, isDelta, false,
                                   hitPos, hitNorm);
            
            // Next Event Estimation
            bool isTransmissive = mat.transmission > 0.5f;
            if (sceneLights.hasLights() && !isTransmissive) {
                float lightSelectProb;
                int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
                
                if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                    const Light& light = sceneLights.lights[lightIdx];
                    LightSample ls = sampleLight(light, hit.point, randf(), randf());
                    
                    if (ls.pdf > render::MIN_PDF) {
                        float NdotL = render::dot(shadingNormal, ls.direction);
                        
                        if (NdotL > 1e-6f) {
                            Ray shadowRay(hit.point, ls.direction);
                            Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN_TYPED, ls.distance);
                            
                            bool inShadow = shadowHit.hit;
                            
                            if (!inShadow) {
                                render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                                render::PdfW pdfLight = ls.pdf * lightSelectProb;
                                
                                render::ThroughputRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight);
                                render::RadianceRGB nee_radiance = bsdf_weight * ls.emission;
                                
                                // Record NEE contribution as completed path
                                int nativeLightIdx = lightIdx - sceneLights.nativeLightStartIndex;
                                LightSourceType lightSourceType = to_light_source_type(
                                    static_cast<int>(light.type));
                                
                                // Calculate light position for visualization
                                render::Displacement lightDisp = render::displacement_from_origin(ls.position);
                                render::Vec3f lightPos = lightDisp.numerical_value_in(render::si::metre);
                                
                                // NEE contribution = throughput * BSDF * emission / pdf
                                render::RadianceRGB total_nee = throughput * nee_radiance;
                                render::RGB3f nee_contrib(
                                    total_nee.r.numerical_value_in(render::radiance_unit),
                                    total_nee.g.numerical_value_in(render::radiance_unit),
                                    total_nee.b.numerical_value_in(render::radiance_unit)
                                );
                                
                                // Plan E: Use NEE strategy with geometry
                                recorder.record_completed_path_with_geometry(SamplingStrategy::NEE,
                                                               nativeLightIdx,
                                                               lightSourceType,
                                                               nee_contrib,
                                                               lightPos);
                                
                                result += throughput * nee_radiance;
                            }
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (!bsdfSample.isValid()) {
            break;
        }

        // Update throughput via variant-aware helper
        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (!render::throughput_is_valid(throughput)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;
    }
    
    return result;
}

/**
 * MIS path tracer with diagnostic recording
 */
inline render::RadianceRGB traceMISWithDiagnostics(
    const Scene& scene, 
    const SceneLights& sceneLights, 
    const Ray& ray, 
    int maxDepth,
    render::diagnostics::PathDiagnosticRecorder& recorder) 
{
    using namespace render::diagnostics;
    
    render::RadianceRGB result = render::zero_radiance_rgb();
    render::ThroughputRGB throughput = render::unit_throughput_rgb();
    Ray currentRay = ray;
    render::PdfW lastBsdfPdf = render::zero_pdf_w();
    render::Length tMin = render::metres(0.0f);

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
            // BSDF sampling hit a native light
            LightSourceType lightSourceType = LightSourceType::Unknown;
            if (lightHit.lightIndex >= 0 && 
                lightHit.lightIndex < static_cast<int>(scene.nativeLights.size())) {
                lightSourceType = to_light_source_type(
                    static_cast<int>(scene.nativeLights[lightHit.lightIndex].type));
            }
            
            // Calculate light position for visualization
            // Calculate light position for visualization
            render::Vec3f lightPos = ray_hit_position(currentRay, lightHit.t);
            
            // Calculate MIS weight for BSDF sampling
            render::PdfW lightPdf = render::zero_pdf_w();
            if (sceneLights.hasLights() && lightHit.lightIndex >= 0) {
                int sceneLightIdx = sceneLights.findNativeLightIndex(lightHit.lightIndex);
                if (sceneLightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(sceneLightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[sceneLightIdx];
                        lightPdf = pdfLightSample(l, currentRay.origin, lightHit.point, lightHit.normal) * selectProb;
                    }
                }
            }
            render::PdfW bsdfPdf = lastBsdfPdf;
            render::Dimensionless misWeight = (bsdfPdf > render::MIN_PDF && lightPdf > render::MIN_PDF)
                                ? render::mis_power_heuristic(bsdfPdf, lightPdf)
                                : render::Dimensionless{1.0f};
            
            // Calculate MIS-weighted contribution
            render::RadianceRGB pathContrib = throughput * lightHit.emission * misWeight;
            render::RGB3f contrib = render::RGB3f(
                pathContrib.r.numerical_value_in(render::radiance_unit),
                pathContrib.g.numerical_value_in(render::radiance_unit),
                pathContrib.b.numerical_value_in(render::radiance_unit)
            );
            
            // Plan E: Record as MIS_BSDF with geometry
            recorder.record_completed_path_with_geometry(SamplingStrategy::MIS_BSDF,
                                           lightHit.lightIndex,
                                           lightSourceType,
                                           contrib,
                                           lightPos);
            
            result += pathContrib;
            break;
        }
        
        if (!hit.hit) {
            // Environment hit - use BSDF strategy (no MIS for env in this implementation)
            render::RadianceRGB envContrib = throughput * render::to_radiance(getEnvironmentColor(currentRay, scene.environment));
            render::RGB3f contrib = render::RGB3f(
                envContrib.r.numerical_value_in(render::radiance_unit),
                envContrib.g.numerical_value_in(render::radiance_unit),
                envContrib.b.numerical_value_in(render::radiance_unit)
            );
            
            // Calculate far point for environment visualization
            render::Vec3f envPos = env_hit_position(currentRay);
            
            // Plan E: Record environment hit as MIS_BSDF with geometry
            recorder.record_completed_path_with_geometry(SamplingStrategy::MIS_BSDF,
                                           -1,
                                           LightSourceType::Environment,
                                           contrib,
                                           envPos);
            
            result += envContrib;
            break;
        }
        
        MaterialParams mat = getMaterialParams(hit);
        render::RadianceRGB emission = getEmission(hit);
        
        // Record vertex
        BsdfType bsdfType = classify_bsdf(mat);
        bool isDelta = is_delta_bsdf(mat);
        bool hasEmission = render::is_emissive(emission);
        
        // Setup normals
        render::Direction n = hit.normal.as_direction();
        render::Direction wo = -currentRay.direction;
        bool frontFace = render::dot(wo, n) > 0;
        
        render::Direction shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = -n;
        }
        
        // Add emission with MIS weight
        if (hasEmission) {
            render::Dimensionless misWeight{1.0f};
            if (sceneLights.hasLights() && lastBsdfPdf > render::MIN_PDF) {
                int lightIdx = findLightIndex(hit.meshIdx, hit.triIdx);
                if (lightIdx >= 0) {
                    float selectProb = sceneLights.getPdfForLight(lightIdx);
                    if (selectProb > 0.0f) {
                        const Light& l = sceneLights.lights[lightIdx];
                        render::PdfW lightPdf = pdfLightSample(l, currentRay.origin, hit.point, hit.normal) * selectProb;
                        if (lightPdf > render::MIN_PDF) {
                            misWeight = render::mis_power_heuristic(lastBsdfPdf, lightPdf);
                        }
                    }
                }
            }
            
            // Calculate emissive hit position for visualization
            render::Vec3f emissivePos = position_to_vec3f(hit.point);
            
            // Calculate MIS-weighted contribution
            render::RadianceRGB emissionContrib = throughput * emission * misWeight;
            render::RGB3f contrib = render::RGB3f(
                emissionContrib.r.numerical_value_in(render::radiance_unit),
                emissionContrib.g.numerical_value_in(render::radiance_unit),
                emissionContrib.b.numerical_value_in(render::radiance_unit)
            );
            
            // Plan E: Record emissive mesh hit as MIS_BSDF with geometry
            recorder.record_completed_path_with_geometry(SamplingStrategy::MIS_BSDF,
                                           hit.meshIdx,
                                           LightSourceType::Emissive,
                                           contrib,
                                           emissivePos);
            
            result += emissionContrib;
        } else {
            // Convert hit position and normal to Vec3f for diagnostics
            render::Vec3f hitPos = position_to_vec3f(hit.point);
            render::Vec3f hitNorm = normal_to_vec3f(hit.normal);
            
            // Record vertex first (before NEE) with geometry
            recorder.record_vertex_with_geometry(hit.meshIdx, 0, 
                                   bsdfType, isDelta, false,
                                   hitPos, hitNorm);
            
            // Next Event Estimation with MIS
            bool isTransmissive = mat.transmission > 0.5f;
            if (sceneLights.hasLights() && !hasEmission && !isTransmissive) {
                float lightSelectProb;
                int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
                
                if (lightIdx >= 0 && lightSelectProb > 1e-6f) {
                    const Light& light = sceneLights.lights[lightIdx];
                    LightSample ls = sampleLight(light, hit.point, randf(), randf());
                    
                    if (ls.pdf > render::MIN_PDF) {
                        float NdotL = render::dot(shadingNormal, ls.direction);
                        
                        if (NdotL > 1e-6f) {
                            Ray shadowRay(hit.point, ls.direction);
                            Hit shadowHit = intersectScene(scene, shadowRay, geometry::RAY_T_MIN_TYPED, ls.distance);
                            
                            bool inShadow = shadowHit.hit;
                            
                            if (!inShadow) {
                                render::BSDFRGB f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                                
                                render::PdfW pdfLight_typed = ls.pdf * lightSelectProb;
                                render::PdfW pdfBsdf_typed = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                                
                                render::Dimensionless misWeight = render::mis_power_heuristic(pdfLight_typed, pdfBsdf_typed);
                                
                                render::ThroughputRGB bsdf_weight = render::bsdf_sample_weight(f, NdotL, pdfLight_typed);
                                render::RadianceRGB nee_radiance = (bsdf_weight * misWeight) * ls.emission;
                                
                                // Record NEE contribution as completed path with MIS_NEE strategy
                                int nativeLightIdx = lightIdx - sceneLights.nativeLightStartIndex;
                                LightSourceType lightSourceType = to_light_source_type(
                                    static_cast<int>(light.type));
                                
                                // Calculate light position for visualization
                                render::Displacement lightDisp = render::displacement_from_origin(ls.position);
                                render::Vec3f lightPos = lightDisp.numerical_value_in(render::si::metre);
                                
                                render::RadianceRGB total_nee = throughput * nee_radiance;
                                render::RGB3f nee_contrib(
                                    total_nee.r.numerical_value_in(render::radiance_unit),
                                    total_nee.g.numerical_value_in(render::radiance_unit),
                                    total_nee.b.numerical_value_in(render::radiance_unit)
                                );
                                
                                // Plan E: Record as MIS_NEE with geometry
                                recorder.record_completed_path_with_geometry(SamplingStrategy::MIS_NEE,
                                                               nativeLightIdx,
                                                               lightSourceType,
                                                               nee_contrib,
                                                               lightPos);
                                
                                result += throughput * nee_radiance;
                            }
                        }
                    }
                }
            }
        }
        
        // BSDF Sampling for next bounce
        render::Direction sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (!bsdfSample.isValid()) {
            break;
        }

        // Update throughput via variant-aware helper
        auto weightOpt = compute_throughput_update(bsdfSample, sampleNormal);
        if (!weightOpt) break;
        throughput = throughput * *weightOpt;
        
        // Store PDF for next bounce MIS
        lastBsdfPdf = mis_pdf_for_next_bounce(bsdfSample);
        
        // Russian Roulette
        if (depth >= 3) {
            float maxThroughput = render::throughput_max_component(throughput);
            float rrProb = std::min(maxThroughput, 0.95f);
            if (randf() > rrProb) break;
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (!render::throughput_is_valid(throughput)) {
            break;
        }
        
        // Setup next ray
        currentRay = Ray(hit.point, bsdfSample.wi);
        tMin = geometry::RAY_T_MIN_TYPED;
    }
    
    return result;
}
