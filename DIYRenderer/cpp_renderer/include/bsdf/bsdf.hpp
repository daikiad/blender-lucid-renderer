/**
 * bsdf.hpp - BSDF Evaluation and Sampling (Unit-Safe)
 * ====================================================
 * 
 * BSDF implementation with render:: namespace types:
 * - BSDFRGB [1/sr] for reflectance per solid angle
 * - PdfW [1/sr] for probability densities
 * - Direction for all direction vectors
 * - ColorRGB (dimensionless) for path weights
 */

#pragma once
#include "../units/render_units.hpp"
#include "../math/random.hpp"
#include "../core/ray.hpp"
#include "fresnel.hpp"
#include "ggx.hpp"

// ========== Material Parameters ==========

struct MaterialParams {
    render::ColorRGB albedo;         // Base color (dimensionless [0,1])
    float metallic;
    float roughness;
    float transmission;
    float ior;
    
    MaterialParams() 
        : albedo(0.8f, 0.8f, 0.8f)
        , metallic(0.0f)
        , roughness(0.5f)
        , transmission(0.0f)
        , ior(1.45f) {}
};

// ========== BSDF Sample Result ==========

/**
 * BSDFSample - Result of sampling a direction from BSDF (unit-safe)
 * 
 * - f: BSDF value [1/sr]
 * - pdf: probability density [1/sr]
 * - weight: f × |cosθ| / pdf (dimensionless)
 */
struct BSDFSample {
    render::Direction wi;             // Sampled direction (normalized)
    render::BSDFRGB f;                // BSDF value [1/sr]
    render::PdfW pdf;                 // Probability density [1/sr]
    render::ColorRGB weight;          // Direct throughput = f × |NdotL| / pdf
    bool useWeight;                   // If true, use weight directly
    bool isDelta;                     // Is this a delta distribution?
    
    enum Type { DIFFUSE, SPECULAR, TRANSMISSION } type;
    
    BSDFSample() 
        : wi(render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f)))
        , f(render::zero_bsdf_rgb())
        , pdf(0.0f * render::per_sr)
        , weight(1.0f, 1.0f, 1.0f)
        , useWeight(false)
        , isDelta(false)
        , type(DIFFUSE) {}
    
    bool isValid() const {
        return pdf > render::MIN_PDF || useWeight;
    }
};

// ========== BSDF Evaluation ==========

/**
 * Evaluate Lambertian diffuse BSDF [1/sr]
 */
inline render::BSDFRGB evalDiffuse(const MaterialParams& mat, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return render::zero_bsdf_rgb();
    // albedo [dimensionless] / π → [1/sr]
    return render::make_bsdf_rgb(
        mat.albedo.r * GGX_INV_PI,
        mat.albedo.g * GGX_INV_PI,
        mat.albedo.b * GGX_INV_PI
    );
}

/**
 * Evaluate GGX specular BSDF [1/sr]
 */
inline render::BSDFRGB evalSpecular(const MaterialParams& mat, const render::Direction& wo, 
                                const render::Direction& wi, const render::Direction& n, 
                                float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return render::zero_bsdf_rgb();
    
    auto h_vec = render::normalize(wo.vec() + wi.vec());
    auto h = render::make_direction_or_default(h_vec);
    float NdotH = std::max(render::dot(n.vec(), h.vec()), 0.0f);
    float VdotH = std::max(render::dot(wo.vec(), h.vec()), 0.0f);
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    float D = ggxD(NdotH, roughness);
    
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float lambdaL = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    float G2_over_denom = 0.5f / (NdotV * lambdaL + NdotL * lambdaV + GGX_EPSILON);
    
    // Compute F0 using ColorRGB
    render::ColorRGB f0 = mat.albedo * mat.metallic + render::ColorRGB(0.04f, 0.04f, 0.04f) * (1.0f - mat.metallic);
    render::ColorRGB F = fresnelSchlickColor(VdotH, f0);
    
    float spec = D * G2_over_denom;
    return render::make_bsdf_rgb(spec * F.r, spec * F.g, spec * F.b);
}

/**
 * Evaluate combined BSDF [1/sr]
 */
inline render::BSDFRGB evalBSDF(const MaterialParams& mat, const render::Direction& wo, 
                            const render::Direction& wi, const render::Direction& n) {
    float NdotL = render::dot(n.vec(), wi.vec());
    float NdotV = render::dot(n.vec(), wo.vec());
    
    if (NdotL <= 0.0f || NdotV <= 0.0f) return render::zero_bsdf_rgb();
    
    render::BSDFRGB specular = evalSpecular(mat, wo, wi, n, NdotL, NdotV);
    
    auto h_vec = render::normalize(wo.vec() + wi.vec());
    auto h = render::make_direction_or_default(h_vec);
    float VdotH = std::max(render::dot(wo.vec(), h.vec()), 0.0f);
    
    // Use ColorRGB for Fresnel calculation
    render::ColorRGB f0(0.04f, 0.04f, 0.04f);
    render::ColorRGB F = fresnelSchlickColor(VdotH, f0);
    
    render::BSDFRGB diffuse = evalDiffuse(mat, NdotL, NdotV);
    render::ColorRGB kd((1.0f - F.r) * (1.0f - mat.metallic),
                        (1.0f - F.g) * (1.0f - mat.metallic),
                        (1.0f - F.b) * (1.0f - mat.metallic));
    diffuse = diffuse * kd;  // BSDFRGB * ColorRGB -> BSDFRGB
    
    return diffuse + specular;  // BSDFRGB + BSDFRGB -> BSDFRGB
}

// ========== BSDF Sampling ==========

/**
 * Sample BSDF direction based on material properties
 * Returns BSDFSample with render:: typed members
 */
inline BSDFSample sampleBSDF(const MaterialParams& mat, const render::Direction& wo, const render::Direction& n, 
                              float u1, float u2, float u3) {
    BSDFSample sample;
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Handle glass/transmission
    if (mat.transmission > 0.0f && u3 < mat.transmission) {
        float wo_dot_n = render::dot(wo.vec(), n.vec());
        bool frontFace = wo_dot_n > 0.0f;
        render::Direction faceNormal = frontFace ? n : -n;
        float eta = frontFace ? (1.0f / mat.ior) : mat.ior;
        
        // Sample microfacet normal using VNDF
        render::Direction t, b;
        buildOrthonormalBasis(faceNormal, t, b);
        render::Direction h = sampleGGXVNDF(wo, roughness, u1, u2, faceNormal, t, b);
        
        float cosThetaI = std::abs(render::dot(wo.vec(), h.vec()));
        float F = fresnelDielectric(cosThetaI, eta);
        
        render::Direction incident = -wo;
        
        if (randf() < F) {
            // Fresnel reflection
            sample.wi = render::reflect(incident, h.as_normal());
            
            float NdotL = render::dot(faceNormal.vec(), sample.wi.vec());
            float NdotV = render::dot(faceNormal.vec(), wo.vec());
            if (NdotL <= 0.0f || NdotV <= 0.0f) {
                sample.pdf = render::PdfW::zero();
                sample.f = render::zero_bsdf_rgb();
                return sample;
            }
            
            float G2 = ggxG2(NdotL, NdotV, roughness);
            float G1 = ggxG1(NdotV, roughness);
            float w = G2 / (G1 + GGX_EPSILON);
            
            sample.useWeight = true;
            sample.weight = render::ColorRGB(w, w, w);
            sample.pdf = 1.0f * render::per_sr;
            sample.isDelta = false;
            sample.type = BSDFSample::SPECULAR;
        } else {
            // Refraction
            auto refracted_opt = render::refract(incident, h.as_normal(), eta);
            if (!refracted_opt) {
                // Total internal reflection
                sample.wi = render::reflect(incident, h.as_normal());
                
                float NdotL = render::dot(faceNormal.vec(), sample.wi.vec());
                float NdotV = render::dot(faceNormal.vec(), wo.vec());
                if (NdotL <= 0.0f || NdotV <= 0.0f) {
                    sample.pdf = render::PdfW::zero();
                    sample.f = render::zero_bsdf_rgb();
                    return sample;
                }
                
                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);
                
                sample.useWeight = true;
                sample.weight = render::ColorRGB(w, w, w);
                sample.pdf = 1.0f * render::per_sr;
                sample.isDelta = false;
                sample.type = BSDFSample::SPECULAR;
            } else {
                sample.wi = *refracted_opt;
                
                float NdotL = std::abs(render::dot(sample.wi.vec(), faceNormal.vec()));
                float NdotV = std::abs(render::dot(wo.vec(), faceNormal.vec()));
                
                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);
                
                sample.useWeight = true;
                sample.weight = render::ColorRGB(w, w, w);
                sample.pdf = 1.0f * render::per_sr;
                sample.isDelta = false;
                sample.type = BSDFSample::TRANSMISSION;
            }
        }
        return sample;
    }
    
    // Opaque material: diffuse + specular
    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = std::clamp(specProb, 0.1f, 0.9f);
    }
    
    render::Direction t, b;
    buildOrthonormalBasis(n, t, b);
    float NdotV = std::max(render::dot(n.vec(), wo.vec()), GGX_EPSILON);
    
    if (u1 < specProb) {
        // Specular sampling using GGX VNDF
        render::Direction h = sampleGGXVNDF(wo, roughness, u1 / specProb, u2, n, t, b);
        sample.wi = render::reflect(-wo, h.as_normal());
        
        float NdotL = render::dot(n.vec(), sample.wi.vec());
        if (NdotL <= 0.0f) {
            sample.pdf = render::PdfW::zero();
            sample.f = render::zero_bsdf_rgb();
            return sample;
        }
        
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        sample.pdf = pdf_raw * render::per_sr;
        
        // For near-delta specular (very low roughness), use weight-based sampling
        // This ensures MIS weight = 1.0 because light sampling cannot efficiently
        // sample the narrow specular lobe, so BSDF sampling should get full weight
        constexpr float DELTA_ROUGHNESS_THRESHOLD = 0.05f;
        if (roughness < DELTA_ROUGHNESS_THRESHOLD && mat.metallic > 0.5f) {
            // Compute weight = f * cos / pdf for pre-weighted sampling
            float weight_denom = (sample.pdf > render::MIN_PDF) 
                ? sample.pdf.numerical_value_in(render::per_sr) : 1.0f;
            render::ColorRGB f_color = render::to_color(sample.f);
            float w = f_color.r * NdotL / weight_denom;  // Assume grayscale for simplicity
            sample.weight = render::ColorRGB(w, w, w);
            sample.useWeight = true;
        } else {
            sample.useWeight = false;
        }
        sample.isDelta = false;
        sample.type = BSDFSample::SPECULAR;
    } else {
        // Diffuse sampling
        float u1Adj = (u1 - specProb) / (1.0f - specProb);
        sample.wi = sampleCosineHemisphere(u1Adj, u2, n);
        
        float NdotL = render::dot(n.vec(), sample.wi.vec());
        if (NdotL <= 0.0f) {
            sample.pdf = render::PdfW::zero();
            sample.f = render::zero_bsdf_rgb();
            return sample;
        }
        
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        
        auto h_vec = render::normalize(wo.vec() + sample.wi.vec());
        auto h = render::make_direction_or_default(h_vec);
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        sample.pdf = pdf_raw * render::per_sr;
        
        sample.useWeight = false;
        sample.isDelta = false;
        sample.type = BSDFSample::DIFFUSE;
    }
    
    return sample;
}

// ========== PDF Evaluation ==========

/**
 * Get PDF for a given direction [1/sr]
 */
inline render::PdfW pdfBSDF(const MaterialParams& mat, 
                            const render::Direction& wo, 
                            const render::Direction& wi, 
                            const render::Direction& n) {
    float NdotL = render::dot(n.vec(), wi.vec());
    if (NdotL <= 0.0f) return render::PdfW::zero();
    
    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = std::clamp(specProb, 0.1f, 0.9f);
    }
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    auto h_vec = render::normalize(wo.vec() + wi.vec());
    auto h = render::make_direction_or_default(h_vec);
    
    float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
    float pdfDiff = pdfCosineHemisphere(NdotL);
    
    float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
    return pdf_raw * render::per_sr;
}