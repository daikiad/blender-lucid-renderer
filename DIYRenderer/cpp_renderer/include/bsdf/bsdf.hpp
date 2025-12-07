/**
 * bsdf.hpp - BSDF Evaluation and Sampling
 * ========================================
 * 
 * Complete BSDF (Bidirectional Scattering Distribution Function) implementation:
 * - MaterialParams: Material properties for BSDF evaluation
 * - BSDFSample: Result of BSDF sampling
 * - Diffuse (Lambertian) evaluation and sampling
 * - Specular (GGX Cook-Torrance) evaluation and sampling
 * - Glass (transmission) handling
 */

#pragma once
#include "../math/vec3.hpp"
#include "../math/random.hpp"
#include "../core/ray.hpp"
#include "fresnel.hpp"
#include "ggx.hpp"

// ========== Material Parameters ==========

struct MaterialParams {
    Vec3 albedo;
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

struct BSDFSample {
    Vec3 wi;        // Sampled direction
    Vec3 f;         // BSDF value
    float pdf;      // Probability density
    Vec3 weight;    // Direct throughput multiplier = f * |NdotL| / pdf
    bool useWeight; // If true, use weight directly instead of f*NdotL/pdf
    bool isDelta;   // Is this a delta distribution?
    
    enum Type { DIFFUSE, SPECULAR, TRANSMISSION } type;
    
    BSDFSample() : wi(), f(), pdf(0.0f), weight(1,1,1), useWeight(false), isDelta(false), type(DIFFUSE) {}
};

// ========== BSDF Evaluation ==========

/**
 * Evaluate Lambertian diffuse BSDF
 * f = albedo / π
 */
inline Vec3 evalDiffuse(const MaterialParams& mat, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3(0, 0, 0);
    return mat.albedo * GGX_INV_PI;
}

/**
 * Evaluate GGX specular BSDF (Cook-Torrance microfacet)
 * f = D * G * F / (4 * NdotL * NdotV)
 */
inline Vec3 evalSpecular(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, 
                         const Vec3& n, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3(0, 0, 0);
    
    Vec3 h = wo + wi;
    h.normalize();
    float NdotH = std::max(Vec3::dot(n, h), 0.0f);
    float VdotH = std::max(Vec3::dot(wo, h), 0.0f);
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Microfacet terms
    float D = ggxD(NdotH, roughness);
    
    // Combined G2/(4*NdotL*NdotV) term
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float lambdaL = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    float G2_over_denom = 0.5f / (NdotV * lambdaL + NdotL * lambdaV + GGX_EPSILON);
    
    // Fresnel term
    Vec3 f0 = mat.albedo * mat.metallic + Vec3(0.04f, 0.04f, 0.04f) * (1.0f - mat.metallic);
    Vec3 F = fresnelSchlickVec3(VdotH, f0);
    
    float spec = D * G2_over_denom;
    return Vec3(spec * F.x, spec * F.y, spec * F.z);
}

/**
 * Evaluate combined BSDF (diffuse + specular)
 */
inline Vec3 evalBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, const Vec3& n) {
    float NdotL = Vec3::dot(n, wi);
    float NdotV = Vec3::dot(n, wo);
    
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3(0, 0, 0);
    
    // Specular contribution
    Vec3 specular = evalSpecular(mat, wo, wi, n, NdotL, NdotV);
    
    // Diffuse contribution (only for non-metals)
    Vec3 h = wo + wi;
    h.normalize();
    float VdotH = std::max(Vec3::dot(wo, h), 0.0f);
    
    Vec3 f0 = Vec3(0.04f, 0.04f, 0.04f);
    Vec3 F = fresnelSchlickVec3(VdotH, f0);
    
    Vec3 diffuse = evalDiffuse(mat, NdotL, NdotV);
    Vec3 kd = Vec3((1.0f - F.x) * (1.0f - mat.metallic),
                   (1.0f - F.y) * (1.0f - mat.metallic),
                   (1.0f - F.z) * (1.0f - mat.metallic));
    diffuse = Vec3(diffuse.x * kd.x, diffuse.y * kd.y, diffuse.z * kd.z);
    
    return diffuse + specular;
}

// ========== BSDF Sampling ==========

/**
 * Sample BSDF direction based on material properties
 */
inline BSDFSample sampleBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& n, 
                              float u1, float u2, float u3) {
    BSDFSample sample;
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Handle glass/transmission
    if (mat.transmission > 0.0f && u3 < mat.transmission) {
        bool frontFace = Vec3::dot(wo, n) > 0.0f;
        Vec3 faceNormal = frontFace ? n : n * -1.0f;
        float eta = frontFace ? (1.0f / mat.ior) : mat.ior;
        
        // Sample microfacet normal using VNDF
        Vec3 t, b;
        buildOrthonormalBasis(faceNormal, t, b);
        Vec3 h = sampleGGXVNDF(wo, roughness, u1, u2, faceNormal, t, b);
        
        float cosThetaI = std::abs(Vec3::dot(wo, h));
        float F = fresnelDielectric(cosThetaI, eta);
        
        Vec3 incident = wo * -1.0f;
        
        if (randf() < F) {
            // Fresnel reflection
            sample.wi = reflect(incident, h);
            
            float NdotL = Vec3::dot(faceNormal, sample.wi);
            float NdotV = Vec3::dot(faceNormal, wo);
            if (NdotL <= 0.0f || NdotV <= 0.0f) {
                sample.pdf = 0.0f;
                sample.f = Vec3(0, 0, 0);
                return sample;
            }
            
            float G2 = ggxG2(NdotL, NdotV, roughness);
            float G1 = ggxG1(NdotV, roughness);
            float w = G2 / (G1 + GGX_EPSILON);
            
            sample.useWeight = true;
            sample.weight = Vec3(w, w, w);
            sample.pdf = 1.0f;
            sample.isDelta = false;
            sample.type = BSDFSample::SPECULAR;
        } else {
            // Refraction
            Vec3 refracted = refract(incident, h, eta);
            if (refracted.length() < 0.5f) {
                // Total internal reflection
                sample.wi = reflect(incident, h);
                float NdotL = Vec3::dot(faceNormal, sample.wi);
                float NdotV = Vec3::dot(faceNormal, wo);
                if (NdotL <= 0.0f || NdotV <= 0.0f) {
                    sample.pdf = 0.0f;
                    sample.f = Vec3(0, 0, 0);
                    return sample;
                }
                
                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);
                
                sample.useWeight = true;
                sample.weight = Vec3(w, w, w);
                sample.pdf = 1.0f;
                sample.isDelta = false;
                sample.type = BSDFSample::SPECULAR;
            } else {
                sample.wi = refracted;
                
                float NdotL = std::abs(Vec3::dot(sample.wi, faceNormal));
                float NdotV = std::abs(Vec3::dot(wo, faceNormal));
                
                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);
                
                sample.useWeight = true;
                sample.weight = Vec3(w, w, w);
                sample.pdf = 1.0f;
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
    
    Vec3 t, b;
    buildOrthonormalBasis(n, t, b);
    float NdotV = std::max(Vec3::dot(n, wo), GGX_EPSILON);
    
    if (u1 < specProb) {
        // Specular sampling using GGX VNDF
        Vec3 h = sampleGGXVNDF(wo, roughness, u1 / specProb, u2, n, t, b);
        sample.wi = reflect(wo * -1.0f, h);
        
        float NdotL = Vec3::dot(n, sample.wi);
        if (NdotL <= 0.0f) {
            sample.pdf = 0.0f;
            sample.f = Vec3(0, 0, 0);
            return sample;
        }
        
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        sample.pdf = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        
        sample.useWeight = false;
        sample.isDelta = false;
        sample.type = BSDFSample::SPECULAR;
    } else {
        // Diffuse sampling
        float u1Adj = (u1 - specProb) / (1.0f - specProb);
        sample.wi = sampleCosineHemisphere(u1Adj, u2, n);
        
        float NdotL = Vec3::dot(n, sample.wi);
        if (NdotL <= 0.0f) {
            sample.pdf = 0.0f;
            sample.f = Vec3(0, 0, 0);
            return sample;
        }
        
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        Vec3 h = wo + sample.wi;
        h.normalize();
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        sample.pdf = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        
        sample.useWeight = false;
        sample.isDelta = false;
        sample.type = BSDFSample::DIFFUSE;
    }
    
    return sample;
}

/**
 * Get PDF for a given direction
 */
inline float pdfBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, const Vec3& n) {
    float NdotL = Vec3::dot(n, wi);
    if (NdotL <= 0.0f) return 0.0f;
    
    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = std::clamp(specProb, 0.1f, 0.9f);
    }
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    Vec3 h = wo + wi;
    h.normalize();
    
    float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
    float pdfDiff = pdfCosineHemisphere(NdotL);
    
    return (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
}
