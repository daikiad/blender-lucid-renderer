/**
 * bsdf.hpp - BSDF Evaluation and Sampling (Unit-Safe)
 * ====================================================
 * 
 * BSDF implementation with mp-units:
 * - BSDF3 [1/sr] for reflectance per solid angle
 * - PdfSolidAngle [1/sr] for probability densities
 * - Direction3 for all direction vectors
 * - Throughput3 (dimensionless) for path weights
 */

#pragma once
#include "../math/vec3_unit.hpp"
#include "../math/random.hpp"
#include "../core/ray.hpp"
#include "../units/units.hpp"
#include "fresnel.hpp"
#include "ggx.hpp"

// ========== Material Parameters ==========

struct MaterialParams {
    diy::Vec3U<mp_units::one> albedo;         // Base color (dimensionless [0,1])
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
    diy::Direction3 wi;             // Sampled direction (normalized)
    diy::BSDF3 f;                   // BSDF value [1/sr]
    diy::units::PdfSolidAngle pdf;  // Probability density [1/sr]
    diy::Vec3U<mp_units::one> weight;        // Direct throughput = f × |NdotL| / pdf
    bool useWeight;                 // If true, use weight directly
    bool isDelta;                   // Is this a delta distribution?
    
    enum Type { DIFFUSE, SPECULAR, TRANSMISSION } type;
    
    BSDFSample() 
        : wi()
        , f(0, 0, 0)
        , pdf(0.0f * diy::units::per_steradian)
        , weight(1, 1, 1)
        , useWeight(false)
        , isDelta(false)
        , type(DIFFUSE) {}
    
    bool isValid() const {
        return diy::units::to_per_sr(pdf) > 1e-6f || useWeight;
    }
};

// ========== BSDF Evaluation ==========

/**
 * Evaluate Lambertian diffuse BSDF [1/sr]
 */
inline diy::BSDF3 evalDiffuse(const MaterialParams& mat, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return diy::BSDF3(0, 0, 0);
    // albedo [dimensionless] / π → [1/sr]
    return diy::BSDF3(
        mat.albedo.x_raw() * GGX_INV_PI,
        mat.albedo.y_raw() * GGX_INV_PI,
        mat.albedo.z_raw() * GGX_INV_PI
    );
}

/**
 * Evaluate GGX specular BSDF [1/sr]
 */
inline diy::BSDF3 evalSpecular(const MaterialParams& mat, const diy::Direction3& wo, 
                                const diy::Direction3& wi, const diy::Direction3& n, 
                                float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return diy::BSDF3(0, 0, 0);
    
    auto h = diy::normalize(wo + wi);
    float NdotH = std::max(diy::dot(n, h).numerical_value_in(mp_units::one), 0.0f);
    float VdotH = std::max(diy::dot(wo, h).numerical_value_in(mp_units::one), 0.0f);
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    float D = ggxD(NdotH, roughness);
    
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float lambdaL = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    float G2_over_denom = 0.5f / (NdotV * lambdaL + NdotL * lambdaV + GGX_EPSILON);
    
    // Compute F0 using Color3
    diy::Vec3U<mp_units::one> f0 = mat.albedo * mat.metallic + diy::Vec3U<mp_units::one>(0.04f, 0.04f, 0.04f) * (1.0f - mat.metallic);
    diy::Vec3U<mp_units::one> F = fresnelSchlickColor(VdotH, f0);
    
    float spec = D * G2_over_denom;
    return diy::BSDF3(spec * F.x_raw(), spec * F.y_raw(), spec * F.z_raw());
}

/**
 * Evaluate combined BSDF [1/sr]
 */
inline diy::BSDF3 evalBSDF(const MaterialParams& mat, const diy::Direction3& wo, 
                            const diy::Direction3& wi, const diy::Direction3& n) {
    float NdotL = diy::dot(n, wi).numerical_value_in(mp_units::one);
    float NdotV = diy::dot(n, wo).numerical_value_in(mp_units::one);
    
    if (NdotL <= 0.0f || NdotV <= 0.0f) return diy::BSDF3(0, 0, 0);
    
    diy::BSDF3 specular = evalSpecular(mat, wo, wi, n, NdotL, NdotV);
    
    auto h = diy::normalize(wo + wi);
    float VdotH = std::max(diy::dot(wo, h).numerical_value_in(mp_units::one), 0.0f);
    
    // Use Color3 for Fresnel calculation
    diy::Vec3U<mp_units::one> f0(0.04f, 0.04f, 0.04f);
    diy::Vec3U<mp_units::one> F = fresnelSchlickColor(VdotH, f0);
    
    diy::BSDF3 diffuse = evalDiffuse(mat, NdotL, NdotV);
    diy::Vec3U<mp_units::one> kd((1.0f - F.x_raw()) * (1.0f - mat.metallic),
                        (1.0f - F.y_raw()) * (1.0f - mat.metallic),
                        (1.0f - F.z_raw()) * (1.0f - mat.metallic));
    diffuse = diy::hadamard(diffuse, kd);
    
    return diffuse + specular;
}

// ========== BSDF Sampling ==========

/**
 * Sample BSDF direction based on material properties
 * Returns BSDFSample with mp-units typed members
 */
inline BSDFSample sampleBSDF(const MaterialParams& mat, const diy::Direction3& wo, const diy::Direction3& n, 
                              float u1, float u2, float u3) {
    BSDFSample sample;
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Handle glass/transmission
    if (mat.transmission > 0.0f && u3 < mat.transmission) {
        float wo_dot_n = diy::dot(wo, n).numerical_value_in(mp_units::one);
        bool frontFace = wo_dot_n > 0.0f;
        diy::Direction3 faceNormal = frontFace ? n : n * -1.0f;
        float eta = frontFace ? (1.0f / mat.ior) : mat.ior;
        
        // Sample microfacet normal using VNDF
        diy::Direction3 t, b;
        buildOrthonormalBasis(faceNormal, t, b);
        diy::Direction3 h = sampleGGXVNDF(wo, roughness, u1, u2, faceNormal, t, b);
        
        float cosThetaI = std::abs(diy::dot(wo, h).numerical_value_in(mp_units::one));
        float F = fresnelDielectric(cosThetaI, eta);
        
        diy::Direction3 incident = wo * -1.0f;
        
        if (randf() < F) {
            // Fresnel reflection
            sample.wi = diy::reflect(incident, h);
            
            float NdotL = diy::dot(faceNormal, sample.wi).numerical_value_in(mp_units::one);
            float NdotV = diy::dot(faceNormal, wo).numerical_value_in(mp_units::one);
            if (NdotL <= 0.0f || NdotV <= 0.0f) {
                sample.pdf = diy::units::PdfSolidAngle::zero();
                sample.f = diy::BSDF3(0, 0, 0);
                return sample;
            }
            
            float G2 = ggxG2(NdotL, NdotV, roughness);
            float G1 = ggxG1(NdotV, roughness);
            float w = G2 / (G1 + GGX_EPSILON);
            
            sample.useWeight = true;
            sample.weight = diy::Vec3U<mp_units::one>(w, w, w);
            sample.pdf = 1.0f * diy::units::per_steradian;
            sample.isDelta = false;
            sample.type = BSDFSample::SPECULAR;
        } else {
            // Refraction
            diy::Direction3 refracted = diy::refract(incident, h, eta);
            if (diy::length(refracted) < 0.5f) {
                // Total internal reflection
                sample.wi = diy::reflect(incident, h);
                
                float NdotL = diy::dot(faceNormal, sample.wi).numerical_value_in(mp_units::one);
                float NdotV = diy::dot(faceNormal, wo).numerical_value_in(mp_units::one);
                if (NdotL <= 0.0f || NdotV <= 0.0f) {
                    sample.pdf = diy::units::PdfSolidAngle::zero();
                    sample.f = diy::BSDF3(0, 0, 0);
                    return sample;
                }
                
                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);
                
                sample.useWeight = true;
                sample.weight = diy::Vec3U<mp_units::one>(w, w, w);
                sample.pdf = 1.0f * diy::units::per_steradian;
                sample.isDelta = false;
                sample.type = BSDFSample::SPECULAR;
            } else {
                sample.wi = refracted;
                
                float NdotL = std::abs(diy::dot(refracted, faceNormal).numerical_value_in(mp_units::one));
                float NdotV = std::abs(diy::dot(wo, faceNormal).numerical_value_in(mp_units::one));
                
                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);
                
                sample.useWeight = true;
                sample.weight = diy::Vec3U<mp_units::one>(w, w, w);
                sample.pdf = 1.0f * diy::units::per_steradian;
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
    
    diy::Direction3 t, b;
    buildOrthonormalBasis(n, t, b);
    float NdotV = std::max(diy::dot(n, wo).numerical_value_in(mp_units::one), GGX_EPSILON);
    
    if (u1 < specProb) {
        // Specular sampling using GGX VNDF
        diy::Direction3 h = sampleGGXVNDF(wo, roughness, u1 / specProb, u2, n, t, b);
        sample.wi = diy::reflect(wo * -1.0f, h);
        
        float NdotL = diy::dot(n, sample.wi).numerical_value_in(mp_units::one);
        if (NdotL <= 0.0f) {
            sample.pdf = diy::units::PdfSolidAngle::zero();
            sample.f = diy::BSDF3(0, 0, 0);
            return sample;
        }
        
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        sample.pdf = pdf_raw * diy::units::per_steradian;
        
        sample.useWeight = false;
        sample.isDelta = false;
        sample.type = BSDFSample::SPECULAR;
    } else {
        // Diffuse sampling
        float u1Adj = (u1 - specProb) / (1.0f - specProb);
        sample.wi = sampleCosineHemisphere(u1Adj, u2, n);
        
        float NdotL = diy::dot(n, sample.wi).numerical_value_in(mp_units::one);
        if (NdotL <= 0.0f) {
            sample.pdf = diy::units::PdfSolidAngle::zero();
            sample.f = diy::BSDF3(0, 0, 0);
            return sample;
        }
        
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        
        diy::Direction3 h = diy::normalize(wo + sample.wi);
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        sample.pdf = pdf_raw * diy::units::per_steradian;
        
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
inline diy::units::PdfSolidAngle pdfBSDF(const MaterialParams& mat, 
                                         const diy::Direction3& wo, 
                                         const diy::Direction3& wi, 
                                         const diy::Direction3& n) {
    using namespace mp_units;
    using namespace mp_units::si;
    
    float NdotL = diy::dot(n, wi).numerical_value_in(one);
    if (NdotL <= 0.0f) return diy::units::PdfSolidAngle::zero();
    
    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = std::clamp(specProb, 0.1f, 0.9f);
    }
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    diy::Direction3 h = diy::normalize(wo + wi);
    
    float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
    float pdfDiff = pdfCosineHemisphere(NdotL);
    
    float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
    return pdf_raw * diy::units::per_steradian;
}