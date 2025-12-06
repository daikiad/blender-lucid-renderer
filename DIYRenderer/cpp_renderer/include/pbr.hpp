/**
 * pbr.hpp - Physically Based Rendering with Multiple Importance Sampling
 * ======================================================================
 * 
 * This file implements physically correct BSDF models and sampling strategies.
 * 
 * Key Components:
 * - BSDF evaluation functions (f): Evaluate the reflectance/transmittance
 * - PDF functions: Probability density for a given direction
 * - Sample functions: Generate random directions according to the PDF
 * - Multiple Importance Sampling (MIS): Combine light and BSDF sampling
 * 
 * Mathematical Foundation:
 * - Rendering equation: Lo(p, wo) = Le(p, wo) + ∫ f(p, wi, wo) * Li(p, wi) * cos(θi) dwi
 * - Monte Carlo estimator: Lo ≈ (1/N) * Σ f(p, wi, wo) * Li(p, wi) * cos(θi) / pdf(wi)
 * - MIS weight: w(pf) = pf^β / (pf^β + pg^β), typically β=2 (power heuristic)
 * 
 * BSDF Models:
 * - Lambertian diffuse: f = albedo / π
 * - GGX specular (Cook-Torrance): f = D * G * F / (4 * cos(θi) * cos(θo))
 * - Glass (specular transmission): Fresnel reflection + refraction
 */

#pragma once
#include "renderer.hpp"
#include <cmath>
#include <algorithm>

constexpr float PI = 3.14159265358979323846f;
constexpr float INV_PI = 0.31830988618379067154f;
constexpr float EPSILON = 1e-6f;

// Minimum roughness to avoid delta distributions
// This ensures all BSDFs are continuous and can be properly sampled via NEE/MIS
// 0.01 gives a good balance between sharp reflections and stable MIS
constexpr float MIN_ROUGHNESS = 0.01f;

// ========== Utility Functions ==========

// Clamp value between min and max
inline float clampf(float x, float minVal, float maxVal) {
    return std::max(minVal, std::min(maxVal, x));
}

// Square of a value
inline float sqr(float x) { return x * x; }

// Safe square root (clamps negative values to 0)
inline float safe_sqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

// Build orthonormal basis from normal (Duff et al. 2017)
inline void buildOrthonormalBasis(const Vec3& n, Vec3& tangent, Vec3& bitangent) {
    // Handle edge case where n.z is very close to -1
    if (n.z < -0.9999f) {
        tangent = Vec3(0.0f, -1.0f, 0.0f);
        bitangent = Vec3(-1.0f, 0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + n.z);
    float b = -n.x * n.y * a;
    tangent = Vec3(1.0f - n.x * n.x * a, b, -n.x);
    bitangent = Vec3(b, 1.0f - n.y * n.y * a, -n.y);
}

// Transform direction from local space (z-up) to world space
inline Vec3 localToWorld(const Vec3& local, const Vec3& n, const Vec3& t, const Vec3& b) {
    return Vec3(
        t.x * local.x + b.x * local.y + n.x * local.z,
        t.y * local.x + b.y * local.y + n.y * local.z,
        t.z * local.x + b.z * local.y + n.z * local.z
    );
}

// Transform direction from world space to local space (z-up)
inline Vec3 worldToLocal(const Vec3& world, const Vec3& n, const Vec3& t, const Vec3& b) {
    return Vec3(
        Vec3::dot(world, t),
        Vec3::dot(world, b),
        Vec3::dot(world, n)
    );
}

// ========== Fresnel Functions ==========

// Schlick's approximation for Fresnel reflectance (dielectric)
// F0 = ((n1 - n2) / (n1 + n2))^2
inline float fresnelSchlick(float cosTheta, float f0) {
    return f0 + (1.0f - f0) * std::pow(1.0f - cosTheta, 5.0f);
}

// Fresnel for conductor (metal) - Schlick with F0 = albedo
inline Vec3 fresnelSchlickVec3(float cosTheta, const Vec3& f0) {
    float f = std::pow(1.0f - cosTheta, 5.0f);
    return Vec3(
        f0.x + (1.0f - f0.x) * f,
        f0.y + (1.0f - f0.y) * f,
        f0.z + (1.0f - f0.z) * f
    );
}

// Exact Fresnel for dielectric
// cosThetaI: cosine of incident angle (positive)
// eta: ratio n_incident / n_transmitted (e.g., 1/1.5 when entering glass from air)
inline float fresnelDielectric(float cosThetaI, float eta) {
    // Clamp cosThetaI to valid range
    cosThetaI = clampf(std::abs(cosThetaI), 0.0f, 1.0f);
    
    // Compute sin^2(theta_t) using Snell's law: n1*sin(theta_i) = n2*sin(theta_t)
    // sin^2(theta_t) = (n1/n2)^2 * sin^2(theta_i) = eta^2 * (1 - cos^2(theta_i))
    float sinThetaI2 = 1.0f - cosThetaI * cosThetaI;
    float sinThetaT2 = eta * eta * sinThetaI2;
    
    if (sinThetaT2 >= 1.0f) {
        // Total internal reflection
        return 1.0f;
    }
    
    float cosThetaT = safe_sqrt(1.0f - sinThetaT2);
    
    // Fresnel equations for s and p polarization
    // rs = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    // Using eta = n1/n2: rs = (eta*cos_i - cos_t) / (eta*cos_i + cos_t)
    float etaCosI = eta * cosThetaI;
    float etaCosT = eta * cosThetaT;
    
    float rs = (etaCosI - cosThetaT) / (etaCosI + cosThetaT + EPSILON);
    float rp = (cosThetaI - etaCosT) / (cosThetaI + etaCosT + EPSILON);
    
    return 0.5f * (rs * rs + rp * rp);
}

// ========== GGX/Trowbridge-Reitz Microfacet Model ==========

// GGX Normal Distribution Function (NDF)
// D(m) = α² / (π * (cos²θ * (α² - 1) + 1)²)
// Note: In Blender/Cycles, alpha = roughness², so we use a2 = roughness⁴
inline float ggxD(float NdotH, float roughness) {
    float alpha = roughness * roughness;  // α = roughness²
    float a2 = alpha * alpha;             // α² = roughness⁴
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 * INV_PI / (denom * denom + EPSILON);
}

// GGX Geometry function (Smith's method with height-correlated G2)
// G1(v) = 2 * NdotV / (NdotV + sqrt(α² + (1 - α²) * NdotV²))
inline float ggxG1(float NdotV, float roughness) {
    float alpha = roughness * roughness;  // α = roughness²
    float a2 = alpha * alpha;             // α² = roughness⁴
    return 2.0f * NdotV / (NdotV + safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
}

// Height-correlated Smith G2
// G2(l,v) = 1 / (1 + Lambda(l) + Lambda(v))
// For GGX: Lambda(v) = (-1 + sqrt(1 + α²*tan²θ)) / 2
inline float ggxG2(float NdotL, float NdotV, float roughness) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return 0.0f;
    
    float alpha = roughness * roughness;  // α = roughness²
    float a2 = alpha * alpha;             // α² = roughness⁴
    
    // Lambda terms
    float lambdaL = safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    
    // G2 = 2 * NdotL * NdotV / (NdotV * lambdaL + NdotL * lambdaV)
    float denom = NdotV * lambdaL + NdotL * lambdaV;
    if (denom < EPSILON) return 0.0f;
    
    return 2.0f * NdotL * NdotV / denom;
}

// Sample GGX microfacet normal (VNDF sampling - Heitz 2018)
// Returns half-vector in world space
inline Vec3 sampleGGXVNDF(const Vec3& wo, float roughness, float u1, float u2, 
                          const Vec3& n, const Vec3& t, const Vec3& b) {
    // Transform wo to local space
    Vec3 woLocal = worldToLocal(wo, n, t, b);
    
    // Stretch wo by alpha (in Blender, alpha = roughness²)
    float alpha = roughness * roughness;
    Vec3 woStretched = Vec3(woLocal.x * alpha, woLocal.y * alpha, woLocal.z);
    woStretched.normalize();
    
    // Build orthonormal basis around stretched wo
    Vec3 t1 = (woStretched.z < 0.9999f) ? 
        Vec3::cross(Vec3(0, 0, 1), woStretched) : Vec3(1, 0, 0);
    t1.normalize();
    Vec3 t2 = Vec3::cross(woStretched, t1);
    
    // Sample point on disk
    float r = std::sqrt(u1);
    float phi = 2.0f * PI * u2;
    float p1 = r * std::cos(phi);
    float p2 = r * std::sin(phi);
    float s = 0.5f * (1.0f + woStretched.z);
    p2 = (1.0f - s) * std::sqrt(1.0f - p1 * p1) + s * p2;
    
    // Compute half-vector
    Vec3 hLocal = t1 * p1 + t2 * p2 + woStretched * std::sqrt(std::max(0.0f, 1.0f - p1*p1 - p2*p2));
    
    // Unstretch
    hLocal = Vec3(hLocal.x * alpha, hLocal.y * alpha, std::max(0.0f, hLocal.z));
    hLocal.normalize();
    
    // Transform back to world space
    return localToWorld(hLocal, n, t, b);
}

// PDF for GGX VNDF sampling
// PDF_h = D(h) * G1(wo) * max(0, wo·h) / wo·n
// PDF_wi = PDF_h / (4 * wo·h)  (Jacobian for reflection)
inline float pdfGGXVNDF(const Vec3& wo, const Vec3& h, float roughness, const Vec3& n) {
    float NdotH = Vec3::dot(n, h);
    float VdotH = Vec3::dot(wo, h);
    float NdotV = Vec3::dot(n, wo);
    if (NdotH <= 0.0f || VdotH <= 0.0f || NdotV <= 0.0f) return 0.0f;
    
    float D = ggxD(NdotH, roughness);
    float G1 = ggxG1(NdotV, roughness);
    
    // PDF for the half-vector, then convert to wi using Jacobian (1 / 4*VdotH)
    return D * G1 / (4.0f * NdotV);
}

// ========== Cosine-Weighted Hemisphere Sampling ==========

// Sample direction with cosine-weighted distribution
// PDF = cos(θ) / π
inline Vec3 sampleCosineHemisphere(float u1, float u2, const Vec3& n) {
    // Concentric disk mapping (Shirley)
    float r = std::sqrt(u1);
    float theta = 2.0f * PI * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    
    Vec3 t, b;
    buildOrthonormalBasis(n, t, b);
    
    return localToWorld(Vec3(x, y, z), n, t, b);
}

// PDF for cosine-weighted hemisphere sampling
inline float pdfCosineHemisphere(float NdotL) {
    return std::max(0.0f, NdotL) * INV_PI;
}

// ========== Material BSDF Structure ==========

struct MaterialParams {
    Vec3 albedo;
    float metallic;
    float roughness;
    float transmission;
    float ior;
    
    MaterialParams() : albedo(0.8f, 0.8f, 0.8f), metallic(0.0f), 
                       roughness(0.5f), transmission(0.0f), ior(1.45f) {}
};

// ========== BSDF Evaluation ==========

// Evaluate Lambertian diffuse BSDF
// f = albedo / π
inline Vec3 evalDiffuse(const MaterialParams& mat, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3(0, 0, 0);
    return mat.albedo * INV_PI;
}

// Evaluate GGX specular BSDF (Cook-Torrance microfacet)
// f = D * G * F / (4 * NdotL * NdotV)
// We use the combined G2/(4*NdotL*NdotV) term for efficiency
inline Vec3 evalSpecular(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, 
                         const Vec3& n, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3(0, 0, 0);
    
    Vec3 h = wo + wi;
    h.normalize();
    float NdotH = std::max(Vec3::dot(n, h), 0.0f);
    float VdotH = std::max(Vec3::dot(wo, h), 0.0f);
    
    // Clamp roughness to avoid singularity - use MIN_ROUGHNESS for consistency
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Microfacet terms
    float D = ggxD(NdotH, roughness);
    
    // Smith G2 / (4 * NdotL * NdotV) combined term
    // Note: In Blender, alpha = roughness²
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float lambdaL = safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    float G2_over_denom = 0.5f / (NdotV * lambdaL + NdotL * lambdaV + EPSILON);
    
    // Fresnel term - for metals use albedo as F0, for dielectrics use 0.04
    Vec3 f0 = mat.albedo * mat.metallic + Vec3(0.04f, 0.04f, 0.04f) * (1.0f - mat.metallic);
    Vec3 F = fresnelSchlickVec3(VdotH, f0);
    
    // f = D * G2_over_denom * F
    float spec = D * G2_over_denom;
    return Vec3(spec * F.x, spec * F.y, spec * F.z);
}

// Evaluate combined BSDF (diffuse + specular)
// Energy conservation: diffuse is weighted by (1 - F) * (1 - metallic)
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
    
    // F0 for dielectrics
    Vec3 f0 = Vec3(0.04f, 0.04f, 0.04f);
    Vec3 F = fresnelSchlickVec3(VdotH, f0);
    
    // Diffuse is weighted by (1 - F) for energy conservation
    // And only non-metallic parts have diffuse
    Vec3 diffuse = evalDiffuse(mat, NdotL, NdotV);
    Vec3 kd = Vec3((1.0f - F.x) * (1.0f - mat.metallic),
                   (1.0f - F.y) * (1.0f - mat.metallic),
                   (1.0f - F.z) * (1.0f - mat.metallic));
    diffuse = Vec3(diffuse.x * kd.x, diffuse.y * kd.y, diffuse.z * kd.z);
    
    return diffuse + specular;
}

// ========== BSDF Sampling ==========

struct BSDFSample {
    Vec3 wi;        // Sampled direction
    Vec3 f;         // BSDF value (only used when useWeight=false)
    float pdf;      // Probability density (only used when useWeight=false)
    Vec3 weight;    // Direct throughput multiplier = f * |NdotL| / pdf (used when useWeight=true)
    bool useWeight; // If true, use weight directly instead of f*NdotL/pdf
    bool isDelta;   // Is this a delta distribution (perfect mirror/glass)?
    enum Type { DIFFUSE, SPECULAR, TRANSMISSION } type;
    
    BSDFSample() : wi(), f(), pdf(0.0f), weight(1,1,1), useWeight(false), isDelta(false), type(DIFFUSE) {}
};

// Sample BSDF direction based on material properties
inline BSDFSample sampleBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& n, 
                              float u1, float u2, float u3) {
    BSDFSample sample;
    
    // Use minimum roughness to avoid numerical issues
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Handle glass/transmission
    if (mat.transmission > 0.0f && u3 < mat.transmission) {
        // Glass path - dielectric GGX microfacet model
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
            // Fresnel reflection - sampled with probability F
            sample.wi = reflect(incident, h);
            
            float NdotL = Vec3::dot(faceNormal, sample.wi);
            float NdotV = Vec3::dot(faceNormal, wo);
            if (NdotL <= 0.0f || NdotV <= 0.0f) {
                sample.pdf = 0.0f;
                sample.f = Vec3(0, 0, 0);
                return sample;
            }
            
            // VNDF sampling: weight = G2/G1 (F is importance sampled)
            float G2 = ggxG2(NdotL, NdotV, roughness);
            float G1 = ggxG1(NdotV, roughness);
            float w = G2 / (G1 + EPSILON);
            
            // Use direct weight to avoid numerical issues
            sample.useWeight = true;
            sample.weight = Vec3(w, w, w);
            sample.pdf = 1.0f;  // Dummy, not used when useWeight=true
            
            sample.isDelta = false;
            sample.type = BSDFSample::SPECULAR;
        } else {
            // Refraction - sampled with probability (1-F)
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
                float w = G2 / (G1 + EPSILON);
                
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
                
                // For transmission with VNDF sampling, weight = G2/G1
                // This gives correct energy conservation for rough glass
                float w = G2 / (G1 + EPSILON);
                
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
    // For metals (metallic=1), use pure specular sampling since there's no diffuse
    // For dielectrics, mix based on roughness (smooth = more specular sampling)
    float specProb;
    if (mat.metallic > 0.99f) {
        // Pure metal - no diffuse component, use pure specular sampling
        specProb = 1.0f;
    } else {
        // Mix sampling: higher metallic and lower roughness = more specular
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = clampf(specProb, 0.1f, 0.9f);
    }
    
    Vec3 t, b;
    buildOrthonormalBasis(n, t, b);
    float NdotV = std::max(Vec3::dot(n, wo), EPSILON);
    
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
        
        // Standard approach: evaluate full BSDF and combined PDF
        // weight = f * cos / pdf
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
        
        // Standard approach: evaluate full BSDF and combined PDF
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

// Get PDF for a given direction
// All BSDFs are continuous (no delta distributions) thanks to MIN_ROUGHNESS
inline float pdfBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, const Vec3& n) {
    float NdotL = Vec3::dot(n, wi);
    if (NdotL <= 0.0f) return 0.0f;
    
    // Same probability distribution as sampling
    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = clampf(specProb, 0.1f, 0.9f);
    }
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    Vec3 h = wo + wi;
    h.normalize();
    
    float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
    float pdfDiff = pdfCosineHemisphere(NdotL);
    
    return (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
}

// ========== Light Sampling ==========

// ライトタイプ
enum class LightType {
    EMISSIVE_MESH,  // エミッシブメッシュ（既存）
    POINT,          // 点光源
    SUN,            // 平行光源（太陽）
    SPOT,           // スポットライト
    AREA            // 面光源（矩形/円形）
};

struct Light {
    LightType type = LightType::EMISSIVE_MESH;
    
    Vec3 position;      // Center position (for area lights, center of triangle)
    Vec3 normal;        // Light surface normal / direction (for SUN/SPOT)
    Vec3 emission;      // Emission color * strength
    float area;         // Surface area (for area sampling PDF)
    int meshIndex;      // Index of emissive mesh (-1 for native lights)
    int triangleIndex;  // Index of emissive triangle (-1 for native lights)
    
    // Triangle vertices (for emissive mesh area light sampling)
    Vec3 v0, v1, v2;
    
    // Native light properties
    float energy = 1.0f;       // Light energy/power (Watts)
    float radius = 0.0f;       // Soft shadow radius (Point/Spot)
    float spotAngle = 0.0f;    // Spot cone angle (radians)
    float spotBlend = 0.0f;    // Spot edge softness (0-1)
    
    // Area light properties
    Vec3 right;         // X axis for area light
    Vec3 up;            // Y axis for area light
    float sizeX = 1.0f; // Width
    float sizeY = 1.0f; // Height
    
    Light() : area(0.0f), meshIndex(-1), triangleIndex(-1) {}
};

// Sample a point on a triangle uniformly
inline Vec3 sampleTrianglePoint(const Vec3& v0, const Vec3& v1, const Vec3& v2, 
                                 float u1, float u2) {
    // Uniform sampling on triangle using barycentric coordinates
    float su1 = std::sqrt(u1);
    float b0 = 1.0f - su1;
    float b1 = u2 * su1;
    float b2 = 1.0f - b0 - b1;
    return v0 * b0 + v1 * b1 + v2 * b2;
}

// Calculate triangle area
inline float triangleArea(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 cross = Vec3::cross(e1, e2);
    return 0.5f * cross.length();
}

struct LightSample {
    Vec3 position;      // Point on light
    Vec3 normal;        // Light surface normal at sampled point
    Vec3 emission;      // Light emission
    float pdf;          // Probability density (in solid angle measure)
    float distance;     // Distance to light
    Vec3 direction;     // Direction from shading point to light
    
    LightSample() : pdf(0.0f), distance(0.0f) {}
};

// Sample uniform point on disk (for area lights)
inline Vec3 sampleDisk(float u1, float u2) {
    float r = std::sqrt(u1);
    float theta = 2.0f * M_PI * u2;
    return Vec3(r * std::cos(theta), r * std::sin(theta), 0.0f);
}

// Sample a random point on a light source
inline LightSample sampleLight(const Light& light, const Vec3& shadingPoint, 
                                float u1, float u2) {
    LightSample sample;
    sample.emission = light.emission;
    
    switch (light.type) {
        case LightType::POINT: {
            // Point light: sample from sphere with radius for soft shadows
            if (light.radius > EPSILON) {
                // Sample point on sphere surface
                float z = 1.0f - 2.0f * u1;
                float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
                float phi = 2.0f * M_PI * u2;
                Vec3 offset(r * std::cos(phi), r * std::sin(phi), z);
                sample.position = light.position + offset * light.radius;
            } else {
                sample.position = light.position;
            }
            sample.normal = Vec3(0, 0, 0); // Point lights don't have normal
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            // Point light: PDF = 1 (delta distribution, scaled by distance^2 in radiance calc)
            // For soft point lights, PDF is on sphere surface
            sample.pdf = 1.0f;
            break;
        }
        
        case LightType::SUN: {
            // Sun light: direction is constant, infinite distance
            sample.direction = light.normal * -1.0f; // normal stores direction
            sample.position = shadingPoint + sample.direction * 1000000.0f; // Very far
            sample.normal = light.normal;
            sample.distance = 1000000.0f;
            sample.pdf = 1.0f; // Delta distribution
            break;
        }
        
        case LightType::SPOT: {
            // Similar to point but with angular falloff
            sample.position = light.position;
            sample.normal = light.normal;
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            // Check if point is within spot cone
            float cosAngle = Vec3::dot(light.normal, sample.direction * -1.0f);
            float cosCone = std::cos(light.spotAngle * 0.5f);
            
            if (cosAngle < cosCone) {
                // Outside cone
                sample.emission = Vec3(0, 0, 0);
            } else {
                // Apply spot falloff
                float blend = light.spotBlend;
                if (blend > 0.0f) {
                    float t = (cosAngle - cosCone) / (1.0f - cosCone);
                    float falloff = std::min(1.0f, t / blend);
                    sample.emission = sample.emission * falloff;
                }
            }
            sample.pdf = 1.0f;
            break;
        }
        
        case LightType::AREA: {
            // Area light: sample point on rectangle
            float localX = (u1 - 0.5f) * light.sizeX;
            float localY = (u2 - 0.5f) * light.sizeY;
            sample.position = light.position + light.right * localX + light.up * localY;
            sample.normal = light.normal;
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            // Check if we're on the right side of the light
            float cosLight = Vec3::dot(sample.normal, sample.direction * -1.0f);
            if (cosLight < EPSILON) {
                sample.pdf = 0.0f;
            } else {
                sample.pdf = (sample.distance * sample.distance) / (light.area * cosLight);
            }
            break;
        }
        
        case LightType::EMISSIVE_MESH:
        default: {
            // Original triangle-based sampling
            sample.position = sampleTrianglePoint(light.v0, light.v1, light.v2, u1, u2);
            sample.normal = light.normal;
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            float cosLight = std::abs(Vec3::dot(sample.normal, sample.direction * -1.0f));
            if (cosLight < EPSILON) {
                sample.pdf = 0.0f;
            } else {
                sample.pdf = (sample.distance * sample.distance) / (light.area * cosLight);
            }
            break;
        }
    }
    
    return sample;
}

// PDF for light sampling (for MIS)
inline float pdfLightSample(const Light& light, const Vec3& shadingPoint, 
                            const Vec3& lightPoint, const Vec3& lightNormal) {
    Vec3 toLight = lightPoint - shadingPoint;
    float dist = toLight.length();
    Vec3 dir = toLight * (1.0f / dist);
    float cosLight = std::abs(Vec3::dot(lightNormal, dir * -1.0f));
    
    if (cosLight < EPSILON || light.area < EPSILON) return 0.0f;
    
    return (dist * dist) / (light.area * cosLight);
}

// ========== Multiple Importance Sampling ==========

// Power heuristic with β = 2
// w = pf² / (pf² + pg²)
inline float powerHeuristic(float pf, float pg) {
    float f2 = pf * pf;
    float g2 = pg * pg;
    return f2 / (f2 + g2 + EPSILON);
}

// Balance heuristic
// w = pf / (pf + pg)
inline float balanceHeuristic(float pf, float pg) {
    return pf / (pf + pg + EPSILON);
}

// ========== Scene Light Collection ==========

struct SceneLights {
    std::vector<Light> lights;
    float totalArea;
    std::vector<float> cdf;  // CDF for importance sampling lights by area
    
    SceneLights() : totalArea(0.0f) {}
    
    void buildFromScene(const Scene& scene) {
        lights.clear();
        totalArea = 0.0f;
        
        for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
            const Mesh& mesh = scene.meshes[mi];
            
            // Check if mesh has emission
            Vec3 emission = mesh.material.emission;
            if (mesh.material.useNodes && mesh.material.nodeTree.valid) {
                // Use dummy UV (0,0) for light intensity evaluation
                emission = getEmissionFromNodeTree(mesh.material.nodeTree, Vec2(0.0f, 0.0f));
            }
            
            float emissionStrength = emission.x + emission.y + emission.z;
            if (emissionStrength < EPSILON) continue;
            
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
                
                if (light.area > EPSILON) {
                    lights.push_back(light);
                    totalArea += light.area;
                }
            }
        }
        
        // Add native Blender lights (Point, Sun, Spot, Area)
        for (const Light& nativeLight : scene.nativeLights) {
            Light light = nativeLight;  // Copy
            
            // Ensure area is set for all light types (needed for CDF)
            if (light.area < EPSILON) {
                light.area = 1.0f;  // Default for point-like lights
            }
            
            lights.push_back(light);
            totalArea += light.area;
        }
        
        if (!scene.nativeLights.empty()) {
            std::cerr << "[SceneLights] Added " << scene.nativeLights.size() 
                      << " native lights" << std::endl;
        }
        
        // Build CDF for light selection
        cdf.resize(lights.size());
        float cumulative = 0.0f;
        for (size_t i = 0; i < lights.size(); ++i) {
            cumulative += lights[i].area;
            cdf[i] = cumulative / totalArea;
        }
    }
    
    // Select a light based on area-weighted probability
    // Returns index and selection probability
    int selectLight(float u, float& selectionProb) const {
        if (lights.empty()) {
            selectionProb = 0.0f;
            return -1;
        }
        
        // Binary search in CDF
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        int idx = (int)(it - cdf.begin());
        idx = std::min(idx, (int)lights.size() - 1);
        
        // Selection probability is proportional to area
        selectionProb = lights[idx].area / totalArea;
        
        return idx;
    }
    
    // Get the total PDF for sampling a specific point on any light
    // This accounts for the probability of selecting that light
    float getPdfForLight(int lightIdx) const {
        if (lightIdx < 0 || lightIdx >= (int)lights.size()) return 0.0f;
        return lights[lightIdx].area / totalArea;
    }
    
    bool hasLights() const { return !lights.empty(); }
};

// ========== Path Tracing Implementations ==========

// ==========================================================
// 1. Simple Path Tracing (BSDF sampling only, no NEE)
// ==========================================================
// This is the most basic path tracer - just samples BSDF and accumulates emission
inline Vec3 traceSimple(const Scene& scene, const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, true);
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay);
            break;
        }
        
        // Get material parameters
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
        
        // Enforce minimum roughness to avoid delta distributions
        mat.roughness = std::max(mat.roughness, MIN_ROUGHNESS);
        
        // Get emission
        Vec3 emission = hit.material.emission;
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            emission = getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        
        // Add emission (always, no MIS)
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
        
        // Sample BSDF for next direction
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < EPSILON && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            // Direct weight (for glass etc.) - already computed as f*|NdotL|/pdf
            throughput = Vec3(
                throughput.x * bsdfSample.weight.x,
                throughput.y * bsdfSample.weight.y,
                throughput.z * bsdfSample.weight.z
            );
        } else {
            // Standard: f * |NdotL| / pdf
            float absNdotL = std::abs(Vec3::dot(sampleNormal, bsdfSample.wi));
            if (absNdotL > EPSILON && bsdfSample.pdf > EPSILON) {
                float weightX = bsdfSample.f.x * absNdotL / bsdfSample.pdf;
                float weightY = bsdfSample.f.y * absNdotL / bsdfSample.pdf;
                float weightZ = bsdfSample.f.z * absNdotL / bsdfSample.pdf;
                
                // Debug: clamp excessive weights
                const float MAX_WEIGHT = 10.0f;
                if (weightX > MAX_WEIGHT || weightY > MAX_WEIGHT || weightZ > MAX_WEIGHT) {
                    // This shouldn't happen with correct VNDF sampling
                    weightX = std::min(weightX, MAX_WEIGHT);
                    weightY = std::min(weightY, MAX_WEIGHT);
                    weightZ = std::min(weightZ, MAX_WEIGHT);
                }
                
                throughput = Vec3(
                    throughput.x * weightX,
                    throughput.y * weightY,
                    throughput.z * weightZ
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette after depth 3
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
        if (bsdfSample.type == BSDFSample::TRANSMISSION) {
            currentRay.o = hit.point + bsdfSample.wi * 0.001f;
        } else {
            Vec3 offsetNormal = (Vec3::dot(bsdfSample.wi, sampleNormal) > 0) ? sampleNormal : sampleNormal * -1.0f;
            currentRay.o = hit.point + offsetNormal * 0.001f;
        }
        currentRay.d = bsdfSample.wi;
    }
    
    return result;
}

// ==========================================================
// 2. NEE-only Path Tracing (Next Event Estimation)
// ==========================================================
// Uses light sampling for direct illumination, BSDF sampling for indirect
// No MIS - just uses NEE for direct, skips emission on BSDF hits (except specular)
inline Vec3 traceNEE(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, true);
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay);
            break;
        }
        
        // Get material parameters
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
        
        // Enforce minimum roughness to avoid delta distributions
        mat.roughness = std::max(mat.roughness, MIN_ROUGHNESS);
        
        // Get emission
        Vec3 emission = hit.material.emission;
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            emission = getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        
        // Setup normals
        Vec3 n = hit.normal;
        Vec3 wo = currentRay.d * -1.0f;
        wo.normalize();
        bool frontFace = Vec3::dot(wo, n) > 0;
        
        Vec3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Add emission only on first hit (depth == 0)
        // NEE handles all other light contributions via direct light sampling
        float emissionStrength = emission.x + emission.y + emission.z;
        if (emissionStrength > EPSILON && depth == 0) {
            result = result + throughput * emission;
        }
        
        // ===== Next Event Estimation =====
        // Skip for emissive surfaces and transmission materials
        // (transmission materials need special handling - BSDF sampling only)
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && emissionStrength < EPSILON && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > EPSILON) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > EPSILON) {
                    // NdotL must be positive (light on same side as normal)
                    float NdotL = Vec3::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > EPSILON) {
                        // Shadow test - offset in light direction to avoid self-intersection
                        Ray shadowRay{hit.point + ls.direction * 0.001f, ls.direction};
                        Hit shadowHit = intersectScene(scene, shadowRay, true);
                        
                        // Use larger tolerance to avoid artifacts from light source self-intersection
                        bool inShadow = shadowHit.hit && shadowHit.t < ls.distance - 0.01f;
                        
                        if (!inShadow) {
                            Vec3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            float pdfLight = ls.pdf * lightSelectProb;
                            
                            // No MIS weight - just 1/pdf
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
        
        // ===== BSDF Sampling for next bounce =====
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < EPSILON && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            // Direct weight (for glass etc.) - already computed as f*|NdotL|/pdf
            throughput = Vec3(
                throughput.x * bsdfSample.weight.x,
                throughput.y * bsdfSample.weight.y,
                throughput.z * bsdfSample.weight.z
            );
        } else {
            // Standard: f * |NdotL| / pdf
            float NdotL = std::abs(Vec3::dot(sampleNormal, bsdfSample.wi));
            if (NdotL > EPSILON && bsdfSample.pdf > EPSILON) {
                throughput = Vec3(
                    throughput.x * bsdfSample.f.x * NdotL / bsdfSample.pdf,
                    throughput.y * bsdfSample.f.y * NdotL / bsdfSample.pdf,
                    throughput.z * bsdfSample.f.z * NdotL / bsdfSample.pdf
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette after depth 3
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
        if (bsdfSample.type == BSDFSample::TRANSMISSION) {
            currentRay.o = hit.point + bsdfSample.wi * 0.001f;
        } else {
            Vec3 offsetNormal = (Vec3::dot(bsdfSample.wi, sampleNormal) > 0) ? sampleNormal : sampleNormal * -1.0f;
            currentRay.o = hit.point + offsetNormal * 0.001f;
        }
        currentRay.d = bsdfSample.wi;
    }
    
    return result;
}

// ==========================================================
// 3. MIS Path Tracing (Multiple Importance Sampling)
// ==========================================================
// Combines BSDF sampling and NEE with proper MIS weights
// Both paths contribute, weighted by their relative probabilities
// All BSDFs are continuous (no delta distributions) due to MIN_ROUGHNESS
inline Vec3 traceMIS(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    float lastBsdfPdf = 0.0f;    // PDF of previous BSDF sample (for emission MIS), 0 means first hit
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, true);
        
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay);
            break;
        }
        
        // Get material parameters
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
        
        // Enforce minimum roughness to avoid delta distributions
        mat.roughness = std::max(mat.roughness, MIN_ROUGHNESS);
        
        // Get emission
        Vec3 emission = hit.material.emission;
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            emission = getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        
        // Setup normals
        Vec3 n = hit.normal;
        Vec3 wo = currentRay.d * -1.0f;
        wo.normalize();
        bool frontFace = Vec3::dot(wo, n) > 0;
        
        Vec3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // ===== Add emission with MIS weight =====
        float emissionStrength = emission.x + emission.y + emission.z;
        if (emissionStrength > EPSILON) {
            float emissionWeight = 1.0f;
            
            // Apply MIS weight for all bounces except first hit
            // (first hit: no previous BSDF sample, so no MIS needed)
            if (lastBsdfPdf > EPSILON && sceneLights.hasLights()) {
                // What would be the PDF if we had sampled this point via NEE?
                float cosLight = std::abs(Vec3::dot(hit.normal, currentRay.d));
                if (cosLight > EPSILON) {
                    // Light PDF in solid angle measure
                    float lightPdf = (hit.t * hit.t) / (sceneLights.totalArea * cosLight);
                    emissionWeight = powerHeuristic(lastBsdfPdf, lightPdf);
                }
            }
            result = result + throughput * emission * emissionWeight;
        }
        
        // ===== Next Event Estimation with MIS =====
        // Skip for emissive surfaces and transmission materials
        // (transmission materials are handled via BSDF sampling only)
        bool isEmissive = emissionStrength > EPSILON;
        bool isTransmissive = mat.transmission > 0.5f;
        if (sceneLights.hasLights() && !isEmissive && !isTransmissive) {
            float lightSelectProb;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            
            if (lightIdx >= 0 && lightSelectProb > EPSILON) {
                const Light& light = sceneLights.lights[lightIdx];
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > EPSILON) {
                    // NdotL must be positive (light on same side as normal)
                    float NdotL = Vec3::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > EPSILON) {
                        // Shadow test - offset in light direction to avoid self-intersection
                        Ray shadowRay{hit.point + ls.direction * 0.001f, ls.direction};
                        Hit shadowHit = intersectScene(scene, shadowRay, true);
                        
                        // Use larger tolerance to avoid artifacts from light source self-intersection
                        bool inShadow = shadowHit.hit && shadowHit.t < ls.distance - 0.01f;
                        
                        if (!inShadow) {
                            Vec3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            float pdfLight = ls.pdf * lightSelectProb;
                            float pdfBsdf = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            // MIS weight
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
        
        // ===== BSDF Sampling for next bounce =====
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < EPSILON && !bsdfSample.useWeight) {
            break;
        }
        
        // Update throughput
        if (bsdfSample.useWeight) {
            // Direct weight (for glass etc.) - already computed as f*|NdotL|/pdf
            throughput = Vec3(
                throughput.x * bsdfSample.weight.x,
                throughput.y * bsdfSample.weight.y,
                throughput.z * bsdfSample.weight.z
            );
        } else {
            // Standard: f * |NdotL| / pdf
            float NdotL = std::abs(Vec3::dot(sampleNormal, bsdfSample.wi));
            if (NdotL > EPSILON && bsdfSample.pdf > EPSILON) {
                throughput = Vec3(
                    throughput.x * bsdfSample.f.x * NdotL / bsdfSample.pdf,
                    throughput.y * bsdfSample.f.y * NdotL / bsdfSample.pdf,
                    throughput.z * bsdfSample.f.z * NdotL / bsdfSample.pdf
                );
            } else {
                break;
            }
        }
        
        // Russian Roulette after depth 3
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
        if (bsdfSample.type == BSDFSample::TRANSMISSION) {
            currentRay.o = hit.point + bsdfSample.wi * 0.001f;
        } else {
            Vec3 offsetNormal = (Vec3::dot(bsdfSample.wi, sampleNormal) > 0) ? sampleNormal : sampleNormal * -1.0f;
            currentRay.o = hit.point + offsetNormal * 0.001f;
        }
        currentRay.d = bsdfSample.wi;
        
        // Store BSDF PDF for next emission's MIS weight calculation
        // For transmission (useWeight=true), set to 0 to disable MIS (pdf is not meaningful)
        lastBsdfPdf = bsdfSample.useWeight ? 0.0f : bsdfSample.pdf;
    }
    
    return result;
}
