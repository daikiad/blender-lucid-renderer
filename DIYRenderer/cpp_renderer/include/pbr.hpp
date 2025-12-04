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
inline float ggxD(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 * INV_PI / (denom * denom);
}

// GGX Geometry function (Smith's method with height-correlated G2)
// G1(v) = 2 * NdotV / (NdotV + sqrt(α² + (1 - α²) * NdotV²))
inline float ggxG1(float NdotV, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    return 2.0f * NdotV / (NdotV + safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
}

// Height-correlated Smith G2 (returns G2, NOT G2/(4*NdotL*NdotV))
inline float ggxG2(float NdotL, float NdotV, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ggxL = NdotV * safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float ggxV = NdotL * safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    float denom = ggxL + ggxV;
    if (denom < EPSILON) return 0.0f;
    return 2.0f * NdotL * NdotV / denom;
}

// Sample GGX microfacet normal (VNDF sampling - Heitz 2018)
// Returns half-vector in world space
inline Vec3 sampleGGXVNDF(const Vec3& wo, float roughness, float u1, float u2, 
                          const Vec3& n, const Vec3& t, const Vec3& b) {
    // Transform wo to local space
    Vec3 woLocal = worldToLocal(wo, n, t, b);
    
    // Stretch wo
    float a = roughness * roughness;
    Vec3 woStretched = Vec3(woLocal.x * a, woLocal.y * a, woLocal.z);
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
    hLocal = Vec3(hLocal.x * a, hLocal.y * a, std::max(0.0f, hLocal.z));
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
inline Vec3 evalSpecular(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, 
                         const Vec3& n, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3(0, 0, 0);
    
    Vec3 h = wo + wi;
    h.normalize();
    float NdotH = std::max(Vec3::dot(n, h), 0.0f);
    float VdotH = std::max(Vec3::dot(wo, h), 0.0f);
    
    // Clamp roughness to avoid singularity
    float roughness = std::max(mat.roughness, 0.04f);
    
    // Microfacet terms
    float D = ggxD(NdotH, roughness);
    float G = ggxG2(NdotL, NdotV, roughness);
    
    // Fresnel term - for metals use albedo as F0, for dielectrics use 0.04
    Vec3 f0 = mat.albedo * mat.metallic + Vec3(0.04f, 0.04f, 0.04f) * (1.0f - mat.metallic);
    Vec3 F = fresnelSchlickVec3(VdotH, f0);
    
    // f = D * G * F / (4 * NdotL * NdotV)
    float denom = 4.0f * NdotL * NdotV;
    if (denom < EPSILON) return Vec3(0, 0, 0);
    float spec = D * G / denom;
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
    Vec3 f;         // BSDF value
    float pdf;      // Probability density
    bool isDelta;   // Is this a delta distribution (perfect mirror/glass)?
    enum Type { DIFFUSE, SPECULAR, TRANSMISSION } type;
    
    BSDFSample() : wi(), f(), pdf(0.0f), isDelta(false), type(DIFFUSE) {}
};

// Sample BSDF direction based on material properties
inline BSDFSample sampleBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& n, 
                              float u1, float u2, float u3) {
    BSDFSample sample;
    float NdotV = std::max(Vec3::dot(n, wo), EPSILON);
    
    // Handle glass/transmission
    if (mat.transmission > 0.0f) {
        if (u3 < mat.transmission) {
            // Glass path
            // wo points away from surface (toward camera/eye)
            // n is the geometric normal (pointing outward from surface)
            
            // Check if we're on the front face (outside) or back face (inside)
            bool frontFace = Vec3::dot(wo, n) > 0.0f;
            
            // outward_normal always points to the "outside" of the object
            Vec3 outward_normal = frontFace ? n : n * -1.0f;
            
            // eta = n_incident / n_transmitted
            // frontFace (entering glass): air(1) -> glass(ior), eta = 1/ior
            // backFace (exiting glass): glass(ior) -> air(1), eta = ior
            float eta = frontFace ? (1.0f / mat.ior) : mat.ior;
            
            // incident direction points INTO the surface
            Vec3 incident = wo * -1.0f;
            
            // cos of incident angle (positive value)
            float cosThetaI = std::abs(Vec3::dot(incident, outward_normal));
            float F = fresnelDielectric(cosThetaI, eta);
            
            if (u1 < F) {
                // Reflect
                sample.wi = reflect(incident, outward_normal);
                sample.f = Vec3(1, 1, 1);
                sample.pdf = F * mat.transmission;
                sample.isDelta = true;
                sample.type = BSDFSample::SPECULAR;
            } else {
                // Refract
                Vec3 refracted = refract(incident, outward_normal, eta);
                if (refracted.length() > 0.5f) {
                    sample.wi = refracted;
                    sample.f = Vec3(1, 1, 1);
                    sample.pdf = (1.0f - F) * mat.transmission;
                    sample.isDelta = true;
                    sample.type = BSDFSample::TRANSMISSION;
                } else {
                    // Total internal reflection
                    sample.wi = reflect(incident, outward_normal);
                    sample.f = Vec3(1, 1, 1);
                    sample.pdf = mat.transmission;
                    sample.isDelta = true;
                    sample.type = BSDFSample::SPECULAR;
                }
            }
            return sample;
        }
        // Else continue with opaque BSDF (scaled by 1 - transmission)
    }
    
    // Decide between diffuse and specular sampling based on roughness and metallic
    // Probability of specular sampling increases with metallic and decreases with roughness
    float specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
    specProb = clampf(specProb, 0.25f, 0.75f);
    
    Vec3 t, b;
    buildOrthonormalBasis(n, t, b);
    
    if (u3 >= mat.transmission && u1 < specProb) {
        // Specular sampling using GGX VNDF
        float roughness = std::max(mat.roughness, 0.04f);
        
        Vec3 h = sampleGGXVNDF(wo, roughness, u1 / specProb, u2, n, t, b);
        sample.wi = reflect(wo * -1.0f, h);
        
        float NdotL = Vec3::dot(n, sample.wi);
        if (NdotL <= 0.0f) {
            sample.pdf = 0.0f;
            sample.f = Vec3(0, 0, 0);
            return sample;
        }
        
        // Evaluate full BSDF
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        
        // PDF is GGX VNDF pdf * specProb + diffuse pdf * (1 - specProb)
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        sample.pdf = specProb * pdfSpec + (1.0f - specProb) * pdfDiff;
        sample.pdf *= (1.0f - mat.transmission);
        sample.isDelta = false;
        sample.type = BSDFSample::SPECULAR;
    } else {
        // Diffuse sampling using cosine hemisphere
        float u1Adjusted = (u1 - specProb) / (1.0f - specProb);
        sample.wi = sampleCosineHemisphere(u1Adjusted, u2, n);
        
        float NdotL = Vec3::dot(n, sample.wi);
        if (NdotL <= 0.0f) {
            sample.pdf = 0.0f;
            sample.f = Vec3(0, 0, 0);
            return sample;
        }
        
        // Evaluate full BSDF
        sample.f = evalBSDF(mat, wo, sample.wi, n);
        
        // Combined PDF
        float roughness = std::max(mat.roughness, 0.04f);
        Vec3 h = wo + sample.wi;
        h.normalize();
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        sample.pdf = specProb * pdfSpec + (1.0f - specProb) * pdfDiff;
        sample.pdf *= (1.0f - mat.transmission);
        sample.isDelta = false;
        sample.type = BSDFSample::DIFFUSE;
    }
    
    return sample;
}

// Get PDF for a given direction
inline float pdfBSDF(const MaterialParams& mat, const Vec3& wo, const Vec3& wi, const Vec3& n) {
    float NdotL = Vec3::dot(n, wi);
    if (NdotL <= 0.0f) return 0.0f;
    
    // Same probability distribution as sampling
    float specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
    specProb = clampf(specProb, 0.25f, 0.75f);
    
    float roughness = std::max(mat.roughness, 0.04f);
    Vec3 h = wo + wi;
    h.normalize();
    
    float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
    float pdfDiff = pdfCosineHemisphere(NdotL);
    
    return (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
}

// ========== Light Sampling ==========

struct Light {
    Vec3 position;      // Center position (for area lights, center of triangle)
    Vec3 normal;        // Light surface normal
    Vec3 emission;      // Emission color * strength
    float area;         // Surface area
    int meshIndex;      // Index of emissive mesh
    int triangleIndex;  // Index of emissive triangle
    
    // Triangle vertices (for area light sampling)
    Vec3 v0, v1, v2;
    
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

// Sample a random point on a light source
inline LightSample sampleLight(const Light& light, const Vec3& shadingPoint, 
                                float u1, float u2) {
    LightSample sample;
    
    // Sample point on triangle
    sample.position = sampleTrianglePoint(light.v0, light.v1, light.v2, u1, u2);
    sample.normal = light.normal;
    sample.emission = light.emission;
    
    // Compute direction and distance
    Vec3 toLight = sample.position - shadingPoint;
    sample.distance = toLight.length();
    sample.direction = toLight * (1.0f / sample.distance);
    
    // PDF in solid angle measure:
    // pdf(ω) = pdf(A) * r² / |cos(θ')|
    // where pdf(A) = 1 / area
    float cosLight = std::abs(Vec3::dot(sample.normal, sample.direction * -1.0f));
    if (cosLight < EPSILON) {
        sample.pdf = 0.0f;
    } else {
        sample.pdf = (sample.distance * sample.distance) / (light.area * cosLight);
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
                emission = getEmissionFromNodeTree(mesh.material.nodeTree);
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
        
        // Build CDF for light selection
        cdf.resize(lights.size());
        float cumulative = 0.0f;
        for (size_t i = 0; i < lights.size(); ++i) {
            cumulative += lights[i].area;
            cdf[i] = cumulative / totalArea;
        }
    }
    
    // Select a light based on area-weighted probability
    // Returns index and adjusts pdf accordingly
    int selectLight(float u, float& pdf) const {
        if (lights.empty()) {
            pdf = 0.0f;
            return -1;
        }
        
        // Binary search in CDF
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        int idx = (int)(it - cdf.begin());
        idx = std::min(idx, (int)lights.size() - 1);
        
        // PDF is proportional to area
        pdf = lights[idx].area / totalArea;
        
        return idx;
    }
    
    bool hasLights() const { return !lights.empty(); }
};

// ========== Path Tracing with MIS ==========

// Trace a path with Multiple Importance Sampling
// Combines BSDF sampling (indirect) with Next Event Estimation (direct light sampling)
inline Vec3 traceMIS(const Scene& scene, const SceneLights& sceneLights, 
                     const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    bool specularBounce = true;  // First hit can see lights directly
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        Hit hit = intersectScene(scene, currentRay, true);
        
        if (!hit.hit) {
            // Ray escaped - add environment
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
            mat.albedo = getAlbedoFromNodeTree(hit.material.nodeTree);
            mat.metallic = getMetallicFromNodeTree(hit.material.nodeTree);
            mat.roughness = getRoughnessFromNodeTree(hit.material.nodeTree);
            mat.transmission = getTransmissionFromNodeTree(hit.material.nodeTree);
            mat.ior = getIORFromNodeTree(hit.material.nodeTree);
        }
        
        // Get emission
        Vec3 emission = hit.material.emission;
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            emission = getEmissionFromNodeTree(hit.material.nodeTree);
        }
        
        // Orient normal to face the ray (for non-glass materials)
        Vec3 n = hit.normal;
        Vec3 wo = currentRay.d * -1.0f;
        wo.normalize();
        bool frontFace = Vec3::dot(wo, n) > 0;
        
        // For glass, we need the original geometric normal for inside/outside detection
        // For opaque materials, flip normal to face the ray
        Vec3 shadingNormal = n;
        if (mat.transmission < 0.5f && !frontFace) {
            shadingNormal = n * -1.0f;
        }
        
        // Add emission with MIS weight
        // On first hit or after specular bounce: full emission (no NEE was done)
        // After non-specular bounce: apply MIS weight
        float emissionWeight = 1.0f;
        if (!specularBounce && sceneLights.hasLights() && depth > 0) {
            // We hit a light via BSDF sampling - need MIS weight
            // Find which light we hit (if any)
            float emissionStrength = emission.x + emission.y + emission.z;
            if (emissionStrength > EPSILON) {
                // Approximate: use a representative light PDF
                // For proper MIS, we'd need to find the exact triangle
                // For now, treat as if we sampled this point uniformly from scene lights
                // This is approximate but avoids fireflies
                emissionWeight = 0.5f;  // Rough balance between BSDF and light sampling
            }
        }
        result = result + throughput * emission * emissionWeight;
        
        // ===== Next Event Estimation (Direct Light Sampling) =====
        // Only for non-delta materials (skip for glass)
        if (sceneLights.hasLights() && mat.transmission < 0.5f) {
            // Select a light
            float lightSelectPdf;
            int lightIdx = sceneLights.selectLight(randf(), lightSelectPdf);
            
            if (lightIdx >= 0) {
                const Light& light = sceneLights.lights[lightIdx];
                
                // Sample point on light
                LightSample ls = sampleLight(light, hit.point, randf(), randf());
                
                if (ls.pdf > EPSILON) {
                    float NdotL = Vec3::dot(shadingNormal, ls.direction);
                    
                    if (NdotL > 0.0f) {
                        // Shadow test
                        Ray shadowRay{hit.point + shadingNormal * 0.001f, ls.direction};
                        Hit shadowHit = intersectScene(scene, shadowRay, true);
                        
                        bool inShadow = shadowHit.hit && shadowHit.t < ls.distance - 0.001f;
                        
                        if (!inShadow) {
                            // Evaluate BSDF
                            Vec3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                            
                            // MIS weight: light sampling vs BSDF sampling
                            float pdfLight = ls.pdf * lightSelectPdf;
                            float pdfBsdf = pdfBSDF(mat, wo, ls.direction, shadingNormal);
                            float misWeight = powerHeuristic(pdfLight, pdfBsdf);
                            
                            // Add contribution: f * Li * cos(θ) * misWeight / pdfLight
                            Vec3 directLight = ls.emission;
                            Vec3 contrib = Vec3(
                                f.x * directLight.x * NdotL * misWeight / pdfLight,
                                f.y * directLight.y * NdotL * misWeight / pdfLight,
                                f.z * directLight.z * NdotL * misWeight / pdfLight
                            );
                            result = result + throughput * contrib;
                        }
                    }
                }
            }
        }
        
        // ===== BSDF Sampling (for next bounce) =====
        // For glass, use original geometric normal (n) for correct inside/outside detection
        // For opaque materials, use shading normal (shadingNormal)
        Vec3 sampleNormal = (mat.transmission > 0.5f) ? n : shadingNormal;
        BSDFSample bsdfSample = sampleBSDF(mat, wo, sampleNormal, randf(), randf(), randf());
        
        if (bsdfSample.pdf < EPSILON) {
            break;
        }
        
        // Update throughput
        // For delta distributions: throughput *= f (pdf is implicit, no cos term needed)
        // For continuous: throughput *= f * cos(θ) / pdf
        if (bsdfSample.isDelta) {
            // Delta BSDF: f gives the throughput directly
            throughput = Vec3(
                throughput.x * bsdfSample.f.x,
                throughput.y * bsdfSample.f.y,
                throughput.z * bsdfSample.f.z
            );
        } else {
            float NdotL = std::max(Vec3::dot(sampleNormal, bsdfSample.wi), 0.0f);
            if (NdotL > 0.0f && bsdfSample.pdf > EPSILON) {
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
            if (randf() > rrProb) {
                break;
            }
            throughput = throughput * (1.0f / rrProb);
        }
        
        // Check for NaN/Inf
        if (std::isnan(throughput.x) || std::isinf(throughput.x) ||
            std::isnan(throughput.y) || std::isinf(throughput.y) ||
            std::isnan(throughput.z) || std::isinf(throughput.z)) {
            break;
        }
        
        // Offset ray origin to avoid self-intersection
        if (bsdfSample.type == BSDFSample::TRANSMISSION) {
            // For transmission, offset along the ray direction (into the surface)
            currentRay.o = hit.point + bsdfSample.wi * 0.001f;
        } else {
            // For reflection/diffuse, offset along the normal (away from surface)
            Vec3 offsetNormal = (Vec3::dot(bsdfSample.wi, sampleNormal) > 0) ? sampleNormal : sampleNormal * -1.0f;
            currentRay.o = hit.point + offsetNormal * 0.001f;
        }
        currentRay.d = bsdfSample.wi;
        
        specularBounce = bsdfSample.isDelta;
    }
    
    return result;
}
