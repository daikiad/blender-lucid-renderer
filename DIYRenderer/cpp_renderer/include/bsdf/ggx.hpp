/**
 * ggx.hpp - GGX/Trowbridge-Reitz Microfacet Model
 * ================================================
 * 
 * GGX microfacet distribution for physically-based rendering:
 * - Normal Distribution Function (NDF)
 * - Geometry/Shadowing functions (Smith G1, G2)
 * - VNDF (Visible Normal Distribution Function) sampling
 */

#pragma once
#include "../math/vec3.hpp"
#include <cmath>
#include <algorithm>

// ========== Constants ==========

constexpr float GGX_PI = 3.14159265358979323846f;
constexpr float GGX_INV_PI = 0.31830988618379067154f;
constexpr float GGX_EPSILON = 1e-6f;

// Minimum roughness to avoid delta distributions
constexpr float MIN_ROUGHNESS = 0.01f;

// ========== Utility Functions ==========

inline float ggx_sqr(float x) { return x * x; }

inline float ggx_safe_sqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

// Build orthonormal basis from normal (Duff et al. 2017)
inline void buildOrthonormalBasis(const Vec3& n, Vec3& tangent, Vec3& bitangent) {
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

// ========== GGX Normal Distribution Function ==========

/**
 * GGX NDF: D(m) = α² / (π * (cos²θ * (α² - 1) + 1)²)
 * In Blender/Cycles, alpha = roughness², so a2 = roughness⁴
 */
inline float ggxD(float NdotH, float roughness) {
    float alpha = roughness * roughness;  // α = roughness²
    float a2 = alpha * alpha;             // α² = roughness⁴
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 * GGX_INV_PI / (denom * denom + GGX_EPSILON);
}

// ========== GGX Geometry Functions ==========

/**
 * Smith G1: G1(v) = 2 * NdotV / (NdotV + sqrt(α² + (1 - α²) * NdotV²))
 */
inline float ggxG1(float NdotV, float roughness) {
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    return 2.0f * NdotV / (NdotV + ggx_safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
}

/**
 * Height-correlated Smith G2
 */
inline float ggxG2(float NdotL, float NdotV, float roughness) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return 0.0f;
    
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    
    float lambdaL = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    
    float denom = NdotV * lambdaL + NdotL * lambdaV;
    if (denom < GGX_EPSILON) return 0.0f;
    
    return 2.0f * NdotL * NdotV / denom;
}

// ========== GGX VNDF Sampling (Heitz 2018) ==========

/**
 * Sample GGX microfacet normal using VNDF
 * Returns half-vector in world space
 */
inline Vec3 sampleGGXVNDF(const Vec3& wo, float roughness, float u1, float u2, 
                          const Vec3& n, const Vec3& t, const Vec3& b) {
    // Transform wo to local space
    Vec3 woLocal = worldToLocal(wo, n, t, b);
    
    // Stretch wo by alpha
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
    float phi = 2.0f * GGX_PI * u2;
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

/**
 * PDF for GGX VNDF sampling
 */
inline float pdfGGXVNDF(const Vec3& wo, const Vec3& h, float roughness, const Vec3& n) {
    float NdotH = Vec3::dot(n, h);
    float VdotH = Vec3::dot(wo, h);
    float NdotV = Vec3::dot(n, wo);
    if (NdotH <= 0.0f || VdotH <= 0.0f || NdotV <= 0.0f) return 0.0f;
    
    float D = ggxD(NdotH, roughness);
    float G1 = ggxG1(NdotV, roughness);
    
    return D * G1 / (4.0f * NdotV);
}

// ========== Cosine-Weighted Hemisphere Sampling ==========

/**
 * Sample direction with cosine-weighted distribution
 * PDF = cos(θ) / π
 */
inline Vec3 sampleCosineHemisphere(float u1, float u2, const Vec3& n) {
    float r = std::sqrt(u1);
    float theta = 2.0f * GGX_PI * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    
    Vec3 t, b;
    buildOrthonormalBasis(n, t, b);
    
    return localToWorld(Vec3(x, y, z), n, t, b);
}

inline float pdfCosineHemisphere(float NdotL) {
    return std::max(0.0f, NdotL) * GGX_INV_PI;
}
