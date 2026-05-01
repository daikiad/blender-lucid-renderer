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
#include "../units/render_units.hpp"
#include <cmath>
#include <algorithm>
#include <numbers>

// ========== Constants ==========

constexpr float GGX_PI = std::numbers::pi_v<float>;
constexpr float GGX_INV_PI = std::numbers::inv_pi_v<float>;
constexpr float GGX_EPSILON = 1e-6f;

// Minimum roughness to avoid delta distributions (dimensionless: 0-1)
constexpr float MIN_ROUGHNESS = 0.01f;

// ========== Utility Functions ==========

inline float ggx_sqr(float x) { return x * x; }

inline float ggx_safe_sqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

// Build orthonormal basis from normal (Duff et al. 2017)
inline void buildOrthonormalBasis(const render::Direction& n, render::Direction& tangent, render::Direction& bitangent) {
    if (n.z() < -0.9999f) {
        tangent = render::direction_from_unit_vector(render::Vec3f(0.0f, -1.0f, 0.0f));
        bitangent = render::direction_from_unit_vector(render::Vec3f(-1.0f, 0.0f, 0.0f));
        return;
    }
    float a = 1.0f / (1.0f + n.z());
    float b = -n.x() * n.y() * a;
    tangent = render::make_direction_or_default(render::Vec3f(1.0f - n.x() * n.x() * a, b, -n.x()));
    bitangent = render::make_direction_or_default(render::Vec3f(b, 1.0f - n.y() * n.y() * a, -n.y()));
}

// Transform direction from local space (z-up) to world space
inline render::Direction localToWorld(const render::Direction& local, const render::Direction& n, 
                                       const render::Direction& t, const render::Direction& b) {
    return render::make_direction_or_default(render::Vec3f(
        t.x() * local.x() + b.x() * local.y() + n.x() * local.z(),
        t.y() * local.x() + b.y() * local.y() + n.y() * local.z(),
        t.z() * local.x() + b.z() * local.y() + n.z() * local.z()
    ));
}

// Transform direction from world space to local space (z-up)
inline render::Direction worldToLocal(const render::Direction& world, const render::Direction& n, 
                                       const render::Direction& t, const render::Direction& b) {
    return render::make_direction_or_default(render::Vec3f(
        render::dot(world.vec(), t.vec()),
        render::dot(world.vec(), b.vec()),
        render::dot(world.vec(), n.vec())
    ));
}

// ========== GGX Normal Distribution Function ==========

/**
 * GGX/Trowbridge-Reitz Normal Distribution Function
 * 
 * D(m) = α² / (π * (cos²θ * (α² - 1) + 1)²)
 * 
 * In Blender/Cycles convention: alpha = roughness², so α² = roughness⁴
 * 
 * At peak (NdotH=1): D = 1 / (π * α²)
 *   - Smoother surfaces (small α) → higher D at peak
 *   - Rougher surfaces (large α) → lower D at peak, but wider distribution
 * 
 * Note: GGX_EPSILON is added to denominator for numerical stability.
 * This affects very smooth surfaces (roughness < ~0.1) where theoretical D
 * would be extremely large. For roughness=0.01, theoretical D ≈ 3e7 but
 * actual D ≈ 0.003 due to epsilon clamping.
 * 
 * @param NdotH dot product of normal and half-vector [0, 1]
 * @param roughness surface roughness [0, 1], clamped to MIN_ROUGHNESS in practice
 * @return NDF value (not a probability, can be > 1)
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
inline render::Direction sampleGGXVNDF(const render::Direction& wo, float roughness, float u1, float u2, 
                                        const render::Direction& n, const render::Direction& t, const render::Direction& b) {
    // Transform wo to local space
    render::Direction woLocal = worldToLocal(wo, n, t, b);
    
    // Stretch wo by alpha
    float alpha = roughness * roughness;
    auto woStretchedVec = render::Vec3f(woLocal.x() * alpha, woLocal.y() * alpha, woLocal.z());
    render::Direction woStretched = render::make_direction_or_default(woStretchedVec);
    
    // Build orthonormal basis around stretched wo
    render::Direction t1;
    if (woStretched.z() < 0.9999f) {
        auto cross_opt = render::make_direction(render::cross(render::Vec3f(0, 0, 1), woStretched.vec()));
        t1 = cross_opt ? *cross_opt : render::direction_from_unit_vector(render::Vec3f(1, 0, 0));
    } else {
        t1 = render::direction_from_unit_vector(render::Vec3f(1, 0, 0));
    }
    auto t2_opt = woStretched.cross(t1);
    render::Direction t2 = t2_opt ? *t2_opt : render::direction_from_unit_vector(render::Vec3f(0, 1, 0));
    
    // Sample point on disk
    float r = std::sqrt(u1);
    float phi = 2.0f * GGX_PI * u2;
    float p1 = r * std::cos(phi);
    float p2 = r * std::sin(phi);
    float s = 0.5f * (1.0f + woStretched.z());
    p2 = (1.0f - s) * std::sqrt(1.0f - p1 * p1) + s * p2;
    
    // Compute half-vector
    render::Vec3f hLocalVec = t1.vec() * p1 + t2.vec() * p2 + woStretched.vec() * std::sqrt(std::max(0.0f, 1.0f - p1*p1 - p2*p2));
    
    // Unstretch
    hLocalVec = render::Vec3f(hLocalVec.x * alpha, hLocalVec.y * alpha, std::max(0.0f, hLocalVec.z));
    render::Direction hLocal = render::make_direction_or_default(hLocalVec);
    
    // Transform back to world space
    return localToWorld(hLocal, n, t, b);
}

/**
 * PDF for GGX VNDF sampling
 */
inline float pdfGGXVNDF(const render::Direction& wo, const render::Direction& h, float roughness, const render::Direction& n) {
    float NdotH = render::dot(n.vec(), h.vec());
    float VdotH = render::dot(wo.vec(), h.vec());
    float NdotV = render::dot(n.vec(), wo.vec());
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
inline render::Direction sampleCosineHemisphere(float u1, float u2, const render::Direction& n) {
    float r = std::sqrt(u1);
    float theta = 2.0f * GGX_PI * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    
    render::Direction t, b;
    buildOrthonormalBasis(n, t, b);
    
    return localToWorld(render::direction_from_unit_vector(render::Vec3f(x, y, z)), n, t, b);
}

inline float pdfCosineHemisphere(float NdotL) {
    return std::max(0.0f, NdotL) * GGX_INV_PI;
}
