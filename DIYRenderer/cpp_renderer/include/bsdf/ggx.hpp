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
#include "../math/vec3_unit.hpp"
#include "../units/units.hpp"
#include <cmath>
#include <algorithm>
#include <numbers>

// ========== Constants ==========

constexpr float GGX_PI = std::numbers::pi_v<float>;
constexpr float GGX_INV_PI = std::numbers::inv_pi_v<float>;
constexpr float GGX_EPSILON = 1e-6f;

// Minimum roughness to avoid delta distributions (dimensionless: 0-1)
constexpr float MIN_ROUGHNESS = 0.01f;

// Unit-typed angle constants (for documentation/reference)
// Note: Most BSDF calculations use cosines, which are dimensionless
namespace ggx_units {
    using namespace diy::units;
    inline constexpr auto pi_rad = diy::units::pi_rad;           // π radians
    inline constexpr auto two_pi_rad = diy::units::two_pi_rad;   // 2π radians
    inline constexpr auto hemisphere = diy::units::hemisphere_sr; // 2π steradians
}

// ========== Utility Functions ==========

inline float ggx_sqr(float x) { return x * x; }

inline float ggx_safe_sqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

// Build orthonormal basis from normal (Duff et al. 2017)
inline void buildOrthonormalBasis(const diy::Direction3& n, diy::Direction3& tangent, diy::Direction3& bitangent) {
    if (n.z_raw() < -0.9999f) {
        tangent = diy::Direction3(0.0f, -1.0f, 0.0f);
        bitangent = diy::Direction3(-1.0f, 0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + n.z_raw());
    float b = -n.x_raw() * n.y_raw() * a;
    tangent = diy::Direction3(1.0f - n.x_raw() * n.x_raw() * a, b, -n.x_raw());
    bitangent = diy::Direction3(b, 1.0f - n.y_raw() * n.y_raw() * a, -n.y_raw());
}

// Transform direction from local space (z-up) to world space
inline diy::Direction3 localToWorld(const diy::Direction3& local, const diy::Direction3& n, 
                                     const diy::Direction3& t, const diy::Direction3& b) {
    return diy::Direction3(
        t.x_raw() * local.x_raw() + b.x_raw() * local.y_raw() + n.x_raw() * local.z_raw(),
        t.y_raw() * local.x_raw() + b.y_raw() * local.y_raw() + n.y_raw() * local.z_raw(),
        t.z_raw() * local.x_raw() + b.z_raw() * local.y_raw() + n.z_raw() * local.z_raw()
    );
}

// Transform direction from world space to local space (z-up)
inline diy::Direction3 worldToLocal(const diy::Direction3& world, const diy::Direction3& n, 
                                     const diy::Direction3& t, const diy::Direction3& b) {
    return diy::Direction3(
        diy::dot(world, t).numerical_value_in(mp_units::one),
        diy::dot(world, b).numerical_value_in(mp_units::one),
        diy::dot(world, n).numerical_value_in(mp_units::one)
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
inline diy::Direction3 sampleGGXVNDF(const diy::Direction3& wo, float roughness, float u1, float u2, 
                                      const diy::Direction3& n, const diy::Direction3& t, const diy::Direction3& b) {
    // Transform wo to local space
    diy::Direction3 woLocal = worldToLocal(wo, n, t, b);
    
    // Stretch wo by alpha
    float alpha = roughness * roughness;
    diy::Direction3 woStretched(woLocal.x_raw() * alpha, woLocal.y_raw() * alpha, woLocal.z_raw());
    woStretched.normalize();
    
    // Build orthonormal basis around stretched wo
    diy::Direction3 t1 = (woStretched.z_raw() < 0.9999f) ? 
        diy::cross(diy::Direction3(0, 0, 1), woStretched) : diy::Direction3(1, 0, 0);
    t1.normalize();
    diy::Direction3 t2 = diy::cross(woStretched, t1);
    
    // Sample point on disk
    float r = std::sqrt(u1);
    float phi = 2.0f * GGX_PI * u2;
    float p1 = r * std::cos(phi);
    float p2 = r * std::sin(phi);
    float s = 0.5f * (1.0f + woStretched.z_raw());
    p2 = (1.0f - s) * std::sqrt(1.0f - p1 * p1) + s * p2;
    
    // Compute half-vector
    diy::Direction3 hLocal = t1 * p1 + t2 * p2 + woStretched * std::sqrt(std::max(0.0f, 1.0f - p1*p1 - p2*p2));
    
    // Unstretch
    hLocal = diy::Direction3(hLocal.x_raw() * alpha, hLocal.y_raw() * alpha, std::max(0.0f, hLocal.z_raw()));
    hLocal.normalize();
    
    // Transform back to world space
    return localToWorld(hLocal, n, t, b);
}

/**
 * PDF for GGX VNDF sampling
 */
inline float pdfGGXVNDF(const diy::Direction3& wo, const diy::Direction3& h, float roughness, const diy::Direction3& n) {
    float NdotH = diy::dot(n, h).numerical_value_in(mp_units::one);
    float VdotH = diy::dot(wo, h).numerical_value_in(mp_units::one);
    float NdotV = diy::dot(n, wo).numerical_value_in(mp_units::one);
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
inline diy::Direction3 sampleCosineHemisphere(float u1, float u2, const diy::Direction3& n) {
    float r = std::sqrt(u1);
    float theta = 2.0f * GGX_PI * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    
    diy::Direction3 t, b;
    buildOrthonormalBasis(n, t, b);
    
    return localToWorld(diy::Direction3(x, y, z), n, t, b);
}

inline float pdfCosineHemisphere(float NdotL) {
    return std::max(0.0f, NdotL) * GGX_INV_PI;
}
