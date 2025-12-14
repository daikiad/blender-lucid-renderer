/**
 * fresnel.hpp - Fresnel Reflectance Functions
 * ============================================
 * 
 * Fresnel equations determine the ratio of reflected vs transmitted light:
 * - Schlick approximation for conductors (metals)
 * - Exact Fresnel for dielectrics (glass)
 * 
 * Physical Units:
 * - All inputs/outputs are dimensionless (ratios, cosines, reflectances)
 * - F0 (base reflectance at normal incidence) is dimensionless [0,1]
 * - eta (IOR ratio) is dimensionless (n1/n2)
 * - cosTheta is dimensionless (cos of angle)
 * - Return values are reflectance fractions [0,1]
 */

#pragma once
#include "../units/render_units.hpp"
#include <cmath>
#include <algorithm>

// ========== Constants ==========

constexpr float FRESNEL_EPSILON = 1e-6f;

// ========== Fresnel Functions ==========

/**
 * Schlick's approximation for Fresnel reflectance (dielectric)
 * F0 = ((n1 - n2) / (n1 + n2))^2
 */
inline float fresnelSchlick(float cosTheta, float f0) {
    return f0 + (1.0f - f0) * std::pow(1.0f - cosTheta, 5.0f);
}

/**
 * Fresnel for conductor (metal) - Schlick with F0 = albedo
 * Returns per-channel Fresnel reflectance
 */
inline render::ColorRGB fresnelSchlickColor(float cosTheta, const render::ColorRGB& f0) {
    float f = std::pow(1.0f - cosTheta, 5.0f);
    // f0.r etc are Reflectance (quantity<one>), so arithmetic works naturally
    return render::ColorRGB{
        f0.r + (1.0f - f0.r) * f,
        f0.g + (1.0f - f0.g) * f,
        f0.b + (1.0f - f0.b) * f
    };
}

/**
 * Exact Fresnel for dielectric
 * 
 * Computes Fresnel reflectance using exact dielectric Fresnel equations.
 * Uses Snell's law: sin(θ_t) = eta * sin(θ_i)
 * 
 * @param cosThetaI cosine of incident angle (positive, will be clamped)
 * @param eta ratio n_incident / n_transmitted
 * 
 * Examples:
 *   - Air (n=1.0) → Glass (n=1.5): eta = 1.0/1.5 = 0.667
 *   - Glass (n=1.5) → Air (n=1.0): eta = 1.5/1.0 = 1.5
 * 
 * Total Internal Reflection (TIR):
 *   TIR occurs when eta > 1 and sin(θ_i) > 1/eta (i.e., going from dense to sparse medium).
 *   At TIR, returns 1.0 (perfect reflection).
 *   Critical angle: θ_c = arcsin(1/eta), e.g., ~41.8° for glass→air.
 * 
 * Usage in BSDF sampling (see bsdf.hpp):
 *   float eta = frontFace ? (1.0f / ior) : ior;
 *   float F = fresnelDielectric(cosThetaI, eta);
 * 
 * @return Fresnel reflectance [0, 1]
 */
inline float fresnelDielectric(float cosThetaI, float eta) {
    // Clamp cosThetaI to valid range
    cosThetaI = std::clamp(std::abs(cosThetaI), 0.0f, 1.0f);
    
    // Compute sin^2(theta_t) using Snell's law
    float sinThetaI2 = 1.0f - cosThetaI * cosThetaI;
    float sinThetaT2 = eta * eta * sinThetaI2;
    
    if (sinThetaT2 >= 1.0f) {
        // Total internal reflection
        return 1.0f;
    }
    
    float cosThetaT = std::sqrt(std::max(0.0f, 1.0f - sinThetaT2));
    
    // Fresnel equations for s and p polarization
    float etaCosI = eta * cosThetaI;
    float etaCosT = eta * cosThetaT;
    
    float rs = (etaCosI - cosThetaT) / (etaCosI + cosThetaT + FRESNEL_EPSILON);
    float rp = (cosThetaI - etaCosT) / (cosThetaI + etaCosT + FRESNEL_EPSILON);
    
    return 0.5f * (rs * rs + rp * rp);
}
