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
#include "../math/vec3_unit.hpp"
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
inline diy::Vec3U<mp_units::one> fresnelSchlickColor(float cosTheta, const diy::Vec3U<mp_units::one>& f0) {
    float f = std::pow(1.0f - cosTheta, 5.0f);
    return diy::Vec3U<mp_units::one>(
        f0.x_raw() + (1.0f - f0.x_raw()) * f,
        f0.y_raw() + (1.0f - f0.y_raw()) * f,
        f0.z_raw() + (1.0f - f0.z_raw()) * f
    );
}

/**
 * Exact Fresnel for dielectric
 * 
 * @param cosThetaI cosine of incident angle (positive)
 * @param eta ratio n_incident / n_transmitted (e.g., 1/1.5 when entering glass from air)
 * @return Fresnel reflectance (0 to 1)
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
