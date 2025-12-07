/**
 * ray.hpp - Ray and Hit Structures
 * =================================
 * 
 * Core ray tracing structures:
 * - Ray: Origin and direction
 * - Camera: Rendering camera parameters
 * 
 * Note: Hit structure is defined in material.hpp after Material is defined
 */

#pragma once
#include "../math/vec3.hpp"
#include <cmath>
#include <algorithm>

/**
 * Ray - Ray for ray tracing
 */
struct Ray {
    Vec3 o;  // Origin
    Vec3 d;  // Direction (should be normalized)
};

/**
 * Camera - Rendering camera parameters
 */
struct Camera {
    Vec3 pos;       // Camera position
    Vec3 dir;       // View direction (normalized)
    Vec3 up;        // Up vector (normalized)
    float fovDeg;   // Field of view in degrees
    float aspect;   // Aspect ratio (width/height)
    Vec3 right;     // Right vector (derived)
    Vec3 forward;   // Forward vector (same as dir, derived)
};

// ========== Reflection/Refraction ==========

// Reflect direction around normal
inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - n * 2.0f * Vec3::dot(v, n);
}

// Refract direction using Snell's law
// Returns zero vector if total internal reflection occurs
inline Vec3 refract(const Vec3 &uv, const Vec3 &n, float etai_over_etat) {
    float cos_theta = std::min(-Vec3::dot(uv, n), 1.0f);
    Vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    float r_out_perp_len2 = Vec3::dot(r_out_perp, r_out_perp);
    if (r_out_perp_len2 > 1.0f) {
        // Total internal reflection
        return Vec3(0, 0, 0);
    }
    Vec3 r_out_parallel = n * (-std::sqrt(std::abs(1.0f - r_out_perp_len2)));
    return r_out_perp + r_out_parallel;
}

// Schlick's approximation for Fresnel reflectance
inline float schlickFresnelReflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
}
