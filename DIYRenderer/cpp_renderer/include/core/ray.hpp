/**
 * ray.hpp - Ray and Camera Structures (Unit-Safe)
 * ================================================
 * 
 * Core ray tracing structures using mp-units:
 * - Ray: Origin (Position3) and direction (Direction3)
 * - Camera: Rendering camera parameters
 * 
 * Type-safe units prevent mixing positions and directions.
 */

#pragma once
#include "../math/vec3_unit.hpp"
#include "../units/units.hpp"
#include <cmath>
#include <algorithm>

/**
 * Ray - Ray for ray tracing (unit-safe)
 * 
 * Origin is Position3 [m], direction is Direction3 (dimensionless normalized).
 */
struct Ray {
    diy::Position3 origin;      // Origin [m]
    diy::Direction3 direction;  // Direction (normalized, dimensionless)
    
    Ray() : origin(), direction(0, 0, 1) {}
    
    Ray(const diy::Position3& orig, const diy::Direction3& dir) 
        : origin(orig), direction(dir) {}
    
    // Point along ray at distance t
    diy::Position3 at(diy::units::Distance t) const {
        return origin + direction * t;
    }
};

/**
 * Camera - Rendering camera parameters (unit-safe)
 */
struct Camera {
    diy::Position3 pos;         // Camera position [m]
    diy::Direction3 dir;        // View direction (normalized)
    diy::Direction3 up;         // Up vector (normalized)
    diy::Direction3 right;      // Right vector (derived)
    diy::Direction3 forward;    // Forward vector (same as dir)
    
    diy::units::Angle fov;      // Field of view [rad]
    float aspect;               // Aspect ratio (dimensionless)
    
    Camera() 
        : fov(diy::units::degrees(60.0f))
        , aspect(1.0f) {}
    
    float fovRad() const {
        return diy::units::to_radians(fov);
    }
    
    float halfTanFov() const {
        return std::tan(fovRad() * 0.5f);
    }
};

// ========== Utility Functions ==========

inline float schlickFresnelReflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
}

