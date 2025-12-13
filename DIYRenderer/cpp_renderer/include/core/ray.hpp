/**
 * ray.hpp - Ray and Camera Structures (Unit-Safe)
 * ================================================
 * 
 * Core ray tracing structures using render:: namespace types:
 * - Ray: Origin (Position) and direction (Direction)
 * - Camera: Rendering camera parameters
 * 
 * Type-safe units prevent mixing positions and directions.
 */

#pragma once
#include "../units/render_units.hpp"
#include <cmath>
#include <algorithm>

/**
 * Ray - Ray for ray tracing (unit-safe)
 * 
 * Origin is Position [m], direction is Direction (normalized).
 */
struct Ray {
    render::Position origin;      // Origin [m]
    render::Direction direction;  // Direction (normalized)
    
    Ray() : origin(render::make_position(0.0f, 0.0f, 0.0f)), direction() {}
    
    Ray(const render::Position& orig, const render::Direction& dir) 
        : origin(orig), direction(dir) {}
    
    // Point along ray at distance t
    render::Position at(render::Length t) const {
        return origin + direction * t;
    }
};

/**
 * Camera - Rendering camera parameters (unit-safe)
 */
struct Camera {
    render::Position pos;         // Camera position [m]
    render::Direction dir;        // View direction (normalized)
    render::Direction up;         // Up vector (normalized)
    render::Direction right;      // Right vector (derived)
    render::Direction forward;    // Forward vector (same as dir)
    
    render::Angle fov;            // Field of view [rad]
    float aspect;                 // Aspect ratio (dimensionless)
    
    Camera() 
        : pos(render::make_position(0.0f, 0.0f, 0.0f))
        , fov(render::degrees(60.0f))
        , aspect(1.0f) {}
    
    float fovRad() const {
        return fov.numerical_value_in(mp_units::si::radian);
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

