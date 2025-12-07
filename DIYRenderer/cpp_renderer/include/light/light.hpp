/**
 * light.hpp - Light Structures and Sampling
 * ==========================================
 * 
 * Light sources for path tracing:
 * - Light types (emissive mesh, point, sun, spot, area)
 * - Light sampling for NEE (Next Event Estimation)
 */

#pragma once
#include "../math/vec3.hpp"
#include "../core/ray.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========== Light Types ==========

enum class LightType {
    EMISSIVE_MESH,  // Emissive mesh triangles
    POINT,          // Point light
    SUN,            // Directional light (sun)
    SPOT,           // Spotlight
    AREA            // Area light (rectangle/disk)
};

enum class AreaLightShape {
    SQUARE,     // Square
    RECTANGLE,  // Rectangle
    DISK,       // Disk
    ELLIPSE     // Ellipse
};

// ========== Light Structure ==========

struct Light {
    LightType type = LightType::EMISSIVE_MESH;
    
    Vec3 position;      // Center position
    Vec3 normal;        // Light surface normal / direction
    Vec3 emission;      // Emission color * strength
    float area;         // Surface area
    int meshIndex;      // Index of emissive mesh (-1 for native lights)
    int triangleIndex;  // Index of emissive triangle (-1 for native lights)
    
    // Triangle vertices (for emissive mesh)
    Vec3 v0, v1, v2;
    
    // Native light properties
    float energy = 1.0f;
    float radius = 0.0f;       // Soft shadow radius
    float spotAngle = 0.0f;    // Spot cone angle (radians)
    float spotBlend = 0.0f;    // Spot edge softness (0-1)
    
    // Area light properties
    Vec3 right;         // X axis for area light
    Vec3 up;            // Y axis for area light
    float sizeX = 1.0f;
    float sizeY = 1.0f;
    AreaLightShape shape = AreaLightShape::SQUARE;
    
    Light() : area(0.0f), meshIndex(-1), triangleIndex(-1) {}
};

// ========== Light Sample Result ==========

struct LightSample {
    Vec3 position;      // Point on light
    Vec3 normal;        // Light surface normal at sampled point
    Vec3 emission;      // Light emission
    float pdf;          // Probability density (solid angle measure)
    float distance;     // Distance to light
    Vec3 direction;     // Direction from shading point to light
    
    LightSample() : pdf(0.0f), distance(0.0f) {}
};

// ========== Light-Ray Intersection Result ==========

struct LightHit {
    bool hit = false;
    float t = 1e30f;
    Vec3 point;
    Vec3 normal;
    Vec3 emission;
    int lightIndex = -1;
};

// ========== Utility Functions ==========

inline Vec3 sampleTrianglePoint(const Vec3& v0, const Vec3& v1, const Vec3& v2, 
                                 float u1, float u2) {
    float su1 = std::sqrt(u1);
    float b0 = 1.0f - su1;
    float b1 = u2 * su1;
    float b2 = 1.0f - b0 - b1;
    return v0 * b0 + v1 * b1 + v2 * b2;
}

inline float triangleArea(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 cross = Vec3::cross(e1, e2);
    return 0.5f * cross.length();
}

// Light intersection epsilon
constexpr float LIGHT_EPSILON = 1e-6f;

// ========== Light Sampling ==========

inline LightSample sampleLight(const Light& light, const Vec3& shadingPoint, 
                                float u1, float u2) {
    LightSample sample;
    sample.emission = light.emission;
    
    switch (light.type) {
        case LightType::POINT: {
            if (light.radius > LIGHT_EPSILON) {
                // Soft point light: sample on sphere
                float z = 1.0f - 2.0f * u1;
                float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
                float phi = 2.0f * M_PI * u2;
                Vec3 offset(r * std::cos(phi), r * std::sin(phi), z);
                sample.position = light.position + offset * light.radius;
                sample.normal = offset;
            } else {
                sample.position = light.position;
                sample.normal = Vec3(0, 0, 0);
            }
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            if (light.radius < LIGHT_EPSILON) {
                float invDistSq = 1.0f / (sample.distance * sample.distance);
                sample.emission = light.emission * invDistSq;
                sample.pdf = 1.0f;
            } else {
                float cosLight = Vec3::dot(sample.normal, sample.direction * -1.0f);
                if (cosLight < LIGHT_EPSILON) {
                    sample.pdf = 0.0f;
                } else {
                    sample.pdf = (sample.distance * sample.distance) / (light.area * cosLight);
                }
            }
            break;
        }
        
        case LightType::SUN: {
            sample.direction = light.normal * -1.0f;
            sample.position = shadingPoint + sample.direction * 1000000.0f;
            sample.normal = light.normal;
            sample.distance = 1000000.0f;
            sample.pdf = 1.0f;
            break;
        }
        
        case LightType::SPOT: {
            sample.position = light.position;
            sample.normal = light.normal;
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            float invDistSq = 1.0f / (sample.distance * sample.distance);
            sample.emission = light.emission * invDistSq;
            
            float cosAngle = Vec3::dot(light.normal, sample.direction * -1.0f);
            float cosCone = std::cos(light.spotAngle * 0.5f);
            
            if (cosAngle < cosCone) {
                sample.emission = Vec3(0, 0, 0);
            } else {
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
            float localX, localY;
            
            if (light.shape == AreaLightShape::DISK || light.shape == AreaLightShape::ELLIPSE) {
                float r = std::sqrt(u1);
                float theta = 2.0f * M_PI * u2;
                localX = r * std::cos(theta) * light.sizeX * 0.5f;
                localY = r * std::sin(theta) * light.sizeY * 0.5f;
            } else {
                localX = (u1 - 0.5f) * light.sizeX;
                localY = (u2 - 0.5f) * light.sizeY;
            }
            
            sample.position = light.position + light.right * localX + light.up * localY;
            sample.normal = light.normal;
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            float cosLight = Vec3::dot(sample.normal, sample.direction * -1.0f);
            if (cosLight < LIGHT_EPSILON) {
                sample.pdf = 0.0f;
            } else {
                sample.pdf = (sample.distance * sample.distance) / (light.area * cosLight);
            }
            break;
        }
        
        case LightType::EMISSIVE_MESH:
        default: {
            sample.position = sampleTrianglePoint(light.v0, light.v1, light.v2, u1, u2);
            sample.normal = light.normal;
            
            Vec3 toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            sample.direction = toLight * (1.0f / sample.distance);
            
            float cosLight = std::abs(Vec3::dot(sample.normal, sample.direction * -1.0f));
            if (cosLight < LIGHT_EPSILON) {
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
    
    if (cosLight < LIGHT_EPSILON || light.area < LIGHT_EPSILON) return 0.0f;
    
    return (dist * dist) / (light.area * cosLight);
}
