/**
 * light.hpp - Light Structures and Sampling (Unit-Safe)
 * ======================================================
 * 
 * Light sources for path tracing using mp-units:
 * - Position3 [m] for positions
 * - Direction3 for normals/directions
 * - Radiance3 [W/(sr·m²)] for emission
 * - Distance [m] for distances
 * - Area [m²] for surface areas
 * - PdfSolidAngle [1/sr] for probability densities
 */

#pragma once
#include "../math/vec3_unit.hpp"
#include "../core/ray.hpp"
#include "../units/units.hpp"
#include <cmath>
#include <numbers>

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

/**
 * Light - Light source for path tracing (unit-safe)
 */
struct Light {
    LightType type = LightType::EMISSIVE_MESH;
    
    // Geometric properties
    diy::Position3 position;        // Center position [m]
    diy::Direction3 normal;         // Light surface normal (normalized)
    diy::Radiance3 emission;        // Emission radiance [W/(sr·m²)]
    
    diy::units::Area area;          // Surface area [m²]
    int meshIndex;                  // Index of emissive mesh (-1 for native lights)
    int triangleIndex;              // Index of emissive triangle (-1 for native lights)
    
    // Triangle vertices [m] (for emissive mesh)
    diy::Position3 v0, v1, v2;
    
    // Native light properties
    diy::units::RadiantFlux energy; // Power [W]
    diy::units::Distance radius;    // Soft shadow radius [m]
    diy::units::Angle spotAngle;    // Spot cone angle [rad]
    float spotBlend = 0.0f;         // Spot edge softness (dimensionless, 0-1)
    
    // Area light properties
    diy::Direction3 right;          // X axis for area light (normalized)
    diy::Direction3 up;             // Y axis for area light (normalized)
    diy::units::Distance sizeX;     // Width [m]
    diy::units::Distance sizeY;     // Height [m]
    AreaLightShape shape = AreaLightShape::SQUARE;
    
    Light() 
        : area(0.0f * diy::units::square_metre)
        , meshIndex(-1)
        , triangleIndex(-1)
        , energy(1.0f * mp_units::si::watt)
        , radius(0.0f * mp_units::si::metre)
        , spotAngle(0.0f * mp_units::si::radian)
        , sizeX(1.0f * mp_units::si::metre)
        , sizeY(1.0f * mp_units::si::metre) {}
    
    // Accessors for internal use
    float area_raw() const { return diy::units::to_square_meters(area); }
};

// Light epsilon for comparisons
constexpr float LIGHT_EPSILON = 1e-6f;

// ========== Light Sample Result ==========

/**
 * LightSample - Result of sampling a point on a light source (unit-safe)
 * 
 * pdf is in solid angle measure [1/sr]:
 *   pdf = d² / (A × cosθ)
 */
struct LightSample {
    diy::Position3 position;        // Point on light [m]
    diy::Direction3 normal;         // Light surface normal
    diy::Radiance3 emission;        // Light emission [W/(sr·m²)]
    diy::units::PdfSolidAngle pdf;  // Probability density [1/sr]
    diy::units::Distance distance;  // Distance to light [m]
    diy::Direction3 direction;      // Direction from shading point to light
    
    LightSample() 
        : pdf(0.0f * diy::units::per_steradian)
        , distance(0.0f * mp_units::si::metre) {}
    
    bool isValid() const {
        return diy::units::to_per_sr(pdf) > LIGHT_EPSILON 
            && diy::units::to_meters(distance) > LIGHT_EPSILON;
    }
    
    // Legacy accessors
    float pdf_raw() const { return diy::units::to_per_sr(pdf); }
    float distance_raw() const { return diy::units::to_meters(distance); }
};

// ========== Light-Ray Intersection Result ==========

/**
 * LightHit - Result of ray-light intersection (unit-safe)
 */
struct LightHit {
    bool hit = false;
    diy::units::Distance t;         // Ray parameter [m]
    diy::Position3 point;           // Intersection point [m]
    diy::Direction3 normal;         // Light surface normal
    diy::Radiance3 emission;        // Light emission [W/(sr·m²)]
    int lightIndex = -1;
    
    LightHit() : t(1e30f * mp_units::si::metre) {}
    
    float t_raw() const { return diy::units::to_meters(t); }
};

// ========== Utility Functions ==========

/**
 * Sample a uniform random point on a triangle
 */
inline diy::Position3 sampleTrianglePoint(const diy::Position3& v0, const diy::Position3& v1, 
                                           const diy::Position3& v2, float u1, float u2) {
    float su1 = std::sqrt(u1);
    float b0 = 1.0f - su1;
    float b1 = u2 * su1;
    float b2 = 1.0f - b0 - b1;
    return v0 * b0 + v1 * b1 + v2 * b2;
}

/**
 * Compute triangle area
 */
inline diy::units::Area triangleArea(const diy::Position3& v0, const diy::Position3& v1, 
                                      const diy::Position3& v2) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto cross_vec = diy::cross(e1, e2);
    return cross_vec.length() * 0.5f;
}

// ========== Light Sampling ==========

/**
 * Sample a point on a light source (unit-safe version)
 */
inline LightSample sampleLightTyped(const Light& light, const diy::Position3& shadingPoint, 
                                     float u1, float u2) {
    using namespace mp_units;
    using namespace diy::units;
    
    LightSample sample;
    sample.emission = light.emission;
    
    switch (light.type) {
        case LightType::POINT: {
            float radius_m = to_meters(light.radius);
            if (radius_m > LIGHT_EPSILON) {
                // Soft point light: sample on sphere
                float z = 1.0f - 2.0f * u1;
                float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
                float phi = 2.0f * std::numbers::pi_v<float> * u2;
                diy::Direction3 offset(r * std::cos(phi), r * std::sin(phi), z);
                sample.position = light.position + offset * light.radius;
                sample.normal = offset;
            } else {
                sample.position = light.position;
                sample.normal = diy::Direction3(0, 0, 0);
            }
            
            auto toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            float dist_m = to_meters(sample.distance);
            sample.direction = diy::normalize(toLight);
            
            if (radius_m < LIGHT_EPSILON) {
                float invDistSq = 1.0f / (dist_m * dist_m);
                sample.emission = light.emission * invDistSq;
                sample.pdf = pdf_solid_angle(1.0f);
            } else {
                float cosLight = diy::dot(sample.normal, -sample.direction).numerical_value_in(one);
                if (cosLight < LIGHT_EPSILON) {
                    sample.pdf = pdf_solid_angle(0.0f);
                } else {
                    float area_m2 = to_square_meters(light.area);
                    float pdf_val = (dist_m * dist_m) / (area_m2 * cosLight);
                    sample.pdf = pdf_solid_angle(pdf_val);
                }
            }
            break;
        }
        
        case LightType::SUN: {
            sample.direction = -light.normal;
            sample.position = shadingPoint + sample.direction * meters(1000000.0f);
            sample.normal = light.normal;
            sample.distance = meters(1000000.0f);
            sample.pdf = pdf_solid_angle(1.0f);
            break;
        }
        
        case LightType::SPOT: {
            sample.position = light.position;
            sample.normal = light.normal;
            
            auto toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            float dist_m = to_meters(sample.distance);
            sample.direction = diy::normalize(toLight);
            
            float invDistSq = 1.0f / (dist_m * dist_m);
            sample.emission = light.emission * invDistSq;
            
            float cosAngle = diy::dot(light.normal, -sample.direction).numerical_value_in(one);
            float spotAngle_rad = to_radians(light.spotAngle);
            float cosCone = std::cos(spotAngle_rad * 0.5f);
            
            if (cosAngle < cosCone) {
                sample.emission = diy::Radiance3(0, 0, 0);
            } else {
                float blend = light.spotBlend;
                if (blend > 0.0f) {
                    float t = (cosAngle - cosCone) / (1.0f - cosCone);
                    float falloff = std::min(1.0f, t / blend);
                    sample.emission = sample.emission * falloff;
                }
            }
            sample.pdf = pdf_solid_angle(1.0f);
            break;
        }
        
        case LightType::AREA: {
            float sizeX_m = to_meters(light.sizeX);
            float sizeY_m = to_meters(light.sizeY);
            float localX, localY;
            
            if (light.shape == AreaLightShape::DISK || light.shape == AreaLightShape::ELLIPSE) {
                float r = std::sqrt(u1);
                float theta = 2.0f * std::numbers::pi_v<float> * u2;
                localX = r * std::cos(theta) * sizeX_m * 0.5f;
                localY = r * std::sin(theta) * sizeY_m * 0.5f;
            } else {
                localX = (u1 - 0.5f) * sizeX_m;
                localY = (u2 - 0.5f) * sizeY_m;
            }
            
            sample.position = light.position + light.right * meters(localX) + light.up * meters(localY);
            sample.normal = light.normal;
            
            auto toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            float dist_m = to_meters(sample.distance);
            sample.direction = diy::normalize(toLight);
            
            float cosLight = diy::dot(sample.normal, -sample.direction).numerical_value_in(one);
            if (cosLight < LIGHT_EPSILON) {
                sample.pdf = pdf_solid_angle(0.0f);
            } else {
                float area_m2 = to_square_meters(light.area);
                float pdf_val = (dist_m * dist_m) / (area_m2 * cosLight);
                sample.pdf = pdf_solid_angle(pdf_val);
            }
            break;
        }
        
        case LightType::EMISSIVE_MESH:
        default: {
            sample.position = sampleTrianglePoint(light.v0, light.v1, light.v2, u1, u2);
            sample.normal = light.normal;
            
            auto toLight = sample.position - shadingPoint;
            sample.distance = toLight.length();
            float dist_m = to_meters(sample.distance);
            sample.direction = diy::normalize(toLight);
            
            float cosLight = std::abs(diy::dot(sample.normal, -sample.direction).numerical_value_in(one));
            if (cosLight < LIGHT_EPSILON) {
                sample.pdf = pdf_solid_angle(0.0f);
            } else {
                float area_m2 = to_square_meters(light.area);
                float pdf_val = (dist_m * dist_m) / (area_m2 * cosLight);
                sample.pdf = pdf_solid_angle(pdf_val);
            }
            break;
        }
    }
    
    return sample;
}

// Primary interface
inline LightSample sampleLight(const Light& light, const diy::Position3& shadingPoint, 
                                float u1, float u2) {
    return sampleLightTyped(light, shadingPoint, u1, u2);
}

// PDF for light sampling (for MIS) - unit-safe version
inline diy::units::PdfSolidAngle pdfLightSampleTyped(const Light& light, 
                                                      const diy::Position3& shadingPoint, 
                                                      const diy::Position3& lightPoint, 
                                                      const diy::Direction3& lightNormal) {
    using namespace diy::units;
    
    auto toLight = lightPoint - shadingPoint;
    auto dist = toLight.length();
    float dist_m = to_meters(dist);
    auto dir = diy::normalize(toLight);
    float cosLight = std::abs(diy::dot(lightNormal, -dir).numerical_value_in(mp_units::one));
    
    float area_m2 = to_square_meters(light.area);
    if (cosLight < LIGHT_EPSILON || area_m2 < LIGHT_EPSILON) {
        return pdf_solid_angle(0.0f);
    }
    
    return pdf_solid_angle((dist_m * dist_m) / (area_m2 * cosLight));
}

// Primary interface
inline diy::units::PdfSolidAngle pdfLightSample(const Light& light, const diy::Position3& shadingPoint, 
                            const diy::Position3& lightPoint, const diy::Direction3& lightNormal) {
    return pdfLightSampleTyped(light, shadingPoint, lightPoint, lightNormal);
}
