/**
 * light.hpp - Light Structures and Sampling (Unit-Safe)
 * ======================================================
 * 
 * Light sources for path tracing using render:: namespace types:
 * - Position [m] for positions
 * - Direction for normals/directions
 * - RadianceRGB [W/(sr·m²)] for emission
 * - Length [m] for distances
 * - Area [m²] for surface areas
 * - PdfW [1/sr] for probability densities
 */

#pragma once
#include "../units/render_units.hpp"
#include "../core/ray.hpp"
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
    render::Position position;        // Center position [m]
    render::Normal normal;            // Light surface normal (normalized)
    render::RadianceRGB emission;     // Emission radiance [W/(sr·m²)]
    
    render::Area area;                // Surface area [m²]
    int meshIndex;                    // Index of emissive mesh (-1 for native lights)
    int triangleIndex;                // Index of emissive triangle (-1 for native lights)
    
    // Triangle vertices [m] (for emissive mesh)
    render::Position v0, v1, v2;
    
    // Native light properties
    render::RadiantFlux energy;       // Power [W]
    render::Length radius;            // Soft shadow radius [m]
    render::Angle spotAngle;          // Spot cone angle [rad]
    float spotBlend = 0.0f;           // Spot edge softness (dimensionless, 0-1)
    
    // Area light properties
    render::Direction right;          // X axis for area light (normalized)
    render::Direction up;             // Y axis for area light (normalized)
    render::Length sizeX;             // Width [m]
    render::Length sizeY;             // Height [m]
    AreaLightShape shape = AreaLightShape::SQUARE;
    
    Light() 
        : position(render::make_position(0.0f, 0.0f, 0.0f))
        , normal(render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f)))
        , emission(render::zero_radiance_rgb())
        , area(0.0f * mp_units::square(mp_units::si::metre))
        , meshIndex(-1)
        , triangleIndex(-1)
        , v0(render::make_position(0.0f, 0.0f, 0.0f))
        , v1(render::make_position(0.0f, 0.0f, 0.0f))
        , v2(render::make_position(0.0f, 0.0f, 0.0f))
        , energy(1.0f * mp_units::si::watt)
        , radius(0.0f * mp_units::si::metre)
        , spotAngle(0.0f * mp_units::si::radian)
        , right(render::direction_from_unit_vector(render::Vec3f(1.0f, 0.0f, 0.0f)))
        , up(render::direction_from_unit_vector(render::Vec3f(0.0f, 1.0f, 0.0f)))
        , sizeX(1.0f * mp_units::si::metre)
        , sizeY(1.0f * mp_units::si::metre) {}
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
    render::Position position;        // Point on light [m]
    render::Normal normal;            // Light surface normal
    render::RadianceRGB emission;     // Light emission [W/(sr·m²)]
    render::PdfW pdf;                 // Probability density [1/sr]
    render::Length distance;          // Distance to light [m]
    render::Direction direction;      // Direction from shading point to light
    
    LightSample() 
        : position(render::make_position(0.0f, 0.0f, 0.0f))
        , normal(render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f)))
        , emission(render::zero_radiance_rgb())
        , pdf(0.0f * render::per_sr)
        , distance(0.0f * mp_units::si::metre)
        , direction(render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f))) {}
    
    bool isValid() const {
        return pdf > render::MIN_PDF && distance > render::MIN_LENGTH;
    }
};

// ========== Light-Ray Intersection Result ==========

/**
 * LightHit - Result of ray-light intersection (unit-safe)
 */
struct LightHit {
    bool hit = false;
    render::Length t;                 // Ray parameter [m]
    render::Position point;           // Intersection point [m]
    render::Normal normal;            // Light surface normal
    render::RadianceRGB emission;     // Light emission [W/(sr·m²)]
    int lightIndex = -1;
    
    LightHit() 
        : t(1e30f * mp_units::si::metre)
        , point(render::make_position(0.0f, 0.0f, 0.0f))
        , normal(render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f)))
        , emission(render::zero_radiance_rgb()) {}
};

// ========== Utility Functions ==========

/**
 * Sample a uniform random point on a triangle
 */
inline render::Position sampleTrianglePoint(const render::Position& v0, const render::Position& v1, 
                                             const render::Position& v2, float u1, float u2) {
    // Barycentric coordinates for uniform triangle sampling
    float su1 = std::sqrt(u1);
    float b0 = 1.0f - su1;
    float b1 = u2 * su1;
    float b2 = 1.0f - b0 - b1;
    
    // Use typed edge vectors for interpolation
    // P = v0 + b1*(v1-v0) + b2*(v2-v0) = v0 + b1*e1 + b2*e2
    render::Displacement e1 = v1 - v0;  // [m] (ISQ compliant!)
    render::Displacement e2 = v2 - v0;  // [m]
    
    // Use normalize_with_length to get both direction and length in one extraction per edge
    auto [d1, len1] = render::normalize_with_length(e1);
    auto [d2, len2] = render::normalize_with_length(e2);
    
    // Scale edge vectors by barycentric coordinates (dimensionless * [m] = [m])
    return v0 + d1 * (len1 * b1) + d2 * (len2 * b2);
}

/**
 * Compute triangle area (fully unit-typed)
 * Area = 0.5 * |e1 × e2|
 * where e1 = v1 - v0, e2 = v2 - v0 are Displacements [m] (ISQ compliant!)
 */
inline render::Area triangleArea(const render::Position& v0, const render::Position& v1, 
                                  const render::Position& v2) {
    // Edge vectors [m]
    render::Displacement e1 = v1 - v0;
    render::Displacement e2 = v2 - v0;
    
    // Use disp_cross_magnitude helper: |e1 × e2| [m²], area = |cross|/2
    return render::disp_cross_magnitude(e1, e2) * 0.5f;
}

// ========== Light Sampling ==========

/**
 * Sample a point on a light source (unit-safe version)
 */
inline LightSample sampleLightTyped(const Light& light, const render::Position& shadingPoint, 
                                     float u1, float u2) {
    using namespace mp_units;
    using namespace mp_units::si;
    
    LightSample sample;
    sample.emission = light.emission;
    
    switch (light.type) {
        case LightType::POINT: {
            if (light.radius > render::MIN_LENGTH) {
                // Soft point light: sample on sphere surface
                float z = 1.0f - 2.0f * u1;
                float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
                float phi = 2.0f * std::numbers::pi_v<float> * u2;
                render::Vec3f offset_dir(r * std::cos(phi), r * std::sin(phi), z);
                // Use Direction * Length -> Displacement for typed offset
                auto offset_vec = render::direction_from_unit_vector(offset_dir) * light.radius;
                sample.position = light.position + offset_vec;
                sample.normal = render::normal_from_unit_vector(offset_dir);
            } else {
                sample.position = light.position;
                // Normal must be unit length; for an ideal point light the normal is unused.
                sample.normal = render::Normal{};
            }
            
            // Use typed position_distance instead of float calculation
            // Position - Position = Displacement (ISQ compliant!)
            auto toLight = sample.position - shadingPoint;
            // Get both direction and distance in one extraction
            auto [dir, dist] = render::normalize_with_length(toLight);
            sample.distance = dist;
            sample.direction = dir;
            
            if (light.radius < render::MIN_LENGTH) {
                // Inverse square law using typed helper
                sample.emission = light.emission * render::inverse_square_factor(sample.distance);
                sample.pdf = 1.0f * render::per_sr;
            } else {
                float cosLight = render::dot(sample.normal, -sample.direction);
                if (cosLight < LIGHT_EPSILON) {
                    sample.pdf = render::zero_pdf_w();
                } else {
                    sample.pdf = render::compute_pdf_w_from_area(sample.distance, light.area, cosLight);
                }
            }
            break;
        }
        
        case LightType::SUN: {
            sample.direction = -light.normal.as_direction();
            // Use Direction * Length -> Displacement for typed offset
            render::Length sun_dist = 1000000.0f * metre;
            sample.position = shadingPoint + sample.direction * sun_dist;
            sample.normal = light.normal;
            sample.distance = sun_dist;
            sample.pdf = 1.0f * render::per_sr;
            break;
        }
        
        case LightType::SPOT: {
            sample.position = light.position;
            sample.normal = light.normal;
            
            // Use typed calculation: get direction and distance in one extraction
            auto toLight = sample.position - shadingPoint;
            auto [dir, dist] = render::normalize_with_length(toLight);
            sample.distance = dist;
            sample.direction = dir;
            
            // Inverse square law using typed helper
            sample.emission = light.emission * render::inverse_square_factor(sample.distance);
            
            float cosAngle = render::dot(light.normal, -sample.direction);
            // Use render::cos() helper for typed Angle
            float cosCone = render::cos(light.spotAngle * 0.5f);
            
            if (cosAngle < cosCone) {
                sample.emission = render::zero_radiance_rgb();
            } else {
                float blend = light.spotBlend;
                if (blend > 0.0f) {
                    float t = (cosAngle - cosCone) / (1.0f - cosCone);
                    float falloff = std::min(1.0f, t / blend);
                    sample.emission = sample.emission * falloff;
                }
            }
            sample.pdf = 1.0f * render::per_sr;
            break;
        }
        
        case LightType::AREA: {
            // Use typed Length for local coordinates
            render::Length localX, localY;
            
            if (light.shape == AreaLightShape::DISK || light.shape == AreaLightShape::ELLIPSE) {
                float r_sample = std::sqrt(u1);
                float theta = 2.0f * std::numbers::pi_v<float> * u2;
                localX = light.sizeX * 0.5f * r_sample * std::cos(theta);
                localY = light.sizeY * 0.5f * r_sample * std::sin(theta);
            } else {
                localX = light.sizeX * (u1 - 0.5f);
                localY = light.sizeY * (u2 - 0.5f);
            }
            
            // Use Direction * Length -> Displacement for typed offset
            auto offset_x = light.right * localX;
            auto offset_y = light.up * localY;
            sample.position = light.position + offset_x + offset_y;
            sample.normal = light.normal;
            
            // Use typed calculation: get direction and distance in one extraction
            auto toLight = sample.position - shadingPoint;
            auto [dir, dist] = render::normalize_with_length(toLight);
            sample.distance = dist;
            sample.direction = dir;
            
            float cosLight = render::dot(sample.normal, -sample.direction);
            if (cosLight < LIGHT_EPSILON) {
                sample.pdf = render::zero_pdf_w();
            } else {
                sample.pdf = render::compute_pdf_w_from_area(sample.distance, light.area, cosLight);
            }
            break;
        }
        
        case LightType::EMISSIVE_MESH:
        default: {
            sample.position = sampleTrianglePoint(light.v0, light.v1, light.v2, u1, u2);
            sample.normal = light.normal;
            
            // Use typed calculation: get direction and distance in one extraction
            auto toLight2 = sample.position - shadingPoint;
            auto [dir2, dist2] = render::normalize_with_length(toLight2);
            sample.distance = dist2;
            sample.direction = dir2;
            
            float cosLight = std::abs(render::dot(sample.normal, -sample.direction));
            if (cosLight < LIGHT_EPSILON) {
                sample.pdf = render::zero_pdf_w();
            } else {
                sample.pdf = render::compute_pdf_w_from_area(sample.distance, light.area, cosLight);
            }
            break;
        }
    }
    
    return sample;
}

// Primary interface
inline LightSample sampleLight(const Light& light, const render::Position& shadingPoint, 
                                float u1, float u2) {
    return sampleLightTyped(light, shadingPoint, u1, u2);
}

// PDF for light sampling (for MIS) - unit-safe version
inline render::PdfW pdfLightSampleTyped(const Light& light, 
                                         const render::Position& shadingPoint, 
                                         const render::Position& lightPoint, 
                                         const render::Normal& lightNormal) {
    // Get both direction and distance in one extraction
    auto toLight = lightPoint - shadingPoint;
    auto [dir, dist] = render::normalize_with_length(toLight);
    float cosLight = std::abs(render::dot(lightNormal, -dir));
    
    if (cosLight < LIGHT_EPSILON || light.area < render::MIN_AREA) {
        return render::zero_pdf_w();
    }
    
    return render::compute_pdf_w_from_area(dist, light.area, cosLight);
}

// Primary interface
inline render::PdfW pdfLightSample(const Light& light, const render::Position& shadingPoint, 
                            const render::Position& lightPoint, const render::Normal& lightNormal) {
    return pdfLightSampleTyped(light, shadingPoint, lightPoint, lightNormal);
}
