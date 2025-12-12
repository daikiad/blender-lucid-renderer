/**
 * intersection.hpp - Primitive Shape Intersection Functions
 * ==========================================================
 * 
 * Ray intersection tests for basic geometric primitives:
 * - Sphere intersection (for soft point lights)
 * - Rectangle intersection (for area lights)
 * - Ellipse/Disk intersection (for area lights)
 * - AABB intersection (for BVH traversal)
 * - Triangle intersection with barycentric coordinates
 */

#pragma once
#include "../math/vec3_unit.hpp"
#include "../core/ray.hpp"
#include "../units/units.hpp"
#include <cmath>
#include <algorithm>

namespace geometry {

// Default ray interval constants
constexpr float RAY_T_MIN = 1e-4f;    // Minimum t to avoid self-intersection [m]
constexpr float RAY_T_MAX = 1e30f;    // Maximum t (effectively infinity) [m]
constexpr float INTERSECTION_EPSILON = 1e-6f;  // Geometric epsilon [m]

// Unit-typed versions of constants
inline constexpr auto RAY_OFFSET = diy::units::meters(1e-4f);     // Ray origin offset to avoid self-intersection
inline constexpr auto RAY_MAX_DISTANCE = diy::units::meters(1e30f);  // Max ray travel distance

// ========== Sphere Intersection ==========

/**
 * Intersect ray with sphere
 * @param ray       Ray to test
 * @param center    Sphere center
 * @param radius    Sphere radius
 * @param tHit      Output: distance to hit point
 * @param hitNormal Output: surface normal at hit point
 * @return true if intersection found
 */
inline bool intersectSphere(const Ray& ray, const diy::Position3& center, float radius, 
                            float& tHit, diy::Direction3& hitNormal) {
    float ox = ray.origin.x_raw();
    float oy = ray.origin.y_raw();
    float oz = ray.origin.z_raw();
    float dx = ray.direction.x_raw();
    float dy = ray.direction.y_raw();
    float dz = ray.direction.z_raw();
    float cx = center.x_raw();
    float cy = center.y_raw();
    float cz = center.z_raw();
    
    float ocx = ox - cx;
    float ocy = oy - cy;
    float ocz = oz - cz;
    
    float a = dx*dx + dy*dy + dz*dz;
    float b = 2.0f * (ocx*dx + ocy*dy + ocz*dz);
    float c = ocx*ocx + ocy*ocy + ocz*ocz - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0) return false;
    
    float sqrtD = std::sqrt(discriminant);
    float t = (-b - sqrtD) / (2.0f * a);
    
    if (t < INTERSECTION_EPSILON) {
        t = (-b + sqrtD) / (2.0f * a);
        if (t < INTERSECTION_EPSILON) return false;
    }
    
    tHit = t;
    float hpx = ox + dx * t;
    float hpy = oy + dy * t;
    float hpz = oz + dz * t;
    float invR = 1.0f / radius;
    hitNormal = diy::Direction3((hpx - cx) * invR, (hpy - cy) * invR, (hpz - cz) * invR);
    return true;
}

// ========== Rectangle Intersection ==========

/**
 * Intersect ray with axis-aligned rectangle in 3D
 */
inline bool intersectRectangle(const Ray& ray, const diy::Position3& center, const diy::Direction3& normal,
                               const diy::Direction3& right, const diy::Direction3& up, 
                               float sizeX, float sizeY,
                               float& tHit, diy::Position3& hitPoint) {
    float ox = ray.origin.x_raw();
    float oy = ray.origin.y_raw();
    float oz = ray.origin.z_raw();
    float dx = ray.direction.x_raw();
    float dy = ray.direction.y_raw();
    float dz = ray.direction.z_raw();
    
    float denom = normal.x_raw()*dx + normal.y_raw()*dy + normal.z_raw()*dz;
    if (std::abs(denom) < INTERSECTION_EPSILON) return false;
    
    float cpx = center.x_raw() - ox;
    float cpy = center.y_raw() - oy;
    float cpz = center.z_raw() - oz;
    float t = (cpx*normal.x_raw() + cpy*normal.y_raw() + cpz*normal.z_raw()) / denom;
    if (t < INTERSECTION_EPSILON) return false;
    
    float hpx = ox + dx * t;
    float hpy = oy + dy * t;
    float hpz = oz + dz * t;
    float lhx = hpx - center.x_raw();
    float lhy = hpy - center.y_raw();
    float lhz = hpz - center.z_raw();
    
    float x = lhx*right.x_raw() + lhy*right.y_raw() + lhz*right.z_raw();
    float y = lhx*up.x_raw() + lhy*up.y_raw() + lhz*up.z_raw();
    
    float halfX = sizeX * 0.5f;
    float halfY = sizeY * 0.5f;
    
    if (std::abs(x) > halfX || std::abs(y) > halfY) return false;
    
    tHit = t;
    hitPoint = diy::Position3(hpx, hpy, hpz);
    return true;
}

// ========== Ellipse/Disk Intersection ==========

/**
 * Intersect ray with ellipse or disk in 3D
 */
inline bool intersectEllipse(const Ray& ray, const diy::Position3& center, const diy::Direction3& normal,
                              const diy::Direction3& right, const diy::Direction3& up, 
                              float radiusX, float radiusY,
                              float& tHit, diy::Position3& hitPoint) {
    float ox = ray.origin.x_raw();
    float oy = ray.origin.y_raw();
    float oz = ray.origin.z_raw();
    float dx = ray.direction.x_raw();
    float dy = ray.direction.y_raw();
    float dz = ray.direction.z_raw();
    
    float denom = normal.x_raw()*dx + normal.y_raw()*dy + normal.z_raw()*dz;
    if (std::abs(denom) < INTERSECTION_EPSILON) return false;
    
    float cpx = center.x_raw() - ox;
    float cpy = center.y_raw() - oy;
    float cpz = center.z_raw() - oz;
    float t = (cpx*normal.x_raw() + cpy*normal.y_raw() + cpz*normal.z_raw()) / denom;
    if (t < INTERSECTION_EPSILON) return false;
    
    float hpx = ox + dx * t;
    float hpy = oy + dy * t;
    float hpz = oz + dz * t;
    float lhx = hpx - center.x_raw();
    float lhy = hpy - center.y_raw();
    float lhz = hpz - center.z_raw();
    
    float x = lhx*right.x_raw() + lhy*right.y_raw() + lhz*right.z_raw();
    float y = lhx*up.x_raw() + lhy*up.y_raw() + lhz*up.z_raw();
    
    float normalizedX = x / radiusX;
    float normalizedY = y / radiusY;
    
    if (normalizedX * normalizedX + normalizedY * normalizedY > 1.0f) return false;
    
    tHit = t;
    hitPoint = diy::Position3(hpx, hpy, hpz);
    return true;
}

// ========== AABB Intersection ==========

/**
 * Fast ray-AABB intersection test using precomputed inverse direction
 * @param ray       Ray to test
 * @param invDir    Precomputed 1/ray.d for each component
 * @param bmin      AABB minimum corner (Position3)
 * @param bmax      AABB maximum corner (Position3)
 * @return true if ray intersects AABB
 */
inline bool intersectAABB(const Ray& ray, const diy::Direction3& invDir, 
                          const diy::Position3& bmin, const diy::Position3& bmax) {
    float ox = ray.origin.x_raw();
    float oy = ray.origin.y_raw();
    float oz = ray.origin.z_raw();
    float bminx = bmin.x_raw();
    float bminy = bmin.y_raw();
    float bminz = bmin.z_raw();
    float bmaxx = bmax.x_raw();
    float bmaxy = bmax.y_raw();
    float bmaxz = bmax.z_raw();
    
    float t1 = (bminx - ox) * invDir.x_raw();
    float t2 = (bmaxx - ox) * invDir.x_raw();
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);
    
    t1 = (bminy - oy) * invDir.y_raw();
    t2 = (bmaxy - oy) * invDir.y_raw();
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));
    
    t1 = (bminz - oz) * invDir.z_raw();
    t2 = (bmaxz - oz) * invDir.z_raw();
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));
    
    return tmax >= std::max(tmin, 0.0f);
}

// ========== Triangle Intersection ==========

/**
 * Ray-Triangle intersection using Möller–Trumbore algorithm
 * Returns barycentric coordinates for texture/normal interpolation
 * @param ray   Ray to test
 * @param v0    Triangle vertex 0 (Position3)
 * @param v1    Triangle vertex 1 (Position3)
 * @param v2    Triangle vertex 2 (Position3)
 * @param outU  Output: barycentric u coordinate
 * @param outV  Output: barycentric v coordinate
 * @param tMin  Minimum valid t value (default: RAY_T_MIN)
 * @param tMax  Maximum valid t value (default: RAY_T_MAX)
 * @return distance t if hit within [tMin, tMax], -1 if no hit
 */
inline float intersectTriangle(const Ray& ray, 
                               const diy::Position3& v0, 
                               const diy::Position3& v1, 
                               const diy::Position3& v2, 
                               float& outU, float& outV,
                               float tMin = RAY_T_MIN, float tMax = RAY_T_MAX) {
    float ox = ray.origin.x_raw();
    float oy = ray.origin.y_raw();
    float oz = ray.origin.z_raw();
    float dx = ray.direction.x_raw();
    float dy = ray.direction.y_raw();
    float dz = ray.direction.z_raw();
    
    const float EPS = 1e-6f;
    
    // Edge vectors
    float e1x = v1.x_raw() - v0.x_raw();
    float e1y = v1.y_raw() - v0.y_raw();
    float e1z = v1.z_raw() - v0.z_raw();
    float e2x = v2.x_raw() - v0.x_raw();
    float e2y = v2.y_raw() - v0.y_raw();
    float e2z = v2.z_raw() - v0.z_raw();
    
    // pvec = cross(d, e2)
    float pvecx = dy * e2z - dz * e2y;
    float pvecy = dz * e2x - dx * e2z;
    float pvecz = dx * e2y - dy * e2x;
    
    // det = dot(e1, pvec)
    float det = e1x * pvecx + e1y * pvecy + e1z * pvecz;
    
    if (det > -EPS && det < EPS) return -1.0f;
    
    float invDet = 1.0f / det;
    
    // tvec = o - v0
    float tvecx = ox - v0.x_raw();
    float tvecy = oy - v0.y_raw();
    float tvecz = oz - v0.z_raw();
    
    // u = dot(tvec, pvec) * invDet
    float u = (tvecx * pvecx + tvecy * pvecy + tvecz * pvecz) * invDet;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    
    // qvec = cross(tvec, e1)
    float qvecx = tvecy * e1z - tvecz * e1y;
    float qvecy = tvecz * e1x - tvecx * e1z;
    float qvecz = tvecx * e1y - tvecy * e1x;
    
    // v = dot(d, qvec) * invDet
    float v = (dx * qvecx + dy * qvecy + dz * qvecz) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    
    // t = dot(e2, qvec) * invDet
    float t = (e2x * qvecx + e2y * qvecy + e2z * qvecz) * invDet;
    if (t >= tMin && t <= tMax) {
        outU = u;
        outV = v;
        return t;
    }
    return -1.0f;
}

/**
 * Compute inverse direction for AABB tests
 * Handles near-zero components to avoid division issues
 */
inline diy::Direction3 computeInverseDirection(const diy::Direction3& dir) {
    return diy::Direction3(
        std::abs(dir.x_raw()) > 1e-8f ? 1.0f / dir.x_raw() : 1e30f,
        std::abs(dir.y_raw()) > 1e-8f ? 1.0f / dir.y_raw() : 1e30f,
        std::abs(dir.z_raw()) > 1e-8f ? 1.0f / dir.z_raw() : 1e30f
    );
}

} // namespace geometry
