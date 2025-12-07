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
#include "../math/vec3.hpp"
#include "../core/ray.hpp"
#include <cmath>
#include <algorithm>

namespace geometry {

constexpr float INTERSECTION_EPSILON = 1e-6f;

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
inline bool intersectSphere(const Ray& ray, const Vec3& center, float radius, 
                            float& tHit, Vec3& hitNormal) {
    Vec3 oc = ray.o - center;
    float a = Vec3::dot(ray.d, ray.d);
    float b = 2.0f * Vec3::dot(oc, ray.d);
    float c = Vec3::dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0) return false;
    
    float sqrtD = std::sqrt(discriminant);
    float t = (-b - sqrtD) / (2.0f * a);
    
    if (t < INTERSECTION_EPSILON) {
        t = (-b + sqrtD) / (2.0f * a);
        if (t < INTERSECTION_EPSILON) return false;
    }
    
    tHit = t;
    Vec3 hitPoint = ray.o + ray.d * t;
    hitNormal = (hitPoint - center) * (1.0f / radius);
    return true;
}

// ========== Rectangle Intersection ==========

/**
 * Intersect ray with axis-aligned rectangle in 3D
 * @param ray       Ray to test
 * @param center    Rectangle center
 * @param normal    Rectangle surface normal
 * @param right     Rectangle X axis
 * @param up        Rectangle Y axis
 * @param sizeX     Rectangle width
 * @param sizeY     Rectangle height
 * @param tHit      Output: distance to hit point
 * @param hitPoint  Output: intersection point
 * @return true if intersection found
 */
inline bool intersectRectangle(const Ray& ray, const Vec3& center, const Vec3& normal,
                               const Vec3& right, const Vec3& up, 
                               float sizeX, float sizeY,
                               float& tHit, Vec3& hitPoint) {
    float denom = Vec3::dot(normal, ray.d);
    if (std::abs(denom) < INTERSECTION_EPSILON) return false;
    
    float t = Vec3::dot(center - ray.o, normal) / denom;
    if (t < INTERSECTION_EPSILON) return false;
    
    hitPoint = ray.o + ray.d * t;
    Vec3 localHit = hitPoint - center;
    
    float x = Vec3::dot(localHit, right);
    float y = Vec3::dot(localHit, up);
    
    float halfX = sizeX * 0.5f;
    float halfY = sizeY * 0.5f;
    
    if (std::abs(x) > halfX || std::abs(y) > halfY) return false;
    
    tHit = t;
    return true;
}

// ========== Ellipse/Disk Intersection ==========

/**
 * Intersect ray with ellipse or disk in 3D
 * @param ray       Ray to test
 * @param center    Ellipse center
 * @param normal    Ellipse surface normal
 * @param right     Ellipse X axis
 * @param up        Ellipse Y axis
 * @param radiusX   Ellipse X radius (for disk: radius)
 * @param radiusY   Ellipse Y radius (for disk: radius)
 * @param tHit      Output: distance to hit point
 * @param hitPoint  Output: intersection point
 * @return true if intersection found
 */
inline bool intersectEllipse(const Ray& ray, const Vec3& center, const Vec3& normal,
                              const Vec3& right, const Vec3& up, 
                              float radiusX, float radiusY,
                              float& tHit, Vec3& hitPoint) {
    float denom = Vec3::dot(normal, ray.d);
    if (std::abs(denom) < INTERSECTION_EPSILON) return false;
    
    float t = Vec3::dot(center - ray.o, normal) / denom;
    if (t < INTERSECTION_EPSILON) return false;
    
    hitPoint = ray.o + ray.d * t;
    Vec3 localHit = hitPoint - center;
    
    float x = Vec3::dot(localHit, right);
    float y = Vec3::dot(localHit, up);
    
    float normalizedX = x / radiusX;
    float normalizedY = y / radiusY;
    
    if (normalizedX * normalizedX + normalizedY * normalizedY > 1.0f) return false;
    
    tHit = t;
    return true;
}

// ========== AABB Intersection ==========

/**
 * Fast ray-AABB intersection test using precomputed inverse direction
 * @param ray       Ray to test
 * @param invDir    Precomputed 1/ray.d for each component
 * @param bmin      AABB minimum corner
 * @param bmax      AABB maximum corner
 * @return true if ray intersects AABB
 */
inline bool intersectAABB(const Ray& ray, const Vec3& invDir, const Vec3& bmin, const Vec3& bmax) {
    float t1 = (bmin.x - ray.o.x) * invDir.x;
    float t2 = (bmax.x - ray.o.x) * invDir.x;
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);
    
    t1 = (bmin.y - ray.o.y) * invDir.y;
    t2 = (bmax.y - ray.o.y) * invDir.y;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));
    
    t1 = (bmin.z - ray.o.z) * invDir.z;
    t2 = (bmax.z - ray.o.z) * invDir.z;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));
    
    return tmax >= std::max(tmin, 0.0f);
}

// ========== Triangle Intersection ==========

/**
 * Ray-Triangle intersection using Möller–Trumbore algorithm
 * Returns barycentric coordinates for texture/normal interpolation
 * @param ray   Ray to test
 * @param v0    Triangle vertex 0
 * @param v1    Triangle vertex 1
 * @param v2    Triangle vertex 2
 * @param outU  Output: barycentric u coordinate
 * @param outV  Output: barycentric v coordinate
 * @return distance t if hit (>0), -1 if no hit
 */
inline float intersectTriangle(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, 
                               float& outU, float& outV) {
    const float EPS = 1e-6f;
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = Vec3::cross(ray.d, e2);
    float det = Vec3::dot(e1, pvec);
    
    if (det > -EPS && det < EPS) return -1.0f;
    
    float invDet = 1.0f / det;
    Vec3 tvec = ray.o - v0;
    float u = Vec3::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    
    Vec3 qvec = Vec3::cross(tvec, e1);
    float v = Vec3::dot(ray.d, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    
    float t = Vec3::dot(e2, qvec) * invDet;
    if (t > EPS) {
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
inline Vec3 computeInverseDirection(const Vec3& dir) {
    return Vec3(
        std::abs(dir.x) > 1e-8f ? 1.0f / dir.x : 1e30f,
        std::abs(dir.y) > 1e-8f ? 1.0f / dir.y : 1e30f,
        std::abs(dir.z) > 1e-8f ? 1.0f / dir.z : 1e30f
    );
}

} // namespace geometry
