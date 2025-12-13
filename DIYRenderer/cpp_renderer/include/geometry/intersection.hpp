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
#include "../units/render_units.hpp"
#include "../core/ray.hpp"
#include <cmath>
#include <algorithm>

namespace geometry {

// ========== Ray Constants ==========

// Unit-typed ray constants (PREFERRED)
inline constexpr render::Length RAY_T_MIN_TYPED = render::metres(1e-4f);       // Ray origin offset to avoid self-intersection
inline constexpr render::Length RAY_T_MAX_TYPED = render::metres(1e30f);       // Max ray travel distance

// Shorter aliases for convenience
inline constexpr auto RAY_T_MIN = RAY_T_MIN_TYPED;      // Preferred: use typed version
inline constexpr auto RAY_T_MAX = RAY_T_MAX_TYPED;      // Preferred: use typed version
inline constexpr auto RAY_OFFSET = render::metres(1e-4f);         // Alias for compatibility
inline constexpr auto RAY_MAX_DISTANCE = render::metres(1e30f);   // Alias for compatibility

// ========== Sphere Intersection ==========

/**
 * Intersect ray with sphere (fully unit-typed)
 * @param ray       Ray to test
 * @param center    Sphere center
 * @param radius    Sphere radius [m]
 * @param tHit      Output: distance to hit point [m]
 * @param hitNormal Output: surface normal at hit point
 * @return true if intersection found
 * 
 * Math: For ray P = O + tD and sphere |P - C|² = r²
 *   |O + tD - C|² = r²
 *   |oc + tD|² = r²  where oc = O - C
 *   (D·D)t² + 2(oc·D)t + (oc·oc - r²) = 0
 *   
 * Units:
 *   D is Direction (dimensionless unit vector), so D·D = 1 (dimensionless)
 *   oc is Displacement [m] (Position - Position = Displacement)
 *   oc·D = Length [m]
 *   oc·oc = Area [m²]
 *   r² = Area [m²]
 *   discriminant = b² - 4ac = [m²] - 4·1·[m²] = [m²]
 *   t = (-b ± √discriminant) / 2a = [m] / 1 = [m]
 */
inline bool intersectSphere(const Ray& ray, const render::Position& center, render::Length radius, 
                            render::Length& tHit, render::Normal& hitNormal) {
    using namespace mp_units::si;
    
    // oc = ray.origin - center [m] (Displacement - ISQ compliant!)
    render::Displacement oc = ray.origin - center;
    render::Direction d = ray.direction;
    
    // Quadratic coefficients (fully typed)
    // a = D·D = 1 (Direction is unit vector, dimensionless)
    constexpr float a = 1.0f;
    
    // b = 2(oc·D) [m]
    render::Length b = 2.0f * render::project_onto(oc, d);
    
    // c = oc·oc - r² [m²]
    render::Area c_coeff = render::disp_length_squared(oc) - radius * radius;
    
    // discriminant = b² - 4ac [m²]
    render::Area discriminant = b * b - 4.0f * a * c_coeff;
    
    // Check discriminant sign with typed comparison
    render::Area zero_area = 0.0f * mp_units::square(metre);
    if (discriminant < zero_area) return false;
    
    // √discriminant [m] - uses area_sqrt helper (encapsulates extraction)
    render::Length sqrtD = render::area_sqrt(discriminant);
    
    // t = (-b - √discriminant) / 2a [m]
    render::Length t = (-b - sqrtD) / (2.0f * a);
    
    if (t < RAY_T_MIN_TYPED) {
        t = (-b + sqrtD) / (2.0f * a);
        if (t < RAY_T_MIN_TYPED) return false;
    }
    
    tHit = t;
    
    // Hit point: P = O + tD
    render::Position hitPoint = ray.origin + d * t;
    
    // Normal = (P - C) / r (unit vector)
    render::Displacement toHit = hitPoint - center;
    hitNormal = render::normalize_to_direction(toHit).as_normal();
    
    return true;
}

// ========== Rectangle Intersection ==========

/**
 * Intersect ray with axis-aligned rectangle in 3D (fully unit-typed)
 * @param sizeX, sizeY Rectangle dimensions [m]
 * @param tHit Output: distance to hit point [m]
 * 
 * Math: Ray-plane intersection, then bounds check
 *   Plane: (P - center) · normal = 0
 *   Ray: P = origin + t * direction
 *   
 *   Substitute: (origin + t*d - center) · n = 0
 *               (origin - center) · n + t*(d · n) = 0
 *               t = -(origin - center) · n / (d · n)
 *               t = (center - origin) · n / (d · n)
 *   
 * Units:
 *   (center - origin) = Displacement [m] (ISQ compliant!)
 *   (center - origin) · n = Length [m]  (Direction is dimensionless)
 *   d · n = dimensionless (Direction · Direction)
 *   t = [m] / dimensionless = [m]
 */
inline bool intersectRectangle(const Ray& ray, const render::Position& center, const render::Direction& normal,
                               const render::Direction& right, const render::Direction& up, 
                               render::Length sizeX, render::Length sizeY,
                               render::Length& tHit, render::Position& hitPoint) {
    using namespace mp_units::si;
    
    render::Direction d = ray.direction;
    
    // denom = d · n (dimensionless)
    float denom = d.dot(normal);
    if (std::abs(denom) < render::kGeometryEpsilon) return false;
    
    // cp = center - origin [m] (Displacement - ISQ compliant!)
    render::Displacement cp = center - ray.origin;
    
    // numerator = cp · n [m]
    render::Length numerator = render::project_onto(cp, normal);
    
    // t = numerator / denom [m]
    render::Length t = numerator / denom;
    
    if (t < RAY_T_MIN_TYPED) return false;
    
    // Hit point: P = origin + t * d
    render::Position hp = ray.origin + d * t;
    
    // Local offset from center: lh = hp - center [m] (Displacement)
    render::Displacement lh = hp - center;
    
    // Project onto local axes
    // x = lh · right [m], y = lh · up [m]
    render::Length x = render::project_onto(lh, right);
    render::Length y = render::project_onto(lh, up);
    
    // Bounds check: |x| > halfX is equivalent to x² > halfX²
    // Use Area [m²] comparison to avoid numerical_value_in extraction
    render::Length halfX = sizeX * 0.5f;
    render::Length halfY = sizeY * 0.5f;
    render::Area halfX_sq = halfX * halfX;
    render::Area halfY_sq = halfY * halfY;
    
    if (x * x > halfX_sq || y * y > halfY_sq) return false;
    
    tHit = t;
    hitPoint = hp;
    return true;
}

// ========== Ellipse/Disk Intersection ==========

/**
 * Intersect ray with ellipse or disk in 3D (fully unit-typed)
 * @param radiusX, radiusY Ellipse radii [m]
 * @param tHit Output: distance to hit point [m]
 * 
 * Math: Same as rectangle intersection, but with ellipse bounds check
 *   (x/rx)² + (y/ry)² ≤ 1
 *   
 * Units:
 *   x, y = Length [m]
 *   rx, ry = Length [m]
 *   x/rx = dimensionless
 *   Comparison is dimensionless
 */
inline bool intersectEllipse(const Ray& ray, const render::Position& center, const render::Direction& normal,
                              const render::Direction& right, const render::Direction& up, 
                              render::Length radiusX, render::Length radiusY,
                              render::Length& tHit, render::Position& hitPoint) {
    using namespace mp_units::si;
    
    render::Direction d = ray.direction;
    
    // denom = d · n (dimensionless)
    float denom = d.dot(normal);
    if (std::abs(denom) < render::kGeometryEpsilon) return false;
    
    // cp = center - origin [m] (Displacement - ISQ compliant!)
    render::Displacement cp = center - ray.origin;
    
    // numerator = cp · n [m]
    render::Length numerator = render::project_onto(cp, normal);
    
    // t = numerator / denom [m]
    render::Length t = numerator / denom;
    
    if (t < RAY_T_MIN_TYPED) return false;
    
    // Hit point: P = origin + t * d
    render::Position hp = ray.origin + d * t;
    
    // Local offset from center: lh = hp - center [m] (Displacement)
    render::Displacement lh = hp - center;
    
    // Project onto local axes
    // x = lh · right [m], y = lh · up [m]
    render::Length x = render::project_onto(lh, right);
    render::Length y = render::project_onto(lh, up);
    
    // Ellipse bounds check: (x/rx)² + (y/ry)² ≤ 1
    // x/radiusX and y/radiusY are dimensionless quantities
    auto nx = x / radiusX;  // dimensionless
    auto ny = y / radiusY;  // dimensionless
    
    if (nx * nx + ny * ny > 1.0f * mp_units::one) return false;
    
    tHit = t;
    hitPoint = hp;
    return true;
}

// ========== AABB Intersection ==========

/**
 * Fast ray-AABB intersection test using precomputed inverse direction
 * @param ray       Ray to test
 * @param invDir    Precomputed 1/ray.direction for each component (dimensionless)
 *                  Direction is a unit vector, so invDir = 1/d_component is also dimensionless.
 * @param bmin      AABB minimum corner (Position)
 * @param bmax      AABB maximum corner (Position)
 * @return true if ray intersects AABB
 * 
 * Dimensional analysis for slab test:
 *   delta = (bmin - ray.origin) = Displacement [m]
 *   invDir = 1 / direction_component = dimensionless (since Direction is unit vector)
 *   t = delta.x [m] * invDir.x [1] = Length [m]
 *
 * This is consistent with the ray equation: P = origin + t * direction
 * where t is Length [m] and direction is dimensionless unit vector.
 */
inline bool intersectAABB(const Ray& ray, const render::Vec3f& invDir, 
                          const render::Position& bmin, const render::Position& bmax) {
    using namespace mp_units::si;
    const float parallelEps = 1e-8f;

    // Compute Displacements from origin to bmin/bmax (ISQ compliant!)
    render::Displacement toMin = bmin - ray.origin;  // [m]
    render::Displacement toMax = bmax - ray.origin;  // [m]

    // Extract direction once for parallel checks
    const auto dir = ray.direction.vec();

    auto axis_slab = [&](float dirComp, float invDirComp,
                         render::Length dMin, render::Length dMax,
                         render::Length& tmin, render::Length& tmax) -> bool {
        // Parallel to slab: origin must be inside bounds on this axis
        if (std::abs(dirComp) < parallelEps) {
            render::Length zero = 0.0f * metre;
            const bool outsidePositive = (dMin > zero && dMax > zero);
            const bool outsideNegative = (dMin < zero && dMax < zero);
            if (outsidePositive || outsideNegative) {
                return false;  // Ray misses box on this axis
            }
            // Inside slab: no t update for this axis
            return true;
        }

        render::Length t1 = dMin * invDirComp;
        render::Length t2 = dMax * invDirComp;
        render::Length axisMin = std::min(t1, t2);
        render::Length axisMax = std::max(t1, t2);
        tmin = std::max(tmin, axisMin);
        tmax = std::min(tmax, axisMax);
        return tmax >= tmin;
    };

    render::Length tmin = RAY_T_MIN_TYPED * 0.0f;  // zero Length
    render::Length tmax = RAY_T_MAX_TYPED;

    if (!axis_slab(dir.x, invDir.x, render::disp_x(toMin), render::disp_x(toMax), tmin, tmax)) return false;
    if (!axis_slab(dir.y, invDir.y, render::disp_y(toMin), render::disp_y(toMax), tmin, tmax)) return false;
    if (!axis_slab(dir.z, invDir.z, render::disp_z(toMin), render::disp_z(toMax), tmin, tmax)) return false;

    render::Length zero_length = 0.0f * metre;
    return tmax >= std::max(tmin, zero_length);
}

// ========== Triangle Intersection ==========

/**
 * Ray-Triangle intersection using Möller–Trumbore algorithm (fully unit-typed)
 * 
 * All intermediate quantities have correct physical dimensions enforced at compile time.
 * This is the canonical implementation for type-safe ray-triangle intersection.
 * 
 * @param ray   Ray to test
 * @param v0    Triangle vertex 0 (Position)
 * @param v1    Triangle vertex 1 (Position)
 * @param v2    Triangle vertex 2 (Position)
 * @param outUV Output: barycentric coordinates (u, v) as BarycentricCoord2
 * @param tMin  Minimum valid t value (default: RAY_T_MIN_TYPED)
 * @param tMax  Maximum valid t value (default: RAY_T_MAX_TYPED)
 * @return distance t if hit within [tMin, tMax], negative Length if no hit
 * 
 * Dimensional Analysis (all enforced by mp-units):
 *   e1, e2, tvec: Displacement [m]
 *   pvec = cross(Direction, Displacement) = Displacement [m]
 *   det = dot(Displacement, Displacement) = Area [m²]
 *   u_num = dot(Displacement, Displacement) = Area [m²]
 *   u = u_num / det = Area/Area = dimensionless
 *   qvec = cross(Displacement, Displacement) = OrientedArea [m²]
 *   v_num = dot(Direction, OrientedArea) = Area [m²]
 *   v = v_num / det = Area/Area = dimensionless
 *   t_num = dot(Displacement, OrientedArea) = Volume [m³]
 *   t = t_num / det = Volume/Area = Length [m]
 * 
 * @return std::optional<Length> - hit distance if intersection found, std::nullopt otherwise
 */
inline std::optional<render::Length> intersectTriangle(const Ray& ray, 
                                        const render::Position& v0, 
                                        const render::Position& v1, 
                                        const render::Position& v2, 
                                        render::BarycentricCoord2& outUV,
                                        render::Length tMin = RAY_T_MIN_TYPED, 
                                        render::Length tMax = RAY_T_MAX_TYPED) {
    // Edge vectors: Displacement [m]
    render::Displacement e1 = v1 - v0;
    render::Displacement e2 = v2 - v0;
    
    // pvec = cross(Direction, Displacement) -> Displacement [m]
    render::Displacement pvec = render::dir_cross_disp(ray.direction, e2);
    
    // det = dot(Displacement, Displacement) -> Area [m²]
    render::Area det = render::disp_inner(e1, pvec);
    
    // Check if ray is parallel to triangle (det ≈ 0)
    // Use typed Area epsilon for comparison
    if (det > -render::kAreaEpsilon && det < render::kAreaEpsilon) {
        return std::nullopt;
    }
    
    // tvec = ray.origin - v0 -> Displacement [m]
    render::Displacement tvec = ray.origin - v0;
    
    // u_numerator = dot(tvec, pvec) -> Area [m²]
    render::Area u_num = render::disp_inner(tvec, pvec);
    
    // u = Area / Area -> dimensionless (keep as typed quantity for comparison)
    auto u_ratio = u_num / det;
    if (u_ratio < 0.0f * mp_units::one || u_ratio > 1.0f * mp_units::one) {
        return std::nullopt;
    }
    
    // qvec = cross(Displacement, Displacement) -> OrientedArea [m²]
    render::OrientedArea qvec = render::disp_cross(tvec, e1);
    
    // v_numerator = dot(Direction, OrientedArea) -> Area [m²]
    render::Area v_num = render::dir_dot_oriented(ray.direction, qvec);
    
    // v = Area / Area -> dimensionless (keep as typed quantity for comparison)
    auto v_ratio = v_num / det;
    if (v_ratio < 0.0f * mp_units::one || u_ratio + v_ratio > 1.0f * mp_units::one) {
        return std::nullopt;
    }
    
    // t_numerator = dot(Displacement, OrientedArea) -> Volume [m³]
    render::Volume t_num = render::disp_dot_oriented(e2, qvec);
    
    // t = Volume / Area -> Length [m]
    render::Length t = t_num / det;
    
    if (t >= tMin && t <= tMax) {
        // Dimensionless quantities passed directly - extraction encapsulated in make_barycentric2
        outUV = render::make_barycentric2(u_ratio, v_ratio);
        return t;
    }
    return std::nullopt;
}

/**
 * Compute inverse direction for AABB slab tests
 * 
 * Since Direction is a unit vector (dimensionless), invDir = 1/d_component
 * is also dimensionless. This allows: t [m] = delta [m] * invDir [1]
 * 
 * For near-zero components, we use a large value (1e30) as a stand-in.
 * This causes the slab test to produce very large t values, effectively
 * treating the ray as parallel to that axis's slabs. The final tmin/tmax
 * comparison handles this correctly for most cases.
 * 
 * @param dir Ray direction (unit vector)
 * @return Inverse of each direction component (dimensionless Vec3f)
 */
inline render::Vec3f computeInverseDirection(const render::Direction& dir) {
    auto d = dir.vec();
    return render::Vec3f(
        std::abs(d.x) > 1e-8f ? 1.0f / d.x : 0.0f,
        std::abs(d.y) > 1e-8f ? 1.0f / d.y : 0.0f,
        std::abs(d.z) > 1e-8f ? 1.0f / d.z : 0.0f
    );
}

} // namespace geometry
