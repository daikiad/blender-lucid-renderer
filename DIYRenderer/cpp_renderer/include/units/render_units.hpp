/**
 * render_units.hpp - Physics-aware type system for rendering using mp-units ISQ
 * ==============================================================================
 *
 * This header leverages mp-units v2.4's ISQ (International System of Quantities)
 * to provide type-safe rendering primitives.
 *
 * Design principles:
 * 1. Use mp-units ISQ quantity specifications directly (isq::displacement, etc.)
 * 2. Vec3f is registered as a vector representation type for mp-units
 * 3. Position uses quantity_point for affine space semantics
 * 4. PDFs are mp-units quantities with custom units (1/sr, 1/m²)
 * 5. RGB is a separate 3-channel bundle (not a geometric vector)
 *
 * Key types:
 * - Displacement: quantity<isq::displacement[m], Vec3f>   (vector, P-P result)
 * - Velocity:     quantity<isq::velocity[m/s], Vec3f>     (vector)
 * - Position:     quantity_point<isq::displacement[m], ...> (affine point)
 * - Direction:    Normalized Vec3f (dimensionless, semantic wrapper)
 * - Normal:       Surface normal (dimensionless, semantic wrapper)
 * - PdfW:         quantity<per_sr, float>  [1/sr]
 * - PdfA:         quantity<per_m2, float>  [1/m²]
 * - ColorRGB:     RGB<float> for albedo/reflectance
 * - RadianceRGB:  RGB<Radiance> for spectral radiance
 * - BSDF:         quantity<per_sr, float> [1/sr] for BSDF evaluation
 * - BSDFRGB:      RGB<BSDF> for spectral BSDF
 */

#pragma once

#include <mp-units/compat_macros.h>  // QUANTITY_SPEC macro
#include <mp-units/framework.h>
#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>
#include <algorithm>  // std::min, std::max
#include <cmath>
#include <concepts>
#include <iterator>  // for std::indirectly_readable_traits
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>    // std::pair

namespace render {

using namespace mp_units;
namespace si = mp_units::si;

// ============================================================================
// Part A: Vec3f - Basic 3D Vector Type (float precision)
// ============================================================================

struct Vec3f {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    constexpr Vec3f() = default;
    constexpr Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Vector arithmetic
    constexpr Vec3f operator+(Vec3f o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3f operator-(Vec3f o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3f operator-() const { return {-x, -y, -z}; }

    // Scalar multiplication/division (template for any arithmetic type)
    template <std::floating_point T>
    constexpr Vec3f operator*(T s) const {
        return {x * static_cast<float>(s), y * static_cast<float>(s), z * static_cast<float>(s)};
    }

    template <std::integral T>
    constexpr Vec3f operator*(T s) const {
        return {x * static_cast<float>(s), y * static_cast<float>(s), z * static_cast<float>(s)};
    }

    template <std::floating_point T>
    constexpr Vec3f operator/(T s) const {
        return {x / static_cast<float>(s), y / static_cast<float>(s), z / static_cast<float>(s)};
    }

    template <std::integral T>
    constexpr Vec3f operator/(T s) const {
        return {x / static_cast<float>(s), y / static_cast<float>(s), z / static_cast<float>(s)};
    }

    // Compound assignment
    constexpr Vec3f& operator+=(Vec3f o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
    constexpr Vec3f& operator-=(Vec3f o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }
    template <std::floating_point T>
    constexpr Vec3f& operator*=(T s) {
        x *= static_cast<float>(s);
        y *= static_cast<float>(s);
        z *= static_cast<float>(s);
        return *this;
    }
    template <std::integral T>
    constexpr Vec3f& operator*=(T s) {
        x *= static_cast<float>(s);
        y *= static_cast<float>(s);
        z *= static_cast<float>(s);
        return *this;
    }
    template <std::floating_point T>
    constexpr Vec3f& operator/=(T s) {
        x /= static_cast<float>(s);
        y /= static_cast<float>(s);
        z /= static_cast<float>(s);
        return *this;
    }
    template <std::integral T>
    constexpr Vec3f& operator/=(T s) {
        x /= static_cast<float>(s);
        y /= static_cast<float>(s);
        z /= static_cast<float>(s);
        return *this;
    }

    // Comparison
    constexpr bool operator==(Vec3f o) const { return x == o.x && y == o.y && z == o.z; }

    // Utility
    [[nodiscard]] constexpr float length_squared() const { return x * x + y * y + z * z; }
    [[nodiscard]] float length() const { return std::sqrt(length_squared()); }
};

// Commutative scalar multiplication (free functions)
template <std::floating_point T>
constexpr Vec3f operator*(T s, Vec3f v) {
    return v * s;
}

template <std::integral T>
constexpr Vec3f operator*(T s, Vec3f v) {
    return v * s;
}

// Dot product
constexpr float dot(Vec3f a, Vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Cross product
constexpr Vec3f cross(Vec3f a, Vec3f b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Normalize
inline Vec3f normalize(Vec3f v) {
    float len = v.length();
    if (len > 0.0f) {
        return v / len;
    }
    return v;
}

// Free function length() for Vec3f (convenience)
inline float length(Vec3f v) { return v.length(); }

// ============================================================================
// Part A.1b: Vec2f - Basic 2D Vector Type (float precision)
// ============================================================================

struct Vec2f {
    float x = 0.0f, y = 0.0f;

    constexpr Vec2f() = default;
    constexpr Vec2f(float x_, float y_) : x(x_), y(y_) {}

    // Vector arithmetic
    constexpr Vec2f operator+(Vec2f o) const { return {x + o.x, y + o.y}; }
    constexpr Vec2f operator-(Vec2f o) const { return {x - o.x, y - o.y}; }
    constexpr Vec2f operator-() const { return {-x, -y}; }

    // Scalar multiplication/division
    template <std::floating_point T>
    constexpr Vec2f operator*(T s) const {
        return {x * static_cast<float>(s), y * static_cast<float>(s)};
    }

    template <std::integral T>
    constexpr Vec2f operator*(T s) const {
        return {x * static_cast<float>(s), y * static_cast<float>(s)};
    }

    template <std::floating_point T>
    constexpr Vec2f operator/(T s) const {
        return {x / static_cast<float>(s), y / static_cast<float>(s)};
    }

    template <std::integral T>
    constexpr Vec2f operator/(T s) const {
        return {x / static_cast<float>(s), y / static_cast<float>(s)};
    }

    // Compound assignment
    constexpr Vec2f& operator+=(Vec2f o) { x += o.x; y += o.y; return *this; }
    constexpr Vec2f& operator-=(Vec2f o) { x -= o.x; y -= o.y; return *this; }

    template <std::floating_point T>
    constexpr Vec2f& operator*=(T s) { x *= static_cast<float>(s); y *= static_cast<float>(s); return *this; }

    template <std::integral T>
    constexpr Vec2f& operator*=(T s) { x *= static_cast<float>(s); y *= static_cast<float>(s); return *this; }

    template <std::floating_point T>
    constexpr Vec2f& operator/=(T s) { x /= static_cast<float>(s); y /= static_cast<float>(s); return *this; }

    template <std::integral T>
    constexpr Vec2f& operator/=(T s) { x /= static_cast<float>(s); y /= static_cast<float>(s); return *this; }

    // Comparison
    constexpr bool operator==(Vec2f o) const { return x == o.x && y == o.y; }

    // Utility
    [[nodiscard]] constexpr float length_squared() const { return x * x + y * y; }
    [[nodiscard]] float length() const { return std::sqrt(length_squared()); }
};

// Commutative scalar multiplication for Vec2f
template <std::floating_point T>
constexpr Vec2f operator*(T s, Vec2f v) { return v * s; }

template <std::integral T>
constexpr Vec2f operator*(T s, Vec2f v) { return v * s; }

// Dot product for Vec2f
constexpr float dot(Vec2f a, Vec2f b) { return a.x * b.x + a.y * b.y; }

// Normalize for Vec2f
inline Vec2f normalize(Vec2f v) {
    float len = v.length();
    if (len > 0.0f) return v / len;
    return v;
}

// Free function length() for Vec2f
inline float length(Vec2f v) { return v.length(); }

}  // namespace render

// ============================================================================
// Part A.2: mp-units integration for Vec3f and Vec2f
// ============================================================================

// 1. Tell mp-units that Vec3f is a vector type
template <>
inline constexpr bool mp_units::is_vector<render::Vec3f> = true;

// 1b. Tell mp-units that Vec2f is a vector type
template <>
inline constexpr bool mp_units::is_vector<render::Vec2f> = true;

// 2. Provide wrapped_type_t via std::indirectly_readable_traits
//    This is REQUIRED by mp-units Scalable concept for non-scalar types.
//    wrapped_type_t<Vec3f> = float tells mp-units the underlying scalar type.
template <>
struct std::indirectly_readable_traits<render::Vec3f> {
    using value_type = float;
};

// 2b. Same for Vec2f
template <>
struct std::indirectly_readable_traits<render::Vec2f> {
    using value_type = float;
};

namespace render {

// ============================================================================
// Part B: Scalar Physical Quantities (using mp-units)
// ============================================================================

// Length [m]
using Length = quantity<isq::length[si::metre], float>;

// Area [m²]
using Area = quantity<isq::area[mp_units::square(si::metre)], float>;

// Volume [m³]
using Volume = quantity<isq::volume[mp_units::cubic(si::metre)], float>;

// ============================================================================
// Part B.1: Barycentric Coordinates (dimensionless with vector character)
// ============================================================================
// Barycentric coordinates (u, v) represent a point inside a triangle.
// w = 1 - u - v is computed on demand. These are dimensionless ratios for interpolation.

QUANTITY_SPEC(barycentric_coord2, dimensionless, quantity_character::vector);
using BarycentricCoord2 = quantity<barycentric_coord2[one], Vec2f>;

// Dimensionless scalar type alias for convenience
using Dimensionless = quantity<dimensionless[one], float>;

// Helper: Create BarycentricCoord2 from float u, v values
inline BarycentricCoord2 make_barycentric2(float u, float v) {
    return BarycentricCoord2{Vec2f{u, v}};
}

// Helper: Create BarycentricCoord2 from dimensionless quantities
inline BarycentricCoord2 make_barycentric2(Dimensionless u, Dimensionless v) {
    return BarycentricCoord2{Vec2f{u.numerical_value_in(one), v.numerical_value_in(one)}};
}

// Accessors for barycentric coordinates (encapsulate extraction)
inline float bary_u(const BarycentricCoord2& bary) { return bary.numerical_value_in(one).x; }
inline float bary_v(const BarycentricCoord2& bary) { return bary.numerical_value_in(one).y; }
inline float bary_w(const BarycentricCoord2& bary) { 
    Vec2f uv = bary.numerical_value_in(one);
    return 1.0f - uv.x - uv.y;
}

// ============================================================================
// Part B.2: OrientedArea - Area with Vector Character (for cross products)
// ============================================================================
// In 3D geometry, cross(Displacement, Displacement) produces an "oriented area"
// vector - it has area dimension [m²] but is a 3D vector (not a scalar).
// Standard isq::area has scalar character, so we define a custom quantity_spec.
//
// This is physically correct: the cross product of two displacement vectors
// is a pseudovector (axial vector) representing signed area with orientation.

// Define quantity_spec: area dimension + vector character
// Using QUANTITY_SPEC macro for proper mp-units integration
QUANTITY_SPEC(oriented_area, isq::area, quantity_character::vector);

// OrientedArea: Area dimension [m²] with Vec3f representation
// Used in Möller-Trumbore: qvec = cross(tvec, e1) is an oriented area vector
using OrientedArea = quantity<oriented_area[mp_units::square(si::metre)], Vec3f>;

// Angle [rad]
using Angle = quantity<isq::angular_measure[si::radian], float>;

// Solid angle [sr]
using SolidAngle = quantity<isq::solid_angular_measure[si::steradian], float>;

// Helper: convert degrees to Angle [rad]
constexpr Angle degrees(float deg) {
    return deg * (std::numbers::pi_v<float> / 180.0f) * si::radian;
}

// Helper: create Length from metres
constexpr Length metres(float val) {
    return val * si::metre;
}

// Trigonometric functions for Angle type
// These encapsulate the extraction boundary for std::cos/sin
inline float cos(Angle a) {
    return std::cos(a.numerical_value_in(si::radian));
}

inline float sin(Angle a) {
    return std::sin(a.numerical_value_in(si::radian));
}

// ============================================================================
// Part C: Vector Physical Quantities (using mp-units ISQ)
// ============================================================================

// Displacement [m] - vector quantity (used for Position differences and offsets)
// This is THE vector type for spatial calculations:
//   - Position - Position → Displacement
//   - Direction * Length → Displacement
//   - Position + Displacement → Position
using Displacement = quantity<isq::displacement[si::metre], Vec3f>;

// ============================================================================
// Part D: Position as Affine Point (using quantity_point)
// ============================================================================

// World origin for position coordinates
// Uses isq::displacement so that Position - Position naturally returns Displacement
inline constexpr struct world_origin final : mp_units::absolute_point_origin<isq::displacement> {
} world_origin;

// Position in world space - affine point
// P - P -> Displacement (via subtraction) - ISQ compliant!
// P + Displacement -> P
// P + P -> COMPILE ERROR (affine space semantics)
using Position = quantity_point<isq::displacement[si::metre], world_origin, Vec3f>;

// Helper to create Position from Vec3f
inline Position make_position(Vec3f v) { return world_origin + quantity{v, isq::displacement[si::metre]}; }

// Helper to create Position from coordinates
inline Position make_position(float x, float y, float z) { return make_position(Vec3f{x, y, z}); }

// Get Displacement from world origin (for low-level geometry ONLY)
// Prefer using Position - Position = Displacement for relative calculations
inline Displacement displacement_from_origin(const Position& p) {
    return p.quantity_from(world_origin);
}

// Helper to compute component-wise min of two Positions (for AABB)
// Note: Uses displacement_from_origin internally since Position is an affine point
// and component-wise operations require access to coordinate values
inline Position pos_component_min(const Position& a, const Position& b) {
    Vec3f va = displacement_from_origin(a).numerical_value_in(si::metre);
    Vec3f vb = displacement_from_origin(b).numerical_value_in(si::metre);
    return make_position(std::min(va.x, vb.x), std::min(va.y, vb.y), std::min(va.z, vb.z));
}

// Helper to compute component-wise max of two Positions (for AABB)
inline Position pos_component_max(const Position& a, const Position& b) {
    Vec3f va = displacement_from_origin(a).numerical_value_in(si::metre);
    Vec3f vb = displacement_from_origin(b).numerical_value_in(si::metre);
    return make_position(std::max(va.x, vb.x), std::max(va.y, vb.y), std::max(va.z, vb.z));
}

// Helper to compute both component-wise min and max in one pass (for AABB construction)
// Saves 2 extractions compared to calling pos_min + pos_max separately
inline std::pair<Position, Position> pos_component_minmax(const Position& a, const Position& b) {
    Vec3f va = displacement_from_origin(a).numerical_value_in(si::metre);
    Vec3f vb = displacement_from_origin(b).numerical_value_in(si::metre);
    return {
        make_position(std::min(va.x, vb.x), std::min(va.y, vb.y), std::min(va.z, vb.z)),
        make_position(std::max(va.x, vb.x), std::max(va.y, vb.y), std::max(va.z, vb.z))
    };
}

// Deprecated shims for backward compatibility (to be removed later)
[[deprecated("Use pos_component_min() for component-wise min of Position coordinates")]]
inline Position pos_min(const Position& a, const Position& b) { return pos_component_min(a, b); }

[[deprecated("Use pos_component_max() for component-wise max of Position coordinates")]]
inline Position pos_max(const Position& a, const Position& b) { return pos_component_max(a, b); }

[[deprecated("Use pos_component_minmax() for component-wise min/max of Position coordinates")]]
inline std::pair<Position, Position> pos_minmax(const Position& a, const Position& b) { return pos_component_minmax(a, b); }

// ============================================================================
// Part E: Direction and Normal (Semantic Wrappers for Dimensionless Vectors)
// ============================================================================
// These types wrap Vec3f with a normalization invariant (always unit length).
// They are dimensionless but semantically distinct from raw Vec3f.
// Direction: ray/light direction, Normal: surface normal.
//
// NOTE: v_ is private to enforce the invariant. Use factory functions to create,
// and vec() to access the underlying Vec3f.

// Forward declaration (needed for Direction::as_normal())
struct Normal;

// Direction: A normalized vector representing a direction in space.
// Invariant: Always unit length (enforced by private v_ and factory functions).
struct Direction {
public:
    // Explicit construction only (must be normalized)
    constexpr Direction() : v_{0.0f, 0.0f, 1.0f} {}  // Default: +Z

    // Access components
    [[nodiscard]] constexpr float x() const { return v_.x; }
    [[nodiscard]] constexpr float y() const { return v_.y; }
    [[nodiscard]] constexpr float z() const { return v_.z; }

    // Access underlying Vec3f (read-only reference to preserve invariant)
    [[nodiscard]] constexpr const Vec3f& vec() const { return v_; }

    // Negate direction
    [[nodiscard]] constexpr Direction operator-() const {
        Direction d;
        d.v_ = -v_;
        return d;
    }

    // Dot product between directions
    [[nodiscard]] constexpr float dot(Direction other) const { return render::dot(v_, other.v_); }

    // Cross product between directions (returns nullopt if result is zero)
    [[nodiscard]] std::optional<Direction> cross(Direction other) const;

    // Convert to Normal (same underlying vector, different semantic type)
    [[nodiscard]] Normal as_normal() const;

    constexpr bool operator==(Direction other) const { return v_ == other.v_; }

private:
    Vec3f v_;  // Always normalized - private to enforce invariant

    friend std::optional<Direction> make_direction(Vec3f);
    friend std::optional<Direction> make_direction(float, float, float);
    friend Direction direction_from_unit_vector(Vec3f);
    friend struct Normal;  // Allow Normal::as_direction() to access v_
    explicit constexpr Direction(Vec3f normalized) : v_(normalized) {}
};

// Threshold for considering a vector as zero [m]
// Used consistently across all normalization functions
inline constexpr float kDirectionEpsilon = 1e-6f;

// Threshold for considering determinant/area as zero [m²]
// Used in Möller-Trumbore and other geometric intersection tests
// Value: (1e-6)² = 1e-12, squared from kDirectionEpsilon for dimensional consistency
inline const Area kAreaEpsilon = kDirectionEpsilon * kDirectionEpsilon * mp_units::square(si::metre);

// Geometric epsilon for dimensionless comparisons (Direction · Direction, etc.)
inline constexpr float kGeometryEpsilon = 1e-6f;

// Factory function to create Direction (normalizes input)
// Returns nullopt if input vector is too small to normalize safely
[[nodiscard]] inline std::optional<Direction> make_direction(Vec3f v) {
    float len_sq = v.length_squared();
    if (len_sq < kDirectionEpsilon * kDirectionEpsilon) {
        return std::nullopt;  // Cannot normalize zero/near-zero vector
    }
    Direction d;
    d.v_ = v / std::sqrt(len_sq);
    return d;
}

[[nodiscard]] inline std::optional<Direction> make_direction(float x, float y, float z) {
    return make_direction(Vec3f{x, y, z});
}

// Preferred convenience: normalize input, fall back to a provided default on near-zero.
[[nodiscard]] inline Direction make_direction_or_default(Vec3f v, Direction fallback = Direction{}) {
    if (auto d = make_direction(v)) {
        return *d;
    }
    return fallback;
}

// Construct from an already unit-length vector.
// Precondition: `unit` must be normalized (unit length). This does not renormalize.
[[nodiscard]] inline Direction direction_from_unit_vector(Vec3f unit) {
    Direction d;
    d.v_ = unit;
    return d;
}

// Axis direction constants (useful for padding, offsets, etc.)
inline const Direction kAxisX = direction_from_unit_vector(Vec3f{1.0f, 0.0f, 0.0f});
inline const Direction kAxisY = direction_from_unit_vector(Vec3f{0.0f, 1.0f, 0.0f});
inline const Direction kAxisZ = direction_from_unit_vector(Vec3f{0.0f, 0.0f, 1.0f});

// Direction cross product implementation (deferred due to forward declaration)
inline std::optional<Direction> Direction::cross(Direction other) const {
    return make_direction(render::cross(v_, other.v_));
}

// Direction * Length -> Displacement
// Returns Displacement so it can be added to Position directly
// This is ISQ-correct: direction × distance = displacement (offset)
// Note: Direction is a semantic wrapper (not a mp-units quantity), so we extract Vec3f
// and scale by Length. This is acceptable because Direction is dimensionless (unit vector).
inline Displacement operator*(Direction d, Length len) {
    return quantity{d.vec() * len.numerical_value_in(si::metre), isq::displacement[si::metre]};
}

inline Displacement operator*(Length len, Direction d) { return d * len; }

// ============================================================================
// Part E.2: Displacement Utilities
// ============================================================================

// Get the squared length of a Displacement -> Area [m²]
// This is the fundamental operation; disp_length uses this
inline Area disp_length_squared(const Displacement& d) {
    Vec3f v = d.numerical_value_in(si::metre);
    return v.length_squared() * mp_units::square(si::metre);
}

// Square root of Area -> Length [m]
// Note: mp_units::sqrt is consteval only, so we extract and reattach units
// This encapsulates the extraction boundary for sqrt operations
inline Length area_sqrt(Area a) {
    float val = a.numerical_value_in(mp_units::square(si::metre));
    return std::sqrt(std::max(0.0f, val)) * si::metre;
}

// Get the length of a Displacement -> Length [m]
// Direct extraction + sqrt (mp_units::sqrt is consteval only, so we extract once)
inline Length disp_length(const Displacement& d) {
    Vec3f v = d.numerical_value_in(si::metre);
    return v.length() * si::metre;
}

// Component accessors for Displacement -> Length [m]
// Useful for AABB slab tests and other component-wise operations
inline Length disp_x(const Displacement& d) {
    return d.numerical_value_in(si::metre).x * si::metre;
}
inline Length disp_y(const Displacement& d) {
    return d.numerical_value_in(si::metre).y * si::metre;
}
inline Length disp_z(const Displacement& d) {
    return d.numerical_value_in(si::metre).z * si::metre;
}

// Component accessors for Position -> Length [m]
// Useful for BVH construction, sorting by axis, etc.
inline Length pos_x(const Position& p) { return disp_x(displacement_from_origin(p)); }
inline Length pos_y(const Position& p) { return disp_y(displacement_from_origin(p)); }
inline Length pos_z(const Position& p) { return disp_z(displacement_from_origin(p)); }

// Get Position component by axis index (0=x, 1=y, 2=z)
inline Length pos_component(const Position& p, int axis) {
    return axis == 0 ? pos_x(p) : (axis == 1 ? pos_y(p) : pos_z(p));
}

// Get Displacement component by axis index (0=x, 1=y, 2=z)
inline Length disp_component(const Displacement& d, int axis) {
    return axis == 0 ? disp_x(d) : (axis == 1 ? disp_y(d) : disp_z(d));
}

// Extract Vec3f from Displacement (for cross/dot operations in algorithms like Möller-Trumbore)
// Note: This is needed because mp-units doesn't support Displacement cross/dot directly.
// The extraction strips units but allows component-wise math; results must be reattached.
inline Vec3f disp_to_vec(const Displacement& d) {
    return d.numerical_value_in(si::metre);
}

// Cross product magnitude: |e1 × e2| -> Area [m²]
// Useful for triangle area calculation: Area = 0.5 * |e1 × e2|
// Encapsulates the extraction boundary for cross product operations
inline Area disp_cross_magnitude(const Displacement& e1, const Displacement& e2) {
    Vec3f a = e1.numerical_value_in(si::metre);
    Vec3f b = e2.numerical_value_in(si::metre);
    return length(cross(a, b)) * mp_units::square(si::metre);
}

// Cross product: Displacement × Displacement -> OrientedArea [m²] (vector)
// Returns the full cross product as an oriented area vector (not just magnitude).
// Used in Möller-Trumbore: qvec = cross(tvec, e1) has dimension [m²] and direction.
inline OrientedArea disp_cross(const Displacement& a, const Displacement& b) {
    Vec3f va = a.numerical_value_in(si::metre);
    Vec3f vb = b.numerical_value_in(si::metre);
    return quantity{cross(va, vb), oriented_area[mp_units::square(si::metre)]};
}

// Cross product: Direction × Displacement -> Displacement [m]
// Direction is dimensionless (unit vector), so result has same dimension as Displacement.
// Used in Möller-Trumbore: pvec = cross(d, e2) where d is ray direction.
inline Displacement dir_cross_disp(const Direction& d, const Displacement& disp) {
    Vec3f result = cross(d.vec(), disp.numerical_value_in(si::metre));
    return quantity{result, isq::displacement[si::metre]};
}

// Dot products involving OrientedArea:
// These complete the dimension chain for Möller-Trumbore algorithm

// Displacement · OrientedArea -> Volume [m³]
// Used in Möller-Trumbore: dot(e2, qvec) where qvec is [m²] vector
inline Volume disp_dot_oriented(const Displacement& d, const OrientedArea& oa) {
    Vec3f vd = d.numerical_value_in(si::metre);
    Vec3f voa = oa.numerical_value_in(mp_units::square(si::metre));
    return dot(vd, voa) * mp_units::cubic(si::metre);
}

// Direction · OrientedArea -> Area [m²]
// Direction is dimensionless, so result has same dimension as OrientedArea.
// Used in Möller-Trumbore: dot(D, qvec) for computing barycentric v.
inline Area dir_dot_oriented(const Direction& d, const OrientedArea& oa) {
    Vec3f voa = oa.numerical_value_in(mp_units::square(si::metre));
    return dot(d.vec(), voa) * mp_units::square(si::metre);
}

// ----------------------------------------------------------------------------
// Displacement operations: inner product and projection
// ----------------------------------------------------------------------------

// Inner product: Displacement · Displacement -> Area [m²]
// This is the true inner product of two displacement vectors.
// Dimensional analysis: [m] × [m] = [m²]
inline Area disp_inner(const Displacement& a, const Displacement& b) {
    Vec3f va = a.numerical_value_in(si::metre);
    Vec3f vb = b.numerical_value_in(si::metre);
    return dot(va, vb) * mp_units::square(si::metre);
}

// Projection: Displacement onto Direction -> signed Length [m]
// Projects a displacement vector onto a unit direction.
// This is NOT an inner product (different dimensions), but a projection operation.
// Direction is dimensionless (unit vector, |d̂| = 1), so:
//   project_onto(disp, d̂) = |disp| × cos(θ) = signed projection length
// Dimensional analysis: [m] × [1] = [m]
// This operation is ISQ-valid: ISO 80000 explicitly uses {unit vector} notation.
inline Length project_onto(const Displacement& disp, const Direction& dir) {
    Vec3f v = disp.numerical_value_in(si::metre);
    return dot(v, dir.vec()) * si::metre;
}

// Commutative overload for convenience
inline Length project_onto(const Direction& dir, const Displacement& disp) {
    return project_onto(disp, dir);
}

// Normalize a Displacement to get Direction
// Caller guarantees non-zero length. Still normalizes to handle floating-point drift.
[[nodiscard]] inline Direction normalize_to_direction(const Displacement& d) {
    Vec3f v = d.numerical_value_in(si::metre);
    return direction_from_unit_vector(normalize(v));
}

// Normalize and return both direction and length in one extraction
// Caller guarantees non-zero length.
[[nodiscard]] inline std::pair<Direction, Length> normalize_with_length(const Displacement& d) {
    Vec3f v = d.numerical_value_in(si::metre);
    float len = v.length();
    return {direction_from_unit_vector(v / len), len * si::metre};
}

// Normal: Surface normal (semantically distinct from Direction)
// Invariant: Always unit length (enforced by private v_ and factory functions).
struct Normal {
public:
    constexpr Normal() : v_{0.0f, 1.0f, 0.0f} {}  // Default: +Y (up)

    [[nodiscard]] constexpr float x() const { return v_.x; }
    [[nodiscard]] constexpr float y() const { return v_.y; }
    [[nodiscard]] constexpr float z() const { return v_.z; }

    // Access underlying Vec3f (read-only reference to preserve invariant)
    [[nodiscard]] constexpr const Vec3f& vec() const { return v_; }

    [[nodiscard]] constexpr Normal operator-() const {
        Normal n;
        n.v_ = -v_;
        return n;
    }

    // Dot with direction
    [[nodiscard]] constexpr float dot(Direction d) const { return render::dot(v_, d.vec()); }

    // Convert to Direction (same underlying vector, different semantic type)
    [[nodiscard]] Direction as_direction() const {
        return direction_from_unit_vector(v_);
    }

    constexpr bool operator==(Normal other) const { return v_ == other.v_; }

private:
    Vec3f v_;  // Always normalized - private to enforce invariant

    friend std::optional<Normal> make_normal(Vec3f);
    friend std::optional<Normal> make_normal(float, float, float);
    friend Normal normal_from_unit_vector(Vec3f);
    friend struct Direction;  // Allow Direction::as_normal() to access v_
    explicit constexpr Normal(Vec3f normalized) : v_(normalized) {}
};

// Factory function to create Normal (normalizes input)
// Returns nullopt if input vector is too small to normalize safely
[[nodiscard]] inline std::optional<Normal> make_normal(Vec3f v) {
    float len_sq = v.length_squared();
    if (len_sq < kDirectionEpsilon * kDirectionEpsilon) {
        return std::nullopt;  // Cannot normalize zero/near-zero vector
    }
    Normal n;
    n.v_ = v / std::sqrt(len_sq);
    return n;
}

[[nodiscard]] inline std::optional<Normal> make_normal(float x, float y, float z) {
    return make_normal(Vec3f{x, y, z});
}

// Preferred convenience: normalize input, fall back to a provided default on near-zero.
[[nodiscard]] inline Normal make_normal_or_default(Vec3f v, Normal fallback = Normal{}) {
    if (auto n = make_normal(v)) {
        return *n;
    }
    return fallback;
}

// Construct from an already unit-length vector.
// Precondition: `unit` must be normalized (unit length). This does not renormalize.
[[nodiscard]] inline Normal normal_from_unit_vector(Vec3f unit) {
    Normal n;
    n.v_ = unit;
    return n;
}

// Direction::as_normal() implementation (deferred due to forward declaration)
inline Normal Direction::as_normal() const {
    Normal n;
    n.v_ = v_;  // Already normalized, just copy
    return n;
}

// Dot products between Direction and Normal
inline float dot(Direction d, Normal n) { return render::dot(d.vec(), n.vec()); }
inline float dot(Normal n, Direction d) { return render::dot(n.vec(), d.vec()); }

// ============================================================================
// Part F: Geometric Utilities
// ============================================================================

// Reflect direction around normal
// Note: Reflection of a unit vector around a unit normal always produces a unit vector,
// so we use make_direction_or_default here (normalizes defensively for float drift).
inline Direction reflect(Direction incident, Normal normal) {
    // r = i - 2(i·n)n
    float cos_i = dot(incident, normal);
    Vec3f reflected = incident.vec() - 2.0f * cos_i * normal.vec();
    return make_direction_or_default(reflected);
}

// Refract direction through surface (returns nullopt for total internal reflection)
// eta = n1/n2 (ratio of refractive indices)
inline std::optional<Direction> refract(Direction incident, Normal normal, float eta) {
    float cos_i = -dot(incident, normal);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);

    if (sin2_t > 1.0f) {
        return std::nullopt;  // Total internal reflection
    }

    float cos_t = std::sqrt(1.0f - sin2_t);
    Vec3f refracted = eta * incident.vec() + (eta * cos_i - cos_t) * normal.vec();
    // Refraction should preserve unit length when inputs are unit vectors, but we normalize
    // defensively for numeric stability.
    return make_direction(refracted);
}

// ============================================================================
// Part G: PDF Types (using mp-units quantity)
// ============================================================================

// Define units for PDFs
inline constexpr auto per_sr = one / si::steradian;
inline constexpr auto per_m2 = one / mp_units::square(si::metre);

// PDF per solid angle [1/sr]
using PdfW = quantity<per_sr, float>;

// PDF per area [1/m²]
using PdfA = quantity<per_m2, float>;

// Typed constants for PDF comparisons (defined early for use throughout)
inline constexpr auto MIN_PDF = 1e-6f * per_sr;
inline constexpr auto MIN_PDF_A = 1e-6f * per_m2;

// Note: kGeometryEpsilon is defined earlier in the file near kAreaEpsilon

// ============================================================================
// Part H: RGB Color Types (Channel-wise operations only)
// ============================================================================
// RGB is NOT a geometric vector - no dot/cross/magnitude allowed.

template <typename T>
struct RGB {
    T r, g, b;

    constexpr RGB() : r{}, g{}, b{} {}
    constexpr RGB(T r_, T g_, T b_) : r(r_), g(g_), b(b_) {}

    // Channel-wise arithmetic
    constexpr RGB operator+(RGB other) const { return {r + other.r, g + other.g, b + other.b}; }
    constexpr RGB operator-(RGB other) const { return {r - other.r, g - other.g, b - other.b}; }

    constexpr RGB& operator+=(RGB other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }
    constexpr RGB& operator-=(RGB other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
        return *this;
    }

    constexpr bool operator==(RGB other) const { return r == other.r && g == other.g && b == other.b; }

    // NOTE: dot(), cross(), length() are intentionally NOT defined.
    // RGB is a color, not a geometric vector.
};

// Scalar multiplication for RGB<float>
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr RGB<float> operator*(RGB<float> c, T s) {
    return {c.r * static_cast<float>(s), c.g * static_cast<float>(s), c.b * static_cast<float>(s)};
}

template <typename T>
    requires std::is_arithmetic_v<T>
constexpr RGB<float> operator*(T s, RGB<float> c) {
    return c * s;
}

template <typename T>
    requires std::is_arithmetic_v<T>
constexpr RGB<float> operator/(RGB<float> c, T s) {
    return {c.r / static_cast<float>(s), c.g / static_cast<float>(s), c.b / static_cast<float>(s)};
}

// Type aliases
using ColorRGB = RGB<float>;  // For albedo, reflectance (dimensionless, [0,1])

// Luminance (ITU-R BT.709 coefficients)
constexpr float luminance(ColorRGB c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; }

// Operator* for ColorRGB * ColorRGB (element-wise product for modulating colors)
constexpr ColorRGB operator*(ColorRGB a, ColorRGB b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }

// ============================================================================
// Part I: Radiance Type
// ============================================================================

// Unit constant for radiance extraction
inline constexpr auto radiance_unit = si::watt / (si::steradian * mp_units::square(si::metre));

// Radiance [W/(sr·m²)]
using Radiance = quantity<isq::radiance[si::watt / (si::steradian * mp_units::square(si::metre))], float>;

// Radiant flux (power) [W]
using RadiantFlux = quantity<isq::power[si::watt], float>;

// BSDF [1/sr] - for BSDF evaluation
using BSDF = quantity<per_sr, float>;

// RGB radiance for spectral rendering
using RadianceRGB = RGB<Radiance>;

// RGB BSDF for spectral BSDF evaluation
using BSDFRGB = RGB<BSDF>;

// Helper to create RadianceRGB from float values (assuming W/(sr·m²))
inline RadianceRGB make_radiance_rgb(float r, float g, float b) {
    return {r * radiance_unit, g * radiance_unit, b * radiance_unit};
}

// Helper to create zero RadianceRGB
inline RadianceRGB zero_radiance_rgb() {
    return {0.0f * radiance_unit, 0.0f * radiance_unit, 0.0f * radiance_unit};
}

// Helper to create BSDFRGB from float values (in 1/sr)
inline BSDFRGB make_bsdf_rgb(float r, float g, float b) {
    return {r * per_sr, g * per_sr, b * per_sr};
}

// Helper to create zero BSDFRGB
inline BSDFRGB zero_bsdf_rgb() {
    return {0.0f * per_sr, 0.0f * per_sr, 0.0f * per_sr};
}

// Scalar multiplication for RadianceRGB
inline RadianceRGB operator*(RadianceRGB c, float s) { return {c.r * s, c.g * s, c.b * s}; }

inline RadianceRGB operator*(float s, RadianceRGB c) { return c * s; }

// Scalar multiplication for BSDFRGB
inline BSDFRGB operator*(BSDFRGB c, float s) { return {c.r * s, c.g * s, c.b * s}; }

inline BSDFRGB operator*(float s, BSDFRGB c) { return c * s; }

// ============================================================================
// Part I.2: Cross-Type RGB Operators (Unit-Typed Arithmetic)
// ============================================================================
// These operators enable unit-typed calculations without stripping units mid-computation.
// Note: Same-type +, +=, - are handled by the RGB<T> template member functions.

// --- BSDFRGB Operators ---

// BSDFRGB * ColorRGB -> BSDFRGB (modulate by albedo/kd)
// Physical: [1/sr] × [dimensionless] -> [1/sr]
inline BSDFRGB operator*(BSDFRGB bsdf, ColorRGB color) {
    return {bsdf.r * color.r, bsdf.g * color.g, bsdf.b * color.b};
}
inline BSDFRGB operator*(ColorRGB color, BSDFRGB bsdf) { return bsdf * color; }

// BSDFRGB / PdfW -> ColorRGB (importance sampling weight)
// Physical: [1/sr] / [1/sr] -> [dimensionless]
inline ColorRGB operator/(BSDFRGB f, PdfW pdf) {
    if (pdf < MIN_PDF) return ColorRGB(0.0f, 0.0f, 0.0f);
    // Unit-typed division: [1/sr] / [1/sr] = [one] (dimensionless)
    return ColorRGB(
        (f.r / pdf).numerical_value_in(one),
        (f.g / pdf).numerical_value_in(one),
        (f.b / pdf).numerical_value_in(one)
    );
}

// --- RadianceRGB / ColorRGB Cross-Type Operators ---

// ColorRGB * RadianceRGB -> RadianceRGB (throughput × emission)
// Physical: [dimensionless] × [W/(sr·m²)] -> [W/(sr·m²)]
inline RadianceRGB operator*(ColorRGB throughput, RadianceRGB rad) {
    return {rad.r * throughput.r, rad.g * throughput.g, rad.b * throughput.b};
}
inline RadianceRGB operator*(RadianceRGB rad, ColorRGB throughput) { return throughput * rad; }

// RadianceRGB / float -> RadianceRGB (averaging)
inline RadianceRGB operator/(RadianceRGB rad, float s) {
    return {rad.r / s, rad.g / s, rad.b / s};
}

// ============================================================================
// Part I.3: Type Conversion Helpers (Interface Boundaries Only)
// ============================================================================
// Use these ONLY at interface boundaries (final output, debug, etc.)

// RadianceRGB -> ColorRGB (strip units for final output)
inline ColorRGB to_color(RadianceRGB rad) {
    return ColorRGB(
        rad.r.numerical_value_in(radiance_unit),
        rad.g.numerical_value_in(radiance_unit),
        rad.b.numerical_value_in(radiance_unit)
    );
}

// BSDFRGB -> ColorRGB (strip units)
inline ColorRGB to_color(BSDFRGB bsdf) {
    return ColorRGB(
        bsdf.r.numerical_value_in(per_sr),
        bsdf.g.numerical_value_in(per_sr),
        bsdf.b.numerical_value_in(per_sr)
    );
}

// ColorRGB -> RadianceRGB (create radiance from emission color values)
inline RadianceRGB to_radiance(ColorRGB color) {
    return make_radiance_rgb(color.r, color.g, color.b);
}

// ============================================================================
// Part I.4: Typed Constants
// ============================================================================

// Zero helper for cleaner code (used in light sampling)
inline constexpr PdfW zero_pdf_w() { return 0.0f * per_sr; }

// Typed constants for comparisons (use these instead of extracting numerical values)
// Note: MIN_PDF and MIN_PDF_A are defined in Part G for early use
inline constexpr auto MIN_AREA = 1e-6f * mp_units::square(mp_units::si::metre);
inline constexpr auto MIN_LENGTH = 1e-6f * mp_units::si::metre;

// Inverse square falloff factor: 1/d² -> dimensionless
// Common pattern for point/spot light attenuation
// Returns: 1 m² / d² (dimensionless)
inline float inverse_square_factor(Length dist) {
    Area dist_sq = dist * dist;
    return (1.0f * mp_units::square(mp_units::si::metre) / dist_sq).numerical_value_in(mp_units::one);
}

// Helper: compute dimensionless ratio between two Area values
// Useful for light selection probability = area_i / total_area
inline float area_ratio(Area numerator, Area denominator) {
    if (denominator < MIN_AREA) return 0.0f;
    return (numerator / denominator).numerical_value_in(mp_units::one);
}

// ============================================================================
// Part I.5: Rendering Calculation Helpers
// ============================================================================

// Compute throughput weight from BSDF sample: f × |cosθ| / pdf
// This is the main importance sampling weight calculation.
// Returns dimensionless ColorRGB that can be multiplied with throughput.
inline ColorRGB bsdf_sample_weight(BSDFRGB f, float abs_cos_theta, PdfW pdf) {
    if (pdf < MIN_PDF) return ColorRGB(0.0f, 0.0f, 0.0f);
    // Use typed division: BSDFRGB [1/sr] / PdfW [1/sr] -> ColorRGB [dimensionless]
    ColorRGB base = f / pdf;
    return base * abs_cos_theta;
}

// MIS power heuristic (balance heuristic with power=2)
// Returns weight for sampling strategy with pdf pf.
// Usage: When you sampled via strategy F and want to weight the contribution,
//        call mis_power_heuristic(pdf_of_F, pdf_of_alternative_G)
// Property: mis_power_heuristic(pf, pg) + mis_power_heuristic(pg, pf) ≈ 1
inline float mis_power_heuristic(PdfW pf, PdfW pg) {
    auto f2 = pf * pf;
    auto g2 = pg * pg;
    auto epsilon = 1e-10f * per_sr * per_sr;
    return (f2 / (f2 + g2 + epsilon)).numerical_value_in(mp_units::one);
}

// ============================================================================
// Part I.6: Typed Geometry Helpers
// ============================================================================

// Compute PDF in solid angle measure from area sampling
// Physics: pdf_ω = d² / (A × |cosθ|)   [1/sr]
inline PdfW compute_pdf_w_from_area(Length dist, Area area, float abs_cos_theta) {
    if (area < MIN_AREA || abs_cos_theta < 1e-6f) {
        return 0.0f * per_sr;
    }
    // dist² [m²] / area [m²] = dimensionless ratio
    auto dist_sq = dist * dist;
    // Unit-typed division, extract dimensionless result at the end
    return ((dist_sq / area) / abs_cos_theta).numerical_value_in(one) * per_sr;
}

// ============================================================================
// Part J: Static Assertions for Type Safety Verification
// ============================================================================

namespace detail {

// Helper to check if an expression is valid
template <typename T, typename = void>
struct is_addable : std::false_type {};

template <typename T>
struct is_addable<T, std::void_t<decltype(std::declval<T>() + std::declval<T>())>> : std::true_type {};

}  // namespace detail

// 1. Position - Position -> Displacement (ISQ compliant!)
static_assert(std::is_same_v<decltype(std::declval<Position>() - std::declval<Position>()),
                             quantity<isq::displacement[si::metre], Vec3f>>,
              "Position - Position must yield Displacement");

// 2. Position + Position should NOT compile
// (Uncomment to verify - this will cause a compilation error)
// static_assert(detail::is_addable<Position>::value, "This should fail");

// 3. Direction * Length -> Displacement (ISQ compliant!)
// Returns Displacement so it can be added directly to Position
static_assert(std::is_same_v<decltype(std::declval<Direction>() * std::declval<Length>()), Displacement>,
              "Direction * Length must yield Displacement");

// 4. PdfW and PdfA are distinct types (cannot be mixed)
static_assert(!std::is_same_v<PdfW, PdfA>, "PdfW and PdfA must be distinct types");

// 5. PdfW + PdfW is valid
static_assert(std::is_same_v<decltype(std::declval<PdfW>() + std::declval<PdfW>()), PdfW>,
              "PdfW + PdfW must yield PdfW");

// 6. PdfA + PdfA is valid
static_assert(std::is_same_v<decltype(std::declval<PdfA>() + std::declval<PdfA>()), PdfA>,
              "PdfA + PdfA must yield PdfA");

// 7. PdfW + PdfA should NOT compile
// (Uncomment to verify - this will cause a compilation error)
// static_assert(std::is_same_v<decltype(std::declval<PdfW>() + std::declval<PdfA>()), void>,
//               "This should fail to compile");

}  // namespace render
