/**
 * vec3_unit.hpp - Unit-aware 3D Vector using mp-units
 * ====================================================
 * 
 * Vec3 template where each component has a physical unit.
 * The unit algebra is handled automatically by mp-units.
 * 
 * Spatial vectors:
 *   Position3  = Vec3U<si::metre>    // [m] - position, displacement
 *   Direction3 = Vec3U<one>          // dimensionless - normalized direction
 * 
 * Radiometric vectors (RGB):
 *   Color3      = Vec3U<one>                                  // dimensionless [0,1]
 *   Radiance3   = Vec3U<watt_per_steradian_per_square_metre>  // [W/(sr·m²)]
 *   BSDF3       = Vec3U<per_steradian>                        // [1/sr]
 *   Throughput3 = Vec3U<one>                                  // dimensionless
 * 
 * Unit algebra examples:
 *   position + direction * (5.0f * m)   // OK: Vec3<m> + Vec3<one> * m -> Vec3<m>
 *   color * radiance                    // OK: Vec3<one> * Vec3<W/(sr·m²)> -> Vec3<W/(sr·m²)>
 *   bsdf * radiance / pdf               // OK: [1/sr] * [W/(sr·m²)] / [1/sr] -> [W/(sr·m²)]
 */

#pragma once

#include <mp-units/systems/si.h>
#include <mp-units/math.h>
#include "../units/units.hpp"
#include "vec2.hpp"
#include <cmath>

namespace diy {

using namespace mp_units;

// ========== Unit-aware Vec3 Template ==========
// Unit is a non-type template parameter (e.g., si::metre, one)

template<auto Unit>
struct Vec3U {
    quantity<Unit, float> x, y, z;
    
    // Default constructor
    constexpr Vec3U() : x(0.0f * Unit), y(0.0f * Unit), z(0.0f * Unit) {}
    
    // Constructor from quantities
    constexpr Vec3U(quantity<Unit, float> x_, quantity<Unit, float> y_, quantity<Unit, float> z_) 
        : x(x_), y(y_), z(z_) {}
    
    // Constructor from raw floats (assumes Unit)
    constexpr Vec3U(float x_, float y_, float z_) 
        : x(x_ * Unit), y(y_ * Unit), z(z_ * Unit) {}
    
    // Component access by index (for algorithm convenience)
    constexpr quantity<Unit, float>& operator[](int i) {
        return i == 0 ? x : (i == 1 ? y : z);
    }
    constexpr const quantity<Unit, float>& operator[](int i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }
    
    // Same-unit operations
    constexpr Vec3U operator+(const Vec3U& o) const {
        return Vec3U(x + o.x, y + o.y, z + o.z);
    }
    
    constexpr Vec3U operator-(const Vec3U& o) const {
        return Vec3U(x - o.x, y - o.y, z - o.z);
    }
    
    constexpr Vec3U operator-() const {
        return Vec3U(-x, -y, -z);
    }
    
    Vec3U& operator+=(const Vec3U& o) {
        x += o.x; y += o.y; z += o.z;
        return *this;
    }
    
    Vec3U& operator-=(const Vec3U& o) {
        x -= o.x; y -= o.y; z -= o.z;
        return *this;
    }
    
    // Scalar multiply (dimensionless float)
    constexpr Vec3U operator*(float s) const {
        return Vec3U(x * s, y * s, z * s);
    }
    
    constexpr Vec3U operator/(float s) const {
        return Vec3U(x / s, y / s, z / s);
    }
    
    Vec3U& operator*=(float s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
    
    Vec3U& operator/=(float s) {
        x /= s; y /= s; z /= s;
        return *this;
    }
    
    // Length: |Vec3<U>| -> quantity<U>
    auto length() const {
        auto len_sq = x * x + y * y + z * z;
        return mp_units::sqrt(len_sq);
    }
    
    auto length_squared() const {
        return x * x + y * y + z * z;
    }
    
    // In-place normalize (only makes sense for dimensionless vectors)
    // Returns reference to self for chaining
    Vec3U& normalize() {
        auto len = length();
        if (len.numerical_value_in(Unit) > 0) {
            x /= len.numerical_value_in(Unit);
            y /= len.numerical_value_in(Unit);
            z /= len.numerical_value_in(Unit);
        }
        return *this;
    }
    
    // Return normalized copy
    Vec3U normalized() const {
        auto len = length();
        if (len.numerical_value_in(Unit) > 0) {
            return Vec3U(x / len.numerical_value_in(Unit), 
                         y / len.numerical_value_in(Unit), 
                         z / len.numerical_value_in(Unit));
        }
        return *this;
    }
    
    // Raw float access (for interop with external code)
    float x_raw() const { return x.numerical_value_in(Unit); }
    float y_raw() const { return y.numerical_value_in(Unit); }
    float z_raw() const { return z.numerical_value_in(Unit); }
};

// ========== Basic Operations ==========

// float * Vec3 = Vec3
template<auto U>
constexpr Vec3U<U> operator*(float s, const Vec3U<U>& v) {
    return v * s;
}

// ========== Multiplication with quantity (unit change) ==========
// Vec3<U1> * quantity<U2> -> Vec3<U1*U2>

template<auto U1, auto U2, typename Rep>
constexpr auto operator*(const Vec3U<U1>& v, const quantity<U2, Rep>& q) {
    constexpr auto ResultUnit = U1 * U2;
    return Vec3U<ResultUnit>(v.x * q, v.y * q, v.z * q);
}

template<auto U1, auto U2, typename Rep>
constexpr auto operator*(const quantity<U2, Rep>& q, const Vec3U<U1>& v) {
    return v * q;
}

// ========== Division by quantity ==========
// Vec3<U1> / quantity<U2> -> Vec3<U1/U2>

template<auto U1, auto U2, typename Rep>
constexpr auto operator/(const Vec3U<U1>& v, const quantity<U2, Rep>& q) {
    constexpr auto ResultUnit = U1 / U2;
    return Vec3U<ResultUnit>(v.x / q, v.y / q, v.z / q);
}

// ========== Hadamard (element-wise) product ==========
// Essential for color/radiance calculations
// Vec3<U1> ⊙ Vec3<U2> -> Vec3<U1*U2>

template<auto U1, auto U2>
constexpr auto hadamard(const Vec3U<U1>& a, const Vec3U<U2>& b) {
    constexpr auto ResultUnit = U1 * U2;
    return Vec3U<ResultUnit>(a.x * b.x, a.y * b.y, a.z * b.z);
}

// Operator * for element-wise multiplication between different units
// This enables: color * radiance, throughput * bsdf, etc.
template<auto U1, auto U2>
constexpr auto operator*(const Vec3U<U1>& a, const Vec3U<U2>& b) {
    return hadamard(a, b);
}

// ========== Hadamard division ==========
// Vec3<U1> / Vec3<U2> -> Vec3<U1/U2>

template<auto U1, auto U2>
constexpr auto operator/(const Vec3U<U1>& a, const Vec3U<U2>& b) {
    constexpr auto ResultUnit = U1 / U2;
    return Vec3U<ResultUnit>(a.x / b.x, a.y / b.y, a.z / b.z);
}

// ========== Dot product ==========
// Vec3<U1> · Vec3<U2> -> quantity<U1*U2>

template<auto U1, auto U2>
constexpr auto dot(const Vec3U<U1>& a, const Vec3U<U2>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ========== Cross product ==========
// Vec3<U1> × Vec3<U2> -> Vec3<U1*U2>

template<auto U1, auto U2>
constexpr auto cross(const Vec3U<U1>& a, const Vec3U<U2>& b) {
    constexpr auto ResultUnit = U1 * U2;
    return Vec3U<ResultUnit>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// ========== Normalize ==========
// normalize(Vec3<U>) -> Vec3<one> (dimensionless)

template<auto U>
constexpr Vec3U<one> normalize(const Vec3U<U>& v) {
    auto len = v.length();
    return Vec3U<one>(v.x / len, v.y / len, v.z / len);
}

// ========== Length (standalone function) ==========
// length(Vec3<U>) -> quantity<U>

template<auto U>
constexpr auto length(const Vec3U<U>& v) {
    return v.length();
}

// ========== Abs (element-wise) ==========

template<auto U>
constexpr Vec3U<U> abs(const Vec3U<U>& v) {
    using std::abs;
    return Vec3U<U>(
        abs(v.x.numerical_value_in(U)) * U,
        abs(v.y.numerical_value_in(U)) * U,
        abs(v.z.numerical_value_in(U)) * U
    );
}

// ========== Max/Min Component ==========

template<auto U>
constexpr auto max_component(const Vec3U<U>& v) {
    float mx = std::max({v.x_raw(), v.y_raw(), v.z_raw()});
    return mx * U;
}

template<auto U>
constexpr auto min_component(const Vec3U<U>& v) {
    float mn = std::min({v.x_raw(), v.y_raw(), v.z_raw()});
    return mn * U;
}

// ========== Clamp ==========

template<auto U>
constexpr Vec3U<U> clamp(const Vec3U<U>& v, float lo, float hi) {
    return Vec3U<U>(
        std::clamp(v.x_raw(), lo, hi),
        std::clamp(v.y_raw(), lo, hi),
        std::clamp(v.z_raw(), lo, hi)
    );
}

template<auto U>
constexpr Vec3U<U> clamp(const Vec3U<U>& v, quantity<U, float> lo, quantity<U, float> hi) {
    float lo_f = lo.numerical_value_in(U);
    float hi_f = hi.numerical_value_in(U);
    return Vec3U<U>(
        std::clamp(v.x_raw(), lo_f, hi_f),
        std::clamp(v.y_raw(), lo_f, hi_f),
        std::clamp(v.z_raw(), lo_f, hi_f)
    );
}

// ========== Luminance ==========
// For RGB colors/radiance: Y = 0.2126R + 0.7152G + 0.0722B

template<auto U>
constexpr auto luminance(const Vec3U<U>& v) {
    float Y = 0.2126f * v.x_raw() + 0.7152f * v.y_raw() + 0.0722f * v.z_raw();
    return Y * U;
}

// ========== Is Black/Zero ==========

template<auto U>
constexpr bool is_black(const Vec3U<U>& v, float eps = 1e-6f) {
    return v.x_raw() < eps && v.y_raw() < eps && v.z_raw() < eps;
}

template<auto U>
constexpr bool is_zero(const Vec3U<U>& v, float eps = 1e-6f) {
    return std::abs(v.x_raw()) < eps && std::abs(v.y_raw()) < eps && std::abs(v.z_raw()) < eps;
}

// ========== Lerp ==========

template<auto U>
constexpr Vec3U<U> lerp(const Vec3U<U>& a, const Vec3U<U>& b, float t) {
    return a * (1.0f - t) + b * t;
}

// ========== Type aliases: Spatial ==========

using Position3 = Vec3U<si::metre>;
using Direction3 = Vec3U<one>;

// ========== Type aliases: Radiometric (RGB) ==========

using Color3 = Vec3U<one>;                                            // Reflectance [0,1]
using Radiance3 = Vec3U<units::watt_per_steradian_per_square_metre>;  // [W/(sr·m²)]
using BSDF3 = Vec3U<units::per_steradian>;                            // [1/sr]
using Throughput3 = Vec3U<one>;                                       // Dimensionless path weight

// ========== Reflect ==========
// reflect(incident, normal) -> reflected direction
// r = i - 2(n·i)n

inline Direction3 reflect(const Direction3& incident, const Direction3& normal) {
    float d = dot(normal, incident).numerical_value_in(one);
    return incident - normal * (2.0f * d);
}

// ========== Refract ==========
// refract(incident, normal, eta) -> refracted direction or zero if TIR
// eta = n1/n2

inline Direction3 refract(const Direction3& incident, const Direction3& normal, float eta) {
    float cosi = -dot(normal, incident).numerical_value_in(one);
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f) return Direction3(0, 0, 0);  // Total internal reflection
    return incident * eta + normal * (eta * cosi - std::sqrt(k));
}

} // namespace diy
