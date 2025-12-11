/**
 * units.hpp - Physical Units for Rendering
 * =========================================
 * 
 * Compile-time dimensional analysis using mp-units library.
 * Provides type-safe quantities for rendering calculations:
 * 
 * - Length: meters (m), millimeters (mm)
 * - Angle: radians (rad), degrees (deg)
 * - Solid angle: steradians (sr)
 * - Radiant intensity: watts per steradian (W/sr)
 * - Radiance: watts per steradian per square meter (W/(sr·m²))
 * - Irradiance: watts per square meter (W/m²)
 */

#pragma once

#include <mp-units/systems/si.h>
#include <mp-units/systems/isq.h>
#include <mp-units/math.h>
#include <numbers>

namespace diy::units {

using namespace mp_units;

// Use SI unit symbols (no angular to avoid ambiguity)
using namespace mp_units::si::unit_symbols;

// ========== Quantity Aliases ==========

// Length quantities
using Length = quantity<si::metre, float>;
using Distance = Length;

// Angle quantities (using SI radian)
using Angle = quantity<si::radian, float>;

// Solid angle
using SolidAngle = quantity<si::steradian, float>;

// ========== Radiometry Quantities ==========

// Radiant flux / Power [W]
using RadiantFlux = quantity<si::watt, float>;
using Power = RadiantFlux;

// ========== Helper Functions ==========

/**
 * Convert degrees to radians
 */
constexpr Angle deg_to_rad(float degrees) {
    return degrees * (std::numbers::pi_v<float> / 180.0f) * si::radian;
}

/**
 * Convert radians to degrees  
 */
constexpr float rad_to_deg(Angle angle) {
    return angle.numerical_value_in(si::radian) * (180.0f / std::numbers::pi_v<float>);
}

/**
 * Create angle from radians
 */
constexpr Angle radians(float value) {
    return value * si::radian;
}

/**
 * Create angle from degrees
 */
constexpr Angle degrees(float value) {
    return deg_to_rad(value);
}

/**
 * Create length in meters
 */
constexpr Length meters(float value) {
    return value * si::metre;
}

/**
 * Create solid angle in steradians
 */
constexpr SolidAngle steradians(float value) {
    return value * si::steradian;
}

// ========== Constants ==========

inline constexpr auto pi_rad = std::numbers::pi_v<float> * si::radian;
inline constexpr auto two_pi_rad = 2.0f * std::numbers::pi_v<float> * si::radian;
inline constexpr auto half_pi_rad = std::numbers::pi_v<float> / 2.0f * si::radian;

// Full sphere solid angle (4π sr)
inline constexpr auto full_sphere_sr = 4.0f * std::numbers::pi_v<float> * si::steradian;
// Hemisphere solid angle (2π sr)
inline constexpr auto hemisphere_sr = 2.0f * std::numbers::pi_v<float> * si::steradian;

} // namespace diy::units
