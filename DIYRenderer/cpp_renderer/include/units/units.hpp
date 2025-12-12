/**
 * units.hpp - Physical Units for Rendering
 * =========================================
 * 
 * Compile-time dimensional analysis using mp-units library.
 * Provides type-safe quantities for rendering calculations:
 * 
 * Geometric Quantities:
 * - Length: meters (m), millimeters (mm)
 * - Area: square meters (m²)
 * - Angle: radians (rad), degrees (deg)
 * - Solid angle: steradians (sr)
 * 
 * Radiometric Quantities:
 * - Radiant flux / Power: watts (W)
 * - Radiant intensity: watts per steradian (W/sr)
 * - Radiance: watts per steradian per square meter (W/(sr·m²))
 * - Irradiance: watts per square meter (W/m²)
 * 
 * See: https://en.wikipedia.org/wiki/Radiometry
 */

#pragma once

#include <mp-units/systems/si.h>
#include <mp-units/systems/isq.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/isq/light_and_radiation.h>
#include <mp-units/math.h>
#include <numbers>

namespace diy::units {

using namespace mp_units;

// Use SI unit symbols
using namespace mp_units::si::unit_symbols;

// ========== Derived Unit Definitions ==========
// Define named units for radiometric quantities

// Square metre [m²]
inline constexpr struct square_metre final : named_unit<"m²", square(si::metre)> {} square_metre;

// Watt per steradian [W/sr] - Radiant intensity
inline constexpr struct watt_per_steradian final : named_unit<"W/sr", si::watt / si::steradian> {} watt_per_steradian;

// Watt per square metre [W/m²] - Irradiance / Radiant exitance  
inline constexpr struct watt_per_square_metre final : named_unit<"W/m²", si::watt / square_metre> {} watt_per_square_metre;

// Watt per steradian per square metre [W/(sr·m²)] - Radiance
inline constexpr struct watt_per_steradian_per_square_metre final : 
    named_unit<"W/(sr·m²)", si::watt / (si::steradian * square_metre)> {} watt_per_steradian_per_square_metre;

// Per steradian [1/sr] - BSDF unit
inline constexpr struct per_steradian final : named_unit<"1/sr", one / si::steradian> {} per_steradian;

// Per square metre [1/m²] - Area measure PDF unit
inline constexpr struct per_square_metre final : named_unit<"1/m²", one / square_metre> {} per_square_metre;

// Square metre per steradian [m²/sr] - Geometry factor for PDF conversion (d²/cosθ)
inline constexpr struct square_metre_per_steradian final : named_unit<"m²/sr", square_metre / si::steradian> {} square_metre_per_steradian;

// Steradian per square metre [sr/m²] - Inverse geometry factor (cosθ/d²)
inline constexpr struct steradian_per_square_metre final : named_unit<"sr/m²", si::steradian / square_metre> {} steradian_per_square_metre;

// ========== Geometric Quantities ==========

// Length quantities [m]
using Length = quantity<si::metre, float>;
using Distance = Length;

// Area quantities [m²]
using Area = quantity<square_metre, float>;

// Angle quantities [rad] (using SI radian)
using Angle = quantity<si::radian, float>;

// Solid angle [sr]
using SolidAngle = quantity<si::steradian, float>;

// ========== PDF Types ==========
// PDFs can be in different measures - type safety prevents mixing them up

// PDF in solid angle measure [1/sr] - from BSDF sampling
using PdfSolidAngle = quantity<per_steradian, float>;

// PDF in area measure [1/m²] - from light surface sampling
using PdfArea = quantity<per_square_metre, float>;

// Geometry factor [m²/sr] = d²/cosθ - for converting between PDF measures
using GeometryFactor = quantity<square_metre_per_steradian, float>;

// Inverse geometry factor [sr/m²] = cosθ/d² - for converting between PDF measures
using InverseGeometryFactor = quantity<steradian_per_square_metre, float>;

// ========== Radiometric Quantities ==========
// Reference: https://en.wikipedia.org/wiki/Radiometry#Radiometric_quantities

// Radiant flux / Power [W] - Total power emitted/received/transmitted
using RadiantFlux = quantity<si::watt, float>;
using Power = RadiantFlux;

// Radiant intensity [W/sr] - Power per unit solid angle
// Used for: Point lights, directional emission
using RadiantIntensity = quantity<watt_per_steradian, float>;

// Irradiance [W/m²] - Power per unit area (incident)
// Used for: Sun/directional lights, light arriving at surface
using Irradiance = quantity<watt_per_square_metre, float>;

// Radiance [W/(sr·m²)] - Power per unit solid angle per unit projected area
// The fundamental quantity in rendering equation
// Used for: Surface emission, BSDF evaluation, final pixel values
using Radiance = quantity<watt_per_steradian_per_square_metre, float>;

// Radiant exitance [W/m²] - Power per unit area (emitted)
// Same dimension as Irradiance but different semantics
using RadiantExitance = Irradiance;

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

/**
 * Create area in square meters
 */
constexpr Area square_meters(float value) {
    return value * square_metre;
}

/**
 * Create radiant flux/power in watts
 */
constexpr RadiantFlux watts(float value) {
    return value * si::watt;
}

/**
 * Create radiant intensity in W/sr
 */
constexpr RadiantIntensity watts_per_sr(float value) {
    return value * watt_per_steradian;
}

/**
 * Create irradiance in W/m²
 */
constexpr Irradiance watts_per_m2(float value) {
    return value * watt_per_square_metre;
}

/**
 * Create radiance in W/(sr·m²)
 */
constexpr Radiance watts_per_sr_m2(float value) {
    return value * watt_per_steradian_per_square_metre;
}

// ========== Constants ==========

inline constexpr auto pi_rad = std::numbers::pi_v<float> * si::radian;
inline constexpr auto two_pi_rad = 2.0f * std::numbers::pi_v<float> * si::radian;
inline constexpr auto half_pi_rad = std::numbers::pi_v<float> / 2.0f * si::radian;

// Full sphere solid angle (4π sr)
inline constexpr auto full_sphere_sr = 4.0f * std::numbers::pi_v<float> * si::steradian;
// Hemisphere solid angle (2π sr)
inline constexpr auto hemisphere_sr = 2.0f * std::numbers::pi_v<float> * si::steradian;

// ========== Conversion Utilities ==========

/**
 * Extract numerical value from Length (in meters)
 */
constexpr float to_meters(Length l) {
    return l.numerical_value_in(si::metre);
}

/**
 * Extract numerical value from Area (in m²)
 */
constexpr float to_square_meters(Area a) {
    return a.numerical_value_in(square_metre);
}

/**
 * Extract numerical value from Angle (in radians)
 */
constexpr float to_radians(Angle a) {
    return a.numerical_value_in(si::radian);
}

/**
 * Extract numerical value from SolidAngle (in steradians)
 */
constexpr float to_steradians(SolidAngle sa) {
    return sa.numerical_value_in(si::steradian);
}

/**
 * Extract numerical value from RadiantFlux (in watts)
 */
constexpr float to_watts(RadiantFlux p) {
    return p.numerical_value_in(si::watt);
}

/**
 * Extract numerical value from RadiantIntensity (in W/sr)
 */
constexpr float to_watts_per_sr(RadiantIntensity i) {
    return i.numerical_value_in(watt_per_steradian);
}

/**
 * Extract numerical value from Irradiance (in W/m²)
 */
constexpr float to_watts_per_m2(Irradiance e) {
    return e.numerical_value_in(watt_per_square_metre);
}

/**
 * Extract numerical value from Radiance (in W/(sr·m²))
 */
constexpr float to_watts_per_sr_m2(Radiance l) {
    return l.numerical_value_in(watt_per_steradian_per_square_metre);
}

/**
 * Extract numerical value from Radiance (alias for clarity)
 */
constexpr float to_radiance(Radiance l) {
    return l.numerical_value_in(watt_per_steradian_per_square_metre);
}

/**
 * Extract numerical value from PdfSolidAngle (in 1/sr)
 */
constexpr float to_per_sr(PdfSolidAngle pdf) {
    return pdf.numerical_value_in(per_steradian);
}

/**
 * Extract numerical value from PdfArea (in 1/m²)
 */
constexpr float to_per_m2(PdfArea pdf) {
    return pdf.numerical_value_in(per_square_metre);
}

/**
 * Create PDF in solid angle measure [1/sr]
 */
constexpr PdfSolidAngle pdf_solid_angle(float value) {
    return value * per_steradian;
}

/**
 * Create PDF in area measure [1/m²]
 */
constexpr PdfArea pdf_area(float value) {
    return value * per_square_metre;
}

/**
 * Create geometry factor [m²/sr]
 */
constexpr GeometryFactor geometry_factor(float value) {
    return value * square_metre_per_steradian;
}

// ========== Radiometric Conversions ==========

/**
 * Convert radiant flux to radiance for Lambertian emitter
 * L = Φ / (π × A)
 * 
 * For a perfect Lambertian emitter, this gives the radiance from total power
 * and emitting area.
 */
constexpr Radiance flux_to_radiance_lambertian(RadiantFlux flux, Area area) {
    float L = to_watts(flux) / (std::numbers::pi_v<float> * to_square_meters(area));
    return watts_per_sr_m2(L);
}

/**
 * Convert radiant intensity to irradiance factor at a given distance
 * Point light: E = I / d²
 */
constexpr float intensity_to_irradiance_factor(RadiantIntensity intensity, Distance dist) {
    float d = to_meters(dist);
    return to_watts_per_sr(intensity) / (d * d);
}

/**
 * Convert irradiance to radiance for directional light hitting a surface
 * For parallel rays (sun), treating as if from infinite distance
 */
constexpr Radiance irradiance_to_directional_radiance(Irradiance E) {
    // For directional light, radiance = irradiance (special case)
    return watts_per_sr_m2(to_watts_per_m2(E));
}

// ========== PDF Measure Conversions ==========

/**
 * Compute geometry factor for PDF conversion: G = d²/cosθ
 * 
 * This converts solid angle measure to area measure:
 *   pdf_area = pdf_solid_angle / G
 *   pdf_solid_angle = pdf_area * G
 * 
 * @param distance Distance to the sampled point [m]
 * @param cos_theta Cosine of angle between direction and surface normal
 * @return Geometry factor [m²/sr]
 */
constexpr GeometryFactor compute_geometry_factor(Distance distance, float cos_theta) {
    float d = to_meters(distance);
    float G = (d * d) / std::max(cos_theta, 1e-6f);
    return geometry_factor(G);
}

/**
 * Compute inverse geometry factor: G⁻¹ = cosθ/d²
 * 
 * @param distance Distance to the sampled point [m]
 * @param cos_theta Cosine of angle between direction and surface normal
 * @return Inverse geometry factor [sr/m²]
 */
constexpr InverseGeometryFactor compute_inverse_geometry_factor(Distance distance, float cos_theta) {
    float d = to_meters(distance);
    float G_inv = std::max(cos_theta, 0.0f) / (d * d);
    return G_inv * steradian_per_square_metre;
}

/**
 * Convert PDF from area measure to solid angle measure
 * pdf_solid_angle = pdf_area × G = pdf_area × d²/cosθ
 * 
 * @param pdf_area PDF in area measure [1/m²]
 * @param distance Distance to sampled point [m]
 * @param cos_theta Cosine of angle at sampled point
 * @return PDF in solid angle measure [1/sr]
 */
constexpr PdfSolidAngle convert_pdf_area_to_solid_angle(PdfArea pdf_a, Distance distance, float cos_theta) {
    auto G = compute_geometry_factor(distance, cos_theta);
    return pdf_a * G;  // [1/m²] × [m²/sr] = [1/sr]
}

/**
 * Convert PDF from solid angle measure to area measure
 * pdf_area = pdf_solid_angle / G = pdf_solid_angle × cosθ/d²
 * 
 * @param pdf_solid PDF in solid angle measure [1/sr]
 * @param distance Distance to sampled point [m]
 * @param cos_theta Cosine of angle at sampled point
 * @return PDF in area measure [1/m²]
 */
constexpr PdfArea convert_pdf_solid_angle_to_area(PdfSolidAngle pdf_s, Distance distance, float cos_theta) {
    auto G_inv = compute_inverse_geometry_factor(distance, cos_theta);
    return pdf_s * G_inv;  // [1/sr] × [sr/m²] = [1/m²]
}

} // namespace diy::units
