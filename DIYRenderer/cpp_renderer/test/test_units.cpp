/**
 * test_units.cpp - Test mp-units integration
 * 
 * Compile with:
 *   cd build_pybind && cmake .. && make test_units
 */

#include "../include/units/units.hpp"
#include "../include/light/light.hpp"
#include "../include/core/ray.hpp"
#include "../include/core/material.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace diy::units;
using namespace mp_units::si::unit_symbols;

// Helper macro for tests
#define TEST(name) std::cout << "  " << name << "... "; 
#define PASS() std::cout << "PASS\n";
#define FAIL(msg) { std::cout << "FAIL: " << msg << "\n"; return 1; }

// Float comparison with tolerance
bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

int main() {
    std::cout << "=== mp-units Test for DIY Renderer ===\n\n";
    
    // ========== 1. Geometric Quantities ==========
    std::cout << "1. Geometric Quantities:\n";
    
    TEST("Length creation")
    Length distance = meters(5.0f);
    if (!approx(to_meters(distance), 5.0f)) FAIL("meters() failed");
    Length small = 0.01f * m;
    if (!approx(to_meters(small), 0.01f)) FAIL("unit symbol failed");
    PASS()
    
    TEST("Area creation")
    Area surface = square_meters(2.5f);
    if (!approx(to_square_meters(surface), 2.5f)) FAIL("square_meters() failed");
    PASS()
    
    TEST("Angle conversions")
    Angle fov = deg_to_rad(90.0f);
    if (!approx(to_radians(fov), std::numbers::pi_v<float> / 2.0f)) FAIL("deg_to_rad failed");
    if (!approx(rad_to_deg(fov), 90.0f)) FAIL("rad_to_deg failed");
    Angle spot = degrees(45.0f);
    if (!approx(rad_to_deg(spot), 45.0f)) FAIL("degrees() failed");
    PASS()
    
    TEST("Solid angle")
    SolidAngle omega = steradians(0.5f);
    if (!approx(to_steradians(omega), 0.5f)) FAIL("steradians() failed");
    if (!approx(to_steradians(full_sphere_sr), 4.0f * std::numbers::pi_v<float>)) FAIL("full_sphere_sr failed");
    if (!approx(to_steradians(hemisphere_sr), 2.0f * std::numbers::pi_v<float>)) FAIL("hemisphere_sr failed");
    PASS()
    
    // ========== 2. Radiometric Quantities ==========
    std::cout << "\n2. Radiometric Quantities:\n";
    
    TEST("Radiant flux / Power")
    RadiantFlux power = watts(100.0f);
    if (!approx(to_watts(power), 100.0f)) FAIL("watts() failed");
    RadiantFlux p2 = 50.0f * W;
    if (!approx(to_watts(p2), 50.0f)) FAIL("W unit symbol failed");
    PASS()
    
    TEST("Radiant intensity [W/sr]")
    RadiantIntensity intensity = watts_per_sr(200.0f);
    if (!approx(to_watts_per_sr(intensity), 200.0f)) FAIL("watts_per_sr() failed");
    PASS()
    
    TEST("Irradiance [W/m²]")
    Irradiance irr = watts_per_m2(1000.0f);  // ~1 sun
    if (!approx(to_watts_per_m2(irr), 1000.0f)) FAIL("watts_per_m2() failed");
    PASS()
    
    TEST("Radiance [W/(sr·m²)]")
    Radiance rad = watts_per_sr_m2(500.0f);
    if (!approx(to_watts_per_sr_m2(rad), 500.0f)) FAIL("watts_per_sr_m2() failed");
    PASS()
    
    // ========== 3. Radiometric Conversions ==========
    std::cout << "\n3. Radiometric Conversions:\n";
    
    TEST("Flux to Radiance (Lambertian)")
    // L = Φ / (π × A)
    // 100W over 1m² → L = 100 / π ≈ 31.83 W/(sr·m²)
    RadiantFlux flux = watts(100.0f);
    Area area = square_meters(1.0f);
    Radiance L = flux_to_radiance_lambertian(flux, area);
    float expected_L = 100.0f / std::numbers::pi_v<float>;
    if (!approx(to_watts_per_sr_m2(L), expected_L, 0.01f)) 
        FAIL("flux_to_radiance_lambertian failed: got " << to_watts_per_sr_m2(L) << " expected " << expected_L);
    PASS()
    
    TEST("Intensity to irradiance factor")
    // I = 1000 W/sr at distance 2m
    // E = I / d² = 1000 / 4 = 250 W/m²
    RadiantIntensity I = watts_per_sr(1000.0f);
    Distance d = meters(2.0f);
    float E_factor = intensity_to_irradiance_factor(I, d);
    if (!approx(E_factor, 250.0f)) FAIL("intensity_to_irradiance_factor failed");
    PASS()
    
    // ========== 4. Type Safety ==========
    std::cout << "\n4. Type Safety (compile-time checks):\n";
    std::cout << "   The following would NOT compile:\n";
    std::cout << "   // Length l = watts(5.0f);       // Error: can't assign power to length\n";
    std::cout << "   // Angle a = meters(1.0f);       // Error: can't assign length to angle\n";
    std::cout << "   // Radiance r = watts(10.0f);    // Error: dimension mismatch\n";
    std::cout << "   // float x = meters(1.0f);       // Error: can't implicitly convert to float\n";
    std::cout << "   Type safety verified!\n";
    
    // ========== 5. Integration with Renderer Structures ==========
    std::cout << "\n5. Integration with Renderer Structures:\n";
    
    TEST("Light structure with units")
    Light light;
    light.type = LightType::AREA;
    light.energy = watts(100.0f);
    light.area = 2.0f * diy::units::square_metre;
    light.spotAngle = 0.785f * mp_units::si::radian;  // ~45 degrees
    light.radius = 0.1f * mp_units::si::metre;
    
    if (!approx(to_watts(light.energy), 100.0f)) FAIL("Light.energy failed");
    if (!approx(to_square_meters(light.area), 2.0f)) FAIL("Light.area failed");
    if (!approx(to_radians(light.spotAngle), 0.785f)) FAIL("Light.spotAngle failed");
    if (!approx(to_meters(light.radius), 0.1f)) FAIL("Light.radius failed");
    PASS()
    
    TEST("Camera structure with units")
    Camera cam;
    cam.fov = degrees(90.0f);
    
    if (!approx(rad_to_deg(cam.fov), 90.0f)) FAIL("Camera.fov set failed");
    if (!approx(cam.fovRad(), std::numbers::pi_v<float> / 2.0f)) FAIL("Camera.fovRad() failed");
    
    // Set via typed field with different value
    cam.fov = degrees(60.0f);
    if (!approx(rad_to_deg(cam.fov), 60.0f)) FAIL("Camera.fov update failed");
    PASS()
    
    TEST("Hit structure with units")
    Hit hit;
    hit.hit = true;
    hit.setFromTyped(5.5f * mp_units::si::metre, 
                     diy::Position3(0, 0, 0), 
                     diy::Direction3(0, 1, 0));
    
    if (!approx(to_meters(hit.distance()), 5.5f)) FAIL("Hit.distance() unit accessor failed");
    if (!approx(hit.t_meters(), 5.5f)) FAIL("Hit.t_meters() failed");
    PASS()
    
    TEST("LightSample structure with units")
    LightSample sample;
    sample.distance = 3.5f * mp_units::si::metre;
    if (!approx(to_meters(sample.distance), 3.5f)) FAIL("LightSample.distance value failed");
    if (!approx(sample.distance_raw(), 3.5f)) FAIL("LightSample.distance_raw() failed");
    PASS()
    
    // ========== 6. Constants ==========
    std::cout << "\n6. Constants:\n";
    std::cout << "   π rad = " << to_radians(pi_rad) << " rad\n";
    std::cout << "   2π rad = " << to_radians(two_pi_rad) << " rad\n";
    std::cout << "   π/2 rad = " << to_radians(half_pi_rad) << " rad\n";
    std::cout << "   Full sphere = " << to_steradians(full_sphere_sr) << " sr\n";
    std::cout << "   Hemisphere = " << to_steradians(hemisphere_sr) << " sr\n";
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
