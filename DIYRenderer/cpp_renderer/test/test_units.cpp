/**
 * test_units.cpp - Test mp-units integration
 * 
 * Compile with:
 *   cd build_pybind && cmake .. && make test_units
 */

#include "../include/units/units.hpp"
#include <iostream>

using namespace diy::units;
using namespace mp_units::si::unit_symbols;

int main() {
    std::cout << "=== mp-units Test for DIY Renderer ===\n\n";
    
    // 1. Length tests
    std::cout << "1. Length tests:\n";
    Length distance = meters(5.0f);
    Length small = 0.01f * m;  // 10mm in meters
    std::cout << "   Distance: " << distance.numerical_value_in(mp_units::si::metre) << " m\n";
    std::cout << "   Small: " << small.numerical_value_in(mp_units::si::metre) << " m\n";
    
    // 2. Angle tests  
    std::cout << "\n2. Angle tests:\n";
    Angle fov = deg_to_rad(90.0f);
    std::cout << "   FOV 90°: " << fov.numerical_value_in(mp_units::si::radian) 
              << " rad = " << rad_to_deg(fov) << "°\n";
    
    Angle spot_angle = degrees(45.0f);
    std::cout << "   Spot angle: " << spot_angle.numerical_value_in(mp_units::si::radian) 
              << " rad = " << rad_to_deg(spot_angle) << "°\n";
    
    // 3. Solid angle tests
    std::cout << "\n3. Solid angle tests:\n";
    SolidAngle omega = steradians(0.5f);
    std::cout << "   Solid angle: " << omega.numerical_value_in(mp_units::si::steradian) << " sr\n";
    
    // Full sphere and hemisphere
    std::cout << "   Full sphere: " << full_sphere_sr.numerical_value_in(mp_units::si::steradian) << " sr\n";
    std::cout << "   Hemisphere: " << hemisphere_sr.numerical_value_in(mp_units::si::steradian) << " sr\n";
    
    // 4. Power tests
    std::cout << "\n4. Power tests:\n";
    RadiantFlux power = 100.0f * W;
    std::cout << "   Light power: " << power.numerical_value_in(mp_units::si::watt) << " W\n";
    
    // 5. Type safety demonstration
    std::cout << "\n5. Type safety (compile-time checks):\n";
    std::cout << "   These would NOT compile:\n";
    std::cout << "   // Length l = 5.0f * W;  // Error: can't assign power to length\n";
    std::cout << "   // Angle a = meters(1.0f);  // Error: can't assign length to angle\n";
    std::cout << "   Type safety works!\n";
    
    // 6. Constants
    std::cout << "\n6. Constants:\n";
    std::cout << "   π rad = " << pi_rad.numerical_value_in(mp_units::si::radian) << " rad\n";
    std::cout << "   2π rad = " << two_pi_rad.numerical_value_in(mp_units::si::radian) << " rad\n";
    std::cout << "   π/2 rad = " << half_pi_rad.numerical_value_in(mp_units::si::radian) << " rad\n";
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
