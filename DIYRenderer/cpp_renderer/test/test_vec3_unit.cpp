/**
 * test_vec3_unit.cpp - Test for unit-aware Vec3
 */

#include "../include/math/vec3_unit.hpp"
#include <iostream>

using namespace mp_units::si::unit_symbols;

void test_basic() {
    std::cout << "=== Basic operations ===\n";
    
    // Create position and direction
    diy::Position3 pos(1.0f, 2.0f, 3.0f);
    diy::Direction3 dir(0.0f, 0.0f, 1.0f);
    
    std::cout << "pos = (" << pos.x_raw() << ", " << pos.y_raw() << ", " << pos.z_raw() << ") m\n";
    std::cout << "dir = (" << dir.x_raw() << ", " << dir.y_raw() << ", " << dir.z_raw() << ")\n";
    
    // Direction * distance -> Position
    auto dist = 5.0f * m;
    diy::Position3 offset = dir * dist;
    std::cout << "dir * 5m = (" << offset.x_raw() << ", " << offset.y_raw() << ", " << offset.z_raw() << ") m\n";
    
    // Position + Position -> Position
    diy::Position3 new_pos = pos + offset;
    std::cout << "pos + offset = (" << new_pos.x_raw() << ", " << new_pos.y_raw() << ", " << new_pos.z_raw() << ") m\n";
}

void test_dot_cross() {
    std::cout << "\n=== Dot and Cross ===\n";
    
    diy::Direction3 d1(1.0f, 0.0f, 0.0f);
    diy::Direction3 d2(0.0f, 1.0f, 0.0f);
    
    auto dot_result = diy::dot(d1, d2);
    std::cout << "d1 · d2 = " << dot_result.numerical_value_in(mp_units::one) << "\n";
    
    auto cross_result = diy::cross(d1, d2);
    std::cout << "d1 × d2 = (" << cross_result.x_raw() << ", " << cross_result.y_raw() << ", " << cross_result.z_raw() << ")\n";
}

void test_normalize() {
    std::cout << "\n=== Normalize ===\n";
    
    diy::Position3 pos(3.0f, 4.0f, 0.0f);
    auto len = pos.length();
    std::cout << "|pos| = " << len.numerical_value_in(m) << " m\n";
    
    diy::Direction3 dir = diy::normalize(pos);
    std::cout << "normalize(pos) = (" << dir.x_raw() << ", " << dir.y_raw() << ", " << dir.z_raw() << ")\n";
}

void test_ray_point() {
    std::cout << "\n=== Ray point calculation ===\n";
    
    diy::Position3 origin(0.0f, 0.0f, 0.0f);
    diy::Direction3 direction(0.577f, 0.577f, 0.577f);  // roughly (1,1,1) normalized
    auto t = 10.0f * m;
    
    // origin + direction * t
    diy::Position3 point = origin + direction * t;
    std::cout << "origin + dir * 10m = (" << point.x_raw() << ", " << point.y_raw() << ", " << point.z_raw() << ") m\n";
}

void test_radiometric() {
    std::cout << "\n=== Radiometric ===\n";
    
    using namespace diy::units;
    
    // Color (albedo) - dimensionless [0,1]
    diy::Vec3U<mp_units::one> albedo(0.8f, 0.2f, 0.1f);
    std::cout << "albedo = (" << albedo.x_raw() << ", " << albedo.y_raw() << ", " << albedo.z_raw() << ")\n";
    
    // Incoming radiance [W/(sr·m²)]
    diy::Radiance3 incoming(100.0f, 100.0f, 100.0f);
    std::cout << "incoming = (" << incoming.x_raw() << ", " << incoming.y_raw() << ", " << incoming.z_raw() << ") W/(sr·m²)\n";
    
    // Reflected radiance = albedo * incoming
    // Color3 * Radiance3 = Vec3<one> * Vec3<W/(sr·m²)> = Vec3<W/(sr·m²)> = Radiance3
    diy::Radiance3 reflected = albedo * incoming;
    std::cout << "reflected = albedo * incoming = (" << reflected.x_raw() << ", " 
              << reflected.y_raw() << ", " << reflected.z_raw() << ") W/(sr·m²)\n";
    
    // BSDF [1/sr]
    diy::BSDF3 bsdf(0.318f, 0.318f, 0.318f);  // ~1/π for Lambertian
    std::cout << "bsdf = (" << bsdf.x_raw() << ", " << bsdf.y_raw() << ", " << bsdf.z_raw() << ") 1/sr\n";
    
    // Throughput (dimensionless)
    diy::Vec3U<mp_units::one> throughput(1.0f, 1.0f, 1.0f);
    
    // PDF [1/sr]
    auto pdf = 0.225f * per_steradian;  // ~cos/π for cosine-weighted hemisphere
    
    // Path contribution: throughput * bsdf * radiance * cosθ / pdf
    // Units: [-] * [1/sr] * [W/(sr·m²)] * [-] / [1/sr] = [W/(sr·m²)]
    float cos_theta = 0.7f;
    auto contribution = throughput * bsdf * incoming * cos_theta / pdf;
    std::cout << "contribution = throughput * bsdf * incoming * cos / pdf\n";
    std::cout << "            = (" << contribution.x_raw() << ", "
              << contribution.y_raw() << ", " << contribution.z_raw() << ") W/(sr·m²)\n";
}

// Legacy Vec3 conversion test removed - Vec3 has been eliminated from codebase

int main() {
    test_basic();
    test_dot_cross();
    test_normalize();
    test_ray_point();
    test_radiometric();
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
