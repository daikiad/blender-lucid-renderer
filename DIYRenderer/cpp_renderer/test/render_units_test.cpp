/**
 * render_units_test.cpp - Tests for render_units.hpp
 *
 * Tests the type-safe rendering primitives built on mp-units ISQ.
 */

#include "units/render_units.hpp"

#include <cmath>
#include <iostream>
#include <sstream>

using namespace render;
using namespace mp_units;
using namespace mp_units::si::unit_symbols;

// Simple test framework
#define TEST(name)                                                  \
    void test_##name();                                             \
    struct TestRunner_##name {                                      \
        TestRunner_##name() {                                       \
            std::cout << "Running " #name "... ";                   \
            try {                                                   \
                test_##name();                                      \
                std::cout << "PASSED" << std::endl;                 \
            } catch (const std::exception& e) {                     \
                std::cout << "FAILED: " << e.what() << std::endl;   \
                exit(1);                                            \
            }                                                       \
        }                                                           \
    } runner_##name;                                                \
    void test_##name()

#define EXPECT_TRUE(cond)                                           \
    do {                                                            \
        if (!(cond)) {                                              \
            throw std::runtime_error(                               \
                "Assertion failed: " #cond " at line " +            \
                std::to_string(__LINE__));                          \
        }                                                           \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                      \
    do {                                                            \
        if (std::abs((a) - (b)) > (eps)) {                          \
            std::ostringstream oss;                                 \
            oss << "EXPECT_NEAR failed: " << (a) << " vs " << (b)   \
                << " (eps=" << (eps) << ") at line " << __LINE__;   \
            throw std::runtime_error(oss.str());                    \
        }                                                           \
    } while (0)

constexpr float kEps = 1e-5f;  // Float precision epsilon

// ============================================================================
// Vec3f Tests
// ============================================================================

TEST(Vec3f_arithmetic) {
    Vec3f a{1.0f, 2.0f, 3.0f};
    Vec3f b{4.0f, 5.0f, 6.0f};

    auto sum = a + b;
    EXPECT_NEAR(sum.x, 5.0f, kEps);
    EXPECT_NEAR(sum.y, 7.0f, kEps);
    EXPECT_NEAR(sum.z, 9.0f, kEps);

    auto diff = b - a;
    EXPECT_NEAR(diff.x, 3.0f, kEps);
    EXPECT_NEAR(diff.y, 3.0f, kEps);
    EXPECT_NEAR(diff.z, 3.0f, kEps);

    auto neg = -a;
    EXPECT_NEAR(neg.x, -1.0f, kEps);
}

TEST(Vec3f_scalar_multiply) {
    Vec3f v{1.0f, 2.0f, 3.0f};

    // float
    auto v2 = v * 2.0f;
    EXPECT_NEAR(v2.x, 2.0f, kEps);
    EXPECT_NEAR(v2.y, 4.0f, kEps);
    EXPECT_NEAR(v2.z, 6.0f, kEps);

    // int
    auto v3 = v * 3;
    EXPECT_NEAR(v3.x, 3.0f, kEps);

    // Commutative
    auto v4 = 2.0f * v;
    EXPECT_NEAR(v4.x, 2.0f, kEps);

    auto v5 = 3 * v;
    EXPECT_NEAR(v5.x, 3.0f, kEps);
}

TEST(Vec3f_dot_cross) {
    Vec3f a{1.0f, 0.0f, 0.0f};
    Vec3f b{0.0f, 1.0f, 0.0f};

    EXPECT_NEAR(dot(a, b), 0.0f, kEps);
    EXPECT_NEAR(dot(a, a), 1.0f, kEps);

    auto c = cross(a, b);
    EXPECT_NEAR(c.z, 1.0f, kEps);
}

TEST(Vec3f_normalize) {
    Vec3f v{3.0f, 4.0f, 0.0f};
    auto n = normalize(v);
    EXPECT_NEAR(n.length(), 1.0f, kEps);
    EXPECT_NEAR(n.x, 0.6f, kEps);
    EXPECT_NEAR(n.y, 0.8f, kEps);
}

// ============================================================================
// Vec2f Tests
// ============================================================================

TEST(Vec2f_basic) {
    Vec2f a{1.0f, 2.0f};
    Vec2f b{3.0f, 4.0f};

    auto sum = a + b;
    EXPECT_NEAR(sum.x, 4.0f, kEps);
    EXPECT_NEAR(sum.y, 6.0f, kEps);

    auto diff = b - a;
    EXPECT_NEAR(diff.x, 2.0f, kEps);
    EXPECT_NEAR(diff.y, 2.0f, kEps);

    auto neg = -a;
    EXPECT_NEAR(neg.x, -1.0f, kEps);
    EXPECT_NEAR(neg.y, -2.0f, kEps);
}

TEST(Vec2f_scalar_multiply) {
    Vec2f v{1.0f, 2.0f};

    auto v2 = v * 2.0f;
    EXPECT_NEAR(v2.x, 2.0f, kEps);
    EXPECT_NEAR(v2.y, 4.0f, kEps);

    auto v3 = 3 * v;
    EXPECT_NEAR(v3.x, 3.0f, kEps);
    EXPECT_NEAR(v3.y, 6.0f, kEps);
}

TEST(Vec2f_dot) {
    Vec2f a{1.0f, 0.0f};
    Vec2f b{0.0f, 1.0f};
    EXPECT_NEAR(dot(a, b), 0.0f, kEps);
    EXPECT_NEAR(dot(a, a), 1.0f, kEps);
}

TEST(Vec2f_normalize) {
    Vec2f v{3.0f, 4.0f};
    auto n = normalize(v);
    EXPECT_NEAR(n.length(), 1.0f, kEps);
    EXPECT_NEAR(n.x, 0.6f, kEps);
    EXPECT_NEAR(n.y, 0.8f, kEps);
}

// ============================================================================
// Displacement Tests
// ============================================================================

TEST(Displacement_basic) {
    Displacement d1 = quantity{Vec3f{1.0f, 0.0f, 0.0f}, isq::displacement[m]};
    Displacement d2 = quantity{Vec3f{0.0f, 2.0f, 0.0f}, isq::displacement[m]};

    auto sum = d1 + d2;
    auto v = sum.numerical_value_in(m);
    EXPECT_NEAR(v.x, 1.0f, kEps);
    EXPECT_NEAR(v.y, 2.0f, kEps);
}

TEST(Displacement_scalar_multiply) {
    Displacement d = quantity{Vec3f{1.0f, 2.0f, 3.0f}, isq::displacement[m]};

    // Multiply by float
    auto d2 = d * 2.0f;
    EXPECT_NEAR(d2.numerical_value_in(m).x, 2.0f, kEps);

    // Multiply by int (should work with proper traits)
    auto d3 = d * 3;
    EXPECT_NEAR(d3.numerical_value_in(m).x, 3.0f, kEps);

    // Commutative
    auto d4 = 4.0f * d;
    EXPECT_NEAR(d4.numerical_value_in(m).x, 4.0f, kEps);

    auto d5 = 5 * d;
    EXPECT_NEAR(d5.numerical_value_in(m).x, 5.0f, kEps);
}

// ============================================================================
// Position Tests (Affine Space)
// ============================================================================

TEST(Position_subtraction) {
    Position p1 = make_position(0.0f, 0.0f, 0.0f);
    Position p2 = make_position(3.0f, 4.0f, 0.0f);

    // Position - Position = Displacement (ISQ compliant!)
    auto delta = p2 - p1;
    Vec3f dv = delta.numerical_value_in(m);
    EXPECT_NEAR(dv.x, 3.0f, kEps);
    EXPECT_NEAR(dv.y, 4.0f, kEps);
}

TEST(Position_addition_with_displacement) {
    Position p = make_position(1.0f, 2.0f, 3.0f);
    Displacement offset = quantity{Vec3f{1.0f, 0.0f, 0.0f}, isq::displacement[m]};

    auto p2 = p + offset;
    Vec3f v = (p2 - world_origin).numerical_value_in(m);
    EXPECT_NEAR(v.x, 2.0f, kEps);
    EXPECT_NEAR(v.y, 2.0f, kEps);
}

TEST(Position_distance) {
    Position p1 = make_position(0.0f, 0.0f, 0.0f);
    Position p2 = make_position(3.0f, 4.0f, 0.0f);

    Length dist = disp_length(p2 - p1);
    EXPECT_NEAR(dist.numerical_value_in(m), 5.0f, kEps);
}

// ============================================================================
// Direction Tests
// ============================================================================

TEST(Direction_construction) {
    auto d_opt = make_direction(3.0f, 4.0f, 0.0f);
    EXPECT_TRUE(d_opt.has_value());
    Direction d = *d_opt;
    EXPECT_NEAR(d.x(), 0.6f, kEps);
    EXPECT_NEAR(d.y(), 0.8f, kEps);
    EXPECT_NEAR(d.z(), 0.0f, kEps);

    // Length should be 1
    float len = std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
    EXPECT_NEAR(len, 1.0f, kEps);
}

TEST(Direction_zero_vector) {
    // Zero vector should return nullopt
    auto d = make_direction(0.0f, 0.0f, 0.0f);
    EXPECT_TRUE(!d.has_value());

    // Very small vector should also return nullopt
    auto d2 = make_direction(1e-15f, 0.0f, 0.0f);
    EXPECT_TRUE(!d2.has_value());
}

TEST(Direction_times_Length) {
    Direction d = *make_direction(1.0f, 0.0f, 0.0f);
    Length len = 5.0f * m;

    Displacement disp = d * len;
    Vec3f v = disp.numerical_value_in(m);
    EXPECT_NEAR(v.x, 5.0f, kEps);
    EXPECT_NEAR(v.y, 0.0f, kEps);
    EXPECT_NEAR(v.z, 0.0f, kEps);

    // Commutative
    Displacement disp2 = len * d;
    EXPECT_NEAR(disp2.numerical_value_in(m).x, 5.0f, kEps);
}

TEST(Direction_negation) {
    Direction d = *make_direction(1.0f, 0.0f, 0.0f);
    Direction neg = -d;
    EXPECT_NEAR(neg.x(), -1.0f, kEps);
}

TEST(Direction_cross_product) {
    Direction d1 = *make_direction(1.0f, 0.0f, 0.0f);
    Direction d2 = *make_direction(0.0f, 1.0f, 0.0f);

    auto cross_result = d1.cross(d2);
    EXPECT_TRUE(cross_result.has_value());
    EXPECT_NEAR(cross_result->z(), 1.0f, kEps);

    // Cross of parallel vectors should be nullopt
    auto cross_parallel = d1.cross(d1);
    EXPECT_TRUE(!cross_parallel.has_value());
}

// ============================================================================
// Normal Tests
// ============================================================================

TEST(Normal_construction) {
    auto n_opt = make_normal(0.0f, 3.0f, 4.0f);
    EXPECT_TRUE(n_opt.has_value());
    Normal n = *n_opt;
    EXPECT_NEAR(n.y(), 0.6f, kEps);
    EXPECT_NEAR(n.z(), 0.8f, kEps);
}

TEST(Normal_zero_vector) {
    // Zero vector should return nullopt
    auto n = make_normal(0.0f, 0.0f, 0.0f);
    EXPECT_TRUE(!n.has_value());
}

TEST(Normal_dot_direction) {
    Normal n = *make_normal(0.0f, 1.0f, 0.0f);
    Direction d = *make_direction(0.0f, 1.0f, 0.0f);

    EXPECT_NEAR(dot(n, d), 1.0f, kEps);
    EXPECT_NEAR(dot(d, n), 1.0f, kEps);
    EXPECT_NEAR(n.dot(d), 1.0f, kEps);
}

// ============================================================================
// Reflection Tests
// ============================================================================

TEST(Reflect_basic) {
    // Ray coming from upper-right, hitting horizontal surface
    Direction incident = *make_direction(1.0f, -1.0f, 0.0f);
    Normal normal = *make_normal(0.0f, 1.0f, 0.0f);

    Direction reflected = reflect(incident, normal);

    // Should reflect to upper-right
    EXPECT_NEAR(reflected.x(), incident.x(), kEps);
    EXPECT_NEAR(reflected.y(), -incident.y(), kEps);
}

// ============================================================================
// Refraction Tests
// ============================================================================

TEST(Refract_basic) {
    // Normal incidence - should pass straight through
    Direction incident = *make_direction(0.0f, -1.0f, 0.0f);
    Normal normal = *make_normal(0.0f, 1.0f, 0.0f);

    auto refracted = refract(incident, normal, 1.5f);
    EXPECT_TRUE(refracted.has_value());
    EXPECT_NEAR(refracted->y(), -1.0f, kEps);
}

TEST(Refract_total_internal_reflection) {
    // Grazing angle from denser medium
    Direction incident = *make_direction(0.9f, -0.436f, 0.0f);  // About 64 degrees
    Normal normal = *make_normal(0.0f, 1.0f, 0.0f);

    // eta = 1.5 (glass to air), critical angle ~42 degrees
    auto refracted = refract(incident, normal, 1.5f);
    EXPECT_TRUE(!refracted.has_value());  // TIR
}

// ============================================================================
// PDF Tests (mp-units quantity)
// ============================================================================

TEST(PDF_construction) {
    // PDF per solid angle: 1/(4π) per steradian (uniform sphere)
    constexpr float inv_4pi = 1.0f / (4.0f * static_cast<float>(M_PI));
    PdfW pdf_w = inv_4pi * per_sr;

    EXPECT_NEAR(pdf_w.numerical_value_in(per_sr), inv_4pi, kEps);
}

TEST(PDF_arithmetic) {
    PdfW pdf1 = 1.0f * per_sr;
    PdfW pdf2 = 2.0f * per_sr;

    auto sum = pdf1 + pdf2;
    EXPECT_NEAR(sum.numerical_value_in(per_sr), 3.0f, kEps);

    auto scaled = pdf1 * 3.0f;
    EXPECT_NEAR(scaled.numerical_value_in(per_sr), 3.0f, kEps);
}

TEST(PDF_types_distinct) {
    PdfW pdf_w = 1.0f * per_sr;
    PdfA pdf_a = 2.0f * per_m2;

    // These are different types - cannot be added directly
    // pdf_w + pdf_a; // This should NOT compile

    EXPECT_NEAR(pdf_w.numerical_value_in(per_sr), 1.0f, kEps);
    EXPECT_NEAR(pdf_a.numerical_value_in(per_m2), 2.0f, kEps);
}

// ============================================================================
// RGB Tests
// ============================================================================

TEST(ColorRGB_basic) {
    ColorRGB c1{0.5f, 0.6f, 0.7f};
    ColorRGB c2{0.1f, 0.2f, 0.3f};

    auto sum = c1 + c2;
    EXPECT_NEAR(sum.r, 0.6f, kEps);
    EXPECT_NEAR(sum.g, 0.8f, kEps);
    EXPECT_NEAR(sum.b, 1.0f, kEps);
}

TEST(ColorRGB_scalar) {
    ColorRGB c{0.2f, 0.4f, 0.6f};

    auto c2 = c * 2.0f;
    EXPECT_NEAR(c2.r, 0.4f, kEps);

    auto c3 = 3.0f * c;
    EXPECT_NEAR(c3.r, 0.6f, kEps);
}

TEST(ColorRGB_hadamard) {
    ColorRGB a{0.5f, 0.5f, 0.5f};
    ColorRGB b{0.2f, 0.4f, 0.6f};

    auto c = a * b;  // operator* is element-wise for ColorRGB
    EXPECT_NEAR(c.r, 0.1f, kEps);
    EXPECT_NEAR(c.g, 0.2f, kEps);
    EXPECT_NEAR(c.b, 0.3f, kEps);
}

TEST(ColorRGB_luminance) {
    ColorRGB white{1.0f, 1.0f, 1.0f};
    EXPECT_NEAR(luminance(white), 1.0f, kEps);

    ColorRGB red{1.0f, 0.0f, 0.0f};
    EXPECT_NEAR(luminance(red), 0.2126f, kEps);
}

// ============================================================================
// Radiance Tests
// ============================================================================

TEST(Radiance_basic) {
    using namespace mp_units::si::unit_symbols;

    Radiance r = 100.0f * W / (sr * m2);
    EXPECT_NEAR(r.numerical_value_in(W / (sr * m2)), 100.0f, kEps);
}

TEST(RadianceRGB_hadamard) {
    using namespace mp_units::si::unit_symbols;

    RadianceRGB rad{100.0f * W / (sr * m2), 200.0f * W / (sr * m2), 300.0f * W / (sr * m2)};
    ColorRGB albedo{0.5f, 0.5f, 0.5f};

    auto result = albedo * rad;  // operator* for ColorRGB * RadianceRGB
    EXPECT_NEAR(result.r.numerical_value_in(W / (sr * m2)), 50.0f, kEps);
    EXPECT_NEAR(result.g.numerical_value_in(W / (sr * m2)), 100.0f, kEps);
    EXPECT_NEAR(result.b.numerical_value_in(W / (sr * m2)), 150.0f, kEps);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== render_units.hpp tests ===" << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
