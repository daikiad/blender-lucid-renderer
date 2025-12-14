/**
 * test_geometry.cpp - Geometry Intersection Tests
 *
 * Comprehensive tests for ray-primitive intersections:
 * - Triangle intersection (Möller-Trumbore)
 * - AABB intersection (slab test)
 * - Sphere intersection
 * - Rectangle/Ellipse intersection
 */

#include "geometry/intersection.hpp"
#include "core/ray.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace render;
using namespace geometry;
using namespace mp_units::si::unit_symbols;

constexpr float kEps = 1e-4f;

// ============================================================================
// Test Fixtures
// ============================================================================

class TriangleIntersectionTest : public ::testing::Test {
protected:
    // Standard triangle in XY plane at z=0
    Position v0 = make_position(0.0f, 0.0f, 0.0f);
    Position v1 = make_position(1.0f, 0.0f, 0.0f);
    Position v2 = make_position(0.0f, 1.0f, 0.0f);
    
    // Degenerate (zero-area) triangle
    Position dv0 = make_position(0.0f, 0.0f, 0.0f);
    Position dv1 = make_position(1.0f, 0.0f, 0.0f);
    Position dv2 = make_position(2.0f, 0.0f, 0.0f);  // Collinear
};

class AABBIntersectionTest : public ::testing::Test {
protected:
    // Unit box centered at origin
    Position bmin = make_position(-1.0f, -1.0f, -1.0f);
    Position bmax = make_position(1.0f, 1.0f, 1.0f);
};

class SphereIntersectionTest : public ::testing::Test {
protected:
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Length radius = 1.0f * m;
};

// ============================================================================
// Triangle Intersection Tests
// ============================================================================

TEST_F(TriangleIntersectionTest, HitCenter) {
    // Ray from above, hitting triangle center
    Ray ray(make_position(0.25f, 0.25f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->numerical_value_in(m), 1.0f, kEps);
    EXPECT_NEAR(bary_u(uv), 0.25f, kEps);
    EXPECT_NEAR(bary_v(uv), 0.25f, kEps);
}

TEST_F(TriangleIntersectionTest, HitVertex0) {
    // Ray hitting vertex v0
    Ray ray(make_position(0.0f, 0.0f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(bary_u(uv), 0.0f, kEps);
    EXPECT_NEAR(bary_v(uv), 0.0f, kEps);
    EXPECT_NEAR(bary_w(uv), 1.0f, kEps);
}

TEST_F(TriangleIntersectionTest, HitVertex1) {
    // Ray hitting vertex v1
    Ray ray(make_position(1.0f, 0.0f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(bary_u(uv), 1.0f, kEps);
    EXPECT_NEAR(bary_v(uv), 0.0f, kEps);
}

TEST_F(TriangleIntersectionTest, HitVertex2) {
    // Ray hitting vertex v2
    Ray ray(make_position(0.0f, 1.0f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(bary_u(uv), 0.0f, kEps);
    EXPECT_NEAR(bary_v(uv), 1.0f, kEps);
}

TEST_F(TriangleIntersectionTest, HitEdge01) {
    // Ray hitting edge v0-v1
    Ray ray(make_position(0.5f, 0.0f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(bary_u(uv), 0.5f, kEps);
    EXPECT_NEAR(bary_v(uv), 0.0f, kEps);
}

TEST_F(TriangleIntersectionTest, MissOutside) {
    // Ray missing triangle (outside)
    Ray ray(make_position(2.0f, 2.0f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    EXPECT_FALSE(result.has_value());
}

TEST_F(TriangleIntersectionTest, MissParallel) {
    // Ray parallel to triangle
    Ray ray(make_position(0.25f, 0.25f, 1.0f), 
            *make_direction(1.0f, 0.0f, 0.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    EXPECT_FALSE(result.has_value());
}

TEST_F(TriangleIntersectionTest, MissBehind) {
    // Ray pointing away from triangle
    Ray ray(make_position(0.25f, 0.25f, 1.0f), 
            *make_direction(0.0f, 0.0f, 1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    EXPECT_FALSE(result.has_value());
}

TEST_F(TriangleIntersectionTest, HitFromBehind) {
    // Ray from below, hitting triangle back face
    Ray ray(make_position(0.25f, 0.25f, -1.0f), 
            *make_direction(0.0f, 0.0f, 1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    // Möller-Trumbore should still report hit (double-sided)
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->numerical_value_in(m), 1.0f, kEps);
}

TEST_F(TriangleIntersectionTest, DegenerateTriangle) {
    // Collinear vertices - zero area triangle
    Ray ray(make_position(1.0f, 0.0f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, dv0, dv1, dv2, uv);
    
    EXPECT_FALSE(result.has_value());
}

TEST_F(TriangleIntersectionTest, TMinConstraint) {
    // Hit should be rejected by tMin
    Ray ray(make_position(0.25f, 0.25f, 0.5f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv, 1.0f * m, RAY_T_MAX);
    
    EXPECT_FALSE(result.has_value());  // t=0.5 < tMin=1.0
}

TEST_F(TriangleIntersectionTest, TMaxConstraint) {
    // Hit should be rejected by tMax
    Ray ray(make_position(0.25f, 0.25f, 2.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv, RAY_T_MIN, 1.0f * m);
    
    EXPECT_FALSE(result.has_value());  // t=2.0 > tMax=1.0
}

TEST_F(TriangleIntersectionTest, BarycentricSum) {
    // Barycentric coordinates should sum to 1
    Ray ray(make_position(0.3f, 0.2f, 1.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    BarycentricCoord2 uv;
    auto result = intersectTriangle(ray, v0, v1, v2, uv);
    
    ASSERT_TRUE(result.has_value());
    float sum = bary_u(uv) + bary_v(uv) + bary_w(uv);
    EXPECT_NEAR(sum, 1.0f, kEps);
}

// ============================================================================
// AABB Intersection Tests
// ============================================================================

TEST_F(AABBIntersectionTest, HitFromFront) {
    // Ray from +Z hitting front face
    Ray ray(make_position(0.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, HitFromSide) {
    // Ray from +X hitting side face
    Ray ray(make_position(5.0f, 0.0f, 0.0f), 
            *make_direction(-1.0f, 0.0f, 0.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, HitDiagonal) {
    // Ray hitting box diagonally
    Ray ray(make_position(5.0f, 5.0f, 5.0f), 
            *make_direction(-1.0f, -1.0f, -1.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, MissAbove) {
    // Ray missing above box
    Ray ray(make_position(0.0f, 5.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_FALSE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, MissSide) {
    // Ray missing to the side
    Ray ray(make_position(5.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_FALSE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, RayInsideBox) {
    // Ray starting inside box
    Ray ray(make_position(0.0f, 0.0f, 0.0f), 
            *make_direction(1.0f, 0.0f, 0.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, ParallelToXAxis) {
    // Ray parallel to X axis, inside Y/Z bounds
    Ray ray(make_position(-5.0f, 0.0f, 0.0f), 
            *make_direction(1.0f, 0.0f, 0.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, ParallelToXAxisMiss) {
    // Ray parallel to X axis, outside Y bounds
    Ray ray(make_position(-5.0f, 5.0f, 0.0f), 
            *make_direction(1.0f, 0.0f, 0.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_FALSE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, HitCorner) {
    // Ray hitting exact corner
    Ray ray(make_position(2.0f, 2.0f, 2.0f), 
            *make_direction(-1.0f, -1.0f, -1.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, HitEdge) {
    // Ray hitting box edge
    Ray ray(make_position(0.0f, 2.0f, 2.0f), 
            *make_direction(0.0f, -1.0f, -1.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

TEST_F(AABBIntersectionTest, GrazingRay) {
    // Ray grazing box surface
    Ray ray(make_position(-5.0f, 1.0f, 0.0f), 
            *make_direction(1.0f, 0.0f, 0.0f));
    Vec3f invDir = computeInverseDirection(ray.direction);
    
    // Should hit (touching at y=1)
    EXPECT_TRUE(intersectAABB(ray, invDir, bmin, bmax));
}

// ============================================================================
// Sphere Intersection Tests
// ============================================================================

TEST_F(SphereIntersectionTest, HitCenter) {
    // Ray through sphere center
    Ray ray(make_position(0.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    ASSERT_TRUE(intersectSphere(ray, center, radius, tHit, hitNormal));
    EXPECT_NEAR(tHit.numerical_value_in(m), 4.0f, kEps);  // 5 - 1 = 4
    EXPECT_NEAR(hitNormal.z(), 1.0f, kEps);  // Normal pointing +Z
}

TEST_F(SphereIntersectionTest, HitOffCenter) {
    // Ray hitting sphere off-center
    Ray ray(make_position(0.5f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    ASSERT_TRUE(intersectSphere(ray, center, radius, tHit, hitNormal));
    // Should hit at z ≈ sqrt(1 - 0.5²) ≈ 0.866
    float expectedZ = std::sqrt(1.0f - 0.25f);
    EXPECT_NEAR(tHit.numerical_value_in(m), 5.0f - expectedZ, kEps);
}

TEST_F(SphereIntersectionTest, MissTangent) {
    // Ray tangent to sphere (just missing)
    Ray ray(make_position(1.01f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    EXPECT_FALSE(intersectSphere(ray, center, radius, tHit, hitNormal));
}

TEST_F(SphereIntersectionTest, MissFarAway) {
    // Ray completely missing sphere
    Ray ray(make_position(5.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    EXPECT_FALSE(intersectSphere(ray, center, radius, tHit, hitNormal));
}

TEST_F(SphereIntersectionTest, RayInsideSphere) {
    // Ray starting inside sphere
    Ray ray(make_position(0.0f, 0.0f, 0.0f), 
            *make_direction(0.0f, 0.0f, 1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    ASSERT_TRUE(intersectSphere(ray, center, radius, tHit, hitNormal));
    EXPECT_NEAR(tHit.numerical_value_in(m), 1.0f, kEps);
    EXPECT_NEAR(hitNormal.z(), 1.0f, kEps);
}

TEST_F(SphereIntersectionTest, RayBehindSphere) {
    // Ray pointing away from sphere
    Ray ray(make_position(0.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, 1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    EXPECT_FALSE(intersectSphere(ray, center, radius, tHit, hitNormal));
}

TEST_F(SphereIntersectionTest, NormalIsNormalized) {
    // Verify hit normal is unit length
    Ray ray(make_position(0.5f, 0.5f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Normal hitNormal;
    
    if (intersectSphere(ray, center, radius, tHit, hitNormal)) {
        float len = std::sqrt(hitNormal.x() * hitNormal.x() + 
                              hitNormal.y() * hitNormal.y() + 
                              hitNormal.z() * hitNormal.z());
        EXPECT_NEAR(len, 1.0f, kEps);
    }
}

// ============================================================================
// computeInverseDirection Tests
// ============================================================================

TEST(InverseDirectionTest, BasicDirection) {
    Direction dir = *make_direction(1.0f, 2.0f, 3.0f);
    Vec3f invDir = computeInverseDirection(dir);
    
    // Verify inverse is correct
    float len = std::sqrt(1.0f + 4.0f + 9.0f);
    EXPECT_NEAR(invDir.x * dir.x(), 1.0f, kEps);
    EXPECT_NEAR(invDir.y * dir.y(), 1.0f, kEps);
    EXPECT_NEAR(invDir.z * dir.z(), 1.0f, kEps);
}

TEST(InverseDirectionTest, AxisAligned) {
    Direction dir = *make_direction(1.0f, 0.0f, 0.0f);
    Vec3f invDir = computeInverseDirection(dir);
    
    EXPECT_NEAR(invDir.x, 1.0f, kEps);
    EXPECT_EQ(invDir.y, 0.0f);  // Parallel handling
    EXPECT_EQ(invDir.z, 0.0f);
}

TEST(InverseDirectionTest, NearParallel) {
    // Direction nearly parallel to axis
    Direction dir = *make_direction(1.0f, 1e-10f, 1e-10f);
    Vec3f invDir = computeInverseDirection(dir);
    
    // Near-zero components should return 0 (not inf)
    EXPECT_EQ(invDir.y, 0.0f);
    EXPECT_EQ(invDir.z, 0.0f);
}

// ============================================================================
// Rectangle Intersection Tests
// ============================================================================

TEST(RectangleIntersectionTest, HitCenter) {
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Direction normal = *make_direction(0.0f, 0.0f, 1.0f);
    Direction right = *make_direction(1.0f, 0.0f, 0.0f);
    Direction up = *make_direction(0.0f, 1.0f, 0.0f);
    Length sizeX = 2.0f * m;
    Length sizeY = 2.0f * m;
    
    Ray ray(make_position(0.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Position hitPoint;
    
    ASSERT_TRUE(intersectRectangle(ray, center, normal, right, up, sizeX, sizeY, tHit, hitPoint));
    EXPECT_NEAR(tHit.numerical_value_in(m), 5.0f, kEps);
}

TEST(RectangleIntersectionTest, MissOutside) {
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Direction normal = *make_direction(0.0f, 0.0f, 1.0f);
    Direction right = *make_direction(1.0f, 0.0f, 0.0f);
    Direction up = *make_direction(0.0f, 1.0f, 0.0f);
    Length sizeX = 2.0f * m;
    Length sizeY = 2.0f * m;
    
    Ray ray(make_position(5.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Position hitPoint;
    
    EXPECT_FALSE(intersectRectangle(ray, center, normal, right, up, sizeX, sizeY, tHit, hitPoint));
}

TEST(RectangleIntersectionTest, ParallelRay) {
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Direction normal = *make_direction(0.0f, 0.0f, 1.0f);
    Direction right = *make_direction(1.0f, 0.0f, 0.0f);
    Direction up = *make_direction(0.0f, 1.0f, 0.0f);
    Length sizeX = 2.0f * m;
    Length sizeY = 2.0f * m;
    
    Ray ray(make_position(0.0f, 0.0f, 5.0f), 
            *make_direction(1.0f, 0.0f, 0.0f));
    
    Length tHit;
    Position hitPoint;
    
    EXPECT_FALSE(intersectRectangle(ray, center, normal, right, up, sizeX, sizeY, tHit, hitPoint));
}

// ============================================================================
// Ellipse Intersection Tests
// ============================================================================

TEST(EllipseIntersectionTest, HitCenter) {
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Direction normal = *make_direction(0.0f, 0.0f, 1.0f);
    Direction right = *make_direction(1.0f, 0.0f, 0.0f);
    Direction up = *make_direction(0.0f, 1.0f, 0.0f);
    Length radiusX = 1.0f * m;
    Length radiusY = 1.0f * m;
    
    Ray ray(make_position(0.0f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Position hitPoint;
    
    ASSERT_TRUE(intersectEllipse(ray, center, normal, right, up, radiusX, radiusY, tHit, hitPoint));
    EXPECT_NEAR(tHit.numerical_value_in(m), 5.0f, kEps);
}

TEST(EllipseIntersectionTest, MissOutsideCircle) {
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Direction normal = *make_direction(0.0f, 0.0f, 1.0f);
    Direction right = *make_direction(1.0f, 0.0f, 0.0f);
    Direction up = *make_direction(0.0f, 1.0f, 0.0f);
    Length radiusX = 1.0f * m;
    Length radiusY = 1.0f * m;
    
    Ray ray(make_position(1.5f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Position hitPoint;
    
    EXPECT_FALSE(intersectEllipse(ray, center, normal, right, up, radiusX, radiusY, tHit, hitPoint));
}

TEST(EllipseIntersectionTest, HitEllipseBoundary) {
    Position center = make_position(0.0f, 0.0f, 0.0f);
    Direction normal = *make_direction(0.0f, 0.0f, 1.0f);
    Direction right = *make_direction(1.0f, 0.0f, 0.0f);
    Direction up = *make_direction(0.0f, 1.0f, 0.0f);
    Length radiusX = 2.0f * m;  // Wider
    Length radiusY = 1.0f * m;
    
    // Hit within ellipse (x=1.5, y=0 → (1.5/2)² + (0/1)² = 0.5625 < 1)
    Ray ray(make_position(1.5f, 0.0f, 5.0f), 
            *make_direction(0.0f, 0.0f, -1.0f));
    
    Length tHit;
    Position hitPoint;
    
    ASSERT_TRUE(intersectEllipse(ray, center, normal, right, up, radiusX, radiusY, tHit, hitPoint));
}
