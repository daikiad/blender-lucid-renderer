/**
 * test_path_types.cpp - TDD tests for path type definitions
 * 
 * Test-first approach: These tests define the expected behavior
 * before implementation.
 */

#include <gtest/gtest.h>
#include "diagnostics/path_types.hpp"

using namespace render;
using namespace render::diagnostics;

// ============================================================================
// BsdfType Tests
// ============================================================================

TEST(BsdfTypeTest, EnumValues) {
    // Verify enum values are stable (used in serialization)
    EXPECT_EQ(static_cast<uint8_t>(BsdfType::Diffuse), 0);
    EXPECT_EQ(static_cast<uint8_t>(BsdfType::Glossy), 1);
    EXPECT_EQ(static_cast<uint8_t>(BsdfType::Mirror), 2);
    EXPECT_EQ(static_cast<uint8_t>(BsdfType::Glass), 3);
    EXPECT_EQ(static_cast<uint8_t>(BsdfType::Emission), 4);
    EXPECT_EQ(static_cast<uint8_t>(BsdfType::Environment), 5);
}

TEST(BsdfTypeTest, NameConversion) {
    EXPECT_STREQ(bsdf_type_name(BsdfType::Diffuse), "Diffuse");
    EXPECT_STREQ(bsdf_type_name(BsdfType::Glossy), "Glossy");
    EXPECT_STREQ(bsdf_type_name(BsdfType::Mirror), "Mirror");
    EXPECT_STREQ(bsdf_type_name(BsdfType::Glass), "Glass");
    EXPECT_STREQ(bsdf_type_name(BsdfType::Emission), "Emission");
    EXPECT_STREQ(bsdf_type_name(BsdfType::Environment), "Environment");
}

TEST(BsdfTypeTest, ShortName) {
    EXPECT_EQ(bsdf_type_short(BsdfType::Diffuse), 'D');
    EXPECT_EQ(bsdf_type_short(BsdfType::Glossy), 'G');
    EXPECT_EQ(bsdf_type_short(BsdfType::Mirror), 'M');
    EXPECT_EQ(bsdf_type_short(BsdfType::Glass), 'T');
    EXPECT_EQ(bsdf_type_short(BsdfType::Emission), 'L');
    EXPECT_EQ(bsdf_type_short(BsdfType::Environment), 'E');
}

// ============================================================================
// PathVertex Tests
// ============================================================================

TEST(PathVertexTest, Size) {
    // Must be 8 bytes for efficient storage
    EXPECT_EQ(sizeof(PathVertex), 8);
}

TEST(PathVertexTest, DefaultConstruction) {
    PathVertex v;
    EXPECT_EQ(v.object_id, -1);
    EXPECT_EQ(v.material_id, -1);
    EXPECT_EQ(v.bsdf_type, BsdfType::Diffuse);
    EXPECT_EQ(v.flags, 0);
}

TEST(PathVertexTest, Flags) {
    PathVertex v;
    
    // Initially false
    EXPECT_FALSE(v.is_delta());
    EXPECT_FALSE(v.is_light_sampled());
    
    // Set delta
    v.set_delta(true);
    EXPECT_TRUE(v.is_delta());
    EXPECT_FALSE(v.is_light_sampled());
    
    // Set light_sampled
    v.set_light_sampled(true);
    EXPECT_TRUE(v.is_delta());
    EXPECT_TRUE(v.is_light_sampled());
    
    // Unset delta
    v.set_delta(false);
    EXPECT_FALSE(v.is_delta());
    EXPECT_TRUE(v.is_light_sampled());
}

TEST(PathVertexTest, Equality) {
    PathVertex v1{1, 2, BsdfType::Glossy, 0};
    PathVertex v2{1, 2, BsdfType::Glossy, 0};
    PathVertex v3{1, 2, BsdfType::Mirror, 0};  // Different type
    PathVertex v4{2, 2, BsdfType::Glossy, 0};  // Different object
    
    EXPECT_EQ(v1, v2);
    EXPECT_NE(v1, v3);
    EXPECT_NE(v1, v4);
}

// ============================================================================
// PathTrace Tests
// ============================================================================

TEST(PathTraceTest, Clear) {
    PathTrace path;
    path.add_vertex(1, 0, BsdfType::Diffuse);
    path.contribution = render::RGB3f{1.0f, 0.5f, 0.0f};
    
    path.clear();
    
    EXPECT_EQ(path.depth, 0);
    EXPECT_EQ(path.contribution.r, 0.0f);
    EXPECT_EQ(path.contribution.g, 0.0f);
    EXPECT_EQ(path.contribution.b, 0.0f);
}

TEST(PathTraceTest, AddVertex) {
    PathTrace path;
    
    path.add_vertex(0, 1, BsdfType::Emission);
    EXPECT_EQ(path.depth, 1);
    EXPECT_EQ(path.vertices[0].object_id, 0);
    EXPECT_EQ(path.vertices[0].bsdf_type, BsdfType::Emission);
    
    path.add_vertex(2, 3, BsdfType::Glossy);
    EXPECT_EQ(path.depth, 2);
    EXPECT_EQ(path.vertices[1].object_id, 2);
}

TEST(PathTraceTest, MaxDepth) {
    PathTrace path;
    
    // Fill to max
    for (size_t i = 0; i < MAX_PATH_DEPTH; ++i) {
        path.add_vertex(static_cast<int32_t>(i), 0, BsdfType::Diffuse);
    }
    EXPECT_EQ(path.depth, MAX_PATH_DEPTH);
    
    // Try to add beyond max - should be ignored
    path.add_vertex(999, 0, BsdfType::Diffuse);
    EXPECT_EQ(path.depth, MAX_PATH_DEPTH);
    EXPECT_NE(path.vertices[MAX_PATH_DEPTH - 1].object_id, 999);
}

TEST(PathTraceTest, Hash) {
    PathTrace path1;
    path1.add_vertex(0, 1, BsdfType::Emission);
    path1.add_vertex(2, 3, BsdfType::Glossy);
    
    PathTrace path2;
    path2.add_vertex(0, 1, BsdfType::Emission);
    path2.add_vertex(2, 3, BsdfType::Glossy);
    
    PathTrace path3;
    path3.add_vertex(0, 1, BsdfType::Emission);
    path3.add_vertex(2, 3, BsdfType::Mirror);  // Different type
    
    // Same paths should have same hash
    EXPECT_EQ(path1.hash(), path2.hash());
    
    // Different paths should (very likely) have different hash
    EXPECT_NE(path1.hash(), path3.hash());
}

TEST(PathTraceTest, HashDifferentDepth) {
    PathTrace path1;
    path1.add_vertex(0, 1, BsdfType::Emission);
    
    PathTrace path2;
    path2.add_vertex(0, 1, BsdfType::Emission);
    path2.add_vertex(2, 3, BsdfType::Glossy);
    
    // Different depth = different hash
    EXPECT_NE(path1.hash(), path2.hash());
}

TEST(PathTraceTest, CoarseType) {
    // Direct, no delta, BSDF sampled
    PathTrace direct_diffuse;
    direct_diffuse.add_vertex(0, 0, BsdfType::Emission);
    direct_diffuse.add_vertex(1, 1, BsdfType::Diffuse);
    EXPECT_EQ(direct_diffuse.coarse_type() & 0x04, 0);  // is_direct
    EXPECT_EQ(direct_diffuse.coarse_type() & 0x02, 0);  // no delta
    
    // Indirect (depth > 2)
    PathTrace indirect;
    indirect.add_vertex(0, 0, BsdfType::Emission);
    indirect.add_vertex(1, 1, BsdfType::Diffuse);
    indirect.add_vertex(2, 2, BsdfType::Diffuse);
    EXPECT_NE(indirect.coarse_type() & 0x04, 0);  // not direct (indirect)
    
    // Has delta
    PathTrace with_delta;
    with_delta.add_vertex(0, 0, BsdfType::Emission);
    PathVertex mirror_v{1, 1, BsdfType::Mirror, PathVertex::FLAG_DELTA};
    with_delta.vertices[with_delta.depth++] = mirror_v;
    EXPECT_NE(with_delta.coarse_type() & 0x02, 0);  // has delta
}

TEST(PathTraceTest, CoarseTypeLightSampled) {
    PathTrace light_sampled;
    light_sampled.add_vertex(0, 0, BsdfType::Emission);
    PathVertex v{1, 1, BsdfType::Diffuse, PathVertex::FLAG_LIGHT_SAMPLED};
    light_sampled.vertices[light_sampled.depth++] = v;
    
    EXPECT_NE(light_sampled.coarse_type() & 0x01, 0);  // light sampled
}
