/**
 * test_raw_path_storage.cpp - Tests for RawPathStorage
 */

#include <gtest/gtest.h>
#include "diagnostics/raw_path_storage.hpp"
#include "diagnostics/path_types.hpp"

using namespace render;
using namespace render::diagnostics;

// Helper to create a simple PathTrace
PathTrace make_test_trace(BsdfType type, float contribution_r) {
    PathTrace trace;
    trace.contribution = RGB3f{contribution_r, contribution_r * 0.8f, contribution_r * 0.5f};
    trace.strategy = SamplingStrategy::BSDF;
    trace.light_type = LightSourceType::Area;
    
    // Camera -> Surface -> Light
    trace.add_vertex(0, 0, type);
    trace.add_vertex(1, 0, BsdfType::Emission);
    
    // Set positions
    trace.set_vertex_geometry(0, Vec3f{0.0f, 0.0f, 0.0f}, Vec3f{0.0f, 0.0f, 1.0f});
    trace.set_vertex_geometry(1, Vec3f{1.0f, 1.0f, 1.0f}, Vec3f{0.0f, 0.0f, -1.0f});
    
    return trace;
}

TEST(RawPathTest, InitFromTrace) {
    PathTrace trace = make_test_trace(BsdfType::Diffuse, 0.5f);
    
    RawPath path;
    path.init_from(trace);
    
    EXPECT_EQ(path.depth, 2);
    EXPECT_EQ(path.vertices.size(), 2);
    EXPECT_EQ(path.positions.size(), 2);
    EXPECT_EQ(path.strategy, SamplingStrategy::BSDF);
    EXPECT_EQ(path.light_type, LightSourceType::Area);
    EXPECT_FLOAT_EQ(path.contribution.r, 0.5f);
}

TEST(RawPathTest, SignatureString) {
    PathTrace trace = make_test_trace(BsdfType::Diffuse, 0.5f);
    
    RawPath path;
    path.init_from(trace);
    
    EXPECT_EQ(path.signature_string(), "CDL");
}

TEST(PixelRawPathsTest, AddPaths) {
    PixelRawPaths pixel;
    pixel.reserve(10);
    
    pixel.add_path(make_test_trace(BsdfType::Diffuse, 0.1f));
    pixel.add_path(make_test_trace(BsdfType::Diffuse, 0.2f));
    pixel.add_path(make_test_trace(BsdfType::Glossy, 0.3f));
    
    EXPECT_EQ(pixel.count(), 3);
    EXPECT_FLOAT_EQ(pixel.path(0).contribution.r, 0.1f);
    EXPECT_FLOAT_EQ(pixel.path(1).contribution.r, 0.2f);
    EXPECT_FLOAT_EQ(pixel.path(2).contribution.r, 0.3f);
}

TEST(RawPathStorageTest, InitAndRecord) {
    RawPathStorage storage;
    storage.init(100, 100, 4);
    
    // Record some paths
    storage.record_path(50, 50, make_test_trace(BsdfType::Diffuse, 0.1f));
    storage.record_path(50, 50, make_test_trace(BsdfType::Diffuse, 0.2f));
    storage.record_path(50, 50, make_test_trace(BsdfType::Glossy, 0.3f));
    storage.record_path(10, 10, make_test_trace(BsdfType::Diffuse, 0.4f));
    
    EXPECT_EQ(storage.width(), 100);
    EXPECT_EQ(storage.height(), 100);
    EXPECT_EQ(storage.total_path_count(), 4);
    EXPECT_EQ(storage.active_pixel_count(), 2);
    
    const auto& pixel = storage.pixel(50, 50);
    EXPECT_EQ(pixel.count(), 3);
}

TEST(RawPathStorageTest, OutOfBoundsIgnored) {
    RawPathStorage storage;
    storage.init(10, 10, 4);
    
    // Should not crash
    storage.record_path(100, 100, make_test_trace(BsdfType::Diffuse, 0.1f));
    
    EXPECT_EQ(storage.total_path_count(), 0);
}

TEST(PixelPathAnalyzerTest, GroupBySignature) {
    PixelRawPaths pixel;
    
    // Add 3 diffuse paths (same signature)
    pixel.add_path(make_test_trace(BsdfType::Diffuse, 0.1f));
    pixel.add_path(make_test_trace(BsdfType::Diffuse, 0.2f));
    pixel.add_path(make_test_trace(BsdfType::Diffuse, 0.3f));
    
    // Add 2 glossy paths (different signature)
    pixel.add_path(make_test_trace(BsdfType::Glossy, 0.5f));
    pixel.add_path(make_test_trace(BsdfType::Glossy, 0.6f));
    
    auto groups = PixelPathAnalyzer::analyze(pixel);
    
    EXPECT_EQ(groups.size(), 2);
    
    // Find diffuse group
    auto diffuse_it = std::find_if(groups.begin(), groups.end(),
        [](const PathGroupResult& g) { return g.signature_string == "CDL"; });
    ASSERT_NE(diffuse_it, groups.end());
    EXPECT_EQ(diffuse_it->sample_count, 3);
    EXPECT_EQ(diffuse_it->path_indices.size(), 3);
    
    // Find glossy group
    auto glossy_it = std::find_if(groups.begin(), groups.end(),
        [](const PathGroupResult& g) { return g.signature_string == "CGL"; });
    ASSERT_NE(glossy_it, groups.end());
    EXPECT_EQ(glossy_it->sample_count, 2);
    EXPECT_EQ(glossy_it->path_indices.size(), 2);
}

TEST(PixelPathAnalyzerTest, SortByMean) {
    PixelRawPaths pixel;
    
    // Low contribution diffuse
    pixel.add_path(make_test_trace(BsdfType::Diffuse, 0.1f));
    
    // High contribution glossy
    pixel.add_path(make_test_trace(BsdfType::Glossy, 0.9f));
    
    auto groups = PixelPathAnalyzer::analyze(pixel);
    
    // Should be sorted by mean (glossy first)
    EXPECT_EQ(groups[0].signature_string, "CGL");
    EXPECT_EQ(groups[1].signature_string, "CDL");
}

TEST(RawPathStorageTest, MemoryUsage) {
    RawPathStorage storage;
    storage.init(100, 100, 4);
    
    // Add some paths
    for (int i = 0; i < 100; ++i) {
        storage.record_path(i, 0, make_test_trace(BsdfType::Diffuse, 0.1f));
    }
    
    double mb = storage.memory_usage_mb();
    
    // Should be small for this test case
    EXPECT_GT(mb, 0.0);
    EXPECT_LT(mb, 10.0);  // Should be much less than 10MB
}
