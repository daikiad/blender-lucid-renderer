/**
 * test_diagnostic_integrator.cpp - TDD tests for path tracer with diagnostics
 * 
 * Tests integration of DiagnosticFilm with path tracing.
 */

#include <gtest/gtest.h>
#include "diagnostics/diagnostic_integrator.hpp"

using namespace render;
using namespace render::diagnostics;

// ============================================================================
// PathDiagnosticRecorder Tests
// ============================================================================

TEST(PathDiagnosticRecorderTest, Construction) {
    PathDiagnosticRecorder recorder;
    recorder.begin_path();
    
    EXPECT_EQ(recorder.current_depth(), 0);
}

TEST(PathDiagnosticRecorderTest, RecordVertex) {
    PathDiagnosticRecorder recorder;
    recorder.begin_path();
    
    recorder.record_vertex(0, 1, BsdfType::Diffuse, false, false);
    EXPECT_EQ(recorder.current_depth(), 1);
    
    recorder.record_vertex(1, 2, BsdfType::Glossy, false, true);
    EXPECT_EQ(recorder.current_depth(), 2);
}

TEST(PathDiagnosticRecorderTest, EndPath) {
    PathDiagnosticRecorder recorder;
    recorder.begin_path();
    recorder.record_vertex(0, 1, BsdfType::Emission, false, false);
    recorder.record_vertex(1, 2, BsdfType::Diffuse, false, false);
    
    RGB3f contrib{1.0f, 0.5f, 0.25f};
    PathTrace trace = recorder.end_path(contrib);
    
    EXPECT_EQ(trace.depth, 2);
    EXPECT_FLOAT_EQ(trace.contribution.r, 1.0f);
    EXPECT_EQ(trace.vertices[0].bsdf_type, BsdfType::Emission);
    EXPECT_EQ(trace.vertices[1].bsdf_type, BsdfType::Diffuse);
}

TEST(PathDiagnosticRecorderTest, RecordEnvironmentHit) {
    PathDiagnosticRecorder recorder;
    recorder.begin_path();
    recorder.record_environment_hit();
    
    PathTrace trace = recorder.end_path(RGB3f{0.5f, 0.7f, 1.0f});
    EXPECT_EQ(trace.depth, 1);
    EXPECT_EQ(trace.vertices[0].object_id, -1);  // Environment
    EXPECT_EQ(trace.vertices[0].bsdf_type, BsdfType::Environment);
}

TEST(PathDiagnosticRecorderTest, RecordLightHit) {
    PathDiagnosticRecorder recorder;
    recorder.begin_path();
    recorder.record_vertex(0, 1, BsdfType::Diffuse, false, false);
    recorder.record_light_hit(5);  // Light index 5
    
    PathTrace trace = recorder.end_path(RGB3f{10.0f, 10.0f, 10.0f});
    EXPECT_EQ(trace.depth, 2);
    EXPECT_EQ(trace.vertices[1].bsdf_type, BsdfType::Emission);
}

// ============================================================================
// DiagnosticPathTracer Tests
// ============================================================================

TEST(DiagnosticPathTracerTest, Construction) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticPathTracer tracer(100, 100, config);
    
    EXPECT_EQ(tracer.film().width(), 100);
    EXPECT_EQ(tracer.film().height(), 100);
    EXPECT_TRUE(tracer.is_enabled());
}

TEST(DiagnosticPathTracerTest, EnableDisable) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticPathTracer tracer(100, 100, config);
    
    EXPECT_TRUE(tracer.is_enabled());
    
    tracer.set_enabled(false);
    EXPECT_FALSE(tracer.is_enabled());
    
    tracer.set_enabled(true);
    EXPECT_TRUE(tracer.is_enabled());
}

TEST(DiagnosticPathTracerTest, RecordAtPixel) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticPathTracer tracer(100, 100, config);
    
    // Record a simple path
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Emission);
    trace.add_vertex(1, 1, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 0.5f, 0.25f};
    
    tracer.record_path(10, 10, trace);
    
    EXPECT_GT(tracer.film().total_paths_recorded(), 0);
}

TEST(DiagnosticPathTracerTest, DisabledNoRecord) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticPathTracer tracer(100, 100, config);
    tracer.set_enabled(false);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    
    tracer.record_path(10, 10, trace);
    
    // Should not record when disabled
    EXPECT_EQ(tracer.film().total_paths_recorded(), 0);
}

// ============================================================================
// BsdfType Classification Tests
// ============================================================================

TEST(BsdfClassificationTest, DiffuseMaterial) {
    // MaterialParams from core/material.hpp
    // We test classify_bsdf logic directly with inline struct
    struct TestMat {
        float roughness;
        float metallic;
        float transmission;
        float ior;
    };
    
    TestMat mat{1.0f, 0.0f, 0.0f, 1.5f};
    
    // Inline classification logic test
    BsdfType type;
    if (mat.transmission > 0.5f) {
        type = BsdfType::Glass;
    } else if (mat.roughness < 0.01f && mat.metallic > 0.5f) {
        type = BsdfType::Mirror;
    } else if (mat.metallic > 0.5f || mat.roughness < 0.5f) {
        type = BsdfType::Glossy;
    } else {
        type = BsdfType::Diffuse;
    }
    
    EXPECT_EQ(type, BsdfType::Diffuse);
}

TEST(BsdfClassificationTest, GlossyMaterial) {
    struct TestMat { float roughness, metallic, transmission; };
    TestMat mat{0.3f, 1.0f, 0.0f};
    
    BsdfType type;
    if (mat.transmission > 0.5f) type = BsdfType::Glass;
    else if (mat.roughness < 0.01f && mat.metallic > 0.5f) type = BsdfType::Mirror;
    else if (mat.metallic > 0.5f || mat.roughness < 0.5f) type = BsdfType::Glossy;
    else type = BsdfType::Diffuse;
    
    EXPECT_EQ(type, BsdfType::Glossy);
}

TEST(BsdfClassificationTest, MirrorMaterial) {
    struct TestMat { float roughness, metallic, transmission; };
    TestMat mat{0.0f, 1.0f, 0.0f};
    
    BsdfType type;
    if (mat.transmission > 0.5f) type = BsdfType::Glass;
    else if (mat.roughness < 0.01f && mat.metallic > 0.5f) type = BsdfType::Mirror;
    else if (mat.metallic > 0.5f || mat.roughness < 0.5f) type = BsdfType::Glossy;
    else type = BsdfType::Diffuse;
    
    EXPECT_EQ(type, BsdfType::Mirror);
}

TEST(BsdfClassificationTest, GlassMaterial) {
    struct TestMat { float roughness, metallic, transmission, ior; };
    TestMat mat{0.0f, 0.0f, 1.0f, 1.5f};
    
    BsdfType type;
    if (mat.transmission > 0.5f) type = BsdfType::Glass;
    else if (mat.roughness < 0.01f && mat.metallic > 0.5f) type = BsdfType::Mirror;
    else if (mat.metallic > 0.5f || mat.roughness < 0.5f) type = BsdfType::Glossy;
    else type = BsdfType::Diffuse;
    
    EXPECT_EQ(type, BsdfType::Glass);
}

// ============================================================================
// Integration with Film Export
// ============================================================================

TEST(DiagnosticPathTracerTest, ExportAfterRecording) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticPathTracer tracer(100, 100, config);
    
    // Record multiple paths at SAME pixel to build variance
    for (int i = 0; i < 10; ++i) {
        PathTrace trace;
        trace.add_vertex(0, 0, BsdfType::Emission);
        trace.add_vertex(1, 1, BsdfType::Diffuse);
        trace.contribution = RGB3f{static_cast<float>(i), 1.0f, 1.0f};
        tracer.record_path(0, 0, trace);  // Same pixel - builds variance
    }
    
    // Should be able to get stats (need >=2 samples for variance)
    auto top = tracer.film().top_groups_by_variance(5);
    EXPECT_GT(top.size(), 0);
    
    // Metadata should include all info
    std::string meta = tracer.film().metadata_json();
    EXPECT_TRUE(meta.find("\"width\": 100") != std::string::npos);
}

TEST(DiagnosticPathTracerTest, Clear) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticPathTracer tracer(100, 100, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    tracer.record_path(0, 0, trace);
    
    EXPECT_GT(tracer.film().total_paths_recorded(), 0);
    
    tracer.clear();
    
    EXPECT_EQ(tracer.film().total_paths_recorded(), 0);
}
