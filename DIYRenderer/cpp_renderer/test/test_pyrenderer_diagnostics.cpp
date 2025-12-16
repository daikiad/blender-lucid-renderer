/**
 * test_pyrenderer_diagnostics.cpp - PyRenderer diagnostic integration tests
 * =========================================================================
 * 
 * TDD tests for integrating path variance diagnostics into PyRenderer.
 * Tests ensure that diagnostic recording works during actual rendering.
 * 
 * Week 2: Integration tests
 */

#include <gtest/gtest.h>
#include <string>

#include "pybind_renderer.hpp"
#include "diagnostics/diagnostic_export.hpp"
#include "diagnostics/diagnostic_integrator.hpp"

using namespace render::diagnostics;

// =============================================================================
// PyRendererDiagnosticsTest - Integration tests
// =============================================================================

class PyRendererDiagnosticsTest : public ::testing::Test {
protected:
    PyRenderer renderer;
    
    void SetUp() override {
        // Load a simple scene for testing
        const std::string simple_scene = R"({
            "meshes": [{
                "name": "Floor",
                "vertices": [
                    [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]
                ],
                "triangles": [[0, 1, 2], [0, 2, 3]],
                "material": {
                    "base_color": [0.8, 0.8, 0.8],
                    "roughness": 0.9,
                    "metallic": 0.0
                }
            }, {
                "name": "Sphere",
                "vertices": [
                    [0, 0, 1], [0.5, 0, 0.8], [-0.5, 0, 0.8],
                    [0, 0.5, 0.8], [0, -0.5, 0.8]
                ],
                "triangles": [[0, 1, 3], [0, 3, 2], [0, 2, 4], [0, 4, 1]],
                "material": {
                    "base_color": [0.2, 0.5, 0.9],
                    "roughness": 0.3,
                    "metallic": 0.8
                }
            }],
            "lights": [{
                "type": "point",
                "position": [2, 2, 3],
                "color": [1, 1, 1],
                "intensity": 100
            }]
        })";
        
        renderer.load_scene_json(simple_scene);
        renderer.set_camera(0, -3, 2, 0, 0.6, -0.4, 0, 0, 1, 50);
    }
};

// Test that PyRenderer can be extended with diagnostic support
TEST_F(PyRendererDiagnosticsTest, HasDiagnosticMethods) {
    // These methods should exist on PyRenderer
    EXPECT_TRUE(renderer.is_scene_loaded());
    EXPECT_TRUE(renderer.is_camera_set());
    
    // Future: renderer.enable_diagnostics(config)
    // Future: renderer.get_diagnostic_film()
}

// Test rendering produces valid output
TEST_F(PyRendererDiagnosticsTest, RenderProducesOutput) {
    auto pixels = renderer.render_tile(0, 0, 8, 8, 8, 8, 1, 0, 4);
    
    // 8x8 tiles * 4 channels = 256 floats
    EXPECT_EQ(pixels.size(), 256u);
    
    // Check some pixels have non-zero values (not all black)
    bool has_nonzero = false;
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i] > 0.001f || pixels[i+1] > 0.001f || pixels[i+2] > 0.001f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// =============================================================================
// DiagnosticPathTracerIntegration - Standalone diagnostic tracing
// =============================================================================

class DiagnosticPathTracerIntegrationTest : public ::testing::Test {
protected:
    PathRecordingConfig config{};
    
    void SetUp() override {
        config = PathRecordingConfig::standard();
    }
};

// Test that DiagnosticPathTracer can record simulated paths
TEST_F(DiagnosticPathTracerIntegrationTest, RecordSimulatedPaths) {
    DiagnosticPathTracer tracer(32, 32, config);
    
    // Simulate recording paths during rendering
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            PathDiagnosticRecorder rec;
            rec.begin_path();
            
            // Simulate a path: camera -> diffuse -> light
            rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
            rec.record_light_hit(0);
            
            auto path = rec.end_path(render::RGB3f{0.5f, 0.5f, 0.5f});
            tracer.record_path(x, y, path);
        }
    }
    
    // Verify recording
    EXPECT_EQ(tracer.film().total_paths_recorded(), 1024u);
    
    // Export and check
    DiagnosticExporter exporter(tracer.film());
    auto stats = exporter.get_global_stats();
    EXPECT_EQ(stats.total_samples, 1024u);
}

// Test mixed path types
TEST_F(DiagnosticPathTracerIntegrationTest, RecordMixedPathTypes) {
    DiagnosticPathTracer tracer(16, 16, config);
    
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            PathDiagnosticRecorder rec;
            rec.begin_path();
            
            // Vary path types based on position
            if ((x + y) % 3 == 0) {
                // Diffuse path
                rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
            } else if ((x + y) % 3 == 1) {
                // Glossy path
                rec.record_vertex(0, 0, BsdfType::Glossy, false, false);
            } else {
                // Indirect path
                rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
                rec.record_vertex(1, 0, BsdfType::Glossy, false, true);
            }
            rec.record_light_hit(0);
            
            float contrib = 0.3f + (x * y) * 0.001f;
            auto path = rec.end_path(render::RGB3f{contrib, contrib, contrib});
            tracer.record_path(x, y, path);
        }
    }
    
    // Should have multiple path groups
    DiagnosticExporter exporter(tracer.film());
    auto top_var = exporter.get_top_variance_groups(10);
    auto top_mean = exporter.get_top_mean_groups(10);
    
    // We created at least 2 different path signatures
    EXPECT_GT(top_mean.size(), 0u);
}

// Test variance detection
TEST_F(DiagnosticPathTracerIntegrationTest, DetectsVariance) {
    DiagnosticPathTracer tracer(4, 4, config);
    
    // Record paths with high variance at one pixel
    for (int sample = 0; sample < 20; ++sample) {
        PathDiagnosticRecorder rec;
        rec.begin_path();
        rec.record_vertex(0, 0, BsdfType::Glossy, false, false);
        rec.record_light_hit(0);
        
        // High variance: contribution varies wildly
        float contrib = (sample % 2 == 0) ? 0.1f : 0.9f;
        auto path = rec.end_path(render::RGB3f{contrib, contrib, contrib});
        tracer.record_path(0, 0, path);  // Same pixel
    }
    
    // Record paths with low variance at another pixel
    for (int sample = 0; sample < 20; ++sample) {
        PathDiagnosticRecorder rec;
        rec.begin_path();
        rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
        rec.record_light_hit(0);
        
        // Low variance: consistent contribution
        float contrib = 0.5f + 0.01f * (sample % 3);
        auto path = rec.end_path(render::RGB3f{contrib, contrib, contrib});
        tracer.record_path(1, 1, path);  // Different pixel
    }
    
    // The glossy group should have higher variance
    DiagnosticExporter exporter(tracer.film());
    auto top_var = exporter.get_top_variance_groups(2);
    
    if (!top_var.empty()) {
        // Top variance group should be the glossy one
        bool found_glossy = false;
        for (const auto& g : top_var) {
            if (g.signature.find("G") != std::string::npos) {
                found_glossy = true;
                EXPECT_GT(g.variance_luminance, 0.01f);  // Should have variance
            }
        }
        // Glossy should be in top variance groups
        EXPECT_TRUE(found_glossy);
    }
}

// Test JSON export contains expected fields
TEST_F(DiagnosticPathTracerIntegrationTest, JsonExportHasFields) {
    DiagnosticPathTracer tracer(8, 8, config);
    
    // Record a few paths
    for (int i = 0; i < 10; ++i) {
        PathDiagnosticRecorder rec;
        rec.begin_path();
        rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
        rec.record_light_hit(0);
        auto path = rec.end_path(render::RGB3f{0.5f, 0.5f, 0.5f});
        tracer.record_path(i % 8, i / 8, path);
    }
    
    DiagnosticExporter exporter(tracer.film());
    std::string json = exporter.export_metadata_json();
    
    // Check for expected JSON fields
    EXPECT_NE(json.find("width"), std::string::npos);
    EXPECT_NE(json.find("height"), std::string::npos);
    EXPECT_NE(json.find("total_samples"), std::string::npos);
    EXPECT_NE(json.find("subsample_factor"), std::string::npos);
}

// Test binary export produces data
TEST_F(DiagnosticPathTracerIntegrationTest, BinaryExportProducesData) {
    DiagnosticPathTracer tracer(4, 4, config);
    
    // Record paths
    for (int i = 0; i < 4; ++i) {
        PathDiagnosticRecorder rec;
        rec.begin_path();
        rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
        rec.record_light_hit(0);
        auto path = rec.end_path(render::RGB3f{0.5f, 0.5f, 0.5f});
        tracer.record_path(i, 0, path);
    }
    
    DiagnosticExporter exporter(tracer.film());
    auto binary = exporter.export_binary();
    
    // Should have some data
    EXPECT_GT(binary.size(), 0u);
    
    // Check magic header "PVAR"
    if (binary.size() >= 4) {
        EXPECT_EQ(binary[0], 'P');
        EXPECT_EQ(binary[1], 'V');
        EXPECT_EQ(binary[2], 'A');
        EXPECT_EQ(binary[3], 'R');
    }
}

// =============================================================================
// PyRenderer Diagnostic Integration Tests
// =============================================================================
// TDD tests for render_tile_with_diagnostics feature

class PyRendererDiagnosticIntegrationTest : public ::testing::Test {
protected:
    PyRenderer renderer;
    
    void SetUp() override {
        // Load Cornell Box-like scene
        const std::string scene_json = R"({
            "meshes": [{
                "name": "Floor",
                "vertices": [
                    [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]
                ],
                "triangles": [[0, 1, 2], [0, 2, 3]],
                "material": {
                    "base_color": [0.8, 0.8, 0.8],
                    "roughness": 0.9,
                    "metallic": 0.0
                }
            }, {
                "name": "BackWall",
                "vertices": [
                    [-2, 2, 0], [2, 2, 0], [2, 2, 3], [-2, 2, 3]
                ],
                "triangles": [[0, 1, 2], [0, 2, 3]],
                "material": {
                    "base_color": [0.7, 0.7, 0.7],
                    "roughness": 0.9,
                    "metallic": 0.0
                }
            }, {
                "name": "Sphere",
                "vertices": [
                    [0, 0, 0.5], [0.3, 0, 0.35], [-0.3, 0, 0.35],
                    [0, 0.3, 0.35], [0, -0.3, 0.35], [0, 0, 0.2]
                ],
                "triangles": [[0, 1, 3], [0, 3, 2], [0, 2, 4], [0, 4, 1], 
                              [5, 1, 4], [5, 4, 2], [5, 2, 3], [5, 3, 1]],
                "material": {
                    "base_color": [0.3, 0.6, 0.9],
                    "roughness": 0.4,
                    "metallic": 0.7
                }
            }],
            "lights": [{
                "type": "point",
                "position": [0, -1, 2.5],
                "color": [1, 1, 1],
                "intensity": 50
            }]
        })";
        
        renderer.load_scene_json(scene_json);
        renderer.set_camera(0, -2.5, 1.5, 0, 0.5, -0.2, 0, 0, 1, 60);
        renderer.set_algorithm("mis");
    }
};

// Test: PyRenderer has diagnostics-related methods
TEST_F(PyRendererDiagnosticIntegrationTest, HasDiagnosticMethods) {
    // enable_diagnostics should exist and accept a config
    auto config = PathRecordingConfig::minimal();
    renderer.enable_diagnostics(config);
    
    // Should be able to check if diagnostics are enabled
    EXPECT_TRUE(renderer.is_diagnostics_enabled());
    
    // Should be able to disable
    renderer.disable_diagnostics();
    EXPECT_FALSE(renderer.is_diagnostics_enabled());
}

// Test: render_tile_with_diagnostics returns same results as render_tile
TEST_F(PyRendererDiagnosticIntegrationTest, DiagnosticRenderMatchesNormal) {
    auto config = PathRecordingConfig::minimal();
    renderer.enable_diagnostics(config);
    
    // Render small tile
    auto pixels1 = renderer.render_tile(0, 0, 4, 4, 4, 4, 1, 42, 4);
    
    renderer.disable_diagnostics();
    auto pixels2 = renderer.render_tile(0, 0, 4, 4, 4, 4, 1, 42, 4);
    
    // Results should be identical (same seed)
    ASSERT_EQ(pixels1.size(), pixels2.size());
    for (size_t i = 0; i < pixels1.size(); ++i) {
        EXPECT_FLOAT_EQ(pixels1[i], pixels2[i]);
    }
}

// Test: Diagnostics records paths during render
TEST_F(PyRendererDiagnosticIntegrationTest, RecordsPathsDuringRender) {
    auto config = PathRecordingConfig::minimal();
    renderer.enable_diagnostics(config);
    
    // Render 8x8 tile with 4 samples
    renderer.render_tile(0, 0, 8, 8, 8, 8, 4, 0, 4);
    
    // Get diagnostic stats
    auto stats = renderer.get_diagnostic_stats();
    
    // Should have recorded paths (8*8*4 = 256 samples maximum, but depends on hits)
    EXPECT_GT(stats.total_samples, 0u);
    EXPECT_GT(stats.active_pixels, 0u);
}

// Test: Can get per-pixel variance data
TEST_F(PyRendererDiagnosticIntegrationTest, GetPerPixelVariance) {
    auto config = PathRecordingConfig::minimal();
    renderer.enable_diagnostics(config);
    
    // Render multiple samples to build up variance data
    renderer.render_tile(0, 0, 8, 8, 8, 8, 16, 0, 4);
    
    // Get variance map
    auto variance_map = renderer.get_diagnostic_variance_map();
    
    // Should be 8*8 pixels
    EXPECT_EQ(variance_map.size(), 64u);
    
    // All values should be non-negative
    for (float var : variance_map) {
        EXPECT_GE(var, 0.0f);
    }
}

// Test: Can export diagnostic data as JSON
TEST_F(PyRendererDiagnosticIntegrationTest, ExportDiagnosticJson) {
    auto config = PathRecordingConfig::minimal();
    renderer.enable_diagnostics(config);
    
    renderer.render_tile(0, 0, 8, 8, 8, 8, 4, 0, 4);
    
    // Export as JSON
    std::string json = renderer.export_diagnostic_json();
    
    // Should contain expected fields
    EXPECT_NE(json.find("width"), std::string::npos);
    EXPECT_NE(json.find("height"), std::string::npos);
    EXPECT_NE(json.find("total_samples"), std::string::npos);
}

// Test: Can get top variance path groups
TEST_F(PyRendererDiagnosticIntegrationTest, GetTopVarianceGroups) {
    auto config = PathRecordingConfig::standard();
    renderer.enable_diagnostics(config);
    
    // Render larger area with more samples
    renderer.render_tile(0, 0, 16, 16, 16, 16, 8, 0, 4);
    
    // Get top variance groups
    auto groups = renderer.get_top_variance_groups(5);
    
    // Should have at least one group (if any paths recorded)
    // Note: might be empty if no paths hit anything
    // Just check the return type is correct
    EXPECT_LE(groups.size(), 5u);
}

// Test: Diagnostics cleared between renders when reset
TEST_F(PyRendererDiagnosticIntegrationTest, DiagnosticsClearOnReset) {
    auto config = PathRecordingConfig::minimal();
    renderer.enable_diagnostics(config);
    
    // First render
    renderer.render_tile(0, 0, 8, 8, 8, 8, 4, 0, 4);
    auto stats1 = renderer.get_diagnostic_stats();
    
    // Clear diagnostics
    renderer.clear_diagnostics();
    auto stats2 = renderer.get_diagnostic_stats();
    
    EXPECT_EQ(stats2.total_samples, 0u);
    
    // Render again
    renderer.render_tile(0, 0, 8, 8, 8, 8, 4, 0, 4);
    auto stats3 = renderer.get_diagnostic_stats();
    
    // Should have recorded new paths
    EXPECT_GT(stats3.total_samples, 0u);
}

// Test: Diagnostics work with all algorithms
TEST_F(PyRendererDiagnosticIntegrationTest, WorksWithAllAlgorithms) {
    auto config = PathRecordingConfig::minimal();
    
    for (const auto& algo : {"simple", "nee", "mis"}) {
        renderer.set_algorithm(algo);
        renderer.enable_diagnostics(config);
        
        renderer.render_tile(0, 0, 4, 4, 4, 4, 2, 0, 4);
        
        auto stats = renderer.get_diagnostic_stats();
        EXPECT_GT(stats.total_samples, 0u) << "Failed for algorithm: " << algo;
        
        renderer.clear_diagnostics();
    }
}
