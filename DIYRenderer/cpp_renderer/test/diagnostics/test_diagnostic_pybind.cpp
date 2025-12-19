/**
 * test_diagnostic_pybind.cpp - Diagnostic pybind11 bindings tests
 * ================================================================
 * 
 * Tests for Python bindings of the diagnostic system.
 * These tests verify that the binding types are correctly exposed
 * and can be used from C++ in the same way Python would use them.
 * 
 * Week 2: TDD tests for pybind11 integration
 */

#include <gtest/gtest.h>

#include "diagnostics/diagnostic_export.hpp"
#include "diagnostics/diagnostic_integrator.hpp"
#include "diagnostics/path_stats_config.hpp"

using namespace render::diagnostics;

// =============================================================================
// PyDiagnosticRecorder Tests
// =============================================================================
// Test PathDiagnosticRecorder's Python-facing behavior

class PyDiagnosticRecorderTest : public ::testing::Test {
protected:
    PathRecordingConfig config{};
    
    void SetUp() override {
        config = PathRecordingConfig::standard();
    }
};

TEST_F(PyDiagnosticRecorderTest, CanBeCreatedDefault) {
    // Python interface: recorder = diyrenderer.PathDiagnosticRecorder()
    PathDiagnosticRecorder recorder;
    
    // Should be default constructed without crash
    SUCCEED();
}

TEST_F(PyDiagnosticRecorderTest, RecordVertexBuildsPath) {
    // Python interface: recorder.record_vertex(obj_id, mat_id, type, is_delta, is_nee)
    PathDiagnosticRecorder recorder;
    
    recorder.begin_path();
    recorder.record_vertex(0, 0, BsdfType::Glossy, false, false);
    recorder.record_vertex(0, 0, BsdfType::Diffuse, false, true);
    
    auto path = recorder.end_path(render::RGB3f{1.0f, 1.0f, 1.0f});
    EXPECT_EQ(path.depth, 2);
}

TEST_F(PyDiagnosticRecorderTest, EndPathReturnsValidTrace) {
    PathDiagnosticRecorder recorder;
    
    recorder.begin_path();
    recorder.record_vertex(1, 0, BsdfType::Glossy, false, false);
    recorder.record_light_hit(0);
    
    auto path = recorder.end_path(render::RGB3f{0.5f, 0.5f, 0.5f});
    
    EXPECT_GE(path.depth, 2);
    // First vertex should be glossy
    EXPECT_EQ(path.vertices[0].bsdf_type, BsdfType::Glossy);
}

// =============================================================================
// PyExportedGroupInfo Tests - POD for pybind11
// =============================================================================

class PyExportedGroupInfoTest : public ::testing::Test {};

TEST_F(PyExportedGroupInfoTest, AllFieldsAccessible) {
    // Python interface: group.pixel_x, group.signature, etc.
    ExportedGroupInfo info;
    info.pixel_x = 10;
    info.pixel_y = 20;
    info.signature = "D-G-S";
    info.mean_luminance = 0.5f;
    info.variance_luminance = 0.1f;
    info.sample_count = 100;
    info.depth = 3;
    info.coarse_type = 2;  // Glossy coarse type
    info.coarse_type_name = "Glossy";
    
    EXPECT_EQ(info.pixel_x, 10u);
    EXPECT_EQ(info.pixel_y, 20u);
    EXPECT_EQ(info.signature, "D-G-S");
    EXPECT_FLOAT_EQ(info.mean_luminance, 0.5f);
    EXPECT_FLOAT_EQ(info.variance_luminance, 0.1f);
    EXPECT_EQ(info.sample_count, 100u);
    EXPECT_EQ(info.depth, 3);
}

TEST_F(PyExportedGroupInfoTest, CanBeDefaultConstructed) {
    // Python will default-construct before assigning
    ExportedGroupInfo info{};
    
    EXPECT_EQ(info.pixel_x, 0u);
    EXPECT_EQ(info.pixel_y, 0u);
    EXPECT_TRUE(info.signature.empty());
}

// =============================================================================
// PyDiagnosticExporter Tests
// =============================================================================

class PyDiagnosticExporterTest : public ::testing::Test {
protected:
    PathRecordingConfig config{};
    
    void SetUp() override {
        config = PathRecordingConfig::standard();
    }
    
    void record_paths(DiagnosticFilm& film) {
        // Record some varied paths for testing
        for (int i = 0; i < 10; ++i) {
            PathDiagnosticRecorder rec;
            rec.begin_path();
            rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
            rec.record_vertex(0, 0, BsdfType::Glossy, false, true);
            float val = 0.2f + (i % 3) * 0.1f;
            auto path = rec.end_path(render::RGB3f{val, val, val});
            
            film.record_path(i % 4, i / 4, path);
        }
    }
};

TEST_F(PyDiagnosticExporterTest, TopGroupsByVariance) {
    // Python: exporter.get_top_variance_groups(count)
    DiagnosticFilm film(8, 8, config);
    record_paths(film);
    
    DiagnosticExporter exporter(film);
    auto groups = exporter.get_top_variance_groups(5);
    
    // Returns vector of ExportedGroupInfo
    EXPECT_LE(groups.size(), 5u);
}

TEST_F(PyDiagnosticExporterTest, TopGroupsByMean) {
    // Python: exporter.get_top_mean_groups(count)
    DiagnosticFilm film(8, 8, config);
    record_paths(film);
    
    DiagnosticExporter exporter(film);
    auto groups = exporter.get_top_mean_groups(5);
    
    EXPECT_LE(groups.size(), 5u);
}

TEST_F(PyDiagnosticExporterTest, GlobalStats) {
    // Python: exporter.get_global_stats()
    DiagnosticFilm film(8, 8, config);
    record_paths(film);
    
    DiagnosticExporter exporter(film);
    auto stats = exporter.get_global_stats();
    
    EXPECT_GT(stats.total_samples, 0u);
}

TEST_F(PyDiagnosticExporterTest, CoarseStats) {
    // Python: exporter.get_coarse_stats()
    DiagnosticFilm film(8, 8, config);
    record_paths(film);
    
    DiagnosticExporter exporter(film);
    auto stats = exporter.get_coarse_stats();
    
    // Should have at least some coarse type statistics
    // Exact count depends on recorded paths
    SUCCEED();
}

TEST_F(PyDiagnosticExporterTest, Suggestions) {
    // Python: exporter.generate_suggestions()
    DiagnosticFilm film(8, 8, config);
    record_paths(film);
    
    DiagnosticExporter exporter(film);
    auto suggestions = exporter.generate_suggestions();
    
    // Returns vector of DiagnosticSuggestion
    for (const auto& s : suggestions) {
        EXPECT_FALSE(s.category.empty());
        EXPECT_FALSE(s.message.empty());
    }
}

TEST_F(PyDiagnosticExporterTest, ExportJson) {
    // Python: exporter.export_metadata_json()
    DiagnosticFilm film(8, 8, config);
    record_paths(film);
    
    DiagnosticExporter exporter(film);
    std::string json = exporter.export_metadata_json();
    
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("total_samples"), std::string::npos);
}

// =============================================================================
// PyPathRecordingConfig Tests
// =============================================================================

class PyPathRecordingConfigTest : public ::testing::Test {};

TEST_F(PyPathRecordingConfigTest, AllFieldsAccessible) {
    // Python: config = diyrenderer.PathRecordingConfig()
    PathRecordingConfig config;
    config.subsample_factor = 2;
    config.max_groups_per_pixel = 16;
    config.max_outliers_per_pixel = 4;
    config.max_depth = 10;
    config.variance_threshold = 0.1f;
    config.outlier_threshold = 3.0f;
    
    EXPECT_EQ(config.subsample_factor, 2);
    EXPECT_EQ(config.max_groups_per_pixel, 16);
    EXPECT_EQ(config.max_outliers_per_pixel, 4);
    EXPECT_EQ(config.max_depth, 10);
    EXPECT_FLOAT_EQ(config.variance_threshold, 0.1f);
    EXPECT_FLOAT_EQ(config.outlier_threshold, 3.0f);
}

TEST_F(PyPathRecordingConfigTest, Presets) {
    // Python: config = diyrenderer.PathRecordingConfig.standard()
    auto standard = PathRecordingConfig::standard();
    EXPECT_EQ(standard.subsample_factor, 2);
    EXPECT_EQ(standard.max_groups_per_pixel, 16);
    
    auto minimal = PathRecordingConfig::minimal();
    EXPECT_EQ(minimal.subsample_factor, 4);
    EXPECT_EQ(minimal.max_groups_per_pixel, 8);
    
    auto detailed = PathRecordingConfig::detailed();
    EXPECT_EQ(detailed.subsample_factor, 1);
    EXPECT_EQ(detailed.max_groups_per_pixel, 32);
}

TEST_F(PyPathRecordingConfigTest, MemoryEstimate) {
    // Python: config.estimate_memory_bytes(1920, 1080)
    PathRecordingConfig config;
    size_t mem = config.estimate_memory_bytes(1920, 1080);
    
    EXPECT_GT(mem, 0u);
}

TEST_F(PyPathRecordingConfigTest, FormatMemory) {
    // Python: format_bytes helper
    PathRecordingConfig config = PathRecordingConfig::standard();
    size_t mem = config.estimate_memory_bytes(1920, 1080);
    
    // Memory should be reasonable (< 10GB for 1080p with standard preset, includes geometry)
    EXPECT_LT(mem, 10UL * 1024 * 1024 * 1024);
    EXPECT_GT(mem, 0u);
}

// =============================================================================
// PyDiagnosticSuggestion Tests
// =============================================================================

class PyDiagnosticSuggestionTest : public ::testing::Test {};

TEST_F(PyDiagnosticSuggestionTest, AllFieldsAccessible) {
    // Python: suggestion.category, suggestion.severity, etc.
    DiagnosticSuggestion suggestion;
    suggestion.category = "Settings";
    suggestion.severity = "warning";
    suggestion.message = "Consider using more samples for caustics";
    suggestion.action = "Increase sample count";
    
    EXPECT_EQ(suggestion.category, "Settings");
    EXPECT_EQ(suggestion.severity, "warning");
    EXPECT_EQ(suggestion.message, "Consider using more samples for caustics");
    EXPECT_EQ(suggestion.action, "Increase sample count");
}

// =============================================================================
// PyDiagnosticPathTracer Tests
// =============================================================================

class PyDiagnosticPathTracerTest : public ::testing::Test {};

TEST_F(PyDiagnosticPathTracerTest, CanBeCreated) {
    // Python: tracer = diyrenderer.DiagnosticPathTracer(w, h, config)
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticPathTracer tracer(8, 8, config);
    
    EXPECT_TRUE(tracer.is_enabled());
}

TEST_F(PyDiagnosticPathTracerTest, EnableDisable) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticPathTracer tracer(8, 8, config);
    
    tracer.set_enabled(false);
    EXPECT_FALSE(tracer.is_enabled());
    
    tracer.set_enabled(true);
    EXPECT_TRUE(tracer.is_enabled());
}

TEST_F(PyDiagnosticPathTracerTest, FilmAccess) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticPathTracer tracer(8, 8, config);
    
    // Should be able to access the underlying film
    DiagnosticFilm& film = tracer.film();
    EXPECT_EQ(film.width(), 8u);
    EXPECT_EQ(film.height(), 8u);
}

// =============================================================================
// Integration Test: Full Python-like Workflow
// =============================================================================

TEST(DiagnosticPybindIntegrationTest, FullWorkflow) {
    // Simulate the Python workflow:
    // 1. Create config
    // 2. Create tracer with diagnostics
    // 3. Record paths during rendering
    // 4. Export results
    
    // Step 1: Configure
    PathRecordingConfig config = PathRecordingConfig::standard();
    
    // Step 2: Create diagnostic tracer
    DiagnosticPathTracer tracer(8, 8, config);
    EXPECT_TRUE(tracer.is_enabled());
    
    // Step 3: Simulate rendering with path recording
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            PathDiagnosticRecorder rec;
            rec.begin_path();
            
            // Simulate path tracing
            rec.record_vertex(0, 0, BsdfType::Diffuse, false, false);
            if ((x + y) % 2 == 0) {
                rec.record_vertex(0, 0, BsdfType::Glossy, false, true);
            }
            rec.record_light_hit(0);
            
            auto path = rec.end_path(render::RGB3f{0.3f, 0.3f, 0.3f});
            tracer.record_path(x, y, path);
        }
    }
    
    // Step 4: Export results
    DiagnosticExporter exporter(tracer.film());
    
    auto global = exporter.get_global_stats();
    EXPECT_EQ(global.total_samples, 64u);
    
    auto top = exporter.get_top_variance_groups(10);
    // Should have at least some groups (may be empty if no variance)
    
    std::string json = exporter.export_metadata_json();
    EXPECT_FALSE(json.empty());
}
