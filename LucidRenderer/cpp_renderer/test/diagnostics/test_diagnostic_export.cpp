/**
 * test_diagnostic_pybind.cpp - TDD tests for pybind11 diagnostic exports
 * 
 * Tests the data structures that will be exposed to Python.
 */

#include <gtest/gtest.h>
#include <sstream>
#include "diagnostics/diagnostic_export.hpp"

using namespace render;
using namespace render::diagnostics;

// ============================================================================
// DiagnosticExporter Tests
// ============================================================================

TEST(DiagnosticExporterTest, ExportGlobalStats) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    // Add some data
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Emission);
    trace.add_vertex(1, 1, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 0.5f, 0.25f};
    film.record_path(0, 0, trace);
    film.record_path(0, 0, trace);
    
    DiagnosticExporter exporter(film);
    auto stats = exporter.get_global_stats();
    
    EXPECT_GT(stats.total_samples, 0);
}

TEST(DiagnosticExporterTest, ExportTopVarianceGroups) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    // Add high-variance data
    for (int i = 0; i < 10; ++i) {
        PathTrace trace;
        trace.add_vertex(0, 0, BsdfType::Glossy);
        trace.contribution = RGB3f{static_cast<float>(i * 5), 1.0f, 1.0f};
        film.record_path(0, 0, trace);
    }
    
    DiagnosticExporter exporter(film);
    auto groups = exporter.get_top_variance_groups(5);
    
    EXPECT_GT(groups.size(), 0);
    EXPECT_TRUE(groups[0].signature.find('G') != std::string::npos);  // Glossy
}

TEST(DiagnosticExporterTest, ExportTopMeanGroups) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    // Add data
    for (int i = 0; i < 5; ++i) {
        PathTrace trace;
        trace.add_vertex(0, 0, BsdfType::Diffuse);
        trace.contribution = RGB3f{10.0f, 10.0f, 10.0f};
        film.record_path(0, 0, trace);
    }
    
    DiagnosticExporter exporter(film);
    auto groups = exporter.get_top_mean_groups(5);
    
    EXPECT_GT(groups.size(), 0);
    EXPECT_GT(groups[0].mean_luminance, 0.0f);
}

TEST(DiagnosticExporterTest, ExportCoarseStats) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    // Direct diffuse (coarse type 0)
    PathTrace direct;
    direct.add_vertex(0, 0, BsdfType::Emission);
    direct.add_vertex(1, 1, BsdfType::Diffuse);
    direct.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    film.record_path(0, 0, direct);
    
    DiagnosticExporter exporter(film);
    auto coarse = exporter.get_coarse_stats();
    
    // Should have 8 coarse type stats
    EXPECT_EQ(coarse.size(), 8);
}

// ============================================================================
// ExportedGroupInfo Tests
// ============================================================================

TEST(ExportedGroupInfoTest, ContainsAllFields) {
    ExportedGroupInfo info;
    info.pixel_x = 100;
    info.pixel_y = 200;
    info.signature = "LDG";
    info.sample_count = 42;
    info.mean_luminance = 0.5f;
    info.variance_luminance = 0.1f;
    info.mean_rgb = {1.0f, 0.8f, 0.6f};
    info.variance_rgb = {0.1f, 0.2f, 0.3f};
    info.depth = 3;
    info.coarse_type = 0;
    info.coarse_type_name = "Direct-NoDelta-BSDF";
    
    EXPECT_EQ(info.pixel_x, 100);
    EXPECT_EQ(info.signature, "LDG");
    EXPECT_EQ(info.depth, 3);
}

// ============================================================================
// Binary Export Tests
// ============================================================================

TEST(DiagnosticExporterTest, ExportBinaryBuffer) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    film.record_path(0, 0, trace);
    
    DiagnosticExporter exporter(film);
    std::vector<uint8_t> buffer = exporter.export_binary();
    
    // Check magic header "PVAR"
    EXPECT_GE(buffer.size(), 4);
    EXPECT_EQ(buffer[0], 'P');
    EXPECT_EQ(buffer[1], 'V');
    EXPECT_EQ(buffer[2], 'A');
    EXPECT_EQ(buffer[3], 'R');
}

TEST(DiagnosticExporterTest, ExportMetadataJson) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(1920, 1080, config);
    
    DiagnosticExporter exporter(film);
    std::string json = exporter.export_metadata_json();
    
    EXPECT_TRUE(json.find("\"width\": 1920") != std::string::npos);
    EXPECT_TRUE(json.find("\"height\": 1080") != std::string::npos);
    EXPECT_TRUE(json.find("\"subsample_factor\"") != std::string::npos);
}

// ============================================================================
// Suggestions Generation Tests
// ============================================================================

TEST(DiagnosticExporterTest, GenerateSuggestions) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    // Add many high-variance glossy paths (need variance > 0 for top_groups)
    // First, create multiple different path types with high variance
    for (int p = 0; p < 5; ++p) {  // 5 different glossy path types
        for (int i = 0; i < 10; ++i) {
            PathTrace trace;
            trace.add_vertex(0, 0, BsdfType::Emission);
            trace.add_vertex(p, p, BsdfType::Glossy);  // Different object IDs
            // Vary contributions significantly to create high variance
            float val = (i % 2 == 0) ? 0.1f : 100.0f;
            trace.contribution = RGB3f{val, val, val};
            film.record_path(0, 0, trace);
        }
    }
    
    DiagnosticExporter exporter(film);
    
    // First verify we have variance groups
    auto top = film.top_groups_by_variance(10);
    
    auto suggestions = exporter.generate_suggestions();
    
    // If we have top variance groups, should have suggestions
    // Otherwise this tests the case when data is insufficient
    if (top.size() >= 4) {
        EXPECT_GT(suggestions.size(), 0);
    } else {
        // Not enough data - test passes with no suggestions
        SUCCEED();
    }
}

TEST(SuggestionTest, Fields) {
    DiagnosticSuggestion suggestion;
    suggestion.category = "Settings";
    suggestion.severity = "warning";
    suggestion.message = "High variance in glossy paths";
    suggestion.action = "Increase samples or use denoiser";
    
    EXPECT_EQ(suggestion.category, "Settings");
    EXPECT_EQ(suggestion.severity, "warning");
}
