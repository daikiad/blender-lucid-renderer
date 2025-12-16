/**
 * test_diagnostic_film.cpp - TDD tests for diagnostic film (full image)
 * 
 * Tests film-level management and export functionality.
 */

#include <gtest/gtest.h>
#include <sstream>
#include "diagnostics/diagnostic_film.hpp"

using namespace render;
using namespace render::diagnostics;

// ============================================================================
// DiagnosticFilm Construction Tests
// ============================================================================

TEST(DiagnosticFilmTest, Construction) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 50, config);
    
    EXPECT_EQ(film.width(), 100);
    EXPECT_EQ(film.height(), 50);
    EXPECT_EQ(film.sampled_width(), 25);   // 100 / 4
    EXPECT_EQ(film.sampled_height(), 12);  // 50 / 4 = 12
}

TEST(DiagnosticFilmTest, StandardPreset) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(1920, 1080, config);
    
    EXPECT_EQ(film.sampled_width(), 960);   // 1920 / 2
    EXPECT_EQ(film.sampled_height(), 540);  // 1080 / 2
}

// ============================================================================
// Recording Tests
// ============================================================================

TEST(DiagnosticFilmTest, RecordPath) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(100, 100, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Emission);
    trace.add_vertex(1, 1, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 0.5f, 0.25f};
    
    // Record at pixel (10, 10)
    film.record_path(10, 10, trace);
    
    EXPECT_EQ(film.total_paths_recorded(), 1);
}

TEST(DiagnosticFilmTest, SubsampledCoordinates) {
    // Subsample factor 4
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    
    // Pixel (4,4) maps to sampled (1,1)
    // Pixel (5,5) also maps to sampled (1,1)
    film.record_path(4, 4, trace);
    film.record_path(5, 5, trace);
    
    // Should be in same sampled pixel
    EXPECT_EQ(film.total_paths_recorded(), 2);
    
    const auto* data = film.pixel_data(4, 4);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->total_samples(), 2);
}

TEST(DiagnosticFilmTest, OutOfBoundsIgnored) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(100, 100, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    
    // Out of bounds
    film.record_path(200, 200, trace);
    
    EXPECT_EQ(film.total_paths_recorded(), 0);
}

// ============================================================================
// Statistics Queries
// ============================================================================

TEST(DiagnosticFilmTest, GlobalStats) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(100, 100, config);
    
    // Record paths at different pixels
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            PathTrace trace;
            trace.add_vertex(0, 0, BsdfType::Diffuse);
            trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
            film.record_path(x * 2, y * 2, trace);
        }
    }
    
    auto stats = film.global_stats();
    EXPECT_GT(stats.total_samples, 0);
    EXPECT_GT(stats.active_pixels, 0);
}

TEST(DiagnosticFilmTest, TopGroupsByVariance) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(100, 100, config);
    
    // High variance path
    for (int i = 0; i < 10; ++i) {
        PathTrace trace;
        trace.add_vertex(0, 0, BsdfType::Glossy);
        trace.contribution = RGB3f{static_cast<float>(i * 10), 0.0f, 0.0f};
        film.record_path(0, 0, trace);
    }
    
    // Low variance path
    for (int i = 0; i < 10; ++i) {
        PathTrace trace;
        trace.add_vertex(1, 1, BsdfType::Diffuse);
        trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
        film.record_path(0, 0, trace);
    }
    
    auto top = film.top_groups_by_variance(5);
    EXPECT_GT(top.size(), 0);
    
    // First should be high variance group
    if (!top.empty()) {
        EXPECT_EQ(top[0].group->vertices[0].bsdf_type, BsdfType::Glossy);
    }
}

// ============================================================================
// Export Tests
// ============================================================================

TEST(DiagnosticFilmTest, MetadataJson) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(1920, 1080, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    film.record_path(100, 100, trace);
    
    std::string json = film.metadata_json();
    
    // Should contain key fields
    EXPECT_TRUE(json.find("\"width\"") != std::string::npos);
    EXPECT_TRUE(json.find("\"height\"") != std::string::npos);
    EXPECT_TRUE(json.find("\"subsample_factor\"") != std::string::npos);
    EXPECT_TRUE(json.find("1920") != std::string::npos);
}

TEST(DiagnosticFilmTest, ExportBinary) {
    PathRecordingConfig config = PathRecordingConfig::minimal();
    DiagnosticFilm film(100, 100, config);
    
    // Add some data
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 0.5f, 0.25f};
    film.record_path(0, 0, trace);
    
    std::ostringstream oss(std::ios::binary);
    film.export_binary(oss);
    
    std::string data = oss.str();
    EXPECT_GT(data.size(), 0);
    
    // Check magic header
    EXPECT_EQ(data.substr(0, 4), "PVAR");
}

// ============================================================================
// Clear/Reset Tests
// ============================================================================

TEST(DiagnosticFilmTest, Clear) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(100, 100, config);
    
    PathTrace trace;
    trace.add_vertex(0, 0, BsdfType::Diffuse);
    trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
    film.record_path(0, 0, trace);
    
    EXPECT_GT(film.total_paths_recorded(), 0);
    
    film.clear();
    
    EXPECT_EQ(film.total_paths_recorded(), 0);
}

// ============================================================================
// Thread Safety Tests (basic)
// ============================================================================

TEST(DiagnosticFilmTest, DifferentPixelsParallel) {
    PathRecordingConfig config = PathRecordingConfig::standard();
    DiagnosticFilm film(1000, 1000, config);
    
    // This test just ensures no crash - actual thread safety needs OpenMP
    #pragma omp parallel for if(false)  // Disabled for unit test
    for (int i = 0; i < 100; ++i) {
        PathTrace trace;
        trace.add_vertex(i, i, BsdfType::Diffuse);
        trace.contribution = RGB3f{1.0f, 1.0f, 1.0f};
        film.record_path(i * 10, i * 10, trace);
    }
    
    EXPECT_EQ(film.total_paths_recorded(), 100);
}
