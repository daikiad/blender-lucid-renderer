"""
test_diagnostics_python.py - Tests for Python diagnostics API
==============================================================

Tests the high-level Python wrapper for the diagnostic system.
Runs with or without the C++ module available (stub mode tests).
"""

import pytest
import sys
import os

# Add DIYRenderer to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DIYRenderer.diagnostics import (
    DiagnosticsManager,
    DiagnosticReport,
    PathGroupInfo,
    DiagnosticSuggestion,
    GlobalStats,
    get_global_diagnostics,
    set_global_diagnostics,
    create_diagnostics_for_render,
)


class TestDiagnosticsManagerStubMode:
    """Tests that work without C++ module (stub mode)."""
    
    def test_create_manager(self):
        """Can create manager instance."""
        manager = DiagnosticsManager(100, 100)
        assert manager._width == 100
        assert manager._height == 100
    
    def test_factory_minimal(self):
        """Factory method minimal() works."""
        manager = DiagnosticsManager.minimal(200, 150)
        assert manager._width == 200
        assert manager._height == 150
    
    def test_factory_standard(self):
        """Factory method standard() works."""
        manager = DiagnosticsManager.standard(300, 200)
        assert manager._width == 300
    
    def test_factory_detailed(self):
        """Factory method detailed() works."""
        manager = DiagnosticsManager.detailed(400, 300)
        assert manager._width == 400
    
    def test_is_available_without_cpp(self):
        """is_available returns False without C++ module."""
        # Save original module
        saved = sys.modules.get('diyrenderer')
        sys.modules['diyrenderer'] = None  # Block import
        
        try:
            manager = DiagnosticsManager(100, 100, config="dummy")
            # In stub mode, tracer is None
        finally:
            if saved is not None:
                sys.modules['diyrenderer'] = saved
            else:
                sys.modules.pop('diyrenderer', None)
    
    def test_enable_disable_safe_without_tracer(self):
        """enable/disable don't crash without tracer."""
        manager = DiagnosticsManager.__new__(DiagnosticsManager)
        manager._tracer = None
        manager._enabled = False
        
        # Should not raise
        manager.enable()
        manager.disable()
    
    def test_get_film_returns_none_without_tracer(self):
        """get_film returns None without tracer."""
        manager = DiagnosticsManager.__new__(DiagnosticsManager)
        manager._tracer = None
        
        assert manager.get_film() is None
    
    def test_get_report_returns_none_without_tracer(self):
        """get_report returns None without tracer."""
        manager = DiagnosticsManager.__new__(DiagnosticsManager)
        manager._tracer = None
        
        assert manager.get_report() is None


class TestGlobalDiagnostics:
    """Tests for global singleton access."""
    
    def test_initial_state_is_none(self):
        """Global diagnostics starts as None."""
        set_global_diagnostics(None)
        assert get_global_diagnostics() is None
    
    def test_set_and_get(self):
        """Can set and get global diagnostics."""
        manager = DiagnosticsManager.__new__(DiagnosticsManager)
        manager._width = 123
        
        set_global_diagnostics(manager)
        retrieved = get_global_diagnostics()
        
        assert retrieved is manager
        assert retrieved._width == 123
        
        # Cleanup
        set_global_diagnostics(None)
    
    def test_create_for_render_sets_global(self):
        """create_diagnostics_for_render sets global."""
        set_global_diagnostics(None)
        
        manager = create_diagnostics_for_render(640, 480, "minimal")
        
        assert get_global_diagnostics() is manager
        assert manager._width == 640
        assert manager._height == 480
        
        # Cleanup
        set_global_diagnostics(None)
    
    def test_create_with_presets(self):
        """Different presets work."""
        for preset in ["minimal", "standard", "detailed"]:
            manager = create_diagnostics_for_render(100, 100, preset)
            assert manager is not None
        
        # Cleanup
        set_global_diagnostics(None)


class TestDataClasses:
    """Tests for the dataclass structures."""
    
    def test_path_group_info_cv_calculation(self):
        """PathGroupInfo calculates CV correctly."""
        group = PathGroupInfo(
            pixel_x=10,
            pixel_y=20,
            signature="DE",
            sample_count=100,
            mean_luminance=2.0,
            variance_luminance=1.0,  # stddev = 1.0
            depth=2,
            coarse_type_name="Direct",
        )
        
        # CV = stddev / mean = 1.0 / 2.0 = 0.5
        assert abs(group.coefficient_of_variation - 0.5) < 1e-6
    
    def test_path_group_info_cv_zero_mean(self):
        """CV returns 0 for zero mean."""
        group = PathGroupInfo(
            pixel_x=0, pixel_y=0,
            signature="X",
            sample_count=1,
            mean_luminance=0.0,
            variance_luminance=1.0,
            depth=1,
            coarse_type_name="",
        )
        
        assert group.coefficient_of_variation == 0.0
    
    def test_global_stats_fields(self):
        """GlobalStats has all required fields."""
        stats = GlobalStats(
            total_samples=1000,
            active_pixels=500,
            total_groups=50,
            total_overflow=0,
            total_variance=123.45,
        )
        
        assert stats.total_samples == 1000
        assert stats.active_pixels == 500
        assert stats.total_groups == 50
        assert stats.total_overflow == 0
        assert abs(stats.total_variance - 123.45) < 1e-6
    
    def test_diagnostic_suggestion_fields(self):
        """DiagnosticSuggestion has all required fields."""
        suggestion = DiagnosticSuggestion(
            category="Settings",
            severity="warning",
            message="High variance detected",
            action="Increase samples",
        )
        
        assert suggestion.category == "Settings"
        assert suggestion.severity == "warning"
        assert "variance" in suggestion.message
        assert "samples" in suggestion.action


class TestDiagnosticReport:
    """Tests for DiagnosticReport."""
    
    def test_summary_generation(self):
        """summary() generates readable text."""
        report = DiagnosticReport(
            width=1920,
            height=1080,
            global_stats=GlobalStats(
                total_samples=10000,
                active_pixels=1000,
                total_groups=100,
                total_overflow=5,
                total_variance=500.0,
            ),
            top_variance_groups=[
                PathGroupInfo(
                    pixel_x=100, pixel_y=200,
                    signature="DDSE",
                    sample_count=50,
                    mean_luminance=1.0,
                    variance_luminance=4.0,
                    depth=4,
                    coarse_type_name="Caustic",
                ),
            ],
            top_mean_groups=[],
            suggestions=[
                DiagnosticSuggestion(
                    category="Settings",
                    severity="warning",
                    message="Caustic paths show high variance",
                    action="Enable caustic filter or increase samples",
                ),
            ],
            metadata_json="{}",
        )
        
        summary = report.summary()
        
        # Check key content
        assert "1920x1080" in summary
        assert "10000" in summary
        assert "DDSE" in summary
        assert "Caustic" in summary.replace("caustic", "Caustic")
        assert "warning" in summary
    
    def test_summary_handles_empty_groups(self):
        """summary() works with empty groups."""
        report = DiagnosticReport(
            width=100,
            height=100,
            global_stats=GlobalStats(0, 0, 0, 0, 0.0),
            top_variance_groups=[],
            top_mean_groups=[],
            suggestions=[],
            metadata_json="{}",
        )
        
        summary = report.summary()
        assert "100x100" in summary


# Integration tests - only run if C++ module available
class TestCppIntegration:
    """Tests requiring the C++ module."""
    
    @pytest.fixture(autouse=True)
    def check_cpp_available(self):
        """Skip tests if C++ module not available."""
        try:
            import diyrenderer
            yield
        except ImportError:
            pytest.skip("diyrenderer C++ module not available")
    
    def test_manager_is_available_with_cpp(self):
        """is_available returns True with C++ module."""
        manager = DiagnosticsManager.standard(100, 100)
        assert manager.is_available
    
    def test_memory_estimation(self):
        """Memory estimation returns reasonable values."""
        manager = DiagnosticsManager.standard(1920, 1080)
        
        memory_mb = manager.estimate_memory_mb()
        
        # Standard preset should use several hundred MB at 1080p
        assert memory_mb > 100, "Memory should be > 100 MB"
        assert memory_mb < 2000, "Memory should be < 2000 MB"
    
    def test_enable_disable_tracking(self):
        """Enable/disable properly tracks state."""
        manager = DiagnosticsManager.standard(100, 100)
        
        manager.enable()
        assert manager.is_enabled
        
        manager.disable()
        assert not manager.is_enabled
    
    def test_get_film_returns_film(self):
        """get_film returns DiagnosticFilm instance."""
        manager = DiagnosticsManager.standard(100, 100)
        
        film = manager.get_film()
        
        assert film is not None
        assert hasattr(film, 'clear')  # Has expected method
    
    def test_full_workflow(self):
        """Complete workflow: record paths, get report."""
        import diyrenderer
        
        # Create manager
        manager = DiagnosticsManager.standard(64, 64)
        manager.enable()
        
        film = manager.get_film()
        
        # Simulate recording paths using proper API
        recorder = diyrenderer.PathDiagnosticRecorder()
        recorder.begin_path()
        recorder.record_vertex(0, 0, diyrenderer.BsdfType.Diffuse, False, False)
        recorder.record_emissive_hit(1, 0)
        trace = recorder.end_path(1.5, 1.5, 1.5)
        film.record_path(10, 10, trace)
        
        # Get report
        report = manager.get_report(top_n=5)
        
        assert report is not None
        assert report.width == 64
        assert report.height == 64
        assert report.global_stats.total_samples >= 1
    
    def test_export_binary(self):
        """Binary export produces data."""
        import diyrenderer
        
        manager = DiagnosticsManager.minimal(32, 32)
        film = manager.get_film()
        
        # Record a path
        recorder = diyrenderer.PathDiagnosticRecorder()
        recorder.begin_path()
        recorder.record_vertex(0, 0, diyrenderer.BsdfType.Glossy, False, False)
        trace = recorder.end_path_lum(0.5)
        film.record_path(5, 5, trace)
        
        # Export
        data = manager.export_binary()
        
        assert data is not None
        assert len(data) > 0
    
    def test_clear(self):
        """clear() resets diagnostic data."""
        import diyrenderer
        
        manager = DiagnosticsManager.minimal(16, 16)
        film = manager.get_film()
        
        # Record path
        recorder = diyrenderer.PathDiagnosticRecorder()
        recorder.begin_path()
        recorder.record_vertex(0, 0, diyrenderer.BsdfType.Diffuse, False, False)
        trace = recorder.end_path_lum(1.0)
        film.record_path(0, 0, trace)
        
        # Clear
        manager.clear()
        
        # Report should show no samples
        report = manager.get_report()
        assert report.global_stats.total_samples == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
