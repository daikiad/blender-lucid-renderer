"""
diagnostics.py - Path Variance Analyzer Python API
===================================================

Provides a high-level Python interface for the C++ diagnostic system.
Used by Blender panels and render engine to collect and display
path variance analysis results.

Usage:
    from LucidRenderer.diagnostics import DiagnosticsManager
    
    # Create manager with standard preset
    manager = DiagnosticsManager.standard(width, height)
    
    # During rendering, record paths via C++ bindings
    # ...
    
    # After rendering, get analysis
    report = manager.get_report()
    print(report.summary())
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json


@dataclass
class PathGroupInfo:
    """Information about a single path group (signature)."""
    pixel_x: int
    pixel_y: int
    signature: str
    sample_count: int
    mean_luminance: float
    variance_luminance: float
    depth: int
    coarse_type_name: str
    
    @property
    def coefficient_of_variation(self) -> float:
        """CV = stddev / mean, measure of relative variance."""
        if self.mean_luminance <= 0:
            return 0.0
        return (self.variance_luminance ** 0.5) / self.mean_luminance


@dataclass  
class DiagnosticSuggestion:
    """Improvement suggestion from the analyzer."""
    category: str  # "Settings", "Scene", "Algorithm"
    severity: str  # "info", "warning", "critical"
    message: str
    action: str


@dataclass
class GlobalStats:
    """Global statistics for the entire render."""
    total_samples: int
    active_pixels: int
    total_groups: int
    total_overflow: int
    total_variance: float


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a render session."""
    width: int
    height: int
    global_stats: GlobalStats
    top_variance_groups: List[PathGroupInfo]
    top_mean_groups: List[PathGroupInfo]
    suggestions: List[DiagnosticSuggestion]
    metadata_json: str
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Path Variance Analysis Report ===",
            f"Resolution: {self.width}x{self.height}",
            f"Total samples: {self.global_stats.total_samples}",
            f"Active pixels: {self.global_stats.active_pixels}",
            f"Path groups: {self.global_stats.total_groups}",
            "",
        ]
        
        if self.suggestions:
            lines.append("Suggestions:")
            for s in self.suggestions:
                lines.append(f"  [{s.severity}] {s.message}")
                lines.append(f"    → {s.action}")
            lines.append("")
        
        if self.top_variance_groups:
            lines.append("Top variance contributors:")
            for g in self.top_variance_groups[:5]:
                cv = g.coefficient_of_variation
                lines.append(f"  {g.signature}: CV={cv:.2f}, var={g.variance_luminance:.4f}")
        
        return "\n".join(lines)


class DiagnosticsManager:
    """
    High-level manager for path variance diagnostics.
    
    Wraps the C++ DiagnosticPathTracer and DiagnosticExporter.
    """
    
    def __init__(self, width: int, height: int, config=None):
        """
        Initialize diagnostics manager.
        
        Args:
            width: Render width in pixels
            height: Render height in pixels
            config: PathRecordingConfig (uses standard if None)
        """
        self._width = width
        self._height = height
        self._tracer = None
        self._enabled = False
        
        try:
            import lucidrenderer
            
            if config is None:
                config = lucidrenderer.PathRecordingConfig.standard()
            
            self._tracer = lucidrenderer.DiagnosticPathTracer(width, height, config)
            self._enabled = True
            self._config = config
        except ImportError:
            # C++ module not available, run in stub mode
            self._config = None
    
    @classmethod
    def minimal(cls, width: int, height: int) -> 'DiagnosticsManager':
        """Create manager with minimal preset (~200MB at 1080p)."""
        try:
            import lucidrenderer
            config = lucidrenderer.PathRecordingConfig.minimal()
            return cls(width, height, config)
        except ImportError:
            return cls(width, height, None)
    
    @classmethod
    def standard(cls, width: int, height: int) -> 'DiagnosticsManager':
        """Create manager with standard preset (~800MB at 1080p)."""
        try:
            import lucidrenderer
            config = lucidrenderer.PathRecordingConfig.standard()
            return cls(width, height, config)
        except ImportError:
            return cls(width, height, None)
    
    @classmethod
    def detailed(cls, width: int, height: int) -> 'DiagnosticsManager':
        """Create manager with detailed preset (~3.2GB at 1080p)."""
        try:
            import lucidrenderer
            config = lucidrenderer.PathRecordingConfig.detailed()
            return cls(width, height, config)
        except ImportError:
            return cls(width, height, None)
    
    @property
    def is_available(self) -> bool:
        """Check if C++ diagnostics are available."""
        return self._tracer is not None
    
    @property
    def is_enabled(self) -> bool:
        """Check if diagnostics are enabled."""
        return self._enabled and self._tracer is not None
    
    def enable(self):
        """Enable diagnostic recording."""
        if self._tracer:
            self._tracer.set_enabled(True)
            self._enabled = True
    
    def disable(self):
        """Disable diagnostic recording."""
        if self._tracer:
            self._tracer.set_enabled(False)
            self._enabled = False
    
    def get_film(self):
        """Get the underlying DiagnosticFilm (for C++ integration)."""
        if self._tracer:
            return self._tracer.film()
        return None
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory usage in megabytes."""
        if self._config:
            try:
                import lucidrenderer
                bytes_needed = self._config.estimate_memory_bytes(self._width, self._height)
                return bytes_needed / (1024 * 1024)
            except:
                pass
        return 0.0
    
    def get_report(self, top_n: int = 10) -> Optional[DiagnosticReport]:
        """
        Generate diagnostic report from recorded data.
        
        Args:
            top_n: Number of top groups to include
            
        Returns:
            DiagnosticReport or None if not available
        """
        if not self._tracer:
            return None
        
        try:
            import lucidrenderer
            
            film = self._tracer.film()
            exporter = lucidrenderer.DiagnosticExporter(film)
            
            # Get global stats
            cpp_stats = exporter.get_global_stats()
            global_stats = GlobalStats(
                total_samples=cpp_stats.total_samples,
                active_pixels=cpp_stats.active_pixels,
                total_groups=cpp_stats.total_groups,
                total_overflow=cpp_stats.total_overflow,
                total_variance=cpp_stats.total_variance,
            )
            
            # Get top variance groups
            cpp_var_groups = exporter.get_top_variance_groups(top_n)
            top_variance = [
                PathGroupInfo(
                    pixel_x=g.pixel_x,
                    pixel_y=g.pixel_y,
                    signature=g.signature,
                    sample_count=g.sample_count,
                    mean_luminance=g.mean_luminance,
                    variance_luminance=g.variance_luminance,
                    depth=g.depth,
                    coarse_type_name=g.coarse_type_name,
                )
                for g in cpp_var_groups
            ]
            
            # Get top mean groups
            cpp_mean_groups = exporter.get_top_mean_groups(top_n)
            top_mean = [
                PathGroupInfo(
                    pixel_x=g.pixel_x,
                    pixel_y=g.pixel_y,
                    signature=g.signature,
                    sample_count=g.sample_count,
                    mean_luminance=g.mean_luminance,
                    variance_luminance=g.variance_luminance,
                    depth=g.depth,
                    coarse_type_name=g.coarse_type_name,
                )
                for g in cpp_mean_groups
            ]
            
            # Get suggestions
            cpp_suggestions = exporter.generate_suggestions()
            suggestions = [
                DiagnosticSuggestion(
                    category=s.category,
                    severity=s.severity,
                    message=s.message,
                    action=s.action,
                )
                for s in cpp_suggestions
            ]
            
            # Get metadata JSON
            metadata_json = exporter.export_metadata_json()
            
            return DiagnosticReport(
                width=self._width,
                height=self._height,
                global_stats=global_stats,
                top_variance_groups=top_variance,
                top_mean_groups=top_mean,
                suggestions=suggestions,
                metadata_json=metadata_json,
            )
            
        except Exception as e:
            print(f"[Diagnostics] Error generating report: {e}")
            return None
    
    def export_binary(self) -> Optional[bytes]:
        """Export diagnostic data as binary blob."""
        if not self._tracer:
            return None
        
        try:
            import lucidrenderer
            film = self._tracer.film()
            exporter = lucidrenderer.DiagnosticExporter(film)
            return exporter.export_binary()
        except Exception as e:
            print(f"[Diagnostics] Error exporting binary: {e}")
            return None
    
    def clear(self):
        """Clear all recorded diagnostic data."""
        if self._tracer:
            self._tracer.film().clear()


# Singleton for global access from render engine
_global_diagnostics: Optional[DiagnosticsManager] = None


def get_global_diagnostics() -> Optional[DiagnosticsManager]:
    """Get the global diagnostics manager."""
    return _global_diagnostics


def set_global_diagnostics(manager: Optional[DiagnosticsManager]):
    """Set the global diagnostics manager."""
    global _global_diagnostics
    _global_diagnostics = manager


def create_diagnostics_for_render(width: int, height: int, preset: str = "standard") -> DiagnosticsManager:
    """
    Create and set global diagnostics manager for a render.
    
    Args:
        width: Render width
        height: Render height
        preset: "minimal", "standard", or "detailed"
        
    Returns:
        The created DiagnosticsManager
    """
    if preset == "minimal":
        manager = DiagnosticsManager.minimal(width, height)
    elif preset == "detailed":
        manager = DiagnosticsManager.detailed(width, height)
    else:
        manager = DiagnosticsManager.standard(width, height)
    
    set_global_diagnostics(manager)
    return manager
