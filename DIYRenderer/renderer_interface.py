"""
Abstract interface for renderer communication.

This module defines the abstract base class for renderer backends,
allowing different communication methods (subprocess, pybind11, etc.)
to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np


class BackendType(Enum):
    """Available rendering backends."""
    CPU = "cpu"
    WEBGPU = "webgpu"


class AlgorithmType(Enum):
    """Available path tracing algorithms."""
    NAIVE = "simple"      # Naive path tracing (reference)
    NEE = "nee"           # Next Event Estimation (recommended)
    MIS = "mis"           # Multiple Importance Sampling


@dataclass
class RenderConfig:
    """Configuration for renderer initialization."""
    backend: BackendType = BackendType.CPU
    algorithm: AlgorithmType = AlgorithmType.NEE
    max_depth: int = 8


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    pos: tuple  # (x, y, z)
    dir: tuple  # (x, y, z) - direction vector
    up: tuple   # (x, y, z) - up vector
    fov: float  # field of view in degrees
    
    @classmethod
    def from_blender(cls, cam_params: dict) -> 'CameraParams':
        """Create from Blender camera params dict."""
        return cls(
            pos=(cam_params['pos'].x, cam_params['pos'].y, cam_params['pos'].z),
            dir=(cam_params['dir'].x, cam_params['dir'].y, cam_params['dir'].z),
            up=(cam_params['up'].x, cam_params['up'].y, cam_params['up'].z),
            fov=cam_params['fov']
        )


@dataclass
class TileParams:
    """Parameters for tile rendering."""
    tile_x: int
    tile_y: int
    tile_w: int
    tile_h: int
    full_w: int
    full_h: int
    samples: int
    sample_offset: int = 0


@dataclass
class RenderResult:
    """Result from a render operation."""
    pixels: List[float]     # Flat list [r,g,b,a, r,g,b,a, ...] 
    width: int
    height: int
    samples_rendered: int
    render_time_ms: float = 0.0
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [H, W, 4]."""
        arr = np.array(self.pixels, dtype=np.float32)
        return arr.reshape((self.height, self.width, 4))


class RendererInterface(ABC):
    """
    Abstract interface for renderer communication.
    
    This interface allows different renderer backends to be used
    interchangeably. Implementations include:
    - SubprocessRenderer: Uses stdin/stdout binary protocol
    - (Future) PybindRenderer: Direct C++ function calls via pybind11
    """
    
    @abstractmethod
    def start(self, config: RenderConfig) -> bool:
        """
        Start the renderer.
        
        Args:
            config: Renderer configuration
            
        Returns:
            True if started successfully
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the renderer is running."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the renderer."""
        pass
    
    @abstractmethod
    def update_scene(self, scene_json: str) -> bool:
        """
        Update the scene data.
        
        Args:
            scene_json: JSON string with scene data
            
        Returns:
            True if scene was loaded successfully
        """
        pass
    
    @abstractmethod
    def update_camera(self, camera: CameraParams) -> bool:
        """
        Update camera parameters.
        
        This is a fast path for camera-only updates (no scene reload).
        
        Args:
            camera: Camera parameters
            
        Returns:
            True if camera was updated successfully
        """
        pass
    
    @abstractmethod
    def render_tile(self, tile: TileParams) -> Optional[RenderResult]:
        """
        Render a tile.
        
        Args:
            tile: Tile parameters
            
        Returns:
            RenderResult with pixel data, or None if failed
        """
        pass
    
    @abstractmethod
    def cancel(self) -> None:
        """Cancel current rendering operation."""
        pass
    
    @abstractmethod
    def set_backend(self, backend: BackendType) -> bool:
        """
        Change rendering backend.
        
        Args:
            backend: New backend to use
            
        Returns:
            True if backend was changed successfully
        """
        pass
    
    @abstractmethod
    def set_algorithm(self, algorithm: AlgorithmType) -> bool:
        """
        Change path tracing algorithm.
        
        Args:
            algorithm: New algorithm to use
            
        Returns:
            True if algorithm was changed successfully
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, List[str]]:
        """
        Query renderer capabilities.
        
        Returns:
            Dict with 'backends' and 'algorithms' lists
        """
        pass


class RendererError(Exception):
    """Exception raised by renderer operations."""
    pass
