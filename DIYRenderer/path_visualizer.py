"""
Path Visualizer - 2D/3D path visualization for diagnostics.
============================================================

This module provides visualization of ray paths in both 2D (Image Editor) 
and 3D (Viewport) views. Paths are drawn as polylines from camera through
each bounce point to the light source.

Features:
- 2D polyline overlay in Image Editor
- 3D polyline visualization in Viewport
- Click-to-lock pixel selection
- Contribution-sorted path list
- Max visualized paths (configurable, default 20, max 100)

Usage:
    from .path_visualizer import PathVisualizer
    
    # Get paths for a pixel
    visualizer = PathVisualizer(renderer, settings)
    paths = visualizer.get_paths_for_pixel(x, y)
    
    # Draw in 2D
    visualizer.draw_2d(context, image_region)
    
    # Draw in 3D
    visualizer.draw_3d(context)
"""

from typing import List, Tuple, Optional
import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

# Type alias for positions
Position3D = Tuple[float, float, float]


class PathData:
    """
    Data for a single path to visualize.
    
    Attributes:
        positions: List of 3D positions (camera first, then bounces)
        normals: List of surface normals at each bounce
        contribution: RGB contribution of this path
        signature: Path signature string (e.g., "LDD D DSE")
        object_path: Object names in path (e.g., "Light → Plane → Camera")
        variance: Variance of this path type
        mean: Mean contribution luminance
    """
    
    def __init__(self):
        self.positions: List[Position3D] = []
        self.normals: List[Position3D] = []
        self.contribution: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.signature: str = ""
        self.signature_heckbert: str = ""
        self.object_path: str = ""
        self.variance: float = 0.0
        self.mean: float = 0.0
        self.sample_count: int = 0
        self.strategy: str = ""
    
    @classmethod
    def from_exported_group(cls, group) -> 'PathData':
        """
        Create PathData from ExportedGroupInfo (pybind object).
        
        Args:
            group: ExportedGroupInfo from get_pixel_diagnostic
        
        Returns:
            PathData with positions and metadata
        """
        path = cls()
        path.positions = [tuple(p) for p in group.positions]
        path.normals = [tuple(n) for n in group.normals]
        path.contribution = tuple(group.mean_rgb)
        path.signature = group.signature
        path.signature_heckbert = group.signature_heckbert
        path.object_path = group.object_path
        path.variance = group.variance_luminance
        path.mean = group.mean_luminance
        path.sample_count = group.sample_count
        path.strategy = group.strategy_name
        return path
    
    def luminance(self) -> float:
        """Calculate luminance from contribution RGB."""
        return 0.2126 * self.contribution[0] + 0.7152 * self.contribution[1] + 0.0722 * self.contribution[2]


class PathVisualizer:
    """
    Visualizer for ray paths in 2D and 3D views.
    
    This class manages:
    - Retrieving path data from the renderer
    - Converting between coordinate systems
    - Drawing paths as polylines in 2D/3D
    
    Attributes:
        renderer: The PyRenderer instance with diagnostics enabled
        settings: DIYRendererSettings with visualization preferences
        cached_paths: List of PathData for the current pixel
        cached_pixel: The pixel coordinates for cached data
    """
    
    def __init__(self, renderer, settings):
        """
        Initialize the path visualizer.
        
        Args:
            renderer: PyRenderer instance with diagnostics enabled
            settings: DIYRendererSettings from scene.diy_renderer
        """
        self.renderer = renderer
        self.settings = settings
        self.cached_paths: List[PathData] = []
        self.cached_pixel: Optional[Tuple[int, int]] = None
        
        # GPU shader for drawing lines
        self._shader_2d = gpu.shader.from_builtin('UNIFORM_COLOR')
        self._shader_3d = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    def get_paths_for_pixel(self, x: int, y: int, max_paths: Optional[int] = None) -> List[PathData]:
        """
        Get visualizable paths for a specific pixel.
        
        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
            max_paths: Maximum paths to return (default from settings)
        
        Returns:
            List of PathData sorted by contribution (brightest first)
        """
        if max_paths is None:
            max_paths = self.settings.max_visualized_paths
        
        # Check cache
        if self.cached_pixel == (x, y) and self.cached_paths:
            return self.cached_paths[:max_paths]
        
        # Get diagnostics from renderer
        if not self.renderer or not self.renderer.is_diagnostics_enabled():
            return []
        
        try:
            # Get pixel diagnostic data sorted by mean contribution
            diag = self.renderer.get_pixel_diagnostic(x, y, "mean")
            if not diag.valid:
                return []
            
            # Convert to PathData
            paths = []
            for group in diag.top_groups:
                if group.positions:  # Only include paths with geometry
                    path = PathData.from_exported_group(group)
                    paths.append(path)
            
            # Cache results
            self.cached_pixel = (x, y)
            self.cached_paths = paths
            
            return paths[:max_paths]
            
        except Exception as e:
            print(f"[PathVisualizer] Error getting paths: {e}")
            return []
    
    def invalidate_cache(self):
        """Clear the cached path data."""
        self.cached_paths = []
        self.cached_pixel = None
    
    def get_locked_pixel(self) -> Optional[Tuple[int, int]]:
        """
        Get the locked pixel coordinates if selection is locked.
        
        Returns:
            (x, y) tuple if locked, None otherwise
        """
        if self.settings.path_selection_locked:
            return (self.settings.locked_pixel_x, self.settings.locked_pixel_y)
        return None
    
    def lock_pixel(self, x: int, y: int):
        """
        Lock the pixel selection for path visualization.
        
        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
        """
        self.settings.path_selection_locked = True
        self.settings.locked_pixel_x = x
        self.settings.locked_pixel_y = y
    
    def unlock_pixel(self):
        """Unlock the pixel selection."""
        self.settings.path_selection_locked = False
    
    def toggle_lock(self, x: int, y: int):
        """
        Toggle the pixel selection lock.
        
        If locked at the same position, unlock.
        Otherwise, lock at the new position.
        
        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
        """
        if self.settings.path_selection_locked:
            if self.settings.locked_pixel_x == x and self.settings.locked_pixel_y == y:
                self.unlock_pixel()
            else:
                self.lock_pixel(x, y)
        else:
            self.lock_pixel(x, y)
    
    def world_to_image_coords(self, 
                               position: Position3D, 
                               camera: bpy.types.Object,
                               region: bpy.types.Region,
                               rv3d: bpy.types.RegionView3D,
                               image_width: int,
                               image_height: int) -> Optional[Tuple[float, float]]:
        """
        Convert world position to image coordinates.
        
        Args:
            position: 3D world position
            camera: Camera object
            region: Region for coordinate conversion
            rv3d: RegionView3D
            image_width: Width of the rendered image
            image_height: Height of the rendered image
        
        Returns:
            (x, y) in image pixel coordinates, or None if behind camera
        """
        from bpy_extras.object_utils import world_to_camera_view
        
        scene = bpy.context.scene
        pos_vec = Vector(position)
        
        # Convert to camera view (normalized 0-1 coordinates)
        co = world_to_camera_view(scene, camera, pos_vec)
        
        # Check if behind camera
        if co.z < 0:
            return None
        
        # Convert to image pixel coordinates
        x = co.x * image_width
        y = co.y * image_height
        
        return (x, y)
    
    def draw_paths_2d(self, 
                       paths: List[PathData],
                       camera: bpy.types.Object,
                       region: bpy.types.Region,
                       rv3d: bpy.types.RegionView3D,
                       image_width: int,
                       image_height: int,
                       offset_x: float = 0,
                       offset_y: float = 0,
                       scale: float = 1.0):
        """
        Draw paths as 2D polylines in image space.
        
        Args:
            paths: List of PathData to draw
            camera: Camera object for projection
            region: Region for coordinate conversion
            rv3d: RegionView3D
            image_width: Rendered image width
            image_height: Rendered image height
            offset_x: X offset for image display
            offset_y: Y offset for image display
            scale: Scale factor for image display
        """
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        
        for i, path in enumerate(paths):
            if len(path.positions) < 2:
                continue
            
            # Convert positions to 2D
            coords_2d = []
            for pos in path.positions:
                coord = self.world_to_image_coords(
                    pos, camera, region, rv3d, image_width, image_height
                )
                if coord is None:
                    break
                # Apply offset and scale
                x = offset_x + coord[0] * scale
                y = offset_y + coord[1] * scale
                coords_2d.append((x, y))
            
            if len(coords_2d) < 2:
                continue
            
            # Generate color based on path index (rainbow)
            hue = (i / max(len(paths), 1)) * 0.8  # 0 to 0.8 (red to purple)
            color = self._hue_to_rgb(hue) + (0.8,)  # Add alpha
            
            # Draw polyline
            batch = batch_for_shader(
                self._shader_2d, 'LINE_STRIP',
                {"pos": coords_2d}
            )
            self._shader_2d.bind()
            self._shader_2d.uniform_float("color", color)
            batch.draw(self._shader_2d)
        
        # Restore state
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
    
    def draw_paths_3d(self, paths: List[PathData]):
        """
        Draw paths as 3D polylines in the viewport.
        
        Args:
            paths: List of PathData to draw
        """
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        gpu.state.depth_test_set('LESS_EQUAL')
        
        for i, path in enumerate(paths):
            if len(path.positions) < 2:
                continue
            
            # Generate color based on path index
            hue = (i / max(len(paths), 1)) * 0.8
            color = self._hue_to_rgb(hue) + (0.8,)
            
            # Draw polyline
            batch = batch_for_shader(
                self._shader_3d, 'LINE_STRIP',
                {"pos": path.positions}
            )
            self._shader_3d.bind()
            self._shader_3d.uniform_float("color", color)
            batch.draw(self._shader_3d)
            
            # Draw normal indicators at bounce points
            for j, (pos, normal) in enumerate(zip(path.positions[1:], path.normals)):
                self._draw_normal_indicator_3d(pos, normal, color)
        
        # Restore state
        gpu.state.depth_test_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
    
    def _draw_normal_indicator_3d(self, 
                                   position: Position3D, 
                                   normal: Position3D, 
                                   color: Tuple[float, ...],
                                   length: float = 0.1):
        """Draw a small line indicating surface normal at a bounce point."""
        start = Vector(position)
        end = start + Vector(normal) * length
        
        batch = batch_for_shader(
            self._shader_3d, 'LINES',
            {"pos": [tuple(start), tuple(end)]}
        )
        self._shader_3d.bind()
        self._shader_3d.uniform_float("color", color)
        batch.draw(self._shader_3d)
    
    def _hue_to_rgb(self, hue: float) -> Tuple[float, float, float]:
        """
        Convert hue (0-1) to RGB color.
        
        Args:
            hue: Hue value 0-1
        
        Returns:
            (r, g, b) tuple with values 0-1
        """
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        return (r, g, b)


# =============================================================================
# Draw handlers for Blender
# =============================================================================

_draw_handler_2d = None
_draw_handler_3d = None
_visualizer_instance: Optional[PathVisualizer] = None


def register_draw_handlers():
    """Register the draw handlers for path visualization."""
    global _draw_handler_2d, _draw_handler_3d
    
    if _draw_handler_2d is None:
        _draw_handler_2d = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_callback_2d, (), 'WINDOW', 'POST_PIXEL'
        )
    
    if _draw_handler_3d is None:
        _draw_handler_3d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback_3d, (), 'WINDOW', 'POST_VIEW'
        )


def unregister_draw_handlers():
    """Unregister the draw handlers."""
    global _draw_handler_2d, _draw_handler_3d
    
    if _draw_handler_2d is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handler_2d, 'WINDOW')
        _draw_handler_2d = None
    
    if _draw_handler_3d is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler_3d, 'WINDOW')
        _draw_handler_3d = None


def set_visualizer(visualizer: Optional[PathVisualizer]):
    """Set the active path visualizer instance."""
    global _visualizer_instance
    _visualizer_instance = visualizer


def _draw_callback_2d():
    """Draw callback for 2D path visualization in Image Editor."""
    global _visualizer_instance
    
    if _visualizer_instance is None:
        return
    
    # Get current context
    context = bpy.context
    if context.area is None or context.area.type != 'IMAGE_EDITOR':
        return
    
    settings = _visualizer_instance.settings
    if not settings.enable_diagnostics:
        return
    
    # Get locked or hover pixel
    locked = _visualizer_instance.get_locked_pixel()
    if locked is None:
        return  # No pixel selected
    
    x, y = locked
    paths = _visualizer_instance.get_paths_for_pixel(x, y)
    
    if not paths:
        return
    
    # Get camera and image info
    scene = context.scene
    camera = scene.camera
    if camera is None:
        return
    
    # Get render dimensions
    render = scene.render
    image_width = int(render.resolution_x * render.resolution_percentage / 100)
    image_height = int(render.resolution_y * render.resolution_percentage / 100)
    
    # Get image editor view info
    space = context.space_data
    region = context.region
    
    # Get image offset and scale from the image editor
    # This is complex in Blender, so we'll use a simplified approach
    # TODO: Properly calculate offset and scale from space_data.image_user
    
    _visualizer_instance.draw_paths_2d(
        paths, camera, region, None,
        image_width, image_height,
        offset_x=0, offset_y=0, scale=1.0
    )


def _draw_callback_3d():
    """Draw callback for 3D path visualization in Viewport."""
    global _visualizer_instance
    
    if _visualizer_instance is None:
        return
    
    # Get current context
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return
    
    settings = _visualizer_instance.settings
    if not settings.enable_diagnostics:
        return
    
    # Get locked or hover pixel
    locked = _visualizer_instance.get_locked_pixel()
    if locked is None:
        return
    
    x, y = locked
    paths = _visualizer_instance.get_paths_for_pixel(x, y)
    
    if not paths:
        return
    
    _visualizer_instance.draw_paths_3d(paths)
