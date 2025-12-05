"""
DIY Renderer - Custom Path Tracing Renderer for Blender
========================================================

This addon integrates a custom C++ path tracer with Blender's rendering system.
It exports Blender scenes to JSON, renders them using an external C++ executable,
and displays the results in Blender's render window and viewport.

Architecture:
- Python (Blender addon): Scene export, UI, result display
- C++ (external executable): Ray tracing, path tracing, material evaluation

Features:
- Progressive rendering with sample accumulation
- Node-based material support (Principled BSDF, Emission)
- Debug modes: normals, albedo, emission
- Viewport and F12 rendering
- Asynchronous viewport updates
"""

bl_info = {
    "name": "DIY Renderer (Minimal Example)",
    "author": "You",
    "version": (0, 0, 2),
    "blender": (4, 5, 0),
    "location": "Render > Engine",
    "description": "Minimal renderer with external C++ integration",
    "category": "Render",
}

import bpy

from .preferences import DIYRendererPreferences, DIYRendererSettings, _get_prefs
from .panels import DIY_RENDER_PT_sampling, DIY_RENDER_PT_light_paths, DIY_RENDER_PT_debug, DIY_RENDER_PT_performance
from .engine import DIYRenderEngine


# Global timer handle
_viewport_timer = None
_viewport_timer_interval = 0.1  # 100ms


def _viewport_redraw_timer():
    """
    Timer callback to force viewport redraws.
    This ensures view_draw is called regularly even when viewport is static.
    Only redraws when our render engine is active.
    """
    try:
        scene = bpy.context.scene
        if scene and scene.render.engine == 'DIY_RENDER_MINIMAL':
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D' and space.shading.type == 'RENDERED':
                                area.tag_redraw()
                                break
    except Exception:
        pass
    
    return _viewport_timer_interval


def register():
    global _viewport_timer
    
    bpy.utils.register_class(DIYRendererPreferences)
    bpy.utils.register_class(DIYRendererSettings)
    bpy.utils.register_class(DIY_RENDER_PT_sampling)
    bpy.utils.register_class(DIY_RENDER_PT_light_paths)
    bpy.utils.register_class(DIY_RENDER_PT_debug)
    bpy.utils.register_class(DIY_RENDER_PT_performance)
    bpy.utils.register_class(DIYRenderEngine)
    
    bpy.types.Scene.diy_renderer = bpy.props.PointerProperty(type=DIYRendererSettings)
    
    if not bpy.app.timers.is_registered(_viewport_redraw_timer):
        bpy.app.timers.register(_viewport_redraw_timer, first_interval=_viewport_timer_interval, persistent=True)
    
    try:
        from bl_ui import (
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        )
        
        modules = [
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        ]
        
        exclude_panels = {
            'CYCLES_RENDER_PT_sampling',
            'CYCLES_RENDER_PT_light_paths',
            'CYCLES_RENDER_PT_performance',
        }
        
        for module in modules:
            for panel_name in dir(module):
                if panel_name in exclude_panels:
                    continue
                
                panel = getattr(module, panel_name, None)
                if panel and hasattr(panel, 'COMPAT_ENGINES'):
                    panel.COMPAT_ENGINES.add('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not register panels: {e}")


def unregister():
    global _viewport_timer
    
    if bpy.app.timers.is_registered(_viewport_redraw_timer):
        bpy.app.timers.unregister(_viewport_redraw_timer)
    
    try:
        from bl_ui import (
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        )
        
        modules = [
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        ]
        
        for module in modules:
            for panel_name in dir(module):
                panel = getattr(module, panel_name, None)
                if panel and hasattr(panel, 'COMPAT_ENGINES') and 'DIY_RENDER_MINIMAL' in panel.COMPAT_ENGINES:
                    panel.COMPAT_ENGINES.discard('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not unregister panels: {e}")
    
    del bpy.types.Scene.diy_renderer
    
    bpy.utils.unregister_class(DIYRenderEngine)
    bpy.utils.unregister_class(DIY_RENDER_PT_performance)
    bpy.utils.unregister_class(DIY_RENDER_PT_debug)
    bpy.utils.unregister_class(DIY_RENDER_PT_light_paths)
    bpy.utils.unregister_class(DIY_RENDER_PT_sampling)
    bpy.utils.unregister_class(DIYRendererSettings)
    bpy.utils.unregister_class(DIYRendererPreferences)


if __name__ == "__main__":
    register()
