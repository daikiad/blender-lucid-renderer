"""
Addon preferences and settings.
"""

import bpy


def _get_prefs_entry():
    """Get addon preferences entry from Blender"""
    return bpy.context.preferences.addons.get("DIYRenderer")


def _get_prefs():
    """Get addon preferences object"""
    entry = _get_prefs_entry()
    if entry is not None:
        return getattr(entry, 'preferences', None)
    return None


class DIYRendererPreferences(bpy.types.AddonPreferences):
    """Addon preferences for configuring external renderer path"""
    bl_idname = "DIYRenderer"
    
    external_renderer_path: bpy.props.StringProperty(
        name="External Renderer Path", 
        description="Path to compiled external C++ renderer binary (diyrt)", 
        default="", 
        subtype='FILE_PATH'
    )
    scene_export_directory: bpy.props.StringProperty(
        name="Scene Export Temp Dir", 
        description="Directory to write temporary exported scene files", 
        default="", 
        subtype='DIR_PATH'
    )
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "external_renderer_path")
        layout.prop(self, "scene_export_directory")


class DIYRendererSettings(bpy.types.PropertyGroup):
    """Per-scene settings for DIY Renderer"""
    samples: bpy.props.IntProperty(
        name="Samples",
        description="Number of samples for path tracing",
        default=128,
        min=1,
        max=10000
    )
    viewport_samples: bpy.props.IntProperty(
        name="Viewport Samples",
        description="Maximum samples for viewport rendering",
        default=64,
        min=1,
        max=1000
    )
    max_bounces: bpy.props.IntProperty(
        name="Max Bounces",
        description="Maximum number of light bounces (ray depth)",
        default=8,
        min=1,
        max=128
    )
    sampling_algorithm: bpy.props.EnumProperty(
        name="Sampling Algorithm",
        description="Path tracing algorithm",
        items=[
            ('simple', "Simple", "BSDF sampling only (slow convergence, good for debugging)"),
            ('nee', "NEE", "Next Event Estimation (fast direct lighting)"),
            ('mis', "MIS", "Multiple Importance Sampling (best quality)"),
        ],
        default='nee'
    )
    debug_mode: bpy.props.EnumProperty(
        name="Debug Mode",
        description="Render mode for debugging",
        items=[
            ('NONE', "Path Tracing", "Full path tracing with global illumination"),
            ('normal', "Normals", "Show surface normals as RGB colors"),
            ('albedo', "Albedo", "Show base colors without lighting"),
            ('emission', "Emission", "Show emissive surfaces only"),
        ],
        default='NONE'
    )
    use_server_mode: bpy.props.BoolProperty(
        name="Use Server Mode",
        description="Use persistent renderer process (faster for viewport, experimental)",
        default=False
    )
    backend: bpy.props.EnumProperty(
        name="Backend",
        description="Rendering backend",
        items=[
            ('cpu', "CPU", "Multi-threaded CPU rendering (OpenMP)"),
            # ('webgpu', "WebGPU", "GPU rendering via Dawn (experimental)"),  # Phase 2
        ],
        default='cpu'
    )
