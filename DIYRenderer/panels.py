"""
UI Panels for DIY Renderer.
"""

import bpy


class DIY_RENDER_PT_sampling(bpy.types.Panel):
    bl_label = "Sampling"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        
        # Render section (like Cycles)
        col = layout.column(heading="Render")
        col.prop(diy, "samples", text="Samples")
        
        # Viewport section
        col = layout.column(heading="Viewport")
        col.prop(diy, "viewport_samples", text="Samples")
        
        # Algorithm section
        col = layout.column(heading="Algorithm")
        col.prop(diy, "sampling_algorithm", text="Method")


class DIY_RENDER_PT_light_paths(bpy.types.Panel):
    bl_label = "Light Paths"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        
        col = layout.column(heading="Max Bounces")
        col.prop(diy, "max_bounces", text="Total")


class DIY_RENDER_PT_debug(bpy.types.Panel):
    bl_label = "Debug"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        layout.prop(diy, "debug_mode")
