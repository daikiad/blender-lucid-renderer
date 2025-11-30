bl_info = {
    "name": "DIY Renderer (Minimal Example)",
    "author": "You",
    "version": (0, 0, 1),
    "blender": (4, 5, 0),
    "location": "Render > Engine",
    "description": "Minimal dummy renderer that returns a solid red image",
    "category": "Render",
}

import bpy


class DIYRenderEngine(bpy.types.RenderEngine):
    bl_idname = "DIY_RENDER_MINIMAL"
    bl_label = "DIY Renderer (Minimal)"
    bl_use_preview = True  # マテリアルプレビューでも使えるように

    def render(self, depsgraph):
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)

        # 結果バッファ開始
        result = self.begin_result(0, 0, width, height)

        rlayer = result.layers[0]
        combined = rlayer.passes["Combined"]

        # 真っ赤な RGBA
        if self.is_preview:
            color = [1.0, 0.3, 0.3, 1.0]
        else:
            color = [1.0, 0.0, 0.0, 1.0]

        pixel_count = width * height
        rect = [color] * pixel_count

        combined.rect = rect

        self.end_result(result)


def register():
    bpy.utils.register_class(DIYRenderEngine)


def unregister():
    bpy.utils.unregister_class(DIYRenderEngine)