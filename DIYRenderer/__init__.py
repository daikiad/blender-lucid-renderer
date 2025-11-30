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

        pixels = []

        for y in range(height):
            fy = y / (height - 1) if height > 1 else 0.0
            for x in range(width):
                fx = x / (width - 1) if width > 1 else 0.0

                r = fx
                g = fy
                b = 0.2
                a = 1.0
                pixels.append([r, g, b, a])

        combined.rect = pixels

        self.end_result(result)


def register():
    bpy.utils.register_class(DIYRenderEngine)


def unregister():
    bpy.utils.unregister_class(DIYRenderEngine)