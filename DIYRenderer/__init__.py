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
from mathutils import Vector
import os
import tempfile
import subprocess
import math

def _get_prefs_entry():
    return bpy.context.preferences.addons.get(__name__)

def _get_prefs():
    entry = _get_prefs_entry()
    if entry is not None:
        return getattr(entry, 'preferences', None)
    return None

class DIYRendererPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    external_renderer_path = bpy.props.StringProperty(name="External Renderer Path", description="Path to compiled external C++ renderer binary (diyrt)", default="", subtype='FILE_PATH')
    scene_export_directory = bpy.props.StringProperty(name="Scene Export Temp Dir", description="Directory to write temporary exported scene files", default="", subtype='DIR_PATH')
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "external_renderer_path")
        layout.prop(self, "scene_export_directory")

def find_external_binary():
    # 1) Explicit preference path
    prefs = _get_prefs()
    if prefs:
        renderer_path = getattr(prefs, 'external_renderer_path', '')
        if renderer_path and isinstance(renderer_path, str) and os.path.isfile(renderer_path):
            return renderer_path
    # 2) Environment variable override
    env_path = os.environ.get('DIY_RENDERER_BIN')
    if env_path and os.path.isfile(env_path):
        return env_path
    # 3) Common relative build locations
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(addon_dir, '..'))
    candidates = [
        os.path.join(project_root, 'DIYRenderer', 'cpp_renderer', 'build', 'diyrt'),
        os.path.join(project_root, 'DIYRenderer', 'cpp_renderer', 'build', 'Release', 'diyrt'),
        os.path.join(project_root, 'DIYRenderer', 'cpp_renderer', 'build', 'Debug', 'diyrt'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # 4) If not found, emit debug info once
    if not hasattr(find_external_binary, '_warned'):
        find_external_binary._warned = True
        print('[DIYRenderer] External renderer binary not found. Checked:')
        for c in candidates:
            print('   -', c)
        print('Set Add-on Preferences path or env DIY_RENDERER_BIN.')
    return None

def export_scene_to_file(depsgraph):
    """Export evaluated meshes to a temporary scene file matching external renderer format."""
    prefs = _get_prefs()
    base_dir = None
    if prefs:
        export_dir = getattr(prefs, 'scene_export_directory', '')
        if export_dir and isinstance(export_dir, str) and os.path.isdir(export_dir):
            base_dir = export_dir
    if base_dir is None:
        base_dir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(prefix="diy_scene_", suffix=".txt", dir=base_dir)
    os.close(fd)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for obj_instance in depsgraph.object_instances:
                obj = obj_instance.object
                if obj.type != 'MESH':
                    continue
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                if not mesh:
                    continue
                verts_world = []
                mw = obj_instance.matrix_world
                for v in mesh.vertices:
                    co = mw @ v.co
                    verts_world.append(co)
                # Triangulate polygons (fan)
                tris = []
                for poly in mesh.polygons:
                    v_indices = list(poly.vertices)
                    if len(v_indices) < 3:
                        continue
                    for i in range(1, len(v_indices) - 1):
                        tris.append((v_indices[0], v_indices[i], v_indices[i+1]))
                f.write(f"mesh {obj.name} {len(verts_world)} {len(tris)}\n")
                for co in verts_world:
                    f.write(f"v {co.x} {co.y} {co.z}\n")
                for (a,b,c) in tris:
                    f.write(f"t {a} {b} {c}\n")
                eval_obj.to_mesh_clear()
            f.write("(end)\n")
    except Exception as e:
        print("[DIYRenderer] Scene export failed:", e)
        return None
    return path

def compute_camera_params(scene, width, height):
    cam = scene.camera
    if not cam:
        return None
    cam_matrix = cam.matrix_world
    pos = cam_matrix.translation
    forward = cam_matrix.to_3x3() @ Vector((0,0,-1))
    up = cam_matrix.to_3x3() @ Vector((0,1,0))
    forward.normalize()
    up.normalize()
    sensor_w = cam.data.sensor_width
    lens = cam.data.lens
    fov_rad = 2.0 * math.atan(sensor_w / (2.0 * lens))
    fov_deg = math.degrees(fov_rad)
    return {
        'pos': pos,
        'dir': forward,
        'up': up,
        'fov': fov_deg
    }

def call_external_renderer(scene_file, tile_x, tile_y, tile_w, tile_h, full_w, full_h, cam_params):
    binary = find_external_binary()
    if not binary:
        print("[DIYRenderer] External binary not found. Falling back to internal rendering.")
        return None
    cmd = [binary,
           '--scene', scene_file,
           '--tile', str(tile_x), str(tile_y), str(tile_w), str(tile_h),
           '--full', str(full_w), str(full_h),
           '--campos', str(cam_params['pos'].x), str(cam_params['pos'].y), str(cam_params['pos'].z),
           '--camdir', str(cam_params['dir'].x), str(cam_params['dir'].y), str(cam_params['dir'].z),
           '--camup', str(cam_params['up'].x), str(cam_params['up'].y), str(cam_params['up'].z),
           '--fov', str(cam_params['fov'])]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        print('[DIYRenderer] External renderer invocation failed:', e)
        return None
    lines = proc.stdout.strip().splitlines()
    if len(lines) != tile_w * tile_h:
        print('[DIYRenderer] Unexpected line count from external renderer', len(lines), 'expected', tile_w * tile_h)
    pixels = []
    for ln in lines:
        try:
            r,g,b,a = map(float, ln.split())
            pixels.append([r,g,b,a])
        except ValueError:
            pixels.append([1.0,0.0,1.0,1.0])  # error magenta
    return pixels


class DIYRenderEngine(bpy.types.RenderEngine):
    bl_idname = "DIY_RENDER_MINIMAL"
    bl_label = "DIY Renderer (Minimal)"
    bl_use_preview = True
    bl_use_shading_nodes = True
    # bl_use_shading_nodes_custom = False

    def _render_gradient(self, width, height):
        pixels = []
        for y in range(height):
            fy = y / (height - 1) if height > 1 else 0.0
            for x in range(width):
                fx = x / (width - 1) if width > 1 else 0.0
                pixels.append([fx, fy, 0.2, 1.0])
        return pixels

    def render(self, depsgraph):
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        print(f"[DIYRenderer] Starting render ({width} x {height})")
        cam_params = compute_camera_params(scene, width, height)
        external_pixels = None
        if cam_params:
            scene_file = export_scene_to_file(depsgraph)
            if scene_file:
                external_pixels = call_external_renderer(scene_file, 0, 0, width, height, width, height, cam_params)
                try:
                    if scene_file and os.path.isfile(scene_file):
                        os.remove(scene_file)
                except Exception:
                    pass
        result = self.begin_result(0, 0, width, height)
        combined = result.layers[0].passes["Combined"]
        if external_pixels and len(external_pixels) == width * height:
            combined.rect = external_pixels
            print("[DIYRenderer] External render complete")
        else:
            combined.rect = self._render_gradient(width, height)
            print("[DIYRenderer] External render failed; gradient fallback")
        self.end_result(result)

    def view_update(self, context, depsgraph):
        if hasattr(self, 'texture'):
            try:
                del self.texture
            except Exception:
                pass
            self.texture = None
        self.viewport_pixels_cache = None

    def view_draw(self, context, depsgraph):
        region = context.region
        width = region.width
        height = region.height
        import gpu
        from gpu_extras.presets import draw_texture_2d  # used for drawing texture
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        self.frame_counter += 1
        render_width = max(80, width // 8)
        render_height = max(60, height // 8)
        needs_update = (
            not hasattr(self, 'texture') or self.texture is None or
            not hasattr(self, 'viewport_pixels_cache') or self.viewport_pixels_cache is None or
            self.texture.width != render_width or self.texture.height != render_height or
            self.frame_counter % 30 == 0
        )
        if needs_update:
            pixels = None
            region_data = context.region_data
            if region_data is not None:
                cam_params = {
                    'pos': region_data.view_matrix.inverted().translation,
                    'dir': (region_data.view_matrix.inverted().to_3x3() @ Vector((0,0,-1))).normalized(),
                    'up': (region_data.view_matrix.inverted().to_3x3() @ Vector((0,1,0))).normalized(),
                    'fov': 60.0 if getattr(region_data, 'is_perspective', True) else 5.0
                }
                scene_file = export_scene_to_file(depsgraph)
                if scene_file:
                    ext_pixels = call_external_renderer(scene_file, 0, 0, render_width, render_height, render_width, render_height, cam_params)
                    try:
                        if os.path.isfile(scene_file):
                            os.remove(scene_file)
                    except Exception:
                        pass
                    if ext_pixels and len(ext_pixels) == render_width * render_height:
                        pixels = ext_pixels
            if pixels is None:
                pixels = self._render_gradient(render_width, render_height)
            self.viewport_pixels_cache = pixels
            flat = [c for px in pixels for c in px]
            buffer = gpu.types.Buffer('FLOAT', render_width * render_height * 4, flat)
            if hasattr(self, 'texture') and self.texture is not None:
                try:
                    del self.texture
                except Exception:
                    pass
            self.texture = gpu.types.GPUTexture((render_width, render_height), format='RGBA16F', data=buffer)
        
        # Draw the texture to fill the viewport
        if hasattr(self, 'texture') and self.texture is not None:
            draw_texture_2d(self.texture, (0, 0), width, height)


def register():
    bpy.utils.register_class(DIYRendererPreferences)
    bpy.utils.register_class(DIYRenderEngine)


def unregister():
    bpy.utils.unregister_class(DIYRenderEngine)
    bpy.utils.unregister_class(DIYRendererPreferences)