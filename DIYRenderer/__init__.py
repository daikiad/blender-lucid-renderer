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
    print(f"[DIYRenderer] Exporting scene to: {path}")
    mesh_count = 0
    tri_count = 0
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
                mesh_count += 1
                tri_count += len(tris)
                if mesh_count <= 2:  # Print first 2 meshes for debugging
                    print(f"[DIYRenderer] Exported mesh '{obj.name}': {len(verts_world)} verts, {len(tris)} tris")
                    if len(verts_world) > 0:
                        print(f"[DIYRenderer]   First vertex: ({verts_world[0].x:.3f}, {verts_world[0].y:.3f}, {verts_world[0].z:.3f})")
            f.write("(end)\n")
        print(f"[DIYRenderer] Export complete: {mesh_count} meshes, {tri_count} total triangles")
    except Exception as e:
        print("[DIYRenderer] Scene export failed:", e)
        return None
    return path

def compute_camera_params(scene, width, height):
    cam = scene.camera
    if not cam:
        print("[DIYRenderer] WARNING: No camera in scene!")
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
    print(f"[DIYRenderer] Camera: pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
          f"dir=({forward.x:.3f}, {forward.y:.3f}, {forward.z:.3f}), "
          f"up=({up.x:.3f}, {up.y:.3f}, {up.z:.3f}), fov={fov_deg:.1f}")
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
           '--fov', str(cam_params['fov']),
           '--debug',
           '--disable-aabb']  # Temporarily disable AABB to test raw triangle intersections
    print(f"[DIYRenderer] Calling external renderer: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if proc.stderr:
            print(f"[DIYRenderer] External renderer stderr:\n{proc.stderr[:2000]}")
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
        # Reset progressive rendering state
        self.viewport_last_update = 0
        self.viewport_resolution_level = 0  # Start from lowest resolution

    def view_draw(self, context, depsgraph):
        region = context.region
        width = region.width
        height = region.height
        import gpu
        from gpu_extras.presets import draw_texture_2d  # used for drawing texture
        import time
        
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        if not hasattr(self, 'viewport_last_update'):
            self.viewport_last_update = 0
        if not hasattr(self, 'viewport_resolution_level'):
            self.viewport_resolution_level = 0
            
        self.frame_counter += 1
        current_time = time.time()
        time_since_update = current_time - self.viewport_last_update
        
        # Progressive resolution: start low, increase when idle
        # Level 0: 1/16 (very fast, update every frame)
        # Level 1: 1/8 (fast, update every 5 frames or 0.5s idle)
        # Level 2: 1/4 (medium, 1s idle)
        # Level 3: 1/2 (high, 2s idle)
        resolution_configs = [
            {'scale': 16, 'update_frames': 1, 'idle_time': 0.0},
            {'scale': 8, 'update_frames': 3, 'idle_time': 0.3},
            {'scale': 4, 'update_frames': 10, 'idle_time': 1.0},
            {'scale': 2, 'update_frames': 30, 'idle_time': 2.0},
        ]
        
        # Determine target resolution level based on idle time
        target_level = 0
        for i, config in enumerate(resolution_configs):
            if time_since_update >= config['idle_time']:
                target_level = i
        
        # Progressive upgrade: gradually increase quality
        if time_since_update > resolution_configs[min(self.viewport_resolution_level + 1, len(resolution_configs) - 1)]['idle_time']:
            if self.viewport_resolution_level < len(resolution_configs) - 1:
                self.viewport_resolution_level += 1
                target_level = self.viewport_resolution_level
        
        # Reset to lowest resolution on camera move (detected by frequent updates)
        if time_since_update < 0.1:  # Moving
            target_level = 0
            self.viewport_resolution_level = 0
        
        config = resolution_configs[target_level]
        render_width = max(40, width // config['scale'])
        render_height = max(30, height // config['scale'])
        
        needs_update = (
            not hasattr(self, 'texture') or self.texture is None or
            not hasattr(self, 'viewport_pixels_cache') or self.viewport_pixels_cache is None or
            self.texture.width != render_width or self.texture.height != render_height or
            self.frame_counter % config['update_frames'] == 0 or
            time_since_update > config['idle_time']
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