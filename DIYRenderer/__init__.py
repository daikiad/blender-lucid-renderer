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
import threading
import queue

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

def call_external_renderer(scene_file, tile_x, tile_y, tile_w, tile_h, full_w, full_h, cam_params, mode='raytrace'):
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
           '--mode', mode]
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
    bl_use_shading_nodes_custom = False  # Use standard Blender nodes

    def _init_async_render(self):
        """Lazy initialization of async rendering infrastructure"""
        if not hasattr(self, 'render_queue'):
            self.render_queue = queue.Queue(maxsize=1)
            self.result_queue = queue.Queue()
            self.render_thread = None
            self.stop_thread = False
            self.rendering_in_progress = False
            self.high_res_complete = False

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

    def async_render_viewport(self, job_data):
        """Background thread function to render viewport without blocking UI"""
        while not self.stop_thread:
            try:
                # Get latest job, discard old ones
                depsgraph, cam_params, render_width, render_height, job_id = job_data
                scene_file = export_scene_to_file(depsgraph)
                if scene_file:
                    ext_pixels = call_external_renderer(
                        scene_file, 0, 0, render_width, render_height, 
                        render_width, render_height, cam_params
                    )
                    try:
                        if os.path.isfile(scene_file):
                            os.remove(scene_file)
                    except Exception:
                        pass
                    # Put result in queue
                    try:
                        self.result_queue.put_nowait({
                            'pixels': ext_pixels,
                            'width': render_width,
                            'height': render_height,
                            'job_id': job_id
                        })
                    except queue.Full:
                        pass  # Discard if queue full
                # Check for new job
                try:
                    job_data = self.render_queue.get(timeout=0.1)
                except queue.Empty:
                    break  # No more jobs, exit thread
            except Exception as e:
                print(f"[DIYRenderer] Async render error: {e}")
                break
    
    def view_update(self, context, depsgraph):
        self._init_async_render()
        if hasattr(self, 'texture'):
            try:
                del self.texture
            except Exception:
                pass
            self.texture = None
        self.viewport_pixels_cache = None
        # Mark that scene changed - reset to low resolution
        import time
        self.viewport_last_change_time = time.time()
        self.high_res_complete = False  # Need to render high-res again
        # Cancel any pending render by clearing queue
        try:
            while not self.render_queue.empty():
                self.render_queue.get_nowait()
        except queue.Empty:
            pass

    def view_draw(self, context, depsgraph):
        self._init_async_render()
        region = context.region
        width = region.width
        height = region.height
        import gpu
        from gpu_extras.presets import draw_texture_2d  # used for drawing texture
        import time
        
        if not hasattr(self, 'viewport_last_change_time'):
            self.viewport_last_change_time = time.time()
        if not hasattr(self, 'last_render_width'):
            self.last_render_width = 0
            self.last_render_height = 0
            
        current_time = time.time()
        time_since_change = current_time - self.viewport_last_change_time
        
        # Simple two-level system: low res when moving, high res when idle
        if time_since_change < 1.0:
            # Moving: use low resolution
            render_width = max(40, width // 16)
            render_height = max(30, height // 16)
            update_interval = 1  # Update every frame when moving
        else:
            # Idle for 1+ seconds: use full resolution
            render_width = max(100, width)
            render_height = max(100, height)
            update_interval = 10  # Update every 10 frames when idle
        
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        self.frame_counter += 1
        
        # Check if we need to update
        resolution_changed = (self.last_render_width != render_width or 
                             self.last_render_height != render_height)
        needs_update = (
            not hasattr(self, 'texture') or self.texture is None or
            resolution_changed or
            self.frame_counter % update_interval == 0
        )
        if needs_update:
            self.last_render_width = render_width
            self.last_render_height = render_height
            pixels = None
            region_data = context.region_data
            if region_data is not None:
                cam_params = {
                    'pos': region_data.view_matrix.inverted().translation,
                    'dir': (region_data.view_matrix.inverted().to_3x3() @ Vector((0,0,-1))).normalized(),
                    'up': (region_data.view_matrix.inverted().to_3x3() @ Vector((0,1,0))).normalized(),
                    'fov': 60.0 if getattr(region_data, 'is_perspective', True) else 5.0
                }
                # Submit async render job (non-blocking)
                if not hasattr(self, 'job_counter'):
                    self.job_counter = 0
                self.job_counter += 1
                job_data = (depsgraph, cam_params, render_width, render_height, self.job_counter)
                
                # Clear old job and submit new one
                try:
                    if not self.render_queue.empty():
                        self.render_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.render_queue.put_nowait(job_data)
                    self.rendering_in_progress = True
                    # Start thread if not running
                    if self.render_thread is None or not self.render_thread.is_alive():
                        self.render_thread = threading.Thread(
                            target=self.async_render_viewport,
                            args=(job_data,),
                            daemon=True
                        )
                        self.render_thread.start()
                except queue.Full:
                    pass  # Skip if queue full
        
        # Check for completed renders (non-blocking)
        has_new_result = False
        try:
            result = self.result_queue.get_nowait()
            ext_pixels = result['pixels']
            result_width = result['width']
            result_height = result['height']
            if ext_pixels and len(ext_pixels) == result_width * result_height:
                pixels = ext_pixels
                render_width = result_width
                render_height = result_height
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
                has_new_result = True
                self.rendering_in_progress = False
                # Check if this was a high-resolution render
                if render_width >= width and render_height >= height:
                    self.high_res_complete = True
        except queue.Empty:
            pass  # No result yet, use existing texture
        
        # Request redraw if rendering is in progress or just got new result
        # This keeps the viewport updating until high-res render is complete
        if self.rendering_in_progress or (has_new_result and not self.high_res_complete):
            # Tag viewport for redraw to check for updates
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        
        # Draw the texture to fill the viewport
        if hasattr(self, 'texture') and self.texture is not None:
            draw_texture_2d(self.texture, (0, 0), width, height)


def register():
    bpy.utils.register_class(DIYRendererPreferences)
    bpy.utils.register_class(DIYRenderEngine)
    
    # Add our engine to all relevant panel compatibility
    try:
        from bl_ui import (
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_render,
            properties_output,
            properties_data_modifier,
        )
        
        modules = [
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_render,
            properties_output,
            properties_data_modifier,
        ]
        
        for module in modules:
            for panel_name in dir(module):
                if panel_name.startswith(('MATERIAL_PT_', 'DATA_PT_', 'WORLD_PT_', 
                                         'RENDER_PT_', 'OUTPUT_PT_', 'EEVEE_', 
                                         'CYCLES_', 'NODE_')):
                    panel = getattr(module, panel_name, None)
                    if panel and hasattr(panel, 'COMPAT_ENGINES'):
                        panel.COMPAT_ENGINES.add('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not register panels: {e}")


def unregister():
    # Remove our engine from panel compatibility
    try:
        from bl_ui import (
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_render,
            properties_output,
            properties_data_modifier,
        )
        
        modules = [
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_render,
            properties_output,
            properties_data_modifier,
        ]
        
        for module in modules:
            for panel_name in dir(module):
                if panel_name.startswith(('MATERIAL_PT_', 'DATA_PT_', 'WORLD_PT_', 
                                         'RENDER_PT_', 'OUTPUT_PT_', 'EEVEE_', 
                                         'CYCLES_', 'NODE_')):
                    panel = getattr(module, panel_name, None)
                    if panel and hasattr(panel, 'COMPAT_ENGINES') and 'DIY_RENDER_MINIMAL' in panel.COMPAT_ENGINES:
                        panel.COMPAT_ENGINES.remove('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not unregister panels: {e}")
    
    bpy.utils.unregister_class(DIYRenderEngine)
    bpy.utils.unregister_class(DIYRendererPreferences)