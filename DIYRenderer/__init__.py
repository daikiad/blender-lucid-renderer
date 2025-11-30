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
from mathutils import Vector
import os
import tempfile
import subprocess
import math
import threading
import queue
import json

def _get_prefs_entry():
    """Get addon preferences entry from Blender"""
    return bpy.context.preferences.addons.get(__name__)

def _get_prefs():
    """Get addon preferences object"""
    entry = _get_prefs_entry()
    if entry is not None:
        return getattr(entry, 'preferences', None)
    return None

class DIYRendererPreferences(bpy.types.AddonPreferences):
    """Addon preferences for configuring external renderer path"""
    bl_idname = __name__
    external_renderer_path = bpy.props.StringProperty(
        name="External Renderer Path", 
        description="Path to compiled external C++ renderer binary (diyrt)", 
        default="", 
        subtype='FILE_PATH'
    )
    scene_export_directory = bpy.props.StringProperty(
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
        
        col = layout.column(align=True)
        col.prop(diy, "samples")
        col.prop(diy, "viewport_samples")
        
        layout.separator()
        layout.prop(diy, "debug_mode")

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


def serialize_socket_value(socket):
    """
    Serialize a node socket's default value to JSON-compatible format.
    
    Handles different socket types found in Blender's node system:
    - RGBA sockets → [r, g, b, a] (4-element list)
    - VECTOR sockets → [x, y, z] (3-element list)
    - VALUE sockets → float or int
    - BOOLEAN sockets → true/false
    - STRING sockets → string value
    
    Args:
        socket: Blender NodeSocket object
        
    Returns:
        JSON-compatible value (list, number, boolean, or string)
        None if socket has no default_value attribute
        
    Technical Note:
    This function is critical for exporting node graph data correctly.
    The VEC4 vs VEC3 distinction in the JSON must be preserved,
    as the C++ renderer uses different union fields for each type.
    """
    if not hasattr(socket, 'default_value'):
        return None
    
    val = socket.default_value
    
    # Color/RGBA socket (VEC4 in C++)
    if hasattr(val, '__len__') and len(val) == 4:
        return list(val)
    # Vector socket (VEC3 in C++)
    elif hasattr(val, '__len__') and len(val) == 3:
        return list(val)
    # Float/Int socket (FLOAT in C++)
    elif isinstance(val, (int, float)):
        return val
    # Boolean (BOOL in C++)
    elif isinstance(val, bool):
        return val
    # String (STRING in C++)
    elif isinstance(val, str):
        return val
    else:
        return None


def serialize_node_tree(node_tree):
    """
    Serialize a complete Blender node tree (material shader graph) to JSON.
    
    This function captures the entire node graph including:
    - All nodes (Principled BSDF, Mix, Emission, etc.)
    - Socket default values (colors, vectors, floats)
    - Node connections (links between sockets)
    - Node properties (blend modes, interpolation, etc.)
    
    Args:
        node_tree: Blender NodeTree object (from material.node_tree)
        
    Returns:
        dict: Serialized node tree with 'nodes' and 'links' keys
        None: If node_tree is None or invalid
        
    Structure:
        {
            'nodes': [
                {
                    'name': 'Principled BSDF',
                    'type': 'ShaderNodeBsdfPrincipled',
                    'label': 'Principled BSDF',
                    'inputs': [...socket data...],
                    'outputs': [...socket data...],
                    'properties': {...node properties...}
                },
                ...
            ],
            'links': [
                {
                    'from_node': 'RGB',
                    'from_socket': 'Color',
                    'to_node': 'Principled BSDF',
                    'to_socket': 'Base Color'
                },
                ...
            ]
        }
    
    Technical Note:
    - Uses RNA reflection to access node properties dynamically
    - Converts Blender types (Vector, Color) to JSON-serializable lists
    - Handles object references by storing name strings
    - Skips read-only and system properties
    """
    if not node_tree:
        return None
    
    result = {
        'nodes': [],
        'links': []
    }
    
    # Serialize all nodes
    for node in node_tree.nodes:
        node_data = {
            'name': node.name,
            'type': node.bl_idname,  # e.g., 'ShaderNodeBsdfPrincipled'
            'label': node.label,
            'location': [node.location.x, node.location.y],
            'properties': {},
            'inputs': [],
            'outputs': []
        }
        
        # Serialize node properties using RNA reflection
        # This captures blend modes, interpolation settings, etc.
        for prop in node.bl_rna.properties:
            if prop.is_readonly or prop.identifier in ('rna_type', 'inputs', 'outputs'):
                continue
            try:
                value = getattr(node, prop.identifier)
                # Convert Blender types to JSON-serializable types
                if hasattr(value, '__len__') and not isinstance(value, str):
                    value = list(value)
                elif hasattr(value, 'name'):  # Object reference
                    value = value.name
                node_data['properties'][prop.identifier] = value
            except Exception:
                pass  # Skip properties that can't be serialized
        
        # Serialize input sockets
        for i, socket in enumerate(node.inputs):
            socket_data = {
                'index': i,
                'name': socket.name,
                'type': socket.type,
                'default_value': serialize_socket_value(socket),
                'is_linked': socket.is_linked
            }
            # Debug: Print Base Color socket values
            if socket.name == "Base Color" and node.bl_idname == "ShaderNodeBsdfPrincipled":
                print(f"[DIYRenderer] Exporting Base Color socket: default_value={socket_data['default_value']}, is_linked={socket.is_linked}")
            node_data['inputs'].append(socket_data)
        
        # Serialize output sockets
        for i, socket in enumerate(node.outputs):
            socket_data = {
                'index': i,
                'name': socket.name,
                'type': socket.type,
                'is_linked': socket.is_linked
            }
            node_data['outputs'].append(socket_data)
        
        result['nodes'].append(node_data)
    
    # Serialize all links (connections between sockets)
    for link in node_tree.links:
        link_data = {
            'from_node': link.from_node.name,
            'from_socket': link.from_socket.name,
            'to_node': link.to_node.name,
            'to_socket': link.to_socket.name
        }
        result['links'].append(link_data)
    
    return result


def get_material_properties(obj):
    """
    Extract complete material node tree from object.
    Returns both legacy simple properties and full node graph.
    """
    if not obj.data or not hasattr(obj.data, 'materials') or not obj.data.materials:
        return None
    
    mat = obj.data.materials[0]  # Use first material slot
    if not mat:
        return None
    
    result = {
        'name': mat.name,
        'use_nodes': mat.use_nodes,
        'node_tree': None,
        'legacy_properties': {}  # Fallback for simple rendering
    }
    
    # If material uses nodes, serialize the complete node tree
    if mat.use_nodes and mat.node_tree:
        result['node_tree'] = serialize_node_tree(mat.node_tree)
    
    # Also extract legacy simple properties for backward compatibility
    # Find Principled BSDF node for fallback
    principled = None
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break
    
    if principled:
        # Base Color
        if 'Base Color' in principled.inputs:
            base_color_input = principled.inputs['Base Color']
            if base_color_input.is_linked:
                result['legacy_properties']['base_color'] = [0.8, 0.8, 0.8]  # Default if connected
            else:
                color = base_color_input.default_value
                result['legacy_properties']['base_color'] = [color[0], color[1], color[2]]
        else:
            result['legacy_properties']['base_color'] = [0.8, 0.8, 0.8]
        
        # Metallic
        if 'Metallic' in principled.inputs:
            metallic_input = principled.inputs['Metallic']
            result['legacy_properties']['metallic'] = metallic_input.default_value if not metallic_input.is_linked else 0.0
        else:
            result['legacy_properties']['metallic'] = 0.0
        
        # Roughness
        if 'Roughness' in principled.inputs:
            roughness_input = principled.inputs['Roughness']
            result['legacy_properties']['roughness'] = roughness_input.default_value if not roughness_input.is_linked else 0.5
        else:
            result['legacy_properties']['roughness'] = 0.5
        
        # Emission
        emission_color = [0.0, 0.0, 0.0]
        emission_strength = 0.0
        if 'Emission Color' in principled.inputs:
            emission_input = principled.inputs['Emission Color']
            if not emission_input.is_linked:
                color = emission_input.default_value
                emission_color = [color[0], color[1], color[2]]
        elif 'Emission' in principled.inputs:  # Older Blender versions
            emission_input = principled.inputs['Emission']
            if not emission_input.is_linked:
                color = emission_input.default_value
                emission_color = [color[0], color[1], color[2]]
        
        if 'Emission Strength' in principled.inputs:
            strength_input = principled.inputs['Emission Strength']
            emission_strength = strength_input.default_value if not strength_input.is_linked else 0.0
        
        result['legacy_properties']['emission'] = [
            emission_color[0] * emission_strength,
            emission_color[1] * emission_strength,
            emission_color[2] * emission_strength
        ]
        
        # Debug: print emission values
        print(f"[DIYRenderer] Material '{mat.name}': emission_color={emission_color}, strength={emission_strength}, final={result['legacy_properties']['emission']}")
    else:
        # No Principled BSDF found - use defaults
        result['legacy_properties'] = {
            'base_color': [0.8, 0.8, 0.8],
            'metallic': 0.0,
            'roughness': 0.5,
            'emission': [0.0, 0.0, 0.0]
        }
    
    return result

def export_scene_to_json(depsgraph):
    """
    Export complete scene to JSON format.
    
    Exports the entire scene including:
    - Mesh geometry (vertices, triangles, normals)
    - Material node trees (full node graph with connections)
    - Mesh attributes (vertex colors, UVs, custom attributes)
    - World transforms (objects positioned in world space)
    
    Args:
        depsgraph: Blender's dependency graph (evaluated scene state)
        
    Returns:
        str: Path to the exported JSON file
        
    Technical Details:
    - Uses depsgraph for evaluated geometry (modifiers applied)
    - Triangulates all faces for renderer compatibility
    - Transforms vertices to world space using matrix_world
    - Exports both legacy properties and full node trees
    - Custom attributes from Geometry Nodes are preserved
    
    File Location:
    - Default: System temp directory (tempfile.gettempdir())
    - Configurable: Via addon preferences 'scene_export_directory'
    - Filename: 'diy_scene_debug.json' (fixed for debugging)
    """
    prefs = _get_prefs()
    base_dir = tempfile.gettempdir()
    if prefs:
        export_dir = getattr(prefs, 'scene_export_directory', '')
        if export_dir and isinstance(export_dir, str) and os.path.isdir(export_dir):
            base_dir = export_dir
    
    # Use fixed path for debugging
    path = os.path.join(base_dir, "diy_scene_debug.json")
    print(f"[DIYRenderer] Exporting scene to: {path}")
    
    scene_data = {
        "version": "1.0",
        "meshes": []
    }
    
    # Iterate through all objects in the evaluated scene
    for obj_instance in depsgraph.object_instances:
        obj = obj_instance.object
        if obj.type != 'MESH':
            continue
        
        # Get evaluated mesh (with modifiers applied)
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if not mesh:
            continue
        
        # Transform vertices to world space
        mw = obj_instance.matrix_world
        vertices = []
        for v in mesh.vertices:
            co = mw @ v.co  # Apply world transformation
            vertices.append([co.x, co.y, co.z])
        
        # Triangulate and collect face data
        # Renderer expects triangles only (no quads/ngons)
        triangles = []
        for poly in mesh.polygons:
            v_indices = list(poly.vertices)
            if len(v_indices) < 3:
                continue
            # Fan triangulation: split polygon into triangles
            for i in range(1, len(v_indices) - 1):
                triangles.append([v_indices[0], v_indices[i], v_indices[i+1]])
        
        # Extract material properties (now includes full node tree)
        mat_props = get_material_properties(obj)
        
        # Prepare material data with both legacy and full node graph
        if mat_props:
            material = {
                "name": mat_props['name'],
                "use_nodes": mat_props['use_nodes'],
                # Legacy properties for simple rendering (backward compatibility)
                "base_color": mat_props['legacy_properties'].get('base_color', [0.8, 0.8, 0.8]),
                "metallic": mat_props['legacy_properties'].get('metallic', 0.0),
                "roughness": mat_props['legacy_properties'].get('roughness', 0.5),
                "emission": mat_props['legacy_properties'].get('emission', [0.0, 0.0, 0.0]),
                # Full node tree (for advanced rendering)
                "node_tree": mat_props['node_tree']
            }
            print(f"[DIYRenderer] Mesh '{obj.name}': emission={material['emission']}, base_color={material['base_color']}")
        else:
            material = {
                "name": "default",
                "use_nodes": False,
                "base_color": [0.8, 0.8, 0.8],
                "metallic": 0.0,
                "roughness": 0.5,
                "emission": [0.0, 0.0, 0.0],
                "node_tree": None
            }
        
        # Extract geometry node attributes
        attributes = {}
        
        # Vertex colors
        if mesh.vertex_colors:
            color_layer = mesh.vertex_colors.active
            if color_layer:
                vertex_colors = []
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        color = color_layer.data[loop_idx].color
                        vertex_colors.append([color[0], color[1], color[2], color[3]])
                attributes["vertex_color"] = vertex_colors
        
        # UVs
        if mesh.uv_layers:
            uv_layer = mesh.uv_layers.active
            if uv_layer:
                uvs = []
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        uv = uv_layer.data[loop_idx].uv
                        uvs.append([uv.x, uv.y])
                attributes["uv"] = uvs
        
        # Custom attributes from Geometry Nodes
        if hasattr(mesh, 'attributes'):
            for attr in mesh.attributes:
                if attr.name.startswith('.'):  # Skip internal attributes
                    continue
                attr_name = attr.name
                attr_data = []
                
                # Handle different attribute domains and data types
                if attr.domain == 'POINT':  # Vertex attribute
                    if attr.data_type == 'FLOAT':
                        attr_data = [v.value for v in attr.data]
                    elif attr.data_type == 'FLOAT_VECTOR':
                        attr_data = [[v.vector[0], v.vector[1], v.vector[2]] for v in attr.data]
                    elif attr.data_type == 'FLOAT_COLOR':
                        attr_data = [[v.color[0], v.color[1], v.color[2], v.color[3]] for v in attr.data]
                    elif attr.data_type == 'INT':
                        attr_data = [v.value for v in attr.data]
                
                if attr_data:
                    attributes[f"custom_{attr_name}"] = {
                        "domain": attr.domain,
                        "data_type": attr.data_type,
                        "data": attr_data
                    }
        
        mesh_data = {
            "name": obj.name,
            "vertices": vertices,
            "triangles": triangles,
            "material": material,
            "attributes": attributes
        }
        
        scene_data["meshes"].append(mesh_data)
        eval_obj.to_mesh_clear()
    
    # Write JSON file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2)
    
    print(f"[DIYRenderer] Exported {len(scene_data['meshes'])} meshes to JSON: {path}")
    return path

def export_scene_to_file(depsgraph):
    """Export evaluated meshes - delegates to JSON export."""
    try:
        return export_scene_to_json(depsgraph)
    except Exception as e:
        print("[DIYRenderer] Scene export failed:", e)
        return None

def linear_to_srgb(c):
    """
    Convert linear color value to sRGB gamma corrected value.
    
    Uses the standard sRGB transfer function:
    - Linear segment for c <= 0.0031308
    - Power curve for c > 0.0031308
    
    Args:
        c: Linear color value (0.0-1.0)
    
    Returns:
        float: sRGB gamma corrected value (0.0-1.0)
    
    Note: Currently unused - Blender handles color management internally
    """
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * (c ** (1.0/2.4)) - 0.055

def apply_gamma_correction(pixels):
    """
    Apply sRGB gamma correction to linear pixel values.
    
    Args:
        pixels: List of [r, g, b, a] pixel values in linear space
        
    Returns:
        list: Same pixels with sRGB gamma applied to RGB channels
        
    Note: Currently unused - Blender handles color management internally.
          We render in linear space and Blender converts to sRGB for display.
    """
    corrected = []
    for pixel in pixels:
        r, g, b, a = pixel
        # Apply gamma correction to RGB, leave alpha as-is
        r_srgb = linear_to_srgb(max(0.0, min(1.0, r)))
        g_srgb = linear_to_srgb(max(0.0, min(1.0, g)))
        b_srgb = linear_to_srgb(max(0.0, min(1.0, b)))
        corrected.append([r_srgb, g_srgb, b_srgb, a])
    return corrected

def compute_camera_params(scene, width, height):
    """
    Compute camera parameters for external renderer.
    
    Extracts camera properties from Blender's camera object and converts
    them to the format expected by the C++ renderer.
    
    Args:
        scene: Blender scene object containing camera
        width: Render width in pixels
        height: Render height in pixels
        
    Returns:
        dict: Camera parameters with keys:
            - 'pos': Camera position [x, y, z]
            - 'forward': Camera forward direction (normalized)
            - 'up': Camera up vector (normalized)
            - 'fov': Field of view in degrees (horizontal)
        None: If scene has no camera
        
    Technical Details:
    - Extracts position from matrix_world.translation
    - Computes forward as -Z axis in camera space (Blender convention)
    - Computes up as +Y axis in camera space
    - Calculates FOV from sensor width and focal length:
      fov = 2 * atan(sensor_width / (2 * focal_length))
    
    Coordinate System:
    - Blender uses right-handed Y-up coordinate system
    - Camera looks down -Z axis
    - Renderer expects world-space coordinates
    """
    cam = scene.camera
    if not cam:
        print("[DIYRenderer] WARNING: No camera in scene!")
        return None
    
    # Extract camera world transformation
    cam_matrix = cam.matrix_world
    pos = cam_matrix.translation
    
    # Camera coordinate system: forward = -Z, up = +Y (in camera space)
    forward = cam_matrix.to_3x3() @ Vector((0,0,-1))
    up = cam_matrix.to_3x3() @ Vector((0,1,0))
    forward.normalize()
    up.normalize()
    
    # Calculate field of view from sensor and lens properties
    sensor_w = cam.data.sensor_width  # mm
    lens = cam.data.lens              # mm (focal length)
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

def call_external_renderer(scene_file, tile_x, tile_y, tile_w, tile_h, full_w, full_h, cam_params, mode='raytrace', samples=1, debug_mode=None):
    """
    Call external C++ renderer.
    
    Args:
        debug_mode: Override render mode for debugging. Options:
            - None or 'raytrace': Full path tracing
            - 'normal': Show surface normals
            - 'albedo': Show base color (Principled BSDF Base Color)
            - 'emission': Show emission values
    """
    binary = find_external_binary()
    if not binary:
        print("[DIYRenderer] External binary not found. Falling back to internal rendering.")
        return None
    
    # Use debug_mode if specified, otherwise use provided mode
    render_mode = debug_mode if debug_mode else mode
    
    cmd = [binary,
           '--scene', scene_file,
           '--tile', str(tile_x), str(tile_y), str(tile_w), str(tile_h),
           '--full', str(full_w), str(full_h),
           '--campos', str(cam_params['pos'].x), str(cam_params['pos'].y), str(cam_params['pos'].z),
           '--camdir', str(cam_params['dir'].x), str(cam_params['dir'].y), str(cam_params['dir'].z),
           '--camup', str(cam_params['up'].x), str(cam_params['up'].y), str(cam_params['up'].z),
           '--fov', str(cam_params['fov']),
           '--samples', str(samples),
           '--mode', render_mode]
    print(f"[DIYRenderer] Calling external renderer (mode={render_mode}, samples={samples}): {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stderr_len = len(proc.stderr) if proc.stderr else 0
        print(f"[DIYRenderer] External renderer stderr length: {stderr_len} chars")
        if proc.stderr:
            # Show last 5000 chars of stderr (where our debug output should be)
            print(f"[DIYRenderer] External renderer stderr (last 5000 chars):\n{proc.stderr[-5000:]}")
        
        # DEBUG: Check stdout
        stdout_len = len(proc.stdout) if proc.stdout else 0
        print(f"[DIYRenderer] External renderer stdout length: {stdout_len} chars")
        if stdout_len == 0:
            print(f"[DIYRenderer] ERROR: No stdout from renderer! Command: {' '.join(cmd)}")
            return None
            
    except Exception as e:
        print('[DIYRenderer] External renderer invocation failed:', e)
        return None
    lines = proc.stdout.strip().splitlines()
    print(f"[DIYRenderer] Got {len(lines)} lines from renderer (expected {tile_w * tile_h})")
    if len(lines) != tile_w * tile_h:
        print('[DIYRenderer] Unexpected line count from external renderer', len(lines), 'expected', tile_w * tile_h)
        if len(lines) > 0:
            print(f"[DIYRenderer] First line: {lines[0]}")
            print(f"[DIYRenderer] Last line: {lines[-1]}")
        return None
    pixels = []
    for ln in lines:
        try:
            r,g,b,a = map(float, ln.split())
            pixels.append([r,g,b,a])
        except ValueError:
            pixels.append([1.0,0.0,1.0,1.0])  # error magenta
    
    # Flip Y-axis: C++ outputs top-to-bottom, Blender expects bottom-to-top
    flipped_pixels = []
    for y in range(tile_h - 1, -1, -1):  # Reverse Y order
        for x in range(tile_w):
            flipped_pixels.append(pixels[y * tile_w + x])
    
    # Return linear values without gamma correction
    # Gamma correction will be applied after sample accumulation
    return flipped_pixels


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
            self.accumulated_samples = {}  # tile_key -> (accumulated_pixels, sample_count)
            # target_samples will be set from scene settings in view_draw

    def _render_gradient(self, width, height):
        pixels = []
        for y in range(height):
            fy = y / (height - 1) if height > 1 else 0.0
            for x in range(width):
                fx = x / (width - 1) if width > 1 else 0.0
                pixels.append([fx, fy, 0.2, 1.0])
        return pixels

    def render(self, depsgraph):
        """
        Main render function for F12 rendering.
        Implements progressive rendering: starts with low sample counts and gradually increases
        to reduce wait time for initial preview while achieving high quality final result.
        """
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        
        # Get target samples from scene settings (user-configurable in UI)
        target_samples = scene.diy_renderer.samples
        print(f"[DIYRenderer] Starting progressive render ({width} x {height}, target: {target_samples} samples)")
        cam_params = compute_camera_params(scene, width, height)
        
        if not cam_params:
            # Fallback to gradient
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = self._render_gradient(width, height)
            self.end_result(result)
            return
        
        scene_file = export_scene_to_file(depsgraph)
        if not scene_file:
            # Fallback to gradient
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = self._render_gradient(width, height)
            self.end_result(result)
            return
        
        # Progressive rendering: start with low samples, gradually increase
        # Generate sample iterations dynamically to reach target
        # Strategy: double sample count each iteration (1, 2, 4, 8, ...) until we reach target
        # This gives fast initial preview while converging to final quality
        sample_iterations = []
        current = 1
        while sum(sample_iterations) < target_samples:
            sample_iterations.append(current)
            if sum(sample_iterations) + current * 2 <= target_samples:
                current *= 2  # Double for next iteration
            else:
                # Add final iteration to reach exactly target_samples
                remaining = target_samples - sum(sample_iterations)
                if remaining > 0:
                    sample_iterations.append(remaining)
                break
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        
        for idx, iteration_samples in enumerate(sample_iterations):
            if self.test_break():
                print("[DIYRenderer] Render cancelled by user")
                break
            
            # Update Blender UI with progress (progress bar and status text)
            progress = total_samples / max_samples
            self.update_progress(progress)
            self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples")
            
            print(f"[DIYRenderer] Rendering iteration with {iteration_samples} samples (total: {total_samples + iteration_samples})")
            
            # Get debug mode from settings
            diy = scene.diy_renderer
            debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
            
            # Render this iteration by calling external C++ renderer
            iteration_pixels = call_external_renderer(
                scene_file, 0, 0, width, height, width, height, cam_params, 
                samples=iteration_samples,
                debug_mode=debug_mode
            )
            
            if not iteration_pixels or len(iteration_pixels) != width * height:
                print("[DIYRenderer] Iteration failed, skipping")
                continue
            
            # Accumulate samples: blend new samples with previous iterations
            # Uses weighted average: (old * old_count + new * new_count) / total_count
            if accumulated_pixels is None:
                # First iteration - just use the pixels directly
                accumulated_pixels = iteration_pixels
                total_samples = iteration_samples
            else:
                # Subsequent iterations - blend with accumulated result
                new_total = total_samples + iteration_samples
                blended = []
                for i in range(len(accumulated_pixels)):
                    r = (accumulated_pixels[i][0] * total_samples + iteration_pixels[i][0] * iteration_samples) / new_total
                    g = (accumulated_pixels[i][1] * total_samples + iteration_pixels[i][1] * iteration_samples) / new_total
                    b = (accumulated_pixels[i][2] * total_samples + iteration_pixels[i][2] * iteration_samples) / new_total
                    a = 1.0
                    blended.append([r, g, b, a])
                accumulated_pixels = blended
                total_samples = new_total
            
            # Blender expects linear color space, so pass accumulated pixels directly
            # (Blender handles sRGB conversion internally based on color management settings)
            
            # Update the render result in Blender's render window
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = accumulated_pixels
            self.end_result(result)
            self.update_result(result)  # This refreshes the render window
            
            # Final progress update for this iteration
            progress = total_samples / max_samples
            self.update_progress(progress)
            self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples")
            print(f"[DIYRenderer] Updated render with {total_samples} total samples")
        
        # Clean up
        try:
            if scene_file and os.path.isfile(scene_file):
                os.remove(scene_file)
        except Exception:
            pass
        
        print(f"[DIYRenderer] Progressive render complete ({total_samples} total samples)")

    def async_render_viewport(self, job_data):
        """Background thread function to render viewport without blocking UI"""
        while not self.stop_thread:
            try:
                # Get latest job, discard old ones
                depsgraph, cam_params, render_width, render_height, job_id, samples_per_iteration, tile_key = job_data
                scene_file = export_scene_to_file(depsgraph)
                if scene_file:
                    # Get debug mode from settings
                    scene = depsgraph.scene
                    diy = scene.diy_renderer
                    debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
                    
                    ext_pixels = call_external_renderer(
                        scene_file, 0, 0, render_width, render_height, 
                        render_width, render_height, cam_params, samples=samples_per_iteration,
                        debug_mode=debug_mode
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
                            'job_id': job_id,
                            'samples_per_iteration': samples_per_iteration,
                            'tile_key': tile_key
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
                import traceback
                traceback.print_exc()
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
        """
        Viewport rendering function - called continuously while viewport is active.
        Implements adaptive quality: low resolution while moving, high resolution when idle.
        Progressively accumulates samples when idle to improve quality over time.
        """
        self._init_async_render()
        region = context.region
        width = region.width
        height = region.height
        
        # Get viewport samples from scene settings (user-configurable in UI)
        target_samples = context.scene.diy_renderer.viewport_samples
        
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
        
        # Adaptive quality system: two-level rendering strategy
        # - Moving (< 1 second idle): low resolution, 1 sample for fast feedback
        # - Idle (>= 1 second): full resolution, progressive sampling up to target_samples
        if time_since_change < 1.0:
            # Moving: use low resolution, 1 sample
            render_width = max(40, width // 16)
            render_height = max(30, height // 16)
            update_interval = 1  # Update every frame when moving
            samples_per_iteration = 1
        else:
            # Idle for 1+ seconds: use full resolution, progressive samples
            render_width = max(100, width)
            render_height = max(100, height)
            update_interval = 5  # Update every 5 frames when idle
            samples_per_iteration = 4  # Add 4 samples per iteration
        
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        self.frame_counter += 1
        
        # Create tile key for accumulation (unique per resolution)
        tile_key = f"{render_width}x{render_height}"
        
        # Check current sample count for this tile
        if tile_key in self.accumulated_samples:
            current_sample_count = self.accumulated_samples[tile_key][1]
        else:
            current_sample_count = 0
        
        # Check if we need to update
        resolution_changed = (self.last_render_width != render_width or 
                             self.last_render_height != render_height)
        
        # If resolution changed, reset accumulation (switching between moving/idle)
        if resolution_changed:
            self.accumulated_samples = {}
            current_sample_count = 0
        
        # Determine if we need a new render:
        # - No texture exists yet
        # - Resolution changed (moving <-> idle transition)
        # - Time to update AND haven't reached target sample count yet
        needs_update = (
            not hasattr(self, 'texture') or self.texture is None or
            resolution_changed or
            (self.frame_counter % update_interval == 0 and current_sample_count < target_samples)
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
                job_data = (depsgraph, cam_params, render_width, render_height, self.job_counter, 
                           samples_per_iteration, tile_key)
                
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
            result_tile_key = result.get('tile_key', '')
            samples_per_iteration = result.get('samples_per_iteration', 1)
            
            if ext_pixels and len(ext_pixels) == result_width * result_height:
                # Accumulate samples
                if result_tile_key in self.accumulated_samples:
                    acc_pixels, prev_count = self.accumulated_samples[result_tile_key]
                    # Blend new samples with accumulated
                    new_count = prev_count + samples_per_iteration
                    blended = []
                    for i in range(len(acc_pixels)):
                        r = (acc_pixels[i][0] * prev_count + ext_pixels[i][0] * samples_per_iteration) / new_count
                        g = (acc_pixels[i][1] * prev_count + ext_pixels[i][1] * samples_per_iteration) / new_count
                        b = (acc_pixels[i][2] * prev_count + ext_pixels[i][2] * samples_per_iteration) / new_count
                        a = 1.0
                        blended.append([r, g, b, a])
                    self.accumulated_samples[result_tile_key] = (blended, new_count)
                    pixels = blended
                    print(f"[DIYRenderer] Accumulated samples: {new_count}/{target_samples}")
                else:
                    # First iteration
                    self.accumulated_samples[result_tile_key] = (ext_pixels, samples_per_iteration)
                    pixels = ext_pixels
                    print(f"[DIYRenderer] Initial samples: {samples_per_iteration}/{target_samples}")
                
                # Blender expects linear color space for viewport too
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
    bpy.utils.register_class(DIYRendererSettings)
    bpy.utils.register_class(DIY_RENDER_PT_sampling)
    bpy.utils.register_class(DIYRenderEngine)
    
    # Register property group
    bpy.types.Scene.diy_renderer = bpy.props.PointerProperty(type=DIYRendererSettings)
    
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
    
    # Unregister property group
    del bpy.types.Scene.diy_renderer
    
    bpy.utils.unregister_class(DIYRenderEngine)
    bpy.utils.unregister_class(DIY_RENDER_PT_sampling)
    bpy.utils.unregister_class(DIYRendererSettings)
    bpy.utils.unregister_class(DIYRendererPreferences)