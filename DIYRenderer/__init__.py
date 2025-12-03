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
    max_bounces: bpy.props.IntProperty(
        name="Max Bounces",
        description="Maximum number of light bounces (ray depth)",
        default=8,
        min=1,
        max=128
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
        
        # Render section (like Cycles)
        col = layout.column(heading="Render")
        col.prop(diy, "samples", text="Samples")
        
        # Viewport section
        col = layout.column(heading="Viewport")
        col.prop(diy, "viewport_samples", text="Samples")

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
        
        # Transmission (Glass)
        if 'Transmission' in principled.inputs:
            transmission_input = principled.inputs['Transmission']
            result['legacy_properties']['transmission'] = transmission_input.default_value if not transmission_input.is_linked else 0.0
        elif 'Transmission Weight' in principled.inputs:  # Blender 4.0+
            transmission_input = principled.inputs['Transmission Weight']
            result['legacy_properties']['transmission'] = transmission_input.default_value if not transmission_input.is_linked else 0.0
        else:
            result['legacy_properties']['transmission'] = 0.0
        
        # IOR (Index of Refraction)
        if 'IOR' in principled.inputs:
            ior_input = principled.inputs['IOR']
            result['legacy_properties']['ior'] = ior_input.default_value if not ior_input.is_linked else 1.45
        else:
            result['legacy_properties']['ior'] = 1.45
        
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
            'emission': [0.0, 0.0, 0.0],
            'transmission': 0.0,
            'ior': 1.45
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
                "transmission": mat_props['legacy_properties'].get('transmission', 0.0),
                "ior": mat_props['legacy_properties'].get('ior', 1.45),
                # Full node tree (for advanced rendering)
                "node_tree": mat_props['node_tree']
            }
            print(f"[DIYRenderer] Mesh '{obj.name}': emission={material['emission']}, base_color={material['base_color']}, transmission={material['transmission']}, ior={material['ior']}")
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


# ========== Scene Cache for Performance ==========
# Caches exported scene files to avoid redundant exports when scene hasn't changed

class SceneCache:
    """
    Cache for exported scene files.
    
    Tracks scene state and reuses exported files when the scene hasn't changed.
    This significantly reduces overhead for viewport rendering where the same
    scene is rendered many times (only camera changes).
    
    Cache invalidation triggers:
    - Object count changes
    - Object modifications (vertices, materials)
    - Material changes
    - Explicit invalidation via view_update()
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.cached_file = None
        self.scene_hash = None
        self.last_export_time = 0
    
    def compute_scene_hash(self, depsgraph):
        """
        Compute a hash to detect scene changes.
        
        This is a fast approximation - doesn't check every vertex,
        but catches most common changes (object add/remove, material edits).
        """
        import hashlib
        hasher = hashlib.md5()
        
        # Count objects
        obj_count = 0
        for obj_instance in depsgraph.object_instances:
            obj = obj_instance.object
            if obj.type == 'MESH':
                obj_count += 1
                # Include object name and modification state
                hasher.update(obj.name.encode())
                # Include vertex count (fast proxy for geometry changes)
                if obj.data:
                    hasher.update(str(len(obj.data.vertices)).encode())
                # Include material name
                if obj.active_material:
                    hasher.update(obj.active_material.name.encode())
        
        hasher.update(str(obj_count).encode())
        return hasher.hexdigest()
    
    def get_or_export(self, depsgraph, force=False):
        """
        Get cached scene file or export if changed.
        
        Args:
            depsgraph: Blender dependency graph
            force: Force re-export even if cached
            
        Returns:
            str: Path to JSON scene file
        """
        current_hash = self.compute_scene_hash(depsgraph)
        
        # Check if we can use cached file
        if (not force and 
            self.cached_file and 
            os.path.isfile(self.cached_file) and
            self.scene_hash == current_hash):
            print(f"[DIYRenderer] Using cached scene file")
            return self.cached_file
        
        # Need to export
        print(f"[DIYRenderer] Scene changed, exporting...")
        import time
        start = time.time()
        
        path = export_scene_to_json(depsgraph)
        
        elapsed = time.time() - start
        print(f"[DIYRenderer] Scene export took {elapsed*1000:.1f}ms")
        
        # Update cache
        self.cached_file = path
        self.scene_hash = current_hash
        self.last_export_time = time.time()
        
        return path
    
    def invalidate(self):
        """Force re-export on next request."""
        self.scene_hash = None
        print("[DIYRenderer] Scene cache invalidated")


def get_scene_cache():
    """Get the singleton scene cache instance."""
    return SceneCache()


def export_scene_to_file(depsgraph, use_cache=True):
    """
    Export evaluated meshes - uses cache when possible.
    
    Args:
        depsgraph: Blender dependency graph
        use_cache: If True, use cached file when scene unchanged
        
    Returns:
        str: Path to exported JSON file
    """
    try:
        if use_cache:
            return get_scene_cache().get_or_export(depsgraph)
        else:
            return export_scene_to_json(depsgraph)
    except Exception as e:
        print("[DIYRenderer] Scene export failed:", e)
        import traceback
        traceback.print_exc()
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

def call_external_renderer(scene_file, tile_x, tile_y, tile_w, tile_h, full_w, full_h, cam_params, mode='raytrace', samples=1, depth=8, debug_mode=None, cancel_check=None):
    """
    Call external C++ renderer with cancellation support.
    
    This function invokes the C++ path tracer as a subprocess and handles:
    - Argument passing (scene, camera, samples, depth, mode)
    - Progress monitoring with cancellation support
    - Output parsing (pixel data from stdout)
    - Y-axis flipping (C++ outputs top-to-bottom, Blender expects bottom-to-top)
    
    Args:
        scene_file: Path to exported JSON scene file
        tile_x, tile_y: Tile position (for tiled rendering)
        tile_w, tile_h: Tile dimensions
        full_w, full_h: Full image dimensions
        cam_params: Camera parameters dict with 'pos', 'dir', 'up', 'fov'
        mode: Render mode ('raytrace', 'normal', 'albedo', 'emission')
        samples: Number of samples per pixel
        depth: Maximum ray bounce depth
        debug_mode: Override render mode for debugging (same options as mode)
        cancel_check: Optional callable that returns True to cancel rendering
            This allows the render to be interrupted by checking Blender's
            test_break() or other cancellation flags.
    
    Returns:
        list: Pixel data as [[r,g,b,a], ...] in linear color space, Y-flipped
        None: If rendering failed or was cancelled
    
    Color Space Note:
        Returns LINEAR color space values. Blender's color management handles
        the conversion to sRGB for display. DO NOT apply gamma correction here.
    
    Cancellation:
        If cancel_check is provided and returns True during rendering,
        the subprocess is terminated and None is returned.
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
           '--depth', str(depth),
           '--mode', render_mode]
    print(f"[DIYRenderer] Calling external renderer (mode={render_mode}, samples={samples}, depth={depth}): {' '.join(cmd)}")
    
    try:
        import time
        import threading
        
        # Use Popen for non-blocking execution with cancellation support
        # CRITICAL: Use bufsize and handle stdout in a separate thread to prevent
        # pipe buffer from filling up and blocking the C++ renderer
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        # Use threads to read stdout/stderr to prevent buffer deadlock
        # When the C++ renderer outputs many lines (256x256=65536 lines),
        # the pipe buffer fills up and blocks if we don't read it.
        stdout_lines = []
        stderr_chunks = []
        
        def read_stdout():
            for line in proc.stdout:
                stdout_lines.append(line)
        
        def read_stderr():
            for line in proc.stderr:
                stderr_chunks.append(line)
        
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # Poll for completion with cancellation check
        while True:
            # Check if process has completed
            retcode = proc.poll()
            if retcode is not None:
                break
                
            # Check for cancellation
            if cancel_check and cancel_check():
                print("[DIYRenderer] Render cancelled, terminating subprocess...")
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)  # Give it 2 seconds to terminate gracefully
                except subprocess.TimeoutExpired:
                    proc.kill()  # Force kill if not terminated
                return None
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.05)  # 50ms polling interval
        
        # Wait for threads to finish reading
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        
        # Combine outputs
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_chunks)
        
        if proc.returncode != 0:
            print(f"[DIYRenderer] External renderer failed with code {proc.returncode}")
            if stderr:
                print(f"[DIYRenderer] stderr: {stderr[-2000:]}")
            return None
        
        stderr_len = len(stderr) if stderr else 0
        print(f"[DIYRenderer] External renderer stderr length: {stderr_len} chars")
        if stderr:
            # Show last 5000 chars of stderr (where our debug output should be)
            print(f"[DIYRenderer] External renderer stderr (last 5000 chars):\n{stderr[-5000:]}")
        
        # DEBUG: Check stdout
        stdout_len = len(stdout) if stdout else 0
        print(f"[DIYRenderer] External renderer stdout length: {stdout_len} chars")
        if stdout_len == 0:
            print(f"[DIYRenderer] ERROR: No stdout from renderer! Command: {' '.join(cmd)}")
            return None
            
    except Exception as e:
        print('[DIYRenderer] External renderer invocation failed:', e)
        return None
    
    lines = stdout.strip().splitlines()
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
        """Lazy initialization of async rendering infrastructure for viewport"""
        if not hasattr(self, 'render_queue'):
            self.render_queue = queue.Queue(maxsize=1)
            self.result_queue = queue.Queue()
            self.render_thread = None
            self.stop_thread = False
            self.rendering_in_progress = False
            self.high_res_complete = False
            self.accumulated_samples = {}  # tile_key -> (accumulated_pixels, sample_count)
            # target_samples will be set from scene settings in view_draw
            
            # Camera state tracking for detecting camera movement
            self.last_camera_matrix = None
            self.last_view_perspective = None
            
            # Continuous rendering thread management
            self.viewport_thread_running = False
            self.current_render_cancelled = False

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
        
        Implements progressive rendering with cancellation support:
        - Starts with low sample counts for fast initial preview
        - Gradually increases samples for final quality
        - Supports cancellation via X button (test_break())
        - Even during C++ subprocess execution, cancellation is immediate
        
        Render Pipeline:
        1. Export scene to JSON file (one-time)
        2. For each progressive iteration:
           a. Check for cancellation
           b. Call C++ renderer with cancel_check callback
           c. Accumulate samples using weighted average
           d. Update render result in Blender window
        3. Clean up temporary files
        """
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        
        # Initialize cancellation state
        self._render_cancelled = False
        
        # Get target samples from scene settings (user-configurable in UI)
        # Use original scene (not scene_eval) to ensure we get the user's settings
        original_scene = depsgraph.scene
        target_samples = original_scene.diy_renderer.samples
        print(f"[DIYRenderer] Starting progressive render ({width} x {height}, target: {target_samples} samples)")
        print(f"[DIYRenderer] DEBUG: scene.diy_renderer.samples = {original_scene.diy_renderer.samples}")
        cam_params = compute_camera_params(scene, width, height)
        
        if not cam_params:
            # Fallback to gradient
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = self._render_gradient(width, height)
            self.end_result(result)
            return
        
        # F12 render always exports fresh scene (no cache)
        scene_file = export_scene_to_file(depsgraph, use_cache=False)
        if not scene_file:
            # Fallback to gradient
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = self._render_gradient(width, height)
            self.end_result(result)
            return
        
        # Progressive rendering: start with low samples, gradually increase
        # Generate sample iterations dynamically to reach target
        # Strategy: 
        # - Start with doubling: 1, 2, 4, 8, 16, 32, 64, 128
        # - After 128, use fixed increments of 128 for more frequent updates
        # Example for 1024 samples: [1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128, 128, 128, 128, 9]
        sample_iterations = []
        current = 1
        total = 0
        max_increment = 128  # Cap increment at 128 for more frequent updates
        while total < target_samples:
            to_add = min(current, target_samples - total)
            sample_iterations.append(to_add)
            total += to_add
            if current < max_increment:
                current *= 2  # Double until we hit max_increment
            # After max_increment, stay at max_increment
        
        print(f"[DIYRenderer] Sample iterations: {sample_iterations} (total: {sum(sample_iterations)})")
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        
        # Time tracking for remaining time estimation
        import time
        render_start_time = time.time()
        
        # Define cancellation check function for subprocess
        def check_cancel():
            """Check if render should be cancelled"""
            return self.test_break() or self._render_cancelled
        
        def format_time(seconds):
            """Format seconds into human-readable string"""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins}m {secs:02d}s"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                return f"{hours}h {mins:02d}m"
        
        for idx, iteration_samples in enumerate(sample_iterations):
            # Check for cancellation before starting iteration
            if check_cancel():
                print("[DIYRenderer] Render cancelled by user")
                break
            
            # Calculate elapsed and remaining time
            elapsed = time.time() - render_start_time
            if total_samples > 0:
                time_per_sample = elapsed / total_samples
                remaining_samples = max_samples - total_samples
                remaining_time = time_per_sample * remaining_samples
                time_str = f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining_time)}"
            else:
                time_str = f"Elapsed: {format_time(elapsed)}"
            
            # Update Blender UI with progress (progress bar and status text)
            progress = total_samples / max_samples
            self.update_progress(progress)
            self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples | {time_str}")
            
            print(f"[DIYRenderer] Rendering iteration with {iteration_samples} samples (total: {total_samples + iteration_samples})")
            
            # Get debug mode and max_bounces from settings (use original scene)
            diy = original_scene.diy_renderer
            debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
            max_bounces = diy.max_bounces
            
            # Render this iteration by calling external C++ renderer
            # Pass cancel_check callback so subprocess can be terminated immediately
            iteration_pixels = call_external_renderer(
                scene_file, 0, 0, width, height, width, height, cam_params, 
                samples=iteration_samples,
                depth=max_bounces,
                debug_mode=debug_mode,
                cancel_check=check_cancel  # Enable immediate cancellation
            )
            
            # Check if cancelled during render
            if iteration_pixels is None and check_cancel():
                print("[DIYRenderer] Render cancelled during iteration")
                break
            
            if not iteration_pixels or len(iteration_pixels) != width * height:
                print("[DIYRenderer] Iteration failed, skipping")
                continue
            
            # Accumulate samples: C++ outputs raw SUM of samples (not averaged)
            # We simply add the sums together and divide by total samples for display
            
            # DEBUG: Check center pixel values (more likely to hit geometry)
            center_idx = (height // 2) * width + (width // 2)
            if len(iteration_pixels) > center_idx:
                p0 = iteration_pixels[center_idx]
                print(f"[DIYRenderer] DEBUG iteration {idx}: samples={iteration_samples}, center_pixel_raw=({p0[0]:.4f}, {p0[1]:.4f}, {p0[2]:.4f})")
            
            if accumulated_pixels is None:
                # First iteration - copy the raw sums
                accumulated_pixels = [[p[0], p[1], p[2], 1.0] for p in iteration_pixels]
                total_samples = iteration_samples
            else:
                # Subsequent iterations - add raw sums
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i][0] += iteration_pixels[i][0]
                    accumulated_pixels[i][1] += iteration_pixels[i][1]
                    accumulated_pixels[i][2] += iteration_pixels[i][2]
                total_samples += iteration_samples
            
            # DEBUG: Check accumulated values
            if len(accumulated_pixels) > center_idx:
                a0 = accumulated_pixels[center_idx]
                print(f"[DIYRenderer] DEBUG accumulated: total_samples={total_samples}, center_pixel_sum=({a0[0]:.4f}, {a0[1]:.4f}, {a0[2]:.4f})")
                print(f"[DIYRenderer] DEBUG display value: ({a0[0]/total_samples:.4f}, {a0[1]/total_samples:.4f}, {a0[2]/total_samples:.4f})")
            
            # Create display pixels by dividing accumulated sums by total samples
            display_pixels = []
            for p in accumulated_pixels:
                display_pixels.append([p[0] / total_samples, p[1] / total_samples, p[2] / total_samples, 1.0])
            
            # Update the render result in Blender's render window
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = display_pixels
            self.end_result(result)
            
            # Final progress update for this iteration with time info
            elapsed = time.time() - render_start_time
            if total_samples < max_samples:
                time_per_sample = elapsed / total_samples
                remaining_samples = max_samples - total_samples
                remaining_time = time_per_sample * remaining_samples
                time_str = f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining_time)}"
            else:
                time_str = f"Total time: {format_time(elapsed)}"
            
            progress = total_samples / max_samples
            self.update_progress(progress)
            self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples | {time_str}")
            print(f"[DIYRenderer] Updated render with {total_samples} total samples")
        
        # Clean up
        try:
            if scene_file and os.path.isfile(scene_file):
                os.remove(scene_file)
        except Exception:
            pass
        
        # Final completion message with total time
        total_elapsed = time.time() - render_start_time
        print(f"[DIYRenderer] Progressive render complete ({total_samples} total samples) in {format_time(total_elapsed)}")

    def async_render_viewport(self, initial_job_data):
        """
        Background thread function to render viewport without blocking UI.
        
        This thread runs continuously while jobs are queued, processing each
        render request in turn. It supports cancellation for quick response
        when the user moves the camera or changes the scene.
        
        Args:
            initial_job_data: Tuple of (depsgraph, cam_params, width, height, job_id, samples, tile_key, target_samples)
        """
        self.viewport_thread_running = True
        job_data = initial_job_data
        accumulated_local = 0  # Track how many samples we've rendered
        
        while not self.stop_thread:
            try:
                # Check if this job should be cancelled (new job waiting)
                if self.current_render_cancelled:
                    self.current_render_cancelled = False
                    # Try to get new job
                    try:
                        job_data = self.render_queue.get_nowait()
                        accumulated_local = 0  # Reset for new job
                        continue  # Start fresh with new job
                    except queue.Empty:
                        break  # No more jobs, exit thread
                
                depsgraph, cam_params, render_width, render_height, job_id, samples_per_iteration, tile_key, target_samples = job_data
                
                # Export scene (uses cache if unchanged)
                scene_file = export_scene_to_file(depsgraph)
                if not scene_file:
                    break
                
                # Get debug mode from settings
                scene = depsgraph.scene
                diy = scene.diy_renderer
                debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
                max_bounces = diy.max_bounces
                
                # Define cancel check for this render
                def should_cancel():
                    return self.current_render_cancelled or self.stop_thread
                
                # Call external renderer with cancellation support
                ext_pixels = call_external_renderer(
                    scene_file, 0, 0, render_width, render_height, 
                    render_width, render_height, cam_params, 
                    samples=samples_per_iteration,
                    depth=max_bounces,
                    debug_mode=debug_mode,
                    cancel_check=should_cancel  # Enable cancellation
                )
                
                # Clean up temp file
                try:
                    if scene_file and os.path.isfile(scene_file):
                        os.remove(scene_file)
                except Exception:
                    pass
                
                # If cancelled, don't put result
                if self.current_render_cancelled:
                    self.current_render_cancelled = False
                    try:
                        job_data = self.render_queue.get_nowait()
                        accumulated_local = 0
                        continue
                    except queue.Empty:
                        break
                
                # Put result in queue (non-blocking)
                # Each result will be accumulated in view_draw for progressive refinement
                if ext_pixels:
                    accumulated_local += samples_per_iteration
                    try:
                        self.result_queue.put_nowait({
                            'pixels': ext_pixels,
                            'width': render_width,
                            'height': render_height,
                            'job_id': job_id,
                            'samples_per_iteration': samples_per_iteration,
                            'tile_key': tile_key
                        })
                        print(f"[DIYRenderer] Thread: rendered {samples_per_iteration} samples, total={accumulated_local}/{target_samples}")
                    except queue.Full:
                        pass  # Discard if queue full
                
                # Check if we need to continue rendering for more samples
                if accumulated_local < target_samples and not self.current_render_cancelled:
                    # Continue rendering with same parameters (job_data stays the same)
                    # Check for new job first
                    try:
                        new_job = self.render_queue.get_nowait()
                        job_data = new_job
                        accumulated_local = 0  # Reset for new job
                    except queue.Empty:
                        # Small delay to allow view_draw to process result
                        import time
                        time.sleep(0.05)  # 50ms delay for UI update
                    continue
                else:
                    # Reached target or cancelled, wait for next job
                    print(f"[DIYRenderer] Thread: completed target samples ({accumulated_local})")
                    try:
                        job_data = self.render_queue.get(timeout=0.5)
                        accumulated_local = 0
                    except queue.Empty:
                        break  # No more jobs after timeout, exit thread
                    
            except Exception as e:
                print(f"[DIYRenderer] Async render error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        self.viewport_thread_running = False
        self.rendering_in_progress = False
    
    def view_update(self, context, depsgraph):
        """
        Called when the scene is modified in viewport mode.
        
        Responsibilities:
        - Invalidate scene cache (force re-export)
        - Reset viewport render state
        - Clear accumulated samples
        - Cancel pending render jobs
        """
        self._init_async_render()
        
        # Invalidate scene cache - scene has changed
        get_scene_cache().invalidate()
        
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
        
        # Reset accumulated samples since scene changed
        self.accumulated_samples = {}
        
        # Cancel current render and clear queue
        self.current_render_cancelled = True
        try:
            while not self.render_queue.empty():
                self.render_queue.get_nowait()
        except queue.Empty:
            pass

    def _detect_camera_change(self, context):
        """
        Detect if the viewport camera has changed (rotation, pan, zoom).
        Returns True if camera changed, False otherwise.
        """
        region_data = context.region_data
        if region_data is None:
            return False
        
        current_matrix = region_data.view_matrix.copy()
        current_perspective = region_data.view_perspective
        
        changed = False
        if self.last_camera_matrix is not None:
            # Compare matrices (with small tolerance for floating point)
            diff = 0.0
            for i in range(4):
                for j in range(4):
                    diff += abs(current_matrix[i][j] - self.last_camera_matrix[i][j])
            if diff > 0.0001:
                changed = True
        
        if self.last_view_perspective != current_perspective:
            changed = True
        
        # Store current state
        self.last_camera_matrix = current_matrix
        self.last_view_perspective = current_perspective
        
        return changed

    def view_draw(self, context, depsgraph):
        """
        Viewport rendering function - called continuously while viewport is active.
        
        Implements adaptive quality rendering:
        - Low resolution (1/8) while camera is moving for responsive UI
        - High resolution with progressive sampling when idle
        - Samples accumulate over time to reduce noise
        - All rendering happens in background thread to avoid blocking UI
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
        
        # Detect camera movement (rotation, pan, zoom)
        camera_changed = self._detect_camera_change(context)
        if camera_changed:
            self.viewport_last_change_time = current_time
            self.high_res_complete = False
            # Cancel current render for immediate response
            self.current_render_cancelled = True
            # Clear accumulated samples since view changed
            self.accumulated_samples = {}
        
        time_since_change = current_time - self.viewport_last_change_time
        
        # Adaptive quality system:
        # - Moving (< 0.5 sec): 1/2 resolution, 1 sample - quick feedback
        # - Idle (>= 0.5 sec): full resolution, progressive sampling
        if time_since_change < 0.5:
            # Moving: half resolution for quick feedback
            scale_factor = 2
            render_width = max(256, width // scale_factor)
            render_height = max(192, height // scale_factor)
            samples_per_iteration = 1
            is_moving = True
        else:
            # Idle: full resolution, progressive samples
            render_width = width
            render_height = height
            samples_per_iteration = 4  # Add 4 samples per iteration
            is_moving = False
        
        # DEBUG: Log state changes
        if not hasattr(self, '_last_debug_state'):
            self._last_debug_state = None
            self._last_debug_time = 0
        debug_state = f"moving={is_moving}, res={render_width}x{render_height}"
        # Log every 2 seconds or when state changes
        if debug_state != self._last_debug_state or (current_time - self._last_debug_time > 2.0):
            thread_alive = self.render_thread is not None and self.render_thread.is_alive()
            print(f"[DIYRenderer] State: {debug_state}, time_idle={time_since_change:.2f}s, thread_alive={thread_alive}, queue_size={self.result_queue.qsize()}")
            self._last_debug_state = debug_state
            self._last_debug_time = current_time
        
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
        
        # Check if resolution changed
        resolution_changed = (self.last_render_width != render_width or 
                             self.last_render_height != render_height)
        
        # If resolution changed, reset accumulation
        if resolution_changed:
            self.accumulated_samples = {}
            current_sample_count = 0
        
        # Determine if we need a new render:
        # - No texture exists yet
        # - Resolution changed
        # - Moving and no recent render started (or current one is cancelled)
        # - Idle and haven't reached target samples
        thread_running = self.render_thread is not None and self.render_thread.is_alive()
        thread_effectively_running = thread_running and not self.current_render_cancelled
        
        needs_new_render = (
            not hasattr(self, 'texture') or self.texture is None or
            resolution_changed or
            (is_moving and not thread_effectively_running) or
            (not is_moving and current_sample_count < target_samples and not thread_effectively_running)
        )
        
        # DEBUG: Log render decisions
        if needs_new_render:
            print(f"[DIYRenderer] Starting new render: res={render_width}x{render_height}, samples={current_sample_count}/{target_samples}, thread_running={thread_running}, cancelled={self.current_render_cancelled}")
        
        if needs_new_render:
            self.last_render_width = render_width
            self.last_render_height = render_height
            
            region_data = context.region_data
            if region_data is not None:
                cam_params = {
                    'pos': region_data.view_matrix.inverted().translation,
                    'dir': (region_data.view_matrix.inverted().to_3x3() @ Vector((0,0,-1))).normalized(),
                    'up': (region_data.view_matrix.inverted().to_3x3() @ Vector((0,1,0))).normalized(),
                    'fov': 60.0 if getattr(region_data, 'is_perspective', True) else 5.0
                }
                
                # Prepare job data (including target_samples for thread to know when to stop)
                if not hasattr(self, 'job_counter'):
                    self.job_counter = 0
                self.job_counter += 1
                job_data = (depsgraph, cam_params, render_width, render_height, self.job_counter, 
                           samples_per_iteration, tile_key, target_samples)
                
                # Submit job to render thread
                try:
                    # Clear old jobs first
                    while not self.render_queue.empty():
                        try:
                            self.render_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    self.render_queue.put_nowait(job_data)
                    self.rendering_in_progress = True
                    
                    # Start thread if not running (or if cancelled, wait briefly and restart)
                    if not thread_running:
                        self.stop_thread = False
                        self.current_render_cancelled = False
                        self.render_thread = threading.Thread(
                            target=self.async_render_viewport,
                            args=(job_data,),
                            daemon=True
                        )
                        self.render_thread.start()
                        print("[DIYRenderer] Started new render thread")
                    elif self.current_render_cancelled:
                        # Thread is running but cancelled - it will pick up the new job from queue
                        print("[DIYRenderer] New job queued, waiting for cancelled thread to pick it up")
                except queue.Full:
                    pass  # Skip if queue full
        
        # Check for completed renders (non-blocking) - process ALL available results
        has_new_result = False
        results_processed = 0
        while True:
            try:
                result = self.result_queue.get_nowait()
                results_processed += 1
                ext_pixels = result['pixels']
                result_width = result['width']
                result_height = result['height']
                result_tile_key = result.get('tile_key', '')
                result_samples = result.get('samples_per_iteration', 1)
                
                if ext_pixels and len(ext_pixels) == result_width * result_height:
                    # Accumulate samples (only for same resolution)
                    # Accumulate samples: C++ outputs raw SUM (not averaged)
                    # accumulated_samples stores (raw_sum_pixels, total_sample_count)
                    if result_tile_key in self.accumulated_samples:
                        acc_pixels, prev_count = self.accumulated_samples[result_tile_key]
                        # Add raw sums together
                        new_count = prev_count + result_samples
                        for i in range(len(acc_pixels)):
                            acc_pixels[i][0] += ext_pixels[i][0]
                            acc_pixels[i][1] += ext_pixels[i][1]
                            acc_pixels[i][2] += ext_pixels[i][2]
                        self.accumulated_samples[result_tile_key] = (acc_pixels, new_count)
                        # Display: divide by total samples
                        pixels = [[p[0]/new_count, p[1]/new_count, p[2]/new_count, 1.0] for p in acc_pixels]
                        print(f"[DIYRenderer] Accumulated: {new_count} samples for {result_tile_key}")
                        if new_count >= target_samples:
                            print(f"[DIYRenderer] Viewport render complete: {new_count} samples")
                            self.high_res_complete = True
                    else:
                        # First iteration - store raw sums, display divided
                        acc_pixels = [[p[0], p[1], p[2], 1.0] for p in ext_pixels]
                        self.accumulated_samples[result_tile_key] = (acc_pixels, result_samples)
                        pixels = [[p[0]/result_samples, p[1]/result_samples, p[2]/result_samples, 1.0] for p in ext_pixels]
                        print(f"[DIYRenderer] First result for {result_tile_key}: {result_samples} samples")
                    
                    # Create GPU texture for display (updates progressively)
                    self.viewport_pixels_cache = pixels
                    flat = [c for px in pixels for c in px]
                    buffer = gpu.types.Buffer('FLOAT', result_width * result_height * 4, flat)
                    if hasattr(self, 'texture') and self.texture is not None:
                        try:
                            del self.texture
                        except Exception:
                            pass
                    self.texture = gpu.types.GPUTexture((result_width, result_height), format='RGBA16F', data=buffer)
                    self.texture_width = result_width
                    self.texture_height = result_height
                    has_new_result = True
                    self.rendering_in_progress = False
                else:
                    print(f"[DIYRenderer] Invalid result: pixels={len(ext_pixels) if ext_pixels else 0}, expected={result_width * result_height}")
                    
            except queue.Empty:
                break  # No more results in queue
        
        # Check thread status
        thread_alive = self.render_thread is not None and self.render_thread.is_alive()
        
        # Get updated sample count after potential accumulation
        if tile_key in self.accumulated_samples:
            current_sample_count = self.accumulated_samples[tile_key][1]
        
        # Request continuous redraw while:
        # - Rendering thread is running
        # - Not yet reached target samples at full resolution
        # - Switched to idle mode but haven't started high-res yet
        should_redraw = (
            thread_alive or 
            is_moving or 
            (not is_moving and current_sample_count < target_samples)
        )
        
        # DEBUG
        if not hasattr(self, '_last_redraw_log') or current_time - self._last_redraw_log > 1.0:
            print(f"[DIYRenderer] Redraw check: should_redraw={should_redraw}, thread_alive={thread_alive}, is_moving={is_moving}, samples={current_sample_count}/{target_samples}")
            self._last_redraw_log = current_time
        
        if should_redraw:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        
        # Draw the texture to fill the viewport
        if hasattr(self, 'texture') and self.texture is not None:
            draw_texture_2d(self.texture, (0, 0), width, height)


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
        # Check if our render engine is active
        scene = bpy.context.scene
        if scene and scene.render.engine == 'DIY_RENDER_MINIMAL':
            # Redraw all 3D viewports using our engine
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        # Check if this viewport is using rendered mode
                        for space in area.spaces:
                            if space.type == 'VIEW_3D' and space.shading.type == 'RENDERED':
                                area.tag_redraw()
                                break
    except Exception:
        pass  # Ignore errors during shutdown/reload
    
    return _viewport_timer_interval  # Return interval to keep timer running


def register():
    global _viewport_timer
    
    bpy.utils.register_class(DIYRendererPreferences)
    bpy.utils.register_class(DIYRendererSettings)
    bpy.utils.register_class(DIY_RENDER_PT_sampling)
    bpy.utils.register_class(DIY_RENDER_PT_light_paths)
    bpy.utils.register_class(DIY_RENDER_PT_debug)
    bpy.utils.register_class(DIYRenderEngine)
    
    # Register property group
    bpy.types.Scene.diy_renderer = bpy.props.PointerProperty(type=DIYRendererSettings)
    
    # Register viewport redraw timer
    if not bpy.app.timers.is_registered(_viewport_redraw_timer):
        bpy.app.timers.register(_viewport_redraw_timer, first_interval=_viewport_timer_interval, persistent=True)
    
    # Add our engine to all relevant panel compatibility
    # Note: We only add to generic panels, not Cycles/Eevee specific ones
    # Cycles panels have COMPAT_ENGINES = {'CYCLES'} so they won't show for our engine
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
        
        # Panels to explicitly exclude (only render sampling/performance panels we replace)
        exclude_panels = {
            # Our DIY_RENDER_PT_* panels replace these
            'CYCLES_RENDER_PT_sampling',
            'CYCLES_RENDER_PT_light_paths', 
            'CYCLES_RENDER_PT_performance',
        }
        
        for module in modules:
            for panel_name in dir(module):
                # Skip explicitly excluded panels
                if panel_name in exclude_panels:
                    continue
                
                panel = getattr(module, panel_name, None)
                if panel and hasattr(panel, 'COMPAT_ENGINES'):
                    panel.COMPAT_ENGINES.add('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not register panels: {e}")


def unregister():
    global _viewport_timer
    
    # Unregister viewport redraw timer
    if bpy.app.timers.is_registered(_viewport_redraw_timer):
        bpy.app.timers.unregister(_viewport_redraw_timer)
    
    # Remove our engine from panel compatibility
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
    
    # Unregister property group
    del bpy.types.Scene.diy_renderer
    
    bpy.utils.unregister_class(DIYRenderEngine)
    bpy.utils.unregister_class(DIY_RENDER_PT_debug)
    bpy.utils.unregister_class(DIY_RENDER_PT_light_paths)
    bpy.utils.unregister_class(DIY_RENDER_PT_sampling)
    bpy.utils.unregister_class(DIYRendererSettings)
    bpy.utils.unregister_class(DIYRendererPreferences)