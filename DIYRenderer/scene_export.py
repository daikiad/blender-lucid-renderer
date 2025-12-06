"""
Scene export functionality - exports Blender scenes to JSON for the C++ renderer.
=================================================================================

このモジュールは Blender シーンを JSON 形式でエクスポートする機能を提供します。
C++ レンダラーはこの JSON を読み込んでシーンを再構築します。

エクスポートされるデータ:
- メッシュジオメトリ（頂点、三角形）
- 頂点法線（スムーズシェーディング用）
- マテリアル（Principled BSDF パラメータ）
- ノードツリー（完全なシェーダーグラフ）

主要関数:
- export_scene_to_file(): メインのエクスポート関数
- export_scene_to_json(): JSON ファイルに書き出し
- serialize_node_tree(): ノードグラフをシリアライズ
- get_material_properties(): マテリアルを抽出

キャッシュシステム:
SceneCache クラスはシーンの MD5 ハッシュを計算し、
変更がない場合は再エクスポートをスキップします。
これにより、カメラのみが動いた場合のパフォーマンスが向上します。

JSON フォーマット例:
{
    "version": "1.0",
    "meshes": [
        {
            "name": "Cube",
            "vertices": [[x, y, z], ...],
            "triangles": [[i0, i1, i2], ...],
            "triangle_normals": [[[n0], [n1], [n2]], ...],
            "smooth": true,
            "material": {
                "name": "Material",
                "use_nodes": true,
                "base_color": [0.8, 0.8, 0.8],
                "metallic": 0.0,
                "roughness": 0.5,
                "emission": [0.0, 0.0, 0.0],
                "node_tree": { ... }
            }
        }
    ]
}
"""

import os
import json
import tempfile

import bpy


def _get_prefs():
    """
    アドオン設定を取得します。
    
    Returns:
        DIYRendererPreferences オブジェクト、または None
    """
    entry = bpy.context.preferences.addons.get("DIYRenderer")
    if entry is not None:
        return getattr(entry, 'preferences', None)
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
            'type': node.bl_idname,
            'label': node.label,
            'location': [node.location.x, node.location.y],
            'properties': {},
            'inputs': [],
            'outputs': []
        }
        
        # Serialize node properties using RNA reflection
        for prop in node.bl_rna.properties:
            if prop.is_readonly or prop.identifier in ('rna_type', 'inputs', 'outputs'):
                continue
            try:
                value = getattr(node, prop.identifier)
                if hasattr(value, '__len__') and not isinstance(value, str):
                    value = list(value)
                elif hasattr(value, 'name'):
                    value = value.name
                node_data['properties'][prop.identifier] = value
            except Exception:
                pass
        
        # Serialize input sockets
        for i, socket in enumerate(node.inputs):
            socket_data = {
                'index': i,
                'name': socket.name,
                'type': socket.type,
                'default_value': serialize_socket_value(socket),
                'is_linked': socket.is_linked
            }
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
    
    # Serialize all links
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
    
    mat = obj.data.materials[0]
    if not mat:
        return None
    
    result = {
        'name': mat.name,
        'use_nodes': mat.use_nodes,
        'node_tree': None,
        'legacy_properties': {}
    }
    
    if mat.use_nodes and mat.node_tree:
        result['node_tree'] = serialize_node_tree(mat.node_tree)
    
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
                result['legacy_properties']['base_color'] = [0.8, 0.8, 0.8]
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
        
        # Transmission
        if 'Transmission' in principled.inputs:
            transmission_input = principled.inputs['Transmission']
            result['legacy_properties']['transmission'] = transmission_input.default_value if not transmission_input.is_linked else 0.0
        elif 'Transmission Weight' in principled.inputs:
            transmission_input = principled.inputs['Transmission Weight']
            result['legacy_properties']['transmission'] = transmission_input.default_value if not transmission_input.is_linked else 0.0
        else:
            result['legacy_properties']['transmission'] = 0.0
        
        # IOR
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
        elif 'Emission' in principled.inputs:
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
    else:
        result['legacy_properties'] = {
            'base_color': [0.8, 0.8, 0.8],
            'metallic': 0.0,
            'roughness': 0.5,
            'emission': [0.0, 0.0, 0.0],
            'transmission': 0.0,
            'ior': 1.45
        }
    
    return result


def _export_world_environment(scene):
    """
    Export world environment settings (background color, strength).
    
    Handles:
    - Simple background color from world.color
    - Background node in world shader graph (color + strength)
    
    Returns:
        dict: Environment settings with 'color' [r, g, b] and 'strength' float
    """
    # Default: light gray environment for ambient lighting
    env_data = {
        "color": [0.5, 0.5, 0.5],  # Default mid-gray
        "strength": 1.0
    }
    
    world = scene.world
    if not world:
        print(f"[Environment] No world, using default: color={env_data['color']}, strength={env_data['strength']}")
        return env_data
    
    # Try to get from node tree first
    if world.use_nodes and world.node_tree:
        nodes = world.node_tree.nodes
        
        # Look for Background node connected to World Output
        for node in nodes:
            if node.type == 'BACKGROUND':
                # Get color input
                color_input = node.inputs.get('Color')
                if color_input:
                    if color_input.is_linked:
                        # Follow link to get actual color (simplified - only handles direct color)
                        pass
                    else:
                        color = color_input.default_value
                        env_data["color"] = [color[0], color[1], color[2]]
                
                # Get strength input
                strength_input = node.inputs.get('Strength')
                if strength_input:
                    if not strength_input.is_linked:
                        env_data["strength"] = strength_input.default_value
                
                print(f"[Environment] From Background node: color={env_data['color']}, strength={env_data['strength']}")
                break
    else:
        # Fallback to simple world color
        if hasattr(world, 'color'):
            color = world.color
            env_data["color"] = [color[0], color[1], color[2]]
        print(f"[Environment] From world.color: color={env_data['color']}, strength={env_data['strength']}")
    
    return env_data


def _export_light(obj, matrix_world):
    """
    Blender のライトオブジェクトをエクスポート
    
    サポートするライトタイプ:
    - POINT: 点光源
    - SUN: 平行光源（方向のみ、無限遠）
    - SPOT: スポットライト
    - AREA: 面光源（矩形/円形）
    
    Args:
        obj: Blender ライトオブジェクト
        matrix_world: ワールド変換行列
        
    Returns:
        dict: ライトデータ、または None
    """
    from mathutils import Vector
    
    light = obj.data
    if not light:
        return None
    
    # 位置と方向を取得
    position = matrix_world.translation
    # ライトのローカル -Z がワールド空間の方向
    direction = (matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    
    # 色とパワー
    color = list(light.color)
    energy = light.energy  # Watts (Blender 2.8+)
    
    light_data = {
        "name": obj.name,
        "type": light.type,  # 'POINT', 'SUN', 'SPOT', 'AREA'
        "position": [position.x, position.y, position.z],
        "direction": [direction.x, direction.y, direction.z],
        "color": color,
        "energy": energy,
    }
    
    # タイプ固有のプロパティ
    if light.type == 'POINT':
        light_data["radius"] = light.shadow_soft_size
        
    elif light.type == 'SUN':
        light_data["angle"] = light.angle  # 太陽の角度（ソフトシャドウ用）
        
    elif light.type == 'SPOT':
        light_data["radius"] = light.shadow_soft_size
        light_data["spot_size"] = light.spot_size  # 円錐角度（ラジアン）
        light_data["spot_blend"] = light.spot_blend  # エッジのぼかし (0-1)
        
    elif light.type == 'AREA':
        light_data["shape"] = light.shape  # 'SQUARE', 'RECTANGLE', 'DISK', 'ELLIPSE'
        light_data["size"] = light.size
        if light.shape in ('RECTANGLE', 'ELLIPSE'):
            light_data["size_y"] = light.size_y
        else:
            light_data["size_y"] = light.size
        # 面光源の向きベクトル（X, Y軸）
        right = (matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
        up = (matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()
        light_data["right"] = [right.x, right.y, right.z]
        light_data["up"] = [up.x, up.y, up.z]
    
    return light_data


def export_scene_to_json(depsgraph):
    """
    Export complete scene to JSON format.
    """
    from mathutils import Vector
    
    prefs = _get_prefs()
    base_dir = tempfile.gettempdir()
    if prefs:
        export_dir = getattr(prefs, 'scene_export_directory', '')
        if export_dir and isinstance(export_dir, str) and os.path.isdir(export_dir):
            base_dir = export_dir
    
    path = os.path.join(base_dir, "diy_scene_debug.json")
    
    # Get scene for environment export
    scene = depsgraph.scene
    
    scene_data = {
        "version": "1.0",
        "meshes": [],
        "lights": [],  # Blender ネイティブライト
        "environment": _export_world_environment(scene)
    }
    
    for obj_instance in depsgraph.object_instances:
        obj = obj_instance.object
        
        # ライトオブジェクトの処理
        if obj.type == 'LIGHT':
            light_data = _export_light(obj, obj_instance.matrix_world)
            if light_data:
                scene_data["lights"].append(light_data)
            continue
        
        if obj.type != 'MESH':
            continue
        
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if not mesh:
            continue
        
        mw = obj_instance.matrix_world
        normal_matrix = mw.to_3x3().inverted().transposed()
        vertices = []
        for v in mesh.vertices:
            co = mw @ v.co
            vertices.append([co.x, co.y, co.z])
        
        use_smooth = any(p.use_smooth for p in mesh.polygons)
        
        corner_normals_data = None
        if use_smooth:
            try:
                if hasattr(mesh, 'corner_normals'):
                    corner_normals_data = [n.vector[:] for n in mesh.corner_normals]
                else:
                    mesh.calc_normals_split()
                    corner_normals_data = [loop.normal[:] for loop in mesh.loops]
            except Exception as e:
                print(f"[DIYRenderer] Warning: Could not get corner normals: {e}")
                use_smooth = False
        
        # Get UV layer if available
        uv_layer = None
        if mesh.uv_layers:
            uv_layer = mesh.uv_layers.active
        
        triangles = []
        triangle_normals = []
        triangle_uvs = []  # UV per triangle vertex
        
        for poly in mesh.polygons:
            v_indices = list(poly.vertices)
            loop_indices = list(poly.loop_indices)
            if len(v_indices) < 3:
                continue
            for i in range(1, len(v_indices) - 1):
                triangles.append([v_indices[0], v_indices[i], v_indices[i+1]])
                
                if use_smooth and corner_normals_data:
                    n0_local = corner_normals_data[loop_indices[0]]
                    n1_local = corner_normals_data[loop_indices[i]]
                    n2_local = corner_normals_data[loop_indices[i+1]]
                    
                    n0 = (normal_matrix @ Vector(n0_local)).normalized()
                    n1 = (normal_matrix @ Vector(n1_local)).normalized()
                    n2 = (normal_matrix @ Vector(n2_local)).normalized()
                    
                    triangle_normals.append([
                        [n0.x, n0.y, n0.z],
                        [n1.x, n1.y, n1.z],
                        [n2.x, n2.y, n2.z]
                    ])
                
                # Export UV coordinates for this triangle
                if uv_layer:
                    uv0 = uv_layer.data[loop_indices[0]].uv
                    uv1 = uv_layer.data[loop_indices[i]].uv
                    uv2 = uv_layer.data[loop_indices[i+1]].uv
                    triangle_uvs.append([
                        [uv0.x, uv0.y],
                        [uv1.x, uv1.y],
                        [uv2.x, uv2.y]
                    ])
        
        mat_props = get_material_properties(obj)
        
        if mat_props:
            material = {
                "name": mat_props['name'],
                "use_nodes": mat_props['use_nodes'],
                "base_color": mat_props['legacy_properties'].get('base_color', [0.8, 0.8, 0.8]),
                "metallic": mat_props['legacy_properties'].get('metallic', 0.0),
                "roughness": mat_props['legacy_properties'].get('roughness', 0.5),
                "emission": mat_props['legacy_properties'].get('emission', [0.0, 0.0, 0.0]),
                "transmission": mat_props['legacy_properties'].get('transmission', 0.0),
                "ior": mat_props['legacy_properties'].get('ior', 1.45),
                "node_tree": mat_props['node_tree']
            }
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
        
        attributes = {}
        
        if mesh.vertex_colors:
            color_layer = mesh.vertex_colors.active
            if color_layer:
                vertex_colors = []
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        color = color_layer.data[loop_idx].color
                        vertex_colors.append([color[0], color[1], color[2], color[3]])
                attributes["vertex_color"] = vertex_colors
        
        if mesh.uv_layers:
            uv_layer = mesh.uv_layers.active
            if uv_layer:
                uvs = []
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        uv = uv_layer.data[loop_idx].uv
                        uvs.append([uv.x, uv.y])
                attributes["uv"] = uvs
        
        if hasattr(mesh, 'attributes'):
            for attr in mesh.attributes:
                if attr.name.startswith('.'):
                    continue
                attr_name = attr.name
                attr_data = []
                
                if attr.domain == 'POINT':
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
        
        if use_smooth and triangle_normals:
            mesh_data["triangle_normals"] = triangle_normals
            mesh_data["smooth"] = True
        
        if triangle_uvs:
            mesh_data["triangle_uvs"] = triangle_uvs
        
        scene_data["meshes"].append(mesh_data)
        eval_obj.to_mesh_clear()
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2)
    
    return path


def export_scene_to_json_for_session(depsgraph, session_id: int):
    """
    Export scene to JSON file with session-specific filename.
    This prevents file conflicts between different render contexts.
    
    Uses the same export logic as export_scene_to_json but with a
    session-specific output path.
    """
    # Session-specific filename
    path = os.path.join(tempfile.gettempdir(), f'diy_scene_session_{session_id}.json')
    
    # Reuse the core export logic from export_scene_to_json
    # but with a custom path
    return _export_scene_to_path(depsgraph, path)


def _export_scene_to_path(depsgraph, path: str):
    """
    Internal function to export scene to a specific path.
    This is the core export logic shared by all export functions.
    """
    scene = depsgraph.scene
    scene_data = {
        "version": "1.0",
        "meshes": [],
        "native_lights": [],
        "environment": _export_world_environment(scene)
    }
    
    for obj_instance in depsgraph.object_instances:
        obj = obj_instance.object
        if obj.type == 'LIGHT':
            light_data = _export_light(obj, obj_instance.matrix_world)
            if light_data:
                scene_data["native_lights"].append(light_data)
            continue
            
        if obj.type != 'MESH':
            continue
            
        eval_obj = obj.evaluated_get(depsgraph)
        
        try:
            eval_mesh = eval_obj.to_mesh()
        except RuntimeError:
            continue
        
        if eval_mesh is None:
            continue
        
        matrix = obj_instance.matrix_world
        
        # Get vertices in world space
        verts = []
        for v in eval_mesh.vertices:
            world_pos = matrix @ v.co
            verts.append([world_pos.x, world_pos.y, world_pos.z])
        
        # Check for smooth shading
        use_smooth = False
        if eval_mesh.polygons:
            use_smooth = any(p.use_smooth for p in eval_mesh.polygons)
        
        # Get UV layer
        uv_layer = None
        if eval_mesh.uv_layers.active:
            uv_layer = eval_mesh.uv_layers.active.data
        
        # Triangulate
        eval_mesh.calc_loop_triangles()
        tris = []
        triangle_normals = []
        triangle_uvs = []
        
        normal_matrix = matrix.to_3x3().inverted().transposed()
        
        for lt in eval_mesh.loop_triangles:
            tris.append([lt.vertices[0], lt.vertices[1], lt.vertices[2]])
            
            if use_smooth:
                normals = []
                for loop_idx in lt.loops:
                    loop = eval_mesh.loops[loop_idx]
                    local_normal = eval_mesh.vertices[loop.vertex_index].normal
                    world_normal = (normal_matrix @ local_normal).normalized()
                    normals.append([world_normal.x, world_normal.y, world_normal.z])
                triangle_normals.append(normals)
            
            if uv_layer:
                uvs = []
                for loop_idx in lt.loops:
                    uv = uv_layer[loop_idx].uv
                    uvs.append([uv.x, uv.y])
                triangle_uvs.append(uvs)
        
        # Get material
        mat_data = {}
        if obj.active_material:
            mat_data = get_material_properties(obj)
        
        mesh_data = {
            "name": obj.name,
            "vertices": verts,
            "triangles": tris,
            "material": mat_data
        }
        
        if use_smooth and triangle_normals:
            mesh_data["triangle_normals"] = triangle_normals
            mesh_data["smooth"] = True
        
        if triangle_uvs:
            mesh_data["triangle_uvs"] = triangle_uvs
        
        scene_data["meshes"].append(mesh_data)
        eval_obj.to_mesh_clear()
    
    print(f"[SceneExport _export_scene_to_path] Exported: {len(scene_data['meshes'])} meshes, {len(scene_data['native_lights'])} lights")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2)
    
    return path


class SceneCache:
    """
    Cache for exported scene files.
    Tracks scene state and reuses exported files when the scene hasn't changed.
    
    Now supports per-session caching to avoid conflicts between:
    - Viewport rendering
    - Material preview
    - F12 rendering
    """
    # グローバルキャッシュ（後方互換用）
    _global_instance = None
    
    # セッション別キャッシュ: session_id -> SceneCache
    _session_caches: dict = {}
    
    def __init__(self, session_id: int = 0):
        self.session_id = session_id
        self.cached_file = None
        self.scene_hash = None
        self.last_export_time = 0
    
    @classmethod
    def get_for_session(cls, session_id: int) -> 'SceneCache':
        """セッションID別のキャッシュを取得"""
        if session_id not in cls._session_caches:
            cls._session_caches[session_id] = cls(session_id)
        return cls._session_caches[session_id]
    
    @classmethod
    def clear_session(cls, session_id: int) -> None:
        """セッションのキャッシュをクリア"""
        if session_id in cls._session_caches:
            cache = cls._session_caches.pop(session_id)
            # キャッシュファイルを削除
            if cache.cached_file and os.path.isfile(cache.cached_file):
                try:
                    os.remove(cache.cached_file)
                except Exception:
                    pass
    
    def compute_scene_hash(self, depsgraph):
        """Compute a hash to detect scene changes."""
        import hashlib
        hasher = hashlib.md5()
        
        obj_count = 0
        for obj_instance in depsgraph.object_instances:
            obj = obj_instance.object
            if obj.type == 'MESH':
                obj_count += 1
                hasher.update(obj.name.encode())
                # トランスフォーム（位置・回転・スケール）をハッシュに含める
                matrix = obj_instance.matrix_world
                for row in matrix:
                    for val in row:
                        hasher.update(str(round(val, 6)).encode())
                if obj.data:
                    hasher.update(str(len(obj.data.vertices)).encode())
                if obj.active_material:
                    hasher.update(obj.active_material.name.encode())
            elif obj.type == 'LIGHT':
                # ライトの位置もハッシュに含める
                hasher.update(obj.name.encode())
                matrix = obj_instance.matrix_world
                for row in matrix:
                    for val in row:
                        hasher.update(str(round(val, 6)).encode())
        
        hasher.update(str(obj_count).encode())
        return hasher.hexdigest()
    
    def get_or_export(self, depsgraph, force=False):
        """Get cached scene file or export if changed."""
        current_hash = self.compute_scene_hash(depsgraph)
        
        if (not force and 
            self.cached_file and 
            os.path.isfile(self.cached_file) and
            self.scene_hash == current_hash):
            return self.cached_file
        
        # セッションIDがある場合はセッション固有のファイルを使用
        if self.session_id > 0:
            path = export_scene_to_json_for_session(depsgraph, self.session_id)
        else:
            path = export_scene_to_json(depsgraph)
        
        self.cached_file = path
        self.scene_hash = current_hash
        
        import time
        self.last_export_time = time.time()
        
        return path
    
    def get_cached_file_fast(self):
        """Get cached file without hash computation."""
        if self.cached_file and os.path.isfile(self.cached_file):
            return self.cached_file
        return None
    
    def invalidate(self):
        """Force re-export on next request."""
        self.scene_hash = None


def get_scene_cache(session_id: int = 0):
    """Get a scene cache instance.
    
    Args:
        session_id: Session ID (0 for global/legacy cache)
    
    Returns:
        SceneCache instance for the given session
    """
    if session_id > 0:
        return SceneCache.get_for_session(session_id)
    else:
        # 後方互換: グローバルインスタンス
        if SceneCache._global_instance is None:
            SceneCache._global_instance = SceneCache(0)
        return SceneCache._global_instance


def export_scene_to_file(depsgraph, use_cache=True, session_id: int = 0):
    """Export evaluated meshes - uses cache when possible.
    
    Args:
        depsgraph: Blender dependency graph
        use_cache: Whether to use caching
        session_id: Session ID for per-session caching (0 for global)
    
    Returns:
        Path to exported JSON file
    """
    try:
        if use_cache:
            return get_scene_cache(session_id).get_or_export(depsgraph)
        else:
            if session_id > 0:
                return export_scene_to_json_for_session(depsgraph, session_id)
            else:
                return export_scene_to_json(depsgraph)
    except Exception as e:
        print("[DIYRenderer] Scene export failed:", e)
        import traceback
        traceback.print_exc()
        return None
