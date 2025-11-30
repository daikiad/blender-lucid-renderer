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
from mathutils import Vector
import mathutils.geometry as geom


class DIYRenderEngine(bpy.types.RenderEngine):
    bl_idname = "DIY_RENDER_MINIMAL"
    bl_label = "DIY Renderer (Minimal)"
    bl_use_preview = True  # マテリアルプレビューでも使えるように
    bl_use_shading_nodes_custom = False  # カスタムシェーダーを使わない

    def ray_triangle_intersect(self, ray_origin, ray_direction, v0, v1, v2):
        """レイと三角形の交差判定（Möller–Trumbore algorithm）"""
        epsilon = 0.0000001
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = ray_direction.cross(edge2)
        a = edge1.dot(h)
        
        if -epsilon < a < epsilon:
            return None  # レイが三角形と平行
        
        f = 1.0 / a
        s = ray_origin - v0
        u = f * s.dot(h)
        
        if u < 0.0 or u > 1.0:
            return None
        
        q = s.cross(edge1)
        v = f * ray_direction.dot(q)
        
        if v < 0.0 or u + v > 1.0:
            return None
        
        t = f * edge2.dot(q)
        
        if t > epsilon:
            return t  # 距離を返す
        
        return None

    def trace_ray(self, ray_origin, ray_direction, scene_objects):
        """レイとシーン内のオブジェクトの交差判定（法線を返す）"""
        closest_distance = float('inf')
        hit_normal = None
        
        for obj_data in scene_objects:
            mesh = obj_data['mesh']
            matrix_inv = obj_data['matrix_inv']
            matrix_normal = obj_data['matrix_normal']
            bbox_min = obj_data['bbox_min']
            bbox_max = obj_data['bbox_max']
            
            # レイをオブジェクトのローカル空間に変換
            local_origin = matrix_inv @ ray_origin
            local_direction = matrix_inv.to_3x3() @ ray_direction
            
            # バウンディングボックスとの交差判定（高速カリング）
            tmin = (bbox_min.x - local_origin.x) / local_direction.x if local_direction.x != 0 else float('inf')
            tmax = (bbox_max.x - local_origin.x) / local_direction.x if local_direction.x != 0 else float('inf')
            if tmin > tmax:
                tmin, tmax = tmax, tmin
            
            tymin = (bbox_min.y - local_origin.y) / local_direction.y if local_direction.y != 0 else float('inf')
            tymax = (bbox_max.y - local_origin.y) / local_direction.y if local_direction.y != 0 else float('inf')
            if tymin > tymax:
                tymin, tymax = tymax, tymin
            
            if tmin > tymax or tymin > tmax:
                continue  # バウンディングボックスと交差しない
            
            # 各三角形と交差判定
            for poly in mesh.polygons:
                if len(poly.vertices) < 3:
                    continue
                
                # 三角形の頂点を取得
                v0 = mesh.vertices[poly.vertices[0]].co
                v1 = mesh.vertices[poly.vertices[1]].co
                v2 = mesh.vertices[poly.vertices[2]].co
                
                # 交差判定
                t = self.ray_triangle_intersect(local_origin, local_direction, v0, v1, v2)
                
                if t is not None and t < closest_distance:
                    closest_distance = t
                    # ローカル空間の法線をワールド空間に変換
                    local_normal = poly.normal
                    hit_normal = matrix_normal @ local_normal
                    hit_normal.normalize()
                
                # 4頂点の場合のみ追加の三角形
                if len(poly.vertices) == 4:
                    v3 = mesh.vertices[poly.vertices[3]].co
                    t = self.ray_triangle_intersect(local_origin, local_direction, v0, v2, v3)
                    if t is not None and t < closest_distance:
                        closest_distance = t
                        local_normal = poly.normal
                        hit_normal = matrix_normal @ local_normal
                        hit_normal.normalize()
        
        return hit_normal

    def get_material_color(self, material):
        """マテリアルから色を取得"""
        if material is None:
            return Vector((0.8, 0.8, 0.8))  # デフォルトグレー
        
        # プリンシプルBSDFノードを探す
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # Base Colorを取得
                    base_color_input = node.inputs['Base Color']
                    if base_color_input.is_linked:
                        # リンクされている場合は複雑なので白を返す
                        return Vector((0.8, 0.8, 0.8))
                    else:
                        color = base_color_input.default_value
                        return Vector((color[0], color[1], color[2]))
        
        # ノードを使っていない場合はディフューズ色
        if hasattr(material, 'diffuse_color'):
            color = material.diffuse_color
            return Vector((color[0], color[1], color[2]))
        
        return Vector((0.8, 0.8, 0.8))

    def render_scene(self, depsgraph, width, height):
        """シーンをレイトレーシングでレンダリング"""
        scene = depsgraph.scene_eval
        camera = scene.camera
        
        if not camera:
            # カメラがない場合は赤を返す
            return [[1.0, 0.0, 0.0, 1.0] for _ in range(width * height)]
        
        # カメラ情報
        camera_matrix = camera.matrix_world
        camera_location = camera_matrix.translation
        camera_data = camera.data
        
        # 視野角の計算
        import math
        sensor_width = camera_data.sensor_width
        focal_length = camera_data.lens
        fov = 2.0 * math.atan(sensor_width / (2.0 * focal_length))
        
        # アスペクト比
        aspect_ratio = width / height if height > 0 else 1.0
        
        # シーンオブジェクトを収集
        scene_objects = []
        for obj_instance in depsgraph.object_instances:
            obj = obj_instance.object
            if obj.type == 'MESH':
                mesh = obj.evaluated_get(depsgraph).data
                matrix = obj_instance.matrix_world
                
                # マテリアルを取得
                material = None
                if len(obj.material_slots) > 0:
                    material = obj.material_slots[0].material
                
                scene_objects.append({
                    'mesh': mesh,
                    'matrix': matrix,
                    'material': material
                })
        
        # 背景色
        bg_color = Vector((0.2, 0.2, 0.2))
        if scene.world and scene.world.use_nodes:
            # ワールドの背景色を取得（簡易版）
            for node in scene.world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    color = node.inputs['Color'].default_value
                    bg_color = Vector((color[0], color[1], color[2]))
                    break
        elif scene.world:
            color = scene.world.color
            bg_color = Vector((color[0], color[1], color[2]))
        
        # ピクセルごとにレイトレーシング
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # 正規化デバイス座標（-1 to 1）
                px = (2.0 * x / width - 1.0) * aspect_ratio
                py = 1.0 - 2.0 * y / height
                
                # レイの方向を計算
                scale = math.tan(fov / 2.0)
                ray_dir_local = Vector((px * scale, py * scale, -1.0))
                ray_dir_local.normalize()
                
                # カメラ空間からワールド空間に変換
                ray_direction = camera_matrix.to_3x3() @ ray_dir_local
                ray_direction.normalize()
                
                # レイトレース
                hit_material = self.trace_ray(camera_location, ray_direction, scene_objects)
                
                if hit_material:
                    color = self.get_material_color(hit_material)
                    pixels.append([color.x, color.y, color.z, 1.0])
                else:
                    pixels.append([bg_color.x, bg_color.y, bg_color.z, 1.0])
        
        return pixels

    def _render_gradient(self, width, height):
        """グラデーション画像を生成"""
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
        return pixels

    def render(self, depsgraph):
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)

        # シーン情報をコンソールに出力
        print("\n=== DIY Renderer: Scene Info ===")
        
        # カメラ情報
        camera = scene.camera
        if camera:
            print(f"Camera: {camera.name}")
            print(f"  Location: {camera.matrix_world.translation}")
            print(f"  Lens: {camera.data.lens}mm")
        
        # オブジェクト情報
        mesh_count = 0
        light_count = 0
        for obj_instance in depsgraph.object_instances:
            obj = obj_instance.object
            
            if obj.type == 'MESH':
                mesh_count += 1
                mesh = obj.evaluated_get(depsgraph).data
                print(f"Mesh: {obj.name}")
                print(f"  Vertices: {len(mesh.vertices)}")
                print(f"  Polygons: {len(mesh.polygons)}")
                print(f"  Location: {obj_instance.matrix_world.translation}")
                print(f"  Materials: {len(obj.material_slots)}")
                
            elif obj.type == 'LIGHT':
                light_count += 1
                light = obj.data
                print(f"Light: {obj.name}")
                print(f"  Type: {light.type}")
                print(f"  Energy: {light.energy}")
                print(f"  Color: {light.color}")
                print(f"  Location: {obj_instance.matrix_world.translation}")
        
        print(f"\nTotal: {mesh_count} meshes, {light_count} lights")
        
        # ワールド設定
        if scene.world:
            print(f"World background: {scene.world.color}")
        
        print("=== End Scene Info ===\n")
        print("Raytracing...")

        # タイルサイズ（小さいタイルで分割してレンダリング）
        tile_size = 64
        
        # タイルごとにレンダリング
        for tile_y in range(0, height, tile_size):
            for tile_x in range(0, width, tile_size):
                # キャンセルチェック
                if self.test_break():
                    return
                
                # タイルのサイズを計算
                tw = min(tile_size, width - tile_x)
                th = min(tile_size, height - tile_y)
                
                # タイルのレンダリング結果を取得
                result = self.begin_result(tile_x, tile_y, tw, th)
                rlayer = result.layers[0]
                combined = rlayer.passes["Combined"]
                
                # このタイルだけをレンダリング
                pixels = self.render_tile(depsgraph, tile_x, tile_y, tw, th, width, height)
                combined.rect = pixels
                
                # 結果を更新（これでUIに表示される）
                self.end_result(result)
                
                # 進捗を更新
                progress = ((tile_y * width + tile_x * th) / (width * height))
                self.update_progress(progress)
        
        print("Render complete!")

    def render_tile(self, depsgraph, offset_x, offset_y, tile_width, tile_height, full_width, full_height):
        """指定されたタイル領域のみをレンダリング"""
        scene = depsgraph.scene_eval
        camera = scene.camera
        
        if not camera:
            return [[1.0, 0.0, 0.0, 1.0] for _ in range(tile_width * tile_height)]
        
        # カメラ情報
        camera_matrix = camera.matrix_world
        camera_location = camera_matrix.translation
        camera_data = camera.data
        camera_rotation = camera_matrix.to_3x3()
        
        # 視野角の計算
        import math
        sensor_width = camera_data.sensor_width
        focal_length = camera_data.lens
        fov = 2.0 * math.atan(sensor_width / (2.0 * focal_length))
        scale = math.tan(fov / 2.0)
        
        # アスペクト比
        aspect_ratio = full_width / full_height if full_height > 0 else 1.0
        
        # シーンオブジェクトを収集（一度だけ、事前計算を含む）
        scene_objects = []
        for obj_instance in depsgraph.object_instances:
            obj = obj_instance.object
            if obj.type == 'MESH':
                mesh = obj.evaluated_get(depsgraph).data
                matrix = obj_instance.matrix_world
                
                # マテリアル色を事前に取得
                material = None
                if len(obj.material_slots) > 0:
                    material = obj.material_slots[0].material
                material_color = self.get_material_color(material)
                
                # バウンディングボックスを計算
                if len(mesh.vertices) > 0:
                    bbox_min = Vector((float('inf'), float('inf'), float('inf')))
                    bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
                    for v in mesh.vertices:
                        bbox_min.x = min(bbox_min.x, v.co.x)
                        bbox_min.y = min(bbox_min.y, v.co.y)
                        bbox_min.z = min(bbox_min.z, v.co.z)
                        bbox_max.x = max(bbox_max.x, v.co.x)
                        bbox_max.y = max(bbox_max.y, v.co.y)
                        bbox_max.z = max(bbox_max.z, v.co.z)
                else:
                    continue
                
                # 法線変換用の行列（逆行列の転置）
                matrix_normal = matrix.inverted().transposed().to_3x3()
                
                scene_objects.append({
                    'mesh': mesh,
                    'matrix_inv': matrix.inverted(),
                    'matrix_normal': matrix_normal,
                    'material_color': material_color,
                    'bbox_min': bbox_min,
                    'bbox_max': bbox_max
                })
        
        # 背景色
        bg_color = Vector((0.2, 0.2, 0.2))
        if scene.world and scene.world.use_nodes:
            for node in scene.world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    color = node.inputs['Color'].default_value
                    bg_color = Vector((color[0], color[1], color[2]))
                    break
        elif scene.world:
            color = scene.world.color
            bg_color = Vector((color[0], color[1], color[2]))
        
        # タイル内のピクセルをレンダリング
        pixels = []
        
        for local_y in range(tile_height):
            y = offset_y + local_y
            py = 1.0 - 2.0 * y / full_height
            
            for local_x in range(tile_width):
                x = offset_x + local_x
                px = (2.0 * x / full_width - 1.0) * aspect_ratio
                
                # レイの方向（カメラ空間）
                ray_dir_local = Vector((px * scale, py * scale, -1.0))
                ray_dir_local.normalize()
                
                # ワールド空間に変換
                ray_direction = camera_rotation @ ray_dir_local
                
                # レイトレース（法線を返す）
                normal = self.trace_ray(camera_location, ray_direction, scene_objects)
                
                if normal:
                    # 法線を色に変換（-1~1 -> 0~1）
                    r = normal.x * 0.5 + 0.5
                    g = normal.y * 0.5 + 0.5
                    b = normal.z * 0.5 + 0.5
                    pixels.append([r, g, b, 1.0])
                else:
                    pixels.append([bg_color.x, bg_color.y, bg_color.z, 1.0])
        
        return pixels

    def render_viewport_tile(self, depsgraph, width, height, region_data):
        """ビューポートの視点（region_data）を使って法線色をレンダリング（低解像度全体）"""
        scene = depsgraph.scene_eval
        import math
        # region_data.view_matrix は World->View なので反転で View->World
        view_matrix = region_data.view_matrix.inverted()
        camera_location = view_matrix.translation
        camera_rotation = view_matrix.to_3x3()  # カメラ空間→ワールド空間
        is_persp = getattr(region_data, 'is_perspective', True)

        # FOVの推定（ビューポート固有値が複雑なので簡易に固定60度）
        if is_persp:
            fov = math.radians(60.0)
            scale = math.tan(fov / 2.0)
        else:
            # オルソの場合はスケールを固定（視距離に依存しない簡易版）
            scale = 1.0

        aspect_ratio = width / height if height > 0 else 1.0

        # シーンオブジェクト準備（法線行列含む）
        scene_objects = []
        for obj_instance in depsgraph.object_instances:
            obj = obj_instance.object
            if obj.type != 'MESH':
                continue
            mesh = obj.evaluated_get(depsgraph).data
            matrix = obj_instance.matrix_world
            if len(mesh.vertices) == 0:
                continue
            bbox_min = Vector((float('inf'), float('inf'), float('inf')))
            bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
            for v in mesh.vertices:
                co = v.co
                bbox_min.x = min(bbox_min.x, co.x)
                bbox_min.y = min(bbox_min.y, co.y)
                bbox_min.z = min(bbox_min.z, co.z)
                bbox_max.x = max(bbox_max.x, co.x)
                bbox_max.y = max(bbox_max.y, co.y)
                bbox_max.z = max(bbox_max.z, co.z)
            matrix_normal = matrix.inverted().transposed().to_3x3()
            scene_objects.append({
                'mesh': mesh,
                'matrix_inv': matrix.inverted(),
                'matrix_normal': matrix_normal,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max
            })

        # 背景色（プレビュー用簡易）
        bg_color = Vector((0.15, 0.15, 0.15))
        if scene.world:
            color = scene.world.color
            bg_color = Vector((color[0], color[1], color[2]))

        pixels = []
        # Blender座標系: X右, Y前(奥行き方向 +Y), Z上
        # NDCからローカルカメラ空間へ: 画面上はY下が増えるので反転して +Y を前方に近い向きへ維持
        for y in range(height):
            # 修正: プレビューが上下反転していたため Y を反転
            # 本来上を +1 としたいが表示結果が逆なので符号を反転して補正
            py_ndc = -(1.0 - 2.0 * ((y + 0.5) / height))
            for x in range(width):
                px_ndc = (2.0 * x / width - 1.0) * aspect_ratio
                if is_persp:
                    # 前方は -Z （Blenderカメラローカル）
                    ray_dir_local = Vector((px_ndc * scale, py_ndc * scale, -1.0))
                    ray_dir_local.normalize()
                    ray_direction = camera_rotation @ ray_dir_local
                else:
                    # オルソ：方向一定、原点をずらして擬似的に平行投影
                    ray_direction = camera_rotation @ Vector((0.0, 0.0, -1.0))
                    ray_dir_local = Vector((px_ndc, py_ndc, 0.0))
                    # オルソの場合はレイ原点を平面上にシフト（位置補正）
                    camera_location_shifted = camera_location + camera_rotation @ Vector((px_ndc, py_ndc, 0.0))
                    hit_normal = self.trace_ray(camera_location_shifted, ray_direction, scene_objects)
                    if hit_normal:
                        r = hit_normal.x * 0.5 + 0.5
                        g = hit_normal.y * 0.5 + 0.5
                        b = hit_normal.z * 0.5 + 0.5
                        pixels.append([r, g, b, 1.0])
                    else:
                        pixels.append([bg_color.x, bg_color.y, bg_color.z, 1.0])
                    continue

                hit_normal = self.trace_ray(camera_location, ray_direction, scene_objects)
                if hit_normal:
                    r = hit_normal.x * 0.5 + 0.5
                    g = hit_normal.y * 0.5 + 0.5
                    b = hit_normal.z * 0.5 + 0.5
                    pixels.append([r, g, b, 1.0])
                else:
                    pixels.append([bg_color.x, bg_color.y, bg_color.z, 1.0])

        # 初回デバッグ出力（負荷軽減のため一度だけ）
        if not hasattr(self, '_debug_axis_printed'):
            self._debug_axis_printed = True
            print('[DIYRenderer] Viewport basis (world):')
            print('  Right (X):', camera_rotation @ Vector((1,0,0)))
            print('  Up    (Z):', camera_rotation @ Vector((0,0,1)))
            print('  Fwd  (-Z):', camera_rotation @ Vector((0,0,-1)))
            print('  Sample ray dir (center):', camera_rotation @ Vector((0,0,-1)))
        return pixels

    def view_update(self, context, depsgraph):
        """ビューポートでシーンが更新されたときに呼ばれる"""
        # テクスチャキャッシュをクリア（シーン変更時）
        if hasattr(self, 'texture'):
            del self.texture
            self.texture = None
        # キャッシュをクリア
        if hasattr(self, 'viewport_pixels_cache'):
            self.viewport_pixels_cache = None

    def view_draw(self, context, depsgraph):
        """ビューポートに描画するときに呼ばれる"""
        region = context.region
        width = region.width
        height = region.height
        
        # GPU描画用のバッファを使用
        import gpu
        from gpu_extras.presets import draw_texture_2d
        
        # 更新頻度を制限（フレームカウンター）
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        
        self.frame_counter += 1
        
        # 非常に低解像度でプレビュー（1/8サイズ、固定）
        render_width = max(80, width // 8)
        render_height = max(60, height // 8)
        
        # テクスチャのサイズが変わった場合、またはキャッシュがない場合は再生成
        needs_update = (
            not hasattr(self, 'texture') or 
            self.texture is None or 
            not hasattr(self, 'viewport_pixels_cache') or
            self.viewport_pixels_cache is None or
            self.frame_counter % 30 == 0  # 30フレームごとに更新
        )
        
        # サイズ変更時は必ず更新
        if hasattr(self, 'texture') and self.texture is not None:
            if self.texture.width != render_width or self.texture.height != render_height:
                needs_update = True
        
        if needs_update:
            # ビューポート視点を使用
            region_data = context.region_data
            if region_data is not None:
                pixels = self.render_viewport_tile(depsgraph, render_width, render_height, region_data)
            else:
                # フォールバック（従来のカメラ）
                pixels = self.render_tile(depsgraph, 0, 0, render_width, render_height, render_width, render_height)
            self.viewport_pixels_cache = pixels
            
            # 平坦化（バッファに必要な形式）
            flat_pixels = []
            for pixel in pixels:
                flat_pixels.extend(pixel)
            
            # Bufferに変換
            buffer = gpu.types.Buffer('FLOAT', render_width * render_height * 4, flat_pixels)
            
            # 古いテクスチャを削除
            if hasattr(self, 'texture') and self.texture is not None:
                del self.texture
            
            # 新しいテクスチャを作成
            self.texture = gpu.types.GPUTexture((render_width, render_height), format='RGBA16F', data=buffer)
        
        # テクスチャを描画（拡大して表示）
        draw_texture_2d(self.texture, (0, 0), width, height)


def register():
    bpy.utils.register_class(DIYRenderEngine)


def unregister():
    bpy.utils.unregister_class(DIYRenderEngine)