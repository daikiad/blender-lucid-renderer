"""
DIY Render Engine - Main render engine class for Blender integration.

このファイルは Blender レンダーエンジンの中核です。
F12 レンダリングとビューポートレンダリングの両方を処理します。

処理フロー:
1. Blender が render() または view_update()/view_draw() を呼び出す
2. シーンをエクスポート (scene_export.py)
3. pybind11 で C++ レンダラーを直接呼び出し
4. 結果を Blender に返す

主要クラス:
- DIYRenderEngine: bpy.types.RenderEngine のサブクラス

レンダリングモード:
- pybind11モード: C++ を直接呼び出し（推奨、即座のキャンセル可能）
- レガシーモード: 毎回プロセス起動（互換性用フォールバック）
"""

import os
import sys
import math
import time
import array
import hashlib
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import bpy
from mathutils import Vector

from .scene_export import export_scene_to_file, get_scene_cache
from .renderer import call_external_renderer, compute_camera_params

# =============================================================================
# pybind11 モジュールのインポート
# =============================================================================
# pybind11 でビルドした C++ レンダラーモジュールをインポート
# ビルドされていない場合は None になり、サーバーモードにフォールバック

_addon_dir = os.path.dirname(os.path.abspath(__file__))
_pybind_paths = [
    os.path.join(_addon_dir, 'cpp_renderer', 'build_pybind'),
    os.path.join(_addon_dir, 'cpp_renderer', 'build'),
]
for _path in _pybind_paths:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import diyrenderer
    PYBIND_AVAILABLE = True
    print(f"[DIYRenderer] pybind11 module loaded: version {diyrenderer.__version__}, OpenMP={diyrenderer.openmp_enabled}")
except ImportError as e:
    diyrenderer = None
    PYBIND_AVAILABLE = False
    print(f"[DIYRenderer] pybind11 module not available: {e}")


# =============================================================================
# グローバル状態
# =============================================================================
# レンダラーの状態を管理するグローバル変数

# pybind11 モード用
_pybind_renderer = None  # diyrenderer.Renderer インスタンス
_pybind_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None  # レンダリング用スレッドプール
_pybind_future: Optional[concurrent.futures.Future] = None  # 現在のレンダリングタスク
_pybind_scene_hash: Optional[str] = None  # シーン変更検出用ハッシュ
_pybind_job_id: int = 0  # ジョブ ID（古い結果を破棄するため）


def get_pybind_renderer():
    """Get or create the global pybind11 renderer instance."""
    global _pybind_renderer, _pybind_executor
    
    if not PYBIND_AVAILABLE:
        return None
    
    if _pybind_renderer is None:
        _pybind_renderer = diyrenderer.Renderer()
        _pybind_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        print("[DIYRenderer] pybind11 renderer created")
    
    return _pybind_renderer


def stop_pybind_renderer():
    """Stop and cleanup the pybind11 renderer."""
    global _pybind_renderer, _pybind_executor, _pybind_future, _pybind_scene_hash
    
    if _pybind_renderer is not None:
        _pybind_renderer.cancel()
    
    if _pybind_future is not None:
        try:
            _pybind_future.result(timeout=1.0)
        except Exception:
            pass
        _pybind_future = None
    
    if _pybind_executor is not None:
        _pybind_executor.shutdown(wait=False)
        _pybind_executor = None
    
    _pybind_renderer = None
    _pybind_scene_hash = None
    print("[DIYRenderer] pybind11 renderer stopped")


# =============================================================================
# DIYRenderEngine クラス
# =============================================================================

class DIYRenderEngine(bpy.types.RenderEngine):
    """
    Blender カスタムレンダーエンジン。
    
    Blender に登録される RenderEngine のサブクラスです。
    F12 レンダリング (render) とビューポートレンダリング (view_update/view_draw)
    の両方を実装しています。
    
    Attributes:
        bl_idname: エンジンの内部識別子
        bl_label: UI に表示される名前
        bl_use_preview: マテリアルプレビューをサポート
        bl_use_shading_nodes: シェーディングノードをサポート
    """
    bl_idname = "DIY_RENDER_MINIMAL"
    bl_label = "DIY Renderer (Minimal)"
    bl_use_preview = True
    bl_use_shading_nodes = True
    bl_use_shading_nodes_custom = False

    def _init_pybind_viewport(self):
        """
        pybind11 ビューポートレンダリング用の初期化。
        
        view_draw から呼び出され、インスタンス変数を初期化します。
        """
        global _pybind_executor, _pybind_renderer
        
        # Executor の初期化
        if _pybind_executor is None:
            _pybind_executor = ThreadPoolExecutor(max_workers=1)
        
        # レンダラーの初期化
        if _pybind_renderer is None:
            get_pybind_renderer()
        
        # インスタンス変数の初期化
        if not hasattr(self, '_pybind_viewport_future'):
            self._pybind_viewport_future = None
        if not hasattr(self, '_pybind_viewport_job_id'):
            self._pybind_viewport_job_id = 0
        if not hasattr(self, 'last_camera_matrix'):
            self.last_camera_matrix = None
        if not hasattr(self, 'last_view_perspective'):
            self.last_view_perspective = None
        if not hasattr(self, 'last_view_distance'):
            self.last_view_distance = None
        # 非同期シーンエクスポート用
        if not hasattr(self, '_export_future'):
            self._export_future = None
        if not hasattr(self, '_pending_export_data'):
            self._pending_export_data = None  # (cam_params, render_width, render_height, is_editing, ...)

    def _render_viewport_pybind(self, context, depsgraph, render_width, render_height,
                                 cam_params, samples, max_depth, algorithm, debug_mode,
                                 scene_file):
        """
        pybind11 を使用してビューポートをレンダリング。
        
        協調キャンセルに対応した非同期レンダリングを行います。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
            render_width, render_height: レンダリング解像度
            cam_params: カメラパラメータ
            samples: サンプル数
            max_depth: 最大バウンス数
            algorithm: アルゴリズム名
            debug_mode: デバッグモード名（None で通常レンダリング）
            scene_file: シーンファイルのパス（メインスレッドでエクスポート済み）
        
        Returns:
            (pixels, job_id) または None
        """
        global _pybind_renderer, _pybind_executor, _pybind_scene_hash, _pybind_job_id
        
        renderer = get_pybind_renderer()
        if renderer is None:
            return None
        
        # シーンファイルが渡されていない場合はエラー
        if not scene_file:
            print("[DIYRenderer] No scene file provided")
            return None
        
        # JSON を読み込み
        try:
            with open(scene_file, 'r') as f:
                scene_json = f.read()
        except Exception as e:
            print(f"[DIYRenderer] Failed to read scene: {e}")
            return None
        
        # シーンハッシュをチェック（hashlib は既にファイル先頭でインポート済み）
        scene_hash = hashlib.md5(scene_json.encode()).hexdigest()
        
        if scene_hash != _pybind_scene_hash:
            # シーンを更新
            if not renderer.load_scene_json(scene_json):
                print("[DIYRenderer] Failed to load scene into pybind renderer")
                return None
            _pybind_scene_hash = scene_hash
        
        # カメラを設定
        pos = cam_params['pos']
        dir_ = cam_params['dir']
        up = cam_params['up']
        fov = cam_params['fov']
        renderer.set_camera(
            pos[0], pos[1], pos[2],
            dir_[0], dir_[1], dir_[2],
            up[0], up[1], up[2],
            fov
        )
        
        # アルゴリズムを設定
        renderer.set_algorithm(algorithm)
        
        # サンプルオフセットを取得（pybind モードでは pybind_accumulated_samples を使用）
        tile_key = f"{render_width}x{render_height}"
        sample_offset = 0
        if hasattr(self, 'pybind_accumulated_samples') and tile_key in self.pybind_accumulated_samples:
            sample_offset = self.pybind_accumulated_samples[tile_key][1]
        
        # レンダリング実行（GIL 解放されるので他のスレッドも動ける）
        if debug_mode and debug_mode != 'NONE':
            debug_map = {'NORMAL': 'normal', 'ALBEDO': 'albedo', 'EMISSION': 'emission'}
            mode = debug_map.get(debug_mode, 'normal')
            pixels = renderer.render_debug(
                0, 0, render_width, render_height,
                render_width, render_height,
                mode
            )
        else:
            pixels = renderer.render_tile(
                0, 0, render_width, render_height,
                render_width, render_height,
                samples=samples,
                sample_offset=sample_offset,
                max_depth=max_depth
            )
        
        # キャンセルされたかチェック
        cancelled = renderer.is_cancelled()
        
        _pybind_job_id += 1
        
        return {
            'pixels': pixels,
            'width': render_width,
            'height': render_height,
            'samples': samples,
            'cancelled': cancelled,
            'job_id': _pybind_job_id
        }

    def _start_pybind_viewport_render(self, context, depsgraph, render_width, render_height,
                                       cam_params, samples, max_depth, algorithm, debug_mode,
                                       scene_file, cancel_previous=True):
        """
        pybind11 を使用したビューポートレンダリングを非同期で開始。
        
        Args:
            scene_file: メインスレッドでエクスポート済みのシーンファイルパス
            cancel_previous: 前回のレンダリングをキャンセルするか。
                            編集中モードではFalseにして完了を待つ。
        """
        global _pybind_executor, _pybind_renderer
        
        renderer = get_pybind_renderer()
        if renderer is None or _pybind_executor is None:
            return
        
        # 前回のレンダリングをキャンセル（指定された場合のみ）
        if cancel_previous:
            renderer.cancel()
        
        # 前回の Future が完了するのを待つ（短時間で終わるはず）
        if self._pybind_viewport_future is not None:
            try:
                self._pybind_viewport_future.result(timeout=0.1)
            except Exception:
                pass
            self._pybind_viewport_future = None
        
        # 新しいジョブを開始
        self._pybind_viewport_job_id += 1
        
        def render_task():
            return self._render_viewport_pybind(
                context, depsgraph, render_width, render_height,
                cam_params, samples, max_depth, algorithm, debug_mode,
                scene_file
            )
        
        self._pybind_viewport_future = _pybind_executor.submit(render_task)

    def _poll_pybind_viewport_result(self):
        """
        pybind11 ビューポートレンダリングの結果をポーリング。
        
        Returns:
            完了していれば result dict、未完了なら None
        """
        if self._pybind_viewport_future is None:
            return None
        
        if not self._pybind_viewport_future.done():
            return None
        
        try:
            result = self._pybind_viewport_future.result()
            self._pybind_viewport_future = None
            return result
        except Exception as e:
            print(f"[DIYRenderer] pybind viewport render error: {e}")
            import traceback
            traceback.print_exc()
            self._pybind_viewport_future = None
            return None

    def _view_draw_pybind(self, context, depsgraph):
        """
        pybind11 を使用したビューポートレンダリング。
        
        協調キャンセル方式により、カメラ移動時に即座にキャンセル可能。
        """
        import gpu
        from gpu_extras.presets import draw_texture_2d
        
        region = context.region
        width = region.width
        height = region.height
        target_samples = context.scene.diy_renderer.viewport_samples
        current_time = time.time()
        
        # 初期化
        if not hasattr(self, 'pybind_viewport_last_change_time'):
            self.pybind_viewport_last_change_time = current_time
            self.pybind_last_render_width = 0
            self.pybind_last_render_height = 0
            self.pybind_accumulated_samples = {}
            self.pybind_texture = None
            self.pybind_texture_width = 0
            self.pybind_texture_height = 0
            self._pending_camera_update = False  # カメラ更新待ちフラグ
            self._was_moving = False  # 前フレームで移動中だったか
            self._scene_update_pending = False  # view_update からのシーン変更フラグ
        
        # view_update からのシーン変更フラグをチェック
        scene_update_from_view_update = getattr(self, '_scene_update_pending', False)
        if scene_update_from_view_update:
            self._scene_update_pending = False  # フラグをクリア
        
        # 前回のフレームで最終プレビュー中だったかを記録（変更検出前に）
        time_since_last_change = current_time - self.pybind_viewport_last_change_time
        was_in_final_mode = time_since_last_change >= 0.3
        
        # シーン変更検出（カメラ、オブジェクト、マテリアルなど）
        scene_changed_from_depsgraph, content_changed = self._detect_scene_change(context, depsgraph)
        
        # view_update からの変更も考慮
        scene_changed = scene_changed_from_depsgraph or scene_update_from_view_update
        if scene_update_from_view_update:
            content_changed = True  # view_update はシーン内容変更を示す
        
        if scene_changed:
            self.pybind_viewport_last_change_time = current_time
            self.pybind_accumulated_samples = {}
            
            # 最終プレビューモード中に変更された場合のみキャンセル
            if was_in_final_mode:
                renderer = get_pybind_renderer()
                if renderer is not None:
                    renderer.cancel()
            
            # 次のレンダリングが必要なことを記録
            self._pending_camera_update = True
        
        # シーン内容変更フラグを保存（シーンエクスポートの判断に使用）
        self._content_changed = content_changed
        
        time_since_change = current_time - self.pybind_viewport_last_change_time
        
        # シーン設定から解像度スケールを取得
        diy = context.scene.diy_renderer
        editing_scale = diy.viewport_scale_editing
        final_scale = diy.viewport_scale_final
        
        # 解像度とパラメータを決定
        # ★ シーン内容が変更された場合は、時間に関係なく編集中モードとして扱う
        #    （シーンエクスポートに時間がかかるため、時間ベースの判定だけでは不十分）
        if scene_changed or time_since_change < 0.3:
            # 編集中: 低解像度で高速応答
            scale_factor = editing_scale
            samples_per_iteration = 1
            viewport_bounces = 4
            is_editing = True
        else:
            # 最終プレビュー: 高解像度で品質重視
            scale_factor = final_scale
            samples_per_iteration = 1
            viewport_bounces = 8
            is_editing = False
        
        render_width = max(1, width // scale_factor)
        render_height = max(1, height // scale_factor)
        
        # 解像度変更検出
        resolution_changed = (self.pybind_last_render_width != render_width or 
                              self.pybind_last_render_height != render_height)
        if resolution_changed:
            self.pybind_accumulated_samples = {}
        
        tile_key = f"{render_width}x{render_height}"
        if tile_key in self.pybind_accumulated_samples:
            current_sample_count = self.pybind_accumulated_samples[tile_key][1]
        else:
            current_sample_count = 0
        
        # 現在レンダリング中かどうか
        rendering_in_progress = (self._pybind_viewport_future is not None and 
                                 not self._pybind_viewport_future.done())
        
        # ★まず結果をポーリング（新しいレンダリング開始前に行う）
        result = self._poll_pybind_viewport_result()
        if result is not None:
            # レンダリング完了時刻を記録
            if not hasattr(self, '_last_render_complete_time'):
                self._last_render_complete_time = 0
            self._last_render_complete_time = current_time
            
            ext_pixels = result.get('pixels')
            result_width = result.get('width', 0)
            result_height = result.get('height', 0)
            result_samples = result.get('samples', 1)
            was_cancelled = result.get('cancelled', False)
            
            expected_len = result_width * result_height * 4
            
            # キャンセルされていない完了結果のみテクスチャを更新
            if not was_cancelled and ext_pixels and len(ext_pixels) == expected_len:
                # カメラ更新待ちがなければサンプル累積
                if not self._pending_camera_update:
                    result_tile_key = f"{result_width}x{result_height}"
                    if result_tile_key in self.pybind_accumulated_samples:
                        acc_array, prev_count = self.pybind_accumulated_samples[result_tile_key]
                        new_count = prev_count + result_samples
                        for i in range(len(acc_array)):
                            acc_array[i] += ext_pixels[i]
                        self.pybind_accumulated_samples[result_tile_key] = (acc_array, new_count)
                        inv_count = 1.0 / new_count
                        display_pixels = [v * inv_count for v in acc_array]
                    else:
                        acc_array = array.array('f', ext_pixels)
                        self.pybind_accumulated_samples[result_tile_key] = (acc_array, result_samples)
                        inv_samples = 1.0 / result_samples
                        display_pixels = [v * inv_samples for v in ext_pixels]
                else:
                    # カメラ更新待ち中は累積せず、この結果を表示
                    inv_samples = 1.0 / result_samples
                    display_pixels = [v * inv_samples for v in ext_pixels]
                
                # テクスチャ更新
                buffer = gpu.types.Buffer('FLOAT', expected_len, display_pixels)
                if self.pybind_texture is not None:
                    try:
                        del self.pybind_texture
                    except Exception:
                        pass
                self.pybind_texture = gpu.types.GPUTexture(
                    (result_width, result_height), format='RGBA16F', data=buffer
                )
                self.pybind_texture_width = result_width
                self.pybind_texture_height = result_height
        
        # ★非同期エクスポートの完了をチェックしてレンダリング開始
        if hasattr(self, '_export_future') and self._export_future is not None:
            if self._export_future.done():
                try:
                    scene_file = self._export_future.result()
                    self._export_future = None
                    
                    if scene_file and self._pending_export_data:
                        data = self._pending_export_data
                        self._pending_export_data = None
                        self._start_pybind_viewport_render(
                            context, depsgraph, 
                            data['render_width'], data['render_height'],
                            data['cam_params'], data['samples'], 
                            data['max_bounces'], data['algorithm'], data['debug_mode'],
                            scene_file, cancel_previous=not data['is_editing']
                        )
                except Exception as e:
                    print(f"[DIYRenderer] Async export error: {e}")
                    self._export_future = None
                    self._pending_export_data = None
        
        # rendering_in_progress を再計算（ポーリング後に Future が None になっている可能性）
        rendering_in_progress = (self._pybind_viewport_future is not None and 
                                 not self._pybind_viewport_future.done())
        
        # 新しいレンダリングが必要か判定
        # 編集中の場合は、エクスポートのスロットリング間隔（200ms）を考慮
        last_render_complete = getattr(self, '_last_render_complete_time', 0)
        time_since_render_complete = current_time - last_render_complete
        
        last_export_time = getattr(self, '_last_scene_export_time', 0)
        time_since_last_export = current_time - last_export_time
        
        RENDER_COOLDOWN = 0.05  # 50ms
        EXPORT_THROTTLE_INTERVAL = 0.2  # 200ms
        
        # 非同期エクスポート中かどうか
        export_in_progress = (hasattr(self, '_export_future') and 
                              self._export_future is not None and 
                              not self._export_future.done())
        
        if is_editing:
            # 編集中は以下の条件でレンダリング開始を制御:
            # - テクスチャがない（初回）
            # - エクスポート/レンダリング中でない
            # - スロットリング間隔を経過
            if self.pybind_texture is None and not export_in_progress:
                needs_new_render = True
            elif export_in_progress or rendering_in_progress:
                needs_new_render = False
            elif time_since_render_complete < RENDER_COOLDOWN:
                needs_new_render = False
            elif self._content_changed and time_since_last_export < EXPORT_THROTTLE_INTERVAL:
                needs_new_render = False
            else:
                needs_new_render = True
        else:
            needs_new_render = (
                self.pybind_texture is None or
                self._pending_camera_update or  # カメラ更新待ち
                (not rendering_in_progress and current_sample_count < target_samples)
            )
        
        # ★新しいレンダリングを開始（レンダリング中でなければ）
        if needs_new_render and not rendering_in_progress:
            # カメラ更新待ちをクリア
            if self._pending_camera_update:
                self._pending_camera_update = False
                self.pybind_accumulated_samples = {}  # 累積もリセット
            
            self.pybind_last_render_width = render_width
            self.pybind_last_render_height = render_height
            
            region_data = context.region_data
            if region_data is not None:
                view_matrix_inv = region_data.view_matrix.inverted()
                
                cam_pos = view_matrix_inv.translation
                cam_dir = (view_matrix_inv.to_3x3() @ Vector((0, 0, -1))).normalized()
                cam_up = (view_matrix_inv.to_3x3() @ Vector((0, 1, 0))).normalized()
                
                if region_data.view_perspective == 'CAMERA':
                    camera = context.scene.camera
                    if camera and camera.data:
                        cam_data = camera.data
                        sensor_width = cam_data.sensor_width
                        focal_length = cam_data.lens
                        fov = math.degrees(2 * math.atan(sensor_width / (2 * focal_length)))
                    else:
                        fov = 50.0
                elif region_data.view_perspective == 'PERSP':
                    fov = 50.0
                else:
                    fov = 5.0
                
                cam_params = {
                    'pos': cam_pos,
                    'dir': cam_dir,
                    'up': cam_up,
                    'fov': fov
                }
                
                # シーンエクスポート（内容変更時のみ非同期、それ以外はキャッシュ）
                if not hasattr(self, '_last_scene_export_time'):
                    self._last_scene_export_time = 0
                
                diy = context.scene.diy_renderer
                debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
                max_bounces = viewport_bounces
                algorithm = diy.sampling_algorithm
                
                if self._content_changed:
                    # 内容が変わったので再エクスポート
                    # ★非同期エクスポート: UIをブロックしない
                    if self._export_future is None or self._export_future.done():
                        # 新しい非同期エクスポートを開始
                        self._pending_export_data = {
                            'cam_params': cam_params,
                            'render_width': render_width,
                            'render_height': render_height,
                            'is_editing': is_editing,
                            'samples': samples_per_iteration,
                            'max_bounces': max_bounces,
                            'algorithm': algorithm,
                            'debug_mode': debug_mode,
                        }
                        self._export_future = _pybind_executor.submit(
                            export_scene_to_file, depsgraph
                        )
                        self._last_scene_export_time = current_time
                    # エクスポート完了待ち → レンダリングは開始しない（既存テクスチャを維持）
                elif is_editing:
                    # カメラのみ変更で編集中: キャッシュを使用（同期、高速）
                    scene_file = get_scene_cache().get_cached_file_fast()
                    if not scene_file:
                        scene_file = export_scene_to_file(depsgraph)
                        self._last_scene_export_time = current_time
                    if scene_file:
                        self._start_pybind_viewport_render(
                            context, depsgraph, render_width, render_height,
                            cam_params, samples_per_iteration, max_bounces, algorithm, debug_mode,
                            scene_file, cancel_previous=False
                        )
                else:
                    # 最終プレビュー: 常に再エクスポート（同期）
                    scene_file = export_scene_to_file(depsgraph)
                    self._last_scene_export_time = current_time
                    if scene_file:
                        self._start_pybind_viewport_render(
                            context, depsgraph, render_width, render_height,
                            cam_params, samples_per_iteration, max_bounces, algorithm, debug_mode,
                            scene_file, cancel_previous=True
                        )
        
        # 再描画判定
        if tile_key in self.pybind_accumulated_samples:
            current_sample_count = self.pybind_accumulated_samples[tile_key][1]
        
        should_redraw = (
            rendering_in_progress or
            is_editing or
            current_sample_count < target_samples
        )
        
        if should_redraw:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        
        # テクスチャを描画
        if self.pybind_texture is not None:
            draw_texture_2d(self.pybind_texture, (0, 0), width, height)
        else:
            # テクスチャがまだない場合は、グレーのプレースホルダーを作成して描画
            # （初回レンダリング中や、レンダリング完了前の状態）
            if not hasattr(self, '_placeholder_texture') or self._placeholder_texture is None:
                placeholder_size = 64
                placeholder_pixels = [0.2, 0.2, 0.2, 1.0] * (placeholder_size * placeholder_size)
                placeholder_buffer = gpu.types.Buffer('FLOAT', len(placeholder_pixels), placeholder_pixels)
                self._placeholder_texture = gpu.types.GPUTexture(
                    (placeholder_size, placeholder_size), format='RGBA16F', data=placeholder_buffer
                )
            draw_texture_2d(self._placeholder_texture, (0, 0), width, height)

    def _render_gradient(self, width, height):
        """
        フォールバック用のグラデーションパターンを生成。
        
        シーンエクスポートに失敗した場合などに使用します。
        """
        pixels = []
        for y in range(height):
            fy = y / (height - 1) if height > 1 else 0.0
            for x in range(width):
                fx = x / (width - 1) if width > 1 else 0.0
                pixels.append([fx, fy, 0.2, 1.0])
        return pixels

    def _render_with_pybind(self, depsgraph, width, height, cam_params,
                            scene_file, target_samples, diy):
        """
        pybind11 を使用した F12 レンダリング。
        
        協調キャンセル方式により、即座にキャンセル可能。
        プログレッシブレンダリングを実装。
        
        Args:
            depsgraph: Blender の依存関係グラフ
            width, height: 出力解像度
            cam_params: カメラパラメータ dict
            scene_file: JSON シーンファイルパス
            target_samples: 目標サンプル数
            diy: DIY Renderer 設定オブジェクト
        
        Returns:
            成功時は True、失敗時は None
        """
        global _pybind_renderer, _pybind_scene_hash
        
        renderer = get_pybind_renderer()
        if renderer is None:
            return None
        
        # シーンを読み込み
        try:
            with open(scene_file, 'r') as f:
                scene_json = f.read()
        except Exception as e:
            print(f"[DIYRenderer] Failed to read scene: {e}")
            return None
        
        # シーンハッシュをチェック
        scene_hash = hashlib.md5(scene_json.encode()).hexdigest()
        
        if scene_hash != _pybind_scene_hash:
            if not renderer.load_scene_json(scene_json):
                print("[DIYRenderer] Failed to load scene into pybind renderer")
                return None
            _pybind_scene_hash = scene_hash
        
        # カメラを設定
        pos = cam_params['pos']
        dir_ = cam_params['dir']
        up = cam_params['up']
        fov = cam_params['fov']
        renderer.set_camera(
            pos[0], pos[1], pos[2],
            dir_[0], dir_[1], dir_[2],
            up[0], up[1], up[2],
            fov
        )
        
        # アルゴリズムを設定
        renderer.set_algorithm(diy.sampling_algorithm)
        
        # プログレッシブレンダリング用のサンプル分割
        sample_iterations = []
        current = 1
        total = 0
        max_increment = 128
        while total < target_samples:
            to_add = min(current, target_samples - total)
            sample_iterations.append(to_add)
            total += to_add
            if current < max_increment:
                current *= 2
        
        print(f"[DIYRenderer] pybind11 - sample iterations: {sample_iterations}")
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        render_start_time = time.time()
        
        def check_cancel():
            return self.test_break() or self._render_cancelled
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
            else:
                return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
        
        for iteration_samples in sample_iterations:
            if check_cancel():
                renderer.cancel()
                print("[DIYRenderer] Render cancelled by user")
                break
            
            elapsed = time.time() - render_start_time
            if total_samples > 0:
                time_per_sample = elapsed / total_samples
                remaining_time = time_per_sample * (max_samples - total_samples)
                time_str = f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining_time)}"
            else:
                time_str = f"Elapsed: {format_time(elapsed)}"
            
            self.update_progress(total_samples / max_samples)
            self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples | {time_str}")
            
            # レンダリング実行
            debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
            
            if debug_mode and debug_mode != 'NONE':
                debug_map = {'NORMAL': 'normal', 'ALBEDO': 'albedo', 'EMISSION': 'emission'}
                mode = debug_map.get(debug_mode, 'normal')
                iteration_pixels = renderer.render_debug(
                    0, 0, width, height, width, height, mode
                )
            else:
                iteration_pixels = renderer.render_tile(
                    0, 0, width, height, width, height,
                    samples=iteration_samples,
                    sample_offset=total_samples,
                    max_depth=diy.max_bounces
                )
            
            if renderer.is_cancelled():
                print("[DIYRenderer] Render was cancelled")
                break
            
            expected_len = width * height * 4
            if len(iteration_pixels) != expected_len:
                print(f"[DIYRenderer] Invalid pixel count: {len(iteration_pixels)} vs {expected_len}")
                continue
            
            # ピクセルデータを累積
            if accumulated_pixels is None:
                accumulated_pixels = array.array('f', iteration_pixels)
                total_samples = iteration_samples
            else:
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i] += iteration_pixels[i]
                total_samples += iteration_samples
            
            # 平均化して Blender に表示
            inv_samples = 1.0 / total_samples
            display_pixels = []
            for i in range(0, len(accumulated_pixels), 4):
                display_pixels.append([
                    accumulated_pixels[i] * inv_samples,
                    accumulated_pixels[i+1] * inv_samples,
                    accumulated_pixels[i+2] * inv_samples,
                    1.0
                ])
            
            result_obj = self.begin_result(0, 0, width, height)
            combined = result_obj.layers[0].passes["Combined"]
            combined.rect = display_pixels
            self.end_result(result_obj)
        
        total_elapsed = time.time() - render_start_time
        print(f"[DIYRenderer] pybind11 render complete ({total_samples} samples) in {format_time(total_elapsed)}")
        return True

    # =========================================================================
    # メインレンダリングメソッド
    # =========================================================================

    def render(self, depsgraph):
        """
        F12 レンダリングのメインエントリーポイント。
        
        Blender がレンダリングを開始するとこのメソッドが呼ばれます。
        pybind11 モードを優先し、フォールバックとしてレガシーモードを使用します。
        プログレッシブレンダリングとキャンセル機能を実装しています。
        
        Args:
            depsgraph: Blender の依存関係グラフ (評価済みシーンを含む)
        """
        # ---------------------------------------------------------------------
        # Phase 1: 基本パラメータの取得
        # ---------------------------------------------------------------------
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        
        self._render_cancelled = False
        
        original_scene = depsgraph.scene
        target_samples = original_scene.diy_renderer.samples
        diy = original_scene.diy_renderer
        
        print(f"[DIYRenderer] Starting render ({width} x {height}, samples: {target_samples}, pybind11: {PYBIND_AVAILABLE})")
        
        cam_params = compute_camera_params(scene, width, height)
        
        if not cam_params:
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = self._render_gradient(width, height)
            self.end_result(result)
            return
        
        scene_file = export_scene_to_file(depsgraph, use_cache=False)
        if not scene_file:
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = self._render_gradient(width, height)
            self.end_result(result)
            return
        
        # pybind11 モードでのレンダリング
        if PYBIND_AVAILABLE:
            pybind_result = self._render_with_pybind(
                depsgraph, width, height, cam_params, 
                scene_file, target_samples, diy
            )
            if pybind_result is not None:
                try:
                    if scene_file and os.path.isfile(scene_file):
                        os.remove(scene_file)
                except Exception:
                    pass
                return
            print("[DIYRenderer] pybind11 render failed, falling back to legacy mode")
        
        # Legacy mode rendering
        # Generate sample iterations
        sample_iterations = []
        current = 1
        total = 0
        max_increment = 128
        while total < target_samples:
            to_add = min(current, target_samples - total)
            sample_iterations.append(to_add)
            total += to_add
            if current < max_increment:
                current *= 2
        
        print(f"[DIYRenderer] Sample iterations: {sample_iterations} (total: {sum(sample_iterations)})")
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        render_start_time = time.time()
        
        def check_cancel():
            return self.test_break() or self._render_cancelled
        
        def format_time(seconds):
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
            if check_cancel():
                print("[DIYRenderer] Render cancelled by user")
                break
            
            elapsed = time.time() - render_start_time
            if total_samples > 0:
                time_per_sample = elapsed / total_samples
                remaining_samples = max_samples - total_samples
                remaining_time = time_per_sample * remaining_samples
                time_str = f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining_time)}"
            else:
                time_str = f"Elapsed: {format_time(elapsed)}"
            
            progress = total_samples / max_samples
            self.update_progress(progress)
            self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples | {time_str}")
            
            print(f"[DIYRenderer] Rendering iteration with {iteration_samples} samples (total: {total_samples + iteration_samples})")
            
            diy = original_scene.diy_renderer
            debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
            max_bounces = diy.max_bounces
            algorithm = diy.sampling_algorithm
            
            iteration_pixels = call_external_renderer(
                scene_file, 0, 0, width, height, width, height, cam_params,
                samples=iteration_samples,
                depth=max_bounces,
                debug_mode=debug_mode,
                cancel_check=check_cancel,
                sample_offset=total_samples,
                algorithm=algorithm
            )
            
            if iteration_pixels is None and check_cancel():
                print("[DIYRenderer] Render cancelled during iteration")
                break
            
            expected_len = width * height * 4
            if not iteration_pixels or len(iteration_pixels) != expected_len:
                print(f"[DIYRenderer] Iteration failed, skipping (got {len(iteration_pixels) if iteration_pixels else 0}, expected {expected_len})")
                continue
            
            center_idx = ((height // 2) * width + (width // 2)) * 4
            if len(iteration_pixels) > center_idx + 2:
                print(f"[DIYRenderer] DEBUG iteration {idx}: samples={iteration_samples}, center_pixel_raw=({iteration_pixels[center_idx]:.4f}, {iteration_pixels[center_idx+1]:.4f}, {iteration_pixels[center_idx+2]:.4f})")
            
            if accumulated_pixels is None:
                accumulated_pixels = array.array('f', iteration_pixels)
                total_samples = iteration_samples
            else:
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i] += iteration_pixels[i]
                total_samples += iteration_samples
            
            if len(accumulated_pixels) > center_idx + 2:
                print(f"[DIYRenderer] DEBUG accumulated: total_samples={total_samples}, center_pixel_sum=({accumulated_pixels[center_idx]:.4f}, {accumulated_pixels[center_idx+1]:.4f}, {accumulated_pixels[center_idx+2]:.4f})")
                print(f"[DIYRenderer] DEBUG display value: ({accumulated_pixels[center_idx]/total_samples:.4f}, {accumulated_pixels[center_idx+1]/total_samples:.4f}, {accumulated_pixels[center_idx+2]/total_samples:.4f})")
            
            inv_samples = 1.0 / total_samples
            display_pixels = []
            for i in range(0, len(accumulated_pixels), 4):
                display_pixels.append([
                    accumulated_pixels[i] * inv_samples,
                    accumulated_pixels[i+1] * inv_samples,
                    accumulated_pixels[i+2] * inv_samples,
                    1.0
                ])
            
            result = self.begin_result(0, 0, width, height)
            combined = result.layers[0].passes["Combined"]
            combined.rect = display_pixels
            self.end_result(result)
            
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
        
        try:
            if scene_file and os.path.isfile(scene_file):
                os.remove(scene_file)
        except Exception:
            pass
        
        total_elapsed = time.time() - render_start_time
        print(f"[DIYRenderer] Progressive render complete ({total_samples} total samples) in {format_time(total_elapsed)}")

    def view_update(self, context, depsgraph):
        """Called when the scene is modified in viewport mode."""
        
        needs_reset = False
        
        for update in depsgraph.updates:
            obj = update.id
            
            if update.is_updated_geometry:
                needs_reset = True
                break
            
            if update.is_updated_transform:
                if hasattr(obj, 'type') and obj.type in {'MESH', 'LIGHT', 'CAMERA'}:
                    needs_reset = True
                    break
            
            if isinstance(obj, bpy.types.Material):
                needs_reset = True
                break
            
            if isinstance(obj, bpy.types.World):
                needs_reset = True
                break
            
            if isinstance(obj, bpy.types.Light):
                needs_reset = True
                break
        
        if not needs_reset:
            return
        
        # シーンキャッシュを無効化
        get_scene_cache().invalidate()
        
        # シーンが変更されたことを記録（view_draw で使用）
        self._scene_update_pending = True
        
        # 累積サンプルのみリセット（テクスチャは維持してちらつき防止）
        if hasattr(self, 'pybind_accumulated_samples'):
            self.pybind_accumulated_samples = {}

    def _detect_scene_change(self, context, depsgraph):
        """Detect if viewport camera or scene content has changed.
        
        Returns:
            tuple: (changed, content_changed)
                - changed: True if anything changed (camera or content)
                - content_changed: True if scene content changed (needs re-export)
        """
        camera_changed = False
        content_changed = False
        
        # 1. カメラ（ビュー）の変更検出
        region_data = context.region_data
        if region_data is not None:
            current_matrix = region_data.view_matrix.copy()
            current_perspective = region_data.view_perspective
            current_distance = region_data.view_distance
            
            if self.last_view_perspective != current_perspective:
                camera_changed = True
            
            if hasattr(self, 'last_view_distance') and self.last_view_distance is not None:
                if self.last_view_distance > 0:
                    distance_change = abs(current_distance - self.last_view_distance) / self.last_view_distance
                    if distance_change > 0.01:
                        camera_changed = True
            
            if self.last_camera_matrix is not None and not camera_changed:
                max_diff = 0.0
                for i in range(4):
                    for j in range(4):
                        max_diff = max(max_diff, abs(current_matrix[i][j] - self.last_camera_matrix[i][j]))
                if max_diff > 0.001:
                    camera_changed = True
            
            if camera_changed or self.last_camera_matrix is None:
                self.last_camera_matrix = current_matrix
                self.last_view_perspective = current_perspective
                self.last_view_distance = current_distance
        
        # 2. シーンコンテンツの変更検出（オブジェクト、マテリアル、ライトなど）
        if depsgraph is not None:
            # depsgraph.id_type_updated() でタイプ別の更新を検出
            obj_updated = depsgraph.id_type_updated('OBJECT')
            mesh_updated = depsgraph.id_type_updated('MESH')
            mat_updated = depsgraph.id_type_updated('MATERIAL')
            light_updated = depsgraph.id_type_updated('LIGHT')
            world_updated = depsgraph.id_type_updated('WORLD')
            node_updated = depsgraph.id_type_updated('NODETREE')
            
            if obj_updated or mesh_updated or mat_updated or light_updated or world_updated or node_updated:
                content_changed = True
        
        changed = camera_changed or content_changed
        return changed, content_changed

    def view_draw(self, context, depsgraph):
        """Viewport rendering function - called continuously while viewport is active."""
        # pybind11 モードが利用可能なら使用
        if PYBIND_AVAILABLE:
            self._init_pybind_viewport()
            self._view_draw_pybind(context, depsgraph)
            return
        
        # pybind11 が利用できない場合は警告を表示
        import gpu
        import blf
        
        # 背景を暗くする
        region = context.region
        width = region.width
        height = region.height
        
        # 警告メッセージを描画
        blf.size(0, 20)
        blf.color(0, 1.0, 0.8, 0.2, 1.0)
        blf.position(0, 20, height - 40, 0)
        blf.draw(0, "DIY Renderer: pybind11 module not available")
        blf.position(0, 20, height - 70, 0)
        blf.draw(0, "Please build the C++ module with: cmake .. -DBUILD_PYBIND=ON && make")
