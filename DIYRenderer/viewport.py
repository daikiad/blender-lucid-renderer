"""
Viewport - ビューポートレンダリング専用モジュール
================================================

このモジュールは Blender ビューポートでのリアルタイム
レンダリングを担当します。

主要クラス:
- ViewportRenderer: ビューポートレンダリングのメインクラス

責務:
- 変更検出（カメラ/シーン）
- モード判定（編集中/最終プレビュー）
- 解像度の決定
- 非同期レンダリングの開始
- 結果のポーリングとテクスチャ更新
- 描画

使用例:
    viewport = ViewportRenderer()
    viewport.render(context, depsgraph, state, session)
"""

from __future__ import annotations

import math
import time
import array
from typing import Optional, Tuple, Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import bpy
    from .state import ViewportState
    from .render_session import RenderSession
    from .backend import RendererBackend

from .state import (
    RenderMode, ChangeType, CameraParams, RenderParams, 
    RenderResult, RENDER_CONSTANTS
)
from .scene_export import export_scene_to_file, get_scene_cache


class ViewportRenderer:
    """ビューポートレンダリングのメインクラス
    
    ビューポートレンダリングのすべてのロジックを管理します。
    engine.py の _view_draw_pybind メソッドを分離・整理したものです。
    """
    
    def __init__(self):
        """初期化"""
        pass
    
    # =========================================================================
    # メインエントリーポイント
    # =========================================================================
    
    def render(
        self,
        context: Any,
        depsgraph: Any,
        state: 'ViewportState',
        session: Union['RenderSession', 'RendererBackend']
    ) -> None:
        """ビューポートをレンダリング
        
        view_draw から呼び出されるメインメソッド。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
            state: ビューポート状態
            session: レンダリングセッション（または後方互換の RendererBackend）
        """
        import gpu
        from gpu_extras.presets import draw_texture_2d
        
        # セッションIDを取得（セッション分離用）
        session_id = getattr(session, 'session_id', 0)
        
        region = context.region
        width = region.width
        height = region.height
        current_time = time.time()
        
        # 1. 変更検出
        change_type = self._detect_changes(context, depsgraph, state)
        
        # 2. シーン変更フラグを処理
        if state.scene_update_pending:
            state.scene_update_pending = False
            if change_type == ChangeType.NONE:
                change_type = ChangeType.CONTENT
        
        # 3. 変更があった場合の処理
        was_in_final_mode = (current_time - state.last_change_time) >= RENDER_CONSTANTS.EDITING_TIMEOUT
        if change_type != ChangeType.NONE:
            state.reset_for_scene_change()
            
            # 最終モード中に変更された場合はキャンセル
            if was_in_final_mode:
                session.cancel()
        
        # 4. モード判定
        mode = self._determine_mode(current_time, state, change_type)
        
        # 5. レンダリングパラメータを計算
        params, camera = self._compute_params(context, mode, width, height)
        
        # 6. 解像度変更チェック
        if (state.last_render_width != params.width or 
            state.last_render_height != params.height):
            state.reset_for_resolution_change()
        
        # 7. 結果をポーリング
        self._poll_results(state, session, current_time)
        
        # 8. 非同期エクスポートの完了をチェック
        self._check_export_completion(context, depsgraph, state, session)
        
        # 9. 新しいレンダリングを開始（必要な場合）
        content_changed = (change_type == ChangeType.CONTENT)
        self._maybe_start_render(
            context, depsgraph, state, session,
            params, camera, mode, content_changed, current_time,
            session_id
        )
        
        # 10. 再描画をスケジュール
        self._schedule_redraw(context, state, mode, params)
        
        # 11. テクスチャを描画
        self._draw_texture(context, state, width, height)
    
    # =========================================================================
    # 変更検出
    # =========================================================================
    
    def _detect_changes(
        self,
        context: Any,
        depsgraph: Any,
        state: 'ViewportState'
    ) -> ChangeType:
        """カメラ（ビュー）の変更を検出
        
        注意: シーンコンテンツの変更検出は engine.view_update() で行い、
        state.scene_update_pending フラグで通知されます。
        このメソッドはカメラ（ビューマトリクス）の変更のみを検出します。
        
        Blender の view_update はカメラ操作（回転、パン、ズーム）では
        呼び出されないため、view_draw 内で毎フレーム検出する必要があります。
        
        Returns:
            変更の種類（CAMERA_ONLY または NONE）
        """
        camera_changed = False
        
        # カメラ（ビュー）の変更検出
        region_data = context.region_data
        if region_data is not None:
            current_matrix = region_data.view_matrix.copy()
            current_perspective = region_data.view_perspective
            current_distance = region_data.view_distance
            
            # パースペクティブ変更
            if state.last_view_perspective != current_perspective:
                camera_changed = True
            
            # 距離変更
            if state.last_view_distance is not None and state.last_view_distance > 0:
                distance_change = abs(current_distance - state.last_view_distance) / state.last_view_distance
                if distance_change > 0.01:
                    camera_changed = True
            
            # マトリクス変更
            if state.last_camera_matrix is not None and not camera_changed:
                max_diff = 0.0
                for i in range(4):
                    for j in range(4):
                        max_diff = max(max_diff, abs(current_matrix[i][j] - state.last_camera_matrix[i][j]))
                if max_diff > 0.001:
                    camera_changed = True
            
            # 状態を更新
            if camera_changed or state.last_camera_matrix is None:
                state.last_camera_matrix = current_matrix
                state.last_view_perspective = current_perspective
                state.last_view_distance = current_distance
        
        # 結果を返す（カメラ変更のみ検出）
        if camera_changed:
            return ChangeType.CAMERA_ONLY
        else:
            return ChangeType.NONE
    
    # =========================================================================
    # モード判定
    # =========================================================================
    
    def _determine_mode(
        self,
        current_time: float,
        state: 'ViewportState',
        change_type: ChangeType
    ) -> RenderMode:
        """レンダリングモードを判定
        
        Returns:
            編集モードまたは最終モード
        """
        time_since_change = current_time - state.last_change_time
        
        if change_type != ChangeType.NONE or time_since_change < RENDER_CONSTANTS.EDITING_TIMEOUT:
            return RenderMode.EDITING
        else:
            return RenderMode.FINAL
    
    # =========================================================================
    # パラメータ計算
    # =========================================================================
    
    def _compute_params(
        self,
        context: Any,
        mode: RenderMode,
        width: int,
        height: int
    ) -> Tuple[RenderParams, Optional[CameraParams]]:
        """レンダリングパラメータを計算
        
        Returns:
            (RenderParams, CameraParams) のタプル
        """
        from mathutils import Vector
        
        diy = context.scene.diy_renderer
        
        # 解像度スケール
        if mode == RenderMode.EDITING:
            scale_factor = diy.viewport_scale_editing
            max_bounces = RENDER_CONSTANTS.EDITING_BOUNCES
        else:
            scale_factor = diy.viewport_scale_final
            max_bounces = RENDER_CONSTANTS.FINAL_BOUNCES
        
        render_width = max(1, width // scale_factor)
        render_height = max(1, height // scale_factor)
        
        # デバッグモード
        debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
        
        params = RenderParams(
            width=render_width,
            height=render_height,
            samples=1,
            max_bounces=max_bounces,
            algorithm=diy.sampling_algorithm,
            debug_mode=debug_mode
        )
        
        # カメラパラメータ
        camera = None
        region_data = context.region_data
        if region_data is not None:
            view_matrix_inv = region_data.view_matrix.inverted()
            
            cam_pos = view_matrix_inv.translation
            cam_dir = (view_matrix_inv.to_3x3() @ Vector((0, 0, -1))).normalized()
            cam_up = (view_matrix_inv.to_3x3() @ Vector((0, 1, 0))).normalized()
            
            # FOV を計算
            fov = self._compute_fov(context, region_data)
            
            camera = CameraParams(
                pos=(cam_pos.x, cam_pos.y, cam_pos.z),
                dir=(cam_dir.x, cam_dir.y, cam_dir.z),
                up=(cam_up.x, cam_up.y, cam_up.z),
                fov=fov
            )
        
        return params, camera
    
    def _compute_fov(self, context: Any, region_data: Any) -> float:
        """FOV を計算
        
        window_matrix から逆算することで、カメラビューのズームも考慮。
        """
        if region_data.view_perspective == 'CAMERA':
            # カメラビュー: window_matrix から実際の表示FOVを取得
            wm = region_data.window_matrix
            if wm[1][1] != 0:
                return math.degrees(2 * math.atan(1.0 / wm[1][1]))
            else:
                # フォールバック: カメラ設定から計算
                camera = context.scene.camera
                if camera and camera.data:
                    cam_data = camera.data
                    sensor_width = cam_data.sensor_width
                    focal_length = cam_data.lens
                    return math.degrees(2 * math.atan(sensor_width / (2 * focal_length)))
                return 50.0
        elif region_data.view_perspective == 'PERSP':
            # 透視投影: window_matrix から FOV を逆算
            wm = region_data.window_matrix
            if wm[1][1] != 0:
                return math.degrees(2 * math.atan(1.0 / wm[1][1]))
            return 50.0
        else:
            # 正射影
            return 5.0
    
    # =========================================================================
    # 結果ポーリング
    # =========================================================================
    
    def _poll_results(
        self,
        state: 'ViewportState',
        session: Union['RenderSession', 'RendererBackend'],
        current_time: float
    ) -> None:
        """レンダリング結果をポーリング"""
        import gpu
        
        if state.render_future is None:
            return
        
        if not state.render_future.done():
            return
        
        try:
            result: RenderResult = state.render_future.result()
            state.render_future = None
            state.last_render_complete_time = current_time
            
            if result.cancelled or not result.pixels:
                return
            
            expected_len = result.width * result.height * 4
            if len(result.pixels) != expected_len:
                return
            
            # サンプル累積
            self._accumulate_samples(state, result)
            
            # テクスチャ更新
            self._update_texture(state, result)
            
        except Exception as e:
            print(f"[ViewportRenderer] Render error: {e}")
            import traceback
            traceback.print_exc()
            state.render_future = None
    
    def _accumulate_samples(
        self,
        state: 'ViewportState',
        result: RenderResult
    ) -> None:
        """サンプルを累積"""
        if state.pending_camera_update:
            return
        
        tile_key = f"{result.width}x{result.height}"
        
        if tile_key in state.accumulated_samples:
            acc_array, prev_count = state.accumulated_samples[tile_key]
            new_count = prev_count + result.samples
            for i in range(len(acc_array)):
                acc_array[i] += result.pixels[i]
            state.accumulated_samples[tile_key] = (acc_array, new_count)
        else:
            acc_array = array.array('f', result.pixels)
            state.accumulated_samples[tile_key] = (acc_array, result.samples)
    
    def _update_texture(
        self,
        state: 'ViewportState',
        result: RenderResult
    ) -> None:
        """テクスチャを更新"""
        import gpu
        
        tile_key = f"{result.width}x{result.height}"
        
        if tile_key in state.accumulated_samples:
            acc_array, count = state.accumulated_samples[tile_key]
            inv_count = 1.0 / count
            display_pixels = [v * inv_count for v in acc_array]
        else:
            inv_samples = 1.0 / result.samples
            display_pixels = [v * inv_samples for v in result.pixels]
        
        expected_len = result.width * result.height * 4
        buffer = gpu.types.Buffer('FLOAT', expected_len, display_pixels)
        
        if state.texture is not None:
            try:
                del state.texture
            except Exception:
                pass
        
        state.texture = gpu.types.GPUTexture(
            (result.width, result.height), format='RGBA16F', data=buffer
        )
        state.texture_width = result.width
        state.texture_height = result.height
    
    # =========================================================================
    # 非同期エクスポート
    # =========================================================================
    
    def _check_export_completion(
        self,
        context: Any,
        depsgraph: Any,
        state: 'ViewportState',
        session: Union['RenderSession', 'RendererBackend']
    ) -> None:
        """非同期エクスポートの完了をチェック"""
        if state.export_future is None:
            return
        
        if not state.export_future.done():
            return
        
        try:
            scene_file = state.export_future.result()
            state.export_future = None
            
            if scene_file and state.pending_export_data:
                data = state.pending_export_data
                state.pending_export_data = None
                
                # 既存のレンダリングをキャンセルして完了を待つ
                # （シーンロード中にレンダリングスレッドがアクセスするとセグフォするため）
                if state.render_future is not None:
                    session.cancel()
                    try:
                        state.render_future.result(timeout=0.1)
                    except Exception:
                        pass
                    state.render_future = None
                    session.reset_cancel()
                
                # レンダリングを開始
                camera = CameraParams(
                    pos=data['cam_pos'],
                    dir=data['cam_dir'],
                    up=data['cam_up'],
                    fov=data['fov']
                )
                
                params = RenderParams(
                    width=data['render_width'],
                    height=data['render_height'],
                    samples=data['samples'],
                    max_bounces=data['max_bounces'],
                    algorithm=data['algorithm'],
                    debug_mode=data['debug_mode']
                )
                
                future = session.render_tile_async(params, scene_file, camera)
                if future:
                    state.render_future = future
                    
        except Exception as e:
            print(f"[ViewportRenderer] Async export error: {e}")
            state.export_future = None
            state.pending_export_data = None
    
    # =========================================================================
    # レンダリング開始
    # =========================================================================
    
    def _maybe_start_render(
        self,
        context: Any,
        depsgraph: Any,
        state: 'ViewportState',
        session: Union['RenderSession', 'RendererBackend'],
        params: RenderParams,
        camera: Optional[CameraParams],
        mode: RenderMode,
        content_changed: bool,
        current_time: float,
        session_id: int = 0
    ) -> None:
        """必要に応じてレンダリングを開始"""
        if camera is None:
            return
        
        # レンダリングが必要か判定
        if not self._should_render(state, params, mode, content_changed, current_time):
            return
        
        # レンダリング中なら開始しない
        if state.is_rendering():
            return
        
        # カメラ更新待ちをクリア
        if state.pending_camera_update:
            state.pending_camera_update = False
            state.accumulated_samples = {}
        
        state.last_render_width = params.width
        state.last_render_height = params.height
        
        if content_changed:
            # 内容が変わったので非同期エクスポート
            if not state.is_exporting():
                state.pending_export_data = {
                    'cam_pos': camera.pos,
                    'cam_dir': camera.dir,
                    'cam_up': camera.up,
                    'fov': camera.fov,
                    'render_width': params.width,
                    'render_height': params.height,
                    'samples': params.samples,
                    'max_bounces': params.max_bounces,
                    'algorithm': params.algorithm,
                    'debug_mode': params.debug_mode,
                    'is_editing': (mode == RenderMode.EDITING),
                }
                
                # セッションの executor を使用
                if hasattr(session, '_executor') and session._executor:
                    # セッションIDを渡すためにラムダを使用
                    sid = session_id
                    state.export_future = session._executor.submit(
                        lambda dg, s=sid: export_scene_to_file(dg, session_id=s), depsgraph
                    )
                state.last_scene_export_time = current_time
        elif mode == RenderMode.EDITING:
            # カメラのみ変更で編集中: キャッシュを使用
            scene_file = get_scene_cache(session_id).get_cached_file_fast()
            if not scene_file:
                scene_file = export_scene_to_file(depsgraph, session_id=session_id)
                state.last_scene_export_time = current_time
            
            if scene_file:
                future = session.render_tile_async(params, scene_file, camera)
                if future:
                    state.render_future = future
        else:
            # 最終プレビュー: 常に再エクスポート
            scene_file = export_scene_to_file(depsgraph, session_id=session_id)
            state.last_scene_export_time = current_time
            
            if scene_file:
                # 前回をキャンセル
                session.cancel()
                future = session.render_tile_async(params, scene_file, camera)
                if future:
                    state.render_future = future
    
    def _should_render(
        self,
        state: 'ViewportState',
        params: RenderParams,
        mode: RenderMode,
        content_changed: bool,
        current_time: float
    ) -> bool:
        """レンダリングが必要か判定"""
        # テクスチャがない場合は必要
        if state.texture is None:
            return not state.is_exporting()
        
        # エクスポート/レンダリング中
        if state.is_exporting() or state.is_rendering():
            return False
        
        # クールダウン
        time_since_render = current_time - state.last_render_complete_time
        if time_since_render < RENDER_CONSTANTS.RENDER_COOLDOWN:
            return False
        
        if mode == RenderMode.EDITING:
            # スロットリング
            if content_changed:
                time_since_export = current_time - state.last_scene_export_time
                if time_since_export < RENDER_CONSTANTS.EXPORT_THROTTLE:
                    return False
            return True
        else:
            # 最終モード: サンプル数が足りない場合
            from .preferences import get_scene_settings
            try:
                diy = get_scene_settings()
                target_samples = diy.viewport_samples if diy else 64
            except Exception:
                target_samples = 64
            
            current_samples = state.get_current_sample_count(params.width, params.height)
            return (
                state.pending_camera_update or
                current_samples < target_samples
            )
    
    # =========================================================================
    # 再描画スケジュール
    # =========================================================================
    
    def _schedule_redraw(
        self,
        context: Any,
        state: 'ViewportState',
        mode: RenderMode,
        params: RenderParams
    ) -> None:
        """再描画をスケジュール"""
        current_samples = state.get_current_sample_count(params.width, params.height)
        
        try:
            diy = context.scene.diy_renderer
            target_samples = diy.viewport_samples
        except Exception:
            target_samples = 64
        
        should_redraw = (
            state.is_rendering() or
            state.is_exporting() or
            mode == RenderMode.EDITING or
            current_samples < target_samples
        )
        
        if should_redraw:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    
    # =========================================================================
    # 描画
    # =========================================================================
    
    def _draw_texture(
        self,
        context: Any,
        state: 'ViewportState',
        width: int,
        height: int
    ) -> None:
        """テクスチャを描画"""
        import gpu
        from gpu_extras.presets import draw_texture_2d
        
        if state.texture is not None:
            draw_texture_2d(state.texture, (0, 0), width, height)
        else:
            # プレースホルダーを描画
            if state.placeholder_texture is None:
                placeholder_size = 64
                placeholder_pixels = [0.2, 0.2, 0.2, 1.0] * (placeholder_size * placeholder_size)
                placeholder_buffer = gpu.types.Buffer('FLOAT', len(placeholder_pixels), placeholder_pixels)
                state.placeholder_texture = gpu.types.GPUTexture(
                    (placeholder_size, placeholder_size), format='RGBA16F', data=placeholder_buffer
                )
            draw_texture_2d(state.placeholder_texture, (0, 0), width, height)
