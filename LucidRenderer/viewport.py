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
    RenderMode, CameraParams, RenderParams, 
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
    ) -> Tuple[int, int]:
        """ビューポートをレンダリング

        view_draw から呼び出されるメインメソッド。

        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
            state: ビューポート状態
            session: レンダリングセッション（または後方互換の RendererBackend）

        Returns:
            (current_samples, target_samples) のタプル
        """
        import gpu
        from gpu_extras.presets import draw_texture_2d

        # セッションIDを取得（セッション分離用）
        session_id = getattr(session, 'session_id', 0)

        # GPU backend + non-debug → use the C++ async accumulator path. The
        # worker thread keeps firing 1-sample dispatches; we just poll the
        # snapshot from view_draw and draw it. Decouples GPU progress from
        # Blender's redraw cadence so the viewport feels continuous.
        try:
            lucid = context.scene.lucid_renderer
            use_async = (lucid.backend == 'gpu'
                         and lucid.debug_mode == 'NONE'
                         and getattr(session, 'is_available', False)
                         and hasattr(session, 'start_render_async'))
        except Exception:
            use_async = False
        if use_async:
            return self._render_via_gpu_async(context, depsgraph, state, session, session_id)

        region = context.region
        width = region.width
        height = region.height
        current_time = time.time()
        
        # 1. カメラ変更をチェック（ビューポート操作用）
        camera_changed = self._check_camera_changed(context, state)

        # 2. シーン変更フラグを処理（engine.view_update から）
        content_changed = state.scene_update_pending
        if content_changed:
            state.scene_update_pending = False

        # 2b. バックエンド / デバッグモード切り替えも変更として扱う
        # (Blender は EnumProperty 変更で view_update を呼んでくれないので、
        #  ここでポーリングして変化を検知し、累積をリセットして再 render させる)
        lucid = context.scene.lucid_renderer
        backend_changed = (state.last_backend is not None
                           and state.last_backend != lucid.backend)
        debug_mode_changed = (state.last_debug_mode is not None
                              and state.last_debug_mode != lucid.debug_mode)
        state.last_backend = lucid.backend
        state.last_debug_mode = lucid.debug_mode

        # 3. 何か変更があった場合の処理
        any_change = (camera_changed or content_changed
                      or backend_changed or debug_mode_changed)
        was_in_final_mode = (current_time - state.last_change_time) >= RENDER_CONSTANTS.EDITING_TIMEOUT
        if any_change:
            # 累積サンプルをリセット、時間を更新
            state.accumulated_samples = {}
            state.last_change_time = time.time()
            
            # 最終モード中に変更された場合はキャンセル
            if was_in_final_mode:
                session.cancel()
        
        # 4. モード判定
        mode = self._determine_mode(current_time, state, any_change)
        
        # 5. レンダリングパラメータを計算（累積サンプリング用オフセットを取得）
        # FINALモードでは累積サンプル数をオフセットとして使用
        sample_offset = 0
        if mode == RenderMode.FINAL:
            # 現在の解像度でのサンプル数を取得（仮の解像度で計算）
            lucid = context.scene.lucid_renderer
            scale_factor = lucid.viewport_scale_final
            render_width = max(1, width // scale_factor)
            render_height = max(1, height // scale_factor)
            # accumulated_samples は (array, count) のタプルを保持
            acc_data = state.accumulated_samples.get((render_width, render_height))
            if acc_data is not None:
                sample_offset = acc_data[1]  # count
        
        params, camera = self._compute_params(context, mode, width, height, sample_offset)
        
        # 6. 解像度変更チェック
        if (state.last_render_width != params.width or 
            state.last_render_height != params.height):
            state.reset_for_resolution_change()
        
        # 7. 結果をポーリング（テクスチャ更新があったかを取得）
        texture_updated = self._poll_results(state, session, current_time)
        
        # 8. 非同期エクスポートの完了をチェック
        self._check_export_completion(context, depsgraph, state, session)
        
        # 9. 新しいレンダリングを開始（必要な場合）
        self._maybe_start_render(
            context, depsgraph, state, session,
            params, camera, mode, content_changed, current_time,
            session_id
        )
        
        # 10. 再描画をスケジュール（テクスチャ更新時は必ず再描画）
        self._schedule_redraw(context, state, mode, params, texture_updated)
        
        # 11. テクスチャを描画
        current_samples, target_samples = self._draw_texture(context, state, width, height)
        
        return current_samples, target_samples
    
    # =========================================================================
    # GPU async accumulator path (the "background worker + poll" model)
    # =========================================================================

    def _render_via_gpu_async(
        self,
        context: Any,
        depsgraph: Any,
        state: 'ViewportState',
        session: Union['RenderSession', 'RendererBackend'],
        session_id: int
    ) -> Tuple[int, int]:
        """GPU バックエンド + デバッグ無し時の async accumulator パス

        C++ 側 worker が 1 sample ずつ accumulator に積むのを poll するだけ。
        scene/camera が変わったら stop → 再 export → 再 load → start_async し直す。
        """
        import array as _array
        import gpu as _gpu

        region = context.region
        view_w = region.width
        view_h = region.height
        current_time = time.time()

        # Resolution scale & bounce count come from the props the user sets.
        # We pick scale_factor based on whether we're "actively interacting"
        # (use editing scale → lower-res, snappier) or settled (use final).
        lucid = context.scene.lucid_renderer
        camera_changed = self._check_camera_changed(context, state)
        content_changed = state.scene_update_pending
        if content_changed:
            state.scene_update_pending = False

        backend_changed = (state.last_backend is not None
                           and state.last_backend != lucid.backend)
        debug_mode_changed = (state.last_debug_mode is not None
                              and state.last_debug_mode != lucid.debug_mode)
        state.last_backend = lucid.backend
        state.last_debug_mode = lucid.debug_mode

        any_change = (camera_changed or content_changed
                      or backend_changed or debug_mode_changed)
        if any_change:
            state.last_change_time = current_time
            state.async_target_reached = False

        time_since_change = current_time - state.last_change_time
        if any_change or time_since_change < RENDER_CONSTANTS.EDITING_TIMEOUT:
            scale_factor = lucid.viewport_scale_editing
            max_bounces  = RENDER_CONSTANTS.EDITING_BOUNCES
        else:
            scale_factor = lucid.viewport_scale_final
            max_bounces  = RENDER_CONSTANTS.FINAL_BOUNCES

        render_w = max(1, view_w // scale_factor)
        render_h = max(1, view_h // scale_factor)

        # Hard cap on async render dimensions. Without this, a 4K external
        # monitor at Final scale=1 produces ~9 MP accumulator buffers (~150 MB),
        # which blow past Dawn's MapAsync timeout and Apple Metal's per-binding
        # storage limit. 2048 max-edge gives ~2.4 MP / ~38 MB for 16:9 4K and
        # is visually fine when stretched back to the viewport. 1080p/1440p
        # are below the cap and unaffected.
        ASYNC_RENDER_MAX_DIM = 2048
        max_dim = max(render_w, render_h)
        if max_dim > ASYNC_RENDER_MAX_DIM:
            shrink = float(max_dim) / float(ASYNC_RENDER_MAX_DIM)
            render_w = max(1, int(render_w / shrink))
            render_h = max(1, int(render_h / shrink))

        resolution_changed = (state.last_async_w != render_w
                              or state.last_async_h != render_h)
        # Don't auto-restart after we've already hit the user's sample target —
        # the worker was stopped on purpose; restarting would burn power
        # forever on already-converged pixels.
        worker_idle_unexpectedly = (not session.is_render_async_running()
                                    and not state.async_target_reached)
        needs_restart = (any_change or resolution_changed
                         or worker_idle_unexpectedly)

        # Heavy restart conditions: anything that requires reuploading the
        # scene buffers or reallocating the accumulator. Bare camera moves
        # do NOT need a full stop+start — that path joins the worker thread
        # (up to 30 ms) and reads the scene from disk (another ~30 ms),
        # which is what was freezing Blender's UI during pan/zoom.
        needs_full_restart = (content_changed or resolution_changed
                              or backend_changed or debug_mode_changed
                              or worker_idle_unexpectedly)

        if needs_restart and needs_full_restart:
            # Heavyweight path — used for first render, scene change, viewport
            # resize, backend/debug switch, or after the target-sample stop.
            try:
                session.stop_render_async()
            except Exception as e:
                print(f"[ViewportRenderer] stop_render_async failed: {e}")

            try:
                scene_file = get_scene_cache(session_id).get_cached_file_fast()
                if scene_file is None or content_changed:
                    scene_file = export_scene_to_file(depsgraph, session_id=session_id)
                # Force reload whenever the scene actually changed — the export
                # file path is reused across edits (it's session-specific), so
                # path-equality alone misses content changes. RenderSession's
                # load_scene_if_changed hashes the JSON and no-ops if truly
                # unchanged, so the only extra cost on a path-stable edit is
                # one disk read + parse.
                needs_reload = (scene_file
                                and (scene_file != state.last_async_scene_hash
                                     or content_changed))
                if needs_reload:
                    session.load_scene_file(scene_file)
                    state.last_async_scene_hash = scene_file
            except Exception as e:
                print(f"[ViewportRenderer] async scene export failed: {e}")

            camera = self._compute_camera_only(context)
            if camera is not None:
                try:
                    session.set_camera(camera)
                except Exception as e:
                    print(f"[ViewportRenderer] set_camera failed: {e}")

            if resolution_changed:
                # Keep state.texture so draw_texture_2d stretches the previous
                # frame for one tick instead of flashing the gray placeholder.
                state.accumulated_samples = {}

            try:
                session.start_render_async(render_w, render_h, max_bounces)
            except Exception as e:
                print(f"[ViewportRenderer] start_render_async failed: {e}")

            # The C++ AsyncState is freshly constructed; its snapshot_revision
            # starts at 0. Reset our cached "last polled" so the first new
            # snapshot of the session is picked up rather than compared against
            # leftover state from the previous session.
            state.last_async_snapshot_rev = 0
            state.last_async_w = render_w
            state.last_async_h = render_h
        elif needs_restart:
            # Camera-only hot path: signal the running worker to wipe the
            # accumulator and pick up the new camera. No thread join, no
            # buffer realloc, no scene reload — ~1 ms total on the main thread.
            camera = self._compute_camera_only(context)
            if camera is not None:
                try:
                    session.set_camera(camera)
                except Exception as e:
                    print(f"[ViewportRenderer] set_camera failed: {e}")
            try:
                session.reset_render_async(render_w, render_h, max_bounces)
            except Exception as e:
                print(f"[ViewportRenderer] reset_render_async failed: {e}")

        # Skip the heavy poll path when the worker hasn't produced a new
        # snapshot since last frame. The check itself is a single atomic read
        # on the C++ side (~µs); poll_render_async is ~30-50ms when there's
        # actual data to convert, so this is a big win at high view_draw rates.
        cur_rev = session.snapshot_revision_async()
        samples = 0
        pixels = None
        if cur_rev > state.last_async_snapshot_rev:
            try:
                samples, pixels = session.poll_render_async()
                state.last_async_snapshot_rev = cur_rev
            except Exception as e:
                print(f"[ViewportRenderer] poll_render_async failed: {e}")
                samples, pixels = 0, None

        expected_len = render_w * render_h * 4
        if (samples > 0 and pixels is not None
                and len(pixels) == expected_len):
            # Worker delivers a numpy float32 array of the normalised RGBA
            # snapshot. Hand it to gpu.types.Buffer via the buffer protocol —
            # no list() detour, no per-element conversion.
            tile_key = (render_w, render_h)
            # _draw_texture / get_current_sample_count only read index [1]
            # (the count). Storing None for the array dodges another
            # multi-megabyte Python allocation.
            state.accumulated_samples[tile_key] = (None, int(samples))

            buffer = _gpu.types.Buffer('FLOAT', expected_len, pixels)
            if state.texture is not None:
                try:
                    del state.texture
                except Exception:
                    pass
            state.texture = _gpu.types.GPUTexture(
                (render_w, render_h), format='RGBA16F', data=buffer
            )
            state.texture_width  = render_w
            state.texture_height = render_h

        # User-configured cap (defaults to 64). Once the worker has caught up
        # to the target we stop it so the GPU goes idle — saves power and
        # avoids burning compute on already-converged pixels.
        try:
            target_samples = lucid.viewport_samples
        except Exception:
            target_samples = 64

        if samples >= target_samples and session.is_render_async_running():
            try:
                session.stop_render_async()
                state.async_target_reached = True
            except Exception as e:
                print(f"[ViewportRenderer] stop at target failed: {e}")

        if samples < target_samples and session.is_render_async_running():
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        # Draw the texture to the viewport. _draw_texture also reads the sample
        # count back out of accumulated_samples for the header readout.
        current_samples, _ = self._draw_texture(context, state, view_w, view_h)
        return current_samples, target_samples

    def _compute_camera_only(self, context: Any) -> Optional[CameraParams]:
        """View-matrix → CameraParams, no other render params. Mirrors the
        camera block of `_compute_params` so the async path produces an
        identical basis."""
        from mathutils import Vector
        region_data = context.region_data
        if region_data is None:
            return None
        view_matrix_inv = region_data.view_matrix.inverted()
        cam_pos = view_matrix_inv.translation
        cam_dir = (view_matrix_inv.to_3x3() @ Vector((0, 0, -1))).normalized()
        cam_up  = (view_matrix_inv.to_3x3() @ Vector((0, 1, 0))).normalized()
        fov = self._compute_fov(context, region_data)
        return CameraParams(
            pos=(cam_pos.x, cam_pos.y, cam_pos.z),
            dir=(cam_dir.x, cam_dir.y, cam_dir.z),
            up=(cam_up.x, cam_up.y, cam_up.z),
            fov=fov,
        )

    # =========================================================================
    # カメラ変更検出
    # =========================================================================

    def _check_camera_changed(
        self,
        context: Any,
        state: 'ViewportState'
    ) -> bool:
        """カメラ（ビュー）の変更を検出
        
        Blender の view_update はビューポートのカメラ操作（回転、パン、ズーム）では
        呼び出されないため、view_draw 内で毎フレーム検出する必要があります。
        
        シーンコンテンツの変更検出は engine.view_update() で行い、
        state.scene_update_pending フラグで通知されます。
        
        Returns:
            カメラが変更された場合 True
        """
        camera_changed = False
        
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
        
        return camera_changed
    
    # =========================================================================
    # モード判定
    # =========================================================================
    
    def _determine_mode(
        self,
        current_time: float,
        state: 'ViewportState',
        any_change: bool
    ) -> RenderMode:
        """レンダリングモードを判定
        
        Args:
            current_time: 現在時刻
            state: ビューポート状態
            any_change: 何か変更があったか（カメラまたはシーン）
        
        Returns:
            編集モードまたは最終モード
        """
        time_since_change = current_time - state.last_change_time
        
        if any_change or time_since_change < RENDER_CONSTANTS.EDITING_TIMEOUT:
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
        height: int,
        sample_offset: int = 0
    ) -> Tuple[RenderParams, Optional[CameraParams]]:
        """レンダリングパラメータを計算
        
        Args:
            sample_offset: 累積サンプリング用のオフセット（FINALモード用）
        
        Returns:
            (RenderParams, CameraParams) のタプル
        """
        from mathutils import Vector
        
        lucid = context.scene.lucid_renderer
        
        # 解像度スケール
        if mode == RenderMode.EDITING:
            scale_factor = lucid.viewport_scale_editing
            max_bounces = RENDER_CONSTANTS.EDITING_BOUNCES
        else:
            scale_factor = lucid.viewport_scale_final
            max_bounces = RENDER_CONSTANTS.FINAL_BOUNCES
        
        render_width = max(1, width // scale_factor)
        render_height = max(1, height // scale_factor)
        
        # デバッグモード
        debug_mode = lucid.debug_mode if lucid.debug_mode != 'NONE' else None
        
        # GPU backend pays a fixed dispatch round-trip cost (~5-10 ms) regardless
        # of samples per call; CPU runs in-process and has no such cost. In
        # FINAL mode (camera settled) we bump samples/dispatch to amortise that
        # overhead — accumulation converges noticeably faster. In EDITING mode
        # (camera moving) we stay at 1 so each dispatch returns ASAP and the
        # viewport tracks input snappily.
        if lucid.backend == 'gpu' and mode == RenderMode.FINAL:
            samples_per_dispatch = 4
        else:
            samples_per_dispatch = 1

        params = RenderParams(
            width=render_width,
            height=render_height,
            samples=samples_per_dispatch,
            max_bounces=max_bounces,
            algorithm=lucid.sampling_algorithm,
            debug_mode=debug_mode,
            sample_offset=sample_offset,
            backend=lucid.backend,
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
    ) -> bool:
        """レンダリング結果をポーリング

        Returns:
            テクスチャが更新された場合 True
        """
        import gpu

        if state.render_future is None:
            return False

        if not state.render_future.done():
            return False

        try:
            result: RenderResult = state.render_future.result()
            state.render_future = None
            state.last_render_complete_time = current_time

            if result.cancelled or not result.pixels:
                return False

            expected_len = result.width * result.height * 4
            if len(result.pixels) != expected_len:
                return False

            # サンプル累積
            self._accumulate_samples(state, result)

            # テクスチャ更新
            self._update_texture(state, result)

            return True

        except Exception as e:
            print(f"[ViewportRenderer] Render error: {e}")
            import traceback
            traceback.print_exc()
            state.render_future = None
            return False
    
    def _accumulate_samples(
        self,
        state: 'ViewportState',
        result: RenderResult
    ) -> None:
        """サンプルを累積（重み付き平均）

        Both CPU (`render_tile`) and GPU (`render_tile_gpu`) follow the SUM
        convention: `result.pixels` is the un-normalized sum of radiance over
        `result.samples` paths (with camera sensitivity already applied). The
        accumulator stores a running mean and combines via:
            new_avg = (old_avg * prev_count + new_sum) / (prev_count + N)
        For the first dispatch, the running mean is `new_sum / N` (i.e. divide
        the sum by the number of samples it contained).
        """
        tile_key = (result.width, result.height)

        if tile_key in state.accumulated_samples:
            acc_array, prev_count = state.accumulated_samples[tile_key]
            new_count = prev_count + result.samples

            for i in range(len(acc_array)):
                old_sum = acc_array[i] * prev_count
                new_sum_in_dispatch = result.pixels[i]   # already a sum
                acc_array[i] = (old_sum + new_sum_in_dispatch) / new_count
        else:
            # 最初のディスパッチ: SUM をサンプル数で割って AVG にする
            inv_n = 1.0 / float(result.samples) if result.samples > 0 else 1.0
            acc_array = array.array('f', (p * inv_n for p in result.pixels))
            new_count = result.samples

        state.accumulated_samples[tile_key] = (acc_array, new_count)
    
    def _update_texture(
        self,
        state: 'ViewportState',
        result: RenderResult
    ) -> None:
        """テクスチャを更新（累積バッファから）"""
        import gpu
        
        tile_key = (result.width, result.height)
        acc_data = state.accumulated_samples.get(tile_key)
        
        if acc_data is None:
            return
        
        acc_array, _ = acc_data
        display_pixels = list(acc_array)
        
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
                    sample_offset=data.get('sample_offset', 0),
                    max_bounces=data['max_bounces'],
                    algorithm=data['algorithm'],
                    debug_mode=data['debug_mode'],
                    backend=data.get('backend', 'cpu'),
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
                    'sample_offset': params.sample_offset,
                    'max_bounces': params.max_bounces,
                    'algorithm': params.algorithm,
                    'debug_mode': params.debug_mode,
                    'backend': params.backend,
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
            # 最終プレビュー: キャッシュを使用（シーンが変わっていなければ再エクスポート不要）
            scene_file = get_scene_cache(session_id).get_cached_file_fast()
            if not scene_file:
                scene_file = export_scene_to_file(depsgraph, session_id=session_id)
                state.last_scene_export_time = current_time
            
            if scene_file:
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
                lucid = get_scene_settings()
                target_samples = lucid.viewport_samples if lucid else 64
            except Exception:
                target_samples = 64
            
            current_samples = state.get_current_sample_count(params.width, params.height)
            return current_samples < target_samples
    
    # =========================================================================
    # 再描画スケジュール
    # =========================================================================
    
    def _schedule_redraw(
        self,
        context: Any,
        state: 'ViewportState',
        mode: RenderMode,
        params: RenderParams,
        texture_updated: bool = False
    ) -> None:
        """再描画をスケジュール"""
        current_samples = state.get_current_sample_count(params.width, params.height)
        
        try:
            lucid = context.scene.lucid_renderer
            target_samples = lucid.viewport_samples
        except Exception:
            target_samples = 64
        
        should_redraw = (
            texture_updated or  # テクスチャが更新された場合は必ず再描画
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
    ) -> Tuple[int, int]:
        """テクスチャを描画
        
        Returns:
            (current_samples, target_samples) のタプル
        """
        import gpu
        from gpu_extras.presets import draw_texture_2d
        
        # OpenGL ステートを設定
        gpu.state.blend_set('ALPHA_PREMULT')
        
        current_samples = 0
        target_samples = 64
        
        if state.texture is not None:
            # 現在の累積サンプル数を確認
            tile_key = (state.texture_width, state.texture_height)
            acc_data = state.accumulated_samples.get(tile_key)
            current_samples = acc_data[1] if acc_data else 0
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
        
        # ブレンドをリセット
        gpu.state.blend_set('NONE')
        
        # ターゲットサンプル数を取得
        try:
            lucid = context.scene.lucid_renderer
            target_samples = lucid.viewport_samples
        except Exception:
            pass
        
        return current_samples, target_samples
