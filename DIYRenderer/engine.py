"""
DIY Render Engine - Main render engine class for Blender integration.

このファイルは Blender レンダーエンジンの中核です。
F12 レンダリングとビューポートレンダリングの両方を処理します。

処理フロー:
1. Blender が render() または view_update()/view_draw() を呼び出す
2. シーンをエクスポート (scene_export.py)
3. サーバーモードまたはレガシーモードでレンダリング
4. 結果を Blender に返す

主要クラス:
- DIYRenderEngine: bpy.types.RenderEngine のサブクラス

グローバル変数:
- _server_renderer: 持続する SubprocessRenderer インスタンス
- _server_scene_hash: シーン変更検出用の MD5 ハッシュ
"""

import os
import math
import time
import queue
import threading
import array
from typing import Optional

import bpy
from mathutils import Vector

from .scene_export import export_scene_to_file, get_scene_cache
from .renderer import call_external_renderer, compute_camera_params
from .subprocess_renderer import SubprocessRenderer
from .renderer_interface import (
    RenderConfig, CameraParams as RICameraParams, TileParams,
    BackendType, AlgorithmType
)


# =============================================================================
# グローバル状態 (サーバーモード用)
# =============================================================================
# サーバーモードでは、レンダラープロセスを持続させてオーバーヘッドを削減します。
# これらのグローバル変数でプロセスの状態を管理します。

_server_renderer: Optional[SubprocessRenderer] = None  # 持続するレンダラーインスタンス
_server_scene_hash: Optional[str] = None               # シーン変更検出用ハッシュ


def get_server_renderer() -> Optional[SubprocessRenderer]:
    """Get the global server renderer instance."""
    global _server_renderer
    return _server_renderer


def start_server_renderer(config: RenderConfig) -> bool:
    """
    Start or restart the server renderer with given config.
    
    サーバーモードのレンダラープロセスを起動します。
    すでに起動済みで動作中なら何もしません。
    
    Args:
        config: レンダリング設定 (backend, algorithm, max_depth)
    
    Returns:
        起動成功なら True
    """
    global _server_renderer, _server_scene_hash
    
    # 既存のインスタンスをチェック
    if _server_renderer is not None:
        if _server_renderer.is_running():
            return True  # すでに起動済み
        _server_renderer.stop()  # 停止済みなら再起動
    
    # 新しいインスタンスを作成して起動
    _server_renderer = SubprocessRenderer()
    _server_scene_hash = None  # シーンキャッシュをクリア
    
    success = _server_renderer.start(config)
    if success:
        print("[DIYRenderer] Server mode started")
    else:
        print("[DIYRenderer] Failed to start server mode")
        _server_renderer = None
    return success


def stop_server_renderer():
    """Stop the server renderer if running."""
    global _server_renderer, _server_scene_hash
    if _server_renderer is not None:
        _server_renderer.stop()
        _server_renderer = None
        _server_scene_hash = None
        print("[DIYRenderer] Server mode stopped")


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

    def _init_async_render(self):
        """
        ビューポートレンダリング用の非同期インフラを遅延初期化。
        
        ビューポートではバックグラウンドスレッドでレンダリングを行い、
        UI をブロックしないようにします。
        """
        if not hasattr(self, 'render_queue'):
            self.render_queue = queue.Queue(maxsize=1)    # レンダリングジョブキュー
            self.result_queue = queue.Queue()              # 結果キュー
            self.render_thread = None                      # レンダリングスレッド
            self.stop_thread = False                       # スレッド停止フラグ
            self.rendering_in_progress = False             # レンダリング中フラグ
            self.high_res_complete = False                 # 高解像度完了フラグ
            self.accumulated_samples = {}                  # サンプル累積データ
            self.last_camera_matrix = None                 # カメラ変更検出用
            self.last_view_perspective = None              # ビュー変更検出用
            self.viewport_thread_running = False           # スレッド実行中フラグ
            self.current_render_cancelled = False          # キャンセルフラグ

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

    # =========================================================================
    # サーバーモード関連メソッド
    # =========================================================================

    def _use_server_mode(self, scene) -> bool:
        """
        サーバーモードを使用すべきか判定。
        
        ユーザー設定の use_server_mode プロパティを参照します。
        """
        return scene.diy_renderer.use_server_mode

    def _ensure_server_started(self, scene) -> bool:
        """
        サーバーレンダラーが起動済みか確認し、必要なら起動。
        
        ユーザー設定から RenderConfig を構築して start_server_renderer() を呼びます。
        
        Args:
            scene: Blender シーン (設定読み取り用)
        
        Returns:
            サーバーが利用可能なら True
        """
        global _server_renderer, _server_scene_hash
        
        diy = scene.diy_renderer
        
        # アルゴリズム設定をマッピング
        algo_map = {
            'simple': AlgorithmType.NAIVE,  # 単純なパストレーシング
            'nee': AlgorithmType.NEE,       # Next Event Estimation (直接光サンプリング)
            'mis': AlgorithmType.MIS        # Multiple Importance Sampling
        }
        algorithm = algo_map.get(diy.sampling_algorithm, AlgorithmType.NEE)
        
        config = RenderConfig(
            backend=BackendType.CPU,
            algorithm=algorithm,
            max_depth=diy.max_bounces
        )
        
        return start_server_renderer(config)

    def _update_server_scene(self, scene_file: str) -> bool:
        """
        サーバーにシーンデータを送信（必要な場合のみ）。
        
        MD5 ハッシュでシーンの変更を検出し、変更がなければスキップします。
        これにより、カメラ移動のみの場合などにシーン再送信を避けられます。
        
        Args:
            scene_file: JSON シーンファイルのパス
        
        Returns:
            成功なら True
        """
        global _server_renderer, _server_scene_hash
        
        if _server_renderer is None:
            return False
        
        # JSON ファイルを読み込み
        try:
            with open(scene_file, 'r') as f:
                scene_json = f.read()
            print(f"[DIYRenderer] Scene JSON size: {len(scene_json)} bytes")
        except Exception as e:
            print(f"[DIYRenderer] Failed to read scene file: {e}")
            return False
        
        # ハッシュで変更検出
        import hashlib
        scene_hash = hashlib.md5(scene_json.encode()).hexdigest()
        
        if scene_hash == _server_scene_hash:
            print("[DIYRenderer] Scene unchanged, skipping update")
            return True  # 変更なし
        
        # サーバーに送信
        print("[DIYRenderer] Sending scene to server...")
        if _server_renderer.update_scene(scene_json):
            _server_scene_hash = scene_hash  # ハッシュを保存
            print("[DIYRenderer] Scene update successful")
            return True
        print("[DIYRenderer] Scene update failed")
        return False

    def _render_with_server(self, depsgraph, width, height, cam_params, 
                            scene_file, target_samples, diy):
        """
        サーバーモードでレンダリングを実行。
        
        プログレッシブレンダリングを実装しています。
        サンプル数を [1, 2, 4, 8, 16, 32, 64, ...] と増やしながら、
        各反復ごとに画像を更新してユーザーに表示します。
        
        Args:
            depsgraph: Blender の依存関係グラフ
            width, height: 出力解像度
            cam_params: カメラパラメータ dict
            scene_file: JSON シーンファイルパス
            target_samples: 目標サンプル数
            diy: DIY Renderer 設定オブジェクト
        
        Returns:
            成功時は累積ピクセルデータ、失敗時は None
        """
        global _server_renderer
        
        # ---------------------------------------------------------------------
        # Step 1: シーンデータ更新
        # ---------------------------------------------------------------------
        if not self._update_server_scene(scene_file):
            print("[DIYRenderer] Failed to update scene on server")
            return None
        
        # ---------------------------------------------------------------------
        # Step 2: カメラパラメータ送信
        # ---------------------------------------------------------------------
        camera = RICameraParams(
            pos=(cam_params['pos'].x, cam_params['pos'].y, cam_params['pos'].z),
            dir=(cam_params['dir'].x, cam_params['dir'].y, cam_params['dir'].z),
            up=(cam_params['up'].x, cam_params['up'].y, cam_params['up'].z),
            fov=cam_params['fov']
        )
        
        if not _server_renderer.update_camera(camera):
            print("[DIYRenderer] Failed to update camera on server")
            return None
        
        # ---------------------------------------------------------------------
        # Step 3: プログレッシブレンダリングのサンプル分割を計算
        # ---------------------------------------------------------------------
        # 例: target_samples=128 → [1, 2, 4, 8, 16, 32, 64, 1]
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
        
        print(f"[DIYRenderer] Server mode - sample iterations: {sample_iterations}")
        
        # ---------------------------------------------------------------------
        # Step 4: プログレッシブレンダリングループの初期化
        # ---------------------------------------------------------------------
        accumulated_pixels = None      # 累積ピクセルデータ
        total_samples = 0              # 累積サンプル数
        max_samples = sum(sample_iterations)  # 合計サンプル数
        render_start_time = time.time()
        
        def check_cancel():
            """ユーザーによるキャンセルをチェック"""
            return self.test_break() or self._render_cancelled
        
        def format_time(seconds):
            """秒数を読みやすい形式にフォーマット"""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
            else:
                return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
        
        # ---------------------------------------------------------------------
        # Step 5: プログレッシブレンダリングループ
        # ---------------------------------------------------------------------
        for idx, iteration_samples in enumerate(sample_iterations):
            # キャンセルチェック
            if check_cancel():
                _server_renderer.cancel()
                print("[DIYRenderer] Render cancelled by user")
                break
            
            # 進捗と残り時間を計算して表示
            elapsed = time.time() - render_start_time
            if total_samples > 0:
                time_per_sample = elapsed / total_samples
                remaining_time = time_per_sample * (max_samples - total_samples)
                time_str = f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining_time)}"
            else:
                time_str = f"Elapsed: {format_time(elapsed)}"
            
            self.update_progress(total_samples / max_samples)
            self.update_stats("", f"Path Tracing (Server): {total_samples}/{max_samples} samples | {time_str}")
            
            # -----------------------------------------------------------------
            # Step 5a: C++ レンダラーにタイルレンダリング要求
            # -----------------------------------------------------------------
            tile = TileParams(
                tile_x=0, tile_y=0,             # タイル位置 (全体を1タイルとして扱う)
                tile_w=width, tile_h=height,   # タイルサイズ
                full_w=width, full_h=height,   # 画像全体サイズ
                samples=iteration_samples,      # この反復でのサンプル数
                sample_offset=total_samples     # RNG シード用の累積オフセット
            )
            
            result = _server_renderer.render_tile(tile)
            
            if result is None:
                if check_cancel():
                    print("[DIYRenderer] Render cancelled during iteration")
                    break
                print("[DIYRenderer] Server render failed")
                continue
            
            iteration_pixels = result.pixels
            expected_len = width * height * 4
            
            if len(iteration_pixels) != expected_len:
                print(f"[DIYRenderer] Invalid pixel count: {len(iteration_pixels)} vs {expected_len}")
                continue
            
            # -----------------------------------------------------------------
            # Step 5b: ピクセルデータを累積
            # -----------------------------------------------------------------
            # C++ からは "累積値" が返ってくる (平均ではない)
            # 各反復の累積値を足し合わせて、表示時に平均化する
            if accumulated_pixels is None:
                accumulated_pixels = array.array('f', iteration_pixels)
                total_samples = iteration_samples
            else:
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i] += iteration_pixels[i]
                total_samples += iteration_samples
            
            # -----------------------------------------------------------------
            # Step 5c: 平均化して Blender に表示
            # -----------------------------------------------------------------
            inv_samples = 1.0 / total_samples
            display_pixels = []
            for i in range(0, len(accumulated_pixels), 4):
                display_pixels.append([
                    accumulated_pixels[i] * inv_samples,      # R
                    accumulated_pixels[i+1] * inv_samples,    # G
                    accumulated_pixels[i+2] * inv_samples,    # B
                    1.0                                        # A
                ])
            
            # Blender のレンダー結果に書き込み
            result_obj = self.begin_result(0, 0, width, height)
            combined = result_obj.layers[0].passes["Combined"]
            combined.rect = display_pixels
            self.end_result(result_obj)
        
        total_elapsed = time.time() - render_start_time
        print(f"[DIYRenderer] Server render complete ({total_samples} samples) in {format_time(total_elapsed)}")
        return accumulated_pixels

    # =========================================================================
    # メインレンダリングメソッド
    # =========================================================================

    def render(self, depsgraph):
        """
        F12 レンダリングのメインエントリーポイント。
        
        Blender がレンダリングを開始するとこのメソッドが呼ばれます。
        サーバーモードとレガシーモードの両方をサポートし、
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
        use_server = self._use_server_mode(original_scene)
        
        print(f"[DIYRenderer] Starting render ({width} x {height}, samples: {target_samples}, server_mode: {use_server})")
        
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
        
        # Server mode rendering
        if use_server:
            if not self._ensure_server_started(original_scene):
                print("[DIYRenderer] Failed to start server, falling back to legacy mode")
            else:
                result = self._render_with_server(
                    depsgraph, width, height, cam_params, 
                    scene_file, target_samples, diy
                )
                if result is not None:
                    # Cleanup scene file
                    try:
                        if scene_file and os.path.isfile(scene_file):
                            os.remove(scene_file)
                    except Exception:
                        pass
                    return
                print("[DIYRenderer] Server render failed, falling back to legacy mode")
        
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

    def async_render_viewport(self, initial_job_data):
        """Background thread function to render viewport without blocking UI."""
        try:
            self.viewport_thread_running = True
        except ReferenceError:
            return
            
        job_data = initial_job_data
        accumulated_local = 0
        
        while True:
            try:
                if self.stop_thread:
                    break
                    
                if self.current_render_cancelled:
                    self.current_render_cancelled = False
                    try:
                        job_data = self.render_queue.get_nowait()
                        accumulated_local = 0
                        continue
                    except queue.Empty:
                        break
                
                scene_file, cam_params, render_width, render_height, job_id, samples_per_iteration, tile_key, target_samples, viewport_bounces, max_bounces, debug_mode, algorithm, is_moving = job_data
                
                if not scene_file:
                    break
                
                render_tile_key = (0, 0, render_width, render_height)
                current_sample_offset = 0
                try:
                    if render_tile_key in self.accumulated_samples:
                        current_sample_offset = self.accumulated_samples[render_tile_key][1]
                except ReferenceError:
                    break
                
                def should_cancel():
                    try:
                        return self.current_render_cancelled or self.stop_thread
                    except ReferenceError:
                        return True
                
                ext_pixels = call_external_renderer(
                    scene_file, 0, 0, render_width, render_height,
                    render_width, render_height, cam_params,
                    samples=samples_per_iteration,
                    depth=max_bounces,
                    debug_mode=debug_mode,
                    cancel_check=should_cancel,
                    sample_offset=current_sample_offset,
                    algorithm=algorithm,
                    pass_id=-1,
                    num_passes=1
                )
                
                if ext_pixels and not should_cancel():
                    try:
                        self.result_queue.put_nowait({
                            'pixels': ext_pixels,
                            'width': render_width,
                            'height': render_height,
                            'job_id': job_id,
                            'samples_per_iteration': samples_per_iteration,
                            'tile_key': tile_key,
                            'is_partial': False
                        })
                        if not is_moving:
                            accumulated_local += samples_per_iteration
                    except (queue.Full, ReferenceError):
                        pass
                
                try:
                    if self.current_render_cancelled:
                        self.current_render_cancelled = False
                        try:
                            job_data = self.render_queue.get_nowait()
                            accumulated_local = 0
                            continue
                        except queue.Empty:
                            break
                except ReferenceError:
                    break
                
                if is_moving:
                    try:
                        job_data = self.render_queue.get(timeout=0.2)
                        accumulated_local = 0
                        continue
                    except queue.Empty:
                        try:
                            job_data = self.render_queue.get(timeout=0.5)
                            accumulated_local = 0
                            continue
                        except queue.Empty:
                            break
                    except ReferenceError:
                        break
                
                try:
                    should_continue = accumulated_local < target_samples and not self.current_render_cancelled
                except ReferenceError:
                    break
                    
                if should_continue:
                    try:
                        new_job = self.render_queue.get_nowait()
                        job_data = new_job
                        accumulated_local = 0
                    except queue.Empty:
                        time.sleep(0.01)
                    except ReferenceError:
                        break
                    continue
                else:
                    try:
                        job_data = self.render_queue.get(timeout=0.5)
                        accumulated_local = 0
                    except queue.Empty:
                        break
                    except ReferenceError:
                        break
                    
            except ReferenceError:
                break
            except Exception as e:
                print(f"[DIYRenderer] Async render error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        try:
            self.viewport_thread_running = False
            self.rendering_in_progress = False
        except ReferenceError:
            pass

    def view_update(self, context, depsgraph):
        """Called when the scene is modified in viewport mode."""
        self._init_async_render()
        
        needs_reset = False
        
        for update in depsgraph.updates:
            obj = update.id
            
            if update.is_updated_geometry:
                needs_reset = True
                print(f"[DIYRenderer] Geometry changed: {obj.name if hasattr(obj, 'name') else type(obj)}")
                break
            
            if update.is_updated_transform:
                if hasattr(obj, 'type') and obj.type in {'MESH', 'LIGHT', 'CAMERA'}:
                    needs_reset = True
                    print(f"[DIYRenderer] Transform changed: {obj.name}")
                    break
            
            if isinstance(obj, bpy.types.Material):
                needs_reset = True
                print(f"[DIYRenderer] Material changed: {obj.name}")
                break
            
            if isinstance(obj, bpy.types.World):
                needs_reset = True
                print("[DIYRenderer] World changed")
                break
            
            if isinstance(obj, bpy.types.Light):
                needs_reset = True
                print(f"[DIYRenderer] Light changed: {obj.name}")
                break
        
        if not needs_reset:
            return
        
        get_scene_cache().invalidate()
        
        if hasattr(self, 'texture'):
            try:
                del self.texture
            except Exception:
                pass
            self.texture = None
        self.viewport_pixels_cache = None
        
        self.viewport_last_change_time = time.time()
        self.high_res_complete = False
        self.accumulated_samples = {}
        self.current_render_cancelled = True
        
        try:
            while not self.render_queue.empty():
                self.render_queue.get_nowait()
        except queue.Empty:
            pass

    def _detect_camera_change(self, context):
        """Detect if the viewport camera has changed."""
        region_data = context.region_data
        if region_data is None:
            return False
        
        current_matrix = region_data.view_matrix.copy()
        current_perspective = region_data.view_perspective
        current_distance = region_data.view_distance
        
        changed = False
        
        if self.last_view_perspective != current_perspective:
            changed = True
        
        if hasattr(self, 'last_view_distance') and self.last_view_distance is not None:
            if self.last_view_distance > 0:
                distance_change = abs(current_distance - self.last_view_distance) / self.last_view_distance
                if distance_change > 0.01:
                    changed = True
        
        if self.last_camera_matrix is not None and not changed:
            max_diff = 0.0
            for i in range(4):
                for j in range(4):
                    max_diff = max(max_diff, abs(current_matrix[i][j] - self.last_camera_matrix[i][j]))
            if max_diff > 0.001:
                changed = True
        
        if changed or self.last_camera_matrix is None:
            self.last_camera_matrix = current_matrix
            self.last_view_perspective = current_perspective
            self.last_view_distance = current_distance
        
        return changed

    def view_draw(self, context, depsgraph):
        """Viewport rendering function - called continuously while viewport is active."""
        self._init_async_render()
        region = context.region
        width = region.width
        height = region.height
        
        target_samples = context.scene.diy_renderer.viewport_samples
        
        import gpu
        from gpu_extras.presets import draw_texture_2d
        
        if not hasattr(self, 'viewport_last_change_time'):
            self.viewport_last_change_time = time.time()
        if not hasattr(self, 'last_render_width'):
            self.last_render_width = 0
            self.last_render_height = 0
        if not hasattr(self, 'last_moving_render_time'):
            self.last_moving_render_time = 0
            
        current_time = time.time()
        
        camera_changed = self._detect_camera_change(context)
        if camera_changed:
            self.viewport_last_change_time = current_time
            self.high_res_complete = False
            self.accumulated_samples = {}
        
        time_since_change = current_time - self.viewport_last_change_time
        
        MOVING_RENDER_INTERVAL = 0.1
        
        if time_since_change < 0.3:
            scale_factor = 8
            render_width = width // scale_factor
            render_height = height // scale_factor
            samples_per_iteration = 1
            viewport_bounces = 4
            is_moving = True
        else:
            scale_factor = 2
            render_width = width // scale_factor
            render_height = height // scale_factor
            samples_per_iteration = 1
            viewport_bounces = 8
            is_moving = False
        
        if not hasattr(self, '_last_debug_state'):
            self._last_debug_state = None
            self._last_debug_time = 0
        debug_state = f"moving={is_moving}, res={render_width}x{render_height}"
        if debug_state != self._last_debug_state or (current_time - self._last_debug_time > 2.0):
            if current_time - self._last_debug_time > 5.0:
                print(f"[DIYRenderer] State: {debug_state}, time_idle={time_since_change:.2f}s")
            self._last_debug_state = debug_state
            self._last_debug_time = current_time
        
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        self.frame_counter += 1
        
        tile_key = f"{render_width}x{render_height}"
        
        if tile_key in self.accumulated_samples:
            current_sample_count = self.accumulated_samples[tile_key][1]
        else:
            current_sample_count = 0
        
        resolution_changed = (self.last_render_width != render_width or 
                             self.last_render_height != render_height)
        
        if resolution_changed:
            self.accumulated_samples = {}
            current_sample_count = 0
        
        thread_running = self.render_thread is not None and self.render_thread.is_alive()
        thread_effectively_running = thread_running and not self.current_render_cancelled
        
        time_since_last_moving_render = current_time - self.last_moving_render_time
        moving_render_allowed = time_since_last_moving_render >= MOVING_RENDER_INTERVAL
        
        needs_new_render = (
            not hasattr(self, 'texture') or self.texture is None or
            resolution_changed or
            (is_moving and moving_render_allowed and not thread_effectively_running) or
            (not is_moving and current_sample_count < target_samples and not thread_effectively_running)
        )
        
        if needs_new_render:
            self.last_render_width = render_width
            self.last_render_height = render_height
            
            if is_moving:
                self.last_moving_render_time = current_time
            
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
                
                if is_moving:
                    scene_file = get_scene_cache().get_cached_file_fast()
                    if not scene_file:
                        scene_file = export_scene_to_file(depsgraph)
                else:
                    scene_file = export_scene_to_file(depsgraph)
                
                if not scene_file:
                    return
                
                diy = context.scene.diy_renderer
                debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
                max_bounces = viewport_bounces if viewport_bounces is not None else diy.max_bounces
                algorithm = diy.sampling_algorithm
                
                if not hasattr(self, 'job_counter'):
                    self.job_counter = 0
                self.job_counter += 1
                job_data = (scene_file, cam_params, render_width, render_height, self.job_counter,
                           samples_per_iteration, tile_key, target_samples, viewport_bounces,
                           max_bounces, debug_mode, algorithm, is_moving)
                
                try:
                    while not self.render_queue.empty():
                        try:
                            self.render_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    self.render_queue.put_nowait(job_data)
                    self.rendering_in_progress = True
                    
                    if not thread_running:
                        self.stop_thread = False
                        self.current_render_cancelled = False
                        self.render_thread = threading.Thread(
                            target=self.async_render_viewport,
                            args=(job_data,),
                            daemon=True
                        )
                        self.render_thread.start()
                    elif self.current_render_cancelled:
                        print("[DIYRenderer] New job queued, waiting for cancelled thread to pick it up")
                except queue.Full:
                    pass
        
        results_processed = 0
        max_results_per_frame = 2
        while results_processed < max_results_per_frame:
            try:
                result = self.result_queue.get_nowait()
                results_processed += 1
                ext_pixels = result['pixels']
                result_width = result['width']
                result_height = result['height']
                result_tile_key = result.get('tile_key', '')
                result_samples = result.get('samples_per_iteration', 1)
                is_partial = result.get('is_partial', False)
                
                expected_len = result_width * result_height * 4
                if ext_pixels and len(ext_pixels) == expected_len:
                    if is_partial:
                        inv_samples = 1.0 / result_samples
                        display_pixels = [v * inv_samples for v in ext_pixels]
                        buffer = gpu.types.Buffer('FLOAT', expected_len, display_pixels)
                        if hasattr(self, 'texture') and self.texture is not None:
                            try:
                                del self.texture
                            except Exception:
                                pass
                        self.texture = gpu.types.GPUTexture((result_width, result_height), format='RGBA16F', data=buffer)
                        self.texture_width = result_width
                        self.texture_height = result_height
                    else:
                        if result_tile_key in self.accumulated_samples:
                            acc_array, prev_count = self.accumulated_samples[result_tile_key]
                            new_count = prev_count + result_samples
                            for i in range(len(acc_array)):
                                acc_array[i] += ext_pixels[i]
                            self.accumulated_samples[result_tile_key] = (acc_array, new_count)
                            inv_count = 1.0 / new_count
                            display_pixels = [v * inv_count for v in acc_array]
                            if new_count >= target_samples:
                                self.high_res_complete = True
                        else:
                            acc_array = array.array('f', ext_pixels)
                            self.accumulated_samples[result_tile_key] = (acc_array, result_samples)
                            inv_samples = 1.0 / result_samples
                            display_pixels = [v * inv_samples for v in ext_pixels]
                        
                        buffer = gpu.types.Buffer('FLOAT', expected_len, display_pixels)
                        if hasattr(self, 'texture') and self.texture is not None:
                            try:
                                del self.texture
                            except Exception:
                                pass
                        self.texture = gpu.types.GPUTexture((result_width, result_height), format='RGBA16F', data=buffer)
                        self.texture_width = result_width
                        self.texture_height = result_height
                        self.rendering_in_progress = False
                else:
                    print(f"[DIYRenderer] Invalid result: pixels={len(ext_pixels) if ext_pixels else 0}, expected={expected_len}")
                    
            except queue.Empty:
                break
        
        thread_alive = self.render_thread is not None and self.render_thread.is_alive()
        
        if tile_key in self.accumulated_samples:
            current_sample_count = self.accumulated_samples[tile_key][1]
        
        should_redraw = (
            thread_alive or
            is_moving or
            (not is_moving and current_sample_count < target_samples)
        )
        
        if should_redraw:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        
        if hasattr(self, 'texture') and self.texture is not None:
            draw_texture_2d(self.texture, (0, 0), width, height)
