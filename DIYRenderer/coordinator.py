"""
Coordinator - レンダリングの調整役
==================================

このモジュールはレンダリング全体の調整を担当します。
状態管理を一元化し、ビューポート/F12 レンダリングの切り替えを管理します。

主要クラス:
- RenderCoordinator: レンダリング調整のメインクラス

責務:
- 状態管理の一元化
- ビューポート/F12 の切り替え
- シーン変更の通知処理
- シャットダウン処理

使用例:
    coordinator = RenderCoordinator()
    coordinator.render_viewport(context, depsgraph)
    coordinator.render_f12(depsgraph)
"""

from __future__ import annotations

import time
import array
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import bpy

from .state import ViewportState, RenderParams, CameraParams, RENDER_CONSTANTS
from .backend import RendererBackend, get_backend, shutdown_backend
from .viewport import ViewportRenderer
from .scene_export import get_scene_cache


class RenderCoordinator:
    """レンダリングの調整役
    
    レンダリング全体の状態を管理し、各レンダラーに委譲します。
    engine.py のグローバル状態とインスタンス状態を統合したものです。
    """
    
    def __init__(self):
        """初期化"""
        self._state = ViewportState()
        self._viewport = ViewportRenderer()
        self._backend: Optional[RendererBackend] = None
    
    @property
    def backend(self) -> RendererBackend:
        """バックエンドを取得（遅延初期化）"""
        if self._backend is None:
            self._backend = get_backend()
        return self._backend
    
    @property
    def state(self) -> ViewportState:
        """状態を取得"""
        return self._state
    
    # =========================================================================
    # ビューポートレンダリング
    # =========================================================================
    
    def render_viewport(self, context: Any, depsgraph: Any) -> None:
        """ビューポートをレンダリング
        
        engine.view_draw から呼び出されます。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        if not self.backend.is_available:
            self._draw_unavailable_message(context)
            return
        
        self._viewport.render(context, depsgraph, self._state, self.backend)
    
    def _draw_unavailable_message(self, context: Any) -> None:
        """pybind11 が利用不可の場合のメッセージを描画"""
        import blf
        
        region = context.region
        height = region.height
        
        blf.size(0, 20)
        blf.color(0, 1.0, 0.8, 0.2, 1.0)
        blf.position(0, 20, height - 40, 0)
        blf.draw(0, "DIY Renderer: pybind11 module not available")
        blf.position(0, 20, height - 70, 0)
        blf.draw(0, "Please build the C++ module with: cmake .. -DBUILD_PYBIND=ON && make")
    
    # =========================================================================
    # シーン変更通知
    # =========================================================================
    
    def on_scene_update(self, depsgraph: Any) -> None:
        """シーンが更新された時の処理
        
        engine.view_update から呼び出されます。
        
        Args:
            depsgraph: 依存関係グラフ
        """
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
            
            import bpy
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
        
        # シーンが変更されたことを記録
        self._state.scene_update_pending = True
        
        # 累積サンプルをリセット（テクスチャは維持）
        self._state.accumulated_samples = {}
    
    # =========================================================================
    # F12 レンダリング
    # =========================================================================
    
    def render_f12(
        self,
        engine: Any,
        depsgraph: Any
    ) -> None:
        """F12 レンダリングを実行
        
        engine.render から呼び出されます。
        
        Args:
            engine: RenderEngine インスタンス
            depsgraph: 依存関係グラフ
        """
        from .scene_export import export_scene_to_file
        from .renderer import compute_camera_params
        
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        
        original_scene = depsgraph.scene
        diy = original_scene.diy_renderer
        target_samples = diy.samples
        
        print(f"[RenderCoordinator] Starting F12 render ({width} x {height}, samples: {target_samples})")
        
        # カメラパラメータを計算
        cam_params = compute_camera_params(scene, width, height)
        if not cam_params:
            self._render_fallback(engine, width, height)
            return
        
        # シーンをエクスポート
        scene_file = export_scene_to_file(depsgraph, use_cache=False)
        if not scene_file:
            self._render_fallback(engine, width, height)
            return
        
        # pybind11 でレンダリング
        if self.backend.is_available:
            self._render_f12_pybind(engine, depsgraph, width, height, cam_params, scene_file, target_samples, diy)
        else:
            # レガシーモードへのフォールバック
            self._render_f12_legacy(engine, depsgraph, width, height, cam_params, scene_file, target_samples, diy)
        
        # シーンファイルを削除
        import os
        try:
            if scene_file and os.path.isfile(scene_file):
                os.remove(scene_file)
        except Exception:
            pass
    
    def _render_f12_pybind(
        self,
        engine: Any,
        depsgraph: Any,
        width: int,
        height: int,
        cam_params: dict,
        scene_file: str,
        target_samples: int,
        diy: Any
    ) -> None:
        """pybind11 を使用した F12 レンダリング"""
        # シーンをロード
        if not self.backend.load_scene_file(scene_file):
            self._render_fallback(engine, width, height)
            return
        
        # カメラを設定
        self.backend.set_camera_from_dict(cam_params)
        
        # アルゴリズムを設定
        self.backend.set_algorithm(diy.sampling_algorithm)
        
        # プログレッシブレンダリング
        sample_iterations = self._compute_sample_iterations(target_samples)
        print(f"[RenderCoordinator] Sample iterations: {sample_iterations}")
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        render_start_time = time.time()
        
        for iteration_samples in sample_iterations:
            if engine.test_break():
                self.backend.cancel()
                print("[RenderCoordinator] Render cancelled")
                break
            
            # 進捗を更新
            elapsed = time.time() - render_start_time
            self._update_progress(engine, total_samples, max_samples, elapsed)
            
            # レンダリング実行
            params = RenderParams(
                width=width,
                height=height,
                samples=iteration_samples,
                max_bounces=diy.max_bounces,
                algorithm=diy.sampling_algorithm,
                debug_mode=diy.debug_mode if diy.debug_mode != 'NONE' else None,
                sample_offset=total_samples
            )
            
            result = self.backend.render_tile(params)
            
            if result.cancelled:
                print("[RenderCoordinator] Render was cancelled")
                break
            
            if not result.pixels or len(result.pixels) != width * height * 4:
                continue
            
            # 累積
            if accumulated_pixels is None:
                accumulated_pixels = array.array('f', result.pixels)
                total_samples = iteration_samples
            else:
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i] += result.pixels[i]
                total_samples += iteration_samples
            
            # 表示を更新
            self._update_render_result(engine, width, height, accumulated_pixels, total_samples)
        
        total_elapsed = time.time() - render_start_time
        print(f"[RenderCoordinator] F12 render complete ({total_samples} samples) in {self._format_time(total_elapsed)}")
    
    def _render_f12_legacy(
        self,
        engine: Any,
        depsgraph: Any,
        width: int,
        height: int,
        cam_params: dict,
        scene_file: str,
        target_samples: int,
        diy: Any
    ) -> None:
        """レガシーモードでの F12 レンダリング"""
        from .renderer import call_external_renderer
        
        sample_iterations = self._compute_sample_iterations(target_samples)
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        render_start_time = time.time()
        
        for iteration_samples in sample_iterations:
            if engine.test_break():
                print("[RenderCoordinator] Render cancelled")
                break
            
            elapsed = time.time() - render_start_time
            self._update_progress(engine, total_samples, max_samples, elapsed)
            
            debug_mode = diy.debug_mode if diy.debug_mode != 'NONE' else None
            
            iteration_pixels = call_external_renderer(
                scene_file, 0, 0, width, height, width, height, cam_params,
                samples=iteration_samples,
                depth=diy.max_bounces,
                debug_mode=debug_mode,
                cancel_check=engine.test_break,
                sample_offset=total_samples,
                algorithm=diy.sampling_algorithm
            )
            
            if iteration_pixels is None:
                continue
            
            if len(iteration_pixels) != width * height * 4:
                continue
            
            if accumulated_pixels is None:
                accumulated_pixels = array.array('f', iteration_pixels)
                total_samples = iteration_samples
            else:
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i] += iteration_pixels[i]
                total_samples += iteration_samples
            
            self._update_render_result(engine, width, height, accumulated_pixels, total_samples)
        
        total_elapsed = time.time() - render_start_time
        print(f"[RenderCoordinator] Legacy render complete ({total_samples} samples) in {self._format_time(total_elapsed)}")
    
    def _compute_sample_iterations(self, target_samples: int) -> list:
        """サンプルイテレーションを計算"""
        iterations = []
        current = 1
        total = 0
        max_increment = 128
        while total < target_samples:
            to_add = min(current, target_samples - total)
            iterations.append(to_add)
            total += to_add
            if current < max_increment:
                current *= 2
        return iterations
    
    def _update_progress(
        self,
        engine: Any,
        current_samples: int,
        max_samples: int,
        elapsed: float
    ) -> None:
        """進捗を更新"""
        if current_samples > 0:
            time_per_sample = elapsed / current_samples
            remaining_time = time_per_sample * (max_samples - current_samples)
            time_str = f"Elapsed: {self._format_time(elapsed)} | Remaining: {self._format_time(remaining_time)}"
        else:
            time_str = f"Elapsed: {self._format_time(elapsed)}"
        
        engine.update_progress(current_samples / max_samples)
        engine.update_stats("", f"Path Tracing: {current_samples}/{max_samples} samples | {time_str}")
    
    def _update_render_result(
        self,
        engine: Any,
        width: int,
        height: int,
        accumulated_pixels: array.array,
        total_samples: int
    ) -> None:
        """レンダリング結果を更新"""
        inv_samples = 1.0 / total_samples
        display_pixels = []
        for i in range(0, len(accumulated_pixels), 4):
            display_pixels.append([
                accumulated_pixels[i] * inv_samples,
                accumulated_pixels[i+1] * inv_samples,
                accumulated_pixels[i+2] * inv_samples,
                1.0
            ])
        
        result = engine.begin_result(0, 0, width, height)
        combined = result.layers[0].passes["Combined"]
        combined.rect = display_pixels
        engine.end_result(result)
    
    def _render_fallback(self, engine: Any, width: int, height: int) -> None:
        """フォールバック（グラデーション）レンダリング"""
        pixels = []
        for y in range(height):
            fy = y / (height - 1) if height > 1 else 0.0
            for x in range(width):
                fx = x / (width - 1) if width > 1 else 0.0
                pixels.append([fx, fy, 0.2, 1.0])
        
        result = engine.begin_result(0, 0, width, height)
        combined = result.layers[0].passes["Combined"]
        combined.rect = pixels
        engine.end_result(result)
    
    def _format_time(self, seconds: float) -> str:
        """時間をフォーマット"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
    
    # =========================================================================
    # シャットダウン
    # =========================================================================
    
    def shutdown(self) -> None:
        """シャットダウン"""
        shutdown_backend()
        self._backend = None
        print("[RenderCoordinator] Shutdown complete")


# =============================================================================
# シングルトンインスタンス
# =============================================================================

_coordinator_instance: Optional[RenderCoordinator] = None


def get_coordinator() -> RenderCoordinator:
    """グローバルな RenderCoordinator インスタンスを取得"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = RenderCoordinator()
    return _coordinator_instance


def shutdown_coordinator() -> None:
    """グローバルな RenderCoordinator をシャットダウン"""
    global _coordinator_instance
    if _coordinator_instance is not None:
        _coordinator_instance.shutdown()
        _coordinator_instance = None
