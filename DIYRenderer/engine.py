"""
DIY Render Engine - Blender レンダーエンジン統合
===============================================

このファイルは Blender RenderEngine のサブクラスを定義します。

設計思想（ADR 003 参照）:
- 各 RenderEngine インスタンスが独自の RenderSession を持つ
- シングルトンを使用しない（マルチインスタンス対応）
- SceneSync による効率的な変更検出と差分更新

処理フロー:
1. Blender が RenderEngine インスタンスを作成
2. view_update() で SceneSync が変更を検出・同期
3. view_draw() で RenderSession がレンダリング実行
4. render() で F12 レンダリング実行

主要クラス:
- DIYRenderEngine: bpy.types.RenderEngine のサブクラス
"""

import bpy
import math
import time
import array
from typing import Optional, Any

from mathutils import Vector

from .render_session import RenderSession
from .scene_sync import UpdateFlags
from .state import RenderParams
from .viewport import ViewportRenderer
from .scene_export import export_scene_to_file


# =============================================================================
# カメラパラメータ計算
# =============================================================================

def compute_camera_params(scene, width: int, height: int) -> Optional[dict]:
    """
    外部レンダラー用のカメラパラメータを計算します。
    
    Args:
        scene: Blender シーン
        width: レンダリング幅
        height: レンダリング高さ
    
    Returns:
        dict: {'pos': Vector, 'dir': Vector, 'up': Vector, 'fov': float}
        または、カメラがない場合 None
    """
    cam = scene.camera
    if not cam:
        print("[DIYRenderEngine] WARNING: No camera in scene!")
        return None
    
    # カメラのワールド変換行列から位置と方向を取得
    cam_matrix = cam.matrix_world
    pos = cam_matrix.translation
    
    # カメラのローカル -Z がワールド空間の視線方向
    # カメラのローカル +Y がワールド空間の上方向
    forward = cam_matrix.to_3x3() @ Vector((0, 0, -1))
    up = cam_matrix.to_3x3() @ Vector((0, 1, 0))
    forward.normalize()
    up.normalize()
    
    # Blender のカメラから垂直 FOV を計算
    # Blender は sensor_fit で水平/垂直/自動を選択できる
    cam_data = cam.data
    
    # センサーサイズとアスペクト比を取得
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    lens = cam_data.lens
    aspect = width / height
    
    # sensor_fit に基づいて実際に使用するセンサー寸法を計算
    # これは Blender の view3d_utils.py と同じロジック
    if cam_data.sensor_fit == 'VERTICAL':
        # 垂直フィット: センサー高さを基準
        sensor_size = sensor_height
        vfov_rad = 2.0 * math.atan(sensor_size / (2.0 * lens))
    elif cam_data.sensor_fit == 'HORIZONTAL':
        # 水平フィット: センサー幅を基準
        hfov_rad = 2.0 * math.atan(sensor_width / (2.0 * lens))
        # 水平 FOV から垂直 FOV に変換
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) / aspect)
    else:  # 'AUTO'
        # AUTO: アスペクト比により水平か垂直か決まる
        if aspect >= 1.0:
            # 横長画像: 水平フィット
            hfov_rad = 2.0 * math.atan(sensor_width / (2.0 * lens))
            vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) / aspect)
        else:
            # 縦長画像: 垂直フィット
            sensor_size = sensor_width / aspect
            vfov_rad = 2.0 * math.atan(sensor_size / (2.0 * lens))
    
    vfov_deg = math.degrees(vfov_rad)
    
    return {
        'pos': pos,
        'dir': forward,
        'up': up,
        'fov': vfov_deg  # 垂直 FOV (degrees)
    }


# =============================================================================
# DIYRenderEngine
# =============================================================================

class DIYRenderEngine(bpy.types.RenderEngine):
    """
    Blender カスタムレンダーエンジン。
    
    Blender に登録される RenderEngine のサブクラスです。
    各インスタンスは独自の RenderSession を持ち、
    他のインスタンス（別のビューポート、マテリアルプレビュー等）と完全に分離されています。
    
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

    # =========================================================================
    # セッション管理
    # =========================================================================
    
    def _ensure_session(self) -> RenderSession:
        """RenderSession を取得（遅延初期化）
        
        各 RenderEngine インスタンスは独自の RenderSession を持ちます。
        これにより、複数のビューポート、F12、マテリアルプレビューが
        同時に動作できます。
        
        Returns:
            RenderSession: このインスタンス専用のセッション
        """
        if not hasattr(self, '_session') or self._session is None:
            self._session = RenderSession()
            print(f"[DIYRenderEngine] Created new session: {self._session}")
        return self._session
    
    def _get_viewport_renderer(self) -> ViewportRenderer:
        """ViewportRenderer を取得（遅延初期化）"""
        if not hasattr(self, '_viewport_renderer') or self._viewport_renderer is None:
            self._viewport_renderer = ViewportRenderer()
        return self._viewport_renderer
    
    def __del__(self):
        """デストラクタ"""
        # Blender が StructRNA を既に削除している場合があるので try-except で保護
        try:
            if hasattr(self, '_session') and self._session is not None:
                print(f"[DIYRenderEngine] Destroying session: {self._session}")
                self._session.shutdown()
                self._session = None
        except ReferenceError:
            # Blender が既にオブジェクトを削除している場合は無視
            pass

    # =========================================================================
    # ビューポートレンダリング
    # =========================================================================

    def view_update(self, context, depsgraph):
        """
        ビューポートモードでシーンが変更された時に呼ばれます。
        
        注意: ここではシーンの同期を行わず、フラグを立てるのみ。
        実際のシーンロードは view_draw() -> viewport.render() の非同期処理で行う。
        （レンダリング中にシーンを変更するとセグフォの原因になるため）
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        session = self._ensure_session()
        
        # 変更を検出（シーンロードはしない）
        flags = session._scene_sync.detect_changes(depsgraph)
        
        # 変更があった場合、viewport に通知
        if flags != UpdateFlags.NONE:
            print(f"[DIYRenderEngine] view_update: {flags}")
            # viewport.render() で処理されるよう state にフラグを立てる
            session.state.scene_update_pending = True
            # シーンキャッシュを無効化（次回のエクスポートで再生成）
            from .scene_export import get_scene_cache
            get_scene_cache(session.session_id).invalidate()

    def view_draw(self, context, depsgraph):
        """
        ビューポートレンダリングのエントリーポイント。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        session = self._ensure_session()
        
        if not session.is_available:
            self._draw_unavailable_message(context)
            return
        
        viewport = self._get_viewport_renderer()
        viewport.render(context, depsgraph, session.state, session)
    
    def _draw_unavailable_message(self, context) -> None:
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
    # F12 レンダリング
    # =========================================================================

    def render(self, depsgraph):
        """
        F12 レンダリングのメインエントリーポイント。
        
        Args:
            depsgraph: Blender の依存関係グラフ
        """
        session = self._ensure_session()
        
        scene = depsgraph.scene_eval
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        
        original_scene = depsgraph.scene
        diy = original_scene.diy_renderer
        target_samples = diy.samples
        
        print(f"[DIYRenderEngine] Starting F12 render ({width} x {height}, samples: {target_samples})")
        
        # pybind11 が必要
        if not session.is_available:
            print("[DIYRenderEngine] ERROR: pybind11 module not available")
            self._render_fallback(width, height)
            return
        
        # カメラパラメータを計算
        cam_params = compute_camera_params(scene, width, height)
        if not cam_params:
            self._render_fallback(width, height)
            return
        
        # シーンをエクスポート（セッションIDを渡してファイル分離）
        scene_file = export_scene_to_file(depsgraph, use_cache=False, session_id=session.session_id)
        if not scene_file:
            self._render_fallback(width, height)
            return
        
        # pybind11 でレンダリング
        self._render_f12_pybind(session, depsgraph, width, height, cam_params, scene_file, target_samples, diy)
        
        # シーンファイルを削除
        import os
        try:
            if scene_file and os.path.isfile(scene_file):
                os.remove(scene_file)
        except Exception:
            pass
    
    def _render_f12_pybind(
        self,
        session: RenderSession,
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
        if not session.load_scene_file(scene_file):
            self._render_fallback(width, height)
            return
        
        # カメラを設定
        session.set_camera_from_dict(cam_params)
        
        # アルゴリズムを設定
        session.set_algorithm(diy.sampling_algorithm)
        
        # プログレッシブレンダリング
        sample_iterations = self._compute_sample_iterations(target_samples)
        print(f"[DIYRenderEngine] Sample iterations: {sample_iterations}")
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        render_start_time = time.time()
        
        for iteration_samples in sample_iterations:
            if self.test_break():
                session.cancel()
                print("[DIYRenderEngine] Render cancelled")
                break
            
            # 進捗を更新
            progress = total_samples / max_samples
            elapsed = time.time() - render_start_time
            eta = (elapsed / max(progress, 0.001)) * (1 - progress) if progress > 0 else 0
            self.update_stats("", f"Sample {total_samples}/{max_samples} | ETA: {eta:.1f}s")
            self.update_progress(progress)
            
            # レンダリング
            params = RenderParams(
                width=width,
                height=height,
                samples=iteration_samples,
                sample_offset=total_samples,
                max_bounces=diy.max_bounces,
                algorithm=diy.sampling_algorithm
            )
            result = session.render_tile(params)
            
            if result.cancelled:
                break
            
            # 累積（合計として保持、表示時に平均化）
            if accumulated_pixels is None:
                accumulated_pixels = array.array('f', result.pixels)
            else:
                for i in range(len(accumulated_pixels)):
                    accumulated_pixels[i] += result.pixels[i]
            
            total_samples += iteration_samples
            
            # 途中結果を表示
            self._update_render_result(accumulated_pixels, width, height, total_samples)
        
        # 最終結果
        if accumulated_pixels:
            self._update_render_result(accumulated_pixels, width, height, total_samples)
        
        elapsed = time.time() - render_start_time
        print(f"[DIYRenderEngine] F12 render complete: {total_samples} samples in {elapsed:.2f}s")
    
    def _compute_sample_iterations(self, target_samples: int) -> list:
        """サンプル分割を計算"""
        if target_samples <= 16:
            return [target_samples]
        
        iterations = []
        remaining = target_samples
        batch = 4
        
        while remaining > 0:
            samples = min(batch, remaining)
            iterations.append(samples)
            remaining -= samples
            batch = min(batch * 2, 64)
        
        return iterations
    
    def _update_render_result(self, pixels, width: int, height: int, total_samples: int = 1) -> None:
        """レンダリング結果を更新
        
        Args:
            pixels: フラットなピクセル配列（累積合計）[r,g,b,a,r,g,b,a,...]
            width: 幅
            height: 高さ
            total_samples: 累積サンプル数（平均化に使用）
        """
        result = self.begin_result(0, 0, width, height)
        if result:
            layer = result.layers[0].passes["Combined"]
            # Blender は [[r,g,b,a], [r,g,b,a], ...] 形式を期待
            # フラット配列を変換し、サンプル数で平均化
            inv_samples = 1.0 / total_samples
            display_pixels = []
            for i in range(0, len(pixels), 4):
                display_pixels.append([
                    pixels[i] * inv_samples,
                    pixels[i+1] * inv_samples,
                    pixels[i+2] * inv_samples,
                    1.0  # アルファは常に1.0
                ])
            layer.rect = display_pixels
            self.end_result(result)
    
    def _render_fallback(self, width: int, height: int) -> None:
        """フォールバックレンダリング（エラー表示用）"""
        result = self.begin_result(0, 0, width, height)
        if result:
            layer = result.layers[0].passes["Combined"]
            # 暗い赤でエラーを示す
            pixels = [0.2, 0.0, 0.0, 1.0] * (width * height)
            layer.rect = pixels
            self.end_result(result)


# =============================================================================
# 後方互換性用グローバル関数
# =============================================================================

def get_pybind_renderer():
    """pybind11 レンダラーを取得（後方互換性用）
    
    警告: この関数は非推奨です。
    代わりに RenderSession を使用してください。
    """
    print("[WARNING] get_pybind_renderer() is deprecated. Use RenderSession instead.")
    return None


def stop_pybind_renderer():
    """pybind11 レンダラーを停止（後方互換性用）
    
    警告: この関数は非推奨です。
    RenderSession は自動的にクリーンアップされます。
    """
    print("[WARNING] stop_pybind_renderer() is deprecated.")
