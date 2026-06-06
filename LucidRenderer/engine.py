"""
Lucid Render Engine - Blender レンダーエンジン統合
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
- LucidRenderEngine: bpy.types.RenderEngine のサブクラス
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
# グローバルセッション保持（診断用）
# =============================================================================

# 診断データアクセス用にセッションを保持
_diagnostic_session: Optional[RenderSession] = None

def set_diagnostic_session(session: Optional[RenderSession]):
    """診断用セッションを設定"""
    global _diagnostic_session
    # 古いセッションがあればシャットダウン
    if _diagnostic_session is not None and _diagnostic_session != session:
        try:
            _diagnostic_session.shutdown()
        except:
            pass
    _diagnostic_session = session

def get_diagnostic_session() -> Optional[RenderSession]:
    """診断用セッションを取得"""
    return _diagnostic_session


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
        print("[LucidRenderEngine] WARNING: No camera in scene!")
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
# LucidRenderEngine
# =============================================================================

class LucidRenderEngine(bpy.types.RenderEngine):
    """
    Blender カスタムレンダーエンジン。
    
    Blender に登録される RenderEngine のサブクラスです。
    各インスタンスは独自の RenderSession を持ち、
    他のインスタンス（別のビューポート、マテリアルプレビュー等）と完全に分離されています。
    
    公式の RenderEngine パターンに従い：
    - __init__: インスタンス変数を初期化
    - __del__: クリーンアップして super().__del__() を呼ぶ
    - view_update: Blender データを読み取る（同じスレッドで）
    - view_draw: 描画のみ（重い処理はしない）
    
    Attributes:
        bl_idname: エンジンの内部識別子
        bl_label: UI に表示される名前
        bl_use_preview: マテリアルプレビューをサポート
        bl_use_shading_nodes: シェーディングノードをサポート
    """
    bl_idname = "LUCID_RENDER_MINIMAL"
    bl_label = "Lucid Renderer (Minimal)"
    bl_use_preview = True
    bl_use_shading_nodes = True
    bl_use_shading_nodes_custom = False

    # =========================================================================
    # ライフサイクル
    # =========================================================================
    
    def __init__(self, *args, **kwargs):
        """コンストラクタ - 公式パターンに従う"""
        super().__init__(*args, **kwargs)
        
        # インスタンス変数を明示的に初期化
        self._session: Optional[RenderSession] = None
        self._viewport_renderer: Optional[ViewportRenderer] = None
        
        print("[LucidRenderEngine] __init__ called")
    
    def __del__(self):
        """デストラクタ - 公式パターンに従う"""
        # Blender が StructRNA を既に削除している場合があるので try-except で保護
        try:
            if self._session is not None:
                # 診断用グローバルセッションとして保持されている場合はシャットダウンしない
                if self._session is get_diagnostic_session():
                    print(f"[LucidRenderEngine] Session kept for diagnostics: {self._session}")
                else:
                    print(f"[LucidRenderEngine] Destroying session: {self._session}")
                    self._session.shutdown()
                self._session = None
        except (ReferenceError, AttributeError):
            # Blender が既にオブジェクトを削除している場合は無視
            pass
        
        # 公式パターン: 必ず super().__del__() を呼ぶ
        try:
            super().__del__()
        except (ReferenceError, AttributeError):
            pass

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
        if self._session is None:
            self._session = RenderSession()
            print(f"[LucidRenderEngine] Created new session: {self._session}")
        return self._session
    
    def _get_viewport_renderer(self) -> ViewportRenderer:
        """ViewportRenderer を取得（遅延初期化）"""
        if self._viewport_renderer is None:
            self._viewport_renderer = ViewportRenderer()
        return self._viewport_renderer

    # =========================================================================
    # ビューポートレンダリング
    # =========================================================================

    def view_update(self, context, depsgraph):
        """
        ビューポートモードでシーンが変更された時に呼ばれます。
        
        公式パターンとの違いについて:
        Blender の公式ドキュメントでは view_update でシーンデータを読み取り、
        view_draw では描画のみを行うことが推奨されています。
        
        しかし、本レンダラーでは以下の理由から view_update ではフラグを立てるのみとしています：
        1. pybind11 レンダラーは非同期でレンダリングを行う
        2. レンダリング中にシーンを変更するとセグメンテーション違反の原因となる
        3. シーンのエクスポートと読み込みは viewport.render() 内で、
           前回のレンダリング完了後に安全に行う必要がある
        
        このアプローチにより、ビューポートの応答性を維持しながら
        安全にシーンを更新できます。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        session = self._ensure_session()
        
        # 変更を検出（シーンロードはしない）
        flags = session._scene_sync.detect_changes(depsgraph)
        
        # 変更があった場合、viewport に通知
        if flags != UpdateFlags.NONE:
            print(f"[LucidRenderEngine] view_update: {flags}")
            # viewport.render() で処理されるよう state にフラグを立てる
            session.state.scene_update_pending = True
            # シーンキャッシュを無効化（次回のエクスポートで再生成）
            from .scene_export import get_scene_cache
            get_scene_cache(session.session_id).invalidate()

    def view_draw(self, context, depsgraph):
        """
        ビューポートレンダリングのエントリーポイント。
        
        公式パターンとの違いについて:
        Blender の公式ドキュメントでは view_draw は描画のみを行い、
        重い処理は view_update で行うことが推奨されています。
        
        本実装では、viewport.render() が以下を行います：
        1. 前回のレンダリング結果があれば即座に描画（高速）
        2. シーン更新フラグがあれば、前回レンダリング完了後に
           シーンを再エクスポート・再ロードして新規レンダリングを開始
        3. OpenGL を使用して結果をビューポートに描画
        
        この設計により、レンダリング中でもビューポートは応答し続け、
        シーン更新は安全なタイミングで行われます。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        session = self._ensure_session()
        
        if not session.is_available:
            self._draw_unavailable_message(context)
            return
        
        viewport = self._get_viewport_renderer()
        current_samples, target_samples = viewport.render(context, depsgraph, session.state, session)
        
        # ヘッダーにサンプル数を表示
        self.update_stats("", f"Samples: {current_samples}/{target_samples}")
    
    def _draw_unavailable_message(self, context) -> None:
        """pybind11 が利用不可の場合のメッセージを描画"""
        import blf
        
        region = context.region
        height = region.height
        
        blf.size(0, 20)
        blf.color(0, 1.0, 0.8, 0.2, 1.0)
        blf.position(0, 20, height - 40, 0)
        blf.draw(0, "Lucid Renderer: pybind11 module not available")
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
        lucid = original_scene.lucid_renderer
        target_samples = lucid.samples
        
        print(f"[LucidRenderEngine] Starting F12 render ({width} x {height}, samples: {target_samples})")
        
        # pybind11 が必要
        if not session.is_available:
            print("[LucidRenderEngine] ERROR: pybind11 module not available")
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
        self._render_f12_pybind(session, depsgraph, width, height, cam_params, scene_file, target_samples, lucid)
        
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
        lucid: Any
    ) -> None:
        """pybind11 を使用した F12 レンダリング"""
        # シーンをロード
        if not session.load_scene_file(scene_file):
            self._render_fallback(width, height)
            return
        
        # カメラを設定
        session.set_camera_from_dict(cam_params)
        
        # アルゴリズムを設定
        session.set_algorithm(lucid.sampling_algorithm)
        
        # 診断機能を設定（ユーザー設定に基づく）
        if lucid.enable_diagnostics:
            preset_map = {
                'MINIMAL': 'minimal',
                'STANDARD': 'standard',
                'DETAILED': 'detailed',
            }
            preset = preset_map.get(lucid.diagnostics_preset, 'standard')
            session.enable_diagnostics(preset)
        else:
            session.disable_diagnostics()
        
        # プログレッシブレンダリング
        sample_iterations = self._compute_sample_iterations(target_samples)
        print(f"[LucidRenderEngine] Sample iterations: {sample_iterations}")
        
        accumulated_pixels = None
        total_samples = 0
        max_samples = sum(sample_iterations)
        render_start_time = time.time()
        
        for iteration_samples in sample_iterations:
            if self.test_break():
                session.cancel()
                print("[LucidRenderEngine] Render cancelled")
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
                max_bounces=lucid.max_bounces,
                algorithm=lucid.sampling_algorithm,
                backend=lucid.backend,
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
        
        # 診断データをグローバルに格納（UIパネルからアクセス可能にする）
        if lucid.enable_diagnostics and session.is_diagnostics_enabled():
            self._store_diagnostic_results(session, width, height)
        
        elapsed = time.time() - render_start_time
        print(f"[LucidRenderEngine] F12 render complete: {total_samples} samples in {elapsed:.2f}s")
    
    def _store_diagnostic_results(self, session: RenderSession, width: int, height: int) -> None:
        """診断結果をグローバルマネージャーに格納
        
        Args:
            session: レンダリングセッション
            width: レンダリング幅
            height: レンダリング高さ
        """
        from .diagnostics import (
            GlobalStats, PathGroupInfo, DiagnosticReport, DiagnosticSuggestion,
            set_global_diagnostics
        )
        
        try:
            # 統計を取得
            cpp_stats = session.get_diagnostic_stats()
            if cpp_stats is None:
                print("[LucidRenderEngine] No diagnostic stats available")
                return
            
            # グローバル統計を変換
            global_stats = GlobalStats(
                total_samples=cpp_stats.total_samples,
                active_pixels=cpp_stats.active_pixels,
                total_groups=cpp_stats.total_groups,
                total_overflow=cpp_stats.total_overflow,
                total_variance=cpp_stats.total_variance,
            )
            
            # 上位分散グループを取得
            cpp_groups = session.get_top_variance_groups(10)
            top_variance_groups = []
            if cpp_groups:
                for g in cpp_groups:
                    top_variance_groups.append(PathGroupInfo(
                        pixel_x=g.pixel_x,
                        pixel_y=g.pixel_y,
                        signature=g.signature,
                        sample_count=g.sample_count,
                        mean_luminance=g.mean_luminance,
                        variance_luminance=g.variance_luminance,
                        depth=g.depth,
                        coarse_type_name=g.coarse_type_name,
                    ))
            
            # JSON メタデータを取得
            metadata_json = session.export_diagnostic_json()
            
            # レポートを作成
            report = DiagnosticReport(
                width=width,
                height=height,
                global_stats=global_stats,
                top_variance_groups=top_variance_groups,
                top_mean_groups=[],  # 簡略化
                suggestions=[],       # TODO: C++ suggestions を接続
                metadata_json=metadata_json,
            )
            
            # セッションをグローバルに保持（診断用）
            # これにより、エンジンがデストラクトされても診断データにアクセスできる
            set_diagnostic_session(session)
            
            # 簡易マネージャーラッパーを作成してグローバルに格納
            class _DiagnosticResultsWrapper:
                def __init__(self, report):
                    self._report = report
                    self.is_available = True
                
                def get_report(self, top_n=10):
                    return self._report
                
                def get_pixel_diagnostic(self, x: int, y: int):
                    """ピクセル単位の診断データを取得（グローバルセッションを使用）"""
                    session = get_diagnostic_session()
                    if session is None:
                        print(f"[_DiagnosticResultsWrapper] diagnostic session is None")
                        return None
                    if not session.is_diagnostics_enabled():
                        print(f"[_DiagnosticResultsWrapper] diagnostics not enabled")
                        return None
                    return session.get_pixel_diagnostic(x, y)
                
                def get_film(self):
                    """互換性のためのダミーメソッド"""
                    return None
            
            wrapper = _DiagnosticResultsWrapper(report)
            set_global_diagnostics(wrapper)
            
            print(f"[LucidRenderEngine] Diagnostic results stored: {global_stats.total_samples} samples, "
                  f"{global_stats.active_pixels} active pixels, {len(top_variance_groups)} variance groups")
            
        except Exception as e:
            print(f"[LucidRenderEngine] Failed to store diagnostic results: {e}")
            import traceback
            traceback.print_exc()
    
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
