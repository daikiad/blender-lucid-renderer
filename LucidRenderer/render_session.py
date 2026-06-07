"""
RenderSession - レンダリングセッション管理
==========================================

このモジュールは各 RenderEngine インスタンスに対応する
レンダリングセッションを管理します。

設計思想（ADR 003 参照）:
- 各 RenderEngine インスタンスが独自の RenderSession を持つ
- シングルトンを廃止し、マルチインスタンス対応
- SceneSync と連携して効率的な差分更新を実現

主要クラス:
- RenderSession: レンダリングセッション

使用例:
    session = RenderSession()
    session.load_scene_if_changed(scene_hash, json_str)
    session.set_camera(camera_params)
    result = session.render_tile(params)
"""

from __future__ import annotations

import os
import sys
import hashlib
from typing import Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from .state import CameraParams, RenderParams, RenderResult, ViewportState
from .scene_sync import SceneSync, UpdateFlags


# =============================================================================
# pybind11 モジュールのインポート
# =============================================================================

_addon_dir = os.path.dirname(os.path.abspath(__file__))
_pybind_paths = [
    os.path.join(_addon_dir, 'cpp_renderer', 'build_pybind'),
    os.path.join(_addon_dir, 'cpp_renderer', 'build'),
]
for _path in _pybind_paths:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import lucidrenderer
    PYBIND_AVAILABLE = True
except ImportError as e:
    lucidrenderer = None
    PYBIND_AVAILABLE = False
    print(f"[RenderSession] pybind11 module not available: {e}")


# =============================================================================
# RenderSession クラス
# =============================================================================

class RenderSession:
    """レンダリングセッション
    
    1つの RenderEngine インスタンスに対応するセッションです。
    各セッションは独自の C++ Renderer インスタンスを持ち、
    他のセッションと完全に分離されています。
    
    Attributes:
        is_available: pybind11 モジュールが利用可能か
        scene_sync: シーン同期マネージャー
        state: ビューポート状態
    """
    
    # セッション ID の自動採番
    _session_counter: int = 0
    
    def __init__(self):
        """初期化"""
        # セッション ID を割り当て
        RenderSession._session_counter += 1
        self._session_id = RenderSession._session_counter
        
        # C++ レンダラーインスタンス（セッション固有）
        self._renderer: Any = None
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # シーン管理
        self._scene_hash: Optional[str] = None
        self._scene_sync = SceneSync()
        
        # ビューポート状態（セッション固有）
        self._state = ViewportState()
        
        # ジョブ管理
        self._job_id: int = 0
        
        # 診断機能
        self._diagnostics_enabled: bool = False
        
        # 初期化
        if PYBIND_AVAILABLE:
            self._renderer = lucidrenderer.Renderer()
            self._executor = ThreadPoolExecutor(max_workers=1)
            print(f"[RenderSession #{self._session_id}] Created with pybind11 renderer")
        else:
            print(f"[RenderSession #{self._session_id}] Created (pybind11 not available)")
    
    @property
    def session_id(self) -> int:
        """セッション ID"""
        return self._session_id
    
    @property
    def is_available(self) -> bool:
        """pybind11 モジュールが利用可能か"""
        return PYBIND_AVAILABLE and self._renderer is not None
    
    @property
    def scene_sync(self) -> SceneSync:
        """シーン同期マネージャー"""
        return self._scene_sync
    
    @property
    def state(self) -> ViewportState:
        """ビューポート状態"""
        return self._state
    
    def shutdown(self) -> None:
        """シャットダウン処理"""
        print(f"[RenderSession #{self._session_id}] Shutting down...")

        if self._renderer is not None:
            # Make sure the C++ async worker is joined before the renderer goes
            # away — otherwise the worker can keep touching freed Dawn resources.
            try:
                self._renderer.render_stop_async()
            except AttributeError:
                pass  # Older build without async API
            self._renderer.cancel()
        
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        self._renderer = None
        self._scene_hash = None
        
        # セッションのシーンキャッシュをクリア
        from .scene_export import SceneCache
        SceneCache.clear_session(self._session_id)
        
        print(f"[RenderSession #{self._session_id}] Shutdown complete")
    
    # =========================================================================
    # シーン管理
    # =========================================================================
    
    def load_scene_if_changed(self, scene_hash: str, json_str: str) -> bool:
        """シーンが変更された場合のみロード
        
        Args:
            scene_hash: シーンのハッシュ
            json_str: シーンの JSON 文字列
            
        Returns:
            bool: 成功した場合 True
        """
        if not self.is_available:
            return False
        
        # ハッシュが同じなら何もしない
        if scene_hash == self._scene_hash:
            return True
        
        # シーンをロード
        if self._renderer.load_scene_json(json_str):
            self._scene_hash = scene_hash
            
            # JSONからオブジェクト名を抽出して設定（診断表示用）
            self._set_object_names_from_json(json_str)
            
            print(f"[RenderSession #{self._session_id}] Scene loaded (hash={scene_hash[:8]}...)")
            return True
        
        print(f"[RenderSession #{self._session_id}] Failed to load scene")
        return False
    
    def _set_object_names_from_json(self, json_str: str) -> None:
        """JSONからオブジェクト名を抽出してレンダラーに設定
        
        Args:
            json_str: シーンの JSON 文字列
        """
        try:
            import json
            scene_data = json.loads(json_str)
            meshes = scene_data.get('meshes', [])
            object_names = [mesh.get('name', f'Object_{i}') for i, mesh in enumerate(meshes)]
            self._renderer.set_object_names(object_names)
            print(f"[RenderSession #{self._session_id}] Set {len(object_names)} object names")
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to set object names: {e}")
    
    def load_scene_json(self, json_str: str) -> bool:
        """JSON 文字列からシーンをロード（ハッシュを自動計算）
        
        Args:
            json_str: シーンの JSON 文字列
            
        Returns:
            bool: 成功した場合 True
        """
        scene_hash = hashlib.md5(json_str.encode()).hexdigest()
        return self.load_scene_if_changed(scene_hash, json_str)
    
    def load_scene_file(self, scene_file: str) -> bool:
        """ファイルからシーンを読み込み
        
        Args:
            scene_file: シーンファイルのパス
            
        Returns:
            bool: 成功した場合 True
        """
        try:
            with open(scene_file, 'r') as f:
                json_str = f.read()
            return self.load_scene_json(json_str)
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to read scene file: {e}")
            return False
    
    def invalidate_scene(self) -> None:
        """シーンキャッシュを無効化"""
        self._scene_hash = None
        self._scene_sync.reset()
    
    # =========================================================================
    # シーン同期（SceneSync 経由）
    # =========================================================================
    
    def sync_scene(self, depsgraph: Any) -> UpdateFlags:
        """depsgraph からシーンを同期
        
        SceneSync を使用して変更を検出し、必要に応じてシーンを更新します。
        
        Args:
            depsgraph: Blender の依存関係グラフ
            
        Returns:
            UpdateFlags: 検出された変更フラグ
        """
        flags = self._scene_sync.detect_changes(depsgraph)
        
        if flags != UpdateFlags.NONE:
            self._scene_sync.sync(depsgraph, self, flags)
            
            # 累積サンプルをリセット
            if self._scene_sync.needs_render_reset(flags):
                self._state.reset_accumulation()
        
        return flags
    
    # =========================================================================
    # カメラ設定
    # =========================================================================
    
    def set_camera(self, camera: CameraParams) -> None:
        """カメラを設定
        
        Args:
            camera: カメラパラメータ
        """
        if not self.is_available:
            return
        
        self._renderer.set_camera(
            camera.pos[0], camera.pos[1], camera.pos[2],
            camera.dir[0], camera.dir[1], camera.dir[2],
            camera.up[0], camera.up[1], camera.up[2],
            camera.fov
        )
    
    def set_camera_from_dict(self, cam_params: dict) -> None:
        """辞書形式でカメラを設定
        
        Args:
            cam_params: {'pos': Vector, 'dir': Vector, 'up': Vector, 'fov': float}
        """
        if not self.is_available:
            return
        
        pos = cam_params['pos']
        dir_ = cam_params['dir']
        up = cam_params['up']
        fov = cam_params['fov']
        
        self._renderer.set_camera(
            pos[0], pos[1], pos[2],
            dir_[0], dir_[1], dir_[2],
            up[0], up[1], up[2],
            fov
        )
    
    # =========================================================================
    # Async (GPU accumulator) API
    # =========================================================================
    #
    # The C++ side owns a worker thread that fires 1-sample dispatches into an
    # on-GPU accumulator. The viewport polls the worker's snapshot from
    # view_draw and updates the texture. See cpp_renderer/include/gpu/path_tracer.hpp.

    def start_render_async(self, width: int, height: int, max_depth: int) -> bool:
        """Spawn the background path-trace worker. Stops any prior session
        first. Returns True if the worker is now running."""
        if not self.is_available:
            return False
        if not hasattr(self._renderer, 'render_start_async'):
            return False
        self._renderer.render_start_async(width, height, max_depth)
        return self._renderer.is_render_async_running()

    def poll_render_async(self):
        """Return (samples_completed, pixels). samples=0 means no snapshot
        yet — caller should keep its current texture."""
        if not self.is_available:
            return 0, []
        if not hasattr(self._renderer, 'render_poll_async'):
            return 0, []
        return self._renderer.render_poll_async()

    def stop_render_async(self) -> None:
        """Stop the background worker. No-op if not running."""
        if not self.is_available:
            return
        if not hasattr(self._renderer, 'render_stop_async'):
            return
        self._renderer.render_stop_async()

    def reset_render_async(self, width: int, height: int, max_depth: int) -> None:
        """Signal the running worker to wipe its accumulator and pick up the
        new camera/params. Cheap and non-blocking — does NOT join the worker
        thread, does NOT reload the scene. Use this for camera-move events;
        for content/resolution change, stop_render_async + start_render_async."""
        if not self.is_available:
            return
        if not hasattr(self._renderer, 'render_reset_async'):
            # Older build without the reset path — fall back to full restart.
            self.stop_render_async()
            self.start_render_async(width, height, max_depth)
            return
        self._renderer.render_reset_async(width, height, max_depth)

    def is_render_async_running(self) -> bool:
        if not self.is_available:
            return False
        if not hasattr(self._renderer, 'is_render_async_running'):
            return False
        return self._renderer.is_render_async_running()

    def snapshot_revision_async(self) -> int:
        """Monotonic snapshot counter from the C++ worker. Lets view_draw
        skip the (expensive) poll path when nothing has changed since the
        last time it polled."""
        if not self.is_available:
            return 0
        if not hasattr(self._renderer, 'render_snapshot_revision_async'):
            return 0
        return self._renderer.render_snapshot_revision_async()

    # =========================================================================
    # アルゴリズム設定
    # =========================================================================

    def set_algorithm(self, algorithm: str) -> None:
        """アルゴリズムを設定
        
        Args:
            algorithm: 'simple', 'nee', または 'mis'
        """
        if not self.is_available:
            return
        
        self._renderer.set_algorithm(algorithm)
    
    # =========================================================================
    # 診断機能
    # =========================================================================
    
    def enable_diagnostics(self, preset: str = "standard") -> bool:
        """診断機能を有効化
        
        Args:
            preset: 'minimal', 'standard', または 'detailed'
            
        Returns:
            bool: 成功した場合 True
        """
        if not self.is_available:
            return False
        
        try:
            if preset == "minimal":
                config = lucidrenderer.PathRecordingConfig.minimal()
            elif preset == "detailed":
                config = lucidrenderer.PathRecordingConfig.detailed()
            else:
                config = lucidrenderer.PathRecordingConfig.standard()
            
            self._renderer.enable_diagnostics(config)
            self._diagnostics_enabled = True
            print(f"[RenderSession #{self._session_id}] Diagnostics enabled ({preset})")
            return True
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to enable diagnostics: {e}")
            return False
    
    def disable_diagnostics(self) -> None:
        """診断機能を無効化"""
        if not self.is_available:
            return
        
        try:
            self._renderer.disable_diagnostics()
            self._diagnostics_enabled = False
            print(f"[RenderSession #{self._session_id}] Diagnostics disabled")
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to disable diagnostics: {e}")
    
    def is_diagnostics_enabled(self) -> bool:
        """診断機能が有効か"""
        return self._diagnostics_enabled and self.is_available
    
    def get_diagnostic_stats(self):
        """診断統計を取得
        
        Returns:
            GlobalDiagnosticStats or None
        """
        if not self.is_available or not self._diagnostics_enabled:
            return None
        
        try:
            return self._renderer.get_diagnostic_stats()
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to get diagnostic stats: {e}")
            return None
    
    def get_diagnostic_variance_map(self):
        """分散マップを取得
        
        Returns:
            numpy array or None
        """
        if not self.is_available or not self._diagnostics_enabled:
            return None
        
        try:
            return self._renderer.get_diagnostic_variance_map()
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to get variance map: {e}")
            return None
    
    def get_pixel_diagnostic(self, x: int, y: int):
        """ピクセル単位の診断データを取得
        
        Args:
            x: ピクセルX座標
            y: ピクセルY座標
            
        Returns:
            PixelDiagnosticInfo or None
        """
        if not self.is_available or not self._diagnostics_enabled:
            return None
        
        try:
            return self._renderer.get_pixel_diagnostic(x, y)
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to get pixel diagnostic: {e}")
            return None
    
    def get_top_variance_groups(self, count: int = 10):
        """上位分散グループを取得
        
        Args:
            count: 取得するグループ数
            
        Returns:
            List of ExportedGroupInfo or None
        """
        if not self.is_available or not self._diagnostics_enabled:
            return None
        
        try:
            return self._renderer.get_top_variance_groups(count)
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to get top variance groups: {e}")
            return None
    
    def export_diagnostic_json(self) -> str:
        """診断データをJSON形式でエクスポート
        
        Returns:
            JSON文字列
        """
        if not self.is_available or not self._diagnostics_enabled:
            return "{}"
        
        try:
            return self._renderer.export_diagnostic_json()
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to export diagnostic JSON: {e}")
            return "{}"
    
    def clear_diagnostics(self) -> None:
        """診断データをクリア"""
        if not self.is_available:
            return
        
        try:
            self._renderer.clear_diagnostics()
        except Exception as e:
            print(f"[RenderSession #{self._session_id}] Failed to clear diagnostics: {e}")
    
    # =========================================================================
    # レンダリング（同期）
    # =========================================================================
    
    def render_tile(self, params: RenderParams) -> RenderResult:
        """タイルをレンダリング（同期）
        
        Args:
            params: レンダリングパラメータ
            
        Returns:
            レンダリング結果
        """
        if not self.is_available:
            return RenderResult(
                pixels=[],
                width=params.width,
                height=params.height,
                cancelled=True
            )
        
        # アルゴリズムを設定
        self._renderer.set_algorithm(params.algorithm)
        
        # レンダリング実行
        if params.debug_mode:
            # Preferences uses lowercase enum values for the newer modes; map
            # both forms so old/new property values both reach the C++ side.
            debug_map = {
                'NORMAL': 'normal', 'ALBEDO': 'albedo', 'EMISSION': 'emission',
                'normal': 'normal', 'albedo': 'albedo', 'emission': 'emission',
                'volume': 'volume',
            }
            mode = debug_map.get(params.debug_mode, 'normal')
            if params.backend == 'gpu' and hasattr(self._renderer, 'render_debug_gpu'):
                # GPU path (Phase 1b: only 'normal' is real GPU; others fall back internally)
                pixels = self._renderer.render_debug_gpu(
                    0, 0, params.width, params.height,
                    params.width, params.height,
                    mode
                )
            else:
                pixels = self._renderer.render_debug(
                    0, 0, params.width, params.height,
                    params.width, params.height,
                    mode
                )
        else:
            # Path tracer: route to GPU when backend=='gpu' (Phase 2a).
            if params.backend == 'gpu' and hasattr(self._renderer, 'render_tile_gpu'):
                pixels = self._renderer.render_tile_gpu(
                    0, 0, params.width, params.height,
                    params.width, params.height,
                    samples=params.samples,
                    sample_offset=params.sample_offset,
                    max_depth=params.max_bounces
                )
            else:
                pixels = self._renderer.render_tile(
                    0, 0, params.width, params.height,
                    params.width, params.height,
                    samples=params.samples,
                    sample_offset=params.sample_offset,
                    max_depth=params.max_bounces
                )
        
        # キャンセルチェック
        cancelled = self._renderer.is_cancelled()
        
        self._job_id += 1
        
        return RenderResult(
            pixels=pixels,
            width=params.width,
            height=params.height,
            samples=params.samples,
            cancelled=cancelled,
            job_id=self._job_id
        )
    
    # =========================================================================
    # レンダリング（非同期）
    # =========================================================================
    
    def render_tile_async(
        self,
        params: RenderParams,
        scene_file: str,
        camera: CameraParams
    ) -> Optional[Any]:
        """タイルを非同期でレンダリング
        
        Args:
            params: レンダリングパラメータ
            scene_file: シーンファイルのパス
            camera: カメラパラメータ
            
        Returns:
            Future オブジェクト、または None（利用不可の場合）
        """
        if not self.is_available or self._executor is None:
            return None
        
        def render_task() -> RenderResult:
            # シーンをロード
            if not self.load_scene_file(scene_file):
                return RenderResult(
                    pixels=[],
                    width=params.width,
                    height=params.height,
                    cancelled=True
                )
            
            # カメラを設定
            self.set_camera(camera)
            
            # レンダリング実行
            return self.render_tile(params)
        
        return self._executor.submit(render_task)
    
    # =========================================================================
    # キャンセル
    # =========================================================================
    
    def cancel(self) -> None:
        """現在のレンダリングをキャンセル"""
        if self._renderer is not None:
            self._renderer.cancel()
    
    def is_cancelled(self) -> bool:
        """キャンセルされたかどうか"""
        if self._renderer is not None:
            return self._renderer.is_cancelled()
        return False
    
    def reset_cancel(self) -> None:
        """キャンセルフラグをリセット"""
        if self._renderer is not None:
            self._renderer.reset_cancel()
    
    # =========================================================================
    # デバッグ情報
    # =========================================================================
    
    def __repr__(self) -> str:
        """文字列表現"""
        status = "available" if self.is_available else "unavailable"
        return f"<RenderSession #{self._session_id} ({status})>"
