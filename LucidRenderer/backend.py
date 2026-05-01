"""
Backend - C++ レンダラーへのインターフェース
==========================================

このモジュールは pybind11 経由で C++ レンダラーを呼び出す
ためのラッパークラスを提供します。

主要クラス:
- RendererBackend: C++ レンダラーのラッパー

責務:
- pybind11 モジュールの初期化と管理
- スレッドプールの管理
- 非同期レンダリングジョブの管理
- キャンセル処理
- シーンのロードとカメラ設定

使用例:
    backend = RendererBackend()
    backend.load_scene(scene_json)
    backend.set_camera(camera_params)
    future = backend.render_tile_async(params)
    result = future.result()
"""

from __future__ import annotations

import os
import sys
import hashlib
import warnings
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future

from .state import CameraParams, RenderParams, RenderResult


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
    print(f"[RendererBackend] pybind11 module loaded: version {lucidrenderer.__version__}, OpenMP={lucidrenderer.openmp_enabled}")
except ImportError as e:
    lucidrenderer = None
    PYBIND_AVAILABLE = False
    print(f"[RendererBackend] pybind11 module not available: {e}")


# =============================================================================
# RendererBackend クラス
# =============================================================================

class RendererBackend:
    """C++ レンダラーへのインターフェース
    
    pybind11 経由で C++ レンダラーを呼び出します。
    スレッドプールを管理し、非同期レンダリングをサポートします。
    
    Attributes:
        is_available: pybind11 モジュールが利用可能か
    """
    
    def __init__(self):
        """初期化"""
        self._renderer: Any = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._scene_hash: Optional[str] = None
        self._job_id: int = 0
        
        if PYBIND_AVAILABLE:
            self._renderer = lucidrenderer.Renderer()
            self._executor = ThreadPoolExecutor(max_workers=1)
            print("[RendererBackend] Initialized with pybind11 renderer")
    
    @property
    def is_available(self) -> bool:
        """pybind11 モジュールが利用可能か"""
        return PYBIND_AVAILABLE and self._renderer is not None
    
    def shutdown(self) -> None:
        """シャットダウン処理"""
        if self._renderer is not None:
            self._renderer.cancel()
        
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        self._renderer = None
        self._scene_hash = None
        print("[RendererBackend] Shutdown complete")
    
    # =========================================================================
    # シーン管理
    # =========================================================================
    
    def load_scene_json(self, scene_json: str) -> bool:
        """JSON 文字列からシーンを読み込み
        
        シーンハッシュをチェックし、変更がある場合のみロードします。
        
        Args:
            scene_json: シーンの JSON 文字列
            
        Returns:
            成功した場合 True
        """
        if not self.is_available:
            return False
        
        # ハッシュをチェック
        scene_hash = hashlib.md5(scene_json.encode()).hexdigest()
        if scene_hash == self._scene_hash:
            return True  # 変更なし
        
        # シーンをロード
        if self._renderer.load_scene_json(scene_json):
            self._scene_hash = scene_hash
            return True
        
        print("[RendererBackend] Failed to load scene")
        return False
    
    def load_scene_file(self, scene_file: str) -> bool:
        """ファイルからシーンを読み込み
        
        Args:
            scene_file: シーンファイルのパス
            
        Returns:
            成功した場合 True
        """
        try:
            with open(scene_file, 'r') as f:
                scene_json = f.read()
            return self.load_scene_json(scene_json)
        except Exception as e:
            print(f"[RendererBackend] Failed to read scene file: {e}")
            return False
    
    def invalidate_scene_cache(self) -> None:
        """シーンキャッシュを無効化"""
        self._scene_hash = None
    
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
        """辞書形式でカメラを設定（後方互換性用）
        
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
            debug_map = {'NORMAL': 'normal', 'ALBEDO': 'albedo', 'EMISSION': 'emission'}
            mode = debug_map.get(params.debug_mode, 'normal')
            pixels = self._renderer.render_debug(
                0, 0, params.width, params.height,
                params.width, params.height,
                mode
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
    ) -> Optional[Future]:
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


# =============================================================================
# シングルトンインスタンス (非推奨)
# =============================================================================
#
# ⚠️ 非推奨 (DEPRECATED) - ADR 003 によりシングルトンパターンは非推奨 ⚠️
#
# 代わりに render_session.py の RenderSession を使用してください。
# RenderSession は各インスタンスが独自の RendererBackend を持ちます。
#
# これらの関数は後方互換のために残されていますが、新しいコードでは
# 使用しないでください。

_backend_instance: Optional[RendererBackend] = None


def get_backend() -> RendererBackend:
    """グローバルな RendererBackend インスタンスを取得
    
    ⚠️ 非推奨: RenderSession を使用してください。
    """
    warnings.warn(
        "get_backend() is deprecated. Use RenderSession instead. "
        "See ADR 003 for details.",
        DeprecationWarning,
        stacklevel=2
    )
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = RendererBackend()
    return _backend_instance


def shutdown_backend() -> None:
    """グローバルな RendererBackend をシャットダウン
    
    ⚠️ 非推奨: RenderSession.shutdown() を使用してください。
    """
    warnings.warn(
        "shutdown_backend() is deprecated. Use RenderSession.shutdown() instead. "
        "See ADR 003 for details.",
        DeprecationWarning,
        stacklevel=2
    )
    global _backend_instance
    if _backend_instance is not None:
        _backend_instance.shutdown()
        _backend_instance = None
