"""
pybind_renderer.py - pybind11 レンダラーのPythonラッパー
==========================================================

pybind11 でビルドした C++ レンダラーモジュール (diyrenderer) を
Blender アドオンから使いやすくするラッパークラス。

主な機能:
- ThreadPoolExecutor によるバックグラウンドレンダリング
- 協調キャンセルによる即座のキャンセル応答
- Future ベースの非同期 API

使用例:
    controller = RenderController()
    controller.start_render(scene_json, camera_params, tile_params)
    # ... 後で ...
    if controller.poll():
        pixels = controller.get_result()
    # キャンセルしたい場合
    controller.cancel()
"""

import os
import sys
import threading
import concurrent.futures
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

# pybind11 モジュールのパスを追加
_addon_dir = os.path.dirname(os.path.abspath(__file__))
_pybind_path = os.path.join(_addon_dir, 'cpp_renderer', 'build')
if _pybind_path not in sys.path:
    sys.path.insert(0, _pybind_path)

# diyrenderer モジュールをインポート（ビルドされていない場合は None）
try:
    import diyrenderer
    PYBIND_AVAILABLE = True
except ImportError:
    diyrenderer = None
    PYBIND_AVAILABLE = False
    print("[pybind_renderer] Warning: diyrenderer module not found. "
          "Build it with: cd cpp_renderer/build && cmake .. -DBUILD_PYBIND=ON && make")


@dataclass
class CameraParams:
    """カメラパラメータ"""
    pos: Tuple[float, float, float]
    dir: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float


@dataclass
class TileParams:
    """タイルレンダリングパラメータ"""
    tile_x: int = 0
    tile_y: int = 0
    tile_w: int = 800
    tile_h: int = 600
    full_w: int = 800
    full_h: int = 600
    samples: int = 1
    sample_offset: int = 0
    max_depth: int = 8


@dataclass
class RenderResult:
    """レンダリング結果"""
    job_id: int
    pixels: List[float]
    width: int
    height: int
    cancelled: bool = False


class RenderController:
    """
    pybind11 レンダラーのコントローラー
    
    ThreadPoolExecutor を使って別スレッドでレンダリングを実行し、
    メインスレッドからのキャンセル要求に即座に応答できます。
    
    使用パターン:
        controller = RenderController()
        
        # レンダリング開始（非同期）
        controller.start_render(scene_json, camera, tile)
        
        # メインループで定期的にポーリング
        while True:
            if controller.poll():
                result = controller.get_result()
                update_viewport(result.pixels)
                break
            
            # ユーザーがカメラを動かした場合
            if camera_changed:
                controller.cancel()  # 即座にキャンセル
                controller.start_render(new_scene, new_camera, new_tile)
    """
    
    def __init__(self):
        if not PYBIND_AVAILABLE:
            raise RuntimeError("diyrenderer module not available")
        
        self._renderer = diyrenderer.Renderer()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._current_future: Optional[concurrent.futures.Future] = None
        self._current_job_id = 0
        self._last_result: Optional[RenderResult] = None
        self._lock = threading.Lock()
        
        # シーンキャッシュ
        self._cached_scene_hash: Optional[str] = None
    
    def __del__(self):
        """クリーンアップ"""
        self.shutdown()
    
    def shutdown(self):
        """コントローラーを終了"""
        self.cancel()
        self._executor.shutdown(wait=False)
    
    # =========================================================================
    # シーン・カメラ設定
    # =========================================================================
    
    def load_scene(self, scene_json: str, scene_hash: Optional[str] = None) -> bool:
        """
        シーンを読み込み
        
        Args:
            scene_json: JSON 形式のシーンデータ
            scene_hash: シーンのハッシュ（キャッシュ用、省略可）
        
        Returns:
            読み込み成功なら True
        """
        # ハッシュが同じならスキップ
        if scene_hash and scene_hash == self._cached_scene_hash:
            return True
        
        success = self._renderer.load_scene_json(scene_json)
        if success:
            self._cached_scene_hash = scene_hash
        return success
    
    def set_camera(self, camera: CameraParams):
        """カメラを設定"""
        self._renderer.set_camera(
            camera.pos[0], camera.pos[1], camera.pos[2],
            camera.dir[0], camera.dir[1], camera.dir[2],
            camera.up[0], camera.up[1], camera.up[2],
            camera.fov
        )
    
    def set_algorithm(self, algorithm: str):
        """アルゴリズムを設定 ('simple', 'nee', 'mis')"""
        self._renderer.set_algorithm(algorithm)
    
    # =========================================================================
    # レンダリング制御
    # =========================================================================
    
    def start_render(self, 
                     scene_json: str,
                     camera: CameraParams,
                     tile: TileParams,
                     scene_hash: Optional[str] = None,
                     algorithm: str = "nee") -> int:
        """
        レンダリングを開始（非同期）
        
        現在実行中のレンダリングがあればキャンセルしてから開始。
        
        Args:
            scene_json: JSON 形式のシーンデータ
            camera: カメラパラメータ
            tile: タイルパラメータ
            scene_hash: シーンハッシュ（キャッシュ用）
            algorithm: パストレーシングアルゴリズム
        
        Returns:
            ジョブ ID
        """
        # 現在のジョブをキャンセル
        self.cancel()
        
        with self._lock:
            self._current_job_id += 1
            job_id = self._current_job_id
            self._last_result = None
        
        def render_task():
            # シーンとカメラを設定
            self.load_scene(scene_json, scene_hash)
            self.set_camera(camera)
            self.set_algorithm(algorithm)
            
            # レンダリング実行（GIL 解放済み）
            pixels = self._renderer.render_tile(
                tile.tile_x, tile.tile_y,
                tile.tile_w, tile.tile_h,
                tile.full_w, tile.full_h,
                tile.samples,
                tile.sample_offset,
                tile.max_depth
            )
            
            # キャンセルされたかチェック
            cancelled = self._renderer.is_cancelled()
            
            return RenderResult(
                job_id=job_id,
                pixels=pixels,
                width=tile.tile_w,
                height=tile.tile_h,
                cancelled=cancelled
            )
        
        self._current_future = self._executor.submit(render_task)
        return job_id
    
    def start_render_debug(self,
                           scene_json: str,
                           camera: CameraParams,
                           tile: TileParams,
                           mode: str,
                           scene_hash: Optional[str] = None) -> int:
        """
        デバッグモードでレンダリングを開始
        
        Args:
            mode: 'normal', 'albedo', 'emission'
        """
        self.cancel()
        
        with self._lock:
            self._current_job_id += 1
            job_id = self._current_job_id
            self._last_result = None
        
        def render_task():
            self.load_scene(scene_json, scene_hash)
            self.set_camera(camera)
            
            pixels = self._renderer.render_debug(
                tile.tile_x, tile.tile_y,
                tile.tile_w, tile.tile_h,
                tile.full_w, tile.full_h,
                mode
            )
            
            cancelled = self._renderer.is_cancelled()
            
            return RenderResult(
                job_id=job_id,
                pixels=pixels,
                width=tile.tile_w,
                height=tile.tile_h,
                cancelled=cancelled
            )
        
        self._current_future = self._executor.submit(render_task)
        return job_id
    
    def cancel(self):
        """
        現在のレンダリングをキャンセル
        
        即座に返る。実際のキャンセルは協調的に行われる。
        """
        # キャンセルフラグを立てる（即座）
        self._renderer.cancel()
        
        # Future の完了を待つ（短時間で終わるはず）
        if self._current_future is not None:
            try:
                # タイムアウト付きで待機（キャンセルが効いていれば短時間で終わる）
                self._current_future.result(timeout=1.0)
            except concurrent.futures.TimeoutError:
                print("[RenderController] Warning: cancel timeout")
            except Exception:
                pass
            self._current_future = None
    
    def poll(self) -> bool:
        """
        レンダリング完了をチェック
        
        Returns:
            完了していれば True
        """
        if self._current_future is None:
            return self._last_result is not None
        
        if self._current_future.done():
            try:
                result = self._current_future.result()
                with self._lock:
                    # 最新のジョブ結果のみ保持
                    if result.job_id == self._current_job_id:
                        self._last_result = result
            except Exception as e:
                print(f"[RenderController] Render error: {e}")
            
            self._current_future = None
            return True
        
        return False
    
    def get_result(self) -> Optional[RenderResult]:
        """
        最後のレンダリング結果を取得
        
        Returns:
            RenderResult または None
        """
        with self._lock:
            return self._last_result
    
    def is_rendering(self) -> bool:
        """レンダリング中かどうか"""
        return self._current_future is not None and not self._current_future.done()
    
    # =========================================================================
    # 情報取得
    # =========================================================================
    
    def is_available(self) -> bool:
        """pybind11 モジュールが利用可能か"""
        return PYBIND_AVAILABLE
    
    def get_mesh_count(self) -> int:
        """読み込まれたメッシュ数"""
        return self._renderer.get_mesh_count()
    
    def get_algorithm(self) -> str:
        """現在のアルゴリズム"""
        return self._renderer.get_algorithm()


# =============================================================================
# ユーティリティ関数
# =============================================================================

def is_pybind_available() -> bool:
    """pybind11 モジュールが利用可能かチェック"""
    return PYBIND_AVAILABLE


def get_pybind_info() -> Dict[str, Any]:
    """pybind11 モジュールの情報を取得"""
    if not PYBIND_AVAILABLE:
        return {
            'available': False,
            'version': None,
            'openmp_enabled': False
        }
    
    return {
        'available': True,
        'version': getattr(diyrenderer, '__version__', 'unknown'),
        'openmp_enabled': getattr(diyrenderer, 'openmp_enabled', False)
    }
