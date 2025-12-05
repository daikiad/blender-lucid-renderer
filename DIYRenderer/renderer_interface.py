"""
Abstract interface for renderer communication.
==============================================

このモジュールは、レンダラーバックエンドとの通信を抽象化する
インターフェースを定義しています。

設計パターン: Strategy / Template Method
異なる通信方式（サブプロセス、pybind11、FFI等）を
統一したインターフェースで扱えるようにします。

主要コンポーネント:
- BackendType: レンダリングバックエンド（CPU / WebGPU）
- AlgorithmType: パストレーシングアルゴリズム
- RenderConfig: レンダラー初期化設定
- CameraParams: カメラパラメータ
- TileParams: タイルレンダリングパラメータ
- RenderResult: レンダリング結果
- RendererInterface: 抽象基底クラス

実装クラス:
- SubprocessRenderer (subprocess_renderer.py): stdin/stdout 通信
- (将来) PybindRenderer: pybind11 による直接呼び出し

使用例:
    renderer: RendererInterface = SubprocessRenderer()
    renderer.start(RenderConfig(backend=BackendType.CPU))
    renderer.update_scene(scene_json)
    renderer.update_camera(camera_params)
    result = renderer.render_tile(tile_params)
    renderer.stop()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np


class BackendType(Enum):
    """
    利用可能なレンダリングバックエンド。
    
    CPU: OpenMP による並列 CPU レンダリング（現在の実装）
    WEBGPU: Dawn/WebGPU による GPU レンダリング（Phase 2 で実装予定）
    """
    CPU = "cpu"
    WEBGPU = "webgpu"


class AlgorithmType(Enum):
    """
    利用可能なパストレーシングアルゴリズム。
    
    NAIVE: シンプルな BSDF サンプリングのみ
        - 収束が遅い（ノイズが多い）
        - デバッグや参照実装として使用
    
    NEE: Next Event Estimation（直接光推定）
        - 各バウンスで直接ライトをサンプリング
        - 直接光が支配的なシーンで高速収束
        - 推奨デフォルト
    
    MIS: Multiple Importance Sampling（多重重点サンプリング）
        - BSDF サンプリングと光源サンプリングを組み合わせ
        - Power Heuristic で重み付け
        - 最高品質だがやや遅い
    """
    NAIVE = "simple"  # シンプルなBSDFサンプリング（参照用）
    NEE = "nee"       # Next Event Estimation（推奨）
    MIS = "mis"       # Multiple Importance Sampling


@dataclass
class RenderConfig:
    """
    レンダラー初期化設定。
    
    INIT コマンドで送信される設定パラメータです。
    
    Attributes:
        backend: レンダリングバックエンド（CPU/WebGPU）
        algorithm: パストレーシングアルゴリズム
        max_depth: 最大レイバウンス数（デフォルト8）
    """
    backend: BackendType = BackendType.CPU
    algorithm: AlgorithmType = AlgorithmType.NEE
    max_depth: int = 8


@dataclass
class CameraParams:
    """
    カメラパラメータ。
    
    レンダリングに必要なカメラ情報をカプセル化します。
    Blender の camera オブジェクトから抽出した値を格納します。
    
    Attributes:
        pos: カメラ位置 (x, y, z)
        dir: 視線方向 (x, y, z) - 正規化されたベクトル
        up: 上方向 (x, y, z) - 正規化されたベクトル
        fov: 視野角（度）
    
    座標系:
        Blender の右手座標系（Z上、Y奥）をそのまま使用
    """
    pos: tuple  # (x, y, z)
    dir: tuple  # (x, y, z) - direction vector
    up: tuple   # (x, y, z) - up vector
    fov: float  # field of view in degrees
    
    @classmethod
    def from_blender(cls, cam_params: dict) -> 'CameraParams':
        """
        Blender のカメラパラメータ辞書から作成します。
        
        Args:
            cam_params: compute_camera_params() の戻り値
        
        Returns:
            CameraParams インスタンス
        """
        return cls(
            pos=(cam_params['pos'].x, cam_params['pos'].y, cam_params['pos'].z),
            dir=(cam_params['dir'].x, cam_params['dir'].y, cam_params['dir'].z),
            up=(cam_params['up'].x, cam_params['up'].y, cam_params['up'].z),
            fov=cam_params['fov']
        )


@dataclass
class TileParams:
    """
    タイルレンダリングパラメータ。
    
    レンダリングするタイル領域とサンプリング設定を定義します。
    
    Attributes:
        tile_x, tile_y: タイルの左上座標（ピクセル）
        tile_w, tile_h: タイルのサイズ（ピクセル）
        full_w, full_h: 画像全体のサイズ（ピクセル）
        samples: このパスでのサンプル数
        sample_offset: サンプルオフセット（累積レンダリング用）
    
    タイルベースレンダリングについて:
        大きな画像を小さなタイルに分割してレンダリングすることで、
        メモリ使用量を抑え、進捗を表示しやすくなります。
        DIY Renderer では全体を1タイルとして処理しています。
    """
    tile_x: int
    tile_y: int
    tile_w: int
    tile_h: int
    full_w: int
    full_h: int
    samples: int
    sample_offset: int = 0


@dataclass
class RenderResult:
    """
    レンダリング結果。
    
    RENDER_TILE コマンドの結果として返されるピクセルデータです。
    
    Attributes:
        pixels: フラットなピクセルリスト [r,g,b,a, r,g,b,a, ...]
        width: タイル幅
        height: タイル高さ
        samples_rendered: 実際にレンダリングされたサンプル数
        render_time_ms: レンダリング時間（ミリ秒）
    
    色空間:
        リニアRGB（sRGBではない）
        Blender 側でカラーマネジメントを適用
    """
    pixels: List[float]     # Flat list [r,g,b,a, r,g,b,a, ...] 
    width: int
    height: int
    samples_rendered: int
    render_time_ms: float = 0.0
    
    def to_numpy(self) -> np.ndarray:
        """
        NumPy 配列に変換します。
        
        Returns:
            shape=(height, width, 4) の float32 配列
        """
        arr = np.array(self.pixels, dtype=np.float32)
        return arr.reshape((self.height, self.width, 4))


class RendererInterface(ABC):
    """
    Abstract interface for renderer communication.
    
    This interface allows different renderer backends to be used
    interchangeably. Implementations include:
    - SubprocessRenderer: Uses stdin/stdout binary protocol
    - (Future) PybindRenderer: Direct C++ function calls via pybind11
    """
    
    @abstractmethod
    def start(self, config: RenderConfig) -> bool:
        """
        Start the renderer.
        
        Args:
            config: Renderer configuration
            
        Returns:
            True if started successfully
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the renderer is running."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the renderer."""
        pass
    
    @abstractmethod
    def update_scene(self, scene_json: str) -> bool:
        """
        Update the scene data.
        
        Args:
            scene_json: JSON string with scene data
            
        Returns:
            True if scene was loaded successfully
        """
        pass
    
    @abstractmethod
    def update_camera(self, camera: CameraParams) -> bool:
        """
        Update camera parameters.
        
        This is a fast path for camera-only updates (no scene reload).
        
        Args:
            camera: Camera parameters
            
        Returns:
            True if camera was updated successfully
        """
        pass
    
    @abstractmethod
    def render_tile(self, tile: TileParams) -> Optional[RenderResult]:
        """
        Render a tile.
        
        Args:
            tile: Tile parameters
            
        Returns:
            RenderResult with pixel data, or None if failed
        """
        pass
    
    @abstractmethod
    def cancel(self) -> None:
        """Cancel current rendering operation."""
        pass
    
    @abstractmethod
    def set_backend(self, backend: BackendType) -> bool:
        """
        Change rendering backend.
        
        Args:
            backend: New backend to use
            
        Returns:
            True if backend was changed successfully
        """
        pass
    
    @abstractmethod
    def set_algorithm(self, algorithm: AlgorithmType) -> bool:
        """
        Change path tracing algorithm.
        
        Args:
            algorithm: New algorithm to use
            
        Returns:
            True if algorithm was changed successfully
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, List[str]]:
        """
        Query renderer capabilities.
        
        Returns:
            Dict with 'backends' and 'algorithms' lists
        """
        pass


class RendererError(Exception):
    """Exception raised by renderer operations."""
    pass
