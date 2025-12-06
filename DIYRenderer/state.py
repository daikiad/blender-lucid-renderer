"""
State - データクラスとレンダリング状態の定義
============================================

このモジュールは、レンダリングシステム全体で使用される
データクラスと状態オブジェクトを定義します。

主要クラス:
- CameraParams: カメラパラメータ
- RenderParams: レンダリングパラメータ
- ViewportState: ビューポートの状態
- RenderResult: レンダリング結果

設計方針:
- すべて dataclass で定義し、イミュータブルを推奨
- 型ヒントを完備
- 状態の初期化を明示的に
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import Future


# =============================================================================
# 列挙型
# =============================================================================

class RenderMode(Enum):
    """レンダリングモード"""
    EDITING = auto()      # 編集中（低解像度、高速応答）
    FINAL = auto()        # 最終プレビュー（高解像度、サンプル累積）
    F12 = auto()          # F12 レンダリング


class ChangeType(Enum):
    """シーン変更の種類"""
    NONE = auto()         # 変更なし
    CAMERA_ONLY = auto()  # カメラのみ変更
    CONTENT = auto()      # シーン内容が変更


# =============================================================================
# カメラパラメータ
# =============================================================================

@dataclass(frozen=True)
class CameraParams:
    """カメラパラメータ（イミュータブル）
    
    Attributes:
        pos: カメラ位置 (x, y, z)
        dir: 視線方向 (x, y, z) - 正規化済み
        up: 上方向 (x, y, z) - 正規化済み
        fov: 視野角（度）
    """
    pos: Tuple[float, float, float]
    dir: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（後方互換性用）"""
        from mathutils import Vector
        return {
            'pos': Vector(self.pos),
            'dir': Vector(self.dir),
            'up': Vector(self.up),
            'fov': self.fov
        }


# =============================================================================
# レンダリングパラメータ
# =============================================================================

@dataclass(frozen=True)
class RenderParams:
    """レンダリングパラメータ（イミュータブル）
    
    Attributes:
        width: レンダリング幅
        height: レンダリング高さ
        samples: サンプル数
        max_bounces: 最大バウンス数
        algorithm: アルゴリズム名 ('simple', 'nee', 'mis')
        debug_mode: デバッグモード ('normal', 'albedo', 'emission', None)
        sample_offset: サンプルオフセット（累積用）
    """
    width: int
    height: int
    samples: int = 1
    max_bounces: int = 8
    algorithm: str = 'nee'
    debug_mode: Optional[str] = None
    sample_offset: int = 0


# =============================================================================
# レンダリング結果
# =============================================================================

@dataclass
class RenderResult:
    """レンダリング結果
    
    Attributes:
        pixels: ピクセルデータ (RGBA float配列)
        width: 画像幅
        height: 画像高さ
        samples: サンプル数
        cancelled: キャンセルされたか
        job_id: ジョブID
    """
    pixels: Any  # array.array or list
    width: int
    height: int
    samples: int = 1
    cancelled: bool = False
    job_id: int = 0


# =============================================================================
# ビューポート状態
# =============================================================================

@dataclass
class ViewportState:
    """ビューポートの状態
    
    ビューポートレンダリングに関するすべての状態を一元管理。
    engine.py の散在した hasattr チェックを解消。
    """
    # --- テクスチャ関連 ---
    texture: Any = None  # gpu.types.GPUTexture
    texture_width: int = 0
    texture_height: int = 0
    placeholder_texture: Any = None
    
    # --- サンプル累積 ---
    accumulated_samples: Dict[str, Tuple[Any, int]] = field(default_factory=dict)
    # key: "WxH" 形式, value: (accumulated_array, sample_count)
    
    # --- 時間管理 ---
    last_change_time: float = field(default_factory=time.time)
    last_render_complete_time: float = 0.0
    last_scene_export_time: float = 0.0
    
    # --- 解像度追跡 ---
    last_render_width: int = 0
    last_render_height: int = 0
    
    # --- カメラ追跡 ---
    last_camera_matrix: Any = None  # mathutils.Matrix
    last_view_perspective: Optional[str] = None
    last_view_distance: Optional[float] = None
    
    # --- フラグ ---
    pending_camera_update: bool = False
    scene_update_pending: bool = False
    content_changed: bool = False
    
    # --- 非同期処理 ---
    render_future: Optional[Future] = None
    export_future: Optional[Future] = None
    pending_export_data: Optional[Dict[str, Any]] = None
    
    # --- ジョブ管理 ---
    job_id: int = 0
    
    def reset_for_scene_change(self) -> None:
        """シーン変更時のリセット"""
        self.accumulated_samples = {}
        self.last_change_time = time.time()
        self.pending_camera_update = True
    
    def reset_for_resolution_change(self) -> None:
        """解像度変更時のリセット"""
        self.accumulated_samples = {}
    
    def get_current_sample_count(self, width: int, height: int) -> int:
        """現在のサンプル数を取得"""
        tile_key = f"{width}x{height}"
        if tile_key in self.accumulated_samples:
            return self.accumulated_samples[tile_key][1]
        return 0
    
    def is_rendering(self) -> bool:
        """レンダリング中かどうか"""
        return (self.render_future is not None and 
                not self.render_future.done())
    
    def is_exporting(self) -> bool:
        """エクスポート中かどうか"""
        return (self.export_future is not None and 
                not self.export_future.done())


# =============================================================================
# 定数
# =============================================================================

@dataclass(frozen=True)
class RenderConstants:
    """レンダリング関連の定数"""
    EDITING_TIMEOUT: float = 0.3      # 編集モード→最終モードまでの待機時間
    EXPORT_THROTTLE: float = 0.2      # シーンエクスポートのスロットリング間隔
    RENDER_COOLDOWN: float = 0.05     # レンダリング間のクールダウン
    
    EDITING_BOUNCES: int = 4          # 編集モードのバウンス数
    FINAL_BOUNCES: int = 8            # 最終モードのバウンス数
    
    DEFAULT_EDITING_SCALE: int = 8    # 編集モードの解像度スケール
    DEFAULT_FINAL_SCALE: int = 2      # 最終モードの解像度スケール


# シングルトンインスタンス
RENDER_CONSTANTS = RenderConstants()
