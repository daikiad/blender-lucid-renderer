"""
SceneSync - Blender シーン変更の検出と差分更新
=============================================

このモジュールは Blender の depsgraph を活用して、
シーンの変更を効率的に検出し、差分更新を行います。

設計思想（ADR 003 参照）:
- イベント駆動: Blender の depsgraph.id_type_updated() を活用
- 差分更新: 変更種別に応じて最小限の更新を実行
- BVH 再構築の最小化: Geometry 変更時のみ BVH を再構築

主要クラス:
- UpdateFlags: 更新種別のフラグ
- SceneSync: 変更検出と差分同期

使用例:
    sync = SceneSync()
    flags = sync.detect_changes(depsgraph)
    if flags != UpdateFlags.NONE:
        sync.sync(depsgraph, renderer, flags)
"""

from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Optional, Any
import time

if TYPE_CHECKING:
    from .render_session import RenderSession


# =============================================================================
# UpdateFlags - 更新種別フラグ
# =============================================================================

class UpdateFlags(Flag):
    """シーン更新の種別を表すフラグ
    
    Blender の depsgraph から検出された変更種別を表現します。
    複数の変更が同時に発生した場合は OR で結合されます。
    
    Attributes:
        NONE: 変更なし
        GEOMETRY: ジオメトリ変更（BVH 再構築が必要）
        MATERIALS: マテリアル変更（マテリアルデータのみ更新）
        LIGHTS: ライト変更（ライトデータのみ更新）
        CAMERA: カメラ変更（カメラパラメータのみ更新）
        WORLD: ワールド/環境設定の変更
    """
    NONE = 0
    GEOMETRY = auto()   # メッシュ、オブジェクト追加/削除/変形 → BVH 再構築
    MATERIALS = auto()  # マテリアルプロパティ変更 → マテリアルのみ更新
    LIGHTS = auto()     # ライト変更 → ライトデータのみ更新
    CAMERA = auto()     # カメラ移動/回転 → カメラパラメータのみ更新
    WORLD = auto()      # ワールド設定変更
    
    # 複合フラグ（便利なエイリアス）
    FULL_SCENE = GEOMETRY | MATERIALS | LIGHTS | WORLD
    SHADING_ONLY = MATERIALS | LIGHTS | WORLD  # BVH 再構築不要な変更


# =============================================================================
# SceneSync - 変更検出と同期
# =============================================================================

class SceneSync:
    """Blender シーンの変更検出と差分同期
    
    Blender の depsgraph API を活用して、効率的な変更検出を行います。
    検出された変更種別に応じて、最小限の更新を C++ レンダラーに送信します。
    
    属性:
        _last_update_flags: 直前の更新フラグ（デバッグ用）
        _update_count: 更新回数（デバッグ用）
        _first_sync: 初回同期フラグ
    """
    
    def __init__(self):
        """初期化"""
        self._last_update_flags: UpdateFlags = UpdateFlags.NONE
        self._update_count: int = 0
        self._first_sync: bool = True
        self._last_sync_time: float = 0.0
    
    def detect_changes(self, depsgraph: Any) -> UpdateFlags:
        """depsgraph から変更種別を検出
        
        Blender の depsgraph.id_type_updated() を使用して、
        どの種類のデータが更新されたかを効率的に検出します。
        
        Args:
            depsgraph: Blender の依存関係グラフ
            
        Returns:
            UpdateFlags: 検出された変更種別のフラグ
        """
        flags = UpdateFlags.NONE
        
        # 初回は全更新
        if self._first_sync:
            self._first_sync = False
            return UpdateFlags.FULL_SCENE
        
        # --- 高レベル API: id_type_updated() ---
        # これは Blender が推奨する変更検出方法
        
        # メッシュデータの変更 → BVH 再構築
        if depsgraph.id_type_updated('MESH'):
            flags |= UpdateFlags.GEOMETRY
        
        # マテリアルの変更 → マテリアルのみ更新
        if depsgraph.id_type_updated('MATERIAL'):
            flags |= UpdateFlags.MATERIALS
        
        # ライトの変更 → ライトデータのみ更新
        if depsgraph.id_type_updated('LIGHT'):
            flags |= UpdateFlags.LIGHTS
        
        # ワールド/環境の変更
        if depsgraph.id_type_updated('WORLD'):
            flags |= UpdateFlags.WORLD
        
        # ノードツリーの変更（マテリアルノードの編集など）
        if depsgraph.id_type_updated('NODETREE'):
            flags |= UpdateFlags.MATERIALS
        
        # オブジェクトの変更（追加/削除/移動/変形）
        # 注意: id_type_updated('OBJECT') は選択変更でも True を返すので、
        # 詳細な updates をチェックして本当に変更があったかを確認する
        if depsgraph.id_type_updated('OBJECT'):
            for update in depsgraph.updates:
                obj = update.id
                # オブジェクト以外の更新（シーンなど）はスキップ
                if not hasattr(obj, 'type'):
                    continue
                # ジオメトリが変更された場合のみ
                if update.is_updated_geometry and obj.type in {'MESH', 'LIGHT', 'CAMERA'}:
                    print(f"[SceneSync] Geometry updated: {obj.name} ({obj.type})")
                    flags |= UpdateFlags.GEOMETRY
                    break
                # トランスフォームが変更された場合のみ
                if update.is_updated_transform and obj.type in {'MESH', 'LIGHT'}:
                    print(f"[SceneSync] Transform updated: {obj.name} ({obj.type})")
                    flags |= UpdateFlags.GEOMETRY
                    break
            # 選択変更のみの場合は flags は NONE のまま
        
        # カメラの変更
        if depsgraph.id_type_updated('CAMERA'):
            flags |= UpdateFlags.CAMERA
        
        # デバッグログ
        if flags != UpdateFlags.NONE:
            self._log_changes(flags)
        
        self._last_update_flags = flags
        return flags
    
    def needs_bvh_rebuild(self, flags: UpdateFlags) -> bool:
        """BVH 再構築が必要かどうかを判定
        
        Args:
            flags: 変更フラグ
            
        Returns:
            bool: BVH 再構築が必要な場合 True
        """
        return UpdateFlags.GEOMETRY in flags
    
    def needs_render_reset(self, flags: UpdateFlags) -> bool:
        """レンダリングリセット（累積バッファクリア）が必要かどうか
        
        Args:
            flags: 変更フラグ
            
        Returns:
            bool: リセットが必要な場合 True
        """
        # カメラ以外の変更、またはカメラ変更でもリセットが必要
        return flags != UpdateFlags.NONE
    
    def sync(
        self,
        depsgraph: Any,
        session: 'RenderSession',
        flags: UpdateFlags
    ) -> bool:
        """変更種別に応じた差分更新を実行
        
        Args:
            depsgraph: Blender の依存関係グラフ
            session: レンダリングセッション
            flags: 変更フラグ
            
        Returns:
            bool: 同期が成功した場合 True
        """
        if flags == UpdateFlags.NONE:
            return True
        
        start_time = time.perf_counter()
        success = True
        
        # ジオメトリ変更がある場合は全シーン再エクスポート
        if self.needs_bvh_rebuild(flags):
            success = self._sync_full_scene(depsgraph, session)
        else:
            # 差分更新（将来の拡張用）
            # 現時点では C++ 側に差分 API がないため、
            # シェーディング変更でも全シーン再ロードする
            if flags & UpdateFlags.SHADING_ONLY:
                # TODO: Phase 3 で update_materials(), update_lights() を実装
                success = self._sync_full_scene(depsgraph, session)
        
        elapsed = time.perf_counter() - start_time
        self._last_sync_time = elapsed
        self._update_count += 1
        self._first_sync = False
        
        if success:
            print(f"[SceneSync] Sync completed in {elapsed*1000:.1f}ms (flags={flags})")
        
        return success
    
    def _sync_full_scene(
        self,
        depsgraph: Any,
        session: 'RenderSession'
    ) -> bool:
        """全シーンを同期（BVH 再構築を含む）
        
        Args:
            depsgraph: 依存関係グラフ
            session: レンダリングセッション
            
        Returns:
            bool: 成功した場合 True
        """
        from .scene_export import get_scene_cache
        
        cache = get_scene_cache()
        # SceneSync が変更を検出したので、強制的に再エクスポート
        cache.invalidate()
        scene_file = cache.get_or_export(depsgraph, force=True)
        
        if scene_file is None:
            print("[SceneSync] Failed to export scene")
            return False
        
        # セッションのシーンをファイルからロード
        return session.load_scene_file(scene_file)
    
    def _log_changes(self, flags: UpdateFlags) -> None:
        """変更内容をログ出力"""
        parts = []
        if UpdateFlags.GEOMETRY in flags:
            parts.append("GEOMETRY(BVH rebuild)")
        if UpdateFlags.MATERIALS in flags:
            parts.append("MATERIALS")
        if UpdateFlags.LIGHTS in flags:
            parts.append("LIGHTS")
        if UpdateFlags.CAMERA in flags:
            parts.append("CAMERA")
        if UpdateFlags.WORLD in flags:
            parts.append("WORLD")
        
        if parts:
            print(f"[SceneSync] Changes detected: {', '.join(parts)}")
    
    @property
    def last_update_flags(self) -> UpdateFlags:
        """直前の更新フラグ"""
        return self._last_update_flags
    
    @property
    def update_count(self) -> int:
        """更新回数"""
        return self._update_count
    
    def reset(self) -> None:
        """状態をリセット（シーン切り替え時など）"""
        self._first_sync = True
        self._last_update_flags = UpdateFlags.NONE
        self._update_count = 0
