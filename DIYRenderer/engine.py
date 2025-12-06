"""
DIY Render Engine - Blender レンダーエンジン統合
===============================================

このファイルは Blender RenderEngine の薄いラッパーです。
実際のレンダリングロジックは coordinator.py に委譲します。

処理フロー:
1. Blender が render() または view_update()/view_draw() を呼び出す
2. RenderCoordinator に処理を委譲
3. 結果を Blender に返す

主要クラス:
- DIYRenderEngine: bpy.types.RenderEngine のサブクラス
"""

import bpy

from .coordinator import get_coordinator, shutdown_coordinator


class DIYRenderEngine(bpy.types.RenderEngine):
    """
    Blender カスタムレンダーエンジン。
    
    Blender に登録される RenderEngine のサブクラスです。
    すべての処理を RenderCoordinator に委譲します。
    
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
    # F12 レンダリング
    # =========================================================================

    def render(self, depsgraph):
        """
        F12 レンダリングのメインエントリーポイント。
        
        Args:
            depsgraph: Blender の依存関係グラフ
        """
        coordinator = get_coordinator()
        coordinator.render_f12(self, depsgraph)

    # =========================================================================
    # ビューポートレンダリング
    # =========================================================================

    def view_update(self, context, depsgraph):
        """
        ビューポートモードでシーンが変更された時に呼ばれます。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        coordinator = get_coordinator()
        coordinator.on_scene_update(depsgraph)

    def view_draw(self, context, depsgraph):
        """
        ビューポートレンダリングのエントリーポイント。
        
        Args:
            context: Blender コンテキスト
            depsgraph: 依存関係グラフ
        """
        coordinator = get_coordinator()
        coordinator.render_viewport(context, depsgraph)


# =============================================================================
# グローバル関数（後方互換性用）
# =============================================================================

def get_pybind_renderer():
    """pybind11 レンダラーを取得（後方互換性用）"""
    from .backend import get_backend
    backend = get_backend()
    return backend._renderer if backend.is_available else None


def stop_pybind_renderer():
    """pybind11 レンダラーを停止（後方互換性用）"""
    shutdown_coordinator()
