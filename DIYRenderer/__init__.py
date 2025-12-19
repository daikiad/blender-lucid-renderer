"""
DIY Renderer - Custom Path Tracing Renderer for Blender
========================================================

このアドオンは、カスタム C++ パストレーサーを Blender のレンダリングシステムに
統合します。Blender シーンを JSON にエクスポートし、pybind11 経由で C++ 
レンダラーを呼び出し、結果を Blender のレンダーウィンドウとビューポートに表示します。

アーキテクチャ (ADR 003: イベント駆動 + 差分更新):
================================================
- 各 RenderEngine インスタンスは独自の RenderSession を持つ
- SceneSync による変更検出（depsgraph.id_type_updated() を活用）
- UpdateFlags による差分更新（GEOMETRY/MATERIALS/LIGHTS）
- シングルトンを使用しない（マルチインスタンス対応）

モジュール構成:
==============
- __init__.py (このファイル): アドオン登録・解除
- engine.py: レンダーエンジン本体 (DIYRenderEngine)
- render_session.py: レンダリングセッション (RenderSession) - インスタンス固有
- scene_sync.py: シーン変更検出 (SceneSync, UpdateFlags)
- viewport.py: ビューポートレンダリング (ViewportRenderer)
- backend.py: C++ レンダラーラッパー (RendererBackend) - 後方互換
- state.py: 状態データ構造 (ViewportState, RenderParams, etc.)
- preferences.py: 設定（AddonPreferences, PropertyGroup）
- panels.py: UI パネル
- scene_export.py: シーン→JSON エクスポート

機能:
====
- プログレッシブレンダリング（サンプル累積）
- ノードベースマテリアル（Principled BSDF, Emission）
- デバッグモード: 法線、アルベド、エミッション
- ビューポートレンダリングと F12 レンダリング
- 非同期ビューポート更新
- マルチインスタンス対応（複数ビューポート、マテリアルプレビュー）

インストール:
============
1. DIYRenderer フォルダを Blender のアドオンディレクトリにコピー
2. Edit > Preferences > Add-ons で "DIY Renderer" を有効化
3. Render Engine を "DIY Render (Minimal)" に変更
4. pybind11 モジュール (diyrenderer) がビルドされていることを確認

Blender アドオン API:
====================
- bl_info: アドオンのメタデータ
- register(): アドオン有効化時に呼ばれる
- unregister(): アドオン無効化時に呼ばれる
"""

bl_info = {
    "name": "DIY Renderer (Minimal Example)",
    "author": "You",
    "version": (0, 0, 2),
    "blender": (4, 5, 0),          # 必要な Blender バージョン
    "location": "Render > Engine",  # メニュー上の位置
    "description": "Minimal renderer with external C++ integration",
    "category": "Render",
}

import bpy

# 各モジュールからクラスをインポート
from .preferences import DIYRendererPreferences, DIYRendererSettings, _get_prefs
from .panels import (
    DIY_RENDER_PT_sampling, 
    DIY_RENDER_PT_light_paths, 
    DIY_RENDER_PT_debug, 
    DIY_RENDER_PT_performance,
    DIY_RENDER_PT_diagnostics,
    DIY_RENDER_PT_diagnostics_results,
    DIY_PT_viewport_path_visualization,
)
from .hover_diagnostics import (
    DIY_OT_pixel_inspector,
    DIY_PT_image_editor_diagnostics,
)
from .engine import DIYRenderEngine


# =============================================================================
# ビューポート更新タイマー
# =============================================================================
# ビューポートレンダリング時、定期的に再描画をトリガーするためのタイマー。
# これがないと、カメラが静止している間にビューポートが更新されない。

_viewport_timer = None           # タイマーハンドル
_viewport_timer_interval = 0.1   # 更新間隔（秒）= 10 FPS


def _viewport_redraw_timer():
    """
    ビューポート再描画タイマーのコールバック。
    
    DIY Renderer が選択されているビューポートを定期的に再描画します。
    これにより、プログレッシブレンダリングの途中結果が表示されます。
    
    Returns:
        float: 次のコールバックまでの秒数（タイマーを継続）
    """
    try:
        scene = bpy.context.scene
        # DIY Renderer が選択されている場合のみ処理
        if scene and scene.render.engine == 'DIY_RENDER_MINIMAL':
            # 全ウィンドウの 3D ビューポートを検索
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            # レンダリングモードの 3D ビューを再描画
                            if space.type == 'VIEW_3D' and space.shading.type == 'RENDERED':
                                area.tag_redraw()
                                break
    except Exception:
        # エラーが発生してもタイマーを継続
        pass
    
    return _viewport_timer_interval  # 次のコールバックまでの秒数


# =============================================================================
# アドオン登録・解除
# =============================================================================

def register():
    """
    アドオンを Blender に登録します。
    
    Edit > Preferences > Add-ons でアドオンを有効化したときに呼ばれます。
    
    処理内容:
    1. カスタムクラスを Blender に登録
       - Preferences: 設定画面
       - PropertyGroup: シーン設定
       - Panels: UI パネル
       - RenderEngine: レンダーエンジン
    2. シーンにプロパティを追加
    3. ビューポート更新タイマーを登録
    4. 標準パネルに互換エンジンとして追加
    """
    global _viewport_timer
    
    # ===== クラス登録 =====
    # 順番が重要: 依存関係のあるクラスは後に登録
    bpy.utils.register_class(DIYRendererPreferences)  # 設定画面
    bpy.utils.register_class(DIYRendererSettings)     # シーン設定
    bpy.utils.register_class(DIY_RENDER_PT_sampling)  # サンプリングパネル
    bpy.utils.register_class(DIY_RENDER_PT_light_paths)  # ライトパスパネル
    bpy.utils.register_class(DIY_RENDER_PT_debug)     # デバッグパネル
    bpy.utils.register_class(DIY_RENDER_PT_performance)  # パフォーマンスパネル
    bpy.utils.register_class(DIY_RENDER_PT_diagnostics)  # 診断パネル
    bpy.utils.register_class(DIY_RENDER_PT_diagnostics_results)  # 診断結果パネル
    bpy.utils.register_class(DIY_PT_viewport_path_visualization)  # 3D Viewport パス可視化パネル
    bpy.utils.register_class(DIY_OT_pixel_inspector)  # ピクセルインスペクター
    bpy.utils.register_class(DIY_PT_image_editor_diagnostics)  # Image Editor パネル
    bpy.utils.register_class(DIYRenderEngine)         # レンダーエンジン本体
    
    # ===== シーンプロパティ追加 =====
    # scene.diy_renderer でアクセス可能に
    bpy.types.Scene.diy_renderer = bpy.props.PointerProperty(type=DIYRendererSettings)
    
    # ===== タイマー登録 =====
    # persistent=True: ファイル読み込み後もタイマーを維持
    if not bpy.app.timers.is_registered(_viewport_redraw_timer):
        bpy.app.timers.register(_viewport_redraw_timer, first_interval=_viewport_timer_interval, persistent=True)
    
    # ===== 標準パネルに互換エンジンとして追加 =====
    # これにより、マテリアル、メッシュ、ライト等の標準パネルが
    # DIY Renderer でも表示されるようになります
    try:
        from bl_ui import (
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        )
        
        modules = [
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        ]
        
        # Cycles 固有のパネルは除外（独自パネルで代替）
        exclude_panels = {
            'CYCLES_RENDER_PT_sampling',
            'CYCLES_RENDER_PT_light_paths',
            'CYCLES_RENDER_PT_performance',
        }
        
        # 各モジュールのパネルに COMPAT_ENGINES を追加
        for module in modules:
            for panel_name in dir(module):
                if panel_name in exclude_panels:
                    continue
                
                panel = getattr(module, panel_name, None)
                if panel and hasattr(panel, 'COMPAT_ENGINES'):
                    panel.COMPAT_ENGINES.add('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not register panels: {e}")


def unregister():
    """
    アドオンを Blender から解除します。
    
    Edit > Preferences > Add-ons でアドオンを無効化したときに呼ばれます。
    register() で行った登録をすべて元に戻します。
    
    処理内容:
    1. タイマーを解除
    2. 標準パネルから互換エンジンを削除
    3. シーンプロパティを削除
    4. カスタムクラスを解除（登録の逆順）
    """
    global _viewport_timer
    
    # ===== タイマー解除 =====
    if bpy.app.timers.is_registered(_viewport_redraw_timer):
        bpy.app.timers.unregister(_viewport_redraw_timer)
    
    # ===== 標準パネルから互換エンジンを削除 =====
    try:
        from bl_ui import (
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        )
        
        modules = [
            properties_material,
            properties_data_mesh,
            properties_data_light,
            properties_data_camera,
            properties_world,
            properties_output,
            properties_data_modifier,
        ]
        
        for module in modules:
            for panel_name in dir(module):
                panel = getattr(module, panel_name, None)
                if panel and hasattr(panel, 'COMPAT_ENGINES') and 'DIY_RENDER_MINIMAL' in panel.COMPAT_ENGINES:
                    panel.COMPAT_ENGINES.discard('DIY_RENDER_MINIMAL')
    except Exception as e:
        print(f"[DIYRenderer] Warning: Could not unregister panels: {e}")
    
    # ===== シーンプロパティ削除 =====
    del bpy.types.Scene.diy_renderer
    
    # ===== クラス解除 =====
    # 登録の逆順で解除（依存関係を壊さないため）
    bpy.utils.unregister_class(DIYRenderEngine)
    bpy.utils.unregister_class(DIY_PT_image_editor_diagnostics)  # Image Editor パネル
    bpy.utils.unregister_class(DIY_OT_pixel_inspector)  # ピクセルインスペクター
    bpy.utils.unregister_class(DIY_PT_viewport_path_visualization)  # 3D Viewport パス可視化パネル
    bpy.utils.unregister_class(DIY_RENDER_PT_diagnostics_results)  # 子パネルは先に解除
    bpy.utils.unregister_class(DIY_RENDER_PT_diagnostics)
    bpy.utils.unregister_class(DIY_RENDER_PT_performance)
    bpy.utils.unregister_class(DIY_RENDER_PT_debug)
    bpy.utils.unregister_class(DIY_RENDER_PT_light_paths)
    bpy.utils.unregister_class(DIY_RENDER_PT_sampling)
    bpy.utils.unregister_class(DIYRendererSettings)
    bpy.utils.unregister_class(DIYRendererPreferences)


# スクリプトとして直接実行された場合
if __name__ == "__main__":
    register()
