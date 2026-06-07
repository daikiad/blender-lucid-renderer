"""
Addon preferences and settings.
================================

このモジュールは Lucid Renderer の設定を定義しています。

2種類の設定があります:
1. AddonPreferences（LucidRendererPreferences）
   - グローバル設定（全シーン共通）
   - Edit > Preferences > Add-ons で編集
   - 例: 外部レンダラーのパス、一時ファイルディレクトリ

2. PropertyGroup（LucidRendererSettings）
   - シーン固有の設定
   - Properties > Render パネルで編集
   - 例: サンプル数、バウンス数、デバッグモード

Blender プロパティシステム:
- bpy.props.IntProperty: 整数値
- bpy.props.FloatProperty: 浮動小数点値
- bpy.props.StringProperty: 文字列
- bpy.props.BoolProperty: ブール値
- bpy.props.EnumProperty: 選択肢（ドロップダウン）
"""

import bpy


def _get_prefs_entry():
    """
    アドオン設定エントリを取得します。
    
    Returns:
        アドオンエントリ、または未登録の場合 None
    """
    return bpy.context.preferences.addons.get("LucidRenderer")


def _get_prefs():
    """
    アドオン設定オブジェクトを取得します。
    
    他のモジュールからインポートして使用できます:
        from .preferences import _get_prefs
        prefs = _get_prefs()
        if prefs:
            path = prefs.external_renderer_path
    
    Returns:
        LucidRendererPreferences オブジェクト、または None
    """
    entry = _get_prefs_entry()
    if entry is not None:
        return getattr(entry, 'preferences', None)
    return None


def get_scene_settings():
    """
    現在のシーンの LucidRenderer 設定を取得します。
    
    Returns:
        LucidRendererSettings オブジェクト、または None
    """
    try:
        return bpy.context.scene.lucid_renderer
    except Exception:
        return None


class LucidRendererPreferences(bpy.types.AddonPreferences):
    """
    アドオン全体の設定（グローバル設定）。
    
    Edit > Preferences > Add-ons > Lucid Renderer で編集できます。
    
    Attributes:
        external_renderer_path: C++ レンダラーバイナリ (lucidrt) のパス
        scene_export_directory: シーンエクスポート用の一時ディレクトリ
    """
    # bl_idname はアドオンのパッケージ名と一致する必要がある
    bl_idname = "LucidRenderer"
    
    # 外部レンダラーバイナリのパス
    # subtype='FILE_PATH' でファイル選択ダイアログを表示
    external_renderer_path: bpy.props.StringProperty(
        name="External Renderer Path", 
        description="Path to compiled external C++ renderer binary (lucidrt)", 
        default="", 
        subtype='FILE_PATH'
    )
    
    # シーンエクスポート用一時ディレクトリ
    # subtype='DIR_PATH' でディレクトリ選択ダイアログを表示
    scene_export_directory: bpy.props.StringProperty(
        name="Scene Export Temp Dir", 
        description="Directory to write temporary exported scene files", 
        default="", 
        subtype='DIR_PATH'
    )
    
    def draw(self, context):
        """
        設定パネルの UI を描画します。
        
        Edit > Preferences > Add-ons で表示される設定画面を構築します。
        """
        layout = self.layout
        layout.prop(self, "external_renderer_path")
        layout.prop(self, "scene_export_directory")


class LucidRendererSettings(bpy.types.PropertyGroup):
    """
    シーン固有のレンダリング設定。
    
    各シーンに紐づく設定で、Properties > Render パネルで編集できます。
    scene.lucid_renderer でアクセスできます。
    
    例:
        samples = bpy.context.scene.lucid_renderer.samples
    
    Attributes:
        samples: F12レンダリング時のサンプル数（品質）
        viewport_samples: ビューポートレンダリング時のサンプル数
        max_bounces: 光線の最大バウンス（反射）回数
        sampling_algorithm: パストレーシングアルゴリズム
        debug_mode: デバッグ可視化モード
        backend: レンダリングバックエンド（CPU/WebGPU）
    """
    
    # サンプル数: 多いほどノイズが減るが、時間がかかる
    # モンテカルロ積分の収束速度は √N なので、
    # ノイズを半分にするには4倍のサンプルが必要
    samples: bpy.props.IntProperty(
        name="Samples",
        description="Number of samples for path tracing",
        default=128,
        min=1,
        max=10000
    )
    
    # ビューポート用のサンプル数（インタラクティブ性のため少なめ）
    viewport_samples: bpy.props.IntProperty(
        name="Viewport Samples",
        description="Maximum samples for viewport rendering",
        default=64,
        min=1,
        max=1000
    )
    
    # ビューポート解像度スケール（編集中）
    # シーン編集中は低解像度で高速応答
    viewport_scale_editing: bpy.props.IntProperty(
        name="Editing Scale",
        description="Resolution divisor while editing scene (higher = faster, lower quality)",
        default=8,
        min=1,
        max=16
    )
    
    # ビューポート解像度スケール（最終プレビュー）
    # 編集停止後は高解像度で品質重視
    viewport_scale_final: bpy.props.IntProperty(
        name="Final Scale",
        description="Resolution divisor for final preview (lower = higher quality)",
        default=1,
        min=1,
        max=8
    )
    
    # 最大バウンス数: 光線が何回反射できるか
    # 多いほどリアルな間接光が計算できるが、時間がかかる
    max_bounces: bpy.props.IntProperty(
        name="Max Bounces",
        description="Maximum number of light bounces (ray depth)",
        default=8,
        min=1,
        max=128
    )
    
    # パストレーシングアルゴリズム
    # - simple: 参照実装（デバッグ用、収束が遅い）
    # - nee: Next Event Estimation（直接光を効率的に計算）
    # - mis: Multiple Importance Sampling（最高品質）
    sampling_algorithm: bpy.props.EnumProperty(
        name="Sampling Algorithm",
        description="Path tracing algorithm",
        items=[
            ('simple', "Simple", "BSDF sampling only (slow convergence, good for debugging)"),
            ('nee', "NEE", "Next Event Estimation (fast direct lighting)"),
            ('mis', "MIS", "Multiple Importance Sampling (best quality)"),
        ],
        default='mis'
    )
    
    # デバッグモード: レンダリングの中間結果を可視化
    debug_mode: bpy.props.EnumProperty(
        name="Debug Mode",
        description="Render mode for debugging",
        items=[
            ('NONE', "Path Tracing", "Full path tracing with global illumination"),
            ('normal', "Normals", "Show surface normals as RGB colors"),
            ('albedo', "Albedo", "Show base colors without lighting"),
            ('emission', "Emission", "Show emissive surfaces only"),
            ('volume', "Volumes", "Thickness map of volume-shaded meshes via Beer-Lambert"),
        ],
        default='NONE'
    )
    
    # レンダリングバックエンド
    backend: bpy.props.EnumProperty(
        name="Backend",
        description="Rendering backend",
        items=[
            ('cpu', "CPU", "Multi-threaded CPU rendering (OpenMP)"),
            ('gpu', "GPU", "GPU rendering via Dawn / WebGPU (Phase 1b: normal debug only)"),
        ],
        default='cpu'
    )
    
    # ==========================================================================
    # Path Variance Analyzer (Diagnostics)
    # ==========================================================================
    
    # 診断機能の有効化
    enable_diagnostics: bpy.props.BoolProperty(
        name="Enable Diagnostics",
        description="Enable path variance analysis during rendering",
        default=False
    )
    
    # 診断プリセット（メモリ使用量と精度のトレードオフ）
    diagnostics_preset: bpy.props.EnumProperty(
        name="Preset",
        description="Diagnostics detail level (affects memory usage)",
        items=[
            ('MINIMAL', "Minimal", "Low memory (~200MB at 1080p), basic analysis"),
            ('STANDARD', "Standard", "Balanced memory (~800MB at 1080p), recommended"),
            ('DETAILED', "Detailed", "High memory (~3.2GB at 1080p), full analysis"),
        ],
        default='STANDARD'
    )
    
    # ==========================================================================
    # Path Visualization Settings
    # ==========================================================================
    
    # 最大可視化パス数（2D/3D可視化で同時表示する最大本数）
    max_visualized_paths: bpy.props.IntProperty(
        name="Max Visualized Paths",
        description="Maximum number of paths to visualize simultaneously in 2D/3D view",
        default=20,
        min=1,
        max=100
    )
    
    # パス選択のロック状態（クリックで固定）
    path_selection_locked: bpy.props.BoolProperty(
        name="Lock Selection",
        description="Lock the current pixel selection for path visualization",
        default=False
    )
    
    # ロックされたピクセル座標
    locked_pixel_x: bpy.props.IntProperty(
        name="Locked Pixel X",
        description="X coordinate of locked pixel for path visualization",
        default=0,
        min=0
    )
    
    locked_pixel_y: bpy.props.IntProperty(
        name="Locked Pixel Y",
        description="Y coordinate of locked pixel for path visualization",
        default=0,
        min=0
    )
