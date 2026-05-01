"""
UI Panels for Lucid Renderer.
============================

このファイルは Blender のレンダープロパティパネル (Properties > Render) に
表示される UI パネルを定義しています。

各パネルの役割:
- LUCID_RENDER_PT_sampling: サンプル数の設定（レンダリング品質）
- LUCID_RENDER_PT_light_paths: 最大バウンス数の設定
- LUCID_RENDER_PT_debug: デバッグモード選択（ノーマル、アルベドなど）
- LUCID_RENDER_PT_performance: バックエンドとサーバーモードの設定

Blender Panel の基本構造:
- bl_label: パネルのタイトル
- bl_space_type: 表示するスペースタイプ（PROPERTIES）
- bl_region_type: 表示するリージョン（WINDOW）
- bl_context: 表示するコンテキスト（render）
- COMPAT_ENGINES: 対応するレンダーエンジン名のセット
- poll(): このパネルを表示するかどうか
- draw(): パネルのUI要素を描画
"""

import bpy


class LUCID_RENDER_PT_sampling(bpy.types.Panel):
    """
    サンプリング設定パネル
    
    パストレーシングのサンプル数を設定します。
    サンプル数が多いほど高品質（ノイズが少ない）になりますが、
    レンダリング時間が長くなります。
    
    設定項目:
    - Render Samples: F12レンダリング時のサンプル数
    - Viewport Samples: ビューポートレンダリング時のサンプル数
    - Algorithm: パストレーシングアルゴリズム（Simple/NEE/MIS）
    """
    bl_label = "Sampling"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'LUCID_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        """
        このパネルを表示するかどうかを判定します。
        Lucid Renderer が選択されている場合のみ True を返します。
        
        Args:
            context: Blender のコンテキスト
        
        Returns:
            bool: パネルを表示する場合 True
        """
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        """
        パネルの UI 要素を描画します。
        
        Blender の UI API を使用してプロパティエディタを構築します。
        use_property_split=True でラベルと値を2列に分けて表示します。
        
        Args:
            context: Blender のコンテキスト
        """
        layout = self.layout
        layout.use_property_split = True    # ラベルと値を2列表示
        layout.use_property_decorate = False  # アニメーションキーフレームボタンを非表示
        
        # シーンに紐づいた設定オブジェクトを取得
        lucid = context.scene.lucid_renderer
        
        # レンダリングセクション（Cyclesと同様のレイアウト）
        col = layout.column(heading="Render")
        col.prop(lucid, "samples", text="Render Samples")
        
        # Viewport section
        col = layout.column(heading="Viewport")
        col.prop(lucid, "viewport_samples", text="Viewport Samples")
        col.prop(lucid, "viewport_scale_editing", text="Scale (Editing)")
        col.prop(lucid, "viewport_scale_final", text="Scale (Final)")
        
        # Algorithm section
        col = layout.column(heading="Algorithm")
        col.prop(lucid, "sampling_algorithm", text="Method")


class LUCID_RENDER_PT_light_paths(bpy.types.Panel):
    """
    ライトパス設定パネル
    
    光線の最大バウンス（反射）回数を設定します。
    バウンス数が多いほど複雑な間接照明を計算できますが、
    計算コストが増加します。
    
    推奨値:
    - 簡単なシーン: 4-6
    - 一般的なシーン: 8（デフォルト）
    - ガラスや鏡が多いシーン: 12-16
    """
    bl_label = "Light Paths"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}  # デフォルトで折りたたんで表示
    COMPAT_ENGINES = {'LUCID_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        lucid = context.scene.lucid_renderer
        
        col = layout.column(heading="Max Bounces")
        col.prop(lucid, "max_bounces", text="Total Max Bounces")


class LUCID_RENDER_PT_debug(bpy.types.Panel):
    """
    デバッグモードパネル
    
    レンダリングの中間結果を可視化するためのデバッグモードを選択します。
    マテリアルやライティングの問題を診断するのに便利です。
    
    モード:
    - Path Tracing: 通常のパストレーシング（完全なレンダリング）
    - Normals: 法線ベクトルをRGBとして表示（XYZ → RGB）
    - Albedo: 基本色（Base Color）のみを表示（ライティングなし）
    - Emission: 発光マテリアルのみを表示
    """
    bl_label = "Debug"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}  # デフォルトで折りたたんで表示
    COMPAT_ENGINES = {'LUCID_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        lucid = context.scene.lucid_renderer
        layout.prop(lucid, "debug_mode")


class LUCID_RENDER_PT_performance(bpy.types.Panel):
    """
    パフォーマンス設定パネル
    
    レンダリングバックエンドとプロセス管理の設定を行います。
    
    設定項目:
    - Backend: レンダリングバックエンド（CPU / 将来的にはWebGPU）
    - Server Mode: サーバーモード（持続プロセス）の有効/無効
    
    サーバーモードについて:
    通常モードでは毎回レンダラープロセスを起動しますが、
    サーバーモードではプロセスを持続させてオーバーヘッドを削減します。
    特にビューポートレンダリングでカメラを動かす場合に効果的です。
    """
    bl_label = "Performance"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}  # デフォルトで折りたたんで表示
    COMPAT_ENGINES = {'LUCID_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        lucid = context.scene.lucid_renderer
        
        col = layout.column(heading="Backend")
        col.prop(lucid, "backend", text="Device")


class LUCID_RENDER_PT_diagnostics(bpy.types.Panel):
    """
    診断パネル - Path Variance Analyzer
    
    レンダリングのノイズ原因を分析するための診断ツールです。
    パスの種類ごとに分散を計算し、どの種類のパスが最もノイズに
    寄与しているかを特定します。
    
    設定項目:
    - Enable Diagnostics: 診断データの収集を有効化
    - Preset: メモリ使用量プリセット（Minimal/Standard/Detailed）
    
    使い方:
    1. 診断を有効化してレンダリング
    2. レンダリング完了後、レポートを確認
    3. 提案されたアクションを実行
    """
    bl_label = "Path Diagnostics"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'LUCID_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        lucid = context.scene.lucid_renderer
        
        # 診断の有効化
        layout.prop(lucid, "enable_diagnostics")
        
        if lucid.enable_diagnostics:
            # プリセット選択
            layout.prop(lucid, "diagnostics_preset")
            
            # メモリ使用量の推定表示
            scene = context.scene
            render = scene.render
            width = int(render.resolution_x * render.resolution_percentage / 100)
            height = int(render.resolution_y * render.resolution_percentage / 100)
            
            # メモリ推定（概算）
            memory_factors = {
                'MINIMAL': 0.1,    # ~200MB at 1080p
                'STANDARD': 0.4,   # ~800MB at 1080p
                'DETAILED': 1.6,   # ~3.2GB at 1080p
            }
            factor = memory_factors.get(lucid.diagnostics_preset, 0.4)
            est_memory_mb = (width * height * factor / (1920 * 1080)) * 800
            
            # 情報表示
            box = layout.box()
            box.label(text=f"Resolution: {width}×{height}")
            box.label(text=f"Est. Memory: ~{est_memory_mb:.0f} MB")


class LUCID_RENDER_PT_diagnostics_results(bpy.types.Panel):
    """
    診断結果の表示パネル
    
    レンダリング後の診断結果を表示します。
    """
    bl_label = "Analysis Results"
    bl_parent_id = "LUCID_RENDER_PT_diagnostics"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'LUCID_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        lucid = context.scene.lucid_renderer
        return (context.engine in cls.COMPAT_ENGINES and 
                lucid.enable_diagnostics)

    def draw(self, context):
        layout = self.layout
        
        # グローバル診断マネージャーから結果を取得（利用可能な場合）
        try:
            from .diagnostics import get_global_diagnostics
            manager = get_global_diagnostics()
            
            if manager is None or not manager.is_available:
                layout.label(text="No diagnostics data available")
                layout.label(text="Run a render to collect data")
                return
            
            report = manager.get_report(top_n=5)
            if report is None:
                layout.label(text="No report available")
                return
            
            # グローバル統計
            stats = report.global_stats
            box = layout.box()
            box.label(text="Global Statistics", icon='INFO')
            col = box.column(align=True)
            col.label(text=f"Total Samples: {stats.total_samples:,}")
            col.label(text=f"Active Pixels: {stats.active_pixels:,}")
            col.label(text=f"Path Groups: {stats.total_groups:,}")
            
            # 分散の高いパスグループ
            if report.top_variance_groups:
                box = layout.box()
                box.label(text="High Variance Paths", icon='ERROR')
                for group in report.top_variance_groups[:3]:
                    row = box.row()
                    row.label(text=f"{group.coarse_type_name}")
                    row.label(text=f"CV={group.coefficient_of_variation:.2f}")
            
            # 改善提案
            if report.suggestions:
                box = layout.box()
                box.label(text="Suggestions", icon='LIGHT')
                for suggestion in report.suggestions[:3]:
                    col = box.column(align=True)
                    col.label(text=suggestion.message)
                    col.label(text=f"  → {suggestion.action}")
                    
        except ImportError:
            layout.label(text="Diagnostics module not available")
        except Exception as e:
            layout.label(text=f"Error: {str(e)[:30]}")


class LUCID_PT_viewport_path_visualization(bpy.types.Panel):
    """
    3D Viewport用のパス可視化パネル
    
    Image Editorでロックしたピクセルのパスを3D Viewportに表示します。
    """
    bl_label = "Lucid Path Visualization"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Lucid'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        # 診断データの確認
        has_data = False
        try:
            from .diagnostics import get_global_diagnostics
            manager = get_global_diagnostics()
            has_data = manager is not None and manager.is_available
        except:
            pass
        
        # hover_diagnostics の状態を確認
        try:
            from .hover_diagnostics import _inspector_state
            is_active = _inspector_state.get('is_active', False)
            is_locked = _inspector_state.get('locked', False)
            locked_x = _inspector_state.get('locked_pixel_x', -1)
            locked_y = _inspector_state.get('locked_pixel_y', -1)
            paths_count = len(_inspector_state.get('paths_data', []))
            selected_count = len(_inspector_state.get('selected_path_indices', set()))
        except:
            is_active = False
            is_locked = False
            locked_x = -1
            locked_y = -1
            paths_count = 0
            selected_count = 0
        
        # ステータス表示
        box = layout.box()
        if not has_data:
            box.label(text="No diagnostic data", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="1. Enable diagnostics")
            col.label(text="2. Run F12 render")
            col.label(text="3. Use Pixel Inspector")
        elif not is_active:
            box.label(text="Inspector not active", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Open Image Editor →")
            col.label(text="Lucid panel →")
            col.label(text="Start Inspection")
        elif not is_locked:
            box.label(text="No pixel locked", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Click on a pixel in")
            col.label(text="Image Editor to lock")
        else:
            box.label(text=f"Locked: ({locked_x}, {locked_y})", icon='LOCKED')
            col = box.column(align=True)
            col.label(text=f"Paths: {paths_count}")
            col.label(text=f"Selected: {selected_count}")
            
            # 全選択/全解除ボタン
            if paths_count > 0:
                row = box.row(align=True)
                row.operator("lucid_render.select_all_paths", text="All", icon='CHECKBOX_HLT')
                row.operator("lucid_render.deselect_all_paths", text="None", icon='CHECKBOX_DEHLT')
            
            # 設定
            lucid = context.scene.lucid_renderer
            col.separator()
            col.prop(lucid, "max_visualized_paths", text="Max Paths")
        
        # 使い方
        if is_active:
            box = layout.box()
            box.label(text="Controls:", icon='HELP')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Click: Lock/unlock pixel")
            col.label(text="Ctrl+Click: Force unlock")
            col.label(text="Select paths in Image Editor")
