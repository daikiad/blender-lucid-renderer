"""
UI Panels for DIY Renderer.
============================

このファイルは Blender のレンダープロパティパネル (Properties > Render) に
表示される UI パネルを定義しています。

各パネルの役割:
- DIY_RENDER_PT_sampling: サンプル数の設定（レンダリング品質）
- DIY_RENDER_PT_light_paths: 最大バウンス数の設定
- DIY_RENDER_PT_debug: デバッグモード選択（ノーマル、アルベドなど）
- DIY_RENDER_PT_performance: バックエンドとサーバーモードの設定

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


class DIY_RENDER_PT_sampling(bpy.types.Panel):
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
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        """
        このパネルを表示するかどうかを判定します。
        DIY Renderer が選択されている場合のみ True を返します。
        
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
        diy = context.scene.diy_renderer
        
        # レンダリングセクション（Cyclesと同様のレイアウト）
        col = layout.column(heading="Render")
        col.prop(diy, "samples", text="Render Samples")
        
        # Viewport section
        col = layout.column(heading="Viewport")
        col.prop(diy, "viewport_samples", text="Viewport Samples")
        col.prop(diy, "viewport_scale_editing", text="Scale (Editing)")
        col.prop(diy, "viewport_scale_final", text="Scale (Final)")
        
        # Algorithm section
        col = layout.column(heading="Algorithm")
        col.prop(diy, "sampling_algorithm", text="Method")


class DIY_RENDER_PT_light_paths(bpy.types.Panel):
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
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        
        col = layout.column(heading="Max Bounces")
        col.prop(diy, "max_bounces", text="Total Max Bounces")


class DIY_RENDER_PT_debug(bpy.types.Panel):
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
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        layout.prop(diy, "debug_mode")


class DIY_RENDER_PT_performance(bpy.types.Panel):
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
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        
        col = layout.column(heading="Backend")
        col.prop(diy, "backend", text="Device")


class DIY_RENDER_PT_diagnostics(bpy.types.Panel):
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
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        diy = context.scene.diy_renderer
        
        # 診断の有効化
        layout.prop(diy, "enable_diagnostics")
        
        if diy.enable_diagnostics:
            # プリセット選択
            layout.prop(diy, "diagnostics_preset")
            
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
            factor = memory_factors.get(diy.diagnostics_preset, 0.4)
            est_memory_mb = (width * height * factor / (1920 * 1080)) * 800
            
            # 情報表示
            box = layout.box()
            box.label(text=f"Resolution: {width}×{height}")
            box.label(text=f"Est. Memory: ~{est_memory_mb:.0f} MB")


class DIY_RENDER_PT_diagnostics_results(bpy.types.Panel):
    """
    診断結果の表示パネル
    
    レンダリング後の診断結果を表示します。
    """
    bl_label = "Analysis Results"
    bl_parent_id = "DIY_RENDER_PT_diagnostics"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'DIY_RENDER_MINIMAL'}

    @classmethod
    def poll(cls, context):
        diy = context.scene.diy_renderer
        return (context.engine in cls.COMPAT_ENGINES and 
                diy.enable_diagnostics)

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

