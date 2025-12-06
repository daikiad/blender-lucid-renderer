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
        col.prop(diy, "viewport_scale_moving", text="Scale (Moving)")
        col.prop(diy, "viewport_scale_static", text="Scale (Static)")
        
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
