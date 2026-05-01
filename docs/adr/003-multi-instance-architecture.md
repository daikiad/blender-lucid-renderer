# ADR 003: マルチインスタンス対応アーキテクチャ

## ステータス

採用 (Accepted)

## 日付

2025-12-06

---

## 背景

### 問題の発見

現在の DIY Renderer はシングルトンパターンを使用してグローバルに1つの C++ レンダラーインスタンスを共有している。しかし Blender は複数の RenderEngine インスタンスを同時に作成・使用するため、これが根本的な設計上の問題となっている。

### Blender の RenderEngine ライフサイクル

```
ユーザーが Rendered モードに切り替え
    ↓
RenderEngine.__new__() → 新しいインスタンス作成
    ↓
view_update() → シーン情報の更新（depsgraph から変更を検出）
    ↓
view_draw() → 毎フレーム呼ばれる（カメラ移動、タイマーなど）
    ↓
(ユーザーがモード切り替え or ウィンドウ閉じる)
    ↓
RenderEngine.__del__() → インスタンス破棄
```

### 同時に存在する RenderEngine インスタンス

| コンテキスト | 用途 | シーン | 寿命 |
|------------|------|-------|------|
| 3D Viewport A | ビューポートプレビュー | メインシーン | ウィンドウが開いている間 |
| 3D Viewport B | 別ビューポート | メインシーン（別カメラ角度） | ウィンドウが開いている間 |
| F12 Render | 最終レンダリング | メインシーン | レンダリング完了まで |
| Material Preview | マテリアル編集時のプレビュー | **プレビュー専用の小さなシーン** | パネルが開いている間 |

**重要**: マテリアルプレビューは Blender が自動生成する**別のシーン**（球体 + ライト）を使用する。

### Blender の変更検出機構（Depsgraph）

Blender は `depsgraph`（Dependency Graph）を通じて変更検出機能を提供している。これを活用することで、効率的な差分更新が可能になる。

```python
# Blender API: 変更検出
def view_update(self, context, depsgraph):
    # 型別の変更チェック
    if depsgraph.id_type_updated('MESH'):
        # ジオメトリが変更された → BVH 再構築必要
        pass
    
    if depsgraph.id_type_updated('MATERIAL'):
        # マテリアルが変更された → マテリアルのみ更新
        pass
    
    if depsgraph.id_type_updated('LIGHT'):
        # ライトが変更された → ライト情報のみ更新
        pass
    
    if depsgraph.id_type_updated('OBJECT'):
        # オブジェクトの追加/削除/移動
        pass

    # 詳細な変更リスト
    for update in depsgraph.updates:
        print(f"Updated: {update.id.name}, Geometry: {update.is_updated_geometry}")
```

### 現状のアーキテクチャの問題点

```python
# 現在の設計（シングルトン）
_backend_instance: Optional[RendererBackend] = None

def get_backend() -> RendererBackend:
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = RendererBackend()
    return _backend_instance
```

**問題**:
1. **シーンの上書き**: マテリアルプレビューがシーンをロードすると、ビューポートのシーンが消える
2. **キャンセルの干渉**: F12 をキャンセルすると、ビューポートも止まる
3. **状態の競合**: 累積サンプル、カメラ設定などが混在
4. **同時レンダリング不可**: 1つの C++ Renderer では並列レンダリングできない
5. **非効率な更新**: マテリアル変更でも全シーン再エクスポート（BVH 再構築含む）

### C++ 側の現状

```cpp
class PyRenderer {
    Scene scene_;                        // シーンデータ（1つ）
    Camera camera_;                      // カメラ（1つ）
    std::atomic<bool> cancel_requested_; // キャンセルフラグ（1つ）
    std::string algorithm_;
};
```

C++ 側は**インスタンスベース**で設計されているが、Python 側がシングルトンで1つしかインスタンスを作っていない。

---

## 決定すべき事項

1. 複数の RenderEngine インスタンスをどのように分離するか
2. リソース（シーンデータ、BVH）をどの程度共有するか
3. C++ 側の変更をどこまで行うか
4. メモリ効率とパフォーマンスのバランス
5. Blender の変更検出（depsgraph）をどう活用するか

---

## ドメイン分析

### データのカテゴリと更新特性

Blender のシーンデータを BVH への影響で分類すると：

| カテゴリ | 例 | BVH 再構築 | 更新頻度 |
|---------|---|-----------|---------|
| **Geometry** | 頂点、面、メッシュ追加/削除 | **必要** | 低（編集時のみ） |
| **Materials** | Base Color、Roughness、ノード | 不要 | 中（マテリアル編集時） |
| **Lights** | 位置、色、強度 | 不要 | 中 |
| **Camera** | 位置、向き | 不要 | **高**（ナビゲーション毎） |
| **Settings** | サンプル数、バウンス数 | 不要 | 低 |

**重要な洞察**: BVH 構築は最もコストが高い処理。Geometry 変更時のみ BVH を再構築し、Materials/Lights 変更時はデータ更新のみで済む。

### 理想のアーキテクチャ要件

#### 機能要件

1. **完全な分離**: 各レンダリングコンテキストが独立して動作
2. **同時レンダリング**: Viewport + F12 + Material Preview が同時に動く
3. **独立したキャンセル**: 1つをキャンセルしても他に影響しない
4. **マテリアルプレビュー対応**: 異なるシーンを同時にレンダリング
5. **差分更新**: Blender の変更検出を活用した効率的な更新

#### 非機能要件

1. **メモリ効率**: 同じシーンの重複を最小化
2. **CPU効率**: 不要な BVH 再構築を避ける
3. **保守性**: コードがシンプルで理解しやすい
4. **拡張性**: 将来の機能追加（GPU、分散レンダリング等）に対応できる

---

## 設計案

### 案1: Python 側のみの修正（完全インスタンス分離）

**概要**: 各 DIYRenderEngine インスタンスが独自の PyRenderer (C++) を持つ。C++ 側の変更は不要。

```
┌──────────────────────────────────────────────────────────────────┐
│ アーキテクチャ図                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DIYRenderEngine #1 (Viewport)                                   │
│    └─ RenderSession #1                                           │
│         ├─ PyRenderer (C++) #1                                   │
│         │    ├─ Scene (メインシーン)                             │
│         │    ├─ Camera                                           │
│         │    └─ cancel_flag                                      │
│         └─ ViewportState                                         │
│              ├─ accumulated_pixels                               │
│              └─ gl_texture                                       │
│                                                                  │
│  DIYRenderEngine #2 (Material Preview)                           │
│    └─ RenderSession #2                                           │
│         ├─ PyRenderer (C++) #2                                   │
│         │    ├─ Scene (プレビューシーン)  ← 別シーン！            │
│         │    ├─ Camera                                           │
│         │    └─ cancel_flag                                      │
│         └─ ViewportState                                         │
│                                                                  │
│  共有: SceneExporter (JSON 生成のキャッシュのみ)                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**実装イメージ (Python)**:

```python
# render_session.py
class RenderSession:
    """レンダリングセッション - 1つの RenderEngine に対応"""
    
    def __init__(self):
        import diyrenderer
        self._renderer = diyrenderer.Renderer()  # 独自の C++ インスタンス
        self._state = ViewportState()
        self._scene_hash = ""
    
    def load_scene_if_needed(self, depsgraph) -> bool:
        cache = get_scene_cache()
        new_hash, json_str = cache.get_or_export(depsgraph)
        if new_hash != self._scene_hash:
            self._renderer.load_scene_json(json_str)
            self._scene_hash = new_hash
            return True
        return False

# engine.py
class DIYRenderEngine(bpy.types.RenderEngine):
    def _ensure_session(self) -> RenderSession:
        if not hasattr(self, '_session') or self._session is None:
            self._session = RenderSession()
        return self._session
```

**メリット**:
- ✅ 完全な分離、競合なし
- ✅ C++ 側の変更不要
- ✅ 実装がシンプル
- ✅ デバッグが容易

**デメリット**:
- ❌ 同じシーンを複数の C++ インスタンスが保持（メモリ重複）
- ❌ BVH 構築が重複（CPU 時間）

**メモリ見積もり**:
- シーン: 10万三角形 ≈ 50MB (vertices + triangles + BVH)
- インスタンス数: 3 (Viewport + F12 + Preview)
- 追加メモリ: 100-150MB（現代のマシンでは許容範囲）

---

### 案2: C++ 側でシーン共有（SceneManager 導入）

**概要**: シーンデータを共有し、レンダリング状態（カメラ、キャンセル）のみ分離。

```
┌──────────────────────────────────────────────────────────────────┐
│ アーキテクチャ図                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SceneManager (C++, Singleton)                           │    │
│  │   scenes_: map<string, shared_ptr<Scene>>               │    │
│  │     ├─ "main_abc123" → Scene (メインシーン)             │    │
│  │     └─ "preview_xyz" → Scene (プレビューシーン)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│           ↑ 参照                ↑ 参照                          │
│           │                     │                               │
│  ┌────────┴────────┐   ┌───────┴─────────┐                     │
│  │ RenderContext #1│   │ RenderContext #2 │                     │
│  │ (Viewport)      │   │ (Preview)        │                     │
│  │  scene_ref      │   │  scene_ref       │                     │
│  │  camera         │   │  camera          │                     │
│  │  cancel_flag    │   │  cancel_flag     │                     │
│  └─────────────────┘   └──────────────────┘                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**C++ 実装イメージ**:

```cpp
// scene_manager.hpp
class SceneManager {
    std::map<std::string, std::shared_ptr<Scene>> scenes_;
    std::mutex mutex_;
    
public:
    static SceneManager& instance();
    
    std::shared_ptr<Scene> get_or_load(
        const std::string& scene_id, 
        const std::string& json
    );
    
    void release(const std::string& scene_id);
};

// render_context.hpp
struct RenderContext {
    std::shared_ptr<Scene> scene;  // 共有参照
    Camera camera;
    std::atomic<bool> cancel_requested{false};
    std::string algorithm = "nee";
};

// renderer.hpp
class Renderer {
    std::map<int, RenderContext> contexts_;
    
public:
    int create_context();
    void destroy_context(int ctx_id);
    void set_scene(int ctx_id, const std::string& scene_id, const std::string& json);
    void cancel(int ctx_id);
    std::vector<float> render_tile(int ctx_id, ...);
};
```

**メリット**:
- ✅ 同じシーンはメモリ上で1つだけ
- ✅ BVH 構築も1回だけ
- ✅ メモリ効率が最高
- ✅ 将来の拡張に最適

**デメリット**:
- ❌ C++ 側の中規模な変更が必要
- ❌ スレッドセーフ実装が必要
- ❌ シーンのライフサイクル管理が複雑化

---

### 案3: イベント駆動 + 差分更新アーキテクチャ（推奨）

**概要**: Blender の depsgraph を活用し、変更種別に応じた差分更新を行う。

```
┌────────────────────────────────────────────────────────────────────────────┐
│ イベント駆動 + 差分更新アーキテクチャ                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ Python Layer                                                         │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │                                                                      │ │
│  │  RenderEngine (Blender API)                                          │ │
│  │    │                                                                 │ │
│  │    ├─ view_update(depsgraph)                                         │ │
│  │    │    └─ SceneSync.detect_changes(depsgraph) → UpdateFlags         │ │
│  │    │                                                                 │ │
│  │    └─ view_draw(context)                                             │ │
│  │         └─ Session.render(camera, settings)                          │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ SceneSync                                                      │ │ │
│  │  │   - Blender depsgraph の変更を検出                             │ │ │
│  │  │   - 変更種別に応じた差分データをエクスポート                   │ │ │
│  │  │   - UpdateFlags: GEOMETRY | MATERIALS | LIGHTS                 │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ C++ Layer                                                            │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ SceneData (共有可能)                                           │ │ │
│  │  │   ├─ Geometry (BVH 構築済み)    ← GEOMETRY 更新で再構築       │ │ │
│  │  │   ├─ Materials[]               ← MATERIALS 更新で差し替え     │ │ │
│  │  │   └─ Lights[]                  ← LIGHTS 更新で差し替え        │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                       │ │
│  │                              ▼                                       │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ RenderContext (インスタンス固有)                               │ │ │
│  │  │   ├─ scene_ref: shared_ptr<SceneData>                          │ │ │
│  │  │   ├─ camera: Camera                                            │ │ │
│  │  │   ├─ settings: RenderSettings                                  │ │ │
│  │  │   ├─ cancel_flag: atomic<bool>                                 │ │ │
│  │  │   └─ film: vector<float>                                       │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                       │ │
│  │                              ▼                                       │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Integrator (Stateless)                                         │ │ │
│  │  │   - trace_path(scene, context, ray) → color                    │ │ │
│  │  │   - 純粋な計算ロジック                                         │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 差分更新 API

```cpp
// renderer.hpp - 差分更新対応の API
class Renderer {
public:
    // シーンデータの差分更新
    void update_geometry(const std::string& json);   // BVH 再構築
    void update_materials(const std::string& json);  // マテリアルのみ更新
    void update_lights(const std::string& json);     // ライトのみ更新
    
    // レンダリングコンテキスト（インスタンス固有）
    void set_camera(const Camera& camera);
    void cancel();
    
    // レンダリング
    std::vector<float> render_tile(int x, int y, int w, int h);
};
```

#### Python 側の変更検出（SceneSync）

```python
# scene_sync.py
from enum import Flag, auto

class UpdateFlags(Flag):
    NONE = 0
    GEOMETRY = auto()   # BVH 再構築必要
    MATERIALS = auto()  # マテリアルのみ更新
    LIGHTS = auto()     # ライトのみ更新
    CAMERA = auto()     # カメラ更新（RenderContext側）

class SceneSync:
    """Blender depsgraph の変更を検出し、差分更新を行う"""
    
    def detect_changes(self, depsgraph) -> UpdateFlags:
        """depsgraph から変更種別を検出"""
        flags = UpdateFlags.NONE
        
        # Blender の変更検出 API を活用
        if depsgraph.id_type_updated('MESH'):
            flags |= UpdateFlags.GEOMETRY
        
        if depsgraph.id_type_updated('MATERIAL'):
            flags |= UpdateFlags.MATERIALS
        
        if depsgraph.id_type_updated('LIGHT'):
            flags |= UpdateFlags.LIGHTS
        
        if depsgraph.id_type_updated('OBJECT'):
            # オブジェクトの追加/削除/移動
            for update in depsgraph.updates:
                if update.is_updated_geometry:
                    flags |= UpdateFlags.GEOMETRY
                    break
        
        return flags
    
    def sync(self, depsgraph, renderer, flags: UpdateFlags):
        """変更種別に応じた差分更新を実行"""
        if UpdateFlags.GEOMETRY in flags:
            # ジオメトリ変更 → 全シーン再エクスポート（BVH 再構築）
            json_str = self._export_full_scene(depsgraph)
            renderer.load_scene_json(json_str)
        else:
            # 差分更新
            if UpdateFlags.MATERIALS in flags:
                materials_json = self._export_materials(depsgraph)
                renderer.update_materials(materials_json)
            
            if UpdateFlags.LIGHTS in flags:
                lights_json = self._export_lights(depsgraph)
                renderer.update_lights(lights_json)
```

#### 更新フローの例

```
ユーザーがマテリアルの Base Color を変更
    ↓
Blender: view_update() を呼び出し
    ↓
SceneSync.detect_changes(depsgraph)
    → depsgraph.id_type_updated('MATERIAL') = True
    → UpdateFlags.MATERIALS を返す
    ↓
SceneSync.sync(depsgraph, renderer, UpdateFlags.MATERIALS)
    → renderer.update_materials(materials_json)
    → BVH は再構築しない！
    ↓
累積バッファをリセット（新しいマテリアルで再レンダリング）
```

**メリット**:
- ✅ Blender の設計思想に沿った実装
- ✅ BVH 再構築を最小限に抑える
- ✅ マテリアル編集のレスポンスが大幅改善
- ✅ 責務が明確に分離（SceneSync / SceneData / RenderContext / Integrator）
- ✅ テスト容易性

**デメリット**:
- ⚠️ C++ 側に差分更新 API の追加が必要
- ⚠️ 実装コストは案1より高い

---

### 案4: 段階的移行アプローチ

**概要**: 案1から開始し、段階的に案3へ移行する。

```
Phase 1: インスタンス分離（即座に問題解決）
  ├─ 各 RenderEngine が独自の PyRenderer を持つ
  ├─ シングルトン廃止
  └─ 期間: 1-2日

Phase 2: 差分更新の基盤（SceneSync 導入）
  ├─ SceneSync クラスの導入
  ├─ depsgraph.id_type_updated() の活用
  ├─ UpdateFlags による変更種別管理
  └─ 期間: 2-3日

Phase 3: C++ 差分更新 API
  ├─ update_geometry/materials/lights API 追加
  ├─ BVH 再構築の最適化
  └─ 期間: 3-5日

Phase 4: シーン共有（オプション）
  ├─ SceneManager 導入
  ├─ shared_ptr によるシーン共有
  └─ 期間: 3-5日
```

**メリット**:
- ✅ 今すぐ問題を解決できる
- ✅ 将来の最適化パスが明確
- ✅ インターフェースを変えずに内部実装を改善可能
- ✅ 低リスク

**デメリット**:
- ⚠️ Phase 1 ではメモリ効率・更新効率が最適ではない
- ⚠️ 各 Phase への移行コストはゼロではない

---

## 比較表

| 観点 | 案1: Python のみ | 案2: C++ シーン共有 | 案3: イベント駆動+差分更新 | 案4: 段階的移行 |
|------|-----------------|-------------------|------------------------|-----------------|
| **問題解決** | ✅ 完全 | ✅ 完全 | ✅ 完全 | ✅ 完全 |
| **実装コスト** | ✅ 小（1-2日） | ⚠️ 中（3-5日） | ⚠️ 中（1-2週） | ✅ 小→中 |
| **C++ 変更** | 不要 | 必要（中規模） | 必要（差分API追加） | 段階的 |
| **メモリ効率** | ⚠️ 低 | ✅ 高 | ✅ 高 | ⚠️→✅ |
| **更新効率** | ❌ 全再構築 | ❌ 全再構築 | ✅ 差分更新 | ⚠️→✅ |
| **Blender 統合** | ⚠️ 低 | ⚠️ 低 | ✅ 最高（depsgraph活用） | ⚠️→✅ |
| **保守性** | ✅ 高 | ⚠️ 中 | ✅ 高 | ✅ 高 |
| **拡張性** | ⚠️ 中 | ✅ 高 | ✅ 最高 | ✅ 高 |
| **移行リスク** | ✅ 低 | ⚠️ 中 | ⚠️ 中 | ✅ 低 |

---

## 推奨

### 推奨案: **案3（イベント駆動 + 差分更新）** を目標とした **案4（段階的移行）**

**理由**:

1. **Blender の設計思想に沿う**: depsgraph.id_type_updated() を活用した変更検出は Blender の想定する使い方
2. **パフォーマンス最適化**: マテリアル変更時に BVH を再構築しないことで、編集体験が大幅に向上
3. **即時の問題解決**: Phase 1 で現在のマルチインスタンス問題を解決
4. **低リスク**: 段階的に移行することで、各フェーズでの検証が可能
5. **責務の明確化**: SceneSync / SceneData / RenderContext / Integrator の分離で保守性向上

### 実装ロードマップ

```
Phase 1（今回）: Python 側のインスタンス分離
  ├─ RenderSession クラスの導入
  ├─ シングルトン廃止
  ├─ SceneCache は共有のまま
  └─ 期間: 1-2日

Phase 2（次回）: SceneSync + UpdateFlags 導入
  ├─ SceneSync クラス作成
  ├─ depsgraph.id_type_updated() による変更検出
  ├─ Python 側のみで UpdateFlags 管理
  ├─ まだ C++ は全シーン再ロード
  └─ 期間: 2-3日

Phase 3（将来）: C++ 差分更新 API
  ├─ update_geometry() - BVH 再構築
  ├─ update_materials() - マテリアルのみ更新
  ├─ update_lights() - ライトのみ更新
  └─ 期間: 3-5日

Phase 4（オプション）: シーン共有
  ├─ SceneManager 導入
  ├─ 複数ビューポート間でシーン共有
  └─ 期間: 3-5日
```

---

## 決定

**案4（段階的移行）を採用し、案3（イベント駆動 + 差分更新）を最終目標とする**

### 実装計画

#### Phase 1: インスタンス分離

1. **新規ファイル作成**
   - `render_session.py`: レンダリングセッション管理

2. **既存ファイルの修正**
   - `engine.py`: `RenderSession` をインスタンス変数として保持
   - `coordinator.py`: 削除またはマージ
   - `backend.py`: シングルトン関数を削除

3. **維持するもの**
   - `scene_export.py` の `SceneCache`: JSON キャッシュとして共有
   - `state.py`: データ構造
   - `viewport.py`: 描画ロジック

#### Phase 2: SceneSync 導入

1. **新規ファイル作成**
   - `scene_sync.py`: 変更検出と差分更新

2. **実装内容**
   - `UpdateFlags` enum の定義
   - `SceneSync.detect_changes(depsgraph)` の実装
   - `engine.py` の `view_update()` での活用

#### Phase 3: C++ 差分更新 API

1. **C++ 側の変更**
   - `update_materials()` API 追加
   - `update_lights()` API 追加
   - pybind11 バインディング更新

### 成功基準

#### Phase 1
- [x] 複数の 3D ビューポートで同時に Rendered モードが動作
- [x] F12 レンダリング中にビューポートプレビューが動作
- [x] マテリアルプレビューが正しく表示される
- [x] 1つのレンダリングをキャンセルしても他に影響しない

#### Phase 2
- [x] マテリアル変更時に UpdateFlags.MATERIALS が検出される
- [x] ジオメトリ変更時のみ UpdateFlags.GEOMETRY が検出される
- [x] ログで変更種別が確認できる

#### Phase 3
- [ ] マテリアル変更時に BVH 再構築が発生しない
- [ ] マテリアルプレビューのレスポンスが向上

---

## 実装状態 (2025-12-06)

### 完了した実装

#### 新規ファイル

| ファイル | 説明 | 状態 |
|---------|------|------|
| `scene_sync.py` | UpdateFlags enum と SceneSync クラス | ✅ 完了 |
| `render_session.py` | RenderSession クラス（per-instance） | ✅ 完了 |

#### 修正されたファイル

| ファイル | 変更内容 | 状態 |
|---------|---------|------|
| `engine.py` | RenderSession を使用するよう完全書き換え | ✅ 完了 |
| `state.py` | `reset_accumulation()` メソッド追加 | ✅ 完了 |
| `viewport.py` | `backend` → `session` パラメータに変更 | ✅ 完了 |
| `__init__.py` | ドキュメント更新 | ✅ 完了 |
| `coordinator.py` | 非推奨マーク追加 | ✅ 完了 |
| `backend.py` | `get_backend()`, `shutdown_backend()` に非推奨警告 | ✅ 完了 |

### アーキテクチャの変化

**Before (シングルトン)**:
```
DIYRenderEngine (複数) → get_backend() → 1つの RendererBackend → 1つの C++ Renderer
```

**After (インスタンス分離)**:
```
DIYRenderEngine #1 → RenderSession #1 → RendererBackend #1 → C++ Renderer #1
DIYRenderEngine #2 → RenderSession #2 → RendererBackend #2 → C++ Renderer #2
(それぞれ独立)
```

### 次のステップ

1. **テスト**: Blender で複数ビューポート、F12、マテリアルプレビューの動作確認
2. **Phase 3**: C++ 側に `update_materials()`, `update_lights()` API を追加
3. **最適化**: 同じシーンの場合に BVH を共有する仕組み（将来）

---

## 参考資料

- [Blender RenderEngine API](https://docs.blender.org/api/current/bpy.types.RenderEngine.html)
- [Blender Depsgraph API](https://docs.blender.org/api/current/bpy.types.Depsgraph.html)
  - `id_type_updated(id_type)`: 指定した型のデータブロックが更新されたかチェック
  - `updates`: 更新されたデータブロックのイテレータ
- [Cycles Source Code](https://github.com/blender/cycles) - Session, Scene, Integrator の分離
- [PBRT-v4](https://github.com/mmp/pbrt-v4) - 参考アーキテクチャ
