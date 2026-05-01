# ADR-002: リファクタリング設計案の比較

**日付**: 2025-12-06  
**ステータス**: 検討中  
**関連**: engine.py のリファクタリング

---

## 背景

`engine.py` が1184行に膨らみ、以下の問題が発生している：
- 責務の混在（ビューポート/F12/状態管理/テクスチャ管理）
- グローバル状態とインスタンス状態の混在
- `hasattr` チェックによる遅延初期化の散在
- F12とビューポートの重複コード

## 設計案

### 案1: レイヤー分離型

```
┌─────────────────────────────────┐
│  LucidRenderEngine (薄いラッパー)  │
├─────────────────────────────────┤
│  RenderCoordinator (状態管理)    │
├─────────────────────────────────┤
│  ViewportRenderer / F12Renderer │
├─────────────────────────────────┤
│  RendererBackend (C++ラッパー)   │
└─────────────────────────────────┘
```

**ファイル構成**:
```
LucidRenderer/
├── engine.py         # 薄いラッパー (50行)
├── coordinator.py    # 状態管理
├── viewport.py       # ビューポート専用
├── f12_renderer.py   # F12専用
├── backend.py        # C++ ラッパー
└── scene_export.py   # シーンエクスポート
```

**メリット**:
- 責務が明確に分離
- 各レイヤーを独立してテスト可能
- 将来の拡張が容易（新しいレンダラー追加など）

**デメリット**:
- ファイル数が増える（5-6ファイル）
- レイヤー間のデータ受け渡しが多い
- 小規模プロジェクトには過剰設計かも

---

### 案2: シンプル2分割型

**ファイル構成**:
```
LucidRenderer/
├── engine.py         # RenderEngine + ビューポート/F12 ロジック
└── backend.py        # C++ ラッパー + シーンエクスポート
```

**コード例**:
```python
# engine.py - すべての Blender 側ロジック
class LucidRenderEngine(bpy.types.RenderEngine):
    def render(self, depsgraph):
        # F12 ロジック直接実装
    
    def view_draw(self, context, depsgraph):
        # ビューポートロジック直接実装
        # ただし、状態管理はデータクラスで整理

@dataclass
class ViewportState:
    texture: Optional[GPUTexture] = None
    accumulated_samples: dict = field(default_factory=dict)
    last_change_time: float = 0.0
    # ...

# backend.py - C++ とのやり取りすべて
class RendererBackend:
    def export_scene(self, depsgraph) -> str: ...
    def render_tile(self, params) -> RenderResult: ...
    def render_tile_async(self, params) -> Future: ...
```

**メリット**:
- ファイル数が最小（2ファイル）
- データの流れがシンプル
- 現状からの変更量が少ない

**デメリット**:
- `engine.py` がまだ大きい（500-600行）
- ビューポートとF12の共通処理が重複する可能性

---

### 案3: 機能モジュール型

**ファイル構成**:
```
LucidRenderer/
├── engine.py              # エントリーポイントのみ
├── rendering/
│   ├── __init__.py
│   ├── viewport.py        # ビューポート専用
│   ├── f12.py             # F12専用
│   ├── accumulator.py     # サンプル累積
│   └── texture_manager.py # GPUテクスチャ管理
├── scene/
│   ├── __init__.py
│   ├── exporter.py        # シーンエクスポート
│   ├── cache.py           # シーンキャッシュ
│   └── change_detector.py # 変更検出
└── backend/
    ├── __init__.py
    └── pybind_wrapper.py  # C++ ラッパー
```

**メリット**:
- 機能ごとに完全分離
- 各機能を独立して理解・テスト可能
- 大規模化しても対応可能

**デメリット**:
- ファイル数が多い（10+ファイル）
- import が複雑になる
- 小規模プロジェクトには過剰

---

### 案4: イベント駆動型

**コード例**:
```python
# engine.py
class LucidRenderEngine(bpy.types.RenderEngine):
    def __init__(self):
        self._event_bus = EventBus()
        self._state = RenderState()
        
        # イベントハンドラを登録
        self._event_bus.on('scene_changed', self._on_scene_changed)
        self._event_bus.on('camera_changed', self._on_camera_changed)
        self._event_bus.on('render_complete', self._on_render_complete)
    
    def view_update(self, context, depsgraph):
        changes = detect_changes(depsgraph)
        for change in changes:
            self._event_bus.emit(change.type, change)
    
    def view_draw(self, context, depsgraph):
        self._event_bus.emit('draw_requested', context)

# events.py
class EventBus:
    def on(self, event: str, handler: Callable): ...
    def emit(self, event: str, data: Any): ...
```

**メリット**:
- 疎結合で拡張しやすい
- 新しい機能（例：プログレスバー更新）を追加しやすい
- デバッグ時にイベントをログできる

**デメリット**:
- 処理フローが追いにくい
- イベントの順序管理が複雑
- Blender の同期的な API と相性が悪い

---

### 案5: 状態マシン型

**コード例**:
```python
# engine.py
class RenderMode(Enum):
    IDLE = auto()
    EDITING = auto()
    FINAL_PREVIEW = auto()
    F12_RENDERING = auto()

class RenderStateMachine:
    def __init__(self):
        self._mode = RenderMode.IDLE
        self._transitions = {
            RenderMode.IDLE: {
                'scene_change': RenderMode.EDITING,
                'f12_start': RenderMode.F12_RENDERING,
            },
            RenderMode.EDITING: {
                'stable_timeout': RenderMode.FINAL_PREVIEW,
                'scene_change': RenderMode.EDITING,  # 自己遷移
            },
            RenderMode.FINAL_PREVIEW: {
                'scene_change': RenderMode.EDITING,
                'samples_complete': RenderMode.IDLE,
            },
        }
    
    def handle_event(self, event: str):
        if event in self._transitions[self._mode]:
            self._mode = self._transitions[self._mode][event]
            self._on_enter(self._mode)
    
    def _on_enter(self, mode: RenderMode):
        # モードごとの初期化処理
        if mode == RenderMode.EDITING:
            self._start_low_res_render()
        elif mode == RenderMode.FINAL_PREVIEW:
            self._start_high_res_render()
```

**状態遷移図**:
```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
    ┌─────────┐  scene  ┌──────────┐  0.3s  ┌─────────┐      │
    │  IDLE   │───────▶│ EDITING  │───────▶│  FINAL  │───────┘
    └─────────┘ change  └──────────┘ stable └─────────┘
                    ▲                          │
                    │      camera change       │
                    └──────────────────────────┘
```

**メリット**:
- モード遷移が明示的で理解しやすい
- バグの原因になりやすい状態管理が整理される
- 現在のモードに基づいた処理が書きやすい

**デメリット**:
- 状態が増えると複雑になる
- 複数の状態が同時に必要な場合に対応しにくい
- 実装がやや冗長

---

### 案6: 現状維持 + リファクタ

**変更点**:
1. `hasattr` チェックを `__init__` で初期化に統一
2. 巨大メソッドを private メソッドに分割
3. 定数を上部にまとめる
4. 状態をデータクラスにまとめる

**コード例**:
```python
# engine.py - 現在の構造を維持しつつ整理
class LucidRenderEngine(bpy.types.RenderEngine):
    EDITING_TIMEOUT = 0.3
    EXPORT_THROTTLE = 0.2
    RENDER_COOLDOWN = 0.05
    
    def __init__(self):
        super().__init__()
        self._state = ViewportState()
        self._backend = RendererBackend()
    
    def _view_draw_pybind(self, context, depsgraph):
        # 現在の350行を以下に分割:
        changes = self._detect_changes(context, depsgraph)  # 50行
        mode = self._determine_mode(changes)                 # 30行
        params = self._compute_params(context, mode)         # 40行
        self._process_pending_results()                      # 50行
        self._maybe_start_render(params)                     # 80行
        self._draw_texture(context)                          # 30行
```

**メリット**:
- 変更量が最小
- 既存の動作を壊すリスクが低い
- 段階的に改善できる

**デメリット**:
- 根本的な問題は残る
- ファイルが大きいまま
- グローバル状態の問題が残る

---

## 比較表

| 案 | ファイル数 | 変更量 | 複雑さ | テスト容易性 | 拡張性 | 推奨シナリオ |
|----|-----------|--------|--------|--------------|--------|--------------|
| 1. レイヤー分離 | 5-6 | 大 | 中 | ◎ | ◎ | 長期開発 |
| 2. シンプル2分割 | 2 | 中 | 低 | ○ | ○ | **バランス重視** |
| 3. 機能モジュール | 10+ | 大 | 高 | ◎ | ◎ | 大規模化予定 |
| 4. イベント駆動 | 3-4 | 大 | 高 | ○ | ◎ | プラグイン的拡張 |
| 5. 状態マシン | 2-3 | 中 | 中 | ◎ | ○ | **状態管理重視** |
| 6. 現状維持+整理 | 1 | 小 | 低 | △ | △ | 最小リスク |

---

## 編集モード vs 最終モードの比較

| 項目 | 編集モード (Editing) | 最終モード (Final) |
|------|---------------------|-------------------|
| トリガー | シーン/カメラ変更 | 0.3秒安定 |
| 解像度スケール | 1/8 (デフォルト) | 1/2 (デフォルト) |
| サンプル数 | 1 | 累積 |
| バウンス数 | 4 | 8 |
| シーンエクスポート | 非同期 | 同期 |
| キャンセル | しない | する |

---

## 推奨

**案2（シンプル2分割）** または **案5（状態マシン）+ 案2の組み合わせ**

理由:
1. 現在の問題の本質は「状態管理の混乱」と「責務の混在」
2. 案2で C++ とのやり取りを分離し、案5でモード管理を明示化
3. ファイル数を抑えつつ、主要な問題を解決

**推奨ファイル構成**:
```
LucidRenderer/
├── engine.py         # RenderEngine + StateMachine
├── backend.py        # C++ ラッパー + シーンエクスポート
├── state.py          # データクラス（ViewportState, RenderParams等）
└── (既存) panels.py, preferences.py
```

---

## 決定

**案1: レイヤー分離型** を採用する。

### 理由
1. 変更量が多くても問題ない
2. 長期的な拡張性を重視
3. 責務の明確な分離によりバグ発生時の原因特定が容易
4. 各コンポーネントの独立したテストが可能

### 採用するファイル構成
```
LucidRenderer/
├── engine.py         # RenderEngine (薄いラッパー, ~50行)
├── coordinator.py    # RenderCoordinator (状態管理の中心)
├── viewport.py       # ViewportRenderer (ビューポート専用ロジック)
├── backend.py        # RendererBackend (C++ ラッパー)
├── state.py          # データクラス (ViewportState, RenderParams等)
├── scene_export.py   # シーンエクスポート (既存を整理)
├── panels.py         # UI (既存)
└── preferences.py    # 設定 (既存)
```

### 削除するファイル
- `protocol.py` - 未使用
- `renderer_interface.py` - 未使用
- `pybind_renderer.py` - 未使用
- `renderer.py` - レガシーモード (pybind11必須化により不要)

---

## 実装計画

### Phase 1: Python リファクタリング

| 順序 | タスク | 依存 | 説明 |
|------|--------|------|------|
| 1 | `state.py` 作成 | なし | データクラスを先に作成 |
| 2 | `backend.py` 作成 | state.py | C++ ラッパーを抽出 |
| 3 | `viewport.py` 作成 | state.py, backend.py | ビューポートレンダリングを抽出 |
| 4 | `coordinator.py` 作成 | state.py, viewport.py, backend.py | 状態管理を抽出 |
| 5 | `engine.py` リファクタ | coordinator.py | 薄いラッパーに書き換え |
| 6 | 不要ファイル削除 | engine.py 完了後 | protocol.py 等を削除 |
| 7 | 動作確認 | 全完了後 | Blenderでテスト |

### Phase 2: C++ リファクタリング (将来)

1. ヘッダ分割 (`math/`, `scene/`, `integrator/`)
2. `Scene` クラスの抽象化
3. `Integrator` インターフェース

---

## 更新履歴

- 2025-12-06: 案1を採用、実装計画を追加
