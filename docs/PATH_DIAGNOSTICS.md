# Path Diagnostics 仕様書

## 概要

Path Diagnosticsは、レンダリング中の光路（パス）を記録・分析し、ピクセルごとの寄与を可視化する機能です。
Hover Diagnostics（マウスオーバー診断）により、Image Editor上でピクセルの詳細情報を表示できます。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Blender UI (Python)                          │
│  ┌─────────────────┐  ┌──────────────────────────────────────────┐ │
│  │  hover_diagnostics.py                                         │ │
│  │  - マウス座標取得                                              │ │
│  │  - Y座標反転 (Blender座標 → レンダラー座標)                    │ │
│  │  - 診断データ表示                                              │ │
│  └─────────────────┴──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      C++ Renderer (pybind11)                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  pybind_renderer.hpp                                            ││
│  │  - get_pixel_diagnostic(): ピクセル診断データ取得              ││
│  │  - object_path_string(): パス文字列生成                        ││
│  │  - light_source_name(): 光源タイプ名                           ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  DiagnosticFilm (diagnostic_film.hpp)                           ││
│  │  - ピクセルごとのPathGroupを管理                               ││
│  │  - subsample_factor でメモリ削減                               ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  PathDiagnosticRecorder (diagnostic_integrator.hpp)             ││
│  │  - パストレーシング中に頂点を記録                              ││
│  │  - record_vertex(): 通常頂点                                   ││
│  │  - record_environment_hit(): 環境光ヒット                      ││
│  │  - record_light_hit(): Native Lightヒット                      ││
│  │  - record_emissive_hit(): Emissiveメッシュヒット               ││
│  │  - record_nee_contribution(): NEE寄与（別パスとして記録）      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Path Tracer (path_tracer.hpp)                                  ││
│  │  - traceSimpleWithDiagnostics()                                 ││
│  │  - traceNEEWithDiagnostics()                                    ││
│  │  - traceMISWithDiagnostics()                                    ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## 光源タイプ (LightSourceType)

```cpp
enum class LightSourceType : uint8_t {
    Unknown = 0,      // 光源に到達していない（terminated）
    Point = 1,        // Blender Point Light
    Area = 2,         // Blender Area Light
    Directional = 3,  // Blender Sun Light
    Spot = 4,         // Blender Spot Light
    Environment = 5,  // 環境光（HDRI等）
    Emissive = 6      // Emissiveマテリアルのメッシュ
};
```

### Heckbert記法での表現

| LightSourceType | Heckbert記号 | 説明 |
|-----------------|--------------|------|
| Point           | Lp           | 点光源 |
| Area            | La           | 面光源 |
| Directional     | Ld           | 平行光源（太陽） |
| Spot            | Ls           | スポットライト |
| Environment     | Le           | 環境光 |
| Emissive        | Lo           | 発光オブジェクト |
| Unknown         | L?           | 不明（未到達） |

## パス記録の仕組み

### 1. 通常のBSDF Sampling

```
Camera → Surface1 → Surface2 → Light
         record_vertex  record_vertex  record_light_hit / record_environment_hit / record_emissive_hit
```

- 各サーフェスで `record_vertex()` を呼ぶ
- 光源に到達したら対応する `record_*_hit()` を呼ぶ
- `light_type` が設定される

### 2. NEE (Next Event Estimation)

```
Camera → Surface1 → Surface2 → ...
                    ↓
                    NEE → Light (直接サンプリング)
```

- 各頂点で光源を直接サンプリング
- 成功したら `record_nee_contribution()` で**別パス**として記録
- main pathとは独立してDiagnosticFilmに記録される

### 3. MIS (Multiple Importance Sampling)

NEEとBSDF Samplingを組み合わせ、balance heuristicで重み付け：

```
Each vertex:
  1. NEE: 光源を直接サンプリング → MIS weight適用 → record_nee_contribution()
  2. BSDF: 次の方向をサンプリング → 光源に当たったらMIS weight適用

Final contribution = NEE contribution + BSDF hit contribution
```

## 座標系

### レンダラー座標系
- **Y = 0 が画像の上**
- NDC計算: `ndc_y = 1.0f - 2.0f * (global_y + 0.5f) / full_h`

### Blender Image Editor座標系
- **Y = 0 が画像の下**
- UV座標 (0,0) が左下

### 変換
Python側で変換（hover_diagnostics.py）:
```python
pixel_y = height - 1 - int(view_y * height)
```

## object_path_string の動作

`object_ids` と `light_type` からパス文字列を生成：

| light_type | 表示 | 例 |
|------------|------|-----|
| Environment | "Environment" | Environment → Plane → Camera |
| Point/Area/Directional/Spot | 光源タイプ名 | PointLight → Plane → Camera |
| Emissive | オブジェクト名 | EmissiveSphere → Plane → Camera |
| Unknown | "(terminated)" | (terminated) → Plane → Camera |

### object_id の規則

| object_id | 意味 |
|-----------|------|
| >= 0 | メッシュオブジェクトのインデックス |
| -1 | 環境光（Environment） |
| <= -2 | Native Light: `light_index = -(object_id + 2)` |

## 既知の問題

### 1. MIS contribution の二重カウント問題

**現状の問題**:
- main pathの `contribution` は `sample_radiance` 全体を使用
- `sample_radiance` にはNEE寄与も含まれている
- NEEパスは別途記録されるため、寄与が二重カウントされる可能性

**影響**:
- 診断表示の mean/variance が正確でない可能性
- パスの寄与合計がピクセルの実際の輝度と一致しない可能性

**解決案**:
1. integrator内でNEE寄与とBSDF寄与を分離してトラッキング
2. main pathのcontributionをBSDF sampling で光源に当たった場合のみにする
3. 現在は `light_type == Unknown` のパスをフィルタリングで対応

### 2. terminated パスのフィルタリング

**対応済み**:
```cpp
// Only record if path reached a light source
if (trace.light_type != LightSourceType::Unknown) {
    diagnostic_film_->record_path(px, py, trace);
}
```

光源に到達しなかったパス（Russian Roulette、max depth等で終了）は記録しない。

## 使用方法

### Blender UI

1. レンダリング時に「Diagnostics」を有効化
2. Image Editorで画像を表示
3. `N`キーでサイドパネルを開く
4. 「DIY Render」→「Pixel Inspector」をクリック
5. 画像上でマウスを動かすと診断情報が表示される

### 表示される情報

- **Pixel座標**: (x, y)
- **Sample数**: 記録されたサンプル数
- **Mean RGB**: 平均色
- **Variance**: 分散（ノイズ指標）
- **Top Groups**: 寄与の大きいパスグループ
  - パス表記（Heckbert notation）
  - オブジェクトパス
  - サンプル数
  - Mean / Variance

## ファイル構成

```
DIYRenderer/
├── hover_diagnostics.py          # Hover UI実装
├── diagnostics.py                # 診断マネージャー
└── cpp_renderer/
    └── include/
        ├── pybind_renderer.hpp   # Python binding, get_pixel_diagnostic
        └── diagnostics/
            ├── path_types.hpp           # LightSourceType, PathVertex, PathTrace
            ├── path_stats.hpp           # PathGroup, PathStatistics
            ├── pixel_path_data.hpp      # PixelPathData (Top-N管理)
            ├── diagnostic_film.hpp      # DiagnosticFilm
            └── diagnostic_integrator.hpp # PathDiagnosticRecorder
```

## 今後の改善案

1. **MIS contribution の正確な分離**
   - BSDF寄与とNEE寄与を別々にトラッキング
   - main pathのcontributionを正確にする

2. **パフォーマンス最適化**
   - subsample_factor の調整
   - パスグループのマージ戦略改善

3. **UI改善**
   - variance mapのオーバーレイ表示
   - クリックでパス詳細表示
   - 特定パスのハイライト機能
