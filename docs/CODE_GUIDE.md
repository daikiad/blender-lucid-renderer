# DIY Renderer コードガイド

このドキュメントは、DIY Renderer アドオンの設計思想とコード構造を解説します。

## 目次

1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [Python側の設計](#python側の設計)
3. [C++側の設計](#c側の設計)
4. [データフロー](#データフロー)
5. [主要なアルゴリズム](#主要なアルゴリズム)
6. [セッション管理とマルチインスタンス](#セッション管理とマルチインスタンス)

---

## アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                      Blender                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Viewport   │  │  Material   │  │    F12 Render       │  │
│  │  Preview    │  │  Preview    │  │                     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │             │
│         └────────────────┼────────────────────┘             │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   DIYRenderEngine     │                      │
│              │   (engine.py)         │                      │
│              └───────────┬───────────┘                      │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
              ┌───────────────────────┐
              │    RenderSession      │  ← セッション単位で分離
              │  (render_session.py)  │
              └───────────┬───────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
│ SceneExport │  │  Viewport   │  │   SceneSync     │
│(scene_export│  │  Renderer   │  │ (scene_sync.py) │
│    .py)     │  │(viewport.py)│  │                 │
└──────┬──────┘  └──────┬──────┘  └─────────────────┘
       │                │
       │    JSON        │ レンダリング要求
       ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                   C++ Renderer (pybind11)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PyRenderer  │  │   Scene     │  │   Path Tracer       │  │
│  │ (pybind_    │  │ (renderer.  │  │   (pbr.hpp)         │  │
│  │ renderer.   │  │   hpp)      │  │                     │  │
│  │   hpp)      │  │             │  │ - traceSimple       │  │
│  └─────────────┘  └─────────────┘  │ - traceNEE          │  │
│                                    │ - traceMIS          │  │
│                                    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### コア設計原則

1. **マルチインスタンス対応**: 各 RenderEngine が独立した RenderSession を持つ
2. **非同期レンダリング**: ThreadPoolExecutor でバックグラウンドレンダリング
3. **協調キャンセル**: `std::atomic<bool>` でスレッドセーフなキャンセル
4. **GIL解放**: C++ レンダリング中は Python GIL を解放

---

## Python側の設計

### ファイル構成

```
DIYRenderer/
├── __init__.py          # アドオン登録
├── engine.py            # Blender RenderEngine サブクラス
├── render_session.py    # レンダリングセッション管理
├── viewport.py          # ビューポートレンダリングロジック
├── scene_export.py      # シーンをJSONにエクスポート
├── scene_sync.py        # シーン変更検出
├── state.py             # データクラス定義
├── panels.py            # UIパネル
├── preferences.py       # アドオン設定
└── backend.py           # (レガシー) C++バックエンド
```

### engine.py - RenderEngine

Blender の `bpy.types.RenderEngine` サブクラス。

```python
class DIYRenderEngine(bpy.types.RenderEngine):
    bl_idname = "DIY_RENDER_MINIMAL"
    bl_label = "DIY Renderer (Minimal)"
    bl_use_preview = True           # マテリアルプレビュー対応
    bl_use_shading_nodes = True     # ノードマテリアル対応
```

**主要メソッド:**

| メソッド | 呼び出しタイミング | 処理内容 |
|---------|-------------------|---------|
| `view_update()` | シーン変更時 | 変更フラグを立てる |
| `view_draw()` | ビューポート描画時 | ViewportRenderer に委譲 |
| `render()` | F12 押下時 | フルレンダリング実行 |

**セッション管理:**

```python
def _ensure_session(self) -> RenderSession:
    """各 RenderEngine インスタンスが独自の RenderSession を持つ"""
    if not hasattr(self, '_session') or self._session is None:
        self._session = RenderSession()
    return self._session
```

### render_session.py - RenderSession

各レンダリングコンテキスト（ビューポート、マテリアルプレビュー、F12）に対応するセッション。

```python
class RenderSession:
    _session_counter: int = 0  # セッションID自動採番
    
    def __init__(self):
        RenderSession._session_counter += 1
        self._session_id = RenderSession._session_counter
        
        # C++ レンダラー（セッション固有）
        self._renderer = diyrenderer.Renderer()
        
        # スレッドプール（非同期レンダリング用）
        self._executor = ThreadPoolExecutor(max_workers=1)
```

**重要**: 各セッションは独自の C++ `Renderer` インスタンスを持ち、完全に分離されています。

### viewport.py - ViewportRenderer

ビューポートレンダリングの中核ロジック。

**処理フロー (render メソッド):**

```python
def render(self, context, depsgraph, state, session):
    # 1. セッションIDを取得
    session_id = getattr(session, 'session_id', 0)
    
    # 2. 変更検出
    change_type = self._detect_changes(context, depsgraph, state)
    
    # 3. モード判定（編集中 or 最終プレビュー）
    mode = self._determine_mode(current_time, state, change_type)
    
    # 4. レンダリングパラメータ計算
    params, camera = self._compute_params(context, mode, width, height)
    
    # 5. 結果をポーリング（非同期完了チェック）
    self._poll_results(state, session, current_time)
    
    # 6. 新しいレンダリングを開始
    self._maybe_start_render(...)
    
    # 7. テクスチャを描画
    self._draw_texture(context, state, width, height)
```

**編集モード vs 最終モード:**

| モード | 解像度 | バウンス数 | 用途 |
|-------|--------|-----------|-----|
| 編集中 | 1/8 | 4 | カメラ操作中の高速プレビュー |
| 最終 | 1/2 | 8 | 静止時の高品質プレビュー |

### scene_export.py - シーンエクスポート

Blender シーンを JSON に変換して C++ に渡す。

**主要関数:**

```python
def export_scene_to_file(depsgraph, use_cache=True, session_id=0):
    """メインのエクスポート関数"""
    if use_cache:
        return get_scene_cache(session_id).get_or_export(depsgraph)
    else:
        return export_scene_to_json_for_session(depsgraph, session_id)
```

**エクスポートされるデータ:**

```json
{
  "version": "1.0",
  "meshes": [
    {
      "name": "Cube",
      "vertices": [[x, y, z], ...],
      "triangles": [[i0, i1, i2], ...],
      "triangle_normals": [[[n0], [n1], [n2]], ...],
      "triangle_uvs": [[[u0, v0], [u1, v1], [u2, v2]], ...],
      "smooth": true,
      "material": {
        "name": "Material",
        "use_nodes": true,
        "node_tree": { ... }
      }
    }
  ],
  "native_lights": [
    {
      "name": "Light",
      "type": "AREA",
      "position": [x, y, z],
      "direction": [dx, dy, dz],
      "color": [r, g, b],
      "energy": 1000.0,
      "size_x": 1.0,
      "size_y": 1.0,
      "shape": "SQUARE"
    }
  ],
  "environment": {
    "color": [0.5, 0.5, 0.5],
    "strength": 1.0
  }
}
```

**SceneCache (セッション別キャッシュ):**

```python
class SceneCache:
    _session_caches: dict = {}  # session_id -> SceneCache
    
    @classmethod
    def get_for_session(cls, session_id: int) -> 'SceneCache':
        """セッションID別のキャッシュを取得"""
        if session_id not in cls._session_caches:
            cls._session_caches[session_id] = cls(session_id)
        return cls._session_caches[session_id]
```

これにより、ビューポートとマテリアルプレビューが別々のシーンファイルを使用し、競合を防ぎます。

### state.py - データクラス

不変（イミュータブル）なデータクラスを定義。

```python
@dataclass(frozen=True)
class CameraParams:
    pos: Tuple[float, float, float]
    dir: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float

@dataclass(frozen=True)
class RenderParams:
    width: int
    height: int
    samples: int = 1
    max_bounces: int = 8
    algorithm: str = 'nee'
    debug_mode: Optional[str] = None

@dataclass
class ViewportState:
    texture: Any = None
    accumulated_samples: Dict[str, Tuple[Any, int]] = field(default_factory=dict)
    render_future: Optional[Future] = None
    # ...
```

---

## C++側の設計

### ファイル構成

```
cpp_renderer/
├── include/
│   ├── renderer.hpp       # 基本データ構造（Vec3, Ray, Hit, Scene）
│   ├── pbr.hpp            # パストレーシングアルゴリズム
│   ├── pybind_renderer.hpp # Python バインディング用ラッパー
│   ├── protocol.hpp       # (未使用) サーバー通信用
│   └── json.hpp           # nlohmann/json ライブラリ
├── src/
│   ├── main.cpp           # スタンドアロン実行ファイル
│   ├── pybind_module.cpp  # pybind11 モジュール定義
│   └── node_evaluator.cpp # ノードツリー評価
└── CMakeLists.txt
```

### renderer.hpp - 基本データ構造

**ベクトル演算:**

```cpp
struct Vec3 {
    float x, y, z;
    
    Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    static float dot(const Vec3& a, const Vec3& b);
    static Vec3 cross(const Vec3& a, const Vec3& b);
    void normalize();
};
```

**レイとヒット情報:**

```cpp
struct Ray {
    Vec3 o;  // origin
    Vec3 d;  // direction (normalized)
};

struct Hit {
    bool hit = false;
    float t = 1e30f;        // 距離
    Vec3 point;             // ヒット位置
    Vec3 normal;            // 法線
    Vec3 barycentrics;      // 重心座標
    int meshIndex = -1;
    int triangleIndex = -1;
};
```

**メッシュとシーン:**

```cpp
struct Mesh {
    std::string name;
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    std::vector<TriangleNormals> triangleNormals;  // スムーズシェーディング用
    std::vector<TriangleUVs> triangleUVs;          // UV座標
    Material material;
    BoundingBox bbox;  // BVH用
};

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Light> nativeLights;  // Blender ネイティブライト
    Environment environment;           // 環境光
};
```

**ライトタイプ:**

```cpp
enum class LightType {
    POINT,   // 点光源
    SUN,     // 平行光源
    SPOT,    // スポットライト
    AREA     // 面光源
};

enum class AreaLightShape {
    SQUARE,     // 正方形
    RECTANGLE,  // 長方形
    DISK,       // 円形
    ELLIPSE     // 楕円形
};

struct Light {
    LightType type;
    Vec3 position;
    Vec3 normal;       // 方向（Area/Sun/Spot用）
    Vec3 color;
    Vec3 emission;     // 計算済みの放射輝度
    float energy;      // パワー（ワット）
    float radius;      // 点光源の半径（ソフトシャドウ用）
    float sizeX, sizeY; // 面光源のサイズ
    AreaLightShape shape;
    // ...
};
```

### pbr.hpp - パストレーシング

**3つのアルゴリズム:**

| アルゴリズム | 関数 | 特徴 |
|------------|------|-----|
| Simple | `traceSimple()` | BSDF サンプリングのみ |
| NEE | `traceNEE()` | Next Event Estimation（直接照明）|
| MIS | `traceMIS()` | Multiple Importance Sampling |

**traceNEE の流れ:**

```cpp
Vec3 traceNEE(const Scene& scene, const SceneLights& sceneLights,
              const Ray& ray, int maxDepth) {
    Vec3 result(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray currentRay = ray;
    
    for (int depth = 0; depth < maxDepth; ++depth) {
        // 1. シーンと交差判定
        Hit hit = intersectScene(scene, currentRay, true);
        
        // 2. ネイティブライトとの交差（depth > 0 のみ）
        LightHit lightHit = intersectNativeLights(scene, currentRay, depth);
        
        // 3. ヒットなし → 環境光
        if (!hit.hit) {
            result = result + throughput * getEnvironmentColor(currentRay, scene.environment);
            break;
        }
        
        // 4. マテリアル取得
        Material mat = getMaterialAtHit(scene, hit);
        
        // 5. エミッション追加（発光マテリアル）
        Vec3 emission = getEmission(mat, hit);
        if (emission.length() > 0 && depth == 0) {
            result = result + throughput * emission;
        }
        
        // 6. Next Event Estimation（直接照明）
        if (sceneLights.hasLights()) {
            // ライトをランダム選択
            int lightIdx = sceneLights.selectLight(randf(), lightSelectProb);
            const Light& light = sceneLights.lights[lightIdx];
            
            // ライト上の点をサンプリング
            LightSample ls = sampleLight(light, hit.point, randf(), randf());
            
            // シャドウレイ
            Ray shadowRay{hit.point + ls.direction * 0.001f, ls.direction};
            Hit shadowHit = intersectScene(scene, shadowRay, true);
            bool inShadow = shadowHit.hit && shadowHit.t < ls.distance;
            
            if (!inShadow) {
                // BSDF評価 & 寄与計算
                Vec3 f = evalBSDF(mat, wo, ls.direction, shadingNormal);
                result = result + throughput * f * ls.emission * NdotL / pdf;
            }
        }
        
        // 7. BSDFサンプリング（次のバウンス方向）
        BSDFSample bsdfSample = sampleBSDF(mat, wo, shadingNormal, randf(), randf());
        throughput = throughput * bsdfSample.f * NdotL / bsdfSample.pdf;
        
        // 8. ロシアンルーレット（パス打ち切り）
        if (depth > 3) {
            float q = std::max(0.05f, 1.0f - throughput.length());
            if (randf() < q) break;
            throughput = throughput * (1.0f / (1.0f - q));
        }
        
        currentRay = Ray{hit.point + offset, bsdfSample.wi};
    }
    
    return result;
}
```

**BSDF モデル (Disney-like):**

```cpp
BSDFSample sampleBSDF(const Material& mat, const Vec3& wo,
                      const Vec3& N, float u1, float u2) {
    // 1. Fresnel 項を計算
    float F0 = lerp(0.04f, 1.0f, mat.metallic);
    float F = fresnelSchlick(cosTheta, F0);
    
    // 2. 確率的に Specular か Diffuse を選択
    bool specular = (randf() < F);
    
    if (specular) {
        // GGX マイクロファセットモデル
        Vec3 H = sampleGGX(u1, u2, mat.roughness);
        Vec3 wi = reflect(-wo, H);
        // ...
    } else {
        // コサイン重み付き半球サンプリング
        Vec3 wi = sampleCosineHemisphere(u1, u2);
        // ...
    }
}
```

**ネイティブライトのカメラ不可視化:**

```cpp
inline LightHit intersectNativeLights(const Scene& scene, const Ray& ray, int depth) {
    LightHit result;
    
    // Blender の仕様: ネイティブライトはカメラから直接見えない
    // 反射/屈折を通してのみ見える (depth > 0)
    if (depth == 0) {
        return result;  // 空のヒットを返す
    }
    
    // depth > 0: ライトとの交差判定を実行
    for (const Light& light : scene.nativeLights) {
        // ...
    }
}
```

### pybind_renderer.hpp - Python バインディング

**PyRenderer クラス:**

```cpp
class PyRenderer {
public:
    // キャンセル機構（別スレッドから呼び出し可能）
    std::atomic<bool> cancel_requested_;
    
    void cancel() {
        cancel_requested_.store(true, std::memory_order_relaxed);
    }
    
    // シーン管理
    Scene scene_;
    Camera camera_;
    
    bool load_scene_json(const std::string& json_str) {
        scene_ = loadSceneFromJsonString(json_str);
        return true;
    }
    
    // レンダリング（GIL解放状態で実行）
    std::vector<float> render_tile(
        int tile_x, int tile_y, int tile_w, int tile_h,
        int full_w, int full_h, int samples, int sample_offset, int max_depth
    ) {
        reset_cancel();
        std::vector<float> pixels(tile_w * tile_h * 4, 0.0f);
        
        // マイクロタイル単位でレンダリング（キャンセルチェック頻度を上げる）
        for (int my = 0; my < tile_h; my += 8) {
            for (int mx = 0; mx < tile_w; mx += 8) {
                // キャンセルチェック
                if (cancel_requested_.load()) {
                    return pixels;  // 部分結果を返す
                }
                
                // OpenMP で並列レンダリング
                #pragma omp parallel for collapse(2)
                for (int y = my; y < my + 8; ++y) {
                    for (int x = mx; x < mx + 8; ++x) {
                        // ピクセルごとにサンプリング
                        Vec3 color = traceNEE(scene_, sceneLights, ray, max_depth);
                        // ...
                    }
                }
            }
        }
        return pixels;
    }
};
```

**pybind11 モジュール定義 (pybind_module.cpp):**

```cpp
PYBIND11_MODULE(diyrenderer, m) {
    py::class_<PyRenderer>(m, "Renderer")
        .def(py::init<>())
        .def("cancel", &PyRenderer::cancel)
        .def("load_scene_json", &PyRenderer::load_scene_json)
        .def("set_camera", &PyRenderer::set_camera)
        .def("render_tile", &PyRenderer::render_tile,
             py::call_guard<py::gil_scoped_release>())  // GIL解放
        // ...
}
```

### node_evaluator.cpp - ノードツリー評価

Blender のシェーダーノードグラフを評価してマテリアルプロパティを取得。

**対応ノード:**

| ノードタイプ | 処理内容 |
|------------|---------|
| `ShaderNodeBsdfPrincipled` | Base Color, Metallic, Roughness 等を取得 |
| `ShaderNodeBsdfDiffuse` | Color を取得 |
| `ShaderNodeEmission` | Color × Strength を取得 |
| `ShaderNodeTexImage` | テクスチャサンプリング |
| `ShaderNodeRGB` | 定数カラー |
| `ShaderNodeMix` | 2色の混合 |

**評価フロー:**

```cpp
Vec3 evaluateNode(const NodeTree& tree, const std::string& nodeName,
                  const std::string& socketName, const Vec2& uv) {
    const MaterialNode* node = tree.findNode(nodeName);
    
    if (node->type == "ShaderNodeBsdfPrincipled") {
        if (socketName == "BSDF") {
            // Base Color ソケットを評価
            const NodeSocket* baseColorSocket = node->findInput("Base Color");
            
            if (baseColorSocket->is_linked) {
                // リンク先ノードを再帰評価
                return evaluateNode(tree, baseColorSocket->linked_node,
                                    baseColorSocket->linked_socket, uv);
            } else {
                // デフォルト値を使用
                return baseColorSocket->default_value.v4;  // RGBA
            }
        }
    }
    // ...
}
```

---

## データフロー

### ビューポートレンダリング

```
1. Blender が view_draw() を呼び出し
   │
   ▼
2. ViewportRenderer.render() が処理開始
   │
   ├─→ 変更検出（カメラ/シーン）
   │
   ├─→ シーン変更あり？
   │    │
   │    └─ Yes → 非同期エクスポート開始
   │              export_future = executor.submit(export_scene_to_file)
   │
   ├─→ 前回のレンダリング完了？
   │    │
   │    └─ Yes → テクスチャ更新、サンプル累積
   │
   └─→ 新しいレンダリング開始
        render_future = session.render_tile_async(params)
```

### F12 レンダリング

```
1. Blender が render() を呼び出し
   │
   ▼
2. DIYRenderEngine.render() が処理開始
   │
   ├─→ シーンをエクスポート（session_id 付き）
   │
   ├─→ カメラパラメータを計算
   │
   └─→ 反復レンダリング
        │
        ├─→ 4 samples → プログレス更新
        ├─→ 8 samples → プログレス更新
        ├─→ 16 samples → プログレス更新
        │   ...
        └─→ 128 samples → 完了
```

---

## 主要なアルゴリズム

### ライトの物理単位

Blender のライト energy (ワット) を放射輝度に変換:

| ライトタイプ | 変換式 | 単位 |
|------------|--------|-----|
| Point | `I = Power / (4π)` | W/sr |
| Sun | `E = Power` | W/m² |
| Spot | `I = Power / solidAngle` | W/sr |
| Area | `L = Power / (π × area)` | W/m²/sr |

### 面光源のサンプリング

**矩形 (SQUARE/RECTANGLE):**
```cpp
Vec2 offset((u1 - 0.5f) * sizeX, (u2 - 0.5f) * sizeY);
Vec3 samplePoint = center + right * offset.x + up * offset.y;
```

**円形/楕円 (DISK/ELLIPSE):**
```cpp
// 同心円マッピング（一様分布）
float r = sqrt(u1);
float theta = 2π * u2;
Vec2 offset(r * cos(theta) * radiusX, r * sin(theta) * radiusY);
```

### サンプル累積

```python
# state.accumulated_samples: Dict[str, Tuple[array, count]]
tile_key = f"{width}x{height}"

if tile_key in accumulated_samples:
    acc_array, prev_count = accumulated_samples[tile_key]
    new_count = prev_count + result.samples
    for i in range(len(acc_array)):
        acc_array[i] += result.pixels[i]
    accumulated_samples[tile_key] = (acc_array, new_count)
else:
    accumulated_samples[tile_key] = (array.array('f', result.pixels), result.samples)

# 表示時は平均化
display_pixels = [v / count for v in acc_array]
```

---

## セッション管理とマルチインスタンス

### 問題: シーンの競合

Blender では複数のレンダリングコンテキストが同時に動作:
- ビューポート (3D View)
- マテリアルプレビュー (Properties Panel)
- F12 レンダリング

シングルトンパターンを使うと、これらが同じリソースを共有してしまい、競合が発生。

### 解決策: セッション分離

```
Session #1 (Viewport)
├── _renderer: PyRenderer instance
├── _scene_hash: "abc123..."
└── SceneCache: diy_scene_session_1.json

Session #2 (Material Preview)
├── _renderer: PyRenderer instance
├── _scene_hash: "def456..."
└── SceneCache: diy_scene_session_2.json

Session #3 (F12 Render)
├── _renderer: PyRenderer instance
├── _scene_hash: "ghi789..."
└── SceneCache: diy_scene_session_3.json
```

**実装:**

```python
# RenderSession
class RenderSession:
    _session_counter = 0
    
    def __init__(self):
        RenderSession._session_counter += 1
        self._session_id = RenderSession._session_counter
        self._renderer = diyrenderer.Renderer()  # 独自インスタンス

# SceneCache
class SceneCache:
    _session_caches: dict = {}  # session_id -> SceneCache
    
    @classmethod
    def get_for_session(cls, session_id):
        if session_id not in cls._session_caches:
            cls._session_caches[session_id] = cls(session_id)
        return cls._session_caches[session_id]

# エクスポート時
def export_scene_to_file(depsgraph, session_id=0):
    cache = get_scene_cache(session_id)
    # セッション固有のファイルパス: diy_scene_session_{id}.json
    return cache.get_or_export(depsgraph)
```

### セッションのライフサイクル

```
1. RenderEngine 作成
   └─→ _ensure_session() で RenderSession 作成
       └─→ session_id = 1, 2, 3, ...

2. レンダリング中
   └─→ session_id を使ってリソースを分離
       ├─→ SceneCache (JSON ファイル)
       └─→ C++ Renderer インスタンス

3. RenderEngine 破棄
   └─→ __del__() で session.shutdown()
       └─→ SceneCache.clear_session(session_id)
           └─→ キャッシュファイル削除
```

---

## デバッグのヒント

### ログ出力

Python 側:
```python
print(f"[RenderSession #{self._session_id}] Scene loaded")
print(f"[SceneExport] Exported: {len(meshes)} meshes, {len(lights)} lights")
```

C++ 側:
```cpp
std::cerr << "[PyRenderer] Scene loaded: " << scene_.meshes.size() << " meshes\n";
std::cerr << "[SceneLights] Added " << lights.size() << " lights\n";
```

### よくある問題

| 症状 | 原因 | 解決策 |
|-----|------|-------|
| 真っ暗 | ライトがパースされていない | JSON キー名確認 (`native_lights` vs `lights`) |
| マテリアルが黒 | ノード評価エラー | ノードタイプのログを確認 |
| シーン混在 | セッション分離されていない | session_id が正しく渡されているか確認 |
| クラッシュ | GIL 解放中に Python API 呼び出し | C++ 内で Python コードを呼ばない |

---

## 今後の拡張ポイント

1. **テクスチャサポートの強化**: 現在は基本的な画像テクスチャのみ
2. **BVH 高速化**: 現在は単純な線形探索
3. **GPU レンダリング**: CUDA/Metal 対応
4. **デノイザー**: Intel Open Image Denoise 統合
5. **AOV 出力**: Normal, Albedo, Depth 等の分離出力
