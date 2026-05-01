# DIY Renderer 実装ガイド

このドキュメントは、DIY Renderer の内部アーキテクチャと実装の詳細を解説します。
コードリーディングのガイドとしても使えます。

## 目次

1. [システム概要](#システム概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [処理フロー詳細（最初から最後まで）](#処理フロー詳細最初から最後まで)
4. [データフロー](#データフロー)
5. [プロトコル仕様](#プロトコル仕様)
6. [コア実装の解説](#コア実装の解説)
7. [ソースコードの読み方](#ソースコードの読み方)
8. [技術的課題と解決策](#技術的課題と解決策)

---

## システム概要

DIY Renderer は Blender 用のカスタムレンダラーアドオンです。以下の特徴があります：

- **Python + C++ ハイブリッド**: Blender 連携は Python、レンダリングコアは C++
- **サーバーモード**: 持続プロセスで高速な再レンダリング
- **パストレーシング**: Monte Carlo 法による物理ベースレンダリング
- **MIS (Multiple Importance Sampling)**: 効率的なライトサンプリング

### 動作モード

```
┌─────────────────────────────────────────────────────────────────┐
│  Legacy Mode (非推奨)                                            │
│  - 毎レンダリングで新プロセス起動                                   │
│  - JSONファイル経由でシーン受け渡し                                 │
│  - 起動オーバーヘッドが大きい                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Server Mode (推奨) ★現在のデフォルト                             │
│  - 持続プロセスでオーバーヘッド削減                                 │
│  - stdin/stdout バイナリプロトコル                                 │
│  - シーンキャッシュで変更差分のみ更新                               │
│  - プログレッシブレンダリング対応                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## アーキテクチャ

### 全体構成図

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Blender                                    │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │   panels.py │    │  engine.py   │    │   scene_export.py      │ │
│  │   UI/設定   │───▶│ RenderEngine │───▶│  Blender → JSON 変換    │ │
│  │             │    │   統合制御    │    │                        │ │
│  └─────────────┘    └──────┬───────┘    └────────────────────────┘ │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                  subprocess_renderer.py                          ││
│  │              プロトコルクライアント / プロセス管理                   ││
│  └──────────────────────────┬──────────────────────────────────────┘│
└─────────────────────────────│───────────────────────────────────────┘
                              │
                              │  stdin/stdout (Binary Protocol)
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                      C++ Renderer (diyrt)                            │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐│
│  │  server.cpp  │    │renderer.hpp  │    │      pbr.hpp           ││
│  │ プロトコル   │───▶│ シーン/BVH   │───▶│  パストレーシング       ││
│  │ サーバー     │    │ 交差判定     │    │  BSDF/MIS              ││
│  └──────────────┘    └──────────────┘    └────────────────────────┘│
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐                               │
│  │ protocol.hpp │    │node_evaluator│                               │
│  │ ヘッダー定義 │    │ .cpp         │                               │
│  │              │    │ マテリアル   │                               │
│  └──────────────┘    └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### レイヤー構造

```
┌───────────────────────────────────────────┐
│  Layer 4: Blender Integration             │  engine.py, panels.py
│  - RenderEngine 登録                      │
│  - UI パネル                              │
│  - 設定管理                               │
├───────────────────────────────────────────┤
│  Layer 3: Scene Translation               │  scene_export.py
│  - Blender オブジェクト → JSON            │
│  - マテリアル/ノードツリー変換            │
│  - カメラパラメータ変換                   │
├───────────────────────────────────────────┤
│  Layer 2: Communication                   │  subprocess_renderer.py
│  - プロセス管理                           │  protocol.py
│  - バイナリプロトコル                     │  protocol.hpp
│  - エラーハンドリング                     │
├───────────────────────────────────────────┤
│  Layer 1: Rendering Core                  │  renderer.hpp, pbr.hpp
│  - パストレーシング                       │  server.cpp
│  - BVH 交差判定                           │
│  - BSDF 評価                              │
└───────────────────────────────────────────┘
```

---

## 処理フロー詳細（最初から最後まで）

F12 を押してからレンダリング結果が表示されるまでの、完全な処理の流れを追います。

### Step 1: Blender がレンダリングを開始

```
ユーザーが F12 キーを押す
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Blender 内部処理                                                │
│                                                                  │
│  1. レンダリングジョブをキューに追加                              │
│  2. 登録された RenderEngine を検索                               │
│  3. DIYRenderEngine を発見 (bl_idname = "DIY_RENDER_MINIMAL")   │
│  4. depsgraph を評価（シーンの依存関係グラフを構築）             │
│  5. DIYRenderEngine.render(depsgraph) を呼び出し                │
└─────────────────────────────────────────────────────────────────┘
```

### Step 2: engine.py - render() メソッド開始

**ファイル**: `DIYRenderer/engine.py`
**メソッド**: `DIYRenderEngine.render(depsgraph)`

```python
def render(self, depsgraph):
    # =====================================================
    # Phase 2-1: 基本パラメータの取得
    # =====================================================
    scene = depsgraph.scene_eval          # 評価済みシーン
    scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * scale)   # 出力解像度
    height = int(scene.render.resolution_y * scale)
    
    # ユーザー設定を取得
    diy = original_scene.diy_renderer     # カスタムプロパティ
    target_samples = diy.samples          # 目標サンプル数 (e.g., 128)
    use_server = diy.use_server_mode      # サーバーモード有効?
    
    # =====================================================
    # Phase 2-2: カメラパラメータ計算
    # =====================================================
    cam_params = compute_camera_params(scene, width, height)
    # → {'pos': Vector, 'dir': Vector, 'up': Vector, 'fov': float}
```

### Step 3: scene_export.py - シーンをJSONにエクスポート

**ファイル**: `DIYRenderer/scene_export.py`
**関数**: `export_scene_to_file(depsgraph, use_cache=False)`

```python
# engine.py から呼び出し:
scene_file = export_scene_to_file(depsgraph, use_cache=False)

# =====================================================
# 処理内容:
# =====================================================
# 1. 可視オブジェクトを収集
for obj in depsgraph.objects:
    if obj.type == 'MESH' and obj.visible_get():
        # 2. メッシュを三角形化
        mesh = obj.evaluated_get(depsgraph).to_mesh()
        mesh.calc_loop_triangles()
        
        # 3. 頂点・三角形・法線を抽出
        vertices = [[v.co.x, v.co.y, v.co.z] for v in mesh.vertices]
        triangles = [[t.vertices[0], t.vertices[1], t.vertices[2]] for t in mesh.loop_triangles]
        
        # 4. マテリアル情報を抽出
        material = extract_material(obj)  # Base Color, Metallic, etc.
        
        # 5. ノードツリーをシリアライズ
        node_tree = serialize_node_tree(obj.material_slots[0].material.node_tree)

# 6. JSON ファイルに書き出し
# → /tmp/diy_scene_debug.json (844KB 程度)
```

**出力される JSON の構造**:
```json
{
  "meshes": [
    {
      "name": "Cube",
      "vertices": [[x, y, z], ...],
      "triangles": [[i0, i1, i2], ...],
      "triangle_normals": [[[nx, ny, nz], ...], ...],
      "material": {
        "base_color": [r, g, b],
        "metallic": 0.0,
        "roughness": 0.5,
        "emission": [0, 0, 0],
        "transmission": 0.0,
        "ior": 1.45,
        "node_tree": { ... }
      }
    }
  ]
}
```

### Step 4: engine.py - サーバーモードの開始

```python
# =====================================================
# Phase 4-1: サーバー起動判定
# =====================================================
if use_server:
    if not self._ensure_server_started(original_scene):
        # フォールバック: レガシーモードへ
        pass
```

**`_ensure_server_started()` の内部**:

```python
def _ensure_server_started(self, scene) -> bool:
    # 設定を RenderConfig に変換
    config = RenderConfig(
        backend=BackendType.CPU,
        algorithm=AlgorithmType.MIS,  # or NEE, NAIVE
        max_depth=8
    )
    
    # グローバル関数を呼び出し
    return start_server_renderer(config)
```

### Step 5: subprocess_renderer.py - C++ プロセス起動

**ファイル**: `DIYRenderer/subprocess_renderer.py`
**メソッド**: `SubprocessRenderer.start(config)`

```python
def start(self, config: RenderConfig) -> bool:
    # =====================================================
    # Phase 5-1: バイナリを探す
    # =====================================================
    binary = find_renderer_binary()
    # → DIYRenderer/cpp_renderer/build/diyrt
    
    # =====================================================
    # Phase 5-2: サブプロセス起動
    # =====================================================
    self._process = subprocess.Popen(
        [binary, '--server'],           # サーバーモードで起動
        stdin=subprocess.PIPE,          # 入力パイプ
        stdout=subprocess.PIPE,         # 出力パイプ
        stderr=subprocess.PIPE,         # エラー出力パイプ
        bufsize=0                       # バッファなし
    )
    
    # =====================================================
    # Phase 5-3: stderr 読み取りスレッド起動
    # =====================================================
    self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
    self._stderr_thread.start()
    # → C++ の std::cerr 出力を "[C++]" プレフィックスで表示
    
    # =====================================================
    # Phase 5-4: INIT コマンド送信
    # =====================================================
    init_cmd = ProtocolEncoder.encode_init(backend="cpu", algorithm="mis")
    # → バイナリデータ: [Magic][CMD_INIT][PayloadSize][backend_str][algo_str]
    self._send(init_cmd)
    
    # =====================================================
    # Phase 5-5: ACK レスポンス待ち
    # =====================================================
    resp_type, status, _ = self._recv_header()
    # C++ から: [Magic][RESP_ACK][STATUS_OK][0]
    return resp_type == ResponseType.ACK
```

### Step 6: C++ server.cpp - サーバー初期化

**ファイル**: `DIYRenderer/cpp_renderer/src/server.cpp`

```cpp
// main.cpp から:
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--server") {
        RenderServer server;
        server.run();  // ← ここに入る
        return 0;
    }
}

// server.cpp:
void RenderServer::run() {
    log("Entering server loop");
    
    while (true) {
        // =====================================================
        // Phase 6-1: コマンドヘッダー読み取り
        // =====================================================
        protocol::CommandHeader header;
        protocol::CommandHeader::read(std::cin, header);
        // header = {magic: 0x44495952, type: INIT, payload_size: 14}
        
        // =====================================================
        // Phase 6-2: ペイロード読み取り
        // =====================================================
        std::vector<uint8_t> payload;
        readPayload(header.payload_size, payload);
        
        // =====================================================
        // Phase 6-3: コマンドディスパッチ
        // =====================================================
        switch (header.type) {
            case CommandType::INIT:
                handleInit(payload);
                // → backend_ = "cpu", algorithm_ = "mis"
                // → response_->writeAck() で ACK を返す
                break;
            // ...
        }
    }
}
```

### Step 7: engine.py - シーンデータ送信

```python
# engine.py: _render_with_server() から
if not self._update_server_scene(scene_file):
    return None
```

**`_update_server_scene()` の内部**:

```python
def _update_server_scene(self, scene_file: str) -> bool:
    # =====================================================
    # Phase 7-1: JSON ファイル読み込み
    # =====================================================
    with open(scene_file, 'r') as f:
        scene_json = f.read()  # 844KB
    
    # =====================================================
    # Phase 7-2: ハッシュで変更検出
    # =====================================================
    scene_hash = hashlib.md5(scene_json.encode()).hexdigest()
    if scene_hash == _server_scene_hash:
        return True  # 変更なし、スキップ
    
    # =====================================================
    # Phase 7-3: SubprocessRenderer に送信
    # =====================================================
    _server_renderer.update_scene(scene_json)
```

### Step 8: subprocess_renderer.py - シーン送信 (スレッド化)

```python
def update_scene(self, scene_json: str) -> bool:
    scene_bytes = scene_json.encode('utf-8')  # 844KB
    cmd = ProtocolEncoder.encode_update_scene(scene_bytes)
    # cmd = [Header 12bytes] + [JSON 844KB] = 844,248 bytes
    
    # =====================================================
    # Phase 8-1: 別スレッドでチャンク送信
    # =====================================================
    # 理由: パイプバッファ (64KB) より大きいデータを送ると
    #       write() がブロックし、C++ 側の read() もブロックして
    #       デッドロックが発生するため
    
    def send_chunked():
        CHUNK = 65536  # 64KB ずつ
        offset = 0
        while offset < len(cmd):
            chunk = cmd[offset:offset+CHUNK]
            self._process.stdin.write(chunk)
            self._process.stdin.flush()
            offset += len(chunk)
    
    send_thread = threading.Thread(target=send_chunked, daemon=True)
    send_thread.start()
    
    # =====================================================
    # Phase 8-2: 完了待ち + レスポンス受信
    # =====================================================
    send_done.wait(timeout=60.0)
    
    resp_type, status, _ = self._recv_header()
    return resp_type == ResponseType.ACK
```

### Step 9: C++ server.cpp - シーン読み込み

```cpp
void RenderServer::handleUpdateScene(const std::vector<uint8_t>& payload) {
    // payload = 844,236 bytes の JSON 文字列
    
    // =====================================================
    // Phase 9-1: JSON パース
    // =====================================================
    std::string jsonStr(payload.begin(), payload.end());
    scene_ = loadSceneFromJsonString(jsonStr);
    
    // =====================================================
    // Phase 9-2: メッシュごとに処理
    // =====================================================
    for (const auto& meshj : j["meshes"]) {
        Mesh m;
        // 頂点読み込み
        for (const auto& v : meshj["vertices"]) {
            m.vertices.emplace_back(v[0], v[1], v[2]);
        }
        // 三角形読み込み
        for (const auto& t : meshj["triangles"]) {
            Triangle tri;
            tri.i0 = t[0]; tri.i1 = t[1]; tri.i2 = t[2];
            m.triangles.push_back(tri);
        }
        // マテリアル読み込み
        m.material = Material(albedo, metallic, roughness, emission);
        
        // BVH 用のバウンディングボックス計算
        finalizeMeshBounds(m);
        
        scene_.meshes.push_back(std::move(m));
    }
    
    scene_loaded_ = true;
    log("Scene loaded: 10 meshes");
    response_->writeAck();
}
```

### Step 10: カメラ更新

```python
# engine.py: _render_with_server()
camera = RICameraParams(
    pos=(cam_params['pos'].x, cam_params['pos'].y, cam_params['pos'].z),
    dir=(cam_params['dir'].x, cam_params['dir'].y, cam_params['dir'].z),
    up=(cam_params['up'].x, cam_params['up'].y, cam_params['up'].z),
    fov=cam_params['fov']
)
_server_renderer.update_camera(camera)
```

```cpp
// C++ server.cpp:
void RenderServer::handleUpdateCamera(const std::vector<uint8_t>& payload) {
    CameraParams params = CameraParams::fromBytes(payload.data());
    
    camera_.pos = Vec3(params.pos_x, params.pos_y, params.pos_z);
    camera_.dir = Vec3(params.dir_x, params.dir_y, params.dir_z).normalize();
    camera_.up = Vec3(params.up_x, params.up_y, params.up_z).normalize();
    camera_.fovDeg = params.fov;
    camera_.right = Vec3::cross(camera_.dir, camera_.up).normalize();
    
    camera_set_ = true;
    response_->writeAck();
}
```

### Step 11: プログレッシブレンダリングループ

```python
# engine.py: _render_with_server()

# サンプル数の分割: [1, 2, 4, 8, 16, 32, 64, 1] (合計128)
sample_iterations = [1, 2, 4, 8, 16, 32, 64, ...]

for idx, iteration_samples in enumerate(sample_iterations):
    # =====================================================
    # Phase 11-1: キャンセルチェック
    # =====================================================
    if self.test_break():
        _server_renderer.cancel()
        break
    
    # =====================================================
    # Phase 11-2: 進捗更新 (UI に表示)
    # =====================================================
    self.update_progress(total_samples / max_samples)
    self.update_stats("", f"Path Tracing: {total_samples}/{max_samples} samples")
    
    # =====================================================
    # Phase 11-3: タイルレンダリング要求
    # =====================================================
    tile = TileParams(
        tile_x=0, tile_y=0,
        tile_w=width, tile_h=height,
        full_w=width, full_h=height,
        samples=iteration_samples,      # この反復のサンプル数
        sample_offset=total_samples     # 累積サンプル数 (RNG seed 用)
    )
    
    result = _server_renderer.render_tile(tile)
    # result.pixels = [r, g, b, a, r, g, b, a, ...] (width*height*4 floats)
```

### Step 12: C++ パストレーシング実行

```cpp
// server.cpp:
void RenderServer::handleRenderTile(const std::vector<uint8_t>& payload) {
    RenderTileParams params = RenderTileParams::fromBytes(payload.data());
    
    // =====================================================
    // Phase 12-1: 出力バッファ確保
    // =====================================================
    std::vector<float> pixels(params.tile_w * params.tile_h * 4, 0.0f);
    
    // =====================================================
    // Phase 12-2: レンダリング実行
    // =====================================================
    // renderer.hpp の renderRegion() を呼び出し
    renderRegion(
        scene_,                    // シーンデータ
        camera_,                   // カメラ
        pixels.data(),
        params.tile_x, params.tile_y,
        params.tile_w, params.tile_h,
        params.full_w, params.full_h,
        params.samples,            // サンプル数
        params.max_depth,          // 最大バウンス数
        params.sample_offset       // RNG シード用オフセット
    );
    
    // =====================================================
    // Phase 12-3: ピクセルデータを返送
    // =====================================================
    response_->writePixels(pixels.data(), pixels.size() * sizeof(float));
}
```

**renderer.hpp - renderRegion()**:

```cpp
// renderer.hpp:
void renderRegion(...) {
    // OpenMP で並列化
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = tile_y; y < tile_y + tile_h; y++) {
        for (int x = tile_x; x < tile_x + tile_w; x++) {
            // =====================================================
            // Phase 12-2a: 各ピクセルのサンプリング
            // =====================================================
            Vec3 color(0, 0, 0);
            
            for (int s = 0; s < samples; s++) {
                // RNG シード (ピクセル座標 + サンプルインデックス)
                seed_random_xyz(x, y, sample_offset + s);
                
                // カメラからレイを生成
                Ray ray = generateCameraRay(camera, x, y, width, height);
                
                // パストレーシング (pbr.hpp)
                color = color + renderPixelMIS(scene, ray, max_depth);
            }
            
            // 結果を格納 (累積値、まだ平均化しない)
            int idx = ((y - tile_y) * tile_w + (x - tile_x)) * 4;
            pixels[idx + 0] = color.x;
            pixels[idx + 1] = color.y;
            pixels[idx + 2] = color.z;
            pixels[idx + 3] = samples;  // アルファにサンプル数
        }
    }
}
```

**pbr.hpp - renderPixelMIS()**:

```cpp
// pbr.hpp: Multiple Importance Sampling
Vec3 renderPixelMIS(const Scene& scene, Ray ray, int maxDepth) {
    Vec3 throughput(1, 1, 1);  // パスのスループット
    Vec3 radiance(0, 0, 0);    // 累積放射輝度
    
    for (int depth = 0; depth < maxDepth; depth++) {
        // =====================================================
        // レイとシーンの交差判定
        // =====================================================
        HitRecord hit;
        if (!intersectScene(scene, ray, hit)) {
            break;  // 何にも当たらなかった
        }
        
        // =====================================================
        // 自己発光 (光源に直接当たった場合)
        // =====================================================
        radiance = radiance + throughput * hit.material.emission;
        
        // =====================================================
        // 直接光サンプリング (NEE: Next Event Estimation)
        // =====================================================
        Vec3 directLight = sampleDirectLightMIS(scene, hit);
        radiance = radiance + throughput * directLight;
        
        // =====================================================
        // BSDF サンプリング (次のバウンス方向)
        // =====================================================
        Vec3 wi;
        float pdf;
        Vec3 bsdf = sampleBSDF(hit, ray.dir, wi, pdf);
        
        // スループット更新
        throughput = throughput * bsdf * abs(dot(wi, hit.normal)) / pdf;
        
        // =====================================================
        // ロシアンルーレット (パス終了判定)
        // =====================================================
        if (depth > 3) {
            float p = std::min(0.95f, maxComponent(throughput));
            if (randf() > p) break;
            throughput = throughput / p;
        }
        
        // 次のレイを設定
        ray = Ray(hit.point + hit.normal * 0.001f, wi);
    }
    
    return radiance;
}
```

### Step 13: ピクセルデータ受信と累積

```python
# subprocess_renderer.py: render_tile()
resp_type, status, payload_size = self._recv_header()
pixel_data = self._recv(payload_size)  # width*height*4*sizeof(float)
pixels = ProtocolDecoder.decode_pixels(pixel_data, width, height)

return RenderResult(pixels=pixels, ...)
```

```python
# engine.py: _render_with_server()
iteration_pixels = result.pixels

# =====================================================
# Phase 13-1: 累積
# =====================================================
if accumulated_pixels is None:
    accumulated_pixels = array.array('f', iteration_pixels)
    total_samples = iteration_samples
else:
    for i in range(len(accumulated_pixels)):
        accumulated_pixels[i] += iteration_pixels[i]
    total_samples += iteration_samples

# =====================================================
# Phase 13-2: 平均化して表示用ピクセル作成
# =====================================================
inv_samples = 1.0 / total_samples
display_pixels = []
for i in range(0, len(accumulated_pixels), 4):
    display_pixels.append([
        accumulated_pixels[i] * inv_samples,
        accumulated_pixels[i+1] * inv_samples,
        accumulated_pixels[i+2] * inv_samples,
        1.0
    ])
```

### Step 14: Blender に画像を反映

```python
# engine.py: _render_with_server()

# =====================================================
# Phase 14: Blender のレンダー結果に書き込み
# =====================================================
result_obj = self.begin_result(0, 0, width, height)
combined = result_obj.layers[0].passes["Combined"]
combined.rect = display_pixels  # [[r,g,b,a], [r,g,b,a], ...]
self.end_result(result_obj)

# → Blender がレンダービューを更新
# → ユーザーにプログレッシブに画像が表示される
```

### Step 15: レンダリング完了

```python
# すべてのサンプル反復が完了
total_elapsed = time.time() - render_start_time
print(f"[DIYRenderer] Server render complete (128 samples) in 5m 23s")

# シーンファイルをクリーンアップ
os.remove(scene_file)
```

---

## シーケンス図 (全体)

```
┌──────────┐     ┌──────────┐     ┌────────────────┐     ┌─────────────┐
│  User    │     │ Blender  │     │ engine.py      │     │ subprocess_ │
│          │     │          │     │                │     │ renderer.py │
└────┬─────┘     └────┬─────┘     └───────┬────────┘     └──────┬──────┘
     │                │                   │                     │
     │   F12 押下     │                   │                     │
     │───────────────▶│                   │                     │
     │                │                   │                     │
     │                │  render()         │                     │
     │                │──────────────────▶│                     │
     │                │                   │                     │
     │                │                   │  export_scene()     │
     │                │                   │─────────┐           │
     │                │                   │         │           │
     │                │                   │◀────────┘           │
     │                │                   │  (JSON file)        │
     │                │                   │                     │
     │                │                   │  start()            │
     │                │                   │────────────────────▶│
     │                │                   │                     │
```

```
┌────────────────┐                              ┌─────────────┐
│ subprocess_    │                              │ C++ diyrt   │
│ renderer.py    │                              │ (server)    │
└───────┬────────┘                              └──────┬──────┘
        │                                              │
        │  subprocess.Popen([diyrt, --server])         │
        │─────────────────────────────────────────────▶│
        │                                              │
        │  INIT command (stdin)                        │
        │─────────────────────────────────────────────▶│
        │                                              │ handleInit()
        │                                              │────┐
        │                                              │    │
        │                    ACK (stdout)              │◀───┘
        │◀─────────────────────────────────────────────│
        │                                              │
        │  UPDATE_SCENE (844KB JSON)                   │
        │─────────────────────────────────────────────▶│
        │                                              │ handleUpdateScene()
        │                                              │ loadSceneFromJson()
        │                                              │────┐
        │                                              │    │
        │                    ACK                       │◀───┘
        │◀─────────────────────────────────────────────│
        │                                              │
        │  UPDATE_CAMERA (40 bytes)                    │
        │─────────────────────────────────────────────▶│
        │                    ACK                       │
        │◀─────────────────────────────────────────────│
        │                                              │
        │  ┌─── Progressive Render Loop ───────────┐   │
        │  │                                       │   │
        │  │  RENDER_TILE (1 sample)               │   │
        │  │──────────────────────────────────────▶│   │
        │  │                                       │   │ renderRegion()
        │  │                                       │   │────┐
        │  │                 PIXELS                │   │    │
        │  │◀──────────────────────────────────────│◀──┘   │
        │  │                                       │   │
        │  │  RENDER_TILE (2 samples)              │   │
        │  │──────────────────────────────────────▶│   │
        │  │                 PIXELS                │   │
        │  │◀──────────────────────────────────────│   │
        │  │                                       │   │
        │  │  ... (4, 8, 16, 32, 64 samples) ...   │   │
        │  │                                       │   │
        │  └───────────────────────────────────────┘   │
        │                                              │
```

---

## データフロー

### レンダリング全体フロー

```
1. ユーザーが F12 または Render ボタンを押す
                │
                ▼
2. Blender が DIYRenderEngine.render() を呼び出す
                │
                ▼
3. engine.py: _use_server_mode() でモード判定
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
 Server Mode            Legacy Mode
    │                       │
    ▼                       ▼
4. _ensure_server_started()  call_external_renderer()
   サーバープロセス起動       (毎回新プロセス)
                │
                ▼
5. scene_export.py: export_scene_to_file()
   Blender シーン → JSON ファイル
                │
                ▼
6. _update_server_scene()
   - MD5 ハッシュで変更検出
   - 変更があれば UPDATE_SCENE コマンド送信
                │
                ▼
7. _render_with_server()
   - UPDATE_CAMERA コマンド送信
   - プログレッシブループ:
     for samples in [1, 2, 4, 8, 16, 32, 64, ...]:
         RENDER_TILE コマンド送信
         PIXELS レスポンス受信
         Blender に画像反映
                │
                ▼
8. レンダリング完了
```

### プログレッシブレンダリング

```
サンプル数の増加パターン: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 残り

┌────────────────────────────────────────────────────────────┐
│  Sample 1    │  低品質だが即座にプレビュー表示             │
├──────────────┼─────────────────────────────────────────────┤
│  Sample 2    │  ノイズ軽減開始                             │
├──────────────┼─────────────────────────────────────────────┤
│  Sample 4    │  形状がより明確に                           │
├──────────────┼─────────────────────────────────────────────┤
│  ...         │  段階的に品質向上                           │
├──────────────┼─────────────────────────────────────────────┤
│  Sample 128  │  最終品質（設定値）                         │
└──────────────┴─────────────────────────────────────────────┘

利点:
- ユーザーは早い段階で構図を確認できる
- 問題があれば早期にキャンセル可能
- CPU 時間の効率的な活用
```

---

## プロトコル仕様

### バイナリプロトコル概要

Python と C++ 間の通信には、カスタムバイナリプロトコルを使用します。
テキストベース (JSON over stdio) より効率的で、ピクセルデータの転送に適しています。

### コマンドフォーマット

```
┌─────────────────────────────────────────────────────────────┐
│                   Command Header (12 bytes)                  │
├─────────────┬─────────────┬─────────────────────────────────┤
│   Magic     │  Command    │       Payload Size              │
│  (4 bytes)  │  (4 bytes)  │       (4 bytes)                 │
│  0x44495952 │  enum       │       uint32_t                  │
│  "DIYR"     │             │                                 │
├─────────────┴─────────────┴─────────────────────────────────┤
│                   Payload (variable length)                  │
└─────────────────────────────────────────────────────────────┘
```

### コマンド一覧

| Code | Command       | Payload                                       |
|------|---------------|-----------------------------------------------|
| 0x01 | INIT          | backend 文字列 + algorithm 文字列             |
| 0x02 | UPDATE_SCENE  | JSON シーンデータ (UTF-8)                     |
| 0x03 | UPDATE_CAMERA | 40 bytes (10 floats: pos, dir, up, fov)       |
| 0x04 | RENDER_TILE   | 36 bytes (9 ints: tile params)                |
| 0x05 | CANCEL        | なし                                          |
| 0xFF | SHUTDOWN      | なし                                          |

### レスポンスフォーマット

```
┌─────────────────────────────────────────────────────────────┐
│                  Response Header (16 bytes)                  │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│   Magic     │    Type     │   Status    │  Payload Size     │
│  (4 bytes)  │  (4 bytes)  │  (4 bytes)  │  (4 bytes)        │
│  0x44495952 │  enum       │  enum       │  uint32_t         │
├─────────────┴─────────────┴─────────────┴───────────────────┤
│                   Payload (variable length)                  │
└─────────────────────────────────────────────────────────────┘
```

### レスポンス一覧

| Code | Type         | Payload                                        |
|------|--------------|------------------------------------------------|
| 0x81 | ACK          | なし (成功応答)                                |
| 0x82 | PIXELS       | float32 配列 (width × height × 4 RGBA)         |
| 0x83 | PROGRESS     | 進捗情報                                       |
| 0x85 | ERROR        | エラーメッセージ (UTF-8)                       |

### カメラパラメータ構造 (40 bytes)

```c
struct CameraParams {
    float pos_x, pos_y, pos_z;    // カメラ位置 (12 bytes)
    float dir_x, dir_y, dir_z;    // 視線方向 (12 bytes)
    float up_x, up_y, up_z;       // 上方向 (12 bytes)
    float fov;                     // 視野角 (4 bytes)
};
```

### タイルパラメータ構造 (36 bytes)

```c
struct TileParams {
    uint32_t tile_x, tile_y;      // タイル開始位置
    uint32_t tile_w, tile_h;      // タイルサイズ
    uint32_t full_w, full_h;      // 画像全体サイズ
    uint32_t samples;             // サンプル数
    uint32_t sample_offset;       // 累積サンプル数
    uint32_t max_depth;           // 最大バウンス数
};
```

---

## コア実装の解説

### 1. シーンエクスポート (scene_export.py)

Blender の内部データを JSON に変換します。

```python
# 主要な変換対象:
# - メッシュ (頂点、三角形、法線)
# - マテリアル (Base Color, Metallic, Roughness, Emission, Transmission)
# - ノードツリー (Principled BSDF のパラメータ)
# - カメラ (位置、方向、FOV)

def export_scene_to_file(scene, filepath, cam_params):
    """
    Blender シーンを JSON ファイルにエクスポート
    
    処理フロー:
    1. 可視メッシュオブジェクトを収集
    2. 各メッシュを三角形に変換 (evaluated mesh)
    3. マテリアル情報を抽出
    4. ノードツリーをシリアライズ
    5. JSON ファイルに書き出し
    """
```

### 2. サブプロセス管理 (subprocess_renderer.py)

C++ レンダラープロセスとの通信を管理します。

```python
class SubprocessRenderer:
    """
    主要メソッド:
    
    start(config)       - サーバープロセス起動 + INIT コマンド
    update_scene(json)  - シーンデータ送信 (スレッド化)
    update_camera(cam)  - カメラパラメータ送信
    render_tile(params) - レンダリング実行 + ピクセル受信
    stop()              - SHUTDOWN コマンド + プロセス終了
    """
```

### 3. パストレーシング (pbr.hpp)

Monte Carlo 法による光輸送シミュレーション。

```cpp
/**
 * レンダリング方程式:
 * Lo(p, ωo) = Le(p, ωo) + ∫ f(p, ωi, ωo) * Li(p, ωi) * cos(θi) dωi
 *
 * Monte Carlo 推定:
 * Lo ≈ (1/N) * Σ f(p, ωi, ωo) * Li(p, ωi) * cos(θi) / pdf(ωi)
 *
 * 実装されているアルゴリズム:
 * - Naive: BSDF サンプリングのみ
 * - NEE (Next Event Estimation): 直接光サンプリング
 * - MIS: BSDF + Light サンプリングの組み合わせ
 */
```

### 4. BVH (renderer.hpp)

レイとジオメトリの交差判定を高速化。

```cpp
/**
 * Bounding Volume Hierarchy (BVH):
 * - 各メッシュの AABB (軸平行境界箱) を計算
 * - 二分木構造で空間分割
 * - O(log N) でレイキャスト
 */
```

---

## ソースコードの読み方

### 推奨の読み進め方

#### Step 1: エントリーポイントを理解する

```
DIYRenderer/__init__.py
    ↓ (アドオン登録)
DIYRenderer/engine.py :: DIYRenderEngine
    ↓ (render メソッド)
```

`engine.py` の `render()` メソッドが全ての起点です。

#### Step 2: データフローを追う

```
engine.py :: render()
    │
    ├─→ scene_export.py :: export_scene_to_file()
    │       Blender → JSON 変換を理解
    │
    ├─→ subprocess_renderer.py :: SubprocessRenderer
    │       プロセス管理とプロトコルを理解
    │
    └─→ protocol.py :: ProtocolEncoder/Decoder
            バイナリフォーマットを理解
```

#### Step 3: C++ レンダラーを理解する

```
cpp_renderer/src/main.cpp
    │
    ├─→ src/server.cpp :: RenderServer
    │       コマンドハンドラを理解
    │
    ├─→ include/renderer.hpp
    │       シーン構造体、BVH、基本レンダリングを理解
    │
    └─→ include/pbr.hpp
            BSDF、MIS、物理ベースレンダリングを理解
```

### ファイル別の役割

| ファイル | 役割 | 読む優先度 |
|---------|------|-----------|
| `engine.py` | Blender統合、全体制御 | ★★★ 最初に読む |
| `subprocess_renderer.py` | プロセス管理、通信 | ★★★ |
| `scene_export.py` | シーン変換 | ★★☆ |
| `protocol.py` | プロトコル定義 | ★★☆ |
| `server.cpp` | C++ サーバー | ★★★ |
| `renderer.hpp` | レンダリングコア | ★★★ |
| `pbr.hpp` | 物理ベースシェーディング | ★★☆ 数学的 |
| `node_evaluator.cpp` | ノード評価 | ★☆☆ |

### 重要な関数/メソッド

#### Python 側

```python
# engine.py
DIYRenderEngine.render()           # レンダリング開始
DIYRenderEngine._ensure_server_started()  # サーバー起動
DIYRenderEngine._update_server_scene()    # シーン更新
DIYRenderEngine._render_with_server()     # サーバーモードレンダリング

# subprocess_renderer.py
SubprocessRenderer.start()         # プロセス起動 + INIT
SubprocessRenderer.update_scene()  # シーン送信 (スレッド化)
SubprocessRenderer.render_tile()   # タイルレンダリング

# scene_export.py
export_scene_to_file()            # メインエクスポート関数
serialize_node_tree()             # ノードツリー変換
```

#### C++ 側

```cpp
// server.cpp
RenderServer::run()               // メインループ
RenderServer::handleUpdateScene() // シーン読み込み
RenderServer::handleRenderTile()  // タイルレンダリング

// renderer.hpp
renderRegion()                    // ピクセルループ
traceRay()                        // パストレーシング
intersectScene()                  // レイキャスト

// pbr.hpp
renderPixelMIS()                  // MIS レンダリング
sampleBSDF()                      // BSDF サンプリング
evalBSDF()                        // BSDF 評価
```

---

## 技術的課題と解決策

### 1. パイプバッファ問題

**問題**: OS のパイプバッファは通常 64KB。844KB のシーンデータを送ると、
バッファが満杯になり `write()` がブロック。同時に C++ 側も全データ待ちで
`read()` がブロック → デッドロック。

```
Python (write) ──────▶ [64KB buffer] ──────▶ C++ (read)
      │                                           │
      │ バッファ満杯で                             │ 全データ待ちで
      │ ブロック                                   │ ブロック
      │                                           │
      └─────────────── デッドロック ───────────────┘
```

**解決策**: スレッド化送信 + チャンク読み込み

```python
# Python: 別スレッドで送信
def send_chunked():
    for chunk in chunks(data, 64KB):
        stdin.write(chunk)
        stdin.flush()

send_thread = Thread(target=send_chunked)
send_thread.start()
# メインスレッドはレスポンス待ち
```

```cpp
// C++: チャンク読み込み
while (bytesRead < size) {
    toRead = min(32KB, remaining);
    cin.read(buffer + bytesRead, toRead);
    bytesRead += toRead;
}
```

### 2. is_running() デッドロック

**問題**: `update_scene()` 内で `self._lock` を取得後、
`is_running()` を呼ぶと再度ロック取得を試みてデッドロック。

```python
def update_scene(self):
    with self._lock:            # ロック取得
        if not self.is_running():  # ← ここでまたロック取得を試みる
            ...

def is_running(self):
    with self._lock:            # デッドロック！
        return self._process is not None
```

**解決策**: `is_running()` はロックを取らない設計に変更

```python
def is_running(self) -> bool:
    # Note: Does not acquire lock - caller should hold lock if needed
    return self._process is not None and self._process.poll() is None
```

### 3. Blender メインスレッドブロッキング

**問題**: Blender のメインスレッドでブロッキング I/O を行うと UI がフリーズ。

**解決策**: 
- 大きなデータ送信は別スレッドで実行
- タイムアウト付きの待機
- select() でノンブロッキングチェック

---

## 今後の拡張予定

### Phase 2: WebGPU/Dawn 対応

```
┌─────────────────────────────────────────────────────────────────┐
│  Current: CPU Backend                                            │
│  - OpenMP による並列化                                           │
│  - 14 スレッド (M1 Pro)                                         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Future: WebGPU Backend                                          │
│  - Dawn ライブラリ使用                                           │
│  - GPU コンピュートシェーダー                                    │
│  - 10-100x 高速化の見込み                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: ビューポートレンダリング最適化

- インタラクティブなカメラ操作
- 低解像度プレビュー → 高解像度仕上げ
- デノイズ統合

---

## 参考資料

- [Physically Based Rendering: From Theory to Implementation](https://pbr-book.org/)
- [Blender Python API](https://docs.blender.org/api/current/)
- [Dawn WebGPU Implementation](https://dawn.googlesource.com/dawn)
