# Blender-Renderer通信アーキテクチャ分析

## 現在の実装（Current Architecture）

### データフロー
```
Blender Scene
    ↓
[Python] export_scene_to_json() 
    ↓ 一時JSONファイル書き込み
Temp File (/tmp/diy_scene_XXXXX.json)
    ↓
[Python] subprocess.run(diyrt --scene file.json ...)
    ↓ プロセス起動・完了待ち
[C++] main.cpp: loadSceneFromJson()
    ↓
[C++] レンダリング実行
    ↓ stdout にピクセルデータ出力
[Python] proc.stdout を読み取り
    ↓
[Python] ピクセル配列に変換
    ↓
Blender Texture/Result に書き込み
```

### 現在の問題点

#### 1. **ビューポートレンダリングの度にシーン全体をエクスポート**
- `view_draw()` が呼ばれる度に `export_scene_to_json()` を実行
- メッシュデータ、マテリアル、属性すべてを毎回JSONシリアライズ
- ファイルI/O のオーバーヘッド大

**コスト**: 数MB〜数十MBのJSONを毎フレーム書き込み

#### 2. **プロセス起動コスト**
- `subprocess.run()` で毎回新しいプロセスを起動
- プロセス起動時間: 数十〜数百ms
- シーンロード時間: JSONパース + データ構造構築

**コスト**: ビューポート更新ごとに100ms以上のレイテンシ

#### 3. **同期的な実行**
- `subprocess.run()` は完了を待つ（現在は `async_render_viewport()` スレッドで緩和済み）
- ただし、スレッド内でも同期的にプロセス完了を待つ

#### 4. **データ転送の非効率性**
- JSONテキスト形式（パース重い、サイズ大）
- stdoutでピクセルデータをテキスト転送（`"r g b a\n"` 形式）
- バイナリなら 4x小さい、パース不要

---

## 改善案（Proposed Improvements）

### 方式1: **永続的なレンダラープロセス（推奨）**

#### アーキテクチャ
```
[Blender Python Addon]
    ↓
[長期実行プロセス: diyrt --server]
    ↑↓ 双方向通信
    stdin:  コマンド受信（バイナリプロトコル）
    stdout: ピクセルデータ送信（バイナリ）
```

#### 実装戦略

##### A. **起動・停止管理**
```python
class DIYRenderEngine:
    def __init__(self):
        self.renderer_process = None
    
    def _ensure_renderer_running(self):
        if self.renderer_process is None or self.renderer_process.poll() is not None:
            # プロセス起動（サーバーモード）
            self.renderer_process = subprocess.Popen(
                [binary, '--server'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    
    def __del__(self):
        if self.renderer_process:
            self.renderer_process.terminate()
```

##### B. **通信プロトコル（バイナリ）**

**コマンド形式**:
```
[4 bytes: command type]
[4 bytes: payload size]
[N bytes: payload data]
```

**コマンド種類**:
- `CMD_UPDATE_SCENE = 1`: シーンデータ更新（初回 or 変更時のみ）
- `CMD_UPDATE_CAMERA = 2`: カメラのみ更新（頻繁）
- `CMD_RENDER_TILE = 3`: タイルレンダリング要求
- `CMD_CANCEL = 4`: 現在のレンダリングをキャンセル
- `CMD_SHUTDOWN = 5`: プロセス終了

**Python側送信例**:
```python
import struct

def send_camera_update(proc, cam_params):
    # コマンド種別
    cmd = struct.pack('I', 2)  # CMD_UPDATE_CAMERA
    
    # カメラデータ（float x 10）
    data = struct.pack('10f', 
        cam_params['pos'].x, cam_params['pos'].y, cam_params['pos'].z,
        cam_params['dir'].x, cam_params['dir'].y, cam_params['dir'].z,
        cam_params['up'].x, cam_params['up'].y, cam_params['up'].z,
        cam_params['fov']
    )
    
    # サイズ + データ
    size = struct.pack('I', len(data))
    proc.stdin.write(cmd + size + data)
    proc.stdin.flush()

def send_render_request(proc, x, y, w, h, samples):
    cmd = struct.pack('I', 3)  # CMD_RENDER_TILE
    data = struct.pack('5I', x, y, w, h, samples)
    size = struct.pack('I', len(data))
    proc.stdin.write(cmd + size + data)
    proc.stdin.flush()
```

**C++側受信例**:
```cpp
enum Command {
    CMD_UPDATE_SCENE = 1,
    CMD_UPDATE_CAMERA = 2,
    CMD_RENDER_TILE = 3,
    CMD_CANCEL = 4,
    CMD_SHUTDOWN = 5
};

void serverLoop() {
    Scene scene;
    Camera camera;
    
    while(true) {
        uint32_t cmd_type;
        std::cin.read((char*)&cmd_type, 4);
        
        uint32_t payload_size;
        std::cin.read((char*)&payload_size, 4);
        
        switch(cmd_type) {
            case CMD_UPDATE_CAMERA: {
                float cam_data[10];
                std::cin.read((char*)cam_data, sizeof(cam_data));
                updateCamera(camera, cam_data);
                break;
            }
            case CMD_RENDER_TILE: {
                uint32_t params[5];
                std::cin.read((char*)params, sizeof(params));
                int x = params[0], y = params[1], w = params[2], h = params[3], samples = params[4];
                renderTile(scene, camera, x, y, w, h, samples);
                break;
            }
            case CMD_SHUTDOWN:
                return;
        }
    }
}
```

##### C. **ピクセルデータ転送（バイナリ）**

**C++側送信**:
```cpp
void sendPixels(int width, int height, const std::vector<Vec3> &pixels) {
    // ヘッダー: [width][height]
    uint32_t header[2] = {(uint32_t)width, (uint32_t)height};
    std::cout.write((char*)header, sizeof(header));
    
    // ピクセルデータ: RGB float32 x 3
    for(const auto &px : pixels) {
        float rgb[3] = {px.x, px.y, px.z};
        std::cout.write((char*)rgb, sizeof(rgb));
    }
    std::cout.flush();
}
```

**Python側受信**:
```python
import struct
import numpy as np

def receive_pixels(proc):
    # ヘッダー読み取り
    header = proc.stdout.read(8)
    width, height = struct.unpack('II', header)
    
    # ピクセルデータ読み取り
    pixel_count = width * height
    pixel_bytes = proc.stdout.read(pixel_count * 12)  # 3 floats * 4 bytes
    
    # NumPy配列に変換
    pixels = np.frombuffer(pixel_bytes, dtype=np.float32)
    pixels = pixels.reshape((height, width, 3))
    
    return pixels
```

#### メリット
- ✅ プロセス起動コスト: 1回のみ
- ✅ シーンロード: 変更時のみ
- ✅ カメラ更新: 40バイトの送信のみ（数マイクロ秒）
- ✅ データ転送: バイナリで高速
- ✅ 低レイテンシ: 数ms〜数十ms

#### デメリット
- ❌ 実装複雑度UP
- ❌ プロセス管理が必要
- ❌ デバッグが難しい

---

### 方式2: **共有メモリ（最速）**

#### アーキテクチャ
```
[Blender Python]     [C++ Renderer]
       ↓                    ↓
   [Shared Memory Segment]
       - Scene Data
       - Camera Data
       - Render Result
       ↓                    ↑
   [Named Semaphore/Event]
   （同期・通知用）
```

#### 実装例（Python側）
```python
import mmap
import posix_ipc  # or multiprocessing.shared_memory

# 共有メモリ作成
shm = posix_ipc.SharedMemory('/diy_renderer_scene', posix_ipc.O_CREAT, size=100*1024*1024)
mapfile = mmap.mmap(shm.fd, shm.size)

# シーンデータ書き込み
scene_bytes = serialize_scene(depsgraph)
mapfile.seek(0)
mapfile.write(scene_bytes)

# イベント通知
event = posix_ipc.Semaphore('/diy_render_event', posix_ipc.O_CREAT)
event.release()  # レンダラーに通知
```

#### メリット
- ✅ 最速のデータ転送（メモリコピーのみ）
- ✅ ゼロシリアライゼーション（構造体を直接共有可能）

#### デメリット
- ❌ プラットフォーム依存（POSIX/Windows API異なる）
- ❌ 同期制御が複雑
- ❌ メモリ破損リスク

---

### 方式3: **ソケット通信（柔軟）**

#### アーキテクチャ
```
[Blender Python]  <--TCP/Unix Socket-->  [C++ Renderer]
   localhost:12345
```

#### メリット
- ✅ リモートレンダリング可能
- ✅ 複数レンダラーインスタンス対応
- ✅ クロスプラットフォーム

#### デメリット
- ❌ ネットワークオーバーヘッド（ローカルでも）
- ❌ 共有メモリより遅い

---

## 推奨実装ロードマップ

### Phase 1: **差分更新の実装**（即効性あり）
現在の `subprocess.run()` 方式のままで改善:

```python
class DIYRenderEngine:
    def __init__(self):
        self.last_scene_hash = None
        self.cached_scene_file = None
    
    def view_draw(self, context, depsgraph):
        # シーンハッシュ計算（メッシュ数、頂点数、マテリアルなど）
        scene_hash = compute_scene_hash(depsgraph)
        
        # シーンが変更されていなければエクスポートスキップ
        if scene_hash == self.last_scene_hash and self.cached_scene_file:
            scene_file = self.cached_scene_file
        else:
            scene_file = export_scene_to_json(depsgraph)
            self.last_scene_hash = scene_hash
            self.cached_scene_file = scene_file
        
        # カメラだけ更新してレンダリング
        call_external_renderer(scene_file, ...)
```

**効果**: シーン静止時のエクスポートコスト削減

### Phase 2: **永続プロセス化**（中期目標）
- サーバーモード実装（stdin/stdoutバイナリ通信）
- カメラ専用更新コマンド
- タイルベースレンダリング

**効果**: プロセス起動コスト削減、レスポンス向上

### Phase 3: **共有メモリ最適化**（長期目標）
- 大規模シーン対応
- リアルタイム性追求

---

## Cycles レンダラーの実装参考

Cyclesは以下の戦略を使用:
1. **Blender内蔵C++レンダラー** - プロセス分離なし
2. **セッション管理** - `Session` オブジェクトで状態保持
3. **タイルベース** - 画面を分割して並列レンダリング
4. **プログレッシブ** - 低サンプルから開始、徐々に精度向上
5. **デノイザー統合** - ノイズ除去でサンプル数削減

**我々との違い**:
- 外部プロセス vs 内蔵C++
- JSON通信 vs 直接関数呼び出し

**参考にできる点**:
- プログレッシブレンダリング戦略（✅ 実装済み）
- タイル分割
- サンプル数の動的調整

---

## 次のステップ提案

### Option A: **段階的改善（推奨）**
1. まず差分更新を実装（1-2時間）
2. 効果測定
3. 必要なら永続プロセス化

### Option B: **一気に永続プロセス化**
1. `--server` モード実装
2. バイナリプロトコル設計
3. Python側通信レイヤー実装

どちらを進めますか？
