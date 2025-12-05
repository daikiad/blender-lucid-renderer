# ADR-001: レンダラーアーキテクチャの選定

## ステータス
**提案中 (Proposed)** - 2025-12-05

## コンテキスト

DIY Renderer は Blender のカスタムレンダーエンジンとして、Python アドオンと C++ パストレーサーで構成されている。現在の実装には以下の課題がある：

### 現状の問題

1. **毎回のプロセス起動コスト**
   - `subprocess.Popen()` で毎フレーム新プロセス起動
   - シーンJSONを毎回ロード・パース

2. **拡張性の欠如**
   - GPU レンダリング (WebGPU/Dawn) への対応が困難
   - アルゴリズム切り替え (NEE, MIS, BDPT) の仕組みがない

3. **将来の要件**
   - WebGPU (Dawn) による GPU レンダリング対応
   - UI から CPU/GPU バックエンド切り替え
   - パストレーシングアルゴリズムの選択

## 検討した選択肢

### 選択肢1: サーバーモード (stdin/stdout バイナリ通信)

```
Python Addon ←──stdin/stdout──→ diyrt --server (常駐プロセス)
```

- **実装工数**: 1-2週間
- **メリット**:
  - 現行コードからの自然な移行
  - プロセス分離による安全性（C++クラッシュでもBlender無事）
  - デバッグ容易（C++単独実行可能）
  - クロスプラットフォーム
- **デメリット**:
  - パイプI/Oのオーバーヘッド（数ms）
  - プロセス間でGPUリソース共有不可

### 選択肢2: Python C拡張 (pybind11)

```
Python Addon ←──直接呼出──→ diyrender.so (同一プロセス)
```

- **実装工数**: 3-4週間
- **メリット**:
  - 最高性能（関数呼び出しコスト≒0）
  - BlenderとGPUコンテキスト共有可能
  - NumPyでゼロコピーデータ共有
- **デメリット**:
  - Blender内蔵Python版に合わせたビルド必要
  - OS/Blenderバージョン毎にバイナリ配布
  - C++バグでBlender全体クラッシュ

### 選択肢3: 共有メモリ (mmap)

```
Python Addon ←──mmap + semaphore──→ diyrt --server
```

- **実装工数**: 3-4週間
- **メリット**:
  - 大規模シーンで最速（ゼロコピー）
  - プロセス分離を維持
- **デメリット**:
  - 同期制御が複雑
  - プラットフォーム依存（POSIX/Windows）
  - メモリ破損リスク

### 選択肢4: ソケット通信 (TCP/Unix)

```
Python Addon ←──TCP localhost:12345──→ diyrt --server
```

- **実装工数**: 1-2週間
- **メリット**:
  - リモートレンダリング対応可能
  - ネットワークデバッグツール活用
- **デメリット**:
  - stdin/stdoutより遅い
  - 接続管理のコード必要

### 選択肢5: ハイブリッド（段階的移行）

```
Phase 1: サーバーモード (stdin/stdout)
    ↓
Phase 2: WebGPU追加 (サーバーモード上)
    ↓
Phase 3: pybind11版追加 (オプション・性能最適化)
```

- **実装工数**: 段階的
- **メリット**:
  - リスク分散、段階的に機能追加
  - 開発時は安全なsubprocess、本番は高速版
  - C++コア共通、通信レイヤーのみ差し替え
- **デメリット**:
  - 複数実装のメンテナンス

## 決定

**選択肢5: ハイブリッド（サーバーモードから開始）** を採用する。

### Phase 1 (今回): サーバーモード基盤
- C++ に `--server` モード追加
- バイナリプロトコル設計・実装
- Python `SubprocessRenderer` クラス実装
- 既存 CPU Backend + NEE をサーバー化

### Phase 2: WebGPU 統合
- Dawn ビルド統合
- `WebGPUBackend` 実装
- WGSL シェーダー作成

### Phase 3 (オプション): pybind11 最適化
- 必要に応じて同じC++コアをpybind11でラップ
- `PybindRenderer` クラス追加

## 理由

1. **段階的リスク軽減**: サーバーモードは現行コードから自然に移行でき、動作確認しながら進められる

2. **安全性**: プロセス分離により、開発中のC++バグでBlenderがクラッシュしない

3. **デバッグ容易性**: C++レンダラーを単独で実行・テストできる

4. **WebGPU対応**: サーバーモード上でも WebGPU Backend は問題なく動作する

5. **将来の拡張**: C++コアを共通化しておけば、後からpybind11版を追加できる

## 結果

### 期待される効果

| 指標 | 現在 | Phase 1後 |
|------|------|-----------|
| カメラ更新時間 | 250-300ms | 200ms (シーンロード省略) |
| シーン変更時間 | 250-300ms | 210ms |
| プロセス起動 | 毎回 | 1回のみ |

### アーキテクチャ

```
DIYRenderer/
├── __init__.py
├── engine.py                 # DIYRenderEngine
├── renderer_interface.py     # [NEW] Abstract interface
├── subprocess_renderer.py    # [NEW] stdin/stdout 実装
├── protocol.py               # [NEW] Binary protocol
│
└── cpp_renderer/
    ├── include/
    │   ├── backend.hpp       # [NEW] IBackend interface
    │   ├── algorithm.hpp     # [NEW] IAlgorithm interface
    │   ├── protocol.hpp      # [NEW] Binary protocol
    │   └── server.hpp        # [NEW] Server loop
    ├── src/
    │   ├── main.cpp          # --server モード追加
    │   ├── server.cpp        # [NEW]
    │   ├── backend_cpu.cpp   # [NEW] 既存コード移動
    │   └── backend_webgpu.cpp # [Phase 2]
    └── shaders/              # [Phase 2] WGSL
```

### バイナリプロトコル

```
Command:  [Magic 4B][CmdType 4B][PayloadSize 4B][Payload...]
Response: [Magic 4B][RespType 4B][Status 4B][PayloadSize 4B][Payload...]

Commands:
  0x01 CMD_INIT          - 初期化
  0x02 CMD_UPDATE_SCENE  - シーン更新
  0x03 CMD_UPDATE_CAMERA - カメラ更新 (40B: pos, dir, up, fov)
  0x04 CMD_RENDER_TILE   - レンダリング要求
  0x05 CMD_CANCEL        - キャンセル
  0x06 CMD_SET_BACKEND   - バックエンド切替 (cpu/webgpu)
  0x07 CMD_SET_ALGORITHM - アルゴリズム切替 (nee/mis/...)
  0xFF CMD_SHUTDOWN      - 終了
```

## 補足

### 参考資料
- [ARCHITECTURE_v2.md](../ARCHITECTURE_v2.md) - 詳細設計
- [Cycles Render Engine](https://wiki.blender.org/wiki/Source/Render/Cycles) - Blender内蔵レンダラー参考

### 関連ADR
- (なし - 初回)

### 変更履歴
- 2025-12-05: 初版作成
