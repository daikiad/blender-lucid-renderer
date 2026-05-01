# subprocess と threading 完全ガイド

`subprocess_renderer.py` で使われている Python の `subprocess` と `threading` を、
**一行ずつ完全に解説**します。

---

## 目次

1. [subprocess モジュールの基礎](#subprocess-モジュールの基礎)
2. [threading モジュールの基礎](#threading-モジュールの基礎)
3. [コード完全解説（一行ずつ）](#コード完全解説一行ずつ)

---

## subprocess モジュールの基礎

### subprocess とは？

Python から**別のプログラムを起動**するためのモジュールです。

```python
import subprocess

# 例: ls コマンドを実行
result = subprocess.run(['ls', '-la'])
```

### 3つの通信チャネル

プロセスには3つの入出力チャネルがあります：

```
┌─────────────────┐
│  親プロセス      │  (Python)
│  (このスクリプト)│
└────┬─┬─┬────────┘
     │ │ │
     │ │ └─ stderr ← エラー出力
     │ └─── stdout ← 標準出力
     └───── stdin  → 標準入力
          ↓
┌─────────────────┐
│  子プロセス      │  (diyrt)
│  (C++ レンダラー)│
└─────────────────┘
```

### subprocess.Popen の使い方

```python
process = subprocess.Popen(
    ['プログラムのパス', '引数1', '引数2'],  # 実行するコマンド
    stdin=subprocess.PIPE,    # stdin をパイプとして開く
    stdout=subprocess.PIPE,   # stdout をパイプとして開く
    stderr=subprocess.PIPE,   # stderr をパイプとして開く
    bufsize=0                 # バッファサイズ (0 = アンバッファード)
)
```

**各引数の意味**:

- **`stdin=subprocess.PIPE`**: 
  - 子プロセスの stdin をパイプにする
  - `process.stdin.write(data)` で子プロセスにデータを送れる

- **`stdout=subprocess.PIPE`**:
  - 子プロセスの stdout をパイプにする
  - `process.stdout.read(size)` で子プロセスからデータを受け取れる

- **`stderr=subprocess.PIPE`**:
  - 子プロセスの stderr をパイプにする
  - `process.stderr.readline()` でエラーメッセージを読める

- **`bufsize=0`** (アンバッファード):
  - データを即座に送受信 (バッファに溜めない)
  - リアルタイム通信に必要

### パイプとは？

**パイプ = プロセス間でデータをやり取りする管**

```
Python                         C++
  ↓                            ↑
stdin.write("Hello")  →  [パイプ]  →  std::cin >> data
  ↑                            ↓
stdout.read()  ←  [パイプ]  ←  std::cout << result
```

### プロセスの状態確認

```python
# プロセスがまだ動いているか？
if process.poll() is None:
    print("まだ実行中")
else:
    print("終了した、終了コード:", process.poll())
```

`poll()` の返り値:
- `None`: まだ実行中
- `0`: 正常終了
- `1以上`: エラー終了

---

## threading モジュールの基礎

### threading とは？

**複数の処理を同時に実行**するためのモジュールです。

```python
import threading

def worker():
    print("バックグラウンドで実行中...")

# スレッドを作成して起動
thread = threading.Thread(target=worker)
thread.start()  # worker() が別スレッドで実行される
```

### なぜスレッドが必要？

**問題**: データ送信中にブロックすると、受信ができない

```
# 悪い例 (デッドロック)
process.stdin.write(large_data)  # ← バッファが満杯でブロック
result = process.stdout.read()   # ← 永遠に届かない
```

**解決**: 別スレッドで送信し、メインスレッドで受信

```
スレッド1 (送信)              スレッド2 (メイン)
     ↓                              ↓
 stdin.write() ────→         stdout.read()
     ↓                              ↓
  完了通知  ────────────→      処理続行
```

### threading.Lock (ロック)

**複数スレッドから同じデータを触ると壊れる** → ロックで保護

```python
lock = threading.Lock()

# スレッド1
with lock:
    data = shared_list.pop()  # ロック中なので安全

# スレッド2
with lock:
    shared_list.append(item)  # スレッド1の処理が終わるまで待つ
```

`with lock:` の中は**一度に1つのスレッドしか入れない**

### threading.Event (イベント)

**スレッド間で「完了した」を通知**するためのフラグ

```python
done = threading.Event()

# スレッド1
def worker():
    do_something()
    done.set()  # 「完了した」フラグを立てる

# メインスレッド
thread.start()
done.wait()  # スレッド1が done.set() するまで待つ
print("完了！")
```

---

## コード完全解説（一行ずつ）

### 1. インポート部分

```python
import os
```
**os モジュール**: ファイルパスの操作、環境変数の取得など OS 関連の機能

```python
import subprocess
```
**subprocess モジュール**: 別プログラムを起動して通信する

```python
import threading
```
**threading モジュール**: マルチスレッド (並行処理) を実現する

```python
import time
```
**time モジュール**: 時間計測、スリープなど

```python
from typing import Optional, Dict, List
```
**型ヒント用**: コードの可読性を上げるための型情報
- `Optional[str]` = `str` または `None`
- `Dict[str, int]` = キーが `str`、値が `int` の辞書
- `List[str]` = `str` のリスト

### 2. find_renderer_binary() 関数

```python
def find_renderer_binary() -> Optional[str]:
```
**関数定義**: 
- `-> Optional[str]` は返り値が `str` または `None` であることを示す型ヒント

```python
    env_path = os.environ.get('DIY_RENDERER_BIN')
```
**環境変数取得**:
- `os.environ` = システムの環境変数の辞書
- `.get('キー')` = 存在すれば値、なければ `None`
- 例: `export DIY_RENDERER_BIN=/custom/path/diyrt` と設定していれば取得できる

```python
    if env_path and os.path.isfile(env_path):
        return env_path
```
**条件分岐**:
- `env_path and` = `env_path` が `None` でも空文字でもない
- `os.path.isfile(path)` = そのパスがファイルとして存在するか？
- 両方真なら早期リターン

```python
    addon_dir = os.path.dirname(os.path.abspath(__file__))
```
**現在のファイルのディレクトリを取得**:
- `__file__` = このスクリプトファイルのパス (例: `/path/to/subprocess_renderer.py`)
- `os.path.abspath(__file__)` = 絶対パスに変換 (例: `/Users/.../subprocess_renderer.py`)
- `os.path.dirname(...)` = ディレクトリ部分を取得 (例: `/Users/.../DIYRenderer`)

```python
    candidates = [
        os.path.join(addon_dir, 'cpp_renderer', 'build', 'diyrt'),
        ...
    ]
```
**候補パスのリストを作成**:
- `os.path.join(a, b, c)` = パスを結合 (OS に応じたセパレータを使用)
- Windows: `a\b\c`
- Mac/Linux: `a/b/c`

```python
    for c in candidates:
        if os.path.isfile(c):
            return c
```
**リストを順に試す**:
- `for c in candidates:` = リストの各要素を順に取り出す
- 見つかったら即座に `return` (ループを抜ける)

```python
    return None
```
**何も見つからなかった場合**: `None` を返す

### 3. SubprocessRenderer クラス - __init__

```python
class SubprocessRenderer(RendererInterface):
```
**クラス定義**: 
- `(RendererInterface)` = 継承 (RendererInterface の子クラス)

```python
    def __init__(self):
```
**コンストラクタ**: インスタンス作成時に自動で呼ばれる

```python
        self._process: Optional[subprocess.Popen] = None
```
**インスタンス変数の初期化**:
- `self._process` = サブプロセスオブジェクト (最初は `None`)
- `subprocess.Popen` = プロセスを表すクラス
- 型ヒント `Optional[subprocess.Popen]` = `Popen` オブジェクトまたは `None`

```python
        self._config: Optional[RenderConfig] = None
```
**設定を保存**: 後で参照できるように `None` で初期化

```python
        self._lock = threading.Lock()
```
**ロックオブジェクト作成**:
- `threading.Lock()` = 新しいロックを作成
- スレッドセーフにするため (複数スレッドからの同時アクセスを防ぐ)

```python
        self._binary_path: Optional[str] = None
```
**バイナリパスを保存**: 最初は `None`

### 4. start() メソッド - サブプロセス起動

```python
    def start(self, config: RenderConfig) -> bool:
```
**関数定義**:
- `config: RenderConfig` = 引数の型ヒント
- `-> bool` = 返り値は `True` または `False`

```python
        with self._lock:
```
**ロック取得 (with 文)**:
- ブロックに入る時: `self._lock.acquire()` (ロック取得)
- ブロックを出る時: `self._lock.release()` (ロック解放)
- 他のスレッドが同じロックを取ろうとすると、解放されるまで待たされる

```python
            if self._process is not None and self._process.poll() is None:
                return True
```
**すでに起動済みかチェック**:
- `self._process is not None` = プロセスオブジェクトが存在する
- `self._process.poll() is None` = プロセスがまだ実行中
- 両方真なら `True` を返して終了 (二重起動を防ぐ)

```python
            binary = find_renderer_binary()
```
**バイナリを検索**: 先ほどの関数を呼ぶ

```python
            if not binary:
                print("[SubprocessRenderer] Renderer binary not found")
                return False
```
**バイナリが見つからなければエラー**:
- `not binary` = `binary` が `None` または空文字
- `print(...)` = エラーメッセージを表示
- `return False` = 失敗を返す

```python
            self._binary_path = binary
            self._config = config
```
**インスタンス変数に保存**: 後で参照できるように

```python
            try:
```
**例外処理ブロック開始**: 
- この中でエラーが起きたら `except` ブロックに飛ぶ

```python
                self._process = subprocess.Popen(
```
**サブプロセス起動**:
- `Popen` = Process Open (プロセスを開く)
- 返り値はプロセスオブジェクト

```python
                    [binary, '--server'],
```
**実行するコマンド**:
- リストの1番目 = 実行ファイルのパス
- 2番目以降 = 引数
- 例: `['/path/to/diyrt', '--server']` → シェルで言う `./diyrt --server`

```python
                    stdin=subprocess.PIPE,
```
**stdin をパイプにする**:
- `subprocess.PIPE` = 特殊な定数
- Python から子プロセスの標準入力にデータを送れるようになる

```python
                    stdout=subprocess.PIPE,
```
**stdout をパイプにする**:
- 子プロセスの標準出力を Python で読み取れる

```python
                    stderr=subprocess.PIPE,
```
**stderr をパイプにする**:
- 子プロセスのエラー出力を Python で読み取れる

```python
                    bufsize=0
```
**バッファサイズを0に**:
- `0` = アンバッファードモード (バッファを使わない)
- データを即座に送受信 (遅延なし)
- リアルタイム通信に必須

```python
                )
```
ここで `Popen` のコンストラクタが完了し、子プロセスが起動する

### 5. stderr 読み取りスレッドの起動

```python
                self._stderr_thread = threading.Thread(
```
**新しいスレッドを作成**:
- `threading.Thread()` = スレッドオブジェクトの作成 (まだ起動していない)

```python
                    target=self._read_stderr,
```
**実行する関数を指定**:
- `target=関数名` = このスレッドで実行する関数
- 関数名だけ渡す (括弧をつけない)
- スレッドが起動すると `self._read_stderr()` が別スレッドで実行される

```python
                    daemon=True
```
**デーモンスレッドに設定**:
- `daemon=True` = メインスレッドが終了したら、このスレッドも自動で終了
- `daemon=False` (デフォルト) だと、このスレッドが終わるまでプログラムが終了しない

```python
                )
                self._stderr_thread.start()
```
**スレッドを起動**:
- `.start()` を呼ぶと別スレッドで `target` の関数が実行開始
- メインスレッドはここでブロックせず、次の行に進む

### 6. INIT コマンド送信

```python
                init_cmd = ProtocolEncoder.encode_init(
                    backend=config.backend.value,
                    algorithm=config.algorithm.value
                )
```
**バイナリコマンドを作成**:
- `ProtocolEncoder.encode_init()` = INIT コマンドをバイナリデータに変換
- 返り値は `bytes` (バイナリデータ)
- 例: `b'\x44\x49\x59\x52\x00\x00\x00\x01...'`

```python
                self._send(init_cmd)
```
**コマンドを送信**:
- `_send()` は後で定義される (stdin にデータを書き込む)

```python
                resp_type, status, _ = self._recv_header()
```
**レスポンスヘッダーを受信**:
- `_recv_header()` = stdout からヘッダーを読み取る
- 返り値は3つのタプル: `(response_type, status_code, payload_size)`
- `_` = 使わない値 (payload_size) を捨てる慣習

```python
                if resp_type != ResponseType.ACK or status != StatusCode.OK:
```
**レスポンスをチェック**:
- `ResponseType.ACK` = ACK レスポンスの定数 (例: 129)
- `StatusCode.OK` = 成功ステータスの定数 (例: 0)
- `or` = どちらか一方でも偽なら

```python
                    print(f"[SubprocessRenderer] Init failed: {status}")
                    self.stop()
                    return False
```
**初期化失敗時の処理**:
- `f"..."` = f-string (変数を埋め込める文字列)
- `self.stop()` = プロセスを停止するメソッド呼び出し
- `return False` = 失敗を返す

```python
                print(f"[SubprocessRenderer] Started: backend={config.backend.value}, algorithm={config.algorithm.value}")
                return True
```
**成功メッセージ**:
- 起動成功を通知して `True` を返す

```python
            except Exception as e:
```
**例外をキャッチ**:
- `try` ブロック内で任意のエラーが発生したらここに来る
- `Exception` = すべてのエラーの基底クラス
- `as e` = エラーオブジェクトを変数 `e` に代入

```python
                print(f"[SubprocessRenderer] Failed to start: {e}")
                self._process = None
                return False
```
**エラー処理**:
- エラー内容を表示
- プロセスを `None` にリセット
- 失敗を返す

### 7. _read_stderr() メソッド - バックグラウンドスレッド

```python
    def _read_stderr(self):
```
**別スレッドで実行される関数**:
- `threading.Thread(target=self._read_stderr)` で指定されている

```python
        try:
```
**例外処理ブロック**: エラーが起きても無視して終了

```python
            while self._process and self._process.poll() is None:
```
**プロセスが実行中の間ループ**:
- `self._process` = プロセスオブジェクトが存在する
- `self._process.poll() is None` = プロセスがまだ動いている
- `while 条件:` = 条件が真の間ずっとループ

```python
                line = self._process.stderr.readline()
```
**stderr から1行読み取る**:
- `.readline()` = 改行まで読み取る (ブロッキング)
- C++ の `std::cerr << "..." << std::endl;` の出力を読む
- データが来るまで**ここで待つ**

```python
                if line:
```
**空でなければ**:
- `if line:` = `line` が空バイト列 (`b''`) でない

```python
                    print(f"[C++] {line.decode('utf-8', errors='replace').rstrip()}")
```
**表示**:
- `line.decode('utf-8')` = バイト列を文字列に変換
- `errors='replace'` = デコードできない文字は `?` に置き換え
- `.rstrip()` = 右側の空白 (改行含む) を削除
- `f"[C++] ..."` = プレフィックスを付けて表示

```python
        except Exception:
            pass
```
**エラーは無視**:
- スレッドが終了するだけ
- `pass` = 何もしない

### 8. _send() メソッド - データ送信

```python
    def _send(self, data: bytes) -> None:
```
**関数定義**:
- `data: bytes` = バイナリデータ (例: `b'\x44\x49...'`)
- `-> None` = 返り値なし

```python
        if self._process is None or self._process.stdin is None:
            raise RendererError("Renderer not running")
```
**プロセスが動いているかチェック**:
- `is None` = プロセスまたは stdin が存在しない
- `raise エラークラス("メッセージ")` = 例外を発生させる (関数を中断)

```python
        self._process.stdin.write(data)
```
**stdin にデータを書き込む**:
- `stdin.write(data)` = パイプ経由で子プロセスに送信
- 子プロセス側では `std::cin` から読み取れる

```python
        self._process.stdin.flush()
```
**バッファをフラッシュ**:
- `flush()` = バッファに溜まったデータを即座に送信
- これを呼ばないとデータが遅延する可能性がある

### 9. _recv() メソッド - データ受信

```python
    def _recv(self, size: int) -> bytes:
```
**関数定義**:
- `size: int` = 受信する**バイト数**
- `-> bytes` = バイナリデータを返す

```python
        if self._process is None or self._process.stdout is None:
            raise RendererError("Renderer not running")
```
**プロセスチェック**: 先ほどと同じ

```python
        data = b''
```
**空のバイト列で初期化**:
- `b''` = 空のバイナリデータ

```python
        while len(data) < size:
```
**必要なサイズに達するまでループ**:
- `len(data)` = 現在受信したバイト数
- `< size` = 目標に達していない

```python
            chunk = self._process.stdout.read(size - len(data))
```
**stdout から読み取る**:
- `.read(n)` = 最大 n バイト読み取る (ブロッキング)
- `size - len(data)` = まだ必要なバイト数
- **重要**: `read()` は指定バイト数より少なく返す可能性がある
- データが来るまで**ここで待つ**

```python
            if not chunk:
                raise RendererError("Connection closed")
```
**接続切断のチェック**:
- `not chunk` = 空のデータが返ってきた
- パイプが閉じられた = プロセスが終了した

```python
            data += chunk
```
**データを累積**:
- `+=` = 末尾に追加
- 例: `b'abc' + b'def'` → `b'abcdef'`

```python
        return data
```
**完全なデータを返す**: ループが終了 = 必要なバイト数が揃った

### 10. update_scene() メソッド - 大きなデータ送信

```python
    def update_scene(self, scene_json: str) -> bool:
```
**関数定義**: JSON 文字列を受け取って送信

```python
        with self._lock:
```
**ロック取得**: 他のスレッドとの競合を防ぐ

```python
            if not self.is_running():
                return False
```
**プロセス確認**: 動いていなければ失敗

```python
            try:
```
**例外処理開始**

```python
                scene_bytes = scene_json.encode('utf-8')
```
**文字列をバイト列に変換**:
- `.encode('utf-8')` = UTF-8 エンコーディングでバイナリ化
- 例: `"Hello"` → `b'Hello'`

```python
                cmd = ProtocolEncoder.encode_update_scene(scene_bytes)
```
**コマンドを作成**:
- ヘッダー (12バイト) + ペイロード (scene_bytes) を結合
- 例: 844KB の JSON → 844,248 バイトのコマンド

```python
                print(f"[SubprocessRenderer] Sending scene: {len(cmd)} bytes total ({len(scene_bytes)} payload)")
```
**デバッグ出力**: サイズを表示

```python
                import threading
```
**threading モジュールをインポート** (関数内でも可能)

```python
                send_error = [None]
```
**エラーを格納するリスト**:
- なぜリスト? → スレッド間で共有するため
- 変数をスレッド間で共有するには、ミュータブル (変更可能) なオブジェクトが必要
- `send_error = None` だと、スレッド内で `send_error = e` しても外から見えない
- `send_error[0] = e` なら、リストは共有されているので外から見える

```python
                send_done = threading.Event()
```
**イベントオブジェクト作成**:
- `Event()` = スレッド間通知用のフラグ
- 初期状態: セットされていない (False)

```python
                def send_chunked():
```
**ネストした関数定義** (ローカル関数):
- この関数は別スレッドで実行される

```python
                    """64KB ずつチャンク送信"""
                    try:
```
**docstring とエラー処理**

```python
                        CHUNK = 65536  # 64KB (パイプバッファサイズ)
```
**チャンクサイズを定義**:
- `65536` = 64 × 1024 = 64KB
- OS のパイプバッファサイズに合わせている

```python
                        offset = 0
```
**オフセット初期化**: 送信済みバイト数を追跡

```python
                        while offset < len(cmd):
```
**全データを送るまでループ**

```python
                            chunk = cmd[offset:offset+CHUNK]
```
**スライスでチャンクを取得**:
- `cmd[offset:offset+CHUNK]` = `offset` から `CHUNK` バイト分
- 例: `cmd[0:65536]` → 最初の 64KB
- 例: `cmd[65536:131072]` → 次の 64KB

```python
                            self._process.stdin.write(chunk)
```
**チャンクを送信**: stdin に書き込む

```python
                            self._process.stdin.flush()
```
**即座に送信**: バッファをフラッシュ

```python
                            offset += len(chunk)
```
**オフセットを進める**:
- 次のループで続きから送信

```python
                        send_done.set()
```
**イベントをセット**:
- `send_done.set()` = フラグを立てる (True にする)
- メインスレッドで待っている `send_done.wait()` が起きる

```python
                    except Exception as e:
```
**エラーをキャッチ**

```python
                        send_error[0] = e
```
**エラーをリストに格納**:
- `send_error[0] = e` → メインスレッドから `send_error[0]` で読み取れる

```python
                        send_done.set()
```
**エラー時もイベントをセット**: メインスレッドを起こす

```python
                send_thread = threading.Thread(target=send_chunked, daemon=True)
```
**スレッドを作成**:
- `target=send_chunked` = 上で定義した関数を実行
- `daemon=True` = デーモンスレッド

```python
                send_thread.start()
```
**スレッドを起動**:
- `send_chunked()` が別スレッドで実行開始
- メインスレッドはここでブロックせず、次の行に進む

```python
                if not send_done.wait(timeout=60.0):
```
**イベントを待つ (タイムアウト付き)**:
- `send_done.wait()` = `send_done.set()` が呼ばれるまで**ここで待つ**
- `timeout=60.0` = 最大60秒待つ
- 返り値: `True` (イベントがセットされた) または `False` (タイムアウト)
- `if not` = タイムアウトなら

```python
                    print("[SubprocessRenderer] Send timeout!")
                    return False
```
**タイムアウト処理**: 60秒以内に送信が終わらなかった

```python
                if send_error[0]:
```
**エラーチェック**:
- `send_error[0]` = スレッド内でエラーが起きたか？
- `None` でなければエラーあり

```python
                    print(f"[SubprocessRenderer] Send error: {send_error[0]}")
                    return False
```
**エラー表示して失敗**

```python
                print("[SubprocessRenderer] Send complete, reading response...")
```
**送信完了メッセージ**

```python
                resp_type, status, payload_size = self._recv_header()
```
**レスポンスヘッダーを受信**:
- C++ 側がデータを受け取って ACK を返す

```python
                if resp_type == ResponseType.ERROR:
                    error_msg = self._recv(payload_size).decode('utf-8') if payload_size > 0 else "Unknown"
                    print(f"[SubprocessRenderer] Scene update failed: {error_msg}")
                    return False
```
**エラーレスポンス処理**:
- `if payload_size > 0` = エラーメッセージがある
- `.decode('utf-8')` = バイト列を文字列に変換
- 三項演算子: `A if 条件 else B`

```python
                print(f"[SubprocessRenderer] Scene update response: {resp_type}, status={status}")
                return resp_type == ResponseType.ACK and status == StatusCode.OK
```
**成功判定**:
- `and` = 両方真なら `True`、それ以外は `False`

```python
            except Exception as e:
                print(f"[SubprocessRenderer] Scene update error: {e}")
                import traceback
                traceback.print_exc()
                return False
```
**例外処理**:
- `traceback.print_exc()` = スタックトレースを表示 (どこでエラーが起きたか)

---

## まとめ: キーコンセプト

### subprocess の使い方

1. **プロセス起動**: `subprocess.Popen([コマンド, 引数], stdin=PIPE, stdout=PIPE)`
2. **データ送信**: `process.stdin.write(data)` + `process.stdin.flush()`
3. **データ受信**: `process.stdout.read(size)` (ブロッキング)
4. **状態確認**: `process.poll()` (`None` = 実行中)

### threading の使い方

1. **スレッド作成**: `threading.Thread(target=関数名, daemon=True)`
2. **スレッド起動**: `thread.start()`
3. **ロック**: `with lock:` ブロックで排他制御
4. **イベント**: `event.set()` で通知、`event.wait()` で待機

### デッドロック回避のパターン

```python
# パイプバッファがあふれる問題
# ↓
# 解決策: 別スレッドで送信、メインスレッドで受信

def send_large_data():
    event = threading.Event()
    
    def sender():
        # 大きなデータをチャンクで送信
        for chunk in chunks(data):
            process.stdin.write(chunk)
        event.set()  # 完了通知
    
    thread = threading.Thread(target=sender)
    thread.start()
    
    # メインスレッドは受信
    response = process.stdout.read(size)
    
    event.wait()  # 送信完了を待つ
```

これで **subprocess と threading の基礎から実装の一行一行まで** 完全に理解できるはずです！
