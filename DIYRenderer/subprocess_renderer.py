"""
Subprocess-based renderer implementation.

サブプロセスベースのレンダラー実装。
stdin/stdout のバイナリプロトコルで持続する C++ レンダラープロセスと通信します。

このモジュールの役割:
1. C++ レンダラープロセス (diyrt) の起動と管理
2. バイナリプロトコルによるコマンド送信
3. レスポンス (ACK, PIXELS, ERROR) の受信と解釈

プロセス管理の特徴:
- 持続プロセス: 毎回起動するのではなく、プロセスを再利用
- パイプ通信: stdin/stdout でバイナリデータをやり取り
- スレッド化: 大きなデータ送信時のデッドロックを回避

通信プロトコル:
- Command: [Magic 4B][Type 4B][PayloadSize 4B][Payload...]
- Response: [Magic 4B][Type 4B][Status 4B][PayloadSize 4B][Payload...]
"""

import os
import subprocess
import threading
import time
from typing import Optional, Dict, List

from .renderer_interface import (
    RendererInterface, 
    RenderConfig, 
    CameraParams, 
    TileParams, 
    RenderResult,
    BackendType,
    AlgorithmType,
    RendererError
)
from .protocol import (
    ProtocolEncoder,
    ProtocolDecoder,
    CameraParams as ProtocolCameraParams,
    RenderTileParams,
    ResponseType,
    StatusCode,
    RESPONSE_HEADER_SIZE
)


def find_renderer_binary() -> Optional[str]:
    """
    C++ レンダラーバイナリ (diyrt) のパスを検索。
    
    検索順序:
    1. 環境変数 DIY_RENDERER_BIN
    2. アドオンディレクトリ内の build/diyrt
    3. Release/Debug ビルドディレクトリ
    
    Returns:
        バイナリのパス、見つからなければ None
    """
    # 環境変数を優先
    env_path = os.environ.get('DIY_RENDERER_BIN')
    if env_path and os.path.isfile(env_path):
        return env_path
    
    # アドオンディレクトリからの相対パス
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(addon_dir, 'cpp_renderer', 'build', 'diyrt'),
        os.path.join(addon_dir, 'cpp_renderer', 'build', 'Release', 'diyrt'),
        os.path.join(addon_dir, 'cpp_renderer', 'build', 'Debug', 'diyrt'),
    ]
    
    for c in candidates:
        if os.path.isfile(c):
            return c
    
    return None


class SubprocessRenderer(RendererInterface):
    """
    サブプロセスベースのレンダラー実装。
    
    C++ レンダラーをサブプロセスとして起動し、バイナリプロトコルで通信します。
    レンダラーは持続プロセスとして動作し、複数のレンダリング要求を処理できます。
    
    使用例:
        renderer = SubprocessRenderer()
        renderer.start(config)
        renderer.update_scene(scene_json)
        renderer.update_camera(camera)
        result = renderer.render_tile(tile)
        renderer.stop()
    
    スレッドセーフ:
        内部でロックを使用し、複数スレッドからの同時アクセスを防止します。
    
    Attributes:
        _process: サブプロセスオブジェクト
        _config: レンダリング設定
        _lock: スレッドセーフ用ロック
        _binary_path: レンダラーバイナリのパス
    """
    
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._config: Optional[RenderConfig] = None
        self._lock = threading.Lock()
        self._binary_path: Optional[str] = None
        
    def start(self, config: RenderConfig) -> bool:
        """
        レンダラーサーバープロセスを起動。
        
        処理フロー:
        1. バイナリを検索
        2. サブプロセスを起動 (--server モード)
        3. stderr 読み取りスレッドを起動
        4. INIT コマンドを送信
        5. ACK レスポンスを待機
        
        Args:
            config: レンダリング設定 (backend, algorithm, max_depth)
        
        Returns:
            起動成功なら True
        """
        with self._lock:
            # すでに起動済みならスキップ
            if self._process is not None and self._process.poll() is None:
                return True
            
            # バイナリを検索
            binary = find_renderer_binary()
            if not binary:
                print("[SubprocessRenderer] Renderer binary not found")
                return False
            
            self._binary_path = binary
            self._config = config
            
            try:
                # サブプロセスを起動
                # bufsize=0: アンバッファードモード (即座に送受信)
                self._process = subprocess.Popen(
                    [binary, '--server'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                
                # stderr 読み取りスレッドを起動
                # C++ の std::cerr 出力を "[C++]" プレフィックスで表示
                self._stderr_thread = threading.Thread(
                    target=self._read_stderr,
                    daemon=True
                )
                self._stderr_thread.start()
                
                # INIT コマンドを送信
                init_cmd = ProtocolEncoder.encode_init(
                    backend=config.backend.value,
                    algorithm=config.algorithm.value
                )
                self._send(init_cmd)
                
                # ACK レスポンスを待機
                resp_type, status, _ = self._recv_header()
                if resp_type != ResponseType.ACK or status != StatusCode.OK:
                    print(f"[SubprocessRenderer] Init failed: {status}")
                    self.stop()
                    return False
                
                print(f"[SubprocessRenderer] Started: backend={config.backend.value}, algorithm={config.algorithm.value}")
                return True
                
            except Exception as e:
                print(f"[SubprocessRenderer] Failed to start: {e}")
                self._process = None
                return False
    
    def _read_stderr(self):
        """
        バックグラウンドで stderr を読み取るスレッド関数。
        
        C++ レンダラーのログ出力 (std::cerr) をキャプチャして、
        "[C++]" プレフィックス付きで Python 側に表示します。
        """
        try:
            while self._process and self._process.poll() is None:
                line = self._process.stderr.readline()
                if line:
                    print(f"[C++] {line.decode('utf-8', errors='replace').rstrip()}")
        except Exception:
            pass
    
    def is_running(self) -> bool:
        """
        レンダラーが実行中かチェック。
        
        注意: ロックを取得しません。ロックを持っている状態で呼び出す場合に使用します。
        外部から呼び出す場合は _is_running_locked() を使用してください。
        """
        return self._process is not None and self._process.poll() is None
    
    def _is_running_locked(self) -> bool:
        """レンダラーが実行中かチェック (ロック取得版)。"""
        with self._lock:
            return self._process is not None and self._process.poll() is None
    
    def stop(self) -> None:
        """
        レンダラープロセスを停止。
        
        SHUTDOWN コマンドを送信し、プロセスの終了を待機します。
        タイムアウトした場合は強制終了 (kill) します。
        """
        with self._lock:
            if self._process is None:
                return
            
            try:
                # SHUTDOWN コマンド送信
                shutdown_cmd = ProtocolEncoder.encode_shutdown()
                self._send(shutdown_cmd)
                
                # プロセス終了を待機 (最大2秒)
                self._process.wait(timeout=2.0)
            except Exception as e:
                print(f"[SubprocessRenderer] Error during shutdown: {e}")
                try:
                    self._process.kill()  # タイムアウト時は強制終了
                except Exception:
                    pass
            
            self._process = None
            print("[SubprocessRenderer] Stopped")
    
    # =========================================================================
    # 低レベル通信メソッド
    # =========================================================================
    
    def _send(self, data: bytes) -> None:
        """
        レンダラーにデータを送信。
        
        stdin パイプに書き込み、即座にフラッシュします。
        """
        if self._process is None or self._process.stdin is None:
            raise RendererError("Renderer not running")
        self._process.stdin.write(data)
        self._process.stdin.flush()
    
    def _recv(self, size: int) -> bytes:
        """
        レンダラーからデータを受信。
        
        指定サイズのデータを完全に受信するまでループします。
        ネットワークソケットと異なり、パイプは部分読み取りの可能性があるため、
        必要なバイト数が揃うまで繰り返し読み取ります。
        """
        if self._process is None or self._process.stdout is None:
            raise RendererError("Renderer not running")
        
        data = b''
        while len(data) < size:
            chunk = self._process.stdout.read(size - len(data))
            if not chunk:
                raise RendererError("Connection closed")
            data += chunk
        return data
    
    def _recv_header(self):
        """
        レスポンスヘッダーを受信してデコード。
        
        Returns:
            (response_type, status_code, payload_size) のタプル
        """
        header_data = self._recv(RESPONSE_HEADER_SIZE)
        return ProtocolDecoder.decode_header(header_data)
    
    # =========================================================================
    # 高レベル通信メソッド (コマンド送信)
    # =========================================================================
    
    def update_scene(self, scene_json: str) -> bool:
        """
        シーンデータをレンダラーに送信。
        
        大きなシーンデータ (数百KB〜数MB) を送信する場合、パイプバッファ (64KB)
        が満杯になってデッドロックする可能性があります。これを回避するため、
        別スレッドでチャンク送信を行います。
        
        デッドロックの仕組み:
        1. Python: write() でパイプバッファが満杯 → ブロック
        2. C++: 全データが来るまで read() → ブロック
        3. 両方がブロックしてデッドロック
        
        解決策:
        - 送信を別スレッドで行い、メインスレッドは読み取り可能になるまで待機
        - C++ 側もチャンク読み取りで対応
        
        Args:
            scene_json: JSON 形式のシーンデータ文字列
        
        Returns:
            成功なら True
        """
        with self._lock:
            if not self.is_running():
                return False
            
            try:
                scene_bytes = scene_json.encode('utf-8')
                cmd = ProtocolEncoder.encode_update_scene(scene_bytes)
                print(f"[SubprocessRenderer] Sending scene: {len(cmd)} bytes total ({len(scene_bytes)} payload)")
                
                # -------------------------------------------------------------
                # 別スレッドでチャンク送信 (デッドロック回避)
                # -------------------------------------------------------------
                import threading
                send_error = [None]
                send_done = threading.Event()
                
                def send_chunked():
                    """64KB ずつチャンク送信"""
                    try:
                        CHUNK = 65536  # 64KB (パイプバッファサイズ)
                        offset = 0
                        while offset < len(cmd):
                            chunk = cmd[offset:offset+CHUNK]
                            self._process.stdin.write(chunk)
                            self._process.stdin.flush()
                            offset += len(chunk)
                        send_done.set()
                    except Exception as e:
                        send_error[0] = e
                        send_done.set()
                
                send_thread = threading.Thread(target=send_chunked, daemon=True)
                send_thread.start()
                
                # 送信完了を待機 (タイムアウト60秒)
                if not send_done.wait(timeout=60.0):
                    print("[SubprocessRenderer] Send timeout!")
                    return False
                
                if send_error[0]:
                    print(f"[SubprocessRenderer] Send error: {send_error[0]}")
                    return False
                
                print("[SubprocessRenderer] Send complete, reading response...")
                
                # レスポンス受信
                resp_type, status, payload_size = self._recv_header()
                
                if resp_type == ResponseType.ERROR:
                    error_msg = self._recv(payload_size).decode('utf-8') if payload_size > 0 else "Unknown"
                    print(f"[SubprocessRenderer] Scene update failed: {error_msg}")
                    return False
                
                print(f"[SubprocessRenderer] Scene update response: {resp_type}, status={status}")
                return resp_type == ResponseType.ACK and status == StatusCode.OK
                
            except Exception as e:
                print(f"[SubprocessRenderer] Scene update error: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def update_camera(self, camera: CameraParams) -> bool:
        """
        カメラパラメータを更新。
        
        カメラデータは 40 バイト固定長なので、デッドロックの心配はありません。
        
        Args:
            camera: カメラパラメータ (位置、方向、上、FOV)
        
        Returns:
            成功なら True
        """
        with self._lock:
            if not self.is_running():
                return False
            
            try:
                proto_cam = ProtocolCameraParams(
                    pos_x=camera.pos[0], pos_y=camera.pos[1], pos_z=camera.pos[2],
                    dir_x=camera.dir[0], dir_y=camera.dir[1], dir_z=camera.dir[2],
                    up_x=camera.up[0], up_y=camera.up[1], up_z=camera.up[2],
                    fov=camera.fov
                )
                
                cmd = ProtocolEncoder.encode_update_camera(proto_cam)
                self._send(cmd)
                
                resp_type, status, _ = self._recv_header()
                return resp_type == ResponseType.ACK and status == StatusCode.OK
                
            except Exception as e:
                print(f"[SubprocessRenderer] Camera update error: {e}")
                return False
    
    def render_tile(self, tile: TileParams) -> Optional[RenderResult]:
        """
        タイルレンダリングを要求。
        
        RENDER_TILE コマンドを送信し、レンダリング結果 (ピクセルデータ) を受信します。
        C++ 側でパストレーシングが実行され、ピクセルの累積値が返されます。
        
        Args:
            tile: タイルパラメータ (位置、サイズ、サンプル数など)
        
        Returns:
            RenderResult オブジェクト、失敗時は None
        """
        with self._lock:
            if not self.is_running():
                return None
            
            try:
                start_time = time.perf_counter()
                
                # タイルパラメータをプロトコル形式に変換
                params = RenderTileParams(
                    tile_x=tile.tile_x,
                    tile_y=tile.tile_y,
                    tile_w=tile.tile_w,
                    tile_h=tile.tile_h,
                    full_w=tile.full_w,
                    full_h=tile.full_h,
                    samples=tile.samples,
                    sample_offset=tile.sample_offset,
                    max_depth=self._config.max_depth if self._config else 8
                )
                
                # コマンド送信
                cmd = ProtocolEncoder.encode_render_tile(params)
                self._send(cmd)
                
                # レスポンス受信
                resp_type, status, payload_size = self._recv_header()
                
                if resp_type == ResponseType.ERROR:
                    error_msg = self._recv(payload_size).decode('utf-8') if payload_size > 0 else "Unknown error"
                    print(f"[SubprocessRenderer] Render failed: {error_msg}")
                    return None
                
                if resp_type != ResponseType.PIXELS:
                    print(f"[SubprocessRenderer] Unexpected response: {resp_type}")
                    return None
                
                # ピクセルデータ受信 (float32 × 4 × width × height)
                pixel_data = self._recv(payload_size)
                pixels = ProtocolDecoder.decode_pixels(pixel_data, tile.tile_w, tile.tile_h)
                
                end_time = time.perf_counter()
                render_time_ms = (end_time - start_time) * 1000
                
                return RenderResult(
                    pixels=pixels,
                    width=tile.tile_w,
                    height=tile.tile_h,
                    samples_rendered=tile.samples,
                    render_time_ms=render_time_ms
                )
                
            except Exception as e:
                print(f"[SubprocessRenderer] Render error: {e}")
                return None
    
    def cancel(self) -> None:
        """Cancel current rendering."""
        with self._lock:
            if not self.is_running():
                return
            
            try:
                cmd = ProtocolEncoder.encode_cancel()
                self._send(cmd)
                # Don't wait for response - it may come later
            except Exception as e:
                print(f"[SubprocessRenderer] Cancel error: {e}")
    
    def set_backend(self, backend: BackendType) -> bool:
        """Change rendering backend."""
        with self._lock:
            if not self.is_running():
                return False
            
            try:
                cmd = ProtocolEncoder.encode_set_backend(backend.value)
                self._send(cmd)
                
                resp_type, status, _ = self._recv_header()
                return resp_type == ResponseType.ACK and status == StatusCode.OK
                
            except Exception as e:
                print(f"[SubprocessRenderer] Set backend error: {e}")
                return False
    
    def set_algorithm(self, algorithm: AlgorithmType) -> bool:
        """Change path tracing algorithm."""
        with self._lock:
            if not self.is_running():
                return False
            
            try:
                cmd = ProtocolEncoder.encode_set_algorithm(algorithm.value)
                self._send(cmd)
                
                resp_type, status, _ = self._recv_header()
                return resp_type == ResponseType.ACK and status == StatusCode.OK
                
            except Exception as e:
                print(f"[SubprocessRenderer] Set algorithm error: {e}")
                return False
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Query renderer capabilities."""
        with self._lock:
            if not self.is_running():
                return {'backends': [], 'algorithms': []}
            
            try:
                cmd = ProtocolEncoder.encode_query_caps()
                self._send(cmd)
                
                resp_type, status, payload_size = self._recv_header()
                
                if resp_type != ResponseType.CAPABILITIES:
                    return {'backends': [], 'algorithms': []}
                
                payload = self._recv(payload_size)
                return ProtocolDecoder.decode_capabilities(payload)
                
            except Exception as e:
                print(f"[SubprocessRenderer] Query caps error: {e}")
                return {'backends': [], 'algorithms': []}
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except Exception:
            pass
