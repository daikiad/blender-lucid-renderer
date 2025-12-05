"""
Subprocess-based renderer implementation.

Uses stdin/stdout binary protocol to communicate with a persistent
C++ renderer process. This avoids process startup and scene loading
overhead on each frame.
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
    """Find the external C++ renderer binary."""
    # Try environment variable first
    env_path = os.environ.get('DIY_RENDERER_BIN')
    if env_path and os.path.isfile(env_path):
        return env_path
    
    # Common relative build locations
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
    Renderer implementation using subprocess with binary protocol.
    
    The renderer runs as a persistent server process, accepting commands
    via stdin and returning results via stdout.
    """
    
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._config: Optional[RenderConfig] = None
        self._lock = threading.Lock()
        self._binary_path: Optional[str] = None
        
    def start(self, config: RenderConfig) -> bool:
        """Start the renderer server process."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                # Already running
                return True
            
            binary = find_renderer_binary()
            if not binary:
                print("[SubprocessRenderer] Renderer binary not found")
                return False
            
            self._binary_path = binary
            self._config = config
            
            try:
                self._process = subprocess.Popen(
                    [binary, '--server'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0  # Unbuffered
                )
                
                # Start stderr reader thread
                self._stderr_thread = threading.Thread(
                    target=self._read_stderr,
                    daemon=True
                )
                self._stderr_thread.start()
                
                # Send init command
                init_cmd = ProtocolEncoder.encode_init(
                    backend=config.backend.value,
                    algorithm=config.algorithm.value
                )
                self._send(init_cmd)
                
                # Wait for ACK
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
        """Background thread to read stderr."""
        try:
            while self._process and self._process.poll() is None:
                line = self._process.stderr.readline()
                if line:
                    print(f"[C++] {line.decode('utf-8', errors='replace').rstrip()}")
        except Exception:
            pass
    
    def is_running(self) -> bool:
        """Check if the renderer is running."""
        # Note: Does not acquire lock - caller should hold lock if needed
        return self._process is not None and self._process.poll() is None
    
    def _is_running_locked(self) -> bool:
        """Check if the renderer is running (acquires lock)."""
        with self._lock:
            return self._process is not None and self._process.poll() is None
    
    def stop(self) -> None:
        """Stop the renderer process."""
        with self._lock:
            if self._process is None:
                return
            
            try:
                # Send shutdown command
                shutdown_cmd = ProtocolEncoder.encode_shutdown()
                self._send(shutdown_cmd)
                
                # Wait for process to exit
                self._process.wait(timeout=2.0)
            except Exception as e:
                print(f"[SubprocessRenderer] Error during shutdown: {e}")
                try:
                    self._process.kill()
                except Exception:
                    pass
            
            self._process = None
            print("[SubprocessRenderer] Stopped")
    
    def _send(self, data: bytes) -> None:
        """Send data to the renderer."""
        if self._process is None or self._process.stdin is None:
            raise RendererError("Renderer not running")
        self._process.stdin.write(data)
        self._process.stdin.flush()
    
    def _recv(self, size: int) -> bytes:
        """Receive data from the renderer."""
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
        """Receive and decode response header."""
        header_data = self._recv(RESPONSE_HEADER_SIZE)
        return ProtocolDecoder.decode_header(header_data)
    
    def update_scene(self, scene_json: str) -> bool:
        """Send scene data to renderer."""
        with self._lock:
            if not self.is_running():
                return False
            
            try:
                scene_bytes = scene_json.encode('utf-8')
                cmd = ProtocolEncoder.encode_update_scene(scene_bytes)
                print(f"[SubprocessRenderer] Sending scene: {len(cmd)} bytes total ({len(scene_bytes)} payload)")
                
                # Send in a separate thread to avoid pipe buffer deadlock
                import threading
                send_error = [None]
                send_done = threading.Event()
                
                def send_chunked():
                    try:
                        CHUNK = 65536
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
                
                # Wait for send to complete
                if not send_done.wait(timeout=60.0):
                    print("[SubprocessRenderer] Send timeout!")
                    return False
                
                if send_error[0]:
                    print(f"[SubprocessRenderer] Send error: {send_error[0]}")
                    return False
                
                print("[SubprocessRenderer] Send complete, reading response...")
                
                # Read response
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
        """Send camera update to renderer."""
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
        """Request tile rendering."""
        with self._lock:
            if not self.is_running():
                return None
            
            try:
                start_time = time.perf_counter()
                
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
                
                cmd = ProtocolEncoder.encode_render_tile(params)
                self._send(cmd)
                
                resp_type, status, payload_size = self._recv_header()
                
                if resp_type == ResponseType.ERROR:
                    error_msg = self._recv(payload_size).decode('utf-8') if payload_size > 0 else "Unknown error"
                    print(f"[SubprocessRenderer] Render failed: {error_msg}")
                    return None
                
                if resp_type != ResponseType.PIXELS:
                    print(f"[SubprocessRenderer] Unexpected response: {resp_type}")
                    return None
                
                # Receive pixel data
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
