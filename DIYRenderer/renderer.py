"""
External renderer interface - communicates with C++ path tracer.
================================================================

このモジュールは、レガシーモード（非サーバーモード）での
C++ レンダラーとの通信を担当します。

レガシーモード vs サーバーモード:
- レガシーモード（このファイル）: 毎回プロセスを起動
- サーバーモード（subprocess_renderer.py）: プロセスを持続

レガシーモードの処理フロー:
1. find_external_binary() でレンダラーバイナリを検索
2. compute_camera_params() でカメラ情報を抽出
3. call_external_renderer() でサブプロセスを起動
4. コマンドライン引数でパラメータを渡す
5. stdout からバイナリピクセルデータを読み取り
6. プロセス終了

主要関数:
- find_external_binary(): レンダラーバイナリのパスを検索
- compute_camera_params(): Blender カメラからパラメータを計算
- call_external_renderer(): サブプロセスでレンダリングを実行

注意:
サーバーモードが有効な場合、このモジュールは使用されません。
engine.py で use_server_mode フラグをチェックして分岐します。
"""

import os
import subprocess
import math
import threading
import array

import bpy
from mathutils import Vector


def _get_prefs():
    """
    アドオン設定を取得します。
    
    Returns:
        DIYRendererPreferences オブジェクト、または None
    """
    entry = bpy.context.preferences.addons.get("DIYRenderer")
    if entry is not None:
        return getattr(entry, 'preferences', None)
    return None


def find_external_binary():
    """
    外部 C++ レンダラーバイナリを検索します。
    
    検索順序:
    1. アドオン設定で指定されたパス
    2. 環境変数 DIY_RENDERER_BIN
    3. 一般的なビルドディレクトリ
    
    Returns:
        str: バイナリのパス、または見つからない場合 None
    """
    # 1) アドオン設定で明示的に指定されたパス
    prefs = _get_prefs()
    if prefs:
        renderer_path = getattr(prefs, 'external_renderer_path', '')
        if renderer_path and isinstance(renderer_path, str) and os.path.isfile(renderer_path):
            return renderer_path
    
    # 2) 環境変数によるオーバーライド
    env_path = os.environ.get('DIY_RENDERER_BIN')
    if env_path and os.path.isfile(env_path):
        return env_path
    
    # 3) 一般的な相対ビルドパス
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(addon_dir, '..'))
    candidates = [
        os.path.join(project_root, 'DIYRenderer', 'cpp_renderer', 'build', 'diyrt'),
        os.path.join(project_root, 'DIYRenderer', 'cpp_renderer', 'build', 'Release', 'diyrt'),
        os.path.join(project_root, 'DIYRenderer', 'cpp_renderer', 'build', 'Debug', 'diyrt'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    
    # 4) 見つからない場合は警告を出力（一度だけ）
    if not hasattr(find_external_binary, '_warned'):
        find_external_binary._warned = True
        print('[DIYRenderer] External renderer binary not found. Checked:')
        for c in candidates:
            print('   -', c)
        print('Set Add-on Preferences path or env DIY_RENDERER_BIN.')
    return None


def compute_camera_params(scene, width, height):
    """
    外部レンダラー用のカメラパラメータを計算します。
    
    Blender のカメラオブジェクトから以下を抽出:
    - 位置（ワールド座標）
    - 視線方向（正規化ベクトル）
    - 上方向ベクトル（正規化）
    - 視野角（度）
    
    Args:
        scene: Blender シーン
        width: レンダリング幅（FOV計算に影響しない）
        height: レンダリング高さ（FOV計算に影響しない）
    
    Returns:
        dict: {'pos': Vector, 'dir': Vector, 'up': Vector, 'fov': float}
        または、カメラがない場合 None
    """
    cam = scene.camera
    if not cam:
        print("[DIYRenderer] WARNING: No camera in scene!")
        return None
    
    # カメラのワールド変換行列から位置と方向を取得
    cam_matrix = cam.matrix_world
    pos = cam_matrix.translation
    
    # カメラのローカル -Z がワールド空間の視線方向
    # カメラのローカル +Y がワールド空間の上方向
    forward = cam_matrix.to_3x3() @ Vector((0, 0, -1))
    up = cam_matrix.to_3x3() @ Vector((0, 1, 0))
    forward.normalize()
    up.normalize()
    
    # 視野角を計算: FOV = 2 * atan(sensor_width / (2 * focal_length))
    sensor_w = cam.data.sensor_width
    lens = cam.data.lens
    fov_rad = 2.0 * math.atan(sensor_w / (2.0 * lens))
    fov_deg = math.degrees(fov_rad)
    
    print(f"[DIYRenderer] Camera: pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
          f"dir=({forward.x:.3f}, {forward.y:.3f}, {forward.z:.3f}), "
          f"up=({up.x:.3f}, {up.y:.3f}, {up.z:.3f}), fov={fov_deg:.1f}")
    return {
        'pos': pos,
        'dir': forward,
        'up': up,
        'fov': fov_deg
    }


def call_external_renderer(scene_file, tile_x, tile_y, tile_w, tile_h, full_w, full_h, 
                           cam_params, mode='raytrace', samples=1, depth=8, 
                           debug_mode=None, cancel_check=None, sample_offset=0, 
                           algorithm='nee', pass_id=-1, num_passes=16):
    """
    外部 C++ レンダラーをサブプロセスとして呼び出します（キャンセル対応）。
    
    この関数は C++ パストレーサーをサブプロセスとして起動し、以下を処理します:
    - 引数渡し（シーン、カメラ、サンプル数、バウンス数、モード）
    - キャンセル対応の進捗監視
    - 出力のパース（stdout からピクセルデータ）
    
    Args:
        scene_file: JSON シーンファイルのパス
        tile_x, tile_y: タイルの左上座標
        tile_w, tile_h: タイルのサイズ
        full_w, full_h: 画像全体のサイズ
        cam_params: カメラパラメータ辞書
        mode: レンダリングモード（'raytrace' など）
        samples: サンプル数
        depth: 最大バウンス数
        debug_mode: デバッグモード（'normal', 'albedo', 'emission' など）
        cancel_check: キャンセルチェック関数（None または callable）
        sample_offset: サンプルオフセット（累積用）
        algorithm: アルゴリズム（'simple', 'nee', 'mis'）
        pass_id: パス ID（プログレッシブ用）
        num_passes: 総パス数（プログレッシブ用）
    
    Returns:
        list: フラットなピクセルデータ [r,g,b,a, r,g,b,a, ...] (リニア色空間)
        None: レンダリング失敗またはキャンセル時
    """
    import time
    
    # バイナリを検索
    binary = find_external_binary()
    if not binary:
        print("[DIYRenderer] External binary not found. Falling back to internal rendering.")
        return None
    
    # デバッグモードがあればそれを使用
    render_mode = debug_mode if debug_mode else mode
    
    # コマンドライン引数を構築
    cmd = [binary,
           '--scene', scene_file,
           '--tile', str(tile_x), str(tile_y), str(tile_w), str(tile_h),
           '--full', str(full_w), str(full_h),
           '--campos', str(cam_params['pos'].x), str(cam_params['pos'].y), str(cam_params['pos'].z),
           '--camdir', str(cam_params['dir'].x), str(cam_params['dir'].y), str(cam_params['dir'].z),
           '--camup', str(cam_params['up'].x), str(cam_params['up'].y), str(cam_params['up'].z),
           '--fov', str(cam_params['fov']),
           '--samples', str(samples),
           '--depth', str(depth),
           '--sample-offset', str(sample_offset),
           '--algorithm', algorithm,
           '--mode', render_mode]
    
    # プログレッシブレンダリング用のパラメータ
    if pass_id >= 0:
        cmd.extend(['--pass', str(pass_id), '--num-passes', str(num_passes)])
    
    try:
        timing_start = time.perf_counter()
        
        # サブプロセスを起動
        # bufsize=-1: システムデフォルトのバッファサイズを使用
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
        
        timing_after_popen = time.perf_counter()
        
        # 出力を非同期で読み取り（デッドロック防止）
        stdout_data = bytearray()
        stderr_chunks = []
        
        def read_stdout():
            """stdout を読み取るスレッド"""
            nonlocal stdout_data
            while True:
                chunk = proc.stdout.read(65536)  # 64KB チャンク
                if not chunk:
                    break
                stdout_data.extend(chunk)
        
        def read_stderr():
            """stderr を読み取るスレッド"""
            for line in proc.stderr:
                stderr_chunks.append(line.decode('utf-8', errors='replace'))
        
        # 読み取りスレッドを開始
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # プロセス終了を待機（キャンセルチェック付き）
        while True:
            retcode = proc.poll()
            if retcode is not None:
                break
            
            # キャンセルされた場合はプロセスを終了
            if cancel_check and cancel_check():
                proc.kill()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                return None
            
            time.sleep(0.005)  # 5ms スリープ
        
        timing_after_render = time.perf_counter()
        
        # スレッド終了を待機
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        
        timing_after_threads = time.perf_counter()
        
        stderr = ''.join(stderr_chunks)
        
        # エラーチェック
        if proc.returncode != 0:
            print(f"[DIYRenderer] External renderer failed with code {proc.returncode}")
            if stderr:
                print(f"[DIYRenderer] stderr: {stderr[-2000:]}")
            return None
        
        # 出力サイズ検証
        expected_pixels = tile_w * tile_h
        expected_bytes = expected_pixels * 4 * 4  # RGBA × float32
        if len(stdout_data) != expected_bytes:
            print(f"[DIYRenderer] ERROR: Unexpected binary size: {len(stdout_data)} bytes, expected {expected_bytes}")
            if stderr:
                print(f"[DIYRenderer] stderr: {stderr[-1000:]}")
            return None
            
    except Exception as e:
        print('[DIYRenderer] External renderer invocation failed:', e)
        return None
    
    timing_before_parse = time.perf_counter()
    
    float_array = array.array('f')
    float_array.frombytes(stdout_data)
    
    flipped_pixels = float_array.tolist()
    
    timing_end = time.perf_counter()
    
    popen_time = (timing_after_popen - timing_start) * 1000
    render_time = (timing_after_render - timing_after_popen) * 1000
    thread_time = (timing_after_threads - timing_after_render) * 1000
    parse_time = (timing_end - timing_before_parse) * 1000
    total_time = (timing_end - timing_start) * 1000
    
    print(f"[DIYRenderer] Timing {tile_w}x{tile_h}: popen={popen_time:.1f}ms, C++={render_time:.1f}ms, threads={thread_time:.1f}ms, parse={parse_time:.1f}ms, TOTAL={total_time:.1f}ms")
    
    return flipped_pixels


def linear_to_srgb(c):
    """Convert linear color value to sRGB gamma corrected value."""
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * (c ** (1.0/2.4)) - 0.055


def apply_gamma_correction(pixels):
    """Apply sRGB gamma correction to linear pixel values."""
    corrected = []
    for pixel in pixels:
        r, g, b, a = pixel
        r_srgb = linear_to_srgb(max(0.0, min(1.0, r)))
        g_srgb = linear_to_srgb(max(0.0, min(1.0, g)))
        b_srgb = linear_to_srgb(max(0.0, min(1.0, b)))
        corrected.append([r_srgb, g_srgb, b_srgb, a])
    return corrected
