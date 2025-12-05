"""
External renderer interface - communicates with C++ path tracer.
"""

import os
import subprocess
import math
import threading
import array

import bpy
from mathutils import Vector


def _get_prefs():
    """Get addon preferences object"""
    entry = bpy.context.preferences.addons.get("DIYRenderer")
    if entry is not None:
        return getattr(entry, 'preferences', None)
    return None


def find_external_binary():
    """Find the external C++ renderer binary."""
    # 1) Explicit preference path
    prefs = _get_prefs()
    if prefs:
        renderer_path = getattr(prefs, 'external_renderer_path', '')
        if renderer_path and isinstance(renderer_path, str) and os.path.isfile(renderer_path):
            return renderer_path
    # 2) Environment variable override
    env_path = os.environ.get('DIY_RENDERER_BIN')
    if env_path and os.path.isfile(env_path):
        return env_path
    # 3) Common relative build locations
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
    # 4) If not found, emit debug info once
    if not hasattr(find_external_binary, '_warned'):
        find_external_binary._warned = True
        print('[DIYRenderer] External renderer binary not found. Checked:')
        for c in candidates:
            print('   -', c)
        print('Set Add-on Preferences path or env DIY_RENDERER_BIN.')
    return None


def compute_camera_params(scene, width, height):
    """
    Compute camera parameters for external renderer.
    
    Extracts camera properties from Blender's camera object and converts
    them to the format expected by the C++ renderer.
    """
    cam = scene.camera
    if not cam:
        print("[DIYRenderer] WARNING: No camera in scene!")
        return None
    
    cam_matrix = cam.matrix_world
    pos = cam_matrix.translation
    
    forward = cam_matrix.to_3x3() @ Vector((0, 0, -1))
    up = cam_matrix.to_3x3() @ Vector((0, 1, 0))
    forward.normalize()
    up.normalize()
    
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
    Call external C++ renderer with cancellation support.
    
    This function invokes the C++ path tracer as a subprocess and handles:
    - Argument passing (scene, camera, samples, depth, mode)
    - Progress monitoring with cancellation support
    - Output parsing (pixel data from stdout)
    
    Returns:
        list: Flat pixel data [r,g,b,a, r,g,b,a, ...] in linear color space
        None: If rendering failed or was cancelled
    """
    import time
    
    binary = find_external_binary()
    if not binary:
        print("[DIYRenderer] External binary not found. Falling back to internal rendering.")
        return None
    
    render_mode = debug_mode if debug_mode else mode
    
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
    
    if pass_id >= 0:
        cmd.extend(['--pass', str(pass_id), '--num-passes', str(num_passes)])
    
    try:
        timing_start = time.perf_counter()
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
        
        timing_after_popen = time.perf_counter()
        
        stdout_data = bytearray()
        stderr_chunks = []
        
        def read_stdout():
            nonlocal stdout_data
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                stdout_data.extend(chunk)
        
        def read_stderr():
            for line in proc.stderr:
                stderr_chunks.append(line.decode('utf-8', errors='replace'))
        
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        while True:
            retcode = proc.poll()
            if retcode is not None:
                break
            
            if cancel_check and cancel_check():
                proc.kill()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                return None
            
            time.sleep(0.005)
        
        timing_after_render = time.perf_counter()
        
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        
        timing_after_threads = time.perf_counter()
        
        stderr = ''.join(stderr_chunks)
        
        if proc.returncode != 0:
            print(f"[DIYRenderer] External renderer failed with code {proc.returncode}")
            if stderr:
                print(f"[DIYRenderer] stderr: {stderr[-2000:]}")
            return None
        
        expected_pixels = tile_w * tile_h
        expected_bytes = expected_pixels * 4 * 4
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
