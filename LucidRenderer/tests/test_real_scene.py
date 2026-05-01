#!/usr/bin/env python3
"""
Test with actual Blender scene file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import threading
import time

from protocol import (
    ResponseType, StatusCode,
    ProtocolEncoder, ProtocolDecoder, 
    CameraParams, RenderTileParams, RESPONSE_HEADER_SIZE
)


SCENE_FILE = "/var/folders/r0/v4p7sf1x7rbf6bvvcj37slk80000gn/T/diy_scene_debug.json"
SERVER_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'cpp_renderer', 'build', 'diyrt')


def recv_response(proc, timeout=30.0):
    """Receive a response from the server."""
    # Read header (must read all 16 bytes)
    header = b''
    while len(header) < RESPONSE_HEADER_SIZE:
        chunk = proc.stdout.read(RESPONSE_HEADER_SIZE - len(header))
        if not chunk:
            raise RuntimeError("Connection closed while reading header")
        header += chunk
    
    resp_type, status, payload_size = ProtocolDecoder.decode_header(header)
    
    # Read full payload
    payload = b''
    while len(payload) < payload_size:
        chunk = proc.stdout.read(payload_size - len(payload))
        if not chunk:
            raise RuntimeError(f"Connection closed while reading payload ({len(payload)}/{payload_size})")
        payload += chunk
    
    return resp_type, status, payload


def main():
    print(f"Server: {SERVER_BIN}")
    print(f"Scene: {SCENE_FILE}")
    
    # Load scene
    with open(SCENE_FILE, 'r') as f:
        scene_json = f.read()
    print(f"Scene size: {len(scene_json)} bytes ({len(scene_json)/1024:.1f} KB)")
    
    # Start server
    proc = subprocess.Popen(
        [SERVER_BIN, '--server'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    
    def read_stderr():
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            print(f"  [C++] {line.decode('utf-8', errors='replace').rstrip()}")
    
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    time.sleep(0.1)
    
    try:
        # INIT
        print("\n1. INIT")
        cmd = ProtocolEncoder.encode_init('cpu', 'nee')
        proc.stdin.write(cmd)
        proc.stdin.flush()
        resp_type, status, _ = recv_response(proc)
        print(f"   Response: {resp_type}, {status}")
        assert resp_type == ResponseType.ACK
        
        # UPDATE_SCENE with real file
        print(f"\n2. UPDATE_SCENE ({len(scene_json)} bytes)")
        scene_bytes = scene_json.encode('utf-8')
        cmd = ProtocolEncoder.encode_update_scene(scene_bytes)
        print(f"   Command size: {len(cmd)} bytes")
        
        # Send in thread
        send_error = [None]
        send_done = threading.Event()
        
        def send_chunked():
            try:
                CHUNK = 65536
                offset = 0
                while offset < len(cmd):
                    chunk = cmd[offset:offset+CHUNK]
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
                    offset += len(chunk)
                    pct = 100 * offset / len(cmd)
                    print(f"   Sent {offset}/{len(cmd)} bytes ({pct:.0f}%)")
                send_done.set()
            except Exception as e:
                send_error[0] = e
                send_done.set()
        
        send_thread = threading.Thread(target=send_chunked, daemon=True)
        send_thread.start()
        
        if not send_done.wait(timeout=60):
            print("   TIMEOUT!")
            return 1
        
        if send_error[0]:
            print(f"   ERROR: {send_error[0]}")
            return 1
        
        print("   Waiting for response...")
        resp_type, status, _ = recv_response(proc)
        print(f"   Response: {resp_type}, {status}")
        
        if resp_type != ResponseType.ACK:
            print("   FAILED!")
            return 1
        
        # UPDATE_CAMERA
        print("\n3. UPDATE_CAMERA")
        cam = CameraParams(
            pos_x=0.0, pos_y=-4.0, pos_z=0.0,
            dir_x=0.0, dir_y=1.0, dir_z=0.0,
            up_x=0.0, up_y=0.0, up_z=1.0,
            fov=39.6
        )
        cmd = ProtocolEncoder.encode_update_camera(cam)
        proc.stdin.write(cmd)
        proc.stdin.flush()
        resp_type, status, _ = recv_response(proc)
        print(f"   Response: {resp_type}, {status}")
        assert resp_type == ResponseType.ACK
        
        # RENDER_TILE
        print("\n4. RENDER_TILE (100x100, 1 sample)")
        params = RenderTileParams(
            tile_x=0, tile_y=0, tile_w=100, tile_h=100,
            full_w=100, full_h=100, samples=1, sample_offset=0, max_depth=8
        )
        cmd = ProtocolEncoder.encode_render_tile(params)
        proc.stdin.write(cmd)
        proc.stdin.flush()
        
        start = time.time()
        resp_type, status, payload = recv_response(proc)
        elapsed = time.time() - start
        print(f"   Response: {resp_type}, payload={len(payload)} bytes, time={elapsed*1000:.1f}ms")
        
        if resp_type == ResponseType.PIXELS:
            pixels = ProtocolDecoder.decode_pixels(payload, 100, 100)
            # Find center pixel
            center = 50 * 100 + 50
            r, g, b, a = pixels[center*4], pixels[center*4+1], pixels[center*4+2], pixels[center*4+3]
            print(f"   Center pixel: ({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})")
        
        print("\n✓ All tests passed with real scene!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        proc.terminate()
        proc.wait(timeout=2)


if __name__ == '__main__':
    sys.exit(main())
