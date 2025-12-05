"""
Binary protocol for Python <-> C++ renderer communication.

Protocol format:
  Command:  [Magic 4B][CmdType 4B][PayloadSize 4B][Payload...]
  Response: [Magic 4B][RespType 4B][Status 4B][PayloadSize 4B][Payload...]
"""

import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
import array


# Protocol constants
PROTOCOL_MAGIC = 0x44495952  # "DIYR" in hex
HEADER_SIZE = 12  # Magic + Type + Size
RESPONSE_HEADER_SIZE = 16  # Magic + Type + Status + Size


class CommandType(IntEnum):
    """Commands sent from Python to C++."""
    INIT = 0x01
    UPDATE_SCENE = 0x02
    UPDATE_CAMERA = 0x03
    RENDER_TILE = 0x04
    CANCEL = 0x05
    QUERY_CAPS = 0x06
    SET_BACKEND = 0x07
    SET_ALGORITHM = 0x08
    SHUTDOWN = 0xFF


class ResponseType(IntEnum):
    """Responses sent from C++ to Python."""
    ACK = 0x81
    PIXELS = 0x82
    PROGRESS = 0x83
    CAPABILITIES = 0x84
    ERROR = 0x85


class StatusCode(IntEnum):
    """Response status codes."""
    OK = 0
    ERROR_UNKNOWN = 1
    ERROR_INVALID_COMMAND = 2
    ERROR_SCENE_NOT_LOADED = 3
    ERROR_RENDER_FAILED = 4
    ERROR_CANCELLED = 5


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    pos_x: float
    pos_y: float
    pos_z: float
    dir_x: float
    dir_y: float
    dir_z: float
    up_x: float
    up_y: float
    up_z: float
    fov: float
    
    def to_bytes(self) -> bytes:
        """Serialize to binary (40 bytes)."""
        return struct.pack('10f',
            self.pos_x, self.pos_y, self.pos_z,
            self.dir_x, self.dir_y, self.dir_z,
            self.up_x, self.up_y, self.up_z,
            self.fov
        )
    
    @classmethod
    def from_blender(cls, cam_params: dict) -> 'CameraParams':
        """Create from Blender camera params dict."""
        return cls(
            pos_x=cam_params['pos'].x,
            pos_y=cam_params['pos'].y,
            pos_z=cam_params['pos'].z,
            dir_x=cam_params['dir'].x,
            dir_y=cam_params['dir'].y,
            dir_z=cam_params['dir'].z,
            up_x=cam_params['up'].x,
            up_y=cam_params['up'].y,
            up_z=cam_params['up'].z,
            fov=cam_params['fov']
        )


@dataclass
class RenderTileParams:
    """Parameters for tile rendering request."""
    tile_x: int
    tile_y: int
    tile_w: int
    tile_h: int
    full_w: int
    full_h: int
    samples: int
    sample_offset: int
    max_depth: int
    
    def to_bytes(self) -> bytes:
        """Serialize to binary (36 bytes)."""
        return struct.pack('9I',
            self.tile_x, self.tile_y, self.tile_w, self.tile_h,
            self.full_w, self.full_h,
            self.samples, self.sample_offset, self.max_depth
        )


@dataclass
class InitParams:
    """Initialization parameters."""
    backend: str  # "cpu" or "webgpu"
    algorithm: str  # "naive", "nee", "mis"
    
    def to_bytes(self) -> bytes:
        """Serialize to binary."""
        backend_bytes = self.backend.encode('utf-8')
        algo_bytes = self.algorithm.encode('utf-8')
        return struct.pack('I', len(backend_bytes)) + backend_bytes + \
               struct.pack('I', len(algo_bytes)) + algo_bytes


class ProtocolEncoder:
    """Encode commands to binary protocol."""
    
    @staticmethod
    def _make_header(cmd_type: CommandType, payload_size: int) -> bytes:
        """Create command header."""
        return struct.pack('III', PROTOCOL_MAGIC, int(cmd_type), payload_size)
    
    @staticmethod
    def encode_init(backend: str = "cpu", algorithm: str = "nee") -> bytes:
        """Encode INIT command."""
        params = InitParams(backend, algorithm)
        payload = params.to_bytes()
        return ProtocolEncoder._make_header(CommandType.INIT, len(payload)) + payload
    
    @staticmethod
    def encode_update_scene(scene_json: bytes) -> bytes:
        """Encode UPDATE_SCENE command with JSON scene data."""
        return ProtocolEncoder._make_header(CommandType.UPDATE_SCENE, len(scene_json)) + scene_json
    
    @staticmethod
    def encode_update_camera(cam_params: CameraParams) -> bytes:
        """Encode UPDATE_CAMERA command."""
        payload = cam_params.to_bytes()
        return ProtocolEncoder._make_header(CommandType.UPDATE_CAMERA, len(payload)) + payload
    
    @staticmethod
    def encode_render_tile(params: RenderTileParams) -> bytes:
        """Encode RENDER_TILE command."""
        payload = params.to_bytes()
        return ProtocolEncoder._make_header(CommandType.RENDER_TILE, len(payload)) + payload
    
    @staticmethod
    def encode_cancel() -> bytes:
        """Encode CANCEL command."""
        return ProtocolEncoder._make_header(CommandType.CANCEL, 0)
    
    @staticmethod
    def encode_query_caps() -> bytes:
        """Encode QUERY_CAPS command."""
        return ProtocolEncoder._make_header(CommandType.QUERY_CAPS, 0)
    
    @staticmethod
    def encode_set_backend(backend: str) -> bytes:
        """Encode SET_BACKEND command."""
        payload = backend.encode('utf-8')
        return ProtocolEncoder._make_header(CommandType.SET_BACKEND, len(payload)) + payload
    
    @staticmethod
    def encode_set_algorithm(algorithm: str) -> bytes:
        """Encode SET_ALGORITHM command."""
        payload = algorithm.encode('utf-8')
        return ProtocolEncoder._make_header(CommandType.SET_ALGORITHM, len(payload)) + payload
    
    @staticmethod
    def encode_shutdown() -> bytes:
        """Encode SHUTDOWN command."""
        return ProtocolEncoder._make_header(CommandType.SHUTDOWN, 0)


class ProtocolDecoder:
    """Decode responses from binary protocol."""
    
    @staticmethod
    def decode_header(data: bytes) -> Tuple[ResponseType, StatusCode, int]:
        """
        Decode response header.
        
        Returns: (response_type, status, payload_size)
        Raises: ValueError if invalid header
        """
        if len(data) < RESPONSE_HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} bytes")
        
        magic, resp_type, status, payload_size = struct.unpack('IIII', data[:RESPONSE_HEADER_SIZE])
        
        if magic != PROTOCOL_MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{PROTOCOL_MAGIC:08X}")
        
        return ResponseType(resp_type), StatusCode(status), payload_size
    
    @staticmethod
    def decode_pixels(payload: bytes, width: int, height: int) -> list:
        """
        Decode pixel data from PIXELS response.
        
        Returns: Flat list [r,g,b,a, r,g,b,a, ...] of float values
        """
        expected_size = width * height * 4 * 4  # 4 floats per pixel, 4 bytes per float
        if len(payload) != expected_size:
            raise ValueError(f"Pixel data size mismatch: {len(payload)} bytes, expected {expected_size}")
        
        float_array = array.array('f')
        float_array.frombytes(payload)
        return float_array.tolist()
    
    @staticmethod
    def decode_progress(payload: bytes) -> float:
        """Decode progress percentage from PROGRESS response."""
        if len(payload) != 4:
            raise ValueError(f"Progress payload size mismatch: {len(payload)}")
        return struct.unpack('f', payload)[0]
    
    @staticmethod
    def decode_capabilities(payload: bytes) -> dict:
        """Decode capabilities from CAPABILITIES response."""
        # Format: [num_backends:4][backend1_len:4][backend1:...][...][num_algos:4][algo1_len:4][algo1:...]...
        offset = 0
        
        def read_string_list():
            nonlocal offset
            count = struct.unpack_from('I', payload, offset)[0]
            offset += 4
            items = []
            for _ in range(count):
                length = struct.unpack_from('I', payload, offset)[0]
                offset += 4
                items.append(payload[offset:offset+length].decode('utf-8'))
                offset += length
            return items
        
        backends = read_string_list()
        algorithms = read_string_list()
        
        return {
            'backends': backends,
            'algorithms': algorithms
        }
    
    @staticmethod
    def decode_error(payload: bytes) -> str:
        """Decode error message from ERROR response."""
        return payload.decode('utf-8')
