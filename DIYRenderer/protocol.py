"""
Binary protocol for Python <-> C++ renderer communication.
===========================================================

このモジュールは Python（Blenderアドオン）と C++（レンダラー）間の
バイナリ通信プロトコルを定義しています。

プロトコル概要:
==============
stdin/stdout を介してバイナリデータを送受信します。
テキストではなくバイナリを使用することで、効率的なデータ転送が可能です。

フォーマット:
============
Command (Python → C++):
    [Magic 4B][CmdType 4B][PayloadSize 4B][Payload...]
    - Magic: 0x44495952 ("DIYR") - パケット識別用マジックナンバー
    - CmdType: コマンドタイプ (INIT, UPDATE_SCENE, RENDER_TILE など)
    - PayloadSize: ペイロードのバイト数
    - Payload: コマンド固有のデータ

Response (C++ → Python):
    [Magic 4B][RespType 4B][Status 4B][PayloadSize 4B][Payload...]
    - Magic: 0x44495952
    - RespType: レスポンスタイプ (ACK, PIXELS, ERROR など)
    - Status: ステータスコード (OK=0, ERROR=1, ...)
    - PayloadSize: ペイロードのバイト数
    - Payload: レスポンス固有のデータ

使用例:
======
# コマンド送信
encoder = ProtocolEncoder()
init_cmd = encoder.encode_init(backend="cpu", algorithm="nee")
process.stdin.write(init_cmd)

# レスポンス受信
header_data = process.stdout.read(RESPONSE_HEADER_SIZE)
resp_type, status, payload_size = ProtocolDecoder.decode_header(header_data)
"""

import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
import array


# =============================================================================
# プロトコル定数
# =============================================================================

# マジックナンバー: "DIYR" (DIY Renderer) をASCIIコードで表現
# パケットの開始を識別し、不正なデータを検出するために使用
PROTOCOL_MAGIC = 0x44495952  # 'D'=0x44, 'I'=0x49, 'Y'=0x59, 'R'=0x52

# ヘッダーサイズ（バイト）
HEADER_SIZE = 12          # コマンドヘッダー: Magic + Type + Size
RESPONSE_HEADER_SIZE = 16  # レスポンスヘッダー: Magic + Type + Status + Size


# =============================================================================
# 列挙型定義
# =============================================================================

class CommandType(IntEnum):
    """
    Python から C++ に送信するコマンドタイプ。
    
    各コマンドはレンダラーに対する操作を表します。
    サーバーモードでは、これらのコマンドを順次送信して
    レンダリングを制御します。
    """
    INIT = 0x01           # レンダラー初期化（バックエンド、アルゴリズム設定）
    UPDATE_SCENE = 0x02   # シーンデータ更新（JSON形式のメッシュ/マテリアル）
    UPDATE_CAMERA = 0x03  # カメラパラメータ更新（位置、方向、FOV）
    RENDER_TILE = 0x04    # タイルレンダリング実行
    CANCEL = 0x05         # 現在のレンダリングをキャンセル
    QUERY_CAPS = 0x06     # レンダラーの機能を問い合わせ
    SET_BACKEND = 0x07    # バックエンド変更（CPU/WebGPU）
    SET_ALGORITHM = 0x08  # アルゴリズム変更（Simple/NEE/MIS）
    SHUTDOWN = 0xFF       # レンダラー終了


class ResponseType(IntEnum):
    """
    C++ から Python に送信するレスポンスタイプ。
    
    各レスポンスは操作の結果を表します。
    """
    ACK = 0x81          # 確認応答（コマンド受理）
    PIXELS = 0x82       # ピクセルデータ（レンダリング結果）
    PROGRESS = 0x83     # 進捗情報（将来の拡張用）
    CAPABILITIES = 0x84  # 機能一覧（QUERY_CAPSへの応答）
    ERROR = 0x85        # エラー（エラーメッセージ付き）


class StatusCode(IntEnum):
    """
    レスポンスのステータスコード。
    
    エラーの種類を識別するために使用します。
    """
    OK = 0                    # 成功
    ERROR_UNKNOWN = 1         # 不明なエラー
    ERROR_INVALID_COMMAND = 2  # 無効なコマンド
    ERROR_SCENE_NOT_LOADED = 3 # シーン未ロード
    ERROR_RENDER_FAILED = 4    # レンダリング失敗
    ERROR_CANCELLED = 5        # キャンセルされた


@dataclass
class CameraParams:
    """
    カメラパラメータのデータクラス。
    
    レンダラーに送信するカメラ情報をバイナリ形式で
    シリアライズするためのクラスです。
    
    Attributes:
        pos_x, pos_y, pos_z: カメラ位置（ワールド座標）
        dir_x, dir_y, dir_z: 視線方向（正規化ベクトル）
        up_x, up_y, up_z: 上方向ベクトル（正規化）
        fov: 視野角（度）
    
    バイナリ形式:
        40バイト = 10 × float32
        [pos_x][pos_y][pos_z][dir_x][dir_y][dir_z][up_x][up_y][up_z][fov]
    """
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
        """
        バイナリ形式にシリアライズします。
        
        struct.pack() を使用して float 値を連続したバイト列に変換します。
        '10f' は「10個のfloat (4バイト×10 = 40バイト)」を意味します。
        
        Returns:
            bytes: 40バイトのバイナリデータ
        """
        return struct.pack('10f',
            self.pos_x, self.pos_y, self.pos_z,
            self.dir_x, self.dir_y, self.dir_z,
            self.up_x, self.up_y, self.up_z,
            self.fov
        )
    
    @classmethod
    def from_blender(cls, cam_params: dict) -> 'CameraParams':
        """
        Blender のカメラパラメータ辞書から作成します。
        
        Args:
            cam_params: compute_camera_params() の戻り値
                - 'pos': mathutils.Vector (位置)
                - 'dir': mathutils.Vector (方向)
                - 'up': mathutils.Vector (上方向)
                - 'fov': float (視野角)
        
        Returns:
            CameraParams: プロトコル用のカメラパラメータ
        """
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
    """
    タイルレンダリングパラメータのデータクラス。
    
    レンダリングするタイル領域とサンプリング設定を定義します。
    
    Attributes:
        tile_x, tile_y: タイルの左上座標（ピクセル）
        tile_w, tile_h: タイルのサイズ（ピクセル）
        full_w, full_h: 画像全体のサイズ（ピクセル）
        samples: このパスでのサンプル数
        sample_offset: サンプルオフセット（累積レンダリング用）
        max_depth: 最大レイバウンス数
    
    バイナリ形式:
        36バイト = 9 × uint32
    
    累積レンダリングについて:
        プログレッシブレンダリングでは、複数パスでサンプルを累積します。
        sample_offset はランダムシードのオフセットとして使用され、
        各パスで異なるサンプルを生成します。
        
        例: 64サンプルを7パスで累積
        パス1: samples=1, offset=0
        パス2: samples=2, offset=1
        パス3: samples=4, offset=3
        ...
    """
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
        """
        バイナリ形式にシリアライズします。
        
        '9I' は「9個の unsigned int (4バイト×9 = 36バイト)」を意味します。
        
        Returns:
            bytes: 36バイトのバイナリデータ
        """
        return struct.pack('9I',
            self.tile_x, self.tile_y, self.tile_w, self.tile_h,
            self.full_w, self.full_h,
            self.samples, self.sample_offset, self.max_depth
        )


@dataclass
class InitParams:
    """
    初期化パラメータのデータクラス。
    
    レンダラー起動時の設定を定義します。
    
    Attributes:
        backend: レンダリングバックエンド ("cpu" または "webgpu")
        algorithm: パストレーシングアルゴリズム
            - "simple": シンプルなBSDFサンプリング（参照用）
            - "nee": Next Event Estimation（直接光の高速計算）
            - "mis": Multiple Importance Sampling（最高品質）
    """
    backend: str
    algorithm: str
    
    def to_bytes(self) -> bytes:
        """
        バイナリ形式にシリアライズします。
        
        文字列は [長さ(4バイト)][UTF-8データ] の形式で格納します。
        
        Returns:
            bytes: 可変長のバイナリデータ
        """
        # 文字列をUTF-8でエンコード
        backend_bytes = self.backend.encode('utf-8')
        algo_bytes = self.algorithm.encode('utf-8')
        # [長さ][データ] の形式で連結
        return struct.pack('I', len(backend_bytes)) + backend_bytes + \
               struct.pack('I', len(algo_bytes)) + algo_bytes


class ProtocolEncoder:
    """
    コマンドをバイナリプロトコルにエンコードするクラス。
    
    各 encode_* メソッドは、対応するコマンドを
    [ヘッダー][ペイロード] の形式でバイト列に変換します。
    
    使用例:
        # レンダラー初期化
        cmd = ProtocolEncoder.encode_init("cpu", "nee")
        process.stdin.write(cmd)
        
        # タイルレンダリング
        params = RenderTileParams(0, 0, 640, 480, 640, 480, 64, 0, 8)
        cmd = ProtocolEncoder.encode_render_tile(params)
        process.stdin.write(cmd)
    """
    
    @staticmethod
    def _make_header(cmd_type: CommandType, payload_size: int) -> bytes:
        """
        コマンドヘッダーを作成します。
        
        ヘッダー構造: [Magic 4B][Type 4B][Size 4B] = 12バイト
        
        Args:
            cmd_type: コマンドタイプ
            payload_size: ペイロードのバイト数
        
        Returns:
            bytes: 12バイトのヘッダー
        """
        # 'III' = 3つの unsigned int (4バイト×3)
        return struct.pack('III', PROTOCOL_MAGIC, int(cmd_type), payload_size)
    
    @staticmethod
    def encode_init(backend: str = "cpu", algorithm: str = "nee") -> bytes:
        """
        INIT コマンドをエンコードします。
        
        レンダラーの初期化設定を送信します。
        サーバーモード開始時に最初に送信するコマンドです。
        """
        params = InitParams(backend, algorithm)
        payload = params.to_bytes()
        return ProtocolEncoder._make_header(CommandType.INIT, len(payload)) + payload
    
    @staticmethod
    def encode_update_scene(scene_json: bytes) -> bytes:
        """
        UPDATE_SCENE コマンドをエンコードします。
        
        JSON形式のシーンデータを送信します。
        シーンが変更された場合に送信します（カメラのみの変更時は不要）。
        
        注意: scene_json は bytes 型です。str の場合は .encode('utf-8') が必要です。
        """
        return ProtocolEncoder._make_header(CommandType.UPDATE_SCENE, len(scene_json)) + scene_json
    
    @staticmethod
    def encode_update_camera(cam_params: CameraParams) -> bytes:
        """
        UPDATE_CAMERA コマンドをエンコードします。
        
        カメラパラメータのみを更新します。
        シーンデータは再送信しないため、高速にカメラ移動できます。
        """
        payload = cam_params.to_bytes()
        return ProtocolEncoder._make_header(CommandType.UPDATE_CAMERA, len(payload)) + payload
    
    @staticmethod
    def encode_render_tile(params: RenderTileParams) -> bytes:
        """
        RENDER_TILE コマンドをエンコードします。
        
        指定したタイル領域をレンダリングします。
        レスポンスとして PIXELS（ピクセルデータ）が返されます。
        """
        payload = params.to_bytes()
        return ProtocolEncoder._make_header(CommandType.RENDER_TILE, len(payload)) + payload
    
    @staticmethod
    def encode_cancel() -> bytes:
        """
        CANCEL コマンドをエンコードします。
        
        現在のレンダリング操作をキャンセルします。
        ペイロードは不要です。
        """
        return ProtocolEncoder._make_header(CommandType.CANCEL, 0)
    
    @staticmethod
    def encode_query_caps() -> bytes:
        """
        QUERY_CAPS コマンドをエンコードします。
        
        レンダラーがサポートする機能（バックエンド、アルゴリズム）を問い合わせます。
        レスポンスとして CAPABILITIES が返されます。
        """
        return ProtocolEncoder._make_header(CommandType.QUERY_CAPS, 0)
    
    @staticmethod
    def encode_set_backend(backend: str) -> bytes:
        """
        SET_BACKEND コマンドをエンコードします。
        
        レンダリングバックエンドを動的に変更します。
        """
        payload = backend.encode('utf-8')
        return ProtocolEncoder._make_header(CommandType.SET_BACKEND, len(payload)) + payload
    
    @staticmethod
    def encode_set_algorithm(algorithm: str) -> bytes:
        """
        SET_ALGORITHM コマンドをエンコードします。
        
        パストレーシングアルゴリズムを動的に変更します。
        """
        payload = algorithm.encode('utf-8')
        return ProtocolEncoder._make_header(CommandType.SET_ALGORITHM, len(payload)) + payload
    
    @staticmethod
    def encode_shutdown() -> bytes:
        """
        SHUTDOWN コマンドをエンコードします。
        
        レンダラープロセスを正常終了させます。
        このコマンドの後、プロセスは終了します。
        """
        return ProtocolEncoder._make_header(CommandType.SHUTDOWN, 0)


class ProtocolDecoder:
    """
    バイナリプロトコルからレスポンスをデコードするクラス。
    
    C++ レンダラーから受信したバイナリデータを
    Python オブジェクトに変換します。
    
    使用例:
        # ヘッダー読み取り
        header_data = process.stdout.read(RESPONSE_HEADER_SIZE)
        resp_type, status, payload_size = ProtocolDecoder.decode_header(header_data)
        
        # ペイロード読み取り
        if resp_type == ResponseType.PIXELS:
            payload = process.stdout.read(payload_size)
            pixels = ProtocolDecoder.decode_pixels(payload, width, height)
    """
    
    @staticmethod
    def decode_header(data: bytes) -> Tuple[ResponseType, StatusCode, int]:
        """
        レスポンスヘッダーをデコードします。
        
        ヘッダー構造: [Magic 4B][Type 4B][Status 4B][Size 4B] = 16バイト
        
        Args:
            data: 16バイトのヘッダーデータ
        
        Returns:
            tuple: (レスポンスタイプ, ステータスコード, ペイロードサイズ)
        
        Raises:
            ValueError: ヘッダーが短すぎる、またはマジックナンバーが不正
        """
        if len(data) < RESPONSE_HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} bytes")
        
        # 'IIII' = 4つの unsigned int
        magic, resp_type, status, payload_size = struct.unpack('IIII', data[:RESPONSE_HEADER_SIZE])
        
        # マジックナンバーの検証
        if magic != PROTOCOL_MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{PROTOCOL_MAGIC:08X}")
        
        return ResponseType(resp_type), StatusCode(status), payload_size
    
    @staticmethod
    def decode_pixels(payload: bytes, width: int, height: int) -> list:
        """
        PIXELS レスポンスからピクセルデータをデコードします。
        
        ピクセルデータ形式:
        - 各ピクセル: 4 × float32 (RGBA)
        - 合計バイト数: width × height × 4 × 4
        - 色空間: リニア（sRGBではない）
        
        Args:
            payload: バイナリピクセルデータ
            width: 画像幅
            height: 画像高さ
        
        Returns:
            list: [r, g, b, a, r, g, b, a, ...] 形式のfloatリスト
        
        Raises:
            ValueError: データサイズが期待と異なる
        """
        expected_size = width * height * 4 * 4  # 4 floats per pixel, 4 bytes per float
        if len(payload) != expected_size:
            raise ValueError(f"Pixel data size mismatch: {len(payload)} bytes, expected {expected_size}")
        
        # array モジュールで高速にfloatリストに変換
        # 'f' は float32 を意味する
        float_array = array.array('f')
        float_array.frombytes(payload)
        return float_array.tolist()
    
    @staticmethod
    def decode_progress(payload: bytes) -> float:
        """
        PROGRESS レスポンスから進捗率をデコードします。
        
        Returns:
            float: 進捗率 (0.0 〜 1.0)
        """
        if len(payload) != 4:
            raise ValueError(f"Progress payload size mismatch: {len(payload)}")
        return struct.unpack('f', payload)[0]
    
    @staticmethod
    def decode_capabilities(payload: bytes) -> dict:
        """
        CAPABILITIES レスポンスから機能一覧をデコードします。
        
        ペイロード形式:
        [num_backends:4][backend1_len:4][backend1:...]...
        [num_algos:4][algo1_len:4][algo1:...]...
        
        Returns:
            dict: {'backends': ['cpu', ...], 'algorithms': ['nee', 'mis', ...]}
        """
        offset = 0
        
        def read_string_list():
            """文字列リストを読み取るヘルパー関数"""
            nonlocal offset
            # リストの要素数を読み取り
            count = struct.unpack_from('I', payload, offset)[0]
            offset += 4
            items = []
            for _ in range(count):
                # 文字列の長さを読み取り
                length = struct.unpack_from('I', payload, offset)[0]
                offset += 4
                # 文字列データを読み取り
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
        """
        ERROR レスポンスからエラーメッセージをデコードします。
        
        Returns:
            str: UTF-8エンコードされたエラーメッセージ
        """
        return payload.decode('utf-8')
