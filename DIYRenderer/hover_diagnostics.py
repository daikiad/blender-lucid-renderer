"""
hover_diagnostics.py - Mouse Hover Diagnostic Display for Image Editor
=======================================================================

Image Editor上でマウスオーバー時にピクセルの診断情報を表示する機能を提供します。
Modal Operator とカスタム描画ハンドラを使用して実装されています。

使い方:
1. F12レンダリングを診断機能有効で実行
2. Image Editor で DIY > Pixel Inspector を開く
3. "Start Inspection" ボタンをクリック
4. マウスをレンダリング結果上で動かすとピクセル情報が表示される
5. クリックでピクセルをロック/アンロック（パス可視化に使用）
6. ESCキーまたは右クリックで終了

パス可視化機能:
- ピクセルをクリックでロックすると、そのピクセルのパスが3D Viewportに表示される
- Ctrl+クリックで強制ロック解除
"""

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from typing import Optional, Tuple, Dict, Any, List


# =============================================================================
# ピクセル診断データの取得
# =============================================================================

def get_pixel_diagnostic(pixel_x: int, pixel_y: int) -> Optional[Dict[str, Any]]:
    """
    指定ピクセルの診断データを取得します。
    
    Args:
        pixel_x: ピクセルX座標
        pixel_y: ピクセルY座標
    
    Returns:
        診断データの辞書、またはデータがない場合 None
    """
    try:
        from .diagnostics import get_global_diagnostics
        manager = get_global_diagnostics()
        
        if manager is None:
            print(f"[HoverDiagnostics] manager is None")
            return None
        if not manager.is_available:
            print(f"[HoverDiagnostics] manager not available")
            return None
        
        # get_pixel_diagnostic メソッドがあるかチェック
        if not hasattr(manager, 'get_pixel_diagnostic'):
            print(f"[HoverDiagnostics] manager has no get_pixel_diagnostic method")
            return None
        
        # 新しいAPIを使用: manager.get_pixel_diagnostic(x, y)
        result = manager.get_pixel_diagnostic(pixel_x, pixel_y)
        
        if result is None:
            print(f"[HoverDiagnostics] get_pixel_diagnostic returned None for ({pixel_x}, {pixel_y})")
            return None
        
        # PixelDiagnosticInfo を辞書に変換
        if not result.valid:
            print(f"[HoverDiagnostics] result.valid is False for ({pixel_x}, {pixel_y})")
            return None
        
        return {
            'sample_count': result.sample_count,
            'variance': result.variance,
            'group_count': result.group_count,
            'outlier_count': result.outlier_count,
            'mean_rgb': list(result.mean_rgb),
            'top_groups': result.top_groups,
        }
            
    except Exception as e:
        print(f"[HoverDiagnostics] Error getting pixel data: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_variance_at_pixel(pixel_x: int, pixel_y: int, width: int, height: int) -> Optional[float]:
    """
    variance_map から指定ピクセルの分散値を取得します。
    
    Args:
        pixel_x: ピクセルX座標
        pixel_y: ピクセルY座標
        width: 画像幅
        height: 画像高さ
    
    Returns:
        分散値、またはデータがない場合 None
    """
    try:
        from .diagnostics import get_global_diagnostics
        manager = get_global_diagnostics()
        
        if manager is None or not manager.is_available:
            return None
        
        # レポートから取得（グローバル統計のみ）
        report = manager.get_report(top_n=1)
        if report is None:
            return None
        
        # 座標チェック
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            return None
        
        return report.global_stats.total_variance / max(report.global_stats.active_pixels, 1)
        
    except Exception as e:
        return None


# =============================================================================
# グループ再集計ユーティリティ
# =============================================================================

def _regroup_paths(top_groups, total_mean_rgb=None):
    """
    パスグループを Heckbert 表記と Object Path でそれぞれ再集計します。
    
    Args:
        top_groups: C++ から取得した top_groups リスト
        total_mean_rgb: ピクセル全体の平均RGB [r, g, b]（deviation計算用）
        
    Returns:
        (heckbert_groups, object_groups) の辞書タプル
        各辞書: {key: {'key': str, 'variance': float, 'mean': float, 'deviation': float, 'count': int}}
    """
    import math
    
    heckbert_groups = {}
    object_groups = {}
    
    # total_meanの輝度を計算
    total_lum = 0.0
    if total_mean_rgb:
        total_lum = 0.2126 * total_mean_rgb[0] + 0.7152 * total_mean_rgb[1] + 0.0722 * total_mean_rgb[2]
    
    for g in top_groups:
        # Heckbert 表記でグルーピング
        sig_heckbert = getattr(g, 'signature_heckbert', '') or getattr(g, 'signature', 'Unknown')
        var = getattr(g, 'variance_luminance', 0)
        mean = getattr(g, 'mean_luminance', 0)
        cnt = getattr(g, 'sample_count', 0)
        
        # Strategy name (Plan E: sampling strategy)
        strategy = getattr(g, 'strategy_name', '')
        
        # RGB mean から輝度を計算
        mean_rgb = getattr(g, 'mean_rgb', [0, 0, 0])
        if mean_rgb:
            group_lum = 0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2]
        else:
            group_lum = mean
        
        # ピクセル全体の平均からの偏差（絶対値）
        deviation = abs(group_lum - total_lum) if total_mean_rgb else 0.0
        
        # Heckbert + Strategy でグルーピング（MIS_BSDF と MIS_NEE を分離表示）
        if strategy:
            heckbert_key = f"{sig_heckbert} [{strategy}]"
        else:
            heckbert_key = sig_heckbert
            
        if heckbert_key not in heckbert_groups:
            heckbert_groups[heckbert_key] = {
                'key': heckbert_key,
                'variance': 0.0,
                'mean': 0.0,
                'deviation': 0.0,
                'count': 0
            }
        heckbert_groups[heckbert_key]['variance'] += var * cnt  # weighted sum
        heckbert_groups[heckbert_key]['mean'] += mean * cnt
        heckbert_groups[heckbert_key]['deviation'] += deviation * cnt
        heckbert_groups[heckbert_key]['count'] += cnt
        
        # Object Path + Strategy でグルーピング
        object_path = getattr(g, 'object_path', '') or 'Unknown Path'
        if strategy:
            object_key = f"{object_path} [{strategy}]"
        else:
            object_key = object_path
            
        if object_key not in object_groups:
            object_groups[object_key] = {
                'key': object_key,
                'variance': 0.0,
                'mean': 0.0,
                'deviation': 0.0,
                'count': 0
            }
        object_groups[object_key]['variance'] += var * cnt
        object_groups[object_key]['mean'] += mean * cnt
        object_groups[object_key]['deviation'] += deviation * cnt
        object_groups[object_key]['count'] += cnt
    
    # 加重平均に変換
    for g in heckbert_groups.values():
        if g['count'] > 0:
            g['variance'] /= g['count']
            g['mean'] /= g['count']
            g['deviation'] /= g['count']
    
    for g in object_groups.values():
        if g['count'] > 0:
            g['variance'] /= g['count']
            g['mean'] /= g['count']
            g['deviation'] /= g['count']
    
    return heckbert_groups, object_groups


def _fmt_val(val: float) -> str:
    """
    値の大きさに応じて適切なフォーマットで表示します。
    
    - 0.0001 以上: 通常表記 (0.0012)
    - それ以下: 科学的表記 (1.2e-5)
    - 非常に小さい場合: <1e-9
    """
    if val == 0:
        return "0"
    
    abs_val = abs(val)
    
    if abs_val >= 0.01:
        return f"{val:.4f}"
    elif abs_val >= 0.0001:
        return f"{val:.6f}"
    elif abs_val >= 1e-9:
        return f"{val:.2e}"
    else:
        return "<1e-9"


# =============================================================================
# 描画ユーティリティ
# =============================================================================

def draw_text_box(x: float, y: float, lines: list, font_size: int = 18):
    """
    テキストボックスを描画します。
    
    Args:
        x: 左上X座標
        y: 左上Y座標  
        lines: 表示するテキスト行のリスト
        font_size: フォントサイズ
    """
    font_id = 0
    blf.size(font_id, font_size)
    
    # 行の高さとボックスサイズを計算
    line_height = font_size + 4
    padding = 8
    
    # 最大幅を計算
    max_width = 0
    for line in lines:
        dims = blf.dimensions(font_id, line)
        max_width = max(max_width, dims[0])
    
    box_width = max_width + padding * 2
    box_height = len(lines) * line_height + padding * 2
    
    # 背景ボックスを描画
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    vertices = [
        (x, y),
        (x + box_width, y),
        (x + box_width, y - box_height),
        (x, y - box_height),
    ]
    indices = [(0, 1, 2), (0, 2, 3)]
    
    gpu.state.blend_set('ALPHA')
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    shader.bind()
    shader.uniform_float("color", (0.1, 0.1, 0.1, 0.85))
    batch.draw(shader)
    
    # 枠線を描画
    border_vertices = [
        (x, y),
        (x + box_width, y),
        (x + box_width, y - box_height),
        (x, y - box_height),
        (x, y),  # Close the loop
    ]
    border_batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": border_vertices})
    shader.uniform_float("color", (0.4, 0.6, 1.0, 0.9))
    border_batch.draw(shader)  # Fixed: was border_batch.draw(border_batch)
    
    gpu.state.blend_set('NONE')
    
    # テキストを描画
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    text_y = y - padding - font_size
    for line in lines:
        blf.position(font_id, x + padding, text_y, 0)
        blf.draw(font_id, line)
        text_y -= line_height


# =============================================================================
# Modal Operator
# =============================================================================

# グローバル状態（draw_callback は self を使えないため）
_inspector_state = {
    'is_active': False,
    'handle': None,
    'handle_3d': None,  # 3D Viewport用のハンドラ
    'mouse_x': 0,
    'mouse_y': 0,
    'pixel_x': -1,
    'pixel_y': -1,
    'image_width': 0,
    'image_height': 0,
    'pixel_info': None,
    'variance': None,
    # パス可視化用
    'locked': False,
    'locked_pixel_x': -1,
    'locked_pixel_y': -1,
    'paths_data': [],  # List of path positions for visualization
    'paths_metadata': [],  # List of path metadata (signature, mean, object_path)
    'selected_path_indices': set(),  # Set of selected path indices
}


def _draw_inspector_callback(context_dummy):
    """オーバーレイ描画コールバック（グローバル関数）"""
    state = _inspector_state
    
    if not state['is_active']:
        return
    
    # デバッグ: 常にピクセル座標が負でも簡易表示
    mouse_x = state['mouse_x']
    mouse_y = state['mouse_y']
    pixel_x = state['pixel_x']
    pixel_y = state['pixel_y']
    
    # ロック状態の場合はロックされたピクセルを表示
    if state['locked'] and state['locked_pixel_x'] >= 0:
        pixel_x = state['locked_pixel_x']
        pixel_y = state['locked_pixel_y']
    
    # 表示するテキスト行を構築
    if pixel_x < 0 or pixel_y < 0:
        lines = [
            f"Mouse: ({mouse_x}, {mouse_y})",
        ]
        # デバッグ情報を表示
        debug_info = state.get('debug_info', '')
        if debug_info:
            lines.append(debug_info)
        lines.append("(Outside image bounds)")
    else:
        # ロック状態のインジケーター
        lock_icon = "🔒" if state['locked'] else ""
        lines = [
            f"Pixel: ({pixel_x}, {pixel_y}) {lock_icon}",
        ]
        
        # 診断情報を追加
        pixel_info = state['pixel_info']
        if pixel_info:
            lines.append(f"Samples: {pixel_info.get('sample_count', 'N/A')}")
            lines.append(f"Variance: {_fmt_val(pixel_info.get('variance', 0))}")
            lines.append(f"Groups: {pixel_info.get('group_count', 0)}")
            
            mean_rgb = pixel_info.get('mean_rgb', [0, 0, 0])
            if mean_rgb:
                lines.append(f"RGB: ({_fmt_val(mean_rgb[0])}, {_fmt_val(mean_rgb[1])}, {_fmt_val(mean_rgb[2])})")
            
            # トップグループ情報を表示
            top_groups = pixel_info.get('top_groups', [])
            if top_groups:
                # グループを再集計（mean_rgbを渡してdeviation計算）
                heckbert_groups, object_groups = _regroup_paths(top_groups, mean_rgb)
                
                # 1. Heckbert × Deviation (meanからの距離＝ノイズ寄与)
                lines.append("")
                lines.append("━━ Heckbert × Deviation ━━")
                sorted_heck_dev = sorted(heckbert_groups.values(), 
                                         key=lambda x: x['deviation'], reverse=True)
                for i, g in enumerate(sorted_heck_dev[:5]):
                    lines.append(f"  {g['key']}: d={_fmt_val(g['deviation'])} n={g['count']}")
                
                # 2. Heckbert × Mean (明るさ寄与)
                lines.append("")
                lines.append("━━ Heckbert × Mean ━━")
                sorted_heck_mean = sorted(heckbert_groups.values(), 
                                          key=lambda x: x['mean'], reverse=True)
                for i, g in enumerate(sorted_heck_mean[:5]):
                    lines.append(f"  {g['key']}: m={_fmt_val(g['mean'])} n={g['count']}")
                
                # 3. Object Path × Deviation
                lines.append("")
                lines.append("━━ Object × Deviation ━━")
                sorted_obj_dev = sorted(object_groups.values(), 
                                        key=lambda x: x['deviation'], reverse=True)
                for i, g in enumerate(sorted_obj_dev[:5]):
                    lines.append(f"  {g['key']}")
                    lines.append(f"    d={_fmt_val(g['deviation'])} n={g['count']}")
                
                # 4. Object Path × Mean
                lines.append("")
                lines.append("━━ Object × Mean ━━")
                sorted_obj_mean = sorted(object_groups.values(), 
                                         key=lambda x: x['mean'], reverse=True)
                for i, g in enumerate(sorted_obj_mean[:5]):
                    lines.append(f"  {g['key']}")
                    lines.append(f"    m={_fmt_val(g['mean'])} n={g['count']}")
                    
        elif state['variance'] is not None:
            lines.append(f"Avg Variance: {_fmt_val(state['variance'])}")
        else:
            lines.append("(No pixel diagnostic data)")
        
        # グローバル統計情報を表示
        try:
            from .diagnostics import get_global_diagnostics
            manager = get_global_diagnostics()
            if manager and manager.is_available:
                report = manager.get_report(top_n=1)
                if report:
                    lines.append("")
                    lines.append(f"Total Samples: {report.global_stats.total_samples:,}")
        except:
            pass
    
    # Image Editor の左上に固定表示
    # 現在のリージョンの高さを取得
    region_height = 600  # デフォルト値
    region_obj = None
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region_height = region.height
                        region_obj = region
                        break
                break
    except:
        pass
    
    draw_x = 20  # 左端から20px
    draw_y = region_height - 20  # 上端から20px下
    
    draw_text_box(draw_x, draw_y, lines)
    
    # 2Dパス描画（ロック中かつ選択がある場合）
    if state['locked'] and state.get('selected_path_indices'):
        _draw_paths_2d(region_obj)


def _draw_paths_2d(region):
    """Image Editor上で2Dパスを描画"""
    import colorsys
    from bpy_extras.object_utils import world_to_camera_view
    
    state = _inspector_state
    paths_data = state.get('paths_data', [])
    selected = state.get('selected_path_indices', set())
    
    if not paths_data or not selected or not region:
        return
    
    scene = bpy.context.scene
    camera = scene.camera
    if not camera:
        return
    
    # 画像サイズを取得
    try:
        space = None
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                space = area.spaces.active
                break
        
        if not space or not space.image:
            return
        
        image = space.image
        img_width, img_height = image.size
        
        if img_width == 0 or img_height == 0:
            if image.name == 'Render Result':
                render = scene.render
                img_width = int(render.resolution_x * render.resolution_percentage / 100)
                img_height = int(render.resolution_y * render.resolution_percentage / 100)
            else:
                return
    except:
        return
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    from mathutils import Vector
    
    # 画面クリッピング用のマージン
    CLIP_MARGIN = 5000  # 画面外に大きく出る座標をクリップ
    
    def clip_line_to_screen(p1, p2, margin=CLIP_MARGIN):
        """2点間の線分を画面内にクリップ"""
        x1, y1 = p1
        x2, y2 = p2
        
        # 両方が範囲内ならそのまま
        if (abs(x1) < margin and abs(y1) < margin and 
            abs(x2) < margin and abs(y2) < margin):
            return p1, p2
        
        # p2が非常に遠い場合、p1からp2方向へ一定距離で切る
        dx = x2 - x1
        dy = y2 - y1
        length = (dx*dx + dy*dy) ** 0.5
        
        if length > margin:
            # 方向ベクトルを正規化してmargin分だけ伸ばす
            scale = margin / length
            x2 = x1 + dx * scale
            y2 = y1 + dy * scale
        
        return p1, (x2, y2)
    
    for i, path_positions in enumerate(paths_data):
        # 選択されたパスのみ描画
        if i not in selected:
            continue
        
        if len(path_positions) < 2:
            continue
        
        # ワールド座標を画像座標に変換
        coords_2d = []
        is_behind_camera = []  # カメラの後ろにあるかどうか
        raw_uv = []  # UV座標を保存（後で方向計算に使う）
        for k, pos in enumerate(path_positions):
            try:
                # world_to_camera_view: (0,0) = 左下, (1,1) = 右上
                co = world_to_camera_view(scene, camera, Vector(pos))
                raw_uv.append((co.x, co.y, co.z))
                
                behind = co.z <= 0
                is_behind_camera.append(behind)
                
                # カメラの後ろにある点は一旦Noneにして、後で方向から計算
                if behind:
                    coords_2d.append(None)
                else:
                    reg_x, reg_y = region.view2d.view_to_region(co.x, co.y, clip=False)
                    coords_2d.append((reg_x, reg_y))
            except Exception as e:
                coords_2d.append(None)
                is_behind_camera.append(True)
                raw_uv.append(None)
        
        # カメラの後ろにある点への線は、前の点からの方向で延長
        for j in range(len(coords_2d)):
            if is_behind_camera[j] and j > 0 and raw_uv[j] is not None and raw_uv[j-1] is not None:
                # 前の点と現在の点のUV方向を計算
                prev_uv = raw_uv[j-1]
                curr_uv = raw_uv[j]
                
                # 前の点がカメラの前にある場合のみ
                if prev_uv[2] > 0:
                    # UV空間での方向
                    dir_u = curr_uv[0] - prev_uv[0]
                    dir_v = curr_uv[1] - prev_uv[1]
                    
                    # 方向を反転（z < 0なので逆方向に出るため）
                    # 前の点から正しい方向へ延長
                    ext_u = prev_uv[0] - dir_u * 10  # 反転して延長
                    ext_v = prev_uv[1] - dir_v * 10
                    
                    reg_x, reg_y = region.view2d.view_to_region(ext_u, ext_v, clip=False)
                    coords_2d[j] = (reg_x, reg_y)
        
        # 有効な座標のみで線分を描画
        # パスインデックスに基づいて色を生成（虹色）
        hue = (i / max(len(paths_data), 1)) * 0.8
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        color = (r, g, b, 0.9)
        
        # 隣接する有効座標ペアを線分として描画（クリッピング付き）
        for j in range(len(coords_2d) - 1):
            p1 = coords_2d[j]
            p2 = coords_2d[j + 1]
            
            if p1 is None or p2 is None:
                continue
            
            # 画面外に大きく出る線分をクリップ
            p1_clipped, p2_clipped = clip_line_to_screen(p1, p2)
            
            batch = batch_for_shader(shader, 'LINES', {"pos": [p1_clipped, p2_clipped]})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
        
        # バウンス点にマーカーを描画（カメラの前にある点のみ）
        for j, coord in enumerate(coords_2d):
            if coord is not None and j > 0:  # カメラ位置はスキップ
                # カメラの後ろにある点はスキップ
                if is_behind_camera[j]:
                    continue
                # 画面内の点のみマーカーを描画
                if abs(coord[0]) < CLIP_MARGIN and abs(coord[1]) < CLIP_MARGIN:
                    _draw_bounce_marker_2d(shader, coord, color)
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_bounce_marker_2d(shader, position: Tuple[float, float], color: Tuple[float, float, float, float]):
    """2Dでバウンス点にマーカーを描画"""
    size = 4
    x, y = position
    
    # 小さな十字
    lines = [
        [(x - size, y), (x + size, y)],
        [(x, y - size), (x, y + size)],
    ]
    
    for line in lines:
        batch = batch_for_shader(shader, 'LINES', {"pos": line})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)


def _draw_paths_3d_callback(context_dummy):
    """3D Viewport用のパス描画コールバック"""
    state = _inspector_state
    
    if not state['is_active'] or not state['locked']:
        return
    
    paths_data = state.get('paths_data', [])
    if not paths_data:
        return
    
    selected = state.get('selected_path_indices', set())
    # 選択がなければ描画しない
    if not selected:
        return
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    gpu.state.depth_test_set('LESS_EQUAL')
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    for i, path_positions in enumerate(paths_data):
        # 選択されたパスのみ描画
        if i not in selected:
            continue
        
        if len(path_positions) < 2:
            continue
        
        # パスインデックスに基づいて色を生成（虹色）
        import colorsys
        hue = (i / max(len(paths_data), 1)) * 0.8  # 0 to 0.8 (赤から紫)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        color = (r, g, b, 0.8)
        
        # ポリラインを描画
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": path_positions})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
        
        # バウンス点にマーカーを描画
        for pos in path_positions[1:]:  # カメラ位置をスキップ
            _draw_bounce_marker_3d(shader, pos, color)
    
    # 状態を復元
    gpu.state.depth_test_set('NONE')
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_bounce_marker_3d(shader, position: Tuple[float, float, float], color: Tuple[float, float, float, float]):
    """バウンス点に小さな十字マーカーを描画"""
    size = 0.05
    x, y, z = position
    
    # 3軸方向に短い線を描画
    lines = [
        [(x - size, y, z), (x + size, y, z)],
        [(x, y - size, z), (x, y + size, z)],
        [(x, y, z - size), (x, y, z + size)],
    ]
    
    for line in lines:
        batch = batch_for_shader(shader, 'LINES', {"pos": line})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)


def _update_paths_data(pixel_x: int, pixel_y: int):
    """ロックされたピクセルのパスデータを更新"""
    global _inspector_state
    
    try:
        from .diagnostics import get_global_diagnostics
        manager = get_global_diagnostics()
        
        if manager is None or not manager.is_available:
            _inspector_state['paths_data'] = []
            _inspector_state['paths_metadata'] = []
            _inspector_state['selected_path_indices'] = set()
            return
        
        # 診断データを取得
        result = manager.get_pixel_diagnostic(pixel_x, pixel_y)
        if result is None or not result.valid:
            _inspector_state['paths_data'] = []
            _inspector_state['paths_metadata'] = []
            _inspector_state['selected_path_indices'] = set()
            return
        
        # max_visualized_paths を取得
        max_paths = 20
        try:
            settings = bpy.context.scene.diy_renderer
            max_paths = settings.max_visualized_paths
        except:
            pass
        
        # パスデータとメタデータを抽出（positions配列を持つグループのみ）
        paths_data = []
        paths_metadata = []
        for group in result.top_groups[:max_paths]:
            if hasattr(group, 'positions') and group.positions:
                # positions は list of list/tuple で [x, y, z]
                positions = [tuple(p) for p in group.positions]
                if len(positions) >= 2:
                    paths_data.append(positions)
                    # メタデータを保存
                    paths_metadata.append({
                        'signature': getattr(group, 'signature_heckbert', '') or getattr(group, 'signature', ''),
                        'mean': getattr(group, 'mean_luminance', 0.0),
                        'object_path': getattr(group, 'object_path', ''),
                        'sample_count': getattr(group, 'sample_count', 0),
                        'strategy': getattr(group, 'strategy_name', ''),
                    })
        
        _inspector_state['paths_data'] = paths_data
        _inspector_state['paths_metadata'] = paths_metadata
        _inspector_state['selected_path_indices'] = set()  # 選択をリセット
        
        # 3D Viewport を更新
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
    except Exception as e:
        print(f"[HoverDiagnostics] Error updating paths data: {e}")
        import traceback
        traceback.print_exc()
        _inspector_state['paths_data'] = []
        _inspector_state['paths_metadata'] = []
        _inspector_state['selected_path_indices'] = set()


class DIY_OT_pixel_inspector(bpy.types.Operator):
    """
    ピクセル診断インスペクター
    
    Image Editor上でマウスオーバー時にピクセルの診断情報を
    オーバーレイ表示します。
    """
    bl_idname = "diy_render.pixel_inspector"
    bl_label = "Pixel Inspector"
    bl_description = "Inspect pixel diagnostic data on mouse hover"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        """Image Editor でのみ有効"""
        return context.area and context.area.type == 'IMAGE_EDITOR'
    
    @classmethod
    def is_active(cls):
        """インスペクターがアクティブかどうか"""
        return _inspector_state['is_active']
    
    def invoke(self, context, event):
        """オペレーター起動"""
        global _inspector_state
        
        if context.area.type != 'IMAGE_EDITOR':
            self.report({'WARNING'}, "Image Editor で実行してください")
            return {'CANCELLED'}
        
        # 既にアクティブなら停止
        if _inspector_state['is_active']:
            self._cleanup(context)
            self.report({'INFO'}, "ピクセルインスペクター終了")
            return {'CANCELLED'}
        
        # 診断データの確認
        try:
            from .diagnostics import get_global_diagnostics
            manager = get_global_diagnostics()
            if manager is None or not manager.is_available:
                self.report({'WARNING'}, "診断データがありません。診断を有効にしてレンダリングしてください")
                return {'CANCELLED'}
        except ImportError:
            self.report({'ERROR'}, "診断モジュールが利用できません")
            return {'CANCELLED'}
        
        # 画像サイズを取得
        space = context.space_data
        if space.image:
            _inspector_state['image_width'], _inspector_state['image_height'] = space.image.size
        else:
            self.report({'WARNING'}, "画像が選択されていません")
            return {'CANCELLED'}
        
        # 状態をリセット
        _inspector_state['pixel_x'] = -1
        _inspector_state['pixel_y'] = -1
        _inspector_state['pixel_info'] = None
        _inspector_state['variance'] = None
        _inspector_state['locked'] = False
        _inspector_state['locked_pixel_x'] = -1
        _inspector_state['locked_pixel_y'] = -1
        _inspector_state['paths_data'] = []
        
        # 描画ハンドラを登録（グローバル関数を使用）
        _inspector_state['handle'] = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_inspector_callback, (None,), 'WINDOW', 'POST_PIXEL'
        )
        # 3D Viewport用ハンドラも登録
        _inspector_state['handle_3d'] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_paths_3d_callback, (None,), 'WINDOW', 'POST_VIEW'
        )
        _inspector_state['is_active'] = True
        
        # モーダルハンドラを登録
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        
        self.report({'INFO'}, "ピクセルインスペクター開始 (クリックでロック, ESC で終了)")
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        """モーダルイベント処理"""
        global _inspector_state
        
        # Image Editor 以外では処理しない
        if not context.area or context.area.type != 'IMAGE_EDITOR':
            return {'PASS_THROUGH'}
        
        # エリアの再描画をトリガー
        context.area.tag_redraw()
        
        # 終了条件
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._cleanup(context)
            self.report({'INFO'}, "ピクセルインスペクター終了")
            return {'CANCELLED'}
        
        # クリック処理（画像領域内のみ）
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # マウスがUIパネル上にあるかチェック（UIパネル上ならスキップ）
            if self._is_mouse_over_ui_panel(context, event.mouse_x, event.mouse_y):
                return {'PASS_THROUGH'}
            
            # クリック時に現在のマウス座標から画像範囲内かを判定
            click_pixel = self._get_pixel_at_mouse(context, event.mouse_region_x, event.mouse_region_y)
            
            # 画像範囲内のクリックでピクセルロックを更新
            if click_pixel is not None:
                pixel_x, pixel_y = click_pixel
                _inspector_state['locked'] = True
                _inspector_state['locked_pixel_x'] = pixel_x
                _inspector_state['locked_pixel_y'] = pixel_y
                _update_paths_data(pixel_x, pixel_y)
                path_count = len(_inspector_state['paths_data'])
                self.report({'INFO'}, f"ピクセル ({pixel_x}, {pixel_y}) ロック - {path_count} パス")
                
                # 3D Viewport も更新
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                
                return {'RUNNING_MODAL'}
            # 画像範囲外のクリックは PASS_THROUGH（UIパネル等に渡す）
        
        # マウス移動（ロック中はスキップ）
        if not _inspector_state['locked']:
            _inspector_state['mouse_x'] = event.mouse_region_x
            _inspector_state['mouse_y'] = event.mouse_region_y
            self._update_pixel_info(context)
        else:
            # ロック中でもマウス座標は更新（表示用）
            _inspector_state['mouse_x'] = event.mouse_region_x
            _inspector_state['mouse_y'] = event.mouse_region_y
        
        # 他のイベントはパススルー（通常操作を許可）
        return {'PASS_THROUGH'}
    
    def _is_mouse_over_ui_panel(self, context, mouse_x: int, mouse_y: int) -> bool:
        """マウスがUIパネル（またはヘッダー等）上にあるかチェック"""
        if not context.area:
            return False
        
        for region in context.area.regions:
            # WINDOW以外のリージョン（UI, HEADER, TOOLS等）をチェック
            if region.type != 'WINDOW':
                # マウスがこのリージョン内にあるか
                if (region.x <= mouse_x < region.x + region.width and
                    region.y <= mouse_y < region.y + region.height):
                    return True
        return False
    
    def _get_pixel_at_mouse(self, context, mouse_x: int, mouse_y: int):
        """マウス座標から画像ピクセル座標を取得。範囲外なら None を返す。"""
        region = context.region
        space = context.space_data
        
        if not region or not space or not space.image:
            return None
        
        # 画像サイズを取得
        image = space.image
        width, height = image.size
        
        if width == 0 or height == 0:
            if image.name == 'Render Result':
                render = context.scene.render
                width = int(render.resolution_x * render.resolution_percentage / 100)
                height = int(render.resolution_y * render.resolution_percentage / 100)
            else:
                return None
        
        # リージョン座標をビュー座標に変換
        try:
            view_x, view_y = region.view2d.region_to_view(mouse_x, mouse_y)
        except:
            return None
        
        # view座標をピクセル座標に変換
        pixel_x = int(view_x * width)
        pixel_y = height - 1 - int(view_y * height)
        
        # 範囲チェック
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            return (pixel_x, pixel_y)
        return None
    
    def _update_pixel_info(self, context):
        """マウス位置からピクセル情報を更新"""
        global _inspector_state
        
        region = context.region
        space = context.space_data
        
        if not region or not space or not space.image:
            _inspector_state['pixel_x'] = -1
            _inspector_state['pixel_y'] = -1
            _inspector_state['debug_info'] = "No region/space/image"
            return
        
        # 画像サイズを取得
        # Render Result の場合は image.size が (0, 0) になるので
        # レンダー設定から取得する
        image = space.image
        width, height = image.size
        
        if width == 0 or height == 0:
            # Render Result の場合、レンダー設定から解像度を取得
            if image.name == 'Render Result':
                render = context.scene.render
                width = int(render.resolution_x * render.resolution_percentage / 100)
                height = int(render.resolution_y * render.resolution_percentage / 100)
            else:
                # 他の画像でサイズが0の場合
                _inspector_state['pixel_x'] = -1
                _inspector_state['pixel_y'] = -1
                _inspector_state['debug_info'] = f"Image '{image.name}' size: 0x0"
                return
        
        mouse_x = _inspector_state['mouse_x']
        mouse_y = _inspector_state['mouse_y']
        
        # リージョン座標をビュー座標に変換
        try:
            view_x, view_y = region.view2d.region_to_view(mouse_x, mouse_y)
        except Exception as e:
            _inspector_state['pixel_x'] = -1
            _inspector_state['pixel_y'] = -1
            _inspector_state['debug_info'] = f"Exception: {e}"
            return
        
        # view座標（正規化UV: 0〜1）をピクセル座標に変換
        pixel_x = int(view_x * width)
        # Y座標を反転（Blenderの画像座標系はY=0が下、レンダラーはY=0が上）
        pixel_y = height - 1 - int(view_y * height)
        
        # デバッグ情報を保存
        _inspector_state['debug_info'] = f"UV: ({view_x:.3f}, {view_y:.3f}), Size: {width}x{height}"
        
        # 範囲チェック
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            _inspector_state['pixel_x'] = pixel_x
            _inspector_state['pixel_y'] = pixel_y
            
            # 診断データを取得
            pixel_info = get_pixel_diagnostic(pixel_x, pixel_y)
            _inspector_state['pixel_info'] = pixel_info
            _inspector_state['variance'] = get_variance_at_pixel(pixel_x, pixel_y, width, height)
            
            # デバッグ: 診断データの状態
            if pixel_info is None:
                _inspector_state['debug_info'] += " | pixel_diag=None"
            else:
                _inspector_state['debug_info'] += f" | pixel_diag OK"
        else:
            _inspector_state['pixel_x'] = -1
            _inspector_state['pixel_y'] = -1
            _inspector_state['pixel_info'] = None
            _inspector_state['variance'] = None
    
    def _cleanup(self, context):
        """クリーンアップ処理"""
        global _inspector_state
        
        _inspector_state['is_active'] = False
        _inspector_state['locked'] = False
        _inspector_state['locked_pixel_x'] = -1
        _inspector_state['locked_pixel_y'] = -1
        _inspector_state['paths_data'] = []
        
        if _inspector_state['handle']:
            bpy.types.SpaceImageEditor.draw_handler_remove(
                _inspector_state['handle'], 'WINDOW'
            )
            _inspector_state['handle'] = None
        
        if _inspector_state['handle_3d']:
            bpy.types.SpaceView3D.draw_handler_remove(
                _inspector_state['handle_3d'], 'WINDOW'
            )
            _inspector_state['handle_3d'] = None
        
        if context.area:
            context.area.tag_redraw()
        
        # 3D Viewport も更新
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# =============================================================================
# Image Editor パネル
# =============================================================================

class DIY_PT_image_editor_diagnostics(bpy.types.Panel):
    """
    Image Editor の診断パネル
    """
    bl_label = "DIY Pixel Inspector"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'DIY'
    
    @classmethod
    def poll(cls, context):
        """DIY Renderer の診断データがある場合のみ表示"""
        return True  # 常に表示（データがない場合は説明を表示）
    
    def draw(self, context):
        layout = self.layout
        
        # 診断データの確認
        has_data = False
        try:
            from .diagnostics import get_global_diagnostics
            manager = get_global_diagnostics()
            has_data = manager is not None and manager.is_available
        except:
            pass
        
        # インスペクターボタン
        if _inspector_state['is_active']:
            layout.operator("diy_render.pixel_inspector", 
                          text="Stop Inspection", 
                          icon='CANCEL')
        else:
            col = layout.column()
            col.enabled = has_data
            col.operator("diy_render.pixel_inspector", 
                        text="Start Inspection", 
                        icon='EYEDROPPER')
        
        # ステータス表示
        box = layout.box()
        if has_data:
            box.label(text="Diagnostic data available", icon='CHECKMARK')
            
            # 簡易統計表示
            try:
                report = manager.get_report(top_n=1)
                if report:
                    col = box.column(align=True)
                    col.scale_y = 0.8
                    col.label(text=f"Samples: {report.global_stats.total_samples:,}")
                    col.label(text=f"Pixels: {report.global_stats.active_pixels:,}")
            except:
                pass
        else:
            box.label(text="No diagnostic data", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="1. Enable diagnostics in")
            col.label(text="   Render > Path Diagnostics")
            col.label(text="2. Run F12 render")
        
        # 使い方
        if _inspector_state['is_active']:
            box = layout.box()
            box.label(text="Controls:", icon='HELP')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Move mouse to inspect")
            col.label(text="Click to lock pixel")
            col.label(text="Ctrl+Click to unlock")
            col.label(text="ESC or Right-click to exit")
            
            # ロック状態表示とパスリスト
            if _inspector_state['locked']:
                px = _inspector_state['locked_pixel_x']
                py = _inspector_state['locked_pixel_y']
                paths_metadata = _inspector_state.get('paths_metadata', [])
                selected = _inspector_state.get('selected_path_indices', set())
                
                box = layout.box()
                box.label(text=f"Locked: ({px}, {py})", icon='LOCKED')
                
                # パスリスト
                if paths_metadata:
                    box.label(text=f"Paths ({len(paths_metadata)}):")
                    
                    # 全選択/全解除ボタン
                    row = box.row(align=True)
                    row.operator("diy_render.select_all_paths", text="All", icon='CHECKBOX_HLT')
                    row.operator("diy_render.deselect_all_paths", text="None", icon='CHECKBOX_DEHLT')
                    
                    # パスリスト表示
                    col = box.column(align=True)
                    for i, meta in enumerate(paths_metadata):
                        is_selected = i in selected
                        
                        # 行: 選択ボタン + パス情報
                        row = col.row(align=True)
                        
                        # 選択トグルボタン
                        op = row.operator(
                            "diy_render.toggle_path_selection",
                            text="",
                            icon='CHECKBOX_HLT' if is_selected else 'CHECKBOX_DEHLT',
                            depress=is_selected
                        )
                        op.path_index = i
                        
                        # パス情報（クリックで選択トグル）
                        sig = meta.get('signature', '?')[:20]
                        mean = meta.get('mean', 0)
                        strategy = meta.get('strategy', '')
                        
                        label = f"{i+1}. {sig}"
                        if strategy:
                            label += f" [{strategy[:3]}]"
                        
                        op2 = row.operator(
                            "diy_render.toggle_path_selection",
                            text=label,
                            depress=is_selected
                        )
                        op2.path_index = i
                        
                        # 寄与度
                        row.label(text=f"{mean:.4f}")
                    
                    # 選択数表示
                    box.label(text=f"Selected: {len(selected)} / {len(paths_metadata)}")
                else:
                    box.label(text="No paths with geometry")


# =============================================================================
# パス選択オペレーター
# =============================================================================

class DIY_OT_toggle_path_selection(bpy.types.Operator):
    """パスの選択をトグルする"""
    bl_idname = "diy_render.toggle_path_selection"
    bl_label = "Toggle Path Selection"
    bl_description = "Toggle selection of this path for 3D visualization"
    bl_options = {'INTERNAL'}
    
    path_index: bpy.props.IntProperty(default=-1)
    
    def execute(self, context):
        global _inspector_state
        idx = self.path_index
        
        if idx < 0 or idx >= len(_inspector_state.get('paths_data', [])):
            return {'CANCELLED'}
        
        selected = _inspector_state.get('selected_path_indices', set())
        
        if idx in selected:
            selected.discard(idx)
        else:
            selected.add(idx)
        
        _inspector_state['selected_path_indices'] = selected
        
        # 全ウィンドウの関連エリアを再描画
        _redraw_all_viewports()
        
        return {'FINISHED'}


class DIY_OT_select_all_paths(bpy.types.Operator):
    """全パスを選択する"""
    bl_idname = "diy_render.select_all_paths"
    bl_label = "Select All Paths"
    bl_description = "Select all paths for 3D visualization"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        global _inspector_state
        paths_count = len(_inspector_state.get('paths_data', []))
        _inspector_state['selected_path_indices'] = set(range(paths_count))
        
        # 全ウィンドウの関連エリアを再描画
        _redraw_all_viewports()
        
        return {'FINISHED'}


class DIY_OT_deselect_all_paths(bpy.types.Operator):
    """全パスの選択を解除する"""
    bl_idname = "diy_render.deselect_all_paths"
    bl_label = "Deselect All Paths"
    bl_description = "Deselect all paths"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        global _inspector_state
        _inspector_state['selected_path_indices'] = set()
        
        # 全ウィンドウの関連エリアを再描画
        _redraw_all_viewports()
        
        return {'FINISHED'}


def _redraw_all_viewports():
    """全ウィンドウの3D ViewportとImage Editorを再描画"""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {'VIEW_3D', 'IMAGE_EDITOR'}:
                area.tag_redraw()


# =============================================================================
# 登録
# =============================================================================

classes = (
    DIY_OT_pixel_inspector,
    DIY_OT_toggle_path_selection,
    DIY_OT_select_all_paths,
    DIY_OT_deselect_all_paths,
    DIY_PT_image_editor_diagnostics,
)


def register():
    """クラスを登録"""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """クラスを解除"""
    global _inspector_state
    
    # アクティブなインスペクターをクリーンアップ
    if _inspector_state['handle']:
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(
                _inspector_state['handle'], 'WINDOW'
            )
        except:
            pass
        _inspector_state['handle'] = None
    
    if _inspector_state['handle_3d']:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(
                _inspector_state['handle_3d'], 'WINDOW'
            )
        except:
            pass
        _inspector_state['handle_3d'] = None
    
    _inspector_state['is_active'] = False
    _inspector_state['locked'] = False
    _inspector_state['paths_data'] = []
    _inspector_state['paths_metadata'] = []
    _inspector_state['selected_path_indices'] = set()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
