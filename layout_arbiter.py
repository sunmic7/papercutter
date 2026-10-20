# -*- coding: utf-8 -*-
"""版面裁判：用 PP-DocLayout_plus-L 独立裁定一页有几栏、栏缝在哪。

为什么需要它
------------
现有分栏是"猜测式"的：`detect_columns` 只看 OCR 文本框的横向空隙，
`_visual_column_split` 只看中间留白，`_local_column_split_candidates` 再看
局部白带/竖线。这些证据都来自页面局部，遇到"右栏末尾一条括号列 + 最右
答案列"这种结构时，会把栏目内部的白带当成第二道栏缝，把一页切成 3 块，
于是右侧整页的四个大题被并进同一个"整页"块。

版面模型给出的是**区域框**（text/table/image/title...），它只看版面几何，
不看文字内容。由此可以稳定地回答两件事：
  - 某条竖直带是否被正文区域框穿过（真栏缝不会被穿过）
  - 栏缝左右是否都是"成片正文"（假栏缝一侧往往只有一条细窄标记列）

本模块只回答栏位，不产出题块；题块切分仍由 cut_questions_v4 负责。

成本与开关
----------
模型约 125MB，CPU 单页推理约 5～6 秒。因此：
  - 只在"可疑页"调用（现有规则分出 ≥3 栏，或对开页被当成单栏）
  - 结果按图片内容哈希落盘缓存，同一页重复运行零成本
  - PAPERFLOW_LAYOUT_ARBITER=0 可整体关闭
  - 模型文件缺失时自动降级为"无意见"，不影响主流程
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import numpy as np

REVISION = 'layout-arbiter-v3-20261019-regions'

# 可当"题图候选"的区域类别（§4.1）：formula / table 单独保留但打 label 标记，
# 由硬拒规则决定去留 —— "公式+图"混排的复合块要靠人看一眼。
IMAGE_LABELS = ('image', 'figure', 'chart', 'formula', 'table')

# image_regions() 的筛图阈值，默认值 = 架构文档 §4.1，可被
# runtime/enrich/candidate_rules.json 的 image_regions 段覆盖（阈值必须外置）。
DEFAULT_IMAGE_RULES = {
    'labels': list(IMAGE_LABELS),
    'score_min': 0.5,
    'area_ratio_min': 0.0002,   # 0.02% 页面积
    'side_min': 64,
}
_RULES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'runtime', 'enrich', 'candidate_rules.json',
)
_RULES_CACHE = {'mtime': None, 'rules': None}

# 参与"正文占用"的版面类别；页眉页脚页码不参与，否则会糊住真实栏缝。
BODY_LABELS = {
    'text', 'paragraph_title', 'doc_title', 'abstract', 'content',
    'table', 'image', 'figure', 'figure_title', 'chart', 'formula',
    'vertical_text', 'reference',
}
IGNORED_LABELS = {
    'header', 'header_image', 'footer', 'footer_image', 'number',
    'page_number', 'footnote', 'seal', 'aside_text',
}

GRID_COLS = 1200
GRID_ROWS = 240

MAX_GUTTER_COVERAGE = 0.08   # 栏缝处被正文区域覆盖的高度比例上限
MIN_SIDE_SHARE = 0.28        # 较窄一侧的正文量占比下限
MIN_SIDE_REGIONS = 3         # 单侧至少几个区域框
# 栏缝宽度上限：不能太宽，否则"栏目末尾的大片留白"会被当成栏缝。
# 但对开扫描的装订缝常因扫描留白而宽达页宽的 10%～13%，因此放到 0.14；
# 真正拦掉"页尾留白"的是上面的两侧正文量配比（较窄一侧 < 28% 即否决）。
MAX_GUTTER_WIDTH_RATIO = 0.14
MIN_COLUMN_WIDTH_RATIO = 0.18
MAX_COLUMNS = 4
# 对开页（宽高比 ≥ 该值）视为两张物理页并排
SPREAD_RATIO = 1.24
# 与现有栏位差异小于该比例时视为"结论一致"，不做覆盖
AGREEMENT_TOLERANCE_RATIO = 0.02

_PIPE_LOCK = threading.Lock()
_PIPE = None
_PIPE_FAILED = False
_MEMORY_CACHE = {}


def _cache_dir():
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, 'runtime', 'layout_cache')


DISABLE_FLAG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'runtime', 'layout_arbiter.disabled',
)


def is_enabled():
    if os.environ.get('PAPERFLOW_LAYOUT_ARBITER', '1') in {'0', 'false', 'off'}:
        return False
    # 现场回滚用哨兵文件：不用改环境变量、不用重启进程，
    # 在 runtime/ 下建一个空文件 layout_arbiter.disabled 即可，下次调用就生效。
    return not os.path.exists(DISABLE_FLAG)


# 小于该尺寸的图多半是测试用合成图，不值得为它加载 125MB 版面模型
MIN_PAGE_WIDTH = 1200
MIN_PAGE_HEIGHT = 900


def _model_dir():
    candidates = [
        os.path.join(os.path.expanduser('~'), '.paddlex', 'official_models',
                     'PP-DocLayout_plus-L'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime', 'models',
                     'paddlex', 'PP-DocLayout_plus-L'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def available():
    if not is_enabled():
        return False
    if _model_dir() is None:
        return False
    try:
        import paddleocr  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _get_pipe():
    global _PIPE, _PIPE_FAILED
    if _PIPE is not None or _PIPE_FAILED:
        return _PIPE
    with _PIPE_LOCK:
        if _PIPE is None and not _PIPE_FAILED:
            try:
                from paddleocr import LayoutDetection
                _PIPE = LayoutDetection(
                    model_name='PP-DocLayout_plus-L', device='cpu', enable_mkldnn=False,
                )
            except Exception as error:  # noqa: BLE001
                _PIPE_FAILED = True
                print(f'[版面裁判] 模型加载失败，已降级为不干预：{error}')
    return _PIPE


def _image_key(img):
    """短小、稳定的内容指纹：缩放后取像素与尺寸。"""
    try:
        import cv2
        small = cv2.resize(img, (160, max(1, int(160 * img.shape[0] / max(1, img.shape[1])))),
                           interpolation=cv2.INTER_AREA)
        digest = hashlib.sha1()
        digest.update(str(img.shape).encode('ascii'))
        digest.update(np.ascontiguousarray(small).tobytes())
        return digest.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _read_cache(key, suffix=''):
    if key is None:
        return None
    memory_key = f'{key}{suffix}'
    if memory_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[memory_key]
    path = os.path.join(_cache_dir(), f'{key}{suffix}.json')
    try:
        with open(path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        if payload.get('revision') == REVISION:
            _MEMORY_CACHE[memory_key] = payload
            return payload
    except (OSError, ValueError):
        pass
    return None


def _write_cache(key, payload, suffix=''):
    if key is None:
        return
    _MEMORY_CACHE[f'{key}{suffix}'] = payload
    try:
        directory = _cache_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'{key}{suffix}.json')
        with open(path + '.tmp', 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False)
        os.replace(path + '.tmp', path)
    except OSError:
        pass


# ---------------------------------------------------------------- 几何推导
def _box_of(region):
    x0, y0, x1, y1 = region['box']
    return float(x0), float(y0), float(x1), float(y1)


def _body_regions(regions, width, height):
    kept = []
    for region in regions:
        label = str(region.get('label', '')).lower()
        if label in IGNORED_LABELS or label not in BODY_LABELS:
            continue
        x0, y0, x1, y1 = _box_of(region)
        if x1 <= x0 or y1 <= y0:
            continue
        if (x1 - x0) < width * 0.004 or (y1 - y0) < height * 0.004:
            continue
        kept.append(region)
    return kept


def _occupancy(regions, width, height):
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    sx = GRID_COLS / float(width)
    sy = GRID_ROWS / float(height)
    for region in regions:
        x0, y0, x1, y1 = _box_of(region)
        c0 = max(0, int(np.floor(x0 * sx)))
        c1 = min(GRID_COLS, max(c0 + 1, int(np.ceil(x1 * sx))))
        r0 = max(0, int(np.floor(y0 * sy)))
        r1 = min(GRID_ROWS, max(r0 + 1, int(np.ceil(y1 * sy))))
        grid[r0:r1, c0:c1] = True
    r0 = int(GRID_ROWS * 0.04)
    r1 = int(GRID_ROWS * 0.965)
    body = grid[r0:r1]
    coverage = body.mean(axis=0) if body.size else np.zeros(GRID_COLS, dtype=float)
    return coverage


def _empty_runs(mask):
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _side_stats(regions, x0, x1, gutter_x):
    left_area = right_area = 0.0
    left_count = right_count = 0
    for region in regions:
        bx0, by0, bx1, by1 = _box_of(region)
        if bx1 <= x0 or bx0 >= x1:
            continue
        area = (bx1 - bx0) * (by1 - by0)
        if (bx0 + bx1) / 2.0 < gutter_x:
            left_area += area
            left_count += 1
        else:
            right_area += area
            right_count += 1
    total = left_area + right_area
    if total <= 0:
        return 0.0, 0.0, 0, 0
    return left_area / total, right_area / total, left_count, right_count


def _gutter_candidates(regions, width, height, x0, x1, coverage):
    sx = GRID_COLS / float(width)
    c0 = max(0, int(round(x0 * sx)))
    c1 = min(GRID_COLS, int(round(x1 * sx)))
    if c1 - c0 < 8:
        return []
    local = coverage[c0:c1] <= MAX_GUTTER_COVERAGE
    min_gap_cells = max(2, int(round(max(8.0, width * 0.004) * sx)))
    min_col_cells = int(round((x1 - x0) * MIN_COLUMN_WIDTH_RATIO * sx))
    candidates = []
    for start, end in _empty_runs(local):
        # 贴着区域边缘的空白是页边距，不是栏缝
        if start == 0 or end == len(local):
            continue
        if end - start < min_gap_cells:
            continue
        if start < min_col_cells or (len(local) - end) < min_col_cells:
            continue
        gutter_c0 = c0 + start
        gutter_c1 = c0 + end
        gutter_x = (gutter_c0 + gutter_c1) / 2.0 / sx
        gutter_width = (gutter_c1 - gutter_c0) / sx
        if gutter_width > (x1 - x0) * MAX_GUTTER_WIDTH_RATIO:
            continue
        left_share, right_share, left_count, right_count = _side_stats(
            regions, x0, x1, gutter_x,
        )
        if min(left_share, right_share) < MIN_SIDE_SHARE:
            continue
        if min(left_count, right_count) < MIN_SIDE_REGIONS:
            continue
        coverage_at_gutter = float(
            coverage[max(0, gutter_c0 - 1):min(GRID_COLS, gutter_c1 + 1)].mean()
        )
        balance = min(left_share, right_share) / max(left_share, right_share)
        score = (1.0 - coverage_at_gutter) * (0.5 + balance) * float(
            np.sqrt(min(left_count, right_count))
        )
        candidates.append({
            'x': round(gutter_x, 1),
            'width': round(gutter_width, 1),
            'coverage': round(coverage_at_gutter, 4),
            'left_share': round(left_share, 3),
            'right_share': round(right_share, 3),
            'left_regions': left_count,
            'right_regions': right_count,
            'score': round(score, 3),
        })
    candidates.sort(key=lambda item: item['score'], reverse=True)
    return candidates


def _split_recursive(regions, width, height, coverage, x0, x1, depth, gutters):
    if depth >= MAX_COLUMNS or (x1 - x0) < width * 0.18:
        return [(x0, x1)]
    candidates = _gutter_candidates(regions, width, height, x0, x1, coverage)
    if not candidates:
        return [(x0, x1)]
    chosen = candidates[0]
    left_ids = {
        id(region) for region in regions
        if (_box_of(region)[0] + _box_of(region)[2]) / 2.0 < chosen['x']
    }
    left_regions = [region for region in regions if id(region) in left_ids]
    right_regions = [region for region in regions if id(region) not in left_ids]
    gutters.append(chosen)
    left_columns = _split_recursive(
        left_regions, width, height, _occupancy(left_regions, width, height),
        x0, chosen['x'], depth + 1, gutters,
    )
    right_columns = _split_recursive(
        right_regions, width, height, _occupancy(right_regions, width, height),
        chosen['x'], x1, depth + 1, gutters,
    )
    return left_columns + right_columns


def image_rules():
    """读 ``runtime/enrich/candidate_rules.json`` 的 ``image_regions`` 段（带 mtime 缓存）。

    文件缺失/损坏时退回 ``DEFAULT_IMAGE_RULES``：现场少一个配置文件不能让版面
    裁判整体失效。阈值必须外置的原因是不同版式（对开扫描 vs 手机拍照）最优值
    明显不同，写死在代码里必然要在现场改代码。
    """
    try:
        mtime = os.path.getmtime(_RULES_PATH)
    except OSError:
        return dict(DEFAULT_IMAGE_RULES)
    if _RULES_CACHE['mtime'] == mtime and _RULES_CACHE['rules'] is not None:
        return _RULES_CACHE['rules']
    rules = dict(DEFAULT_IMAGE_RULES)
    try:
        with open(_RULES_PATH, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        section = payload.get('image_regions') if isinstance(payload, dict) else None
        if isinstance(section, dict):
            for key in ('labels', 'score_min', 'area_ratio_min', 'side_min'):
                if key in section and section[key] is not None:
                    rules[key] = section[key]
    except (OSError, ValueError):
        pass
    _RULES_CACHE['mtime'] = mtime
    _RULES_CACHE['rules'] = rules
    return rules


def detect(img, with_regions=False):
    """返回 {'columns': [(x0, x1)], 'gutters': [...], 'regions': n} 或 None。

    ``with_regions=False``（默认）时返回键集合与改造前**完全相同** ——
    ``cut_questions_v4._apply_layout_arbiter()`` 不改，主切题流程零影响。

    ``with_regions=True`` 时额外返回两个新键（不动 ``regions`` / ``body_regions``）::

        regions_detail: [{'label': 'image', 'score': 0.94, 'box': [x0, y0, x1, y1]}]
        image_boxes:    [[x0, y0, x1, y1]]

    带框的明细单独落一份缓存（``<key>.regions.json``），这样默认调用的缓存
    里永远不会混进新键 —— 键集合不变量靠缓存分层保证，不靠返回前删键。
    """
    pipe = _get_pipe()
    if pipe is None:
        return None
    height, width = img.shape[:2]
    key = _image_key(img)
    suffix = '.regions' if with_regions else ''
    cached = _read_cache(key, suffix)
    if cached:
        return cached
    detection = pipe.predict(img, batch_size=1)
    payload = detection[0].json if hasattr(detection[0], 'json') else detection[0]
    boxes = payload.get('res', {}).get('boxes', []) if isinstance(payload, dict) else []
    regions = []
    for box in boxes:
        coordinate = box.get('coordinate') or []
        if len(coordinate) != 4:
            continue
        regions.append({
            'label': box.get('label'),
            'score': float(box.get('score', 0.0)),
            'box': [float(value) for value in coordinate],
        })
    body = _body_regions(regions, width, height)
    coverage = _occupancy(body, width, height)
    gutters = []
    columns = _split_recursive(body, width, height, coverage, 0.0, float(width), 0, gutters)
    payload = {
        'revision': REVISION,
        'width': width,
        'height': height,
        'regions': len(regions),
        'body_regions': len(body),
        'columns': [[round(a, 1), round(b, 1)] for a, b in columns],
        'gutters': sorted(gutters, key=lambda item: item['x']),
    }
    if with_regions:
        image_labels = {str(label).lower() for label in image_rules().get('labels') or IMAGE_LABELS}
        detail = dict(payload)
        detail['regions_detail'] = [
            {
                'label': str(region.get('label') or '').lower(),
                'score': round(float(region.get('score') or 0.0), 4),
                'box': [round(float(value), 1) for value in region['box']],
            }
            for region in regions
        ]
        detail['image_boxes'] = [
            [round(float(value), 1) for value in region['box']]
            for region in regions
            if str(region.get('label') or '').lower() in image_labels
        ]
        _write_cache(key, detail, '.regions')
        _write_cache(key, payload, '')
        return detail
    _write_cache(key, payload, '')
    return payload


def image_regions(img):
    """本页的题图候选区域（按 (y0, x0) 排序）；算不了就返回 ``None``。

    ``None`` 与 ``[]`` 语义必须能区分：``None`` = 没算（模型缺失/开关关闭），
    ``[]`` = 算了但页面上没有够格的图。把"没算"当成"没图"会让下游
    把"题图缺失"当事实写进交付物。
    """
    if not available():
        return None
    detail = detect(img, with_regions=True)
    if not detail:
        return None
    rules = image_rules()
    labels = {str(label).lower() for label in rules.get('labels') or IMAGE_LABELS}
    score_min = float(rules.get('score_min', DEFAULT_IMAGE_RULES['score_min']))
    area_ratio_min = float(rules.get('area_ratio_min', DEFAULT_IMAGE_RULES['area_ratio_min']))
    side_min = float(rules.get('side_min', DEFAULT_IMAGE_RULES['side_min']))
    page_width = float(detail.get('width') or 1.0)
    page_height = float(detail.get('height') or 1.0)
    page_area = max(1.0, page_width * page_height)

    kept = []
    for region in detail.get('regions_detail') or []:
        label = str(region.get('label') or '').lower()
        if label not in labels:
            continue
        if float(region.get('score') or 0.0) < score_min:
            continue
        box = [float(value) for value in region.get('box') or []]
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            continue
        if (width * height) < page_area * area_ratio_min:
            continue
        if min(width, height) < side_min:
            continue
        kept.append({'label': label, 'score': round(float(region.get('score') or 0.0), 4),
                     'box': box})
    kept.sort(key=lambda item: (item['box'][1], item['box'][0]))
    return kept


# ---------------------------------------------------------------- 对外策略
def _region_bounds(regions):
    return [(float(x0), float(x1)) for x0, x1, _items in regions]


def _same_layout(left, right, width):
    if len(left) != len(right):
        return False
    tolerance = max(24.0, width * AGREEMENT_TOLERANCE_RATIO)
    for (a0, a1), (b0, b1) in zip(left, right):
        if abs(a0 - b0) > tolerance or abs(a1 - b1) > tolerance:
            return False
    return True


def resolve_regions(img, regions, ocr_items=None, log=None):
    """给定现有分栏结果，返回裁判修正后的分栏结果（或原样返回）。

    ``regions`` 为 ``[(x0, x1, items)]``。只有两种情形会改动它：
      1. 现有规则分出 ≥3 栏，而裁判认为栏数不同 —— 以裁判为准
      2. 现有规则分出 ≥3 栏、裁判也只看到一栏、但页面明显是对开扫描
         —— 退化为按装订线二分，而不是把整页当一栏
    其余情况一律保留现状，避免在正常页面上引入回归。
    """
    if not regions or len(regions) < 3 or not available():
        return regions
    if img.shape[1] < MIN_PAGE_WIDTH or img.shape[0] < MIN_PAGE_HEIGHT:
        return regions
    try:
        layout = detect(img)
    except Exception as error:  # noqa: BLE001
        if log:
            log(f'[版面裁判] 推理失败，保留现有分栏：{error}')
        return regions
    if not layout:
        return regions

    width = img.shape[1]
    columns = [(float(a), float(b)) for a, b in layout['columns']]
    if len(columns) >= 2:
        if _same_layout(_region_bounds(regions), columns, width):
            return regions
        corrected = _regions_from_columns(columns, ocr_items, regions)
        if log:
            log(
                '[版面裁判] 栏位覆盖：现有 %s → 裁判 %s'
                % (
                    ' / '.join(f'{int(a)}-{int(b)}' for a, b in _region_bounds(regions)),
                    ' / '.join(f'{int(a)}-{int(b)}' for a, b in columns),
                )
            )
        return corrected

    # 裁判只看到一栏：对开页不能整页当一栏，按装订线二分保住"页"这一级
    ratio = width / max(1, img.shape[0])
    if ratio >= SPREAD_RATIO:
        split_x = _spine_x(img)
        corrected = _regions_from_columns(
            [(0.0, float(split_x)), (float(split_x), float(width))], ocr_items, regions,
        )
        if log:
            log(f'[版面裁判] 裁判判为单栏，对开页按装订线 {split_x} 二分')
        return corrected
    return regions


def _spine_x(img):
    width = img.shape[1]
    try:
        import cut_questions_v4 as core
        split = core._visual_column_split(img)
        if split:
            return int(split)
    except Exception:  # noqa: BLE001
        pass
    return int(width // 2)


def _regions_from_columns(columns, ocr_items, fallback_regions):
    """把 OCR 项按裁判栏位重新分组，跨栏项沿用生产逻辑拆分。"""
    if not ocr_items:
        return [(int(x0), int(x1), []) for x0, x1 in columns]
    remaining = list(ocr_items)
    result = []
    for index, (x0, x1) in enumerate(columns):
        if index == len(columns) - 1:
            result.append((int(x0), int(x1), remaining))
            break
        try:
            import cut_questions_v4 as core
            left, right = core._split_cross_column_items(remaining, x1)
        except Exception:  # noqa: BLE001
            left, right = [], remaining
        result.append((int(x0), int(x1), left))
        remaining = right
    return result
