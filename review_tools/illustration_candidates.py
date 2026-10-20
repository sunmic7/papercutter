# -*- coding: utf-8 -*-
"""题图候选：归属判定 + 机器硬拒（纯几何/像素判定，不发请求）。

产品前提：**100% 人工确认**。所以这里只产出"候选"，永远不产出 `illustrations`。
候选走独立通道下发（UI 画虚线框 + 角标），人工点"采用"后才会通过
``image-area`` 标注变成正式题图。``confirmed`` 只由后端依据 UI 回传的
``crop_sha1`` 置真 —— 除了人，没有第二个入口。

归属判定（第一个命中即返回，**不做多归属**）
-------------------------------------------
① 包含法 ``contain`` 0.95：图框 ≥85% 面积落在某题框内，且两者中心同栏
② 中心法 ``center``  0.80：图框中心点落在某题框内
③ 邻接法 ``nearest`` 0.50：同栏 + 该题框是本列上方最近 + 纵向间隙 ≤1.2×题框高
   + 间隙内无其它题框
④ 否决 ``cross_column`` / ``orphan`` 0.0：图框宽度跨栏缝 ≥20%，或③不满足

④ 的结果不写入候选的所有权，而是让受影响题进入 ``image_owner_ambiguous``（L2）：
"题图属于谁"是这个产品里最贵的错误，交付后无法挽回。

机器硬拒 4 条（§4.4）
----------------------
``page_like``（整页缩略图）→ ``formula_only``（公式条）→ ``number_label``（序号图）
→ ``too_small``（最短边 <128）。按此顺序返回第一个命中原因。被拒的候选**不丢弃**，
仍下发（UI 显示为灰色不可采用）：让人看到"系统为什么认为这是坏图"，
比静默丢弃好，因为人是最终闸门。
"""
from __future__ import annotations

import hashlib
import json
import os

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(ROOT_DIR, 'runtime', 'enrich', 'candidate_rules.json')

DEFAULT_RULES = {
    'version': 'candidate-rules-2026.10.1',
    'image_regions': {
        'labels': ['image', 'figure', 'chart', 'formula', 'table'],
        'score_min': 0.5,
        'area_ratio_min': 0.0002,
        'side_min': 64,
    },
    'owner': {
        'contain_area_ratio': 0.85,
        'confidences': {
            'contain': 0.95, 'center': 0.8, 'nearest': 0.5,
            'cross_column': 0.0, 'orphan': 0.0,
        },
        'nearest_gap_ratio': 1.2,
        'nearest_overlap_tolerance_ratio': 0.15,
        'gutter_cross_ratio': 0.2,
        'column_epsilon': 1.0,
    },
    'hard_reject': {
        'page_like': {
            'hamming_max': 64,
            'hash_size': 32,
            'page_area_ratio_min': 0.9,
            'page_similarity_min': 0.9,
            'page_margin_ratio_max': 0.05,
        },
        # 默认关闭硬拒（B 方案）：需求允许图表/统计图作题图，表格降级为待复核。
        # 内置默认值必须与 runtime/enrich/candidate_rules.json 一致，
        # 否则"配置文件缺失"的现场会退回一个已被否决的行为。
        'table_region': {'enabled': False, 'reason': 'table_region'},
        'formula_only': {
            'vertical_ratio_max': 3,
            'horizontal_ratio_max': 6,
            'thin_area_ratio_max': 0.02,
            'thin_height_ratio_max': 0.05,
            'reason': 'formula_only',
        },
        'number_label': {'area_min': 0.0015, 'side_min': 48},
        'too_small': {'side_min': 128},
    },
}


def _merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_rules(path: str | None = None) -> dict:
    """读阈值文件并与内置默认值深合并（缺键不会 KeyError，少文件不会崩）。"""
    target = path or RULES_PATH
    try:
        with open(target, 'r', encoding='utf-8') as file:
            payload = json.load(file)
    except (OSError, ValueError):
        return _merge(DEFAULT_RULES, {})
    if not isinstance(payload, dict):
        return _merge(DEFAULT_RULES, {})
    return _merge(DEFAULT_RULES, payload)


RULES = load_rules()


def reload_rules(path: str | None = None) -> dict:
    """测试与现场调阈值用：重读配置并刷新模块级 ``RULES``。"""
    global RULES
    RULES = load_rules(path)
    return RULES


# ---------------------------------------------------------------- 几何工具
def _box_of(value):
    if isinstance(value, dict):
        value = value.get('box') or value.get('bbox')
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(left, right):
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _column_bounds(columns) -> list[tuple[float, float]]:
    bounds = []
    for item in columns or []:
        pair = None
        if isinstance(item, dict) and {'x0', 'x1'} <= set(item):
            pair = (item.get('x0'), item.get('x1'))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pair = (item[0], item[1])
        if pair is None:
            continue
        try:
            x0, x1 = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        if x1 > x0:
            bounds.append((x0, x1))
    bounds.sort()
    return bounds


def gutters_of(columns) -> list[float]:
    """相邻两栏交界的中点 x。"""
    bounds = _column_bounds(columns)
    return [(left[1] + right[0]) / 2.0 for left, right in zip(bounds, bounds[1:])]


def _column_index(box, columns) -> int:
    """框中心落在第几栏（-1 = 无栏位信息）。"""
    bounds = _column_bounds(columns)
    if not bounds:
        return -1
    center_x, _ = _center(box)
    for index, (x0, x1) in enumerate(bounds):
        if x0 <= center_x <= x1:
            return index
    return -1


def _same_column(left, right, columns) -> bool:
    """同一栏：栏位信息缺失时退化为"横向有重叠"，不硬判成不同栏。"""
    left_index = _column_index(left, columns)
    right_index = _column_index(right, columns)
    if left_index < 0 or right_index < 0:
        overlap = min(left[2], right[2]) - max(left[0], right[0])
        return overlap > 0
    return left_index == right_index


def _crosses_gutter(box, columns) -> bool:
    gutters = gutters_of(columns)
    if not gutters:
        return False
    limit = float(RULES['owner']['gutter_cross_ratio'])
    width = box[2] - box[0]
    for gutter in gutters:
        if not (box[0] < gutter < box[2]):
            continue
        if min(gutter - box[0], box[2] - gutter) / width >= limit:
            return True
    return False


def _question_boxes(records) -> list[dict]:
    items = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if str(record.get('region_role', 'question')) != 'question':
            continue
        box = _box_of(record)
        if box is None:
            continue
        items.append({'id': str(record.get('id') or ''), 'box': box})
    return items


def _verdict(owner_id: str, method: str) -> dict:
    confidences = RULES['owner']['confidences']
    return {
        'owner_id': owner_id,
        'owner_method': method,
        'owner_confidence': float(confidences.get(method, 0.0)),
    }


# ---------------------------------------------------------------- 归属判定
def assign_owner(image_box, records, columns=None) -> dict:
    """判定这张图属于哪道题。

    返回 ``{'owner_id','owner_method','owner_confidence'}``。
    ``owner_method`` ∈ {contain, center, nearest, cross_column, orphan}，
    后两者的 ``owner_id`` 为空且 ``owner_confidence=0.0`` —— 调用方必须把它们
    转成记录的 ``image_owner_ambiguous``（L2），不能当"没找到主人"静默忽略。
    """
    box = _box_of(image_box)
    questions = _question_boxes(records)
    if box is None or not questions:
        return _verdict('', 'orphan')

    contain_ratio = float(RULES['owner']['contain_area_ratio'])
    for item in questions:
        overlap = _intersection_area(box, item['box']) / _area(box)
        if overlap >= contain_ratio and _same_column(box, item['box'], columns):
            return _verdict(item['id'], 'contain')

    center_x, center_y = _center(box)
    for item in questions:
        x0, y0, x1, y1 = item['box']
        if x0 <= center_x <= x1 and y0 <= center_y <= y1:
            return _verdict(item['id'], 'center')

    nearest = _nearest_owner(box, questions, columns)
    if nearest is not None:
        return _verdict(nearest, 'nearest')

    if _crosses_gutter(box, columns):
        return _verdict('', 'cross_column')
    return _verdict('', 'orphan')


def _nearest_owner(box, questions, columns):
    """③ 邻接法：同栏、上方最近、间隙 ≤1.2×题框高、间隙内无其它题框。"""
    tolerance_ratio = float(RULES['owner']['nearest_overlap_tolerance_ratio'])
    gap_ratio = float(RULES['owner']['nearest_gap_ratio'])
    candidates = []
    for item in questions:
        if not _same_column(box, item['box'], columns):
            continue
        height = item['box'][3] - item['box'][1]
        gap = box[1] - item['box'][3]
        if gap < -tolerance_ratio * height:
            continue
        candidates.append((gap, item))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    gap, closest = candidates[0]
    height = closest['box'][3] - closest['box'][1]
    if gap < 0:
        # 图与题框轻微重叠：仍算"图在题下"，只是间隙按 0 处理
        gap = 0.0
    if gap > gap_ratio * height:
        return None
    # 间隙内不能有其它题框，否则"最近"没有意义（中间那道题才是主人）
    for _other_gap, other in candidates[1:]:
        if other['box'][3] > closest['box'][3] and other['box'][1] < box[1]:
            return None
    return closest['id']


# ---------------------------------------------------------------- 硬拒
def _gray(image):
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array


def perceptual_hash(image, size: int | None = None) -> np.ndarray:
    """自实现感知哈希：灰度 → 32×32 → 与均值比较得 1024 bit（不引入 imagehash）。"""
    side = int(size or RULES['hard_reject']['page_like']['hash_size'])
    gray = _gray(image)
    if gray is None or gray.size == 0:
        return np.zeros((side * side,), dtype=np.uint8)
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    return (small > mean).astype(np.uint8).flatten()


def hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    if left is None or right is None or left.shape != right.shape:
        return int(max(left.size if left is not None else 0,
                       right.size if right is not None else 0))
    return int(np.count_nonzero(left != right))


def hard_reject(crop, image_box, page_shape, label, page=None) -> dict:
    """4 条机器硬拒 + 2 条合同硬拒。返回
    ``{'rejected','reason','similarity_to_page','page_hamming','page_area_ratio','needs_review'}``。

    ``page_like``（整页截图当题图）用**三条判据取或**，任意一条命中即拒：

    ① 32×32 感知哈希汉明距离 ≤ ``hamming_max``
       —— 抓"整页缩略图被贴成一个**小区域**"这类内容级克隆；
    ② 面积 ≥ 页面积 ``page_area_ratio_min`` **且** 与整页相似度 ≥ ``page_similarity_min``
       —— 防止哈希被大面积留白拉偏；
    ③ 候选框四边到页边的距离都 ≤ 该方向页宽/页高的 ``page_margin_ratio_max``
       —— 抓"框**就是**整页"，纯几何、与内容统计特性无关。

    为什么必须有 ③：①② 都依赖内容统计特性，整页框内缩到一定程度就双双失效
    （实测 1000×1400 页、20px 内缩：汉明距离 108 > 64、面积 0.933、相似度 0.895）；
    而 ③ 只看"框是否贴着页边"。

    **③ 的边界不是"内缩 20~40px 一定拦住"，而是"四边到页边各 ≤5% 页宽/页高
    （实际由短边决定，等效面积约 ≥83%~87%）"**：同一 40px 内缩，在 1000×1400
    （高 margin 2.86%）会被拦，在 1000×686（高 margin 5.83%）会被放行；对开真实页
    4929×3382 上 40px（1.18%）拦、175px（5.17%）放行。按像素数理解这条护栏，
    换一个纵横比就会得出相反结论。
    反之只有 ③ 也不行：小区域里装整页缩略图时面积/页边判据完全不触发。

    ``table`` 区域**默认降级为待复核**（``table_region.enabled = false``）：
    需求允许"图表、统计图"作为题图（统计数据表本身就是统计题的题图），但表格数据
    本身也要求可转写为文本/LaTeX，两者冲突时交人判断 —— 硬拒是不可逆的误伤
    （人连"采用"都点不了），所以默认不拒、标 ``needs_review=True``。
    把 ``table_region.enabled`` 置 true 可恢复"一律硬拒"行为（保留该开关与阈值）。

    细条 ``formula``（面积 <2% 页面积 或 高 <5% 页高 或 高宽比 >3 / 宽高比 >6）硬拒；
    其余 formula 保留但标 ``needs_review=True`` —— "公式与题干混排的长框"可能是
    有效题图，一刀切拒会把正常题图也拒掉。
    """
    rules = RULES['hard_reject']
    result = {
        'rejected': False,
        'reason': '',
        'similarity_to_page': None,
        'page_hamming': None,
        'page_area_ratio': None,
        'needs_review': False,
    }
    box = _box_of(image_box)
    if box is None:
        return {**result, 'rejected': True, 'reason': 'invalid_box'}

    width = box[2] - box[0]
    height = box[3] - box[1]
    try:
        page_height, page_width = int(page_shape[0]), int(page_shape[1])
    except (TypeError, IndexError, ValueError):
        page_height, page_width = 0, 0

    page_rules = rules['page_like']
    page_area = float(page_width * page_height) if page_width and page_height else 0.0
    area_ratio = ((width * height) / page_area) if page_area > 0 else 0.0
    result['page_area_ratio'] = round(area_ratio, 4)

    # 先把度量算全（UI 与 QA 报表都要读 similarity_to_page，不能因判定提前返回而空着）
    distance = None
    similarity = None
    if page is not None and crop is not None:
        side = int(page_rules['hash_size'])
        distance = hamming_distance(perceptual_hash(crop, side), perceptual_hash(page, side))
        similarity = 1.0 - distance / float(side * side)
        result['page_hamming'] = distance
        result['similarity_to_page'] = round(similarity, 4)

    # ① 内容判据：小区域里装的是一张整页缩略图
    if distance is not None and distance <= int(page_rules['hamming_max']):
        return {**result, 'rejected': True, 'reason': 'page_like'}

    # ② 面积 + 相似度：防止哈希被大面积留白拉偏
    if similarity is not None and page_area > 0 \
            and area_ratio >= float(page_rules['page_area_ratio_min']) \
            and similarity >= float(page_rules['page_similarity_min']):
        return {**result, 'rejected': True, 'reason': 'page_like'}

    # ③ 几何判据：四边都贴着页边 → 框就是整页（与内容、页面尺寸都无关）
    if page_area > 0:
        margin_limit = float(page_rules['page_margin_ratio_max'])
        margins = (
            box[0] / float(page_width),
            box[1] / float(page_height),
            (page_width - box[2]) / float(page_width),
            (page_height - box[3]) / float(page_height),
        )
        if all(margin <= margin_limit for margin in margins):
            return {**result, 'rejected': True, 'reason': 'page_like'}

    region_label = str(label or '').lower()

    table_rules = rules.get('table_region') or {}
    if region_label == 'table':
        table_reason = str(table_rules.get('reason') or 'table_region')
        if bool(table_rules.get('enabled', False)):
            # 运营显式开启时才硬拒（默认关闭：见 docstring 的裁决理由）
            return {**result, 'rejected': True, 'reason': table_reason}
        # 默认：不拒，但必须交人判断（统计数据表可能正是该题的题图）
        result['needs_review'] = True

    if region_label == 'formula':
        formula_rules = rules['formula_only']
        formula_reason = str(formula_rules.get('reason') or 'formula_only')
        too_tall = height / width > float(formula_rules['vertical_ratio_max'])
        too_wide = width / height > float(formula_rules['horizontal_ratio_max'])
        too_thin = page_area > 0 and area_ratio < float(formula_rules['thin_area_ratio_max'])
        too_short = page_height > 0 and height < page_height * float(
            formula_rules['thin_height_ratio_max'])
        if too_tall or too_wide or too_thin or too_short:
            return {**result, 'rejected': True, 'reason': formula_reason}
        # 公式与题干混排的长框：可能是有效题图，保留但强制人看
        result['needs_review'] = True

    if page_area > 0:
        if area_ratio < float(rules['number_label']['area_min']) \
                and min(width, height) < float(rules['number_label']['side_min']):
            return {**result, 'rejected': True, 'reason': 'number_label'}

    if min(width, height) < float(rules['too_small']['side_min']):
        return {**result, 'rejected': True, 'reason': 'too_small'}

    return result


# ---------------------------------------------------------------- 候选组装
def _as_image(page_image):
    if page_image is None:
        return None
    if isinstance(page_image, str):
        try:
            return cv2.imdecode(np.fromfile(page_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            return None
    array = np.asarray(page_image)
    return array if array.size else None


def _crop_of(image, box):
    height, width = image.shape[:2]
    x0 = max(0, int(round(box[0])))
    y0 = max(0, int(round(box[1])))
    x1 = min(width, int(round(box[2])))
    y1 = min(height, int(round(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _sha1_of(image) -> str:
    if image is None:
        return ''
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


def _layout_regions(layout: dict) -> list[dict]:
    if not isinstance(layout, dict):
        return []
    detail = layout.get('regions_detail')
    if isinstance(detail, list) and detail:
        return [item for item in detail if isinstance(item, dict)]
    labels = layout.get('image_labels')
    boxes = layout.get('image_boxes')
    if isinstance(boxes, list):
        return [
            {'label': (labels[index] if isinstance(labels, list) and index < len(labels) else 'image'),
             'score': 1.0, 'box': box}
            for index, box in enumerate(boxes)
        ]
    return []


def collect(page_image, page_index: int, records, layout=None) -> list[dict]:
    """产出本页的题图候选（``illustration_candidates``）。

    ``page_image`` 可以是 numpy 图或图片路径；``layout`` 为
    ``layout_arbiter.detect(img, with_regions=True)`` 的返回。不传 layout 时
    现场调一次版面裁判（模型缺失则返回空表 —— "没算"不是"没图"，调用方要靠
    ``page['illustration_ready']`` 区分）。
    """
    image = _as_image(page_image)
    if image is None:
        return []
    if layout is None:
        try:
            import layout_arbiter
            layout = layout_arbiter.detect(image, with_regions=True)
        except Exception:  # noqa: BLE001 - 版面裁判是可选依赖，坏了不能拖垮候选
            layout = None
    regions = _layout_regions(layout or {})
    if not regions:
        return []

    rules = RULES['image_regions']
    labels = {str(label).lower() for label in rules.get('labels') or []}
    score_min = float(rules.get('score_min', 0.5))
    area_ratio_min = float(rules.get('area_ratio_min', 0.0002))
    side_min = float(rules.get('side_min', 64))
    height, width = image.shape[:2]
    page_area = float(max(1, width * height))
    columns = (layout or {}).get('columns') or []

    candidates = []
    for region in regions:
        label = str(region.get('label') or '').lower()
        if label not in labels:
            continue
        score = float(region.get('score') or 0.0)
        if score < score_min:
            continue
        box = _box_of(region)
        if box is None:
            continue
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        if (box_width * box_height) < page_area * area_ratio_min:
            continue
        if min(box_width, box_height) < side_min:
            continue

        crop = _crop_of(image, box)
        verdict = hard_reject(crop, box, (height, width), label, page=image)
        owner = assign_owner(box, records, columns)
        candidates.append({
            'id': f'ic-{int(page_index)}-{len(candidates) + 1}',
            'bbox': [int(round(value)) for value in box],
            'label': label,
            'layout_score': round(score, 4),
            'owner_id': owner['owner_id'],
            'owner_method': owner['owner_method'],
            'owner_confidence': owner['owner_confidence'],
            'reject_reason': verdict['reason'],
            'needs_review': bool(verdict['needs_review']),
            'similarity_to_page': verdict['similarity_to_page'],
            'page_hamming': verdict['page_hamming'],
            'page_area_ratio': verdict['page_area_ratio'],
            'crop_sha1': _sha1_of(crop),
            'confirmed': False,
        })
    return candidates
