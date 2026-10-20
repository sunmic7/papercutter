# -*- coding: utf-8 -*-
"""分级复核策略（纯函数，无 IO）。

职责
----
把「一条题记录该不该给人看」这件事算成 ``L0 / L1 / L2`` 三档，并给出规则码，
供 UI 徽标、交付门禁、抽检分层共用。判定只读现成字段，**不新增信号源、不发请求、
不读工程文件** —— 这是它能被单测覆盖的前提。

三条硬口径（改一处就会整批算错，动之前先读架构文档 §8.2 与 §12.3）
---------------------------------------------------------------
1. 配对判定一律用**原始分合取**：``path_match AND score>=180 AND margin>=80``。
   ``display_confidence`` 只供展示（``clamp(score/300, 0, 1)``），任何判定不得读它。
   把它当阈值会退化成单阈值放行：路径不一致但 score=300 的误配对会被放过。
2. L0 的充要条件里**显式包含** ``margin >= 120``，不是"没命中 L1 就自动 L0"。
   否则未配对记录（``pair == {}``，margin=0）会被误判免检 —— 这是最容易写错的一行。
3. ``page['illustration_ready']`` 为假时**不得**命中 ``no_image_candidate``。
   候选还没算时该条件恒真，会把 100% 的题判成 L2，等于分级失效。

阈值集中在 ``THRESHOLDS`` 这一张表里（D1 决议值），代码中不出现裸数字；
需要按批换口径时由批处理层覆盖表项，函数本身保持纯。
"""
from __future__ import annotations

import re
from collections import Counter

RULE_VERSION = 'review-levels-v1'

# D1 决议的阈值。值是业务决议，不是可现场调的参数：改它等于改复核口径，
# 必须同时升 RULE_VERSION，否则历史 review_trace 无法解释。
THRESHOLDS = {
    # 配对判定（三条件合取，全部用原始分）
    'pair_score_min': 180.0,
    'pair_margin_min': 80.0,
    'pair_margin_l0': 120.0,
    # 仅用于展示的归一化尺度，禁止参与判定
    'display_confidence_scale': 300.0,
    # 同页题量（L1 many_subquestions）
    'many_subquestions': 6,
    # 新书前置页（L2 new_book_front，口径 U7：按每本前 20 页）
    'new_book_front_pages': 20,
    # 跨栏判定：题框跨过栏缝后，两侧各占题框宽度的比例下限
    'cross_column_overlap': 0.20,
    # 难度分布坍缩告警
    'difficulty_collapse_ratio': 0.60,
    # 目标占比（只用于 batch_stats 报"是否达标"，不参与单条判定）
    'l2_ratio_target': 0.15,
    'l0_ratio_range': (0.60, 0.75),
}

LEVELS = ('L0', 'L1', 'L2')
DEFAULT_LEVEL = 'L2'  # 未计算即需人看（§12.3 第 4 条）

# L2 规则码（顺序即优先级，便于人读；判定本身是"命中任一"）
L2_REASONS = (
    'minor_inferred',
    'region_group',
    'region_continuation',
    'cross_column',
    'pair_fail',
    'no_image_candidate',
    'image_owner_ambiguous',
    'analysis_fail',
    'enrich_failed',
    'new_book_front',
    'batch_degraded',
    'page_density_outlier',
    'major_sequence_gap',
)
# L1 规则码
L1_REASONS = (
    'pair_margin_low',
    'many_subquestions',
    'multi_part_answer',
    'image_owner_nearest',
    'analysis_unverifiable',
)
L0_REASON = 'default'

AMBIGUOUS_OWNER_METHODS = ('cross_column', 'orphan')
GROUP_ROLES = ('group', 'continuation')
FAILED_ENRICH_STATUS = ('failed', 'rejected', 'stale')
MULTI_PART_TYPES = ('解答', '应用', '证明', '计算')

# 小问标记：(1) （2） ① ② …
_SUBQUESTION_RE = re.compile(r'[（(]\s*[1-9１-９]\s*[）)]|[①②③④⑤⑥⑦⑧⑨⑩]')


# ---------------------------------------------------------------- 配对
def _pair_of(record: dict) -> dict:
    pair = record.get('pair')
    return pair if isinstance(pair, dict) else {}


def pair_confidence(score: float) -> float:
    """展示用置信度：``clamp(score/300, 0, 1)``。**任何判定都不得读它。**"""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    scale = float(THRESHOLDS['display_confidence_scale'])
    return max(0.0, min(1.0, value / scale))


def pair_ok(record: dict) -> bool:
    """D1：三条件合取，全部用原始分。不要写成单阈值，不要读 display_confidence。

    ``path_match`` 必须是**布尔真值本身**（``is True``），不能只判 truthy：
    字符串 ``'false'`` / ``'0'`` 都是 truthy，用 ``bool()`` 会把"路径不一致"放行成
    L0 —— 这是最危险的方向（该看的没看）。链路里的 pair 来自 JSON 布尔，正常不触发；
    但上游一旦把标志写成字符串（配置、表单、旧版导出），这一行就是唯一防线。
    """
    pair = _pair_of(record)
    try:
        score = float(pair.get('score', 0.0) or 0.0)
        margin = float(pair.get('margin', 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return pair.get('path_match') is True \
        and score >= THRESHOLDS['pair_score_min'] \
        and margin >= THRESHOLDS['pair_margin_min']


def pair_margin(record: dict) -> float:
    """配对余量原始分。未配对返回 0.0（因此未配对记录不可能进 L0）。"""
    try:
        return float(_pair_of(record).get('margin', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- 版面辅助
def _column_bounds(columns) -> list[tuple[float, float]]:
    """把 ``detect()['columns']`` 规范成 ``[(x0, x1)]``，容错 list/tuple/dict 三种写法。"""
    bounds: list[tuple[float, float]] = []
    for item in columns or []:
        pair = None
        if isinstance(item, dict):
            if {'x0', 'x1'} <= set(item):
                pair = (item.get('x0'), item.get('x1'))
            elif 'bounds' in item and isinstance(item['bounds'], (list, tuple)) \
                    and len(item['bounds']) >= 2:
                pair = (item['bounds'][0], item['bounds'][1])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pair = (item[0], item[1])
        if pair is None:
            continue
        try:
            x0, x1 = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        if x1 <= x0:
            continue
        bounds.append((x0, x1))
    bounds.sort()
    return bounds


def column_gutters(columns) -> list[float]:
    """栏缝中心 x：相邻两栏的交界中点。栏数 <2 时返回空表。"""
    bounds = _column_bounds(columns)
    return [
        (left[1] + right[0]) / 2.0
        for left, right in zip(bounds, bounds[1:])
    ]


def _bbox(record: dict):
    box = record.get('bbox')
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crosses_gutter(record: dict, gutters, min_overlap: float | None = None) -> bool:
    """题框宽度跨过栏缝，且栏缝两侧各占题框宽度的比例 ≥ ``min_overlap``。"""
    box = _bbox(record)
    if box is None or not gutters:
        return False
    limit = THRESHOLDS['cross_column_overlap'] if min_overlap is None else float(min_overlap)
    x0, _y0, x1, _y1 = box
    width = x1 - x0
    for gutter in gutters:
        if not (x0 < gutter < x1):
            continue
        ratio = min(gutter - x0, x1 - gutter) / width
        if ratio >= limit:
            return True
    return False


# ---------------------------------------------------------------- 页面字段读取
def _page_questions(page: dict) -> list[dict]:
    records = page.get('questions') if isinstance(page, dict) else None
    if not isinstance(records, list):
        return []
    return [
        record for record in records
        if isinstance(record, dict) and str(record.get('region_role', 'question')) == 'question'
    ]


def _page_check_failed(page: dict, name: str) -> bool:
    """读 ``page['page_consistency']`` 里已算好的页级结论（不在这里重算，避免两套口径）。"""
    consistency = page.get('page_consistency') if isinstance(page, dict) else None
    if not isinstance(consistency, dict):
        return False
    for check in consistency.get('checks') or []:
        if not isinstance(check, dict):
            continue
        if str(check.get('name', '')) == name and check.get('ok') is False:
            return True
    return False


def _owned_candidates(record: dict) -> list[dict]:
    items = record.get('illustration_candidates')
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _has_own_candidate(record: dict) -> bool:
    record_id = str(record.get('id') or '')
    if not record_id:
        return False
    return any(str(item.get('owner_id') or '') == record_id for item in _owned_candidates(record))


def _has_confirmed_illustration(record: dict) -> bool:
    items = record.get('illustrations')
    return bool(items) if isinstance(items, list) else bool(items)


def _multi_part_answer(record: dict) -> bool:
    if str(record.get('question_type', '')) not in MULTI_PART_TYPES:
        return False
    text = str(record.get('text', '') or '')
    return len(set(_SUBQUESTION_RE.findall(text))) >= 2


def _degraded_batch(record: dict, enrichment: dict | None) -> bool:
    """D5 批级/层级降级：该记录的 batch_id 或 stratum_id 出现在 degraded 表里。"""
    if not isinstance(enrichment, dict):
        return False
    degraded = enrichment.get('degraded')
    if not isinstance(degraded, dict) or not degraded:
        return False
    keys = [
        str(enrichment.get('batch_id') or '').strip(),
        str(enrichment.get('stratum_id') or '').strip(),
    ]
    record_batch = str(record.get('batch_id') or '').strip()
    if record_batch:
        keys.append(record_batch)
    return any(key and key in degraded for key in keys)


# ---------------------------------------------------------------- 判定主体
def evaluate(record: dict, page: dict, *, page_index: int = 0, columns=None,
             sibling_count: int = 0, is_new_book: bool = False,
             book_page_count: int = 0, enrichment: dict | None = None) -> dict:
    """算出分级明细：``{'level', 'reasons', 'warnings', 'rule_version'}``。

    与 :func:`compute_review_level` 的唯一区别是它把"规则被跳过"的告警也返回，
    供审计与抽检面板展示。判定逻辑本身只有这一份。
    """
    record = record if isinstance(record, dict) else {}
    page = page if isinstance(page, dict) else {}
    warnings: list[str] = []
    l2: list[str] = []
    l1: list[str] = []

    # ① L2
    if record.get('minor_number_inferred') is True:
        l2.append('minor_inferred')
    role = str(record.get('region_role', 'question') or 'question')
    if role == 'group':
        l2.append('region_group')
    elif role == 'continuation' or str(record.get('continuation_of') or '').strip():
        l2.append('region_continuation')

    if columns is None:
        # 版面裁判不可用时这条规则必须**跳过并记 warn**，不能当成"没跨栏"
        if _bbox(record) is not None:
            warnings.append('cross_column_skipped:no_columns')
    elif crosses_gutter(record, column_gutters(columns)):
        l2.append('cross_column')

    if not pair_ok(record):
        l2.append('pair_fail')

    if page.get('illustration_ready') is True \
            and not _has_own_candidate(record) \
            and not _has_confirmed_illustration(record):
        l2.append('no_image_candidate')

    if any(str(item.get('owner_method') or '') in AMBIGUOUS_OWNER_METHODS
           for item in _owned_candidates(record)):
        l2.append('image_owner_ambiguous')

    analysis_check = record.get('analysis_check')
    analysis_status = ''
    if isinstance(analysis_check, dict):
        analysis_status = str(analysis_check.get('status') or '')
    if analysis_status == 'fail':
        l2.append('analysis_fail')

    if str(record.get('enrich_status') or '') in FAILED_ENRICH_STATUS:
        l2.append('enrich_failed')

    if is_new_book and int(page_index) < int(THRESHOLDS['new_book_front_pages']):
        l2.append('new_book_front')

    if _degraded_batch(record, enrichment):
        l2.append('batch_degraded')

    if _page_check_failed(page, 'question_density'):
        l2.append('page_density_outlier')
    if _page_check_failed(page, 'major_sequence'):
        l2.append('major_sequence_gap')

    # ② L1
    if pair_ok(record) and pair_margin(record) < THRESHOLDS['pair_margin_l0']:
        l1.append('pair_margin_low')

    siblings = int(sibling_count or 0)
    if siblings <= 0:
        siblings = len(_page_questions(page))
    if siblings >= int(THRESHOLDS['many_subquestions']):
        l1.append('many_subquestions')

    if _multi_part_answer(record):
        l1.append('multi_part_answer')

    if any(
        str(item.get('owner_method') or '') == 'nearest' and item.get('confirmed') is True
        for item in _owned_candidates(record)
    ):
        l1.append('image_owner_nearest')

    if analysis_status == 'not_checkable':
        l1.append('analysis_unverifiable')

    # ③ 归并：命中任一 L2 → L2；否则任一 L1 → L1；否则要求配对余量充足才 L0
    if l2:
        level = 'L2'
        reasons = _dedupe(l2)
    elif l1:
        level = 'L1'
        reasons = _dedupe(l1)
    elif pair_ok(record) and pair_margin(record) >= THRESHOLDS['pair_margin_l0']:
        level = 'L0'
        reasons = [L0_REASON]
    else:
        # 走到这里只有一种可能：pair 被判 ok 但余量 < 80（不可能，已被 pair_fail 拦下），
        # 或 pair 未配对（已被 pair_fail 拦下）。保留 L2 是"宁可多看"的兜底。
        level = DEFAULT_LEVEL
        reasons = ['pair_fail']

    return {
        'level': level,
        'reasons': reasons,
        'warnings': warnings,
        'rule_version': RULE_VERSION,
    }


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def compute_review_level(record: dict, page: dict, *, page_index: int = 0, columns=None,
                         sibling_count: int = 0, is_new_book: bool = False,
                         book_page_count: int = 0,
                         enrichment: dict | None = None) -> tuple[str, list[str]]:
    """返回 ``(level, reasons)``。level ∈ {'L0','L1','L2'}。

    参数
    ----
    record / page:
        题记录与所在页（page 提供 ``illustration_ready`` / ``page_consistency``）。
    page_index:
        页在 question 页列表中的序号，用于 ``new_book_front``。
    columns:
        ``layout_arbiter.detect()['columns']``。传 ``None`` 表示版面裁判不可用：
        ``cross_column`` 规则会被跳过并记 warn（见 :func:`evaluate`），
        **不能**当成"没跨栏"。
    sibling_count:
        同页 ``region_role=='question'`` 记录数；传 0 时从 ``page['questions']`` 现算。
    is_new_book / book_page_count:
        新书前置页信号（口径 U7：按"每本前 20 页"）。
    enrichment:
        ``project['enrichment']``，用于 D5 批级降级判定（签名之外的唯一可选通道）。
    """
    detail = evaluate(
        record, page,
        page_index=page_index,
        columns=columns,
        sibling_count=sibling_count,
        is_new_book=is_new_book,
        book_page_count=book_page_count,
        enrichment=enrichment,
    )
    return detail['level'], list(detail['reasons'])


# ---------------------------------------------------------------- 批级统计
def batch_stats(records: list[dict]) -> dict:
    """批级分级统计：目标 L2 ≤ 15%，L0 60~75%。

    返回值里的 ``l2_within_target`` / ``l0_within_target`` 只报口径是否达标，
    不做任何自动降级 —— 降级动作属于抽检（D5），不在这里。
    """
    counts = Counter()
    reasons = Counter()
    new_book_front = 0
    total = 0
    for record in records or []:
        if not isinstance(record, dict):
            continue
        total += 1
        level = str(record.get('review_level') or DEFAULT_LEVEL)
        if level not in LEVELS:
            level = DEFAULT_LEVEL
        counts[level] += 1
        codes = record.get('review_reasons')
        if isinstance(codes, list):
            reasons.update(str(code) for code in codes)
            if 'new_book_front' in codes:
                new_book_front += 1

    def ratio(level: str) -> float:
        return round(counts[level] / total, 4) if total else 0.0

    l0_low, l0_high = THRESHOLDS['l0_ratio_range']
    return {
        'total': total,
        'L0': counts['L0'],
        'L1': counts['L1'],
        'L2': counts['L2'],
        'l0_ratio': ratio('L0'),
        'l1_ratio': ratio('L1'),
        'l2_ratio': ratio('L2'),
        'l2_within_target': ratio('L2') <= float(THRESHOLDS['l2_ratio_target']),
        'l0_within_target': l0_low <= ratio('L0') <= l0_high,
        'new_book_front': new_book_front,
        'by_reason': dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        'rule_version': RULE_VERSION,
    }


# ---------------------------------------------------------------- 难度分布
def difficulty_index(value) -> int | None:
    """把难度规范成 1~5 的整数；非 1~5 值域（含 '中'/'易'/空）返回 None。

    内部一律用整数，导出映射由交付层负责（本函数不做三档/五档转换）。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if value != int(value):
            return None
        number = int(value)
    else:
        text = str(value or '').strip()
        if not text.isdigit():
            return None
        number = int(text)
    return number if 1 <= number <= 5 else None


def check_difficulty_distribution(records: list[dict]) -> dict:
    """D4：任一档占比 > 60% → 判标签失效（模型在偷懒）。

    返回 ``{'scheme','buckets','dominant_ratio','collapsed','action','invalid'}``。
    发现 ``collapsed`` 时调用方只需重跑 ``difficulty`` 一个字段，不是整条记录。
    """
    buckets = {str(index): 0 for index in range(1, 6)}
    invalid: list[str] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        index = difficulty_index(record.get('difficulty'))
        if index is None:
            invalid.append(str(record.get('difficulty', '')))
            continue
        buckets[str(index)] += 1

    valid_total = sum(buckets.values())
    dominant = max(buckets.values()) if buckets else 0
    dominant_ratio = round(dominant / valid_total, 4) if valid_total else 0.0
    collapsed = valid_total > 0 and dominant_ratio > float(THRESHOLDS['difficulty_collapse_ratio'])
    return {
        'scheme': '1-5',
        'buckets': buckets,
        'total': valid_total,
        'invalid_count': len(invalid),
        'invalid': invalid[:20],
        'dominant_ratio': dominant_ratio,
        'collapsed': collapsed,
        'action': 'rerun' if collapsed else 'ok',
        'rule_version': RULE_VERSION,
    }
