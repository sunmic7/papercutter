# -*- coding: utf-8 -*-
"""页级一致性机检：漏题检测（纯函数，无 IO）。

为什么单独做这一层
------------------
漏题是整条链路里**唯一没有错误信号**的失效：少切一条题，数据只是变少，
``_k12_validation`` 照样判"合格"，抽样也抽不到（它不在池子里）。
所以必须有跨记录、跨页的机检，用两个"结构必然性"兜住：

1. 大题号连续性：本页首个印刷大题号必须接上"全书阅读顺序中上一页的最大题号"
   （允许 +0 跨页续写）。跳号意味着中间整段可能被漏切。
2. 页题量偏离：该页题数偏离同书均值 > ``SIGMA_K`` 个 σ（且该页题数 ≥1）。

命中后该页**所有** ``region_role=='question'`` 记录升 L2（"漏了哪一题"本身无法定位），
并写入 ``quality_report.json`` 的 ``suspected_missing`` 段。这属于"告警不阻断"：
它有假阳性（单元末页、纯例题页、整页阅读材料），但假阴性的代价是客户发现少题。

分母口径
--------
``book_density_stats`` 的分母是"同书已确认题页"：只算有题记录的 question 页，
排除答案页（本来就在 ``answer_pages`` 里）、忽略页与零题页。
否则封面/答案页会把均值拉塌，导致处处告警。
"""
from __future__ import annotations

import math
import re

SIGMA_K = 2.0                 # §8.7：偏离同书均值 2σ 以上告警
MAJOR_GAP_TOLERANCE = 1       # 允许 +0（跨页续写）与 +1（正常递进）
MIN_SIGMA_FLOOR = 0.5         # σ 过小时不判偏离（避免 1 题之差也告警）

_CN_DIGITS = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9,
}
_MAJOR_PREFIX_RE = re.compile(
    r'^\s*(?:第\s*)?([一二三四五六七八九十百]+|\d{1,3})(?=\s*(?:大题|[、.．:：]))'
)
_CN_UNITS = {'十': 10, '百': 100}


def cn_to_number(text: str) -> int | None:
    """中文数字 → 整数：'三'→3，'十二'→12，'二十三'→23，'一百'→100。"""
    value = str(text or '').strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    total = 0
    current = 0
    saw_unit = False
    for char in value:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[char]
            current = 0
            saw_unit = True
        else:
            return None
    if saw_unit:
        return total + current
    return current or None


def major_index(label: str) -> int | None:
    """把大题标记规范成序号：'三'/'三、'/'3.'/'第二大题' → 3。"""
    value = str(label or '').strip()
    if not value:
        return None
    match = _MAJOR_PREFIX_RE.match(value)
    if match:
        return cn_to_number(match.group(1))
    return cn_to_number(value)


def _record_major_label(record: dict) -> str:
    """单条记录上的印刷大题标记：优先 ``major_number``，其次 ``major_title`` 前缀。"""
    if not isinstance(record, dict):
        return ''
    number = str(record.get('major_number') or '').strip()
    if number:
        return number
    title = str(record.get('major_title') or record.get('major') or '').strip()
    if not title:
        return ''
    match = _MAJOR_PREFIX_RE.match(title)
    if match:
        return match.group(1)
    return ''


def extract_printed_majors(page_records: list[dict]) -> list[str]:
    """从 ``major_number`` / ``major_title`` 抽印刷大题号，去重保序。

    去重是必要的：同一大题下每道小题都带着同一个 ``major_number``，
    不去重会把"一、一、一"当成三个大题。保序保证"首个大题号"可解释。
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for record in page_records or []:
        label = _record_major_label(record)
        if not label:
            continue
        key = str(major_index(label) if major_index(label) is not None else label)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(label)
    return ordered


def _question_records(page: dict) -> list[dict]:
    records = page.get('questions') if isinstance(page, dict) else None
    if not isinstance(records, list):
        return []
    return [
        record for record in records
        if isinstance(record, dict) and str(record.get('region_role', 'question')) == 'question'
    ]


def _ignored_images(project: dict) -> set[str]:
    keys = set()
    for page in (project or {}).get('ignored_pages') or []:
        if isinstance(page, dict):
            image = str(page.get('image') or '').replace('\\', '/')
        else:
            image = str(page or '').replace('\\', '/')
        if image:
            keys.add(image)
    return keys


def book_density_stats(project: dict) -> dict:
    """同书每页题数的 mean/std（分母只含"有题记录的 question 页"）。"""
    project = project if isinstance(project, dict) else {}
    ignored = _ignored_images(project)
    counts: list[int] = []
    skipped: list[int] = []
    for page_index, page in enumerate(project.get('pages') or []):
        if not isinstance(page, dict):
            continue
        image = str(page.get('image') or '').replace('\\', '/')
        if image and image in ignored:
            skipped.append(page_index)
            continue
        if str(page.get('status') or '') in {'ignored', 'skipped'}:
            skipped.append(page_index)
            continue
        total = len(_question_records(page))
        if total <= 0:
            skipped.append(page_index)   # 封面/答案页/零题页不参与均值
            continue
        counts.append(total)

    page_count = len(counts)
    mean = round(sum(counts) / float(page_count), 4) if page_count else 0.0
    if page_count >= 2:
        variance = sum((value - mean) ** 2 for value in counts) / float(page_count)
        std = round(math.sqrt(variance), 4)
    else:
        std = 0.0
    return {
        'mean': mean,
        'std': std,
        'pages': page_count,
        'counts': counts,
        'skipped_pages': skipped,
        'sigma_k': SIGMA_K,
    }


def _major_sequence_check(project: dict, page_index: int, majors: list[str]) -> dict:
    name = 'major_sequence'
    if not majors:
        return {'name': name, 'ok': True, 'skipped': True,
                'detail': '本页无印刷大题号（封面/答案页/纯图页），跳过'}

    first = major_index(majors[0])
    if first is None:
        return {'name': name, 'ok': True, 'skipped': True,
                'detail': f'首个大题号无法解析：{majors[0]!r}，跳过'}

    previous_max = None
    previous_page = None
    for index in range(min(page_index, len(project.get('pages') or [])) - 1, -1, -1):
        page = project['pages'][index]
        found = [major_index(label) for label in extract_printed_majors(page.get('questions') or [])]
        found = [value for value in found if value is not None]
        if found:
            previous_max = max(found)
            previous_page = index
            break

    if previous_max is None:
        return {'name': name, 'ok': True, 'skipped': False,
                'detail': f'本页首题 {majors[0]}，前面没有可接续的大题号（本册首个题页）'}

    gap = first - previous_max
    if gap < 0:
        return {'name': name, 'ok': True, 'skipped': False,
                'detail': f'本页首题 {majors[0]}，上一页（第 {previous_page + 1} 页）结束于 '
                          f'{previous_max}，判为新单元重新编号，不告警'}
    if gap > MAJOR_GAP_TOLERANCE:
        return {'name': name, 'ok': False, 'skipped': False,
                'detail': f'本页首题 {majors[0]}，上一页（第 {previous_page + 1} 页）结束于 '
                          f'{previous_max}，中间缺 {gap - 1} 个大题，疑漏切'}
    return {'name': name, 'ok': True, 'skipped': False,
            'detail': f'本页 {majors[0]} 接上第 {previous_page + 1} 页的 {previous_max}（+{gap}）'}


def _density_check(question_count: int, stats: dict) -> dict:
    name = 'question_density'
    mean = float(stats.get('mean') or 0.0)
    std = float(stats.get('std') or 0.0)
    sigma_k = float(stats.get('sigma_k') or SIGMA_K)
    if int(stats.get('pages') or 0) < 2 or std <= 0:
        return {'name': name, 'ok': True, 'skipped': True,
                'detail': f'本页 {question_count} 题；同书可比较题页不足或 σ=0，无法判定'}
    deviation = abs(question_count - mean)
    outlier = question_count >= 1 and deviation > sigma_k * max(std, MIN_SIGMA_FLOOR)
    return {
        'name': name,
        'ok': not outlier,
        'skipped': False,
        'detail': f'本页 {question_count} 题，同书均值 {round(mean, 2)}，σ={round(std, 2)}',
        'deviation': round(deviation, 4),
    }


def check_page(project: dict, page_index: int, stats: dict | None = None) -> dict:
    """返回 ``{'ok', 'checks': [{'name','ok','detail'}], 'majors', 'question_density'}``。

    ``stats`` 可传入 ``book_density_stats(project)`` 的预算结果（批处理逐页调用时
    避免每页重算全书均值，否则是 O(页数²)）。
    """
    project = project if isinstance(project, dict) else {}
    pages = project.get('pages') or []
    if not isinstance(page_index, int) or page_index < 0 or page_index >= len(pages):
        return {
            'ok': True,
            'checks': [],
            'majors': [],
            'question_density': 0,
            'warnings': [f'页码越界：{page_index}'],
        }

    page = pages[page_index] if isinstance(pages[page_index], dict) else {}
    records = _question_records(page)
    majors = extract_printed_majors(page.get('questions') or [])
    density_stats = stats if isinstance(stats, dict) else book_density_stats(project)
    checks = [
        _major_sequence_check(project, page_index, majors),
        _density_check(len(records), density_stats),
    ]
    return {
        'ok': all(check.get('ok') for check in checks),
        'checks': checks,
        'majors': majors,
        'question_density': len(records),
    }
