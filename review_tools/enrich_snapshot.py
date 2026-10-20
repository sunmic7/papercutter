# -*- coding: utf-8 -*-
"""快照：冻结人确认过的输入 / 失效判定 / 单页分级回写（T03）。

一句话：**人确认一次 → 冻结成不可变快照 → 之后所有 AI 字段都基于快照生成**。
所以快照里只放"人承诺过的输入"，**绝不放 AI 输出** —— AI 输出是可重算的派生
数据，混进快照后就再也说不清"这条解析是哪个 prompt 算的"。

冻结什么（§3.1）
----------------
题框 ``bbox``、题目 OCR 文本 ``text``、答案文本 ``answer`` + ``answer_ids``、
已人工确认的题图框 ``illustration_boxes``（**不含** ``illustration_candidates``）、
共享材料段、续页片段，以及四个"口径版本"：``taxonomy_version`` /
``prompt_version`` / ``difficulty_scheme`` / ``book_meta_rev``。

失效粒度是 **record**，不是 page（§3.6）。改第 3 题的框只把第 3 题打 stale，
同页其它题保持原级 —— 这是"人工量不翻倍"的前提。

五条失效条件与处置（判定顺序即优先级，§3.5 + A3 决议）
------------------------------------------------------
===  ==============================  ================  ============
#    条件                            enrich_status     级别
===  ==============================  ================  ============
1    bbox 变                         ``stale``         **强制 L2**
2    text / answer / options 变      ``stale``         **强制 L2**
3    page_sha1 变（换源图/旋转）     ``stale``         **强制 L2**
4    taxonomy / prompt 版本升        ``kp_stale``/     **不降级**
                                     ``analysis_stale``
5    book_meta_rev 变（书级四字段）  ``stale``         **强制 L2**
===  ==============================  ================  ============

第 5 条按 A3 决议改成**降 L2**（先前实现按"人没改输入就不降级"处理，被 lead 撤回）：
书级字段一错污染全书，宁可重看一遍框，也不能让"旧学科算出的知识点"混进交付。

级别怎么来：**不在这里另写一套判定**，而是把 ``enrich_status`` 置好之后调
:func:`review_tools.review_levels.compute_review_level`。``stale`` 命中
``enrich_failed`` 规则自然得 L2，所以"分级单一真源"不用打折扣。

三条刻意的取舍（改之前先读架构文档 + QA Round 8）
-------------------------------------------------
1. **内容哈希只认内容身份**：``record_id + bbox + text + options + answer +
   page_sha1``（见 :data:`HASH_KEYS`）。位置（``page_index`` 及其嵌套副本）、
   诊断（``text_conf``/``text_engine``）、批次号（``batch_id``）都**不进哈希** ——
   否则插一页 = 全书重算、重跑一次 OCR = 整页失效、开新批次 = 缓存全废。
2. ``page_sha1`` 取不到（图缺失/换机）时**保留旧值**，不判失效。
   "读不到图"和"换了图"是两件事，把前者判成 stale 会让离机审查的工程集体失效；
   为让交付门禁能区分这两件事，快照另存 ``page_sha1_status``
   （``ok`` / ``unavailable``），**它不参与哈希**。
3. ``PAPERFLOW_ENRICH`` 之类的开关一律**现读环境变量**，不缓存到模块常量 ——
   缓存会让"关一轮再开"的单测和灰度切换读到旧值。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from review_tools import book_meta, review_levels

SNAPSHOT_VERSION = 'snapshot-v1'

# 源图指纹的缩放宽度：够辨认"是不是同一张页"，又不让 JPEG 重压的微差误判失效。
PAGE_HASH_SIDE = 160

# 内容哈希**只认内容身份**（架构 A3 决议口径②，2026-10-20 lead 拍板）：
# ``record_id + bbox + text + options + answer + page_sha1``。
#
# 为什么**不**把 ``page_index`` 或其它位置信息放进哈希：插一页/删一页/重排页序
# 只改变位置、不改变内容，但如果进了哈希，插入点之后的**每一题**都会换 cache key
# → 整本判 stale → 100% 进 L2。QA Round 8 的实测就是这么炸的
# （q1 内容一字未改，前面插一个空页 → hash 变 → L0 掉 L2，AI 字段全标 stale）。
#
# 位置信息（``page_index`` / ``shared_context_segments[*].page_index`` /
# ``continuations[*].page_index``）**照旧存进快照**供回溯与人工核对，只是不参与
# 哈希、也不参与失效判定（见 :func:`diff_snapshots` 的"位置无关比较"）。
#
# 口径版本四件套（taxonomy/prompt/difficulty_scheme/book_meta_rev）同样不进哈希：
# 它们由 ``diff_snapshots`` 单独判定（升版只重算受影响字段，不动级别），
# 而 T04 的 ``cache_key()`` 本来就显式接收 ``taxonomy_version``（架构 §9 T04 验收 2）。
HASH_KEYS = ('question_id', 'bbox', 'text', 'options', 'answer', 'page_sha1')

# 快照必须齐全的键：少一个就当"旧快照"，不能拿来比对
REQUIRED_KEYS = (
    'hash', 'version', 'created_at', 'kind', 'page_index', 'question_id',
    'bbox', 'page_sha1', 'page_sha1_status', 'text', 'options', 'answer', 'answer_ids',
    'answer_source', 'illustration_boxes', 'shared_context_segments',
    'continuations', 'profile', 'taxonomy_version', 'prompt_version',
    'difficulty_scheme', 'book_meta_rev',
)

# 影响交付的 AI 字段；失效时把它们的来源标 stale（不让"旧 AI 结果"冒充有效）
AI_FIELD_KEYS = ('question_type', 'knowledge_points', 'difficulty', 'analysis')

# diff 时的输入字段顺序 = 判定优先级（**只放内容，不放位置**）
INPUT_KEYS = (
    'bbox', 'text', 'options', 'answer', 'answer_ids',
    'illustration_boxes', 'shared_context_segments', 'continuations',
)

# 动作码
ACTION_FROZEN = 'frozen'
ACTION_UNCHANGED = 'unchanged'
ACTION_STALE = 'stale'
ACTION_BOOK_META_STALE = 'book_meta_stale'
ACTION_KP_STALE = 'kp_stale'
ACTION_ANALYSIS_STALE = 'analysis_stale'
ACTION_SKIPPED = 'skipped'

PAGE_HASH_CACHE_LIMIT = 512
_PAGE_HASH_CACHE: dict[tuple, str] = {}
_PAGE_HASH_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def _canonical_json(value) -> str:
    """跨进程稳定的序列化（与 ``online_review_sync`` 同范式）。

    **不要**用内建 ``hash()``：它每进程加盐，缓存会全废。
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


# ------------------------------------------------------------------ 哈希
def hash_of(snapshot: dict) -> str:
    """内容哈希 = :data:`HASH_KEYS` 的规范序列化 SHA256。

    只认"内容身份"，不认位置/诊断/批次（理由见模块 docstring 取舍 1）。
    缺键用 ``None`` 占位，保证任何半截快照也能算出一个确定的哈希。
    """
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    payload = {key: snapshot.get(key) for key in HASH_KEYS}
    return hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()



def _image_array(source):
    """把路径 / numpy 图规范成 BGR ndarray；失败返回 None（不抛）。"""
    if source is None:
        return None
    if not isinstance(source, (str, os.PathLike)):
        try:
            import numpy as np

            array = np.asarray(source)
        except Exception:  # noqa: BLE001 - 形状怪异就当读不到
            return None
        return array if array.size else None
    try:
        import cv2
        import numpy as np

        return cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - 缺 cv2 时降级为"取不到指纹"
        return None


def _image_sha1(image, side: int) -> str:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    resized = image
    if width > side:
        target_height = max(1, int(round(height * side / float(width))))
        resized = cv2.resize(image, (side, target_height), interpolation=cv2.INTER_AREA)
    digest = hashlib.sha1()
    # 拼上原始尺寸：换源图但缩略后像素偶然一致时仍能检出
    digest.update(f'{int(height)}x{int(width)}'.encode('ascii'))
    digest.update(np.ascontiguousarray(resized).tobytes())
    return digest.hexdigest()


def _fingerprint_cache_key(path: Path) -> tuple | None:
    """缓存键 = 路径 + mtime + 大小 + **文件首尾 64KB 的内容摘要**。

    QA Round 8 的 C2 反例：只按 ``(路径, mtime_ns, size)`` 做键时，"等长改写 +
    复原 mtime"（``copy2`` / ``rsync --times`` / 图片处理脚本的常态）会命中旧缓存
    → 换图不失效。加一段内容摘要后，只要文件内容变了键就变，而读 2×64KB 比
    解码整页图便宜两个数量级。
    """
    try:
        stat = path.stat()
        digest = hashlib.sha1()
        with path.open('rb') as handle:
            digest.update(handle.read(65536))
            if stat.st_size > 131072:
                handle.seek(-65536, os.SEEK_END)
                digest.update(handle.read(65536))
    except OSError:
        return None
    return (
        os.path.normcase(str(path)),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        digest.hexdigest()[:16],
    )


def page_sha1(source, *, side: int = PAGE_HASH_SIDE, use_cache: bool = True) -> str:
    """源图指纹：缩放到 ``side`` 宽 + 拼上原始 ``shape`` 再 sha1。

    作用与 ``layout_arbiter._image_key()`` 相同 —— 换源图（重扫/重新旋转）能检出，
    JPEG 重压的微小差异不会误判失效。

    缓存**只存正结果**（算出来的指纹），取不到（图缺失/读失败/解码失败）时
    一律返回 ``''`` 且**不写缓存**，所以"图后来出现了"不会读到陈旧的空值。
    交付/出包前的强校验请传 ``use_cache=False``，或调 :func:`clear_page_hash_cache`。
    """
    path = Path(source) if isinstance(source, (str, os.PathLike)) else None
    cache_key = None
    if path is not None and use_cache:
        cache_key = _fingerprint_cache_key(path)
        if cache_key is not None:
            cached = _PAGE_HASH_CACHE.get(cache_key)
            if cached is not None:
                return cached
    image = _image_array(source)
    if image is None:
        return ''
    try:
        value = _image_sha1(image, max(8, int(side)))
    except Exception:  # noqa: BLE001 - 指纹算不出来不该拦住保存
        return ''
    if cache_key is not None and value:
        # 只缓存**正结果**：空串代表"这会儿读不到图"（图正在拷贝 / 权限 / 临时
        # 不可用），把它缓存下来会让后续所有读取都拿到空值，图回来之后仍判不出
        # 指纹 —— 负缓存一旦命中就只能等进程重启或 clear_page_hash_cache()。
        with _PAGE_HASH_LOCK:
            if len(_PAGE_HASH_CACHE) >= PAGE_HASH_CACHE_LIMIT:
                _PAGE_HASH_CACHE.clear()
            _PAGE_HASH_CACHE[cache_key] = value
    return value


def clear_page_hash_cache() -> None:
    """清指纹缓存：单测用；交付/出包前的强校验路径也应先调一次。"""
    with _PAGE_HASH_LOCK:
        _PAGE_HASH_CACHE.clear()



def page_sha1_of(project: dict, page: dict, kind: str = 'question') -> str:
    """按工程根目录解析页图，再算 :func:`page_sha1`。取不到就返回 ``''``。"""
    if not isinstance(page, dict):
        return ''
    project = project if isinstance(project, dict) else {}
    root_key = 'answer_image_root' if str(kind) == 'answer' else 'image_root'
    root_value = str(project.get(root_key) or project.get('book_root') or '')
    if not root_value:
        return ''
    try:
        root = Path(root_value).resolve()
        rotation = int(page.get('rotation', 0) or 0) % 360
        relative_value = page.get('rotated_image', '') if rotation else page.get('image', '')
        relative = Path(str(relative_value or ''))
        if not str(relative):
            return ''
        if relative.is_absolute():
            return ''
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return ''
    return page_sha1(candidate)


# ------------------------------------------------------------------ 读字段
def _bbox_list(value) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        return [int(round(float(item))) for item in value]
    except (TypeError, ValueError, OverflowError):
        return []


def _float(value) -> float:
    try:
        return round(float(value or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _text_list(value) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def record_index(project: dict) -> dict[str, dict]:
    """``id -> 记录``：题目页与答案页一起索引（答案回写要按 id 找答案）。"""
    index: dict[str, dict] = {}
    project = project if isinstance(project, dict) else {}
    for pages_key, records_key in (('pages', 'questions'), ('answer_pages', 'answers')):
        for page in project.get(pages_key) or []:
            if not isinstance(page, dict):
                continue
            for record in page.get(records_key) or []:
                if isinstance(record, dict) and record.get('id'):
                    index.setdefault(str(record['id']), record)
    return index


def _record_position(project: dict) -> dict[str, int]:
    position: dict[str, int] = {}
    order = 0
    for pages_key, records_key in (('answer_pages', 'answers'), ('pages', 'questions')):
        for page in project.get(pages_key) or []:
            if not isinstance(page, dict):
                continue
            for record in page.get(records_key) or []:
                record_id = str((record or {}).get('id') or '') if isinstance(record, dict) else ''
                if record_id and record_id not in position:
                    position[record_id] = order
                order += 1
    return position


def answer_ids_of(project: dict, question_id: str) -> list[str]:
    """该题在 ``project['pairs']`` 里引用的答案 ID（去重保序）。"""
    ids: list[str] = []
    for pair in (project or {}).get('pairs') or []:
        if not isinstance(pair, dict):
            continue
        if str(pair.get('question_id', '')) != str(question_id):
            continue
        for value in pair.get('answer_ids') or []:
            text = str(value or '').strip()
            if text and text not in ids:
                ids.append(text)
    return ids


def _answer_text(record: dict) -> str:
    """答案记录的文字：正文答案优先，续页片段可能只有解析或题干。"""
    for key in ('answer', 'analysis', 'text'):
        value = str((record or {}).get(key) or '').strip()
        if value:
            return value
    return ''


def answer_segment(project: dict, record: dict) -> tuple[str, list[str]]:
    """拼出快照里的 ``answer`` 段与 ``answer_ids``。

    顺序按答案记录在工程里的自然位置（答案页顺序 + 页内顺序）。这里刻意
    不 import ``delivery_workspace.answer_chain`` —— 交付层属于 T05，快照不该
    跟着它变；两边顺序一致性由"答案页内坐标排序"这条简单规则保证。
    """
    record = record if isinstance(record, dict) else {}
    question_id = str(record.get('id') or '')
    ids = answer_ids_of(project, question_id) if question_id else []
    if not ids:
        return str(record.get('answer') or '').strip(), []
    position = _record_position(project)
    ordered = sorted(ids, key=lambda value: position.get(value, 10 ** 9))
    index = record_index(project)
    texts: list[str] = []
    for answer_id in ordered:
        text = _answer_text(index.get(answer_id) or {})
        if text:
            texts.append(text)
    return '\n'.join(texts), ordered


def illustration_boxes(record: dict) -> list[list[int]]:
    """人确认过的题图框。**不含** ``illustration_candidates``（那是机器候选）。"""
    boxes: list[list[int]] = []
    items = (record or {}).get('illustrations')
    if not isinstance(items, (list, tuple)):
        return boxes
    for item in items:
        box = item.get('bbox') if isinstance(item, dict) else item
        normalized = _bbox_list(box)
        if normalized:
            boxes.append(normalized)
    boxes.sort()
    return boxes


def _shared_context_segments(record: dict) -> list[dict]:
    segments = []
    for item in (record or {}).get('shared_context_segments') or []:
        if not isinstance(item, dict):
            continue
        box = _bbox_list(item.get('bbox'))
        if not box:
            continue
        try:
            page_index = int(item.get('page_index', -1))
        except (TypeError, ValueError):
            page_index = -1
        segments.append({'page_index': page_index, 'bbox': box})
    segments.sort(key=lambda entry: (entry['page_index'], entry['bbox']))
    return segments


def _continuations(project: dict, record_id: str) -> list[dict]:
    result = []
    if not record_id:
        return result
    for page_index, page in enumerate((project or {}).get('pages') or []):
        if not isinstance(page, dict):
            continue
        for record in page.get('questions') or []:
            if not isinstance(record, dict):
                continue
            if str(record.get('region_role', 'question') or '') != 'continuation':
                continue
            if str(record.get('continuation_of') or '') != record_id:
                continue
            box = _bbox_list(record.get('bbox'))
            if box:
                result.append({'page_index': int(page_index), 'bbox': box})
    result.sort(key=lambda entry: (entry['page_index'], entry['bbox']))
    return result


def project_meta(project: dict) -> dict:
    """口径版本四件套。``book_meta_rev`` 由当前书级字段现算（唯一真源）。"""
    project = project if isinstance(project, dict) else {}
    enrichment = project.get('enrichment')
    enrichment = enrichment if isinstance(enrichment, dict) else {}
    return {
        'taxonomy_version': str(enrichment.get('taxonomy_version') or ''),
        'prompt_version': str(enrichment.get('prompt_version') or ''),
        'batch_id': str(enrichment.get('batch_id') or ''),
        'difficulty_scheme': str(project.get('difficulty_scheme') or '1-5'),
        'book_meta_rev': book_meta.book_meta_rev(project),
    }


def page_index_of(project: dict, page: dict, kind: str = 'question') -> int:
    """找回页序号：先按对象同一性，再按 ``image`` 相对路径，最后兜底 0。"""
    pages_key = 'answer_pages' if str(kind) == 'answer' else 'pages'
    pages = (project or {}).get(pages_key) or []
    for index, candidate in enumerate(pages):
        if candidate is page:
            return index
    image = str((page or {}).get('image', '') or '')
    if image:
        for index, candidate in enumerate(pages):
            if isinstance(candidate, dict) and str(candidate.get('image', '') or '') == image:
                return index
    return 0


# ------------------------------------------------------------------ 冻结
def build_snapshot(project: dict, page: dict, record: dict, *, page_index: int = 0,
                   kind: str = 'question', page_hash: str | None = None) -> dict:
    """按 §3.1 组装一条快照（含 ``hash``）。不做任何失效判断。"""
    record = record if isinstance(record, dict) else {}
    meta = project_meta(project)
    answer_text, answer_ids = answer_segment(project, record)
    ocr = record.get('ocr') if isinstance(record.get('ocr'), dict) else {}
    resolved_hash = page_sha1_of(project, page, kind) if page_hash is None else str(page_hash or '')
    snapshot = {
        'version': SNAPSHOT_VERSION,
        'kind': str(kind or 'question'),
        'page_index': int(page_index or 0),
        'question_id': str(record.get('id') or ''),
        'created_at': _now(),
        'bbox': _bbox_list(record.get('bbox')),
        'page_sha1': resolved_hash,
        'text': str(record.get('text') or ''),
        'text_conf': _float(ocr.get('confidence', ocr.get('conf', 0.0))),
        'text_engine': str(ocr.get('engine') or ''),
        'options': _text_list(record.get('options')),
        'answer': answer_text,
        'answer_ids': answer_ids,
        'answer_source': (
            'paired' if answer_ids
            else ('manual' if str(record.get('answer') or '').strip() else 'missing')
        ),
        'shared_context_bbox': _bbox_list(record.get('shared_context_bbox')),
        'shared_context_segments': _shared_context_segments(record),
        'continuations': _continuations(project, str(record.get('id') or '')),
        'illustration_boxes': illustration_boxes(record),
        'profile': str((project or {}).get('profile') or 'general'),
        'taxonomy_version': meta['taxonomy_version'],
        'prompt_version': meta['prompt_version'],
        'difficulty_scheme': meta['difficulty_scheme'],
        'book_meta_rev': meta['book_meta_rev'],
        'batch_id': meta['batch_id'],
        # 交付门禁要能区分"图没变"与"读不到图"（§3.2 的非权威诊断位，不进哈希）
        'page_sha1_status': 'ok' if resolved_hash else 'unavailable',
    }
    snapshot['hash'] = hash_of(snapshot)
    return snapshot


def is_valid_snapshot(snapshot: dict) -> bool:
    """结构齐全 + 版本匹配 + 身份非空 + 内容哈希自洽。

    空 ``question_id`` 一律判旧（QA Round 8 的小洞）：下游按 id 关联快照与记录，
    空 id 的快照会跟任意记录糊在一起。
    """
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    if str(snapshot.get('version') or '') != SNAPSHOT_VERSION:
        return False
    if any(key not in snapshot for key in REQUIRED_KEYS):
        return False
    if not str(snapshot.get('question_id') or '').strip():
        return False
    stored = str(snapshot.get('hash') or '')
    if not stored:
        return False
    return hash_of(snapshot) == stored


def is_valid(record: dict, page: dict | None = None, *, project: dict | None = None,
             kind: str | None = None) -> bool:
    """该记录的快照是否可用于补全/交付。

    传 ``project`` + ``page`` 时**额外**校验源图指纹（换源图/旋转即失效）；
    不传就只校验结构与哈希，够单测和列表页用。
    """
    if not isinstance(record, dict):
        return False
    snapshot = record.get('snapshot')
    if not is_valid_snapshot(snapshot):
        return False
    if str(record.get('snapshot_id') or '') != str(snapshot.get('hash'))[:16]:
        return False
    if isinstance(page, dict) and isinstance(project, dict):
        resolved_kind = str(kind or snapshot.get('kind') or 'question')
        current = page_sha1_of(project, page, resolved_kind)
        stored = str(snapshot.get('page_sha1') or '')
        if current and stored and current != stored:
            return False
    return True


def snapshot_hash(record: dict) -> str:
    snapshot = (record or {}).get('snapshot')
    return str(snapshot.get('hash') or '') if isinstance(snapshot, dict) else ''


def _segment_boxes(value) -> list[list[int]]:
    """把 ``shared_context_segments`` / ``continuations`` 压成"只含题框"的列表。

    位置无关比较用：插一页会让这些片段里的 ``page_index`` 全部平移，但**内容
    （框）没变**，不该判失效。
    """
    boxes: list[list[int]] = []
    for item in value if isinstance(value, (list, tuple)) else []:
        box = item.get('bbox') if isinstance(item, dict) else item
        normalized = _bbox_list(box)
        if normalized:
            boxes.append(normalized)
    boxes.sort()
    return boxes


def _comparable(snapshot: dict, key: str):
    """取"参与失效判定"的字段内容（位置信息已被剥掉）。"""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    if key in ('shared_context_segments', 'continuations'):
        return _segment_boxes(snapshot.get(key))
    return snapshot.get(key)


def diff_snapshots(old: dict | None, new: dict) -> dict:
    """比对旧/新快照，返回 ``{'action','reasons','changed'}``。

    判定顺序即优先级（§3.5 + A3）：内容 1~2 → 源图 3 → 书级 5 → 版本 4 → 兜底。
    **位置不动**：``page_index`` 及片段里的 ``page_index`` 不参与任何比较 ——
    插页/删页/重排页序只改位置，不该让后续每道题重算（QA Round 8 的 A3）。
    """
    if not is_valid_snapshot(old):
        return {'action': ACTION_FROZEN, 'reasons': ['no_previous_snapshot'], 'changed': []}

    changed = [
        key for key in INPUT_KEYS
        if _canonical_json(_comparable(old, key)) != _canonical_json(_comparable(new, key))
    ]
    if changed:
        return {'action': ACTION_STALE, 'reasons': [f'{key}_changed' for key in changed],
                'changed': changed}

    old_page = str((old or {}).get('page_sha1') or '')
    new_page = str((new or {}).get('page_sha1') or '')
    if old_page and new_page and old_page != new_page:
        return {'action': ACTION_STALE, 'reasons': ['page_sha1_changed'], 'changed': ['page_sha1']}

    old_rev = str((old or {}).get('book_meta_rev') or '')
    new_rev = str((new or {}).get('book_meta_rev') or '')
    if old_rev and new_rev and old_rev != new_rev:
        return {'action': ACTION_BOOK_META_STALE, 'reasons': ['book_meta_rev_changed'],
                'changed': ['book_meta_rev']}

    if str((old or {}).get('taxonomy_version') or '') != str((new or {}).get('taxonomy_version') or ''):
        return {'action': ACTION_KP_STALE, 'reasons': ['taxonomy_version_changed'],
                'changed': ['taxonomy_version']}

    if str((old or {}).get('prompt_version') or '') != str((new or {}).get('prompt_version') or ''):
        return {'action': ACTION_ANALYSIS_STALE, 'reasons': ['prompt_version_changed'],
                'changed': ['prompt_version']}

    old_hash = str((old or {}).get('hash') or '')
    new_hash = str((new or {}).get('hash') or '')
    if old_hash and new_hash and old_hash != new_hash:
        # 兜底：HASH_KEYS 里还有上文没逐项比过的键（answer/page_sha1 已比过），
        # 出现这条只可能是"内容哈希口径变了"，按失效处理最安全。
        return {'action': ACTION_STALE, 'reasons': ['snapshot_input_changed'],
                'changed': ['snapshot']}

    return {'action': ACTION_UNCHANGED, 'reasons': [], 'changed': []}


def _mark_field_source(record: dict, keys, value: str) -> None:
    """只改"确实有来源"的键：没有来源的字段不凭空造一条记录。"""
    sources = record.get('field_source')
    sources = dict(sources) if isinstance(sources, dict) else {}
    for key in keys:
        current = str(sources.get(key) or '')
        if current and current != 'human':
            sources[key] = value
    record['field_source'] = sources


def refresh_level(project: dict, page: dict, record: dict, *, page_index: int | None = None,
                  kind: str = 'question', columns=None, enrichment: dict | None = None) -> tuple[str, list[str]]:
    """调 T01 的策略算级别并回写 ``review_level/review_reasons/review_trace``。

    **唯一**的分级实现是 :func:`review_tools.review_levels.compute_review_level`；
    本函数只负责传参与落盘，不复制任何规则。
    """
    index = page_index_of(project, page, kind) if page_index is None else int(page_index)
    resolved_enrichment = enrichment
    if resolved_enrichment is None and isinstance(project, dict):
        value = project.get('enrichment')
        resolved_enrichment = value if isinstance(value, dict) else None
    level, reasons = review_levels.compute_review_level(
        record, page,
        page_index=index,
        columns=columns,
        enrichment=resolved_enrichment,
    )
    record['review_level'] = level
    record['review_reasons'] = list(reasons)
    trace = record.get('review_trace')
    trace = dict(trace) if isinstance(trace, dict) else {}
    trace['computed_at'] = _now()
    trace['rule_version'] = review_levels.RULE_VERSION
    record['review_trace'] = trace
    return level, list(reasons)


def freeze_or_invalidate(project: dict, page: dict, record: dict, old: dict | None = None, *,
                         page_index: int | None = None, kind: str = 'question',
                         page_hash: str | None = None, columns=None,
                         enrichment: dict | None = None) -> dict:
    """冻结或失效**一条**记录的快照，并按需重算级别。

    返回 ``{'action','reasons','changed','snapshot_id','review_level','enrich_status'}``。
    ``action`` 取值见模块级 ``ACTION_*`` 常量。
    """
    if not isinstance(record, dict):
        return {'action': ACTION_SKIPPED, 'reasons': ['invalid_record'], 'changed': []}

    stored = record.get('snapshot')
    stored = stored if isinstance(stored, dict) else {}
    old_stored = (old or {}).get('snapshot')
    old_stored = old_stored if isinstance(old_stored, dict) else {}
    # 升级前留下的旧快照可能"判为不可信"（缺 C1 的 page_sha1_status 等新字段），
    # 但内容哈希仍可能与新算出来的一致 —— 这时要区分"真的要补跑"与"只是补个字段"。
    legacy_hash = str(stored.get('hash') or old_stored.get('hash') or '')

    previous = record.get('snapshot')
    if not is_valid_snapshot(previous):
        previous = (old or {}).get('snapshot')
    previous = previous if is_valid_snapshot(previous) else None

    index = page_index_of(project, page, kind) if page_index is None else int(page_index)
    snapshot = build_snapshot(project, page, record, page_index=index, kind=kind,
                              page_hash=page_hash)
    if previous and not snapshot['page_sha1']:
        # 读不到源图 ≠ 换了源图：保留旧指纹，避免离机审查时整本失效
        snapshot['page_sha1'] = str(previous.get('page_sha1') or '')
        snapshot['hash'] = hash_of(snapshot)

    decision = diff_snapshots(previous, snapshot)
    action = decision['action']

    attached_answer_ids = answer_ids_of(project, str(record.get('id') or ''))
    if action == ACTION_FROZEN and attached_answer_ids and (previous is not None or old):
        # "答案挂接过的题永不复冻"：已经有配对答案、只是旧快照坏了/是旧版本时，
        # 不能走 pending（形同"从零开始"）—— 那会掩盖"AI 字段是基于无答案快照
        # 算出来的"这件事。统一按 stale 处理，进重算/人工队列。
        action = ACTION_STALE
        decision = {'action': ACTION_STALE,
                    'reasons': ['answer_attached_refreeze'],
                    'changed': []}

    if action == ACTION_UNCHANGED:
        # 同输入再冻结：保留旧快照（created_at 与 hash 都不动），级别也不碰。
        # 但**位置字段要跟着走**：插页/移页后快照里的 page_index 必须是真的，
        # 否则它会一直撒谎；page_index 不在 HASH_KEYS 里，改它不影响 hash/valid。
        if 'page_index' in snapshot and int(previous.get('page_index', index)) != index:
            previous['page_index'] = index
        record['snapshot'] = previous
        record['snapshot_id'] = str(previous.get('hash'))[:16]
        return {
            'action': action,
            'reasons': [],
            'changed': [],
            'snapshot_id': record['snapshot_id'],
            'review_level': str(record.get('review_level') or ''),
            'enrich_status': str(record.get('enrich_status') or 'none'),
        }

    record['snapshot'] = snapshot
    record['snapshot_id'] = str(snapshot['hash'])[:16]

    level = str(record.get('review_level') or '')
    status = str(record.get('enrich_status') or 'none')

    if action == ACTION_FROZEN:
        # 只有"内容真的变了或从来没有快照"才置 pending。旧快照因缺新字段被判
        # 不可信而触发的一次性重冻结（内容哈希一致）**不进 pending**：pending 是
        # 六种待办状态之一，一置上就会让现场凭空多出一批"待补全"，而 AI 结果
        # （prompt/taxonomy/快照哈希都没变）本来仍然有效。
        content_unchanged = bool(legacy_hash) and legacy_hash == str(snapshot.get('hash') or '')
        if status != 'done' and not content_unchanged:
            record['enrich_status'] = 'pending'
        record['enrich_error'] = ''
        level, _ = refresh_level(project, page, record, page_index=index, kind=kind,
                                 columns=columns, enrichment=enrichment)
    elif action in (ACTION_STALE, ACTION_BOOK_META_STALE):
        record['enrich_status'] = 'stale'
        record['enrich_error'] = ''
        _mark_field_source(record, AI_FIELD_KEYS, 'stale')
        # 条件 1~3（人改了输入）与条件 5（书级字段变更）**都强制 L2**：
        # 条件 5 原先按 D6 不降级，已由 A3 决议改为降级（书级一错污染全书）。
        level, _ = refresh_level(project, page, record, page_index=index, kind=kind,
                                 columns=columns, enrichment=enrichment)
    elif action == ACTION_KP_STALE:
        record['enrich_status'] = 'kp_stale'
        _mark_field_source(record, ('knowledge_points',), 'stale')
    elif action == ACTION_ANALYSIS_STALE:
        record['enrich_status'] = 'analysis_stale'
        _mark_field_source(record, ('analysis',), 'stale')


    return {
        'action': action,
        'reasons': list(decision['reasons']),
        'changed': list(decision['changed']),
        'snapshot_id': str(snapshot['hash'])[:16],
        'review_level': str(level or ''),
        'enrich_status': str(record.get('enrich_status') or 'none'),
    }


def page_review_summary(page: dict) -> dict:
    """页级徽标用的分级计数。未算过的记录按 L2 计（§12.3 第 4 条）。"""
    summary = {'L0': 0, 'L1': 0, 'L2': 0}
    for record in (page or {}).get('questions') or []:
        if not isinstance(record, dict):
            continue
        if str(record.get('region_role', 'question') or 'question') != 'question':
            continue
        level = str(record.get('review_level') or review_levels.DEFAULT_LEVEL)
        if level not in summary:
            level = review_levels.DEFAULT_LEVEL
        summary[level] += 1
    return summary


def refresh_page_levels(project: dict, page_index: int | None = None, *,
                        kind: str = 'question', columns=None,
                        enrichment: dict | None = None,
                        question_ids=None) -> dict:
    """重算页面（或指定若干题）的级别，并回写 ``review_summary``。

    **分级与待办的唯一来源**：``review_level``、``review_reasons``、
    ``review_trace``、``page['review_summary']`` 全部由这一次计算产出，
    所以"页头风险 N 条"与"待办列表"不可能互相对不上（QA Round 8 第 6 条）。

    ``page_index=None`` 时只处理 ``question_ids`` 指定的记录（答案页保存后
    回刷被引用到的题目用）；``question_ids`` 也给了时只处理交集。
    """
    pages = (project or {}).get('pages') or []
    targets = {
        str(value) for value in (question_ids or []) if str(value or '').strip()
    } or None
    if page_index is not None and (int(page_index) < 0 or int(page_index) >= len(pages)):
        return {'updated': 0, 'review_summary': {'L0': 0, 'L1': 0, 'L2': 0}}
    indices = (
        [int(page_index)] if page_index is not None
        else sorted({
            index for index, page in enumerate(pages)
            if isinstance(page, dict) and any(
                isinstance(record, dict) and targets
                and str(record.get('id') or '') in targets
                for record in page.get('questions') or []
            )
        })
    )
    updated = 0
    summary: dict = {'L0': 0, 'L1': 0, 'L2': 0}
    for index in indices:
        page = pages[index] if isinstance(pages[index], dict) else {}
        if not page:
            continue
        for record in page.get('questions') or []:
            if not isinstance(record, dict):
                continue
            if str(record.get('region_role', 'question') or 'question') != 'question':
                continue
            if targets is not None and str(record.get('id') or '') not in targets:
                continue
            refresh_level(project, page, record, page_index=index, kind=kind,
                          columns=columns, enrichment=enrichment)
            updated += 1
        summary = page_review_summary(page)
        if str(kind) != 'answer':
            page['review_summary'] = summary
    return {'updated': updated, 'review_summary': summary}


def freeze_page(project: dict, page_index: int, *, kind: str = 'question',
                columns=None, enrichment: dict | None = None,
                page_hash: str | None = None) -> dict:
    """冻结/失效一页内全部 ``region_role=='question'`` 的记录，并重算本页分级。

    只处理 question 角色：``group`` / ``continuation`` / ``shared_material`` 不是
    独立交付题，给它们冻快照只会把人工量放大（它们由所属题目带过）。

    冻结完**整页重算一次级别**（而不是只算变化的那几条）：页级信号
    （``illustration_ready`` / ``page_consistency`` / 同页题量）一变，同页没改动
    的题也可能该升降级，两套口径分开算必然对不上（QA Round 8 第 6 条）。
    """
    pages_key, records_key = (
        ('answer_pages', 'answers') if str(kind) == 'answer' else ('pages', 'questions')
    )
    pages = (project or {}).get(pages_key) or []
    index = int(page_index)
    if index < 0 or index >= len(pages):
        return {'frozen': 0, 'stale': 0, 'kp_stale': 0, 'unchanged': 0,
                'decisions': {}, 'review_summary': {'L0': 0, 'L1': 0, 'L2': 0}}
    page = pages[index]
    if not isinstance(page, dict):
        return {'frozen': 0, 'stale': 0, 'kp_stale': 0, 'unchanged': 0,
                'decisions': {}, 'review_summary': {'L0': 0, 'L1': 0, 'L2': 0}}

    counts = {ACTION_FROZEN: 0, ACTION_STALE: 0, ACTION_UNCHANGED: 0,
              ACTION_KP_STALE: 0, ACTION_ANALYSIS_STALE: 0, ACTION_BOOK_META_STALE: 0}
    decisions: dict[str, str] = {}
    for record in page.get(records_key) or []:
        if not isinstance(record, dict):
            continue
        if str(record.get('region_role', 'question') or 'question') != 'question':
            continue
        result = freeze_or_invalidate(
            project, page, record,
            page_index=index, kind=kind, page_hash=page_hash,
            columns=columns, enrichment=enrichment,
        )
        action = str(result.get('action') or ACTION_SKIPPED)
        counts[action] = counts.get(action, 0) + 1
        record_id = str(record.get('id') or '')
        if record_id:
            decisions[record_id] = action

    levels = refresh_page_levels(
        project, index, kind=kind, columns=columns, enrichment=enrichment,
    )
    summary = levels['review_summary']

    stale_like = (counts[ACTION_STALE] + counts[ACTION_BOOK_META_STALE])
    return {
        'frozen': counts[ACTION_FROZEN],
        'stale': stale_like,
        'kp_stale': counts[ACTION_KP_STALE] + counts[ACTION_ANALYSIS_STALE],
        'unchanged': counts[ACTION_UNCHANGED],
        'decisions': decisions,
        'review_summary': summary,
        'levels_refreshed': levels['updated'],
    }



def rebuild_missing(project: dict, *, kind: str = 'question',
                    columns=None, enrichment: dict | None = None) -> dict:
    """给"没有有效快照"的记录补建快照并重算级别（补全开关 0→1 的活锁修复）。

    为什么需要：``PAPERFLOW_ENRICH=0`` 期间保存的页面不会冻结（``_save_page_enrich``
    直接返回 ``{"enabled": False}``）。开关切回 1 之后，这些记录既没有快照，
    ``review_level`` 也永远是默认的 L2 —— **永久 L2**，人工白看。

    幂等：已有有效快照的记录直接跳过，不重写 ``created_at``/``hash``。
    只处理 ``region_role == 'question'``（与 :func:`freeze_page` 同口径）。
    """
    pages_key, records_key = (
        ('answer_pages', 'answers') if str(kind) == 'answer' else ('pages', 'questions')
    )
    pages = (project or {}).get(pages_key) or []
    counts = {'pages': 0, 'frozen': 0, 'skipped': 0}
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        touched = False
        for record in page.get(records_key) or []:
            if not isinstance(record, dict):
                continue
            if str(record.get('region_role', 'question') or 'question') != 'question':
                continue
            if is_valid_snapshot(record.get('snapshot')):
                counts['skipped'] += 1
                continue
            freeze_or_invalidate(
                project, page, record,
                page_index=index, kind=kind, columns=columns, enrichment=enrichment,
            )
            counts['frozen'] += 1
            touched = True
        if touched:
            refresh_page_levels(project, index, kind=kind, columns=columns,
                                enrichment=enrichment)
            counts['pages'] += 1
    return counts


def refresh_answer_segment(project: dict, answer_ids, *, columns=None) -> dict:
    """答案后到（§3.4）：把刚确认的答案回写进引用它的题目快照，并重算级别。

    员工通常先确认题页、后确认答案页。若不在答案页确认时回写，快照会永久缺
    ``answer``，AI 解析只能瞎猜。回写后 ``answer`` 变了 → 该题按条件 2 失效
    （``stale`` + L2），等 T04 拿着含答案的快照重跑。这是**正确**的代价：
    用没有答案的快照算出来的解析本来就是废的。

    回写之后**必然**重算这些题的级别（哪怕快照判定是 unchanged）：答案页一确认，
    答案条数/可匹配性就变了，不重算会出现"答案都到齐了、题目侧还挂着 L2"的
    假风险（QA Round 8 第 4 条）。
    """
    targets = {
        str(value).strip()
        for value in (answer_ids or [])
        if str(value or '').strip()
    }
    result = {'ok': True, 'updated': 0, 'question_ids': [], 'actions': {},
              'levels_refreshed': 0}
    if not targets:
        return result

    enrichment = None
    if isinstance(project, dict) and isinstance(project.get('enrichment'), dict):
        enrichment = project['enrichment']
    touched_ids: list[str] = []
    for page_index, page in enumerate((project or {}).get('pages') or []):
        if not isinstance(page, dict):
            continue
        page_hash = None
        for record in page.get('questions') or []:
            if not isinstance(record, dict):
                continue
            if str(record.get('region_role', 'question') or 'question') != 'question':
                continue
            current_ids = set(answer_ids_of(project, str(record.get('id') or '')))
            if not targets.intersection(current_ids):
                continue
            record_id = str(record.get('id') or '')
            if record_id:
                touched_ids.append(record_id)
            if not is_valid_snapshot(record.get('snapshot')):
                # 没有可用快照 → 不假装"回填过"，但级别照样要重算（见 docstring）
                continue
            if page_hash is None:
                page_hash = page_sha1_of(project, page, 'question')
            decision = freeze_or_invalidate(
                project, page, record,
                page_index=page_index, kind='question', page_hash=page_hash,
                columns=columns, enrichment=enrichment,
            )
            action = str(decision.get('action') or '')
            if action == ACTION_UNCHANGED:
                continue
            result['updated'] += 1
            if record_id:
                result['question_ids'].append(record_id)
                result['actions'][record_id] = action

    if touched_ids:
        levels = refresh_page_levels(
            project, kind='question', columns=columns, enrichment=enrichment,
            question_ids=touched_ids,
        )
        result['levels_refreshed'] = levels['updated']
        result['review_summary'] = levels['review_summary']
    return result

