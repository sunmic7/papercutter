#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零人工切题质量审计。

设计前提：**不做人工标注**。真值来自图像本身——印刷的题号。
同一页跑两次 OCR：

  * 生产口径 ``ocr_page_structure``（稀疏，只扫标题带+题号带）→ 用于切块，代表用户真实看到的
  * 真值口径 ``ocr_image``（整页，更全）→ 只用于抽取"这页印了哪些题号"

两者独立，所以"生产口径漏了多少题号"本身就是可测的。所有指标都能由
图像 + OCR 自动算出，不需要人看一张图。

指标分三层（对应路线图里的 L1/L2/L3）：

  L1 栏位：由 ``layout_arbiter`` 给出，本工具只记录栏数，供横向比较
  L2 大题：整页块（有据）、吞并块、重复标题块、大题塌缩
  L3 小题：题号未识别率、碎块、超长框、切点错位

**只采信硬信号。** 一开始试过"块内印刷题号集合 vs 切出题号集合"的差集，
但整页 OCR 本身会漏读个别题号，差集会把正常切分判成"漏切+多切"（实测
第 7 张就误报了一例）。因此精确定位类指标只作为 ``diagnostic`` 原始数据
保留，不进汇总；汇总只统计几何上无可辩驳的错误。

用法::

    # 单本
    python review_tools/cut_quality_audit.py --book "<书目录>" --profile exam --out reports/audit.jsonl

    # 多本
    python review_tools/cut_quality_audit.py --books-from books.txt --out reports/audit.jsonl

    # 汇总（可重复调用，追加式）
    python review_tools/cut_quality_audit.py --summarize reports/audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# 一组"几乎一定是错"的阈值。都取得很保守，宁可漏报也不误报：
# 误报会让人不再信这份报表。
TINY_CUT_HEIGHT = 40          # px，低于此高度的题框基本是横向碎片
TINY_BLOCK_HEIGHT = 40        # px，低于此高度的大题块是碎屑，几乎必然是切分抖动
GIANT_CUT_HEIGHT_RATIO = 0.50  # 超过页高一半的"小题"多半是吞并后没切开
MAX_MINOR_NUM = 60            # 小学教材小题号不会超过这个数


def _load_core():
    """延迟导入，避免 --summarize 也去加载 paddle。"""
    import cut_questions_v4 as core
    return core


# ─────────────────────────── 印刷题号抽取 ───────────────────────────


def _item_top(item):
    return int(min(point[1] for point in item['bbox']))


def _item_text(item):
    return str(item.get('text') or '').strip()


def printed_major_titles(items):
    """页内印刷的大题标题：返回 [(序号, 文本, y, x)]，同号只留最靠上的一个。"""
    core = _load_core()
    found = {}
    for item in items:
        text = _item_text(item)
        if not text:
            continue
        if not core.RE_MAJOR.match(text):
            continue
        # 正文里的"三、"会被 RE_MAJOR 误命中。大题标题通常很短，
        # 用长度做一次粗过滤；真正的大题标题很少超过 30 字。
        if len(re.sub(r'\s+', '', text)) > 30:
            continue
        try:
            idx = core.extract_major_idx(text)
        except Exception:  # noqa: BLE001
            idx = None
        if idx is None:
            continue
        top = _item_top(item)
        left = min(point[0] for point in item['bbox'])
        if idx not in found or top < found[idx][2]:
            found[idx] = (idx, re.sub(r'\s+', '', text), top, left)
    return sorted(found.values(), key=lambda value: value[2])


def printed_minor_numbers(items):
    """块内印刷的小题号序列（按 y 排序，保留重复以便发现"重新从 1 开始"）。"""
    core = _load_core()
    sequence = []
    for item in sorted(items, key=lambda value: (_item_top(value), value['bbox'][0][0])):
        text = _item_text(item)
        if not text:
            continue
        match = core.RE_MINOR.match(text)
        if not match:
            continue
        num = int(match.group(1))
        if 0 < num <= MAX_MINOR_NUM:
            sequence.append(num)
    return sequence


def _cut_numbers(minors):
    numbers = []
    for minor in minors:
        value = minor.get('num')
        try:
            numbers.append(int(value))
        except (TypeError, ValueError):
            numbers.append(None)
    return numbers


# ─────────────────────────── 单页审计 ───────────────────────────


def audit_page(image_path, profile='general'):
    core = _load_core()
    image = core.cv2_imread(image_path)
    if image is None:
        return {'path': image_path, 'error': 'read_failed'}
    height, width = image.shape[:2]

    timing = {}
    # 生产口径：用户点"自动框题"时真正跑的那条路
    started = time.perf_counter()
    production_items = core.ocr_page_structure(image_path, image, profile=profile)
    timing['structure_ocr'] = round(time.perf_counter() - started, 2)
    started = time.perf_counter()
    blocks = core.split_page_into_majors(image, production_items, profile=profile)
    split_seconds = time.perf_counter() - started
    timing['split'] = round(split_seconds, 2)

    # 真值口径：整页 OCR，只用来读"这页印了哪些题号"
    started = time.perf_counter()
    truth_items = core.ocr_image(image_path)
    timing['truth_ocr'] = round(time.perf_counter() - started, 2)
    timing['minors'] = 0.0
    page_majors = printed_major_titles(truth_items)
    printed_major_set = {entry[0] for entry in page_majors}

    counters = Counter()
    block_records = []
    error_samples = []
    captured = set()
    total_printed_minors = 0
    previous_title = None
    previous_title_idx = None

    for position, block in enumerate(blocks):
        title = str(block.get('title') or '')
        block_height = int(block['crop_img'].shape[0])
        block_width = int(block['crop_img'].shape[1])
        x0 = int(block.get('_x_start', 0))
        y0 = int(block.get('_y_start', 0))
        is_blank_title = title in {'整页', ''}

        try:
            title_idx = core.extract_major_idx(title) if core.RE_MAJOR.match(title) else None
        except Exception:  # noqa: BLE001
            title_idx = None
        if title_idx is not None:
            captured.add(title_idx)

        truth_local = core.ocr_items_for_block(truth_items, block)
        production_local = core.ocr_items_for_block(production_items, block)

        # 本块覆盖了哪些印刷大题：必须同时命中 y 与 x。
        # 分栏页里左右栏的大题可能有相同的 y，只比 y 会把邻栏算进来。
        block_printed_majors = [
            idx for idx, _text, major_y, major_x in page_majors
            if y0 - 8 <= major_y <= y0 + block_height - 8
            and x0 - 4 <= major_x <= x0 + block_width + 4
        ]

        printed_seq = printed_minor_numbers(truth_local)
        printed_set = set(printed_seq)
        total_printed_minors += len(printed_set)

        minor_started = time.perf_counter()
        minors = core.split_major_into_minors(
            block['crop_img'], ocr_items=production_local, profile=profile,
        )
        timing['minors'] = round(
            timing['minors'] + time.perf_counter() - minor_started, 2,
        )
        cut_numbers = _cut_numbers(minors)
        cut_set = {value for value in cut_numbers if value is not None}
        ordered_cut = [value for value in cut_numbers if value is not None]
        # 题号"回头"本身不是错误：两栏页每栏都从 1 开始，切出来的序列就是
        # 1,2,1,2，而印刷的也是 1,2,1,2。只有切出的题号集合与印刷的不一致，
        # 回头才说明切错了（少切或多切）。
        breaks = 0
        if Counter(ordered_cut) != Counter(printed_seq):
            breaks = sum(
                1 for previous, current in zip(ordered_cut, ordered_cut[1:])
                if current <= previous
            )

        block_undetected = sum(
            1 for minor in minors if not minor.get('number_detected', True)
        )
        block_tiny = sum(
            1 for minor in minors
            if int(minor['crop_img'].shape[0]) < TINY_CUT_HEIGHT
        )
        # 只有"块内至少两道题"时，超长框才说明该切没切开。
        # 整块只有一道题时，高度大是因为题目本身长或带图，不是错误。
        # 另外还要求"切出的题号比印刷的少"：印了 2 题、也切出 2 题，其中一题
        # 本身就占了半页（带图或长题干），这不是吞并。旧判据缺这一条，
        # 20 个 giant 里 20 个都是这种正常长题。
        block_giant = 0
        if len(minors) >= 2 and len(cut_set) < len(set(printed_seq)):
            block_giant = sum(
                1 for minor in minors
                if int(minor['crop_img'].shape[0]) > height * GIANT_CUT_HEIGHT_RATIO
            )
        block_bad = sum(
            1 for minor in minors if core.crop_quality_issues(minor['crop_img'])
        )

        # ── 硬信号：几何或文本上无可辩驳的错误 ──
        # 整页块要分两种：块内首题号从 1 开始（本该是个新大题，标题丢了）
        # 才算失败；从 3、4 开始说明它是上一块的跨页延续，属于正常形态。
        # 练习册大量页面本来就没有"一、二、三"大题号，只剩 1./2./3.，
        # 这种页面的整页块不是丢标题，而是它的正常形态——旧判据没有这道
        # 前置条件，把整批 workbook 页都报成错误（10 页里 8 页是误报）。
        first_cut = next((value for value in cut_numbers if value is not None), None)
        starts_fresh = first_cut == 1
        page_has_majors = bool(printed_major_set)
        if is_blank_title and (
            block_printed_majors
            or (page_has_majors and printed_set and starts_fresh)
        ):
            counters['untitled_with_evidence'] += 1
            error_samples.append({
                'type': '标题退化为整页（块内是大题开头）',
                'block': position,
                'title': '',
                'majors': sorted(set(block_printed_majors)),
                'minors': sorted(printed_set),
            })
        elif is_blank_title and page_has_majors and printed_set:
            # 有印大题号、但这块没有标题：既不是丢标题（首题号不回头），
            # 就是上一块大题在本栏/本页的延续，属正常形态，不算错误。
            counters['continuation_blocks'] += 1
        if len(set(block_printed_majors)) >= 2:
            counters['merged_blocks'] += 1
            error_samples.append({
                'type': '一个块含多个大题',
                'block': position,
                'title': title[:20],
                'majors': sorted(set(block_printed_majors)),
            })
        same_major = (
            title_idx is not None
            and previous_title_idx is not None
            and title_idx == previous_title_idx
        )
        if same_major or (
            previous_title is not None and title == previous_title and not is_blank_title
        ):
            counters['duplicate_title_blocks'] += 1
            error_samples.append({
                'type': '同一大题标题被切成两块',
                'block': position,
                'title': title[:20],
                'height': block_height,
            })
        if len(printed_set) >= 3 and len(minors) <= 1:
            counters['collapsed_blocks'] += 1
            error_samples.append({
                'type': '大题没切开（印刷多题只切出1题）',
                'block': position,
                'title': title[:20],
                'printed': sorted(printed_set),
                'height': block_height,
            })
        if block_height < TINY_BLOCK_HEIGHT:
            counters['empty_blocks'] += 1
            error_samples.append({
                'type': '空壳块（高度异常小）',
                'block': position,
                'title': title[:20],
                'height': block_height,
            })

        block_records.append({
            'position': position,
            'title': title[:24],
            'title_idx': title_idx,
            'col': block.get('_col'),
            'bounds': [int(value) for value in block.get('_column_bounds', [])],
            'y_start': y0,
            'height': block_height,
            'printed_majors': sorted(set(block_printed_majors)),
            'cut_numbers': cut_numbers,
            'diagnostic': {
                'printed_minors': printed_seq,
                'missed': sorted(printed_set - cut_set),
                'extra': sorted(cut_set - printed_set),
                'undetected': block_undetected,
                'tiny': block_tiny,
                'giant': block_giant,
                'bad_quality': block_bad,
                'seq_breaks': breaks,
            },
        })
        previous_title = title
        previous_title_idx = title_idx

    # 小题级指标只在"块内确实印了题号"的块上统计。
    # 封面、版权页、答案页的整页块本来就没有题号，整块当一题是合理的，
    # 把它们算进"未识别""超长框"只会制造噪音。
    for record in block_records:
        diagnostic = record['diagnostic']
        if not diagnostic['printed_minors']:
            continue
        counters['undetected'] += diagnostic['undetected']
        counters['tiny_cuts'] += diagnostic['tiny']
        counters['giant_cuts'] += diagnostic['giant']
        counters['bad_quality'] += diagnostic['bad_quality']
        counters['seq_breaks'] += diagnostic['seq_breaks']
        # 小题召回的直接口径：印刷题号里有几个没被切出来。
        # 比"超长框""题号回头"这类几何启发式可信，因为它直接和真值比对。
        counters['minor_missed'] += len(diagnostic['missed'])
        counters['minor_extra'] += len(diagnostic['extra'])

    # 封面、版权页、答案页没有印刷题号，所有比例都不适用
    applicable = bool(printed_major_set) or total_printed_minors > 0

    # 纯答案页也会印着「一、二、三」，但它们下面跟着的是答案而不是题目，
    # 整页并成一个块本来就是对的。先起跑的答案页（image0000911B）就因此
    # 被误判成“一个块含多个大题”。答案页的可靠签名是「数字+略」成串出现。
    answer_marks = sum(1 for item in truth_items if '略' in _item_text(item))
    page_kind = 'answer' if answer_marks >= 4 else 'question'
    if page_kind == 'answer':
        applicable = False
        # 答案页的“错误”全部是口径噪音，连带样本一起清掉。
        counters = Counter()
        error_samples = []

    major_recall = None
    if printed_major_set:
        major_recall = round(len(printed_major_set & captured) / len(printed_major_set), 4)

    return {
        'path': image_path,
        'file': os.path.basename(image_path),
        'profile': profile,
        'image': [width, height],
        'applicable': applicable,
        'page_kind': page_kind,
        'ocr_production': len(production_items),
        'ocr_truth': len(truth_items),
        'columns': len({
            tuple(record['bounds'][:2]) for record in block_records if record['bounds']
        }),
        'block_count': len(block_records),
        'major_printed': len(printed_major_set),
        'major_captured': len(printed_major_set & captured),
        'major_recall': major_recall,
        'minor_printed': total_printed_minors,
        'minor_cuts': sum(len(record['cut_numbers']) for record in block_records),
        # 小题召回：印刷了多少个题号、有几个没被切出来。这是 L3 的主指标，
        # signal 类判据只用来解释"为什么漏"。
        'minor_missed': counters.get('minor_missed', 0),
        'minor_extra': counters.get('minor_extra', 0),
        **{name: counters.get(name, 0) for name in (
            'untitled_with_evidence', 'merged_blocks', 'duplicate_title_blocks',
            'collapsed_blocks', 'empty_blocks', 'undetected', 'tiny_cuts',
            'giant_cuts', 'bad_quality', 'seq_breaks', 'continuation_blocks',
        )},
        'split_seconds': round(split_seconds, 2),
        'timing': timing,
        'error_samples': error_samples,
        'blocks': block_records,
    }


# ─────────────────────────── 批量与汇总 ───────────────────────────


def iter_page_paths(book_dir, sample=0):
    names = sorted(
        name for name in os.listdir(book_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )
    if sample and sample < len(names):
        # 等间隔抽样：跳过首张封面，首尾都取到，避免只测前半本。
        pool = names[1:] if len(names) > 1 else names
        if sample == 1:
            picks = [pool[len(pool) // 2]]
        else:
            last = len(pool) - 1
            picks = [pool[round(index * last / (sample - 1))] for index in range(sample)]
        names = list(dict.fromkeys(picks))
    return [os.path.join(book_dir, name) for name in names]


def _completed_paths(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('path') and not record.get('error'):
                done.add(record['path'])
    return done


def prewarm_models(log=print):
    """把三个重模型提前加载，避免首个样本吞掉加载时间（也会进 avg_seconds）。"""
    core = _load_core()
    started = time.perf_counter()
    core.get_ocr_engine()
    core.get_fast_ocr_engine()
    try:
        import layout_arbiter
        if layout_arbiter.available():
            layout_arbiter._get_pipe()  # noqa: SLF001 预热，避免首栏页吞掉加载时间
            log('[审计] 版面裁判已加载')
    except Exception as error:  # noqa: BLE001
        log(f'[审计] 版面裁判不可用（{type(error).__name__}: {error}），按无裁判口径跑')
    log(f'[审计] 模型就绪 {time.perf_counter() - started:.1f}s')


def _audit_pending(pending, out_path, log=print):
    """逐页审计并追加写盘；每页立刻落盘，中断后可续跑。"""
    written = 0
    with open(out_path, 'a', encoding='utf-8') as handle:
        for index, (path, profile, meta) in enumerate(pending, 1):
            started = time.perf_counter()
            try:
                record = audit_page(path, profile=profile)
            except Exception as error:  # noqa: BLE001
                record = {
                    'path': path,
                    'file': os.path.basename(path),
                    'profile': profile,
                    'error': f'{type(error).__name__}: {error}',
                    'traceback': traceback.format_exc()[-1200:],
                }
            record.update(meta)
            record['seconds'] = round(time.perf_counter() - started, 2)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            written += 1
            if record.get('error'):
                log(f"  [{index}/{len(pending)}] {record['file']} 失败 {record['error']}")
            else:
                log(
                    f"  [{index}/{len(pending)}] {record.get('file', '')[-22:]} "
                    f"块{record.get('block_count', 0)} 整页{record.get('untitled_with_evidence', 0)} "
                    f"吞并{record.get('merged_blocks', 0)} 重复{record.get('duplicate_title_blocks', 0)} "
                    f"塌缩{record.get('collapsed_blocks', 0)} 小题{record.get('minor_cuts', 0)}/"
                    f"{record.get('minor_printed', 0)} 未识别{record.get('undetected', 0)} "
                    f"{record['seconds']}s"
                )
    return written


def load_page_plan(plan_path):
    """读取固定页面清单（JSONL：path / profile / book）。"""
    rows = []
    with open(plan_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('path'):
                rows.append(row)
    return rows


def run_page_plan(plan_path, out_path, log=print):
    """按清单逐页审计。

    `--sample N` 依赖"文件名排序 + 步长"，书目或代码一变就抽到另一批页，
    两次结论无法对比。固定清单把"哪本书、哪几页、按什么版式"写死，
    复跑与首跑是同一批页。
    """
    rows = load_page_plan(plan_path)
    done = _completed_paths(out_path)
    pending = [
        (row['path'], row.get('profile') or 'general', {
            'book': row.get('book') or os.path.basename(os.path.dirname(row['path'])),
            'book_dir': row.get('book_dir') or os.path.dirname(row['path']),
        })
        for row in rows if row['path'] not in done
    ]
    log(f'[审计] 清单 {len(rows)} 页，待处理 {len(pending)} 页')
    prewarm_models(log)
    written = _audit_pending(pending, out_path, log)
    log(f'[审计] 写入 {written} 页 -> {out_path}')
    return out_path


def run_book(book_dir, profile, out_path, limit=0, sample=0, log=print):
    core = _load_core()
    paths = iter_page_paths(book_dir, sample=sample)
    if limit:
        paths = paths[:limit]
    done = _completed_paths(out_path)
    pending = [path for path in paths if path not in done]
    log(f'[审计] {os.path.basename(book_dir)} 共 {len(paths)} 页，待处理 {len(pending)} 页')

    prewarm_models(log)

    with open(out_path, 'a', encoding='utf-8') as handle:
        for index, path in enumerate(pending, 1):
            started = time.perf_counter()
            try:
                record = audit_page(path, profile=profile)
            except Exception as error:  # noqa: BLE001
                record = {
                    'path': path,
                    'file': os.path.basename(path),
                    'profile': profile,
                    'error': f'{type(error).__name__}: {error}',
                    'traceback': traceback.format_exc()[-1200:],
                }
            record['book'] = os.path.basename(book_dir.rstrip('\\/'))
            record['book_dir'] = book_dir
            record['seconds'] = round(time.perf_counter() - started, 2)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            if record.get('error'):
                log(f"  [{index}/{len(pending)}] {record['file']} 失败 {record['error']}")
            else:
                log(
                    f"  [{index}/{len(pending)}] {record['file']} "
                    f"块{record['block_count']} 整页{record['untitled_with_evidence']} "
                    f"吞并{record['merged_blocks']} 重复{record['duplicate_title_blocks']} "
                    f"塌缩{record['collapsed_blocks']} 小题{record['minor_cuts']}/"
                    f"{record['minor_printed']} 未识别{record['undetected']} "
                    f"{record['seconds']}s"
                )
    return out_path


def load_records(paths):
    records = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return [record for record in records if not record.get('error')]


def summarize(records):
    """汇总。所有比率只在 applicable 页上算，避免封面/答案页拉低分母。"""
    if not records:
        return {'total': {'pages': 0}, 'per_book': {}, 'error_types': {}, 'page_errors': []}

    books = {}
    for record in records:
        books.setdefault(record.get('book') or '?', []).append(record)

    signal_names = (
        'untitled_with_evidence', 'merged_blocks', 'duplicate_title_blocks',
        'collapsed_blocks', 'empty_blocks', 'undetected', 'tiny_cuts',
        'giant_cuts', 'bad_quality', 'seq_breaks',
    )
    # `continuation_blocks` 故意不在这里：块没标题但首题号从中间续上，是
    # 大题跨栏/跨页的正常形态。把它算成"有错页"会让整批两栏页全变红。

    def aggregate(items):
        pages = len(items)
        usable = [item for item in items if item.get('applicable')]
        printed = sum(item['major_printed'] for item in usable)
        captured = sum(item['major_captured'] for item in usable)
        stats = {
            'pages': pages,
            'applicable_pages': len(usable),
            'blocks': sum(item['block_count'] for item in items),
            'major_printed': printed,
            'major_captured': captured,
            'major_recall': round(captured / printed, 4) if printed else None,
            'minor_printed': sum(item['minor_printed'] for item in usable),
            'minor_cuts': sum(item['minor_cuts'] for item in usable),
            'minor_missed': sum(item.get('minor_missed', 0) for item in usable),
            'minor_extra': sum(item.get('minor_extra', 0) for item in usable),
            'minor_recall': (
                round(
                    1 - sum(item.get('minor_missed', 0) for item in usable)
                    / sum(item['minor_printed'] for item in usable), 4,
                )
                if sum(item['minor_printed'] for item in usable) else None
            ),
            'pages_with_error': sum(
                1 for item in usable if any(item.get(name, 0) for name in signal_names)
            ),
            'avg_seconds': round(sum(item.get('seconds', 0) for item in items) / pages, 2),
        }
        for name in signal_names:
            stats[name] = sum(item.get(name, 0) for item in items)
        stats['error_rate'] = (
            round(stats['pages_with_error'] / len(usable), 4) if usable else None
        )
        return stats

    per_book = {name: aggregate(items) for name, items in sorted(books.items())}
    error_types = Counter()
    page_errors = []
    for record in records:
        for sample in record.get('error_samples', []):
            error_types[sample['type']] += 1
            page_errors.append({
                'book': record.get('book'),
                'page': record.get('file'),
                'type': sample['type'],
                'block': sample.get('block'),
                'detail': {
                    key: value for key, value in sample.items()
                    if key not in {'type', 'block'}
                },
            })
    return {
        'total': aggregate(records),
        'per_book': per_book,
        'error_types': dict(error_types.most_common()),
        'page_errors': page_errors,
    }


def main():
    parser = argparse.ArgumentParser(description='零人工切题质量审计')
    parser.add_argument('--book', action='append', default=[], help='书目录，可重复')
    parser.add_argument('--books-from', help='每行一个书目录的文件')
    parser.add_argument('--profile', default='auto', help='教材场景；auto 时按书名推断')
    parser.add_argument('--out', help='JSONL 输出路径（追加，可续跑）')
    parser.add_argument('--limit', type=int, default=0, help='每本只跑前 N 页')
    parser.add_argument('--sample', type=int, default=0, help='每本等间隔抽 N 页（优先于 --limit 之外单独使用）')
    parser.add_argument('--pages-from', help='固定页面清单 JSONL（path/profile/book），复跑必须用它而不是 --sample')
    parser.add_argument('--summarize', nargs='*', help='只汇总指定的 JSONL')
    parser.add_argument('--json-out', help='汇总结果另存为 JSON')
    args = parser.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if args.summarize:
        records = load_records(args.summarize)
        report = summarize(records)
        print(json.dumps(report['total'], ensure_ascii=False, indent=2))
        for name, stats in report['per_book'].items():
            print(f"\n{name}")
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        print('\n错误类型分布：')
        for name, count in report['error_types'].items():
            print(f'  {count:5d}  {name}')
        if args.json_out:
            with open(args.json_out, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print(f'\n写入 {args.json_out}')
        return

    books = list(args.book)
    if args.books_from:
        with open(args.books_from, 'r', encoding='utf-8') as handle:
            books.extend(line.strip() for line in handle if line.strip())
    if not books and not args.pages_from:
        parser.error('请提供 --book / --books-from / --pages-from，或使用 --summarize')
    if not args.out:
        parser.error('批量审计需要 --out')

    if args.pages_from:
        # 固定清单自带版式，不再按书名重判，保证复跑同口径。
        run_page_plan(args.pages_from, args.out)
        return

    core = _load_core()
    for book_dir in books:
        profile = args.profile
        if profile in {'', 'auto'}:
            # 审计必须复现生产口径：界面抽样判出的版式落盘后，命令行也要用它，
            # 否则同一本书“界面切得对、审计按 general 跑出另一套结论”。
            profile = core.resolve_book_profile(book_dir)
            print(f'[审计] profile -> {profile}')
        run_book(book_dir, profile, args.out, limit=args.limit, sample=args.sample)


if __name__ == '__main__':
    main()
