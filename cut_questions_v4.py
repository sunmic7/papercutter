#!/usr/bin/env python3
"""cut_questions_v4.py - 切题脚本 v4

核心理念：按小题切分，每张图只含一道小题
目录结构：教材/单元(如有)/大题类型/第X题/

输出示例：
  课前课后同步练二年级下册数学B/
    第一单元/
      一、课前预习/
        第1题/
          题目.jpg
          答案.jpg
        第2题/
          ...
      二、课后练习/
        第1题/
          ...
    第二单元/
      ...

切题策略（两阶段）：
  阶段1：按大题边界（一、二、三...）将页面切成大题块
  阶段2：在每个大题块内，按小题编号（1. 2. 3.）切成单题图
  如果整页没有大题编号也没有小题编号，整页输出
"""

import os
import sys
import re
import json
import time
import argparse
import inspect
import hashlib
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime
from collections import defaultdict, OrderedDict
import multiprocessing

import uuid

import cv2
import numpy as np

import review_queue
import question_detector
import answer_detector
import layout_strategy
import answer_region_detector
import vlm_hybrid

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
ANSWER_DIR_NAMES = {'答案', '参考答案', '答案与解析', '解析'}
QUESTION_DIR_NAMES = {'题目', '试卷', '试题', '练习'}
ANSWER_TITLE_KEYWORDS = ('参考答案', '试题答案', '答案与解析', '参考解答', '答  案', '答案')
COMPLETION_MARKER = '.cut_questions_complete.json'
BOOK_COMPLETION_MARKER = '.cut_questions_book_complete.json'
ALGORITHM_VERSION = 'smart-v47-vlm-answer-context-20260808'
APP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(APP_DIR, 'runtime')
LOCAL_MODEL_ROOT = os.path.join(RUNTIME_DIR, 'models', 'paddleocr')
# 结构扫描只识别标题与题号带，使用随项目携带的轻量中文模型；
# 完整题面和答案仍使用 PaddleOCR 已安装的中等模型，保证现有精度不被牺牲。
FAST_DET_MODEL_DIR = os.path.join(LOCAL_MODEL_ROOT, 'PP-OCRv5_mobile_det')
FAST_REC_MODEL_DIR = os.path.join(LOCAL_MODEL_ROOT, 'PP-OCRv5_mobile_rec')
# 可训练题块检测器默认只做“保守辅助”：仅在模型框与既有规则框一一匹配时
# 微调边界；检测器不存在、低置信度或题数有冲突时，完全保留既有 OCR 规则。
QUESTION_DETECTOR_MODE = os.environ.get('PAPERCUTTER_QUESTION_DETECTOR_MODE', 'assist').strip().lower()
QUESTION_DETECTOR_ASSIST_IOU = 0.55
# The answer detector is optional and is trained only from reviewer-approved
# Chinese answer-page boxes.  OCR still supplies the answer number and is the
# sole authority for question--answer association.
ANSWER_DETECTOR_MODE = os.environ.get('PAPERCUTTER_ANSWER_DETECTOR_MODE', 'assist').strip().lower()
ANSWER_DETECTOR_ASSIST_IOU = 0.55
# HiLEx Answer_Block is a question-page internal area, not an answer-key row.
# It can only conservatively extend a current question crop before the next one.
ANSWER_REGION_DETECTOR_MODE = os.environ.get('PAPERCUTTER_ANSWER_REGION_DETECTOR_MODE', 'assist').strip().lower()
ANSWER_REGION_DETECTOR_ASSIST_CONFIDENCE = 0.78
# 仅在三条以上长线给出高度一致的角度时纠斜，证据不足则保持原图。
ENABLE_DESKEW = True
OCR_MAX_SIDE = 1800
# 快速定位只用于可验证的题号/版块识别；无法通过校验的页面仍回退到 1800。
OCR_FAST_MAX_SIDE = 1400
# 题号通常只位于每栏左侧；下一版优先识别标题和题号带，证据不足再读整页。
OCR_ANCHOR_MAX_SIDE = 1200
# 答案页只需保留题号及答案行的可读性。单页对照验证 1600 与 1800 的答案锚点一致，
# 但识别耗时降低约 16%；文本过少或置信度异常时仍自动回退 1800。
ANSWER_OCR_MAX_SIDE = 1600
DESKEW_ANALYSIS_MAX_SIDE = 1100
MIN_OCR_CONFIDENCE = 0.35
# 同一大题标题被 OCR 拆成两个文字框时，两框的 y 差上限与 x 差上限（像素）。
# 前者用来识别“同一行拆成两块”，后者用来排除对开页左右两页印着同一个大题号。
SPLIT_MAJOR_MERGE_GAP = 90
SPLIT_MAJOR_MERGE_DX = 1200
# 环境变量只供性能基准隔离缓存使用；正常运行保持固定版本并持续复用缓存。
OCR_CACHE_VERSION = os.environ.get('PAPERCUTTER_OCR_CACHE_VERSION', 'ocr-v3-1800-20260729')
# 输出模式：默认 PaperFlow 预标注 JSON；通过 --export-crops 可恢复旧裁图输出。

# 批量任务的故障定位上下文。process_year 会在某本教材意外异常后读取它，
# 只重试/跳过出错单元，而不是让整批任务直接结束。
CURRENT_RUN_CONTEXT = {
    'book_path': None,
    'book_name': None,
    'unit_path': None,
    'unit_num': None,
    'unit_name': None,
    'unit_position': None,
}
PROGRESS_PREFIX = '@@PROGRESS@@'
RUN_STATS = defaultdict(float)
_QUESTION_DETECTOR_NOTICE_EMITTED = False
_ANSWER_DETECTOR_NOTICE_EMITTED = False
_ANSWER_REGION_DETECTOR_NOTICE_EMITTED = False
_VLM_NOTICE_EMITTED = False


def analyze_page_with_vlm(img, page_info, page_name):
    """Use the cloud model as a guarded page-level assistant, never as a cutter."""
    global _VLM_NOTICE_EMITTED
    if not vlm_hybrid.enabled():
        return None
    started = time.perf_counter()
    analysis = vlm_hybrid.analyze_page(img)
    _add_timing('vlm_page_analysis', time.perf_counter() - started)
    RUN_STATS['vlm_page_count'] += 1
    if analysis is None:
        RUN_STATS['vlm_failed_page_count'] += 1
        if not _VLM_NOTICE_EMITTED:
            print('[VLM] 未获得有效结果，当前页继续使用本地 OCR/规则与答案配对。')
            _VLM_NOTICE_EMITTED = True
        return None
    RUN_STATS['vlm_group_count'] += len(analysis.get('groups', []))
    RUN_STATS['vlm_ignore_region_count'] += len(analysis.get('ignore_regions', []))
    if vlm_hybrid.should_skip_page(analysis, page_info.get('kind', 'unknown')):
        # A visible local paper title is stronger evidence than a single VLM
        # answer-sheet label.  Keep this page and let later crop-level guards
        # handle any answer-card area instead of losing an entire test paper.
        if page_info.get('title'):
            RUN_STATS['vlm_skip_overruled_by_title_count'] += 1
            return analysis
        RUN_STATS['vlm_skipped_page_count'] += 1
        print(f"    {page_name}: VLM确认跳过{analysis.get('page_type')}页")
        return {'_skip_page': True, **analysis}
    return analysis


def _ocr_item_bounds(item):
    """Return a resilient rectangular bound for a PaddleOCR item."""
    points = item.get('bbox', []) if isinstance(item, dict) else []
    if not isinstance(points, list) or not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _inside_any_box(item, boxes):
    bounds = _ocr_item_bounds(item)
    if not bounds:
        return False
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    return any(box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3] for box in boxes)


def _dedupe_ocr_items(items):
    """Prefer high-resolution region OCR but keep outside-region page OCR intact."""
    unique = []
    seen = set()
    for item in items:
        bounds = _ocr_item_bounds(item)
        if not bounds:
            continue
        key = (
            re.sub(r'\s+', '', str(item.get('text', ''))),
            *(int(round(value / 4.0)) for value in bounds),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def refine_answer_ocr_with_vlm(ans_path, img, base_items, context_hint=''):
    """Use VLM only to focus OCR on answer-context regions, never to crop answers.

    Dense answer keys are often scanned as a very wide spread.  Running OCR on
    the whole spread makes the type too small and loses many anchors.  The VLM
    is asked for coarse titled regions; each is re-read at high resolution and
    the existing local parser still finds every minor number and final crop.
    """
    if not vlm_hybrid.enabled() or os.environ.get(
        'PAPERCUTTER_VLM_ANSWER_RESCAN', 'off',
    ).strip().lower() != 'on':
        return base_items
    height, width = img.shape[:2]
    # Answer keys in this source are often two-page horizontal spreads.  At
    # that scale Qwen can identify the page but not the individual titled
    # answer contexts.  Analyse each physical half and map boxes back.
    views = [(0, 0, img)]
    if width / max(1, height) >= 1.32:
        split = width // 2
        views = [(0, 0, img[:, :split]), (split, 0, img[:, split:])]
    sections = []
    for offset_x, offset_y, view in views:
        started = time.perf_counter()
        analysis = vlm_hybrid.analyze_answer_sections(view)
        _add_timing('vlm_answer_section_analysis', time.perf_counter() - started)
        RUN_STATS['vlm_answer_page_count'] += 1
        if not analysis or analysis.get('page_type') != 'answer_key':
            RUN_STATS['vlm_failed_answer_page_count'] += 1
            continue
        for section in analysis.get('sections', []):
            box = section.get('bbox', [])
            if not isinstance(box, list) or len(box) != 4:
                continue
            mapped = dict(section)
            mapped['bbox'] = [
                int(box[0]) + offset_x, int(box[1]) + offset_y,
                int(box[2]) + offset_x, int(box[3]) + offset_y,
            ]
            sections.append(mapped)
    RUN_STATS['vlm_answer_section_count'] += len(sections)
    hint = re.sub(r'\s+', '', str(context_hint or ''))
    if hint:
        matched_sections = [
            section for section in sections
            if hint in re.sub(r'\s+', '', str(section.get('id', '')))
        ]
        # A context mismatch is evidence that this answer page belongs to a
        # different paper.  Keep its ordinary OCR candidates, but do not spend
        # time re-reading or accidentally mixing that unrelated region.
        if not matched_sections:
            return base_items
        sections = matched_sections
    if not sections:
        return base_items
    margin = max(10, int(min(width, height) * 0.012))
    boxes = []
    focused_items = []
    for section in sections:
        raw_box = section.get('bbox', [])
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in raw_box)
        box = [
            max(0, x0 - margin), max(0, y0 - margin),
            min(width, x1 + margin), min(height, y1 + margin),
        ]
        if box[2] - box[0] < 80 or box[3] - box[1] < 40:
            continue
        try:
            focused_items.extend(ocr_image(ans_path, max_side=OCR_MAX_SIDE, crop_box=box))
            boxes.append(box)
        except (OSError, ValueError, cv2.error):
            continue
    if not focused_items:
        return base_items
    outside_items = [item for item in base_items if not _inside_any_box(item, boxes)]
    combined = _dedupe_ocr_items([*focused_items, *outside_items])
    RUN_STATS['vlm_answer_region_ocr_count'] += len(boxes)
    RUN_STATS['vlm_answer_region_item_count'] += len(focused_items)
    print(f"    [VLM] 答案上下文区域{len(boxes)}个，高分辨率复核{len(focused_items)}条OCR")
    return combined


def detect_question_blocks_with_model(img):
    """Return high-confidence learned question boxes without risking the rule path."""
    global _QUESTION_DETECTOR_NOTICE_EMITTED
    if QUESTION_DETECTOR_MODE != 'assist':
        return []
    detector = question_detector.get_question_detector()
    if not detector.available:
        if not _QUESTION_DETECTOR_NOTICE_EMITTED:
            print('[模型] 题块检测模型未安装，继续使用既有 OCR/规则切题')
            _QUESTION_DETECTOR_NOTICE_EMITTED = True
        return []
    started = time.perf_counter()
    detections = detector.detect(img)
    _add_timing('question_detector', time.perf_counter() - started)
    RUN_STATS['question_detector_page_count'] += 1
    RUN_STATS['question_detector_box_count'] += len(detections)
    if detector.error and not _QUESTION_DETECTOR_NOTICE_EMITTED:
        print(f'[模型] 题块检测器不可用，继续使用既有 OCR/规则切题：{detector.error}')
        _QUESTION_DETECTOR_NOTICE_EMITTED = True
    return detections


def detect_answer_blocks_with_model(img):
    """Return reviewed-answer model boxes; OCR records remain authoritative."""
    global _ANSWER_DETECTOR_NOTICE_EMITTED
    if ANSWER_DETECTOR_MODE != 'assist':
        return []
    detector = answer_detector.get_answer_detector()
    if not detector.available:
        if not _ANSWER_DETECTOR_NOTICE_EMITTED:
            print('[模型] 答案块检测模型尚未训练，继续使用现有答案 OCR 与配对规则')
            _ANSWER_DETECTOR_NOTICE_EMITTED = True
        return []
    started = time.perf_counter()
    detections = detector.detect(img)
    _add_timing('answer_detector', time.perf_counter() - started)
    RUN_STATS['answer_detector_page_count'] += 1
    RUN_STATS['answer_detector_box_count'] += len(detections)
    if detector.error and not _ANSWER_DETECTOR_NOTICE_EMITTED:
        print(f'[模型] 答案块检测器不可用，继续使用现有答案 OCR 与配对规则：{detector.error}')
        _ANSWER_DETECTOR_NOTICE_EMITTED = True
    return detections


def detect_answer_regions_with_model(img):
    """Return HiLEx internal answer/option boxes; never use them for pairing."""
    global _ANSWER_REGION_DETECTOR_NOTICE_EMITTED
    if ANSWER_REGION_DETECTOR_MODE != 'assist':
        return []
    detector = answer_region_detector.get_answer_region_detector()
    if not detector.available:
        if not _ANSWER_REGION_DETECTOR_NOTICE_EMITTED:
            print('[模型] 题内答案区域模型未部署，继续使用现有 OCR/规则切题')
            _ANSWER_REGION_DETECTOR_NOTICE_EMITTED = True
        return []
    started = time.perf_counter()
    detections = detector.detect(img)
    _add_timing('answer_region_detector', time.perf_counter() - started)
    RUN_STATS['answer_region_detector_page_count'] += 1
    RUN_STATS['answer_region_detector_box_count'] += len(detections)
    if detector.error and not _ANSWER_REGION_DETECTOR_NOTICE_EMITTED:
        print(f'[模型] 题内答案区域检测器不可用，继续使用现有 OCR/规则切题：{detector.error}')
        _ANSWER_REGION_DETECTOR_NOTICE_EMITTED = True
    return detections
# 本次运行的日志文件路径；并行子进程据此把输出追加入同一份日志。
RUN_LOG_PATH = None


def _positive_int_env(name, default):
    """读取正整数环境变量；错误配置时保持默认值。"""
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def ocr_cpu_threads(default=8):
    """单个 OCR 进程的 CPU 线程数；并行 worker 会通过环境变量降到 4。"""
    return max(1, min(_positive_int_env('PAPERCUTTER_OCR_THREADS', default), os.cpu_count() or default))


def structure_ocr_worker_count():
    """结构预识别并行度：默认最多三进程，允许环境变量降回单进程。"""
    configured = os.environ.get('PAPERCUTTER_STRUCTURE_WORKERS')
    if configured is not None:
        return max(1, min(3, _positive_int_env('PAPERCUTTER_STRUCTURE_WORKERS', 1)))
    # 24 逻辑线程的 Xeon 使用 3 个 worker；普通 4～8 线程电脑仍保持单进程，
    # 避免模型副本争抢内存和 CPU。
    return max(1, min(3, (os.cpu_count() or 1) // 8))


def _available_ram_gb():
    """返回当前可用物理内存（GB）；无法探测时返回 None。

    用 Windows API 探测（ctypes 是标准库，不额外引入依赖），异常时安静回退。
    """
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    return None


def _parallel_hardware_cap():
    """批量切题在当前硬件上安全的并行本数上限。

    - 每本教材的 OCR 推理约占满 6 个逻辑线程，CPU 上限按 逻辑线程 // 6 估算；
    - 每个并行进程要独占一份模型和整页图片缓冲（约 4GB），可用内存不足时继续压低。
    返回至少 1，即任何机器都至少允许顺序处理。
    """
    cpu_cap = max(1, (os.cpu_count() or 4) // 6)
    mem_cap = cpu_cap
    available_gb = _available_ram_gb()
    if available_gb is not None:
        mem_cap = max(1, int(available_gb // 4))
    return max(1, min(cpu_cap, mem_cap))


def _book_parallelism(requested, book_count):
    """把请求的并行本数压到当前硬件安全范围（环境变量或 GUI 传入）。"""
    return max(1, min(int(requested), int(book_count), _parallel_hardware_cap()))


def _add_timing(name, seconds):
    RUN_STATS[f'{name}_seconds'] += max(0.0, float(seconds))
NON_QUESTION_SECTION_KEYWORDS = (
    '卷面书写', '答题须知', '考试须知', '温馨提示', '注意事项',
    '考生信息', '考场规则', '密封线', '装订线', '阅卷说明',
)
INSTRUCTIONAL_KEYWORDS = (
    '课前导学', '教材解读', '知识点', '目标点击', '温故知新',
    '思路点拨', '规范解答', '方法归纳', '例题', '典例',
)
EXERCISE_KEYWORDS = (
    '练一练', '巩固练习', '达标检测', '能力提升', '基础训练',
    '课后练习', '自主练习', '应用题', '填空', '判断', '选择',
    '计算', '解决问题', '作业',
)
ACTIVITY_HEADING_KEYWORDS = (
    '数学游戏', '假期体验', '开心乐园', '智慧乐园', '趣味数学',
    '生活数学', '实践活动', '拓展天地',
)
# 奥数/思维类辅导书常用彩色栏目名组织页面，而不是“一、二、三”式大题。
# 这里只收录可以“整行精确识别”的题目栏目；刻意不把“点拨”“拓展”等
# 讲解标签加入，避免把同一道题的讲解拆成另一个题目目录。
OLYMPIAD_FIELD_PATTERNS = (
    (re.compile(r'^难题点拨(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])?$'), 2000),
    # 彩色标题里的“一”常被 OCR 漏掉，允许“想想做一做”这一种受控变体。
    (re.compile(r'^想(?:一)?想做一做$'), 2200),
    (re.compile(r'^(?:例题|典型例题)$'), 2300),
    (re.compile(r'^(?:练一练|试一试|举一反三|挑战自我|闯关练习|巩固提升)$'), 2400),
)
OLYMPIAD_LESSON_RE = re.compile(
    r'^第(?:[一二三四五六七八九十百千万]+|\d{1,3})(?:讲|课|节)'
)
OLYMPIAD_CIRCLED_NUMBERS = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
# “①把动物分类……”这种行 = 圈号 + 题干；只有一个圈号的行是图注，不匹配。
RE_CIRCLED_STEM = re.compile(
    r'^([' + OLYMPIAD_CIRCLED_NUMBERS + r'])\s*(.+)$', re.S
)
COVER_KEYWORDS = ('ISBN', '定价', '责任编辑', '封面设计', '出版发行', '版权所有')
NON_CONTENT_PAGE_KEYWORDS = ('目录', '图书在版编目', 'CIP数据', '版权页')
PAPER_START_RE = re.compile(
    r'(?:第[一二三四五六七八九十百\d]+(?:单元|次|套)?'
    r'|期中|期末|月考|开学|专项|模拟|冲刺|综合|毕业|升学|学业|阶段|培优|达标)'
    r'.{0,22}(?:测试|检测|练习|试卷|卷)'
)
# 仅用于“新试卷边界”的严格卷名。PAPER_START_RE 仍保留给版面画像等
# 宽松场景使用；不能把正文里“期末测试成绩单”一类文字当成新卷标题。
PAPER_TITLE_RE = re.compile(
    r'^(?:第[一二三四五六七八九十百\d]+(?:单元|次|套)?[^。；，、]{0,16}'
    r'|(?:期中|期末|月考|开学|专项|模拟|冲刺|综合|毕业|升学|学业|阶段|培优|达标)[^。；，、]{0,16})'
    r'(?:测试卷|检测卷|练习卷|试卷|测试|检测|练习|卷)(?:[（(][^）)]{0,16}[）)])?$'
)
LESSON_TITLE_RE = re.compile(
    r'^\s*(\d{1,3})\s*(?:[\.．、]|\s)\s*[“"《]?\s*([\u4e00-\u9fff].{0,30})'
)


def emit_progress(phase, current, total, message='', **extra):
    """输出仅供图形界面读取的进度事件，不参与切题逻辑。

    事件带上当前教材名，GUI 才能在多本并行时区分进度来自哪一本书。
    """
    payload = {
        'phase': str(phase),
        'current': max(0, int(current or 0)),
        'total': max(0, int(total or 0)),
        'message': str(message or ''),
        'book': CURRENT_RUN_CONTEXT.get('book_name') or '',
    }
    # Page-preview events carry paths and coordinates only, never image data;
    # the local desktop client reads the original page directly from disk.
    payload.update(extra)
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def reset_run_stats():
    RUN_STATS.clear()


def report_run_stats():
    """输出本次速度诊断，便于在不猜测的前提下持续优化。"""
    print(
        "[性能] 轻量结构OCR{fast:.0f}页/{fast_time:.1f}秒；高精度OCR{medium:.0f}页/{medium_time:.1f}秒；"
        "缓存命中{hit:.0f}次，题面高精度回退{fallback:.0f}页、答案回退{answer_fallback:.0f}页；"
        "题图归一化{norm:.0f}次/{norm_time:.1f}秒".format(
            fast=RUN_STATS.get('fast_ocr_miss_count', 0),
            fast_time=RUN_STATS.get('fast_ocr_infer_seconds', 0.0),
            medium=RUN_STATS.get('medium_ocr_miss_count', RUN_STATS.get('ocr_miss_count', 0)),
            medium_time=RUN_STATS.get('medium_ocr_infer_seconds', RUN_STATS.get('ocr_infer_seconds', 0.0)),
            miss=RUN_STATS.get('ocr_miss_count', 0),
            ocr=RUN_STATS.get('ocr_infer_seconds', 0.0),
            hit=RUN_STATS.get('ocr_cache_hit_count', 0),
            fallback=RUN_STATS.get('ocr_high_precision_fallback_count', 0),
            answer_fallback=RUN_STATS.get('answer_ocr_high_precision_fallback_count', 0),
            norm=RUN_STATS.get('normalize_count', 0),
            norm_time=RUN_STATS.get('normalize_seconds', 0.0),
        )
    )
    print(
        "[性能-阶段] 结构预识别墙钟{structure_wall:.1f}秒（{workers:.0f}进程）；"
        "页面读取{read:.1f}秒；纠斜{deskew:.1f}秒；大题版式{major:.1f}秒；"
        "小题切分{minor:.1f}秒；质量核验{quality:.1f}秒；最终写图{write:.1f}秒；发布{publish:.1f}秒".format(
            structure_wall=RUN_STATS.get('structure_scan_wall_seconds', 0.0),
            workers=RUN_STATS.get('structure_scan_workers', 1.0),
            read=RUN_STATS.get('page_read_seconds', 0.0),
            deskew=RUN_STATS.get('deskew_seconds', 0.0),
            major=RUN_STATS.get('major_split_seconds', 0.0),
            minor=RUN_STATS.get('minor_split_seconds', 0.0),
            quality=RUN_STATS.get('quality_check_seconds', 0.0),
            write=RUN_STATS.get('final_write_seconds', 0.0),
            publish=RUN_STATS.get('publish_seconds', 0.0),
        )
    )

CN_NUM = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
    '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
    '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24,
    '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28,
    '二十九': 29, '三十': 30,
}

# 大题编号正则：必须跟顿号/点号分隔符
# 增强：也匹配 '、' 开头（OCR丢掉中文数字的情况，如"、填空题"缺少"一"）
RE_MAJOR = re.compile(r'^([一二三四五六七八九十百]+)[、．\.、]|^[、．\.、]\s*(填空|选择|判断|计算|解答|应用|操作|画图|看图|解决问题)')
# 小题编号正则。点号后不能紧跟数字，避免把“4.8-1.6”当成第4题。
RE_MINOR = re.compile(r'^([0-9]{1,3})\s*(?:[\.．](?!\d)|、)\s*')
RE_MINOR_PAREN = re.compile(r'^[（(]\s*([0-9]{1,3})\s*[）)]')
RE_MINOR_CIRCLE = re.compile(r'^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])')
# 单元/课时/试卷标记
RE_UNIT_TITLE = re.compile(r'^第\s*([一二三四五六七八九十\d]+)\s*(单元|课时|章节|部分|讲|练|试卷|周|次|套)')
RE_PRACTICE_TITLE = re.compile(
    r'^第\s*([一二三四五六七八九十\d]+)\s*练(?:习)?(?:[：:、.．—-]?\s*(.{1,30}))?$'
)
# 试卷名称标记（如"第一单元过关检测卷"、"专项4式与方程"、"期中综合检测卷"、"期末冲刺夺分卷"）
RE_PAPER_TITLE = re.compile(r'^(第[一二三四五六七八九十\d]+单元|专项[⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲\d一二三四五六七八九十]*|期中|期末|开学|月考|期末冲刺|期中综合|综合检测|过关检测|梳理式诊断|模拟卷|冲刺卷|夺分卷)')

_ocr_engine = None
_ocr_api_v3 = False
_fast_ocr_engine = None
_fast_ocr_api_v3 = False
_fast_ocr_disabled = False
_ocr_memory_cache = {}


class TeeStream:
    """将终端输出同步写入 UTF-8 日志文件，可选回传给 GUI 进度通道。"""

    def __init__(self, original, log_file, progress_queue=None):
        self.original = original
        self.log_file = log_file
        self.progress_queue = progress_queue

    def write(self, data):
        # 并行 worker 的 stdout 可能是 None（父进程管道在 spawn 后已失效）；
        # 原输出不可用时仍要写日志文件，不能因此打断处理。
        if self.original is not None:
            self.original.write(data)
        self.log_file.write(data)
        if self.progress_queue is not None:
            # 回传给主进程以转发到 GUI 管道；队列满时丢弃进度行，绝不影响切题。
            try:
                self.progress_queue.put_nowait(data)
            except Exception:
                pass

    def flush(self):
        if self.original is not None:
            self.original.flush()
        self.log_file.flush()

    def isatty(self):
        return False


def start_run_log(log_dir):
    """创建一次运行日志，并返回用于恢复标准输出的状态。"""
    global RUN_LOG_PATH
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'cut_questions_{timestamp}.log')
    RUN_LOG_PATH = log_path
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f'[日志] {log_path}')
    return log_file, original_stdout, original_stderr


def get_ocr_engine():
    global _ocr_engine, _ocr_api_v3
    if _ocr_engine is None:
        # PaddleOCR 3.x 默认会检查远端模型源；本工具只使用已经安装在本机的模型。
        # 批量运行必须离线：即便系统环境变量曾被改写，也不检查远端模型源。
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        warnings.filterwarnings(
            "ignore",
            message=r"No ccache found.*",
            category=UserWarning,
        )
        from paddleocr import PaddleOCR
        print("[初始化] 加载PaddleOCR引擎...")
        parameters = inspect.signature(PaddleOCR).parameters
        if 'use_doc_orientation_classify' in parameters:
            _ocr_api_v3 = True
            # PaddleOCR 3.x：旧版 use_gpu/show_log/use_angle_cls 参数已被移除。
            # 当前 Windows CPU 组合启用 oneDNN 会触发 PIR 属性转换异常，明确关闭。
            _ocr_engine = PaddleOCR(
                lang='ch',
                device='cpu',
                enable_mkldnn=False,
                cpu_threads=ocr_cpu_threads(),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            print("[初始化] PaddleOCR 3.x CPU兼容模式")
        else:
            _ocr_api_v3 = False
            # 兼容仍在使用 PaddleOCR 2.x 的电脑。
            _ocr_engine = PaddleOCR(
                use_angle_cls=True, lang='ch', show_log=False, use_gpu=False,
            )
    return _ocr_engine


def get_fast_ocr_engine():
    """返回只用于页面结构定位的本地轻量 OCR；异常时让调用方回退高精度模型。

    模型目录明确指向项目 runtime，运行期间不会下载模型，因此批量处理可完全离线。
    """
    global _fast_ocr_engine, _fast_ocr_api_v3, _fast_ocr_disabled
    if _fast_ocr_disabled:
        return None
    if _fast_ocr_engine is not None:
        return _fast_ocr_engine
    if not (os.path.isdir(FAST_DET_MODEL_DIR) and os.path.isdir(FAST_REC_MODEL_DIR)):
        print('[警告] 未找到项目内轻量 OCR 模型，将用高精度模型完成结构扫描。')
        _fast_ocr_disabled = True
        return None
    try:
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        warnings.filterwarnings('ignore', message=r'No ccache found.*', category=UserWarning)
        from paddleocr import PaddleOCR
        parameters = inspect.signature(PaddleOCR).parameters
        print('[初始化] 加载本地轻量 OCR 结构扫描模型...')
        if 'use_doc_orientation_classify' not in parameters:
            print('[警告] 当前 PaddleOCR 版本不支持本地轻量模型参数，将使用高精度结构扫描。')
            _fast_ocr_disabled = True
            return None
        _fast_ocr_api_v3 = True
        _fast_ocr_engine = PaddleOCR(
            device='cpu',
            enable_mkldnn=False,
            cpu_threads=ocr_cpu_threads(),
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_detection_model_dir=FAST_DET_MODEL_DIR,
            text_recognition_model_name='PP-OCRv5_mobile_rec',
            text_recognition_model_dir=FAST_REC_MODEL_DIR,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print('[初始化] 本地轻量 OCR 已就绪（仅结构扫描）')
        return _fast_ocr_engine
    except Exception as exc:
        print(f'[警告] 轻量 OCR 初始化失败，将使用高精度结构扫描：{exc}')
        _fast_ocr_disabled = True
        _fast_ocr_engine = None
        return None


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def cv2_imread(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as e:
        print(f"[警告] 读取图片失败: {path} - {e}")
        return None

def cv2_imwrite(path, img):
    try:
        ext = os.path.splitext(path)[1] or '.jpg'
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
        return False
    except (OSError, cv2.error) as e:
        print(f"[警告] 写入图片失败: {path} - {e}")
        return False
    return False


def _json_safe_crop_info(crop_info):
    """移除仅供内存复用的掩码，并把 NumPy 数值转换为 JSON 基本类型。"""
    result = {}
    for key, value in (crop_info or {}).items():
        if key == 'content_mask':
            continue
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, tuple):
            value = list(value)
        result[key] = value
    return result


def build_question_metadata(
    minor, major_idx, major_title, sequence, crop_info, quality_issues,
    pairing=None, answer_source=None,
):
    """生成与题图并排保存的可训练来源信息；不包含大数组或 OCR 全文。"""
    return {
        'schema_version': 2,
        'algorithm_version': ALGORITHM_VERSION,
        'source': {
            'image': minor.get('_source_path', ''),
            'size': list(minor.get('_source_size', [])),
            'question_box': list(minor.get('_source_box', [])),
            'major_box': list(minor.get('_major_box', [])),
            'page_deskew_angle': float(minor.get('_page_deskew_angle', 0.0)),
            'segments': _minor_source_segments(minor),
        },
        'prediction': {
            'major_idx': int(major_idx),
            'major_title': major_title,
            'sequence': int(sequence),
            'original_question_number': int(minor.get('num', sequence)),
            'number_detected': bool(minor.get('number_detected', True)),
            'number_confidence': float(minor.get('_number_confidence', 0.0)),
            'boundary_source': minor.get('_boundary_source', 'unknown'),
            'anchor_y_in_major': minor.get('_anchor_y'),
            'answer_region_assisted': bool(minor.get('_answer_region_assisted', False)),
            'answer_region_confidence': float(minor.get('_answer_region_confidence', 0.0)),
            'answer_region_box': list(minor.get('_answer_region_box', [])),
            'shared_context_attached': bool(minor.get('_shared_context_attached', False)),
            'shared_context_reason': minor.get('_shared_context_reason', ''),
            'shared_context_box': list(minor.get('_shared_context_box', [])),
            'vlm_primary_group_id': minor.get('_vlm_group_id', ''),
            'vlm_outside_returned_groups': bool(minor.get('_vlm_outside_group', False)),
        },
        'normalization': _json_safe_crop_info(crop_info),
        'quality_issues': list(quality_issues or []),
        'pairing': pairing or {
            'decision_band': 'no_answer',
            'score': 0.0,
            'reasons': ['尚未执行答案配对'],
            'features': {},
        },
        'answer_source': answer_source,
    }


def write_json_atomic(path, data):
    """原子写 JSON，避免任务中断后留下半截训练元数据。"""
    temp_path = path + '.tmp'
    try:
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(review_queue.json_safe(data), file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
        return True
    except (OSError, TypeError, ValueError) as error:
        print(f"[警告] 写入JSON失败: {path} - {error}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return False

def safe_dirname(name, max_len=60):
    """清理目录名"""
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', name)
    name = name.strip('. ')
    return name[:max_len] if len(name) > max_len else name


def image_sort_key(name):
    """优先按纯数字文件名排序，其他文件保持稳定的名称排序。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def list_images(directory):
    """返回目录下的图片文件，不递归扫描。"""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory), key=image_sort_key)
        if os.path.isfile(os.path.join(directory, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTS
    ]


def unique_paths(paths):
    """按首次出现顺序去重，避免同一答案页被处理两次。"""
    seen = set()
    result = []
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_paperflow_book_root(src_dir):
    """返回员工在 PaperFlow 中应打开的实际图片目录。"""
    root = os.path.abspath(src_dir)
    if list_images(root):
        return root
    image_children = [
        path for path in (
            os.path.join(root, name) for name in os.listdir(root)
        )
        if os.path.isdir(path)
        and not os.path.basename(path).startswith('_')
        and list_images(path)
    ]
    return image_children[0] if len(image_children) == 1 else root


def completion_marker_path(unit_dir):
    return os.path.join(unit_dir, COMPLETION_MARKER)


def is_completed_unit(unit_dir):
    """只有带有效完成标记的单元才允许跳过。"""
    marker = completion_marker_path(unit_dir)
    if not os.path.isfile(marker):
        return False, None
    try:
        with open(marker, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if (
            data.get('status') == 'complete'
            and data.get('algorithm_version') == ALGORITHM_VERSION
            and data.get('question_count', 0) > 0
        ):
            return True, data
    except (OSError, ValueError, TypeError) as error:
        print(f"  [警告] 无法读取完成标记，将重新处理: {error}")
    return False, None


def write_completion_marker(unit_dir, question_count, matched_count, answer_count):
    """在全部图片写入完成后记录单元结果，作为下次跳过的唯一依据。"""
    data = {
        'status': 'complete',
        'algorithm_version': ALGORITHM_VERSION,
        'question_count': question_count,
        'matched_answer_count': matched_count,
        'detected_answer_count': answer_count,
        'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    marker = completion_marker_path(unit_dir)
    temp_marker = marker + '.tmp'
    with open(temp_marker, 'w', encoding='utf-8') as file:
        json.dump(review_queue.json_safe(data), file, ensure_ascii=False, indent=2)
    os.replace(temp_marker, marker)


def book_completion_marker_path(book_out_dir):
    return os.path.join(book_out_dir, BOOK_COMPLETION_MARKER)


def source_book_fingerprint(src_dir):
    """生成不读图片像素的教材指纹，用于启动前整本跳过判断。

    只遍历文件名和大小，速度远快于 OCR，也不会受教材所在盘符变化影响。
    """
    digest = hashlib.sha256()
    image_count = 0
    total_size = 0
    for base, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(
            directory for directory in dirs if not directory.startswith('_')
        )
        for filename in sorted(files, key=image_sort_key):
            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(base, filename)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            relative = os.path.relpath(path, src_dir).replace('\\', '/').lower()
            digest.update(relative.encode('utf-8', errors='surrogatepass'))
            digest.update(b'\0')
            digest.update(str(size).encode('ascii'))
            digest.update(b'\n')
            image_count += 1
            total_size += size
    return {
        'sha256': digest.hexdigest(),
        'image_count': image_count,
        'total_size': total_size,
    }


def is_completed_book(src_dir, book_out_dir, fingerprint=None, output_mode='crops'):
    """在 OCR 前验证与当前输出模式匹配的整本完成结果。"""
    actual = fingerprint or source_book_fingerprint(src_dir)
    if output_mode == 'paperflow':
        project_path = os.path.join(src_dir, 'PaperFlow标注项目.json')
        if not os.path.isfile(project_path):
            return False, None
        try:
            with open(project_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            valid = (
                data.get('kind') == 'papercutter_manual_annotation'
                and data.get('preannotation_mode') == 'paperflow'
                # 旧版已发布项目没有 status；兼容它们，避免重跑覆盖人工补标。
                # 新版的不完整任务不会发布项目，若外部显式标为 incomplete 则绝不跳过。
                and data.get('status', 'complete') == 'complete'
                and data.get('algorithm_version') == ALGORITHM_VERSION
                and data.get('source_fingerprint') == actual
                and isinstance(data.get('pages'), list)
                and isinstance(data.get('answer_pages'), list)
            )
            return (True, data) if valid else (False, None)
        except (OSError, ValueError, TypeError) as error:
            print(f'  [警告] 无法读取 PaperFlow 预标注项目，将重新处理：{error}')
            return False, None

    marker = book_completion_marker_path(book_out_dir)
    if not os.path.isfile(marker) or os.path.isdir(os.path.join(book_out_dir, '_运行中')):
        return False, None
    try:
        with open(marker, 'r', encoding='utf-8') as file:
            data = json.load(file)
        valid = (
            data.get('status') == 'complete'
            and data.get('algorithm_version') == ALGORITHM_VERSION
            and data.get('source_fingerprint') == actual
        )
        return (True, data) if valid else (False, None)
    except (OSError, ValueError, TypeError) as error:
        print(f'  [警告] 无法读取整本完成标记，将继续核验：{error}')
        return False, None


def write_book_completion_marker(book_out_dir, fingerprint, unit_count):
    """仅在整本教材完整走完后写入，供下次在 OCR 前直接跳过。"""
    data = {
        'status': 'complete',
        'algorithm_version': ALGORITHM_VERSION,
        'source_fingerprint': fingerprint,
        'unit_count': int(unit_count),
        'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    marker = book_completion_marker_path(book_out_dir)
    temp_marker = marker + '.tmp'
    with open(temp_marker, 'w', encoding='utf-8') as file:
        json.dump(review_queue.json_safe(data), file, ensure_ascii=False, indent=2)
    os.replace(temp_marker, marker)


def adopt_legacy_completed_book(src_dir, book_out_dir, profile, fingerprint):
    """把可明确判断完整的旧版结果补成整本标记，避免首次升级仍做 OCR。

    必须所有可见单元目录都有有效完成标记，且不存在运行中暂存目录。
    “32练”等已知旧版会被错误合并成“第一课时”的场景不在此兼容，
    让新版规则重新按第X练切分。
    """
    if not os.path.isdir(book_out_dir) or os.path.isdir(os.path.join(book_out_dir, '_运行中')):
        return False
    children = [
        entry.path for entry in os.scandir(book_out_dir)
        if entry.is_dir() and not entry.name.startswith('_')
    ]
    if not children:
        return False
    # 对新版已修复的“思维拓展32练”不能沿用旧版整书合并结果。
    book_name = os.path.basename(os.path.normpath(src_dir))
    if (
        layout_strategy.strategy_for(profile).legacy_flat_unit_guard
        and len(children) == 1
        and os.path.basename(children[0]) == '第一课时'
        and '32练' in book_name
    ):
        return False
    if not all(is_completed_unit(child)[0] for child in children):
        return False
    write_book_completion_marker(book_out_dir, fingerprint, len(children))
    print(f'  已兼容旧版完成结果：{len(children)} 个单元，后续可在 OCR 前整本跳过。')
    return True


def publish_staged_unit(staging_dir, final_dir):
    """原子发布一个完成的单元；旧结果先改名保留，避免升级或中断造成混杂。"""
    backup_dir = None
    if os.path.isdir(final_dir):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        backup_root = os.path.join(os.path.dirname(final_dir), '_旧结果备份')
        os.makedirs(backup_root, exist_ok=True)
        backup_name = f'{safe_dirname(os.path.basename(final_dir), 60)}_{stamp}'
        backup_dir = os.path.join(backup_root, backup_name)
        suffix = 2
        while os.path.exists(backup_dir):
            backup_dir = os.path.join(backup_root, f'{backup_name}_{suffix}')
            suffix += 1
        os.replace(final_dir, backup_dir)
    try:
        os.replace(staging_dir, final_dir)
    except OSError:
        if backup_dir and not os.path.exists(final_dir) and os.path.isdir(backup_dir):
            os.replace(backup_dir, final_dir)
        raise
    return backup_dir


def cn_to_num(text):
    """中文数字转阿拉伯数字"""
    text = text.strip()
    if text in CN_NUM:
        return CN_NUM[text]
    # CN_NUM 覆盖常见的 1--30，但暑假作业、练习册会出现“三十一、”
    # 甚至“四十一、”这类大题号。此前这种编号会落到 int() 并返回 None，
    # 后续大题切分再把 None 当数字比较，导致整个批量任务中断。
    digit_values = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    }
    unit_values = {'十': 10, '百': 100}
    if text and all(char in digit_values or char in unit_values for char in text):
        total = 0
        current = 0
        saw_unit = False
        for char in text:
            if char in digit_values:
                current = digit_values[char]
                continue
            saw_unit = True
            # “十、”等省略“一”的写法按十处理。
            total += (current or 1) * unit_values[char]
            current = 0
        if saw_unit:
            return total + current
    try:
        return int(text)
    except (ValueError, TypeError):
        return None

def extract_major_idx(text):
    """提取大题号：'一、选择题' -> 1, '、填空题' -> 1
    增强版：处理OCR丢掉中文数字的情况
    """
    text = text.strip()
    m = RE_MAJOR.match(text)
    if m:
        if m.group(1):
            return cn_to_num(m.group(1))
        else:
            # 匹配到 '、填空题' 这种缺中文数字的情况，推断为第一大题
            return 1
    return None

def extract_minor_idx(text):
    """提取小题号：支持“7.”、“(7)”和圈号，排除小数开头。"""
    normalized = text.strip()
    for pattern in (RE_MINOR, RE_MINOR_PAREN):
        m = pattern.match(normalized)
        if not m:
            continue
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            pass
    m = RE_MINOR_CIRCLE.match(normalized)
    if m:
        circle_numbers = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
        return circle_numbers.index(m.group(1)) + 1
    return None


def extract_minor_anchor(text):
    """返回 (题号, 层级样式)，用于避免把(1)(2)子问与1. 2.主问题混切。"""
    normalized = text.strip()
    match = RE_MINOR.match(normalized)
    if match:
        return int(match.group(1)), 'primary'
    match = RE_MINOR_PAREN.match(normalized)
    if match:
        return int(match.group(1)), 'subpart'
    match = RE_MINOR_CIRCLE.match(normalized)
    if match:
        circle_numbers = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
        return circle_numbers.index(match.group(1)) + 1, 'circle'
    return None, None


def is_non_question_section(title):
    """判断明显不应切成练习题的说明性大题。"""
    normalized = re.sub(r'\s+', '', title or '')
    return any(keyword in normalized for keyword in NON_QUESTION_SECTION_KEYWORDS)


# 版式画像（profile）决定结构 OCR 的画布范围与大题边界规则。整套取值集中
# 在这里，界面只需要展示中文名，不需要知道内部代号。
BOOK_PROFILE_CHOICES = (
    'general', 'exam', 'workbook', 'drill', 'holiday', 'olympiad', 'instructional',
)
BOOK_PROFILE_LABELS = {
    'general': '通用（不套用专用规则）',
    'exam': '试卷 / 单元卷',
    'workbook': '课堂练习 / 作业本',
    'drill': '口算 / 速算题卡',
    'holiday': '假期作业',
    'olympiad': '奥数与思维训练',
    'instructional': '教材解读 / 讲解',
}
# 检测规则或抽样口径变化时必须改这个版本号，否则旧缓存会被继续复用。
BOOK_PROFILE_REVISION = 'book-profile-v2-20261019'
BOOK_PROFILE_SAMPLE_LIMIT = 3
# 抽样页结构 OCR 的总时间上限。试卷对开扫描单页可能要几十秒，宁可少读一页
# 也不能让后台识别无限拖下去（前台预标注会和它抢 CPU）。
BOOK_PROFILE_OCR_BUDGET = 150.0


def infer_book_profile_detail(src_dir):
    """根据书名推断版式，并返回命中的词，供界面解释“为什么是这个版式”。

    判定顺序即优先级，改动会直接影响老项目的识别结果，必须同步回归
    ``tests/test_smart_cutter.py`` 里的 profile 用例。
    """
    name = os.path.basename(os.path.normpath(src_dir))
    # 周测、单元测、达标检测等通常也是完整试卷；若先命中“训练”会被
    # 误归为 workbook，结构 OCR 只抽样两侧窄条，三栏题面会因此漏掉中栏。
    for word in (
        '试卷', '金卷', '卷王', '全能卷', '测试卷', '周测', '单元测',
        '达标检测', '阶段检测', '模拟卷', '月考',
    ):
        if word in name:
            return 'exam', word
    for word in ('口算', '题卡', '速算'):
        if word in name:
            return 'drill', word
    for word in ('暑假', '寒假'):
        if word in name:
            return 'holiday', word
    for word in ('奥数', '思维'):
        if word in name:
            return 'olympiad', word
    for word in ('教材解读', '教材全解', '知识讲解'):
        if word in name:
            return 'instructional', word
    for word in ('课堂', '同步', '作业', '练习', '训练'):
        if word in name:
            return 'workbook', word
    return 'general', ''


def infer_book_profile(src_dir):
    """根据书名和目录结构推断裁剪场景，供后续策略选择。"""
    return infer_book_profile_detail(src_dir)[0]


def refine_profile_from_text_detail(profile, text):
    """同 ``refine_profile_from_text``，额外返回判定依据。

    依据会写进项目，员工在界面上能看到“自动识别为试卷，因为正文含……”，
    判错时才知道该改成什么。
    """
    if profile != 'general':
        return profile, ''
    compact = re.sub(r'\s+', '', text or '')
    for word in ('暑假生活', '寒假生活', '暑假作业', '寒假作业', '假期体验'):
        if word in compact:
            return 'holiday', f'正文含“{word}”'
    if PAPER_START_RE.search(compact):
        return 'exam', '正文含单元/期中/期末测试卷名'
    if (
        '时间' in compact and '总分' in compact
        and any(word in compact for word in ('检测', '试卷', '考试'))
    ):
        return 'exam', '正文含“时间/总分”卷面抬头'
    for word in ('奥数', '举一反三', '思维训练', '竞赛专题'):
        if word in compact:
            return 'olympiad', f'正文含“{word}”'
    instructional_hits = [word for word in INSTRUCTIONAL_KEYWORDS if word in compact]
    if len(instructional_hits) >= 2:
        return 'instructional', '正文含' + '、'.join(f'“{word}”' for word in instructional_hits[:3])
    for word in ('口算题卡', '口算练习', '速算', '计时口算'):
        if word in compact:
            return 'drill', f'正文含“{word}”'
    exercise_hits = [word for word in EXERCISE_KEYWORDS if word in compact]
    if len(exercise_hits) >= 2:
        return 'workbook', '正文含' + '、'.join(f'“{word}”' for word in exercise_hits[:3])
    return profile, ''


def refine_profile_from_text(profile, text):
    """书名信息不足时，用页面正文把 general 场景细化为更合适的策略。"""
    return refine_profile_from_text_detail(profile, text)[0]


def refine_profile_from_images(profile, image_paths, ocr_max_side=OCR_FAST_MAX_SIDE):
    """抽取首尾少量页面做场景识别，优先复用轻量结构 OCR，避免预判阶段整页高精度识别。"""
    if profile != 'general' or not image_paths:
        return profile
    sample_indices = sorted(set(
        list(range(min(3, len(image_paths))))
        + list(range(max(0, len(image_paths) - 2), len(image_paths)))
    ))
    combined_text = []
    for index in sample_indices:
        try:
            img = cv2_imread(image_paths[index])
            if img is None:
                continue
            combined_text.extend(
                item.get('text', '')
                for item in ocr_page_structure(image_paths[index], img)
            )
        except Exception:
            continue
        detected = refine_profile_from_text(profile, '\n'.join(combined_text))
        if detected != 'general':
            return detected
    return profile


def book_profile_cache_path(book_root, runtime_root=None):
    """版式画像按教材根目录落盘，路径不随 Python 进程重启而变化。"""
    root = runtime_root or RUNTIME_DIR
    key = hashlib.sha1(
        os.path.normcase(os.path.abspath(book_root)).encode('utf-8', 'surrogateescape')
    ).hexdigest()
    return os.path.join(str(root), 'profile_cache', f'{key}.json')


def _profile_sample_signature(paths):
    """抽样页的“身份”：文件名 + 大小 + mtime，任一项变化即视为换了页面。"""
    signature = []
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            signature.append([os.path.basename(path), -1, -1])
            continue
        signature.append([os.path.basename(path), int(stat.st_size), int(stat.st_mtime_ns)])
    return signature


def _profile_sample_paths(image_paths, limit=BOOK_PROFILE_SAMPLE_LIMIT):
    """等间隔取页，避开封面与版权页，同时保证首尾都有覆盖。"""
    paths = [path for path in image_paths if path]
    if not paths:
        return []
    if len(paths) <= limit:
        return paths
    fractions = (0.12, 0.5, 0.85) if limit == 3 else tuple(
        (index + 1) / (limit + 1) for index in range(limit)
    )
    picked = []
    for fraction in fractions[:limit]:
        index = min(len(paths) - 1, max(0, int(round(fraction * (len(paths) - 1)))))
        if paths[index] not in picked:
            picked.append(paths[index])
    return picked


def read_book_profile_cache(book_root, runtime_root=None):
    """读取已落盘的版式判定；版本号或教材路径不符时一律当作未命中。"""
    path = book_profile_cache_path(book_root, runtime_root)
    try:
        with open(path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get('revision') != BOOK_PROFILE_REVISION:
        return None
    if payload.get('profile') not in BOOK_PROFILE_CHOICES:
        return None
    return payload


def _write_book_profile_cache(book_root, payload, runtime_root=None):
    path = book_profile_cache_path(book_root, runtime_root)
    temporary = f'{path}.tmp'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass


def _profile_title_lines(ocr_items):
    """只保留像标题的短行，用来判定教材版式。

    整页正文直接喂给 ``refine_profile_from_text`` 会被前言坑到：100 分攻略的
    “编写说明”里写着“分课时同步练习和阶段检测卷两部分”，于是整本书被判成
    试卷。卷名、栏目名都是短行且不含句读，按这个形状过滤能挡掉绝大多数前言
    与讲解段落；漏判只会退回 general，比判错安全。
    """
    lines = []
    for item in ocr_items:
        text = re.sub(r'\s+', '', item.get('text', '') or '')
        if not text or len(text) > 28:
            continue
        if any(mark in text for mark in '。；，、'):
            continue
        lines.append(text)
    return lines


def detect_book_profile(book_root, image_paths=None, runtime_root=None,
                        force=False, sample_limit=BOOK_PROFILE_SAMPLE_LIMIT,
                        ocr_budget=BOOK_PROFILE_OCR_BUDGET, ocr_guard=None):
    """判定教材版式：书名优先，书名判不出来时抽样正文。

    返回 ``profile`` / ``source``（name/text/cache/general）``reason`` /
    ``sampled`` / ``cached``。抽样 OCR 是这里唯一的重活，结果按教材落盘，
    同一本教材只在首次（或抽样页变化后）真正跑一次。

    ``ocr_guard`` 是调用方提供的上下文管理器；桥服务用它逐页借用预标注推理
    锁，使后台抽样最多只挡住一次前台点击，而不是整轮识别。
    """
    book_root = os.path.abspath(book_root)
    name_profile, name_word = infer_book_profile_detail(book_root)
    base = {
        'book_root': book_root,
        'name': os.path.basename(os.path.normpath(book_root)),
        'name_profile': name_profile,
        'revision': BOOK_PROFILE_REVISION,
    }
    if name_profile != 'general':
        return {**base, 'profile': name_profile, 'source': 'name',
                'reason': f'书名含“{name_word}”', 'sampled': [], 'cached': False}

    paths = list(image_paths) if image_paths is not None else list_images(book_root)
    sample = _profile_sample_paths(paths, sample_limit)
    signature = _profile_sample_signature(sample)
    cached = None if force else read_book_profile_cache(book_root, runtime_root)
    if cached and cached.get('sample') == signature:
        return {**base, 'profile': cached['profile'], 'source': 'cache',
                'reason': cached.get('reason', ''), 'sampled': cached.get('sampled', []),
                'cached': True}
    if not sample:
        return {**base, 'profile': 'general', 'source': 'general',
                'reason': '教材目录下没有可抽样的页面', 'sampled': [], 'cached': False}

    started = time.perf_counter()
    texts = []
    sampled = []
    for path in sample:
        if texts and time.perf_counter() - started > ocr_budget:
            break
        try:
            img = cv2_imread(path)
            if img is None:
                continue
            if ocr_guard is None:
                items = ocr_page_structure(path, img)
            else:
                with ocr_guard:
                    items = ocr_page_structure(path, img)
            texts.extend(_profile_title_lines(items))
            sampled.append(os.path.basename(path))
        except Exception as exc:  # 单页失败不能拖垮整本教材的判定
            print(f'[警告] 版式抽样识别失败（{os.path.basename(path)}）：{exc}')
    profile, reason = refine_profile_from_text_detail('general', '\n'.join(texts))
    payload = {
        **base,
        'profile': profile,
        'reason': reason or ('已读抽样页但未命中版式关键词，保持通用' if texts else '抽样页没有读出文字'),
        'sampled': sampled,
        'sample': signature,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    _write_book_profile_cache(book_root, payload, runtime_root)
    return {**payload, 'source': 'text', 'cached': False}


def auto_book_profile(book_root, runtime_root=None):
    """不跑 OCR 的版式判定：书名优先，其次复用抽样结论。返回 (profile, source, reason)。"""
    profile, word = infer_book_profile_detail(book_root)
    if word:
        return profile, 'auto_name', f'书名含“{word}”'
    cached = read_book_profile_cache(book_root, runtime_root)
    if cached:
        return cached['profile'], 'auto_text', str(cached.get('reason', '') or '')
    return 'general', 'default', '书名未提供版式线索，暂用通用版式'


def resolve_book_profile(src_dir, image_root=None):
    """命令行入口使用的版式：书名优先，其次复用界面落盘的抽样结论。

    界面（桥服务）会把抽样正文判出的版式写进 ``runtime/profile_cache``；
    命令行对同一本书必须拿到同一个结论，否则“界面切得对、命令行切得错”。
    """
    profile, word = infer_book_profile_detail(src_dir)
    if word:
        return profile
    for candidate in (image_root, src_dir):
        if not candidate:
            continue
        cached = read_book_profile_cache(candidate)
        if cached and cached.get('profile') in BOOK_PROFILE_CHOICES:
            return cached['profile']
    return profile


def _normalized_page_text(ocr_items):
    return '\n'.join(item.get('text', '').strip() for item in ocr_items if item.get('text', '').strip())


def _top_ocr_items(ocr_items, page_height, ratio=0.35):
    limit = page_height * ratio
    return [
        item for item in ocr_items
        if min(point[1] for point in item['bbox']) <= limit
    ]


def _answer_title_score(top_items):
    """只把独立答案标题视为答案页，排除“选择正确答案”这类题干。"""
    score = 0
    for item in top_items:
        text = re.sub(r'\s+', '', item.get('text', ''))
        if re.fullmatch(r'(?:参考|试题)?答案(?:与解析)?', text):
            score += 6
            continue
        for keyword in ANSWER_TITLE_KEYWORDS:
            compact_keyword = keyword.replace(' ', '')
            if compact_keyword == '答案':
                continue
            if compact_keyword in text and len(text) <= len(compact_keyword) + 8:
                score += 3
                break
    return score


def detect_practice_title(ocr_items, page_height):
    """提取平铺“第X练 + 练名”页眉，作为思维训练册的单元名。

    这类书经常把“第6练”和“乘法速算”分成相邻 OCR 框；仅在页面
    顶部寻找，并要求“第X练”完整锚点，避免把正文的练习说明误当新单元。
    """
    if page_height <= 0:
        return ''
    top_items = _top_ocr_items(ocr_items, page_height, 0.16)
    ordered = sorted(
        top_items,
        key=lambda item: (
            min(point[1] for point in item['bbox']),
            min(point[0] for point in item['bbox']),
        ),
    )
    for item in ordered:
        text = re.sub(r'\s+', '', item.get('text', ''))
        match = RE_PRACTICE_TITLE.fullmatch(text)
        if not match:
            continue
        number = match.group(1)
        suffix = (match.group(2) or '').strip('：:、.．—- ')
        y0 = min(point[1] for point in item['bbox'])
        x1 = max(point[0] for point in item['bbox'])
        if not suffix:
            same_row = []
            for peer in ordered:
                if peer is item:
                    continue
                peer_y0 = min(point[1] for point in peer['bbox'])
                peer_x0 = min(point[0] for point in peer['bbox'])
                peer_text = re.sub(r'\s+', '', peer.get('text', ''))
                if (
                    peer_text
                    and peer_x0 >= x1 - 8
                    and abs(peer_y0 - y0) <= max(24, page_height * 0.018)
                    and len(peer_text) <= 30
                    and not RE_PRACTICE_TITLE.fullmatch(peer_text)
                ):
                    same_row.append((peer_x0, peer_text))
            if same_row:
                suffix = sorted(same_row)[0][1]
        title = f'第{number}练'
        return f'{title} {suffix}' if suffix else title
    return ''


def detect_page_title(ocr_items, page_height):
    """从页面顶部挑选最可能的试卷/课时标题。"""
    practice_title = detect_practice_title(ocr_items, page_height)
    if practice_title:
        return practice_title
    top_items = _top_ocr_items(ocr_items, page_height, 0.28)
    candidates = []
    for item in top_items:
        text = re.sub(r'\s+', '', item.get('text', ''))
        if not (3 <= len(text) <= 70):
            continue
        # “八、下面是某班期末测试成绩单”是大题题干，即使它出现在
        # 页面上方，也绝不可以作为下一份试卷的标题。
        if re.match(r'^[一二三四五六七八九十百]+\s*[、，,．.]', text):
            continue
        score = 0
        if PAPER_TITLE_RE.fullmatch(text):
            score += 8
        if RE_UNIT_TITLE.search(text):
            score += 6
        if any(word in text for word in ('测试', '检测', '试卷', '课时', '单元', '专项')):
            score += 3
        if score == 0:
            continue
        box_height = max(point[1] for point in item['bbox']) - min(point[1] for point in item['bbox'])
        score += min(3, box_height / 30)
        candidates.append((score, text))
    # 试卷标题常因字号大/排版居中而被 OCR 拆成上下两行
    # （如“第二单元”+“达标检测卷”）。只合并明确像同一标题的两行，
    # 避免把“时间/分数/姓名”等页面信息或正文误拼成新的卷名。
    ordered = sorted(top_items, key=lambda it: (
        min(p[1] for p in it['bbox']), min(p[0] for p in it['bbox']),
    ))
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        a_text = re.sub(r'\s+', '', a.get('text', ''))
        b_text = re.sub(r'\s+', '', b.get('text', ''))
        if not a_text or not b_text:
            continue
        if re.search(r'(?:时间|总分|得分|分数|姓名|班级|密封|题号)\s*[:：]?', a_text + b_text):
            continue
        # 大题标题和正文即使恰好相邻，也绝不能参与卷名拼接。
        if RE_MAJOR.match(a_text) or RE_MAJOR.match(b_text):
            continue

        ax0 = min(p[0] for p in a['bbox'])
        ax1 = max(p[0] for p in a['bbox'])
        ay = max(p[1] for p in a['bbox'])
        bx0 = min(p[0] for p in b['bbox'])
        bx1 = max(p[0] for p in b['bbox'])
        by = min(p[1] for p in b['bbox'])
        line_gap = by - ay
        if line_gap < 0 or line_gap > max(36, int(page_height * 0.025)):
            continue
        a_width = max(1, ax1 - ax0)
        b_width = max(1, bx1 - bx0)
        center_distance = abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2)
        # 同一标题的两行通常水平中心接近；不同栏目/正文则相距很远。
        if center_distance > max(80, max(a_width, b_width) * 0.55):
            continue
        combined = a_text + b_text
        if not (4 <= len(combined) <= 70):
            continue
        if PAPER_TITLE_RE.fullmatch(combined):
            # 完整的两行卷名比单独识别到的“达标检测卷”等后半行更可靠。
            candidates.append((13, combined))
    return max(candidates, default=(0, ''), key=lambda candidate: candidate[0])[1]


def classify_page(img, ocr_items, profile='general'):
    """离线页面分类。返回 kind、标题和用于诊断的分数。"""
    height, width = img.shape[:2]
    del width
    text = _normalized_page_text(ocr_items)
    compact = re.sub(r'\s+', '', text)
    top_items = _top_ocr_items(ocr_items, height, 0.30)
    top_text = re.sub(r'\s+', '', '\n'.join(item.get('text', '') for item in top_items))

    major_count = sum(1 for item in ocr_items if extract_major_idx(item.get('text', '')) is not None)
    minor_count = sum(1 for item in ocr_items if extract_minor_idx(item.get('text', '')) is not None)
    answer_score = _answer_title_score(top_items)
    cover_score = sum(1 for word in COVER_KEYWORDS if word.upper() in compact.upper())
    instruction_score = sum(1 for word in INSTRUCTIONAL_KEYWORDS if word in compact)
    exercise_score = sum(1 for word in EXERCISE_KEYWORDS if word in compact)
    title = detect_page_title(ocr_items, height)
    strategy = layout_strategy.strategy_for(profile)

    if answer_score >= 3:
        kind = 'answer'
    elif any(word in compact for word in NON_CONTENT_PAGE_KEYWORDS) and major_count == 0:
        kind = 'non_content'
    elif strategy.wide_title_as_cover and '大试卷' in compact and major_count == 0 and minor_count == 0:
        kind = 'non_content'
    elif cover_score >= 2 and major_count == 0 and minor_count <= 1:
        kind = 'non_content'
    elif strategy.instructional_without_exercise and instruction_score >= 2 and exercise_score == 0:
        kind = 'instruction'
    elif len(ocr_items) <= 3 and major_count == 0:
        kind = 'non_content'
    elif any(word in compact for word in NON_QUESTION_SECTION_KEYWORDS) and major_count <= 1 and minor_count <= 1:
        kind = 'instruction'
    elif major_count or minor_count or exercise_score:
        kind = 'question'
    else:
        kind = 'unknown'

    return {
        'kind': kind,
        'title': title,
        'major_count': major_count,
        'minor_count': minor_count,
        'answer_score': answer_score,
        'cover_score': cover_score,
        'instruction_score': instruction_score,
        'exercise_score': exercise_score,
    }


def needs_high_precision_page_ocr(ocr_items, page_info):
    """快速 OCR 的保守质量门：证据不足时回退到既有 1800 精度。"""
    if not ocr_items:
        return True
    kind = page_info.get('kind', 'unknown')
    if kind == 'unknown':
        return True

    confidences = [float(item.get('conf', 1.0)) for item in ocr_items]
    if confidences and float(np.median(confidences)) < 0.88:
        return True

    anchors = [
        item for item in ocr_items
        if extract_major_idx(item.get('text', '')) is not None
        or extract_minor_idx(item.get('text', '')) is not None
    ]
    # 题号/大题标题决定切线；任何低置信锚点都不能使用低清结果冒险。
    if any(float(item.get('conf', 1.0)) < 0.82 for item in anchors):
        return True
    if kind == 'question' and not anchors:
        return True
    return False


def needs_high_precision_answer_ocr(ocr_items):
    """答案页降采样 OCR 的保守质量门，不能可靠读到文本时回退完整 1800。"""
    if len(ocr_items) < 6:
        return True
    confidences = [float(item.get('conf', 1.0)) for item in ocr_items]
    return bool(confidences and float(np.median(confidences)) < 0.72)


def is_likely_low_text_cover(ocr_items, page_info, page_index, page_total):
    """封面/版权等低文本首尾页无需为确认“跳过”再做整页 OCR。"""
    if page_info.get('kind') != 'unknown' or len(ocr_items) > 12:
        return False
    if any(
        extract_major_idx(item.get('text', '')) is not None
        or extract_minor_idx(item.get('text', '')) is not None
        for item in ocr_items
    ):
        return False
    return page_index < min(4, page_total) or page_index >= max(0, page_total - 2)


def paper_start_reason(page_info, profile='general'):
    """返回新试卷起始证据；只采纳页眉或试卷首页固定区，避免正文误分卷。"""
    title = re.sub(r'\s+', '', page_info.get('title', ''))
    if (
        title
        and PAPER_TITLE_RE.fullmatch(title)
        and not re.match(r'^[一二三四五六七八九十百]+\s*[、，,．.]', title)
    ):
        return '卷名'
    vlm_title = re.sub(r'\s+', '', page_info.get('vlm_paper_title', ''))
    if (
        vlm_title
        and (
            PAPER_TITLE_RE.fullmatch(vlm_title)
            or any(token in vlm_title for token in ('检测卷', '测试卷', '考卷', '达标卷', '月考'))
        )
    ):
        return 'VLM卷名'
    # “思维拓展32练”“奥数练习册”等平铺扫描文件没有子目录；每一练
    # 的页眉“第X练”就是可靠的新单元边界。仅在启用该策略的版式打开，
    # 以免普通教材正文出现“第X练”时被误拆。
    if layout_strategy.strategy_for(profile).practice_title_boundary and title:
        practice_match = RE_PRACTICE_TITLE.fullmatch(title)
        if practice_match:
            return '第X练'
    if not layout_strategy.strategy_for(profile).exam_cover_fallback:
        return ''

    # 某些扫描件的花体卷名会被轻量 OCR 漏读，但每份卷子首页稳定带有
    # “一、卷面书写”及题号/得分表。该文字只在首页出现，作为保守后备边界。
    items = page_info.get('ocr_items', [])
    compact = re.sub(r'\s+', '', _normalized_page_text(items))
    if '卷面书写' not in compact:
        return ''
    has_first_section = any(
        extract_major_idx(item.get('text', '')) == 1
        for item in items
    )
    score_signals = sum(word in compact for word in ('题号', '得分', '总分', '时间'))
    return '卷首版式' if has_first_section or score_signals >= 2 else ''


def is_paper_start(page_info, profile='general'):
    return bool(paper_start_reason(page_info, profile))


# D7-a：整条文本（去空白后）就是一道算式的字符集。只有"数字 / 小数点 / 括号 /
# 运算符 / 等号"这几类，任何一个汉字或字母出现就立刻不匹配。
_ARITHMETIC_ONLY_RE = re.compile(r'^[\d.()（）+\-*/×÷=＝<>≤≥−－]+$')
# `−`(U+2212) 与 `－`(U+FF0D) 是排版里真正的减号，OCR 常输出它们而不是 ASCII '-'
_ARITHMETIC_OPERATORS = '+-*/×÷=＝<>≤≥−－'


def looks_like_arithmetic_expression(text):
    """整段文字是否就是一道算式（`2.7+0.2=`、`10.67−4.07=`、`4.75×0.98`）。

    D7-a 的**双条件**：①归一化全文匹配算式字符集 ②至少含一个运算符或等号。
    两个条件缺一不可：只有数字和点的是小数（`2.7`），有运算符没数字的是噪声。

    **含汉字或字母的一律不判为算式**——"2.0.42的计数单位是" 这种"真题号被 OCR
    粘成小数"的形态必须继续走疑似题号救援，否则密排填空题会塌缩成一道题。
    """
    compact = re.sub(r'\s+', '', str(text or ''))
    if not compact or not _ARITHMETIC_ONLY_RE.match(compact):
        return False
    return any(operator in compact for operator in _ARITHMETIC_OPERATORS)


def find_minor_boundaries(ocr_items, crop_width, crop_height):
    """从 OCR 结果中找可信小题号。

    题号应位于当前栏的左侧，并与相邻题号保有足够的垂直距离；
    这样可过滤正文中被 OCR 误读成“数字+点”的碎片。
    """
    candidates = []
    # 疑似题号：形状像“题号+点+数字”，但被 extract_minor_anchor 判成小数。
    # 不立即采信，等主序列建好后只用来填补空缺。
    suspects = []
    # 题号一般贴着栏的左缘，用来挡住正文里的数字碎片。但横向并排的小问，
    # 题号本来就落在块中部，继续套"左 28%"会把 2. 3. 全丢掉。横排长条改为
    # 只做纵向合法性判断，横向交给 side_by_side 的连续序列校验兜底。
    side_by_side_layout = crop_height < crop_width * 0.6
    left_limit = crop_width if side_by_side_layout else max(90, int(crop_width * 0.28))
    line_heights = []
    for item in ocr_items:
        text = item['text'].strip()
        num, style = extract_minor_anchor(text)
        if num is None:
            # “1.6路公共汽车……”中的 1. 是小题号，OCR 漏掉空格后会被普通规则
            # 当成小数；“N 路公交”语义不会是数学小数，故仅对此极窄格式救回题号。
            route_number = re.match(
                r'^([0-9]{1,3})[\.．]\s*(?=\d{1,3}路(?:公共汽车|公交车|汽车)?)',
                text,
            )
            if route_number:
                num, style = int(route_number.group(1)), 'primary'
        if num is None:
            # OCR偶尔会吞掉题号后的空格，把“5. 36个铁圆锥”识别成“5.36个铁圆锥”。
            # 只对“第二段数字后紧跟明确量词”的形态做救援；普通4.8、35.2元仍按小数处理。
            joined = re.match(
                r'^([0-9]{1,2})[\.．][0-9]{1,4}'
                r'(?=个|只|条|本|名|枚|根|张|辆|块|棵|盒|台|件|位|把|杯|瓶|套|种|排|组|筐|桶|箱)',
                text,
            )
            if joined:
                num, style = int(joined.group(1)), 'primary'
        bbox = item['bbox']
        x = int(min(point[0] for point in bbox))
        y = int(min(point[1] for point in bbox))
        box_height = max(point[1] for point in bbox) - min(point[1] for point in bbox)
        if x < 0 or x > left_limit or y < 0 or y >= crop_height:
            continue
        if num is None:
            # 题号与题目正文之间没有空格时（“2.0.42的计数单位是”），整段开头
            # 会被当成小数 2.0，题号整批丢失——密排填空题因此塌缩成一道题。
            # 这里只登记为疑似，因为正文里的“4.75×0.98”“10.01×0.11”形状相同。
            leading = re.match(r'^([0-9]{1,2})[\.．]\s*\S', text)
            # D7-a：点后**紧邻**非空白、且整段就是一道算式时（`2.7+0.2=`），
            # 这是正文里的算式碎片，不能登记成疑似题号。否则它会被下面"正好补上
            # 已确认序列空缺"的通道救成编号，既把 `10.67−4.07=` 截断，又凭空多
            # 切一道题（minor_extra）。`2. 计算`（点后有空格）与含汉字的粘连题号
            # （`2.0.42的计数单位是`）都不满足这两个条件，救援通道照旧生效。
            arithmetic_fragment = (
                re.match(r'^[0-9]{1,2}[\.．][^\s]', text) is not None
                and looks_like_arithmetic_expression(text)
            )
            if (leading and not arithmetic_fragment
                    and item.get('conf', 1.0) >= MIN_OCR_CONFIDENCE):
                suspect_num = int(leading.group(1))
                if 1 <= suspect_num <= 150:
                    suspects.append({
                        'num': suspect_num, 'style': 'primary', 'text': text,
                        'y': y, 'x': x,
                        'confidence': float(item.get('conf', 1.0)),
                        '_suspect': True,
                    })
            continue
        if num > 150 or item.get('conf', 1.0) < MIN_OCR_CONFIDENCE:
            continue
        line_heights.append(max(1, box_height))
        candidates.append({
            'num': num, 'style': style, 'text': text, 'y': y, 'x': x,
            'confidence': float(item.get('conf', 1.0)),
        })

    if not candidates:
        return []
    primary_candidates = [candidate for candidate in candidates if candidate['style'] == 'primary']
    if primary_candidates:
        candidates = primary_candidates
    else:
        circle_candidates = [candidate for candidate in candidates if candidate['style'] == 'circle']
        if len(circle_candidates) >= 2:
            candidates = circle_candidates
        else:
            candidates = [candidate for candidate in candidates if candidate['style'] == 'subpart']
    candidates.sort(key=lambda item: (item['y'], item['x']))
    typical_height = int(np.median(line_heights)) if line_heights else 20
    min_gap = max(28, int(typical_height * 1.6))
    boundaries = []
    for candidate in candidates:
        if boundaries and candidate['y'] - boundaries[-1]['y'] < min_gap:
            gap = candidate['y'] - boundaries[-1]['y']
            # 判断题等版式的行距很紧，两个连续题号可能只相隔约一个字高。
            # 连续主编号且已经明显换行时仍应保留；真正同一行的选项数字则继续去重。
            compact_sequential = (
                candidate['style'] == 'primary'
                and boundaries[-1]['style'] == 'primary'
                and candidate['num'] == boundaries[-1]['num'] + 1
                and gap >= max(16, int(typical_height * 0.72))
            )
            # 横向并排的小问（1. 2. 3. 排在同一条水平线上）题号 y 差通常只有
            # 几个像素，靠 y 永远分不开。连续编号 + 明显横向间隔是可靠证据，
            # 先保留，由 split_major_into_minors 按 x 切开。
            side_by_side = (
                candidate['style'] == 'primary'
                and boundaries[-1]['style'] == 'primary'
                and candidate['num'] == boundaries[-1]['num'] + 1
                and candidate['x'] - boundaries[-1]['x'] >= max(120, int(crop_width * 0.12))
            )
            if compact_sequential or side_by_side:
                if side_by_side:
                    candidate['side_by_side'] = True
                    boundaries[-1]['side_by_side'] = True
                boundaries.append(candidate)
            elif candidate['x'] < boundaries[-1]['x']:
                # 同一行/相邻噪点只保留靠左的一个题号。
                boundaries[-1] = candidate
            continue
        boundaries.append(candidate)

    # 疑似题号只在“正好补上已确认序列的空缺”时采纳，即前一道已确认题的
    # 编号 +1。这样 2.0.42 / 3.3.47 / 4.1里面有 能救回成第 2、3、4 题，
    # 而第 8 题选项里的 4.75×0.98、10.01×0.11 因为编号不接续而继续被忽略。
    if suspects:
        merged = sorted(
            [(candidate['y'], 0, candidate) for candidate in boundaries]
            + [(suspect['y'], 1, suspect) for suspect in suspects],
            key=lambda value: (value[0], value[1]),
        )
        rescued = []
        for _y, kind, candidate in merged:
            if kind == 1:
                if rescued and candidate['num'] == rescued[-1]['num'] + 1:
                    candidate.pop('_suspect', None)
                    rescued.append(candidate)
                continue
            rescued.append(candidate)
        boundaries = rescued

    # 单个孤立“题号”很容易是正文数字；除非它确实是第1题，否则不据此强切。
    # 例外：多栏延续页/栏的顶部整段是上一题跨栏续文，栏内通常只有栏中部一个
    # 非1题号（如三栏试卷页第二栏只有“35.”，上方全部是上一题的子问续写）。
    # 此时题号距栏顶足够远、其上方存在可观续文，保留该边界由装配阶段剥离
    # 顶部续文；贴近栏顶的孤立数字仍按正文数字处理。
    if len(boundaries) == 1 and boundaries[0]['num'] != 1:
        min_fragment_height = max(60, min(150, int(crop_height * 0.30)))
        if boundaries[0]['y'] < min_fragment_height:
            return []

    # 选择整体最连贯的题号序列。允许同号重新开始，但过滤明显的随机跳号。
    coherent = []
    for candidate in boundaries:
        if not coherent:
            coherent.append(candidate)
            continue
        previous = coherent[-1]['num']
        current = candidate['num']
        if candidate['style'] == 'circle':
            is_coherent = current in {previous + 1, 1}
        else:
            is_coherent = current in {previous, previous + 1, 1} or (1 <= current - previous <= 3)
        if is_coherent:
            coherent.append(candidate)
    # 单独一个圈号通常是题内选项、图片标号，不足以证明存在多道顶层小题。
    if len(coherent) == 1 and coherent[0]['style'] != 'primary':
        return []
    return coherent


def find_unnumbered_boundaries(ocr_items, crop_width, crop_height, profile):
    """对装饰图标题号等 OCR 难以读出的场景，用大段垂直留白恢复题块。"""
    strategy = layout_strategy.strategy_for(profile)
    if strategy.unnumbered_activity:
        activity_starts = []
        for item in ocr_items:
            text = re.sub(r'\s+', '', item.get('text', ''))
            if any(keyword in text for keyword in ACTIVITY_HEADING_KEYWORDS):
                y = int(min(point[1] for point in item['bbox']))
                if crop_height * 0.05 < y < crop_height * 0.95:
                    activity_starts.append(y)
        activity_starts = sorted(set(activity_starts))
        if activity_starts:
            return [
                {'num': index, 'text': '活动分区', 'y': y, 'x': 0, 'detected': False}
                for index, y in enumerate([0] + activity_starts, 1)
            ]

    if not strategy.unnumbered_whitespace or len(ocr_items) < 3:
        return []
    lines = []
    for item in ocr_items:
        if item.get('conf', 1.0) < MIN_OCR_CONFIDENCE:
            continue
        bbox = item['bbox']
        x0 = min(point[0] for point in bbox)
        y0 = min(point[1] for point in bbox)
        y1 = max(point[1] for point in bbox)
        if x0 < 0 or x0 > crop_width * 0.38:
            continue
        lines.append({
            'x': x0,
            'y': y0,
            'bottom': y1,
            'text': item.get('text', '').strip(),
        })
    if len(lines) < 3:
        return []
    lines.sort(key=lambda line: line['y'])
    heights = [max(1, line['bottom'] - line['y']) for line in lines]
    typical_height = float(np.median(heights))
    large_gap = max(typical_height * 2.8, crop_height * 0.035)

    starts = []
    previous_bottom = 0
    for line in lines:
        gap = line['y'] - previous_bottom
        is_body_line = len(re.sub(r'\s+', '', line['text'])) >= 4
        if is_body_line and gap >= large_gap and line['y'] < crop_height * 0.92:
            starts.append(line)
        previous_bottom = max(previous_bottom, line['bottom'])
    if len(starts) < 2:
        return []
    return [
        {
            'num': index, 'text': line['text'], 'y': int(line['y']),
            'x': int(line['x']), 'detected': True,
        }
        for index, line in enumerate(starts, 1)
    ]


# ═══════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════
def _parse_ocr_result(result, api_v3=None):
    """把 PaddleOCR 2.x/3.x 的返回值统一为内部 OCR 项。"""
    if not result:
        return []

    if api_v3 is None:
        api_v3 = _ocr_api_v3
    if api_v3:
        first = result[0]
        payload = first.json if hasattr(first, 'json') else first
        data = payload.get('res', payload) if isinstance(payload, dict) else {}
        texts = data.get('rec_texts') or []
        scores = data.get('rec_scores') or []
        polys = data.get('rec_polys') or data.get('dt_polys') or []
        items = []
        for index, text in enumerate(texts):
            if index >= len(polys):
                break
            poly = np.asarray(polys[index]).reshape(-1, 2).tolist()
            score = float(scores[index]) if index < len(scores) else 1.0
            items.append({'text': str(text), 'bbox': poly, 'conf': score})
        return items

    items = []
    if result[0]:
        for line in result[0]:
            items.append({
                'text': line[1][0],
                'bbox': line[0],
                'conf': line[1][1],
            })
    return items


def _run_ocr(image_input):
    """对图片路径或 BGR numpy 数组运行高精度 OCR；避免临时文件写盘。"""
    engine = get_ocr_engine()
    if _ocr_api_v3:
        return _parse_ocr_result(engine.predict(image_input))
    return _parse_ocr_result(engine.ocr(image_input, cls=True))


_STRIP_OCR_CACHE_ENABLED = True


def _run_ocr_cached_strip(strip):
    """对内存图片做内容寻址缓存后跑 OCR。

    跳号补救会对每处跳号单独重识别一条横条带，同一页反复切分（界面里
    改一次框就重跑一次、离线审计要跑多轮）时会重复计算。缓存只存 OCR
    原文，切分逻辑仍然逐次重跑，所以改判据不受缓存影响。

    单元测试会 patch ``_run_ocr`` 换取不同结果，同一张合成图的缓存会跨
    用例命中；需要时把 ``_STRIP_OCR_CACHE_ENABLED`` 置 False。
    """
    if not _STRIP_OCR_CACHE_ENABLED:
        return _run_ocr(strip)
    if strip is None or not hasattr(strip, 'shape'):
        return _run_ocr(strip)
    digest = hashlib.sha1()
    digest.update(f'{OCR_CACHE_VERSION}|{strip.shape}'.encode('ascii'))
    digest.update(np.ascontiguousarray(strip).tobytes())
    key = digest.hexdigest()
    path = os.path.join(RUNTIME_DIR, 'ocr_cache', 'strip', key[:2], f'{key}.json')
    try:
        with open(path, encoding='utf-8') as file:
            return json.load(file)
    except (OSError, ValueError):
        pass
    items = _run_ocr(strip)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(items, file, ensure_ascii=False)
    except OSError:
        pass
    return items


def _run_fast_ocr(image_input):
    """运行项目内轻量模型；不可用时安全回退到既有高精度 OCR。
    接受图片路径或 BGR numpy 数组，避免为 OCR 输入额外写临时文件。
    """
    engine = get_fast_ocr_engine()
    if engine is None:
        return _run_ocr(image_input), False
    try:
        return _parse_ocr_result(engine.predict(image_input), api_v3=_fast_ocr_api_v3), True
    except Exception as exc:
        # 仅本页回退，既不停止任务，也不尝试联网获取任何模型。
        print(f'[警告] 轻量 OCR 本页识别失败，已回退高精度模型：{exc}')
        return _run_ocr(image_input), False


def _normalized_crop_box(crop_box):
    if crop_box is None:
        return None
    try:
        x0, y0, x1, y1 = (int(value) for value in crop_box)
    except (TypeError, ValueError):
        return None
    return x0, y0, x1, y1


def _ocr_cache_location(img_path, max_side=OCR_MAX_SIDE, crop_box=None, cache_tag=None):
    """生成内容缓存和旧版路径缓存位置；复制后的相同图片可复用 OCR。"""
    try:
        stat = os.stat(img_path)
        digest = hashlib.sha1()
        with open(img_path, 'rb') as file:
            digest.update(file.read(65536))
            if stat.st_size > 65536:
                file.seek(max(0, stat.st_size - 65536))
                digest.update(file.read(65536))
    except OSError:
        return None, None, None
    normalized_box = _normalized_crop_box(crop_box)
    cache_version = OCR_CACHE_VERSION if max_side == OCR_MAX_SIDE else f'{OCR_CACHE_VERSION}|side-{max_side}'
    if normalized_box is not None:
        cache_version += '|box-' + '-'.join(str(value) for value in normalized_box)
    if cache_tag:
        cache_version += '|tag-' + str(cache_tag)
    content_identity = f'{cache_version}|{stat.st_size}|{digest.hexdigest()}'
    key = hashlib.sha1(content_identity.encode('ascii')).hexdigest()
    legacy_identity = (
        f'{cache_version}|{os.path.abspath(img_path)}|{stat.st_size}|{stat.st_mtime_ns}'
    )
    legacy_key = hashlib.sha1(
        legacy_identity.encode('utf-8', errors='surrogatepass'),
    ).hexdigest()
    cache_dir = os.path.join(RUNTIME_DIR, 'ocr_cache')
    return (
        key,
        os.path.join(cache_dir, f'{key}.json'),
        os.path.join(cache_dir, f'{legacy_key}.json'),
    )


def ocr_image(img_path, max_side=OCR_MAX_SIDE, crop_box=None, force_refresh=False):
    """对整图或指定区域做 OCR；区域结果坐标始终映射回原始页面。

    ``force_refresh`` is reserved for an annotator explicitly asking to
    regenerate a page.  That action must exercise the current OCR + cutting
    pipeline, rather than returning an earlier disk or process-memory result.
    The newly recognised result still refreshes the normal cache afterwards.
    """
    max_side = max(640, int(max_side))
    crop_box = _normalized_crop_box(crop_box)
    cache_key, cache_path, legacy_cache_path = _ocr_cache_location(
        img_path, max_side=max_side, crop_box=crop_box,
    )
    if not force_refresh and cache_key and cache_key in _ocr_memory_cache:
        RUN_STATS['ocr_cache_hit_count'] += 1
        return [dict(item) for item in _ocr_memory_cache[cache_key]]
    readable_cache_path = None
    if not force_refresh:
        readable_cache_path = next(
            (
                path for path in (cache_path, legacy_cache_path)
                if path and os.path.isfile(path)
            ),
            None,
        )
    if readable_cache_path:
        try:
            with open(readable_cache_path, 'r', encoding='utf-8') as file:
                cached = json.load(file)
            if isinstance(cached, list):
                RUN_STATS['ocr_cache_hit_count'] += 1
                _ocr_memory_cache[cache_key] = cached
                if cache_path and readable_cache_path != cache_path:
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path + '.tmp', 'w', encoding='utf-8') as file:
                            json.dump(review_queue.json_safe(cached), file, ensure_ascii=False)
                        os.replace(cache_path + '.tmp', cache_path)
                    except OSError:
                        pass
                return [dict(item) for item in cached]
        except (OSError, ValueError, TypeError):
            pass

    engine = get_ocr_engine()
    del engine  # 初始化在这里发生，实际调用统一交给 _run_ocr。
    img = cv2_imread(img_path)
    if img is None:
        return []
    source_height, source_width = img.shape[:2]
    x_offset = y_offset = 0
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        x0 = max(0, min(source_width, x0))
        y0 = max(0, min(source_height, y0))
        x1 = max(0, min(source_width, x1))
        y1 = max(0, min(source_height, y1))
        if x1 <= x0 or y1 <= y0:
            return []
        img = img[y0:y1, x0:x1]
        x_offset, y_offset = x0, y0
    height, width = img.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        ocr_img = cv2.resize(
            img, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        ocr_img = img
    infer_started = time.perf_counter()
    # 直接把内存中的 BGR 数组交给模型，避免写临时文件再读回的磁盘往返。
    items = _run_ocr(ocr_img)
    elapsed = time.perf_counter() - infer_started
    RUN_STATS['ocr_miss_count'] += 1
    RUN_STATS['ocr_infer_seconds'] += elapsed
    RUN_STATS['medium_ocr_miss_count'] += 1
    RUN_STATS['medium_ocr_infer_seconds'] += elapsed

    if scale < 1.0:
        inverse = 1.0 / scale
        for item in items:
            item['bbox'] = [[point[0] * inverse, point[1] * inverse] for point in item['bbox']]
    if x_offset or y_offset:
        for item in items:
            item['bbox'] = [[point[0] + x_offset, point[1] + y_offset] for point in item['bbox']]
    # 按y坐标排序，确保从上到下的顺序（PaddleOCR不保证返回顺序）
    items.sort(key=lambda it: (min(p[1] for p in it['bbox']), min(p[0] for p in it['bbox'])))
    if cache_key and cache_path:
        _ocr_memory_cache[cache_key] = items
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            temp_path = cache_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as file:
                json.dump(review_queue.json_safe(items), file, ensure_ascii=False)
            os.replace(temp_path, cache_path)
        except OSError:
            pass
    return items


def _visual_column_split(img):
    """用中间留白检测双栏，不依赖整页 OCR 正文。证据不足时返回 None。"""
    height, width = img.shape[:2]
    if width < 700 or height < 500:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y0, y1 = int(height * 0.12), int(height * 0.94)
    sample = gray[y0:y1]
    scale = min(1.0, 1200 / max(1, width))
    if scale < 1.0:
        sample = cv2.resize(sample, (max(1, int(width * scale)), max(1, int(sample.shape[0] * scale))),
                            interpolation=cv2.INTER_AREA)
    ink_density = np.count_nonzero(sample < 210, axis=0).astype(np.float32)
    center_start, center_end = int(sample.shape[1] * 0.34), int(sample.shape[1] * 0.66)
    center = ink_density[center_start:center_end]
    if center.size < 12:
        return None
    smooth = np.convolve(center, np.ones(9, dtype=np.float32) / 9, mode='same')
    baseline = float(np.median(ink_density[ink_density > 0])) if np.any(ink_density > 0) else 0.0
    if baseline <= 0:
        return None
    low = smooth <= max(1.0, baseline * 0.16)
    best = None
    run_start = None
    for index, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            if best is None or index - run_start > best[1] - best[0]:
                best = (run_start, index)
            run_start = None
    if run_start is not None and (best is None or len(low) - run_start > best[1] - best[0]):
        best = (run_start, len(low))
    if best is not None and best[1] - best[0] >= max(10, int(sample.shape[1] * 0.018)):
        split_sample = center_start + (best[0] + best[1]) // 2
        split = int(split_sample / scale)
        # 两边都必须有足够墨迹，避免把单栏正文中的普通空隙误判成双栏。
        left_ink = float(np.sum(ink_density[:split_sample]))
        right_ink = float(np.sum(ink_density[split_sample:]))
        if min(left_ink, right_ink) >= max(100.0, baseline * 8):
            return split

    # 有些教辅是“左侧讲解栏 + 右侧练习栏”，中间以长实线分隔而非留白，
    # 且分隔线经常偏在页面左侧。若不识别它，右栏题目会把左栏解析带入。
    return _visual_vertical_divider(img)


def _visual_vertical_divider(img):
    """识别教辅讲解栏与练习栏之间的单条长竖线；证据不足时不分栏。"""
    height, width = img.shape[:2]
    if width < 700 or height < 700:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y0, y1 = int(height * 0.08), int(height * 0.95)
    # 排除装订边和页边线；教辅侧栏通常位于 18%～72% 之间。
    x0, x1 = int(width * 0.18), int(width * 0.72)
    if y1 <= y0 or x1 <= x0:
        return None
    strip = gray[y0:y1, x0:x1]
    darkness = np.count_nonzero(strip < 180, axis=0).astype(np.float32)
    # 用极小横向窗口容忍扫描造成的 1～2 像素线条偏移。
    kernel = np.ones(5, dtype=np.float32)
    line_strength = np.convolve(darkness, kernel, mode='same') / kernel.size
    candidate_offset = int(np.argmax(line_strength))
    candidate_strength = float(line_strength[candidate_offset])
    required_strength = (y1 - y0) * 0.48
    if candidate_strength < required_strength:
        return None
    # 一张表格往往有多条同样长的竖线；只有显著独立的长线才视作版式分隔。
    suppressed = line_strength.copy()
    left = max(0, candidate_offset - max(12, width // 180))
    right = min(len(suppressed), candidate_offset + max(13, width // 180))
    suppressed[left:right] = 0
    runner_up = float(np.max(suppressed)) if suppressed.size else 0.0
    if runner_up > 0 and candidate_strength < runner_up * 1.55:
        return None
    split = x0 + candidate_offset
    # 分隔线两侧都须存在正文，避免把题内图表的边框误当成版式边界。
    left_ink = float(np.sum(darkness[:candidate_offset]))
    right_ink = float(np.sum(darkness[candidate_offset + 1:]))
    if min(left_ink, right_ink) < max(80.0, np.median(darkness[darkness > 0]) * 6 if np.any(darkness > 0) else 80.0):
        return None
    return int(split)


def _merge_ocr_items(*groups):
    """合并标题区与题号带 OCR，去掉重叠区域的重复文本框。"""
    merged = []
    for item in sorted(
        (item for group in groups for item in group),
        key=lambda value: float(value.get('conf', 0.0)), reverse=True,
    ):
        text = str(item.get('text', '')).strip()
        box = item.get('bbox') or []
        if not text or not box:
            continue
        x0, y0 = min(point[0] for point in box), min(point[1] for point in box)
        x1, y1 = max(point[0] for point in box), max(point[1] for point in box)
        duplicate = False
        for existing in merged:
            if existing['text'] != text:
                continue
            existing_box = existing['bbox']
            ex0, ey0 = min(point[0] for point in existing_box), min(point[1] for point in existing_box)
            ex1, ey1 = max(point[0] for point in existing_box), max(point[1] for point in existing_box)
            if abs((x0 + x1) - (ex0 + ex1)) <= max(16, (x1 - x0 + ex1 - ex0) * 0.25) and \
               abs((y0 + y1) - (ey0 + ey1)) <= max(16, (y1 - y0 + ey1 - ey0) * 0.35):
                duplicate = True
                break
        if not duplicate:
            merged.append(dict(item))
    merged.sort(key=lambda value: (
        min(point[1] for point in value['bbox']), min(point[0] for point in value['bbox'])
    ))
    return merged


def ocr_page_structure(img_path, img, profile='general', force_refresh=False):
    """将标题和题号带拼成较小画布，只做一次 OCR 后映射回原页坐标。

    试卷首页的卷名常印在页高 13%～25% 的居中位置；旧版只保留最上方
    13%，会漏掉“第一单元达标检测卷”这类分卷锚点，进而把多套卷合并。
    因此 exam 场景扩大标题带，但仍只用轻量 OCR，不增加整页高精度 OCR。
    """
    height, width = img.shape[:2]
    strategy = layout_strategy.strategy_for(profile)
    header_ratio = strategy.header_ratio
    header_height = min(height, max(150, int(height * header_ratio)))
    header_scale = 0.72
    header_width = max(1, int(width * header_scale))
    header_canvas_height = max(1, int(header_height * header_scale))
    split = _visual_column_split(img)
    # A test sheet may place two or more exercise columns inside one physical
    # page.  Sampling only the outer 30% strips was a useful speed shortcut
    # for ordinary facing-page scans, but it silently discards the middle
    # column and makes later layout detection impossible.  For the exam
    # profile, keep the complete page in the one-pass structure OCR canvas.
    if strategy.full_width_structure:
        left_width = width
        right_source_x = None
        right_width = 0
    elif split is None:
        # 单页竖版没有装订缝，“只读左 30% 边条”就失去了前提：练习册常把
        # 四、五、六大题排在页面中右部，边条采样会整批丢掉，右半页只能退回
        # “整页”单块（全能测控 0002586A、奥赛天天练 006_0069 都是这个形态）。
        # 宽幅对开扫描仍保留边条采样，避免中间的装订白带白跑。
        if width / max(1, height) < 1.24:
            left_width = width
        else:
            left_width = min(width, max(180, int(width * 0.30)))
        right_source_x = None
        right_width = 0
    else:
        left_width = min(split, max(180, int(split * 0.34)))
        right_source_x = split
        right_width = min(width - split, max(180, int((width - split) * 0.34)))

    gap = 18
    right_canvas_x = left_width + gap
    canvas_width = max(header_width, right_canvas_x + right_width)
    content_canvas_y = header_canvas_height + gap
    canvas = np.full((content_canvas_y + height, canvas_width, 3), 255, dtype=np.uint8)
    header = cv2.resize(img[:header_height, :], (header_width, header_canvas_height), interpolation=cv2.INTER_AREA)
    canvas[:header_canvas_height, :header_width] = header
    canvas[content_canvas_y:content_canvas_y + height, :left_width] = img[:, :left_width]
    if right_source_x is not None:
        canvas[content_canvas_y:content_canvas_y + height, right_canvas_x:right_canvas_x + right_width] = (
            img[:, right_source_x:right_source_x + right_width]
        )

    # 缓存不能与旧版标题带范围不同的结果共用，避免升级后仍漏读试卷卷名。
    cache_tag = (
        f'anchor-mobile-v4-{profile}-{header_height}-{left_width}-'
        f'{right_source_x or 0}-{right_width}'
    )
    cache_key, cache_path, legacy_cache_path = _ocr_cache_location(
        img_path, max_side=OCR_ANCHOR_MAX_SIDE, cache_tag=cache_tag,
    )
    if not force_refresh and cache_key and cache_key in _ocr_memory_cache:
        RUN_STATS['ocr_cache_hit_count'] += 1
        return [dict(item) for item in _ocr_memory_cache[cache_key]]
    if not force_refresh and cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as file:
                cached = json.load(file)
            if isinstance(cached, list):
                RUN_STATS['ocr_cache_hit_count'] += 1
                _ocr_memory_cache[cache_key] = cached
                return [dict(item) for item in cached]
        except (OSError, ValueError, TypeError):
            pass

    scale = min(1.0, OCR_ANCHOR_MAX_SIDE / max(canvas.shape[:2]))
    ocr_canvas = canvas
    if scale < 1.0:
        ocr_canvas = cv2.resize(
            canvas,
            (max(1, int(canvas.shape[1] * scale)), max(1, int(canvas.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    started = time.perf_counter()
    # 直接传内存中的 BGR 数组，避免写临时文件再读回的磁盘往返。
    raw_items, used_lightweight = _run_fast_ocr(ocr_canvas)
    elapsed = time.perf_counter() - started
    RUN_STATS['ocr_miss_count'] += 1
    RUN_STATS['ocr_infer_seconds'] += elapsed
    if used_lightweight:
        RUN_STATS['fast_ocr_miss_count'] += 1
        RUN_STATS['fast_ocr_infer_seconds'] += elapsed
    else:
        RUN_STATS['medium_ocr_miss_count'] += 1
        RUN_STATS['medium_ocr_infer_seconds'] += elapsed
    if scale < 1.0:
        inverse = 1.0 / scale
        for item in raw_items:
            item['bbox'] = [[point[0] * inverse, point[1] * inverse] for point in item['bbox']]

    mapped = []
    for item in raw_items:
        box = item.get('bbox') or []
        if not box:
            continue
        center_x = sum(point[0] for point in box) / len(box)
        center_y = sum(point[1] for point in box) / len(box)
        if center_y < header_canvas_height:
            mapped_box = [[point[0] / header_scale, point[1] / header_scale] for point in box]
        elif center_y >= content_canvas_y and center_x < left_width:
            mapped_box = [[point[0], point[1] - content_canvas_y] for point in box]
        elif right_source_x is not None and center_y >= content_canvas_y and center_x >= right_canvas_x:
            mapped_box = [
                [right_source_x + point[0] - right_canvas_x, point[1] - content_canvas_y]
                for point in box
            ]
        else:
            continue
        copied = dict(item)
        copied['bbox'] = mapped_box
        mapped.append(copied)
    mapped = _merge_ocr_items(mapped)
    if cache_key and cache_path:
        _ocr_memory_cache[cache_key] = mapped
        try:
            with open(cache_path + '.tmp', 'w', encoding='utf-8') as file:
                json.dump(review_queue.json_safe(mapped), file, ensure_ascii=False)
            os.replace(cache_path + '.tmp', cache_path)
        except OSError:
            pass
    RUN_STATS['ocr_anchor_region_count'] += 1
    return [dict(item) for item in mapped]


def _init_structure_ocr_worker():
    """Windows 子进程初始化：每个轻量 OCR worker 只占用四个 CPU 线程。"""
    os.environ['PAPERCUTTER_OCR_THREADS'] = str(min(4, os.cpu_count() or 4))


def _structure_ocr_worker(image_path, profile='general'):
    """子进程任务：只做独立页面的结构 OCR，不写正式结果目录。

    ``profile`` 决定结构 OCR 的画布范围（exam 覆盖整页以保留多栏），必须
    与 ``prepare_structured_unit_ocr`` 中该页所属单元的 profile 一致，
    否则预计算缓存会被“只抽样两侧条带”的低覆盖结果取代。
    """
    reset_run_stats()
    img = cv2_imread(image_path)
    items = ocr_page_structure(image_path, img, profile=profile) if img is not None else []
    return image_path, items, dict(RUN_STATS)


def split_tail_answer_pages(image_paths, announce=False):
    """从一组按页序排列的图片中识别末尾连续的答案页。

    只从后 1/3 开始寻找页面顶部的答案标题，避免把题干中的“答案”误判为答案页。
    返回 (题目页, 自动识别出的答案页)。无法识别时保留全部图片为题目页。
    """
    if not image_paths:
        return [], []

    scan_start = max(0, len(image_paths) * 2 // 3)
    scan_total = max(1, len(image_paths) - scan_start)
    emit_progress('answer_scan', 0, scan_total, '预扫描末尾答案页')
    for index in range(scan_start, len(image_paths)):
        items = ocr_image(image_paths[index])
        emit_progress(
            'answer_scan', index - scan_start + 1, scan_total,
            f'预扫描答案页 {index - scan_start + 1}/{scan_total}',
        )
        for item in items[:15]:
            text = re.sub(r'\s+', '', item['text'].strip())
            if not text:
                continue
            if any(keyword.replace(' ', '') in text and len(text) <= len(keyword.replace(' ', '')) + 10
                   for keyword in ANSWER_TITLE_KEYWORDS):
                if announce:
                    print(f"  自动发现答案页: 末尾{len(image_paths) - index}张 (第{index + 1}张起)")
                return image_paths[:index], image_paths[index:]
    return image_paths, []


def rebase_ocr_items(ocr_items, x_offset=0, y_offset=0):
    """将页面坐标中的 OCR 框转换为裁剪图的局部坐标。"""
    rebased = []
    for item in ocr_items:
        copied = dict(item)
        copied['bbox'] = [[point[0] - x_offset, point[1] - y_offset]
                          for point in item['bbox']]
        rebased.append(copied)
    return rebased


def ocr_items_for_block(ocr_items, block):
    """严格按 X/Y 中心点选取当前栏和当前大题块的 OCR 项。"""
    x0 = block.get('_x_start', 0)
    y0 = block.get('_y_start', 0)
    height, width = block['crop_img'].shape[:2]
    selected = []
    for item in ocr_items:
        center_x = (
            min(point[0] for point in item['bbox'])
            + max(point[0] for point in item['bbox'])
        ) / 2
        center_y = (
            min(point[1] for point in item['bbox'])
            + max(point[1] for point in item['bbox'])
        ) / 2
        if x0 <= center_x < x0 + width and y0 <= center_y < y0 + height:
            selected.append(item)
    return rebase_ocr_items(selected, x_offset=x0, y_offset=y0)


def _estimate_skew_angle(gray):
    """从长的近水平线估计扫描倾斜角；证据不足时不旋转。"""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(80, gray.shape[1] // 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=min_len, maxLineGap=12)
    if lines is None or lines.size == 0:
        return 0.0

    angles = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 4.0:
            angles.append(angle)
    if len(angles) < 3 or np.std(angles) > 0.6:
        return 0.0
    angle = float(np.median(angles))
    return angle if abs(angle) >= 0.35 else 0.0


def estimate_page_deskew_angle(img):
    """只分析一次整页倾斜角，供该页的所有题图复用。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    analysis_scale = min(1.0, DESKEW_ANALYSIS_MAX_SIDE / max(height, width))
    if analysis_scale < 1.0:
        analysis_gray = cv2.resize(
            gray,
            (max(1, int(width * analysis_scale)), max(1, int(height * analysis_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis_gray = gray
    return _estimate_skew_angle(analysis_gray)


def _deskew_image(img, angle=None):
    """仅在检测到一致的轻微倾斜时纠正，避免误旋转题图。"""
    if angle is None:
        angle = estimate_page_deskew_angle(img)
    if not angle:
        return img, 0.0
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    corrected = cv2.warpAffine(
        img, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return corrected, angle


def _clean_sparse_edge_band(
    mask, edge_col_density, first_ink, last_ink, width, height, right_edge=False,
):
    """Zero out a sparse seal-line band at either edge of the mask."""
    band_span = last_ink - first_ink + 1
    if band_span < max(14, int(width * 0.025)):
        return
    band_density_slice = edge_col_density[first_ink:last_ink + 1]
    band_mean = float(np.mean(band_density_slice))
    zero_col_ratio = (
        float(np.count_nonzero(band_density_slice == 0)) / band_span
    )
    # 密封线文字竖排：平均列密度低，且列间有大量零墨迹间隙。
    if not (
        band_mean <= max(18, int(height * 0.38))
        and zero_col_ratio >= 0.22
    ):
        return
    if right_edge:
        # 检查 band 左侧是否有清晰空白沟（正文在 band 的左边）。
        gap_end = first_ink
        gap_start = max(0, gap_end - max(12, int(width * 0.035)))
        gap_slice = edge_col_density[gap_start:gap_end]
        if len(gap_slice) >= 3 and int(np.max(gap_slice)) < max(3, int(height * 0.04)):
            mask[:, first_ink:] = 0
    else:
        gap_start = last_ink + 1
        gap_end = min(width, gap_start + max(12, int(width * 0.035)))
        gap_slice = edge_col_density[gap_start:gap_end]
        if len(gap_slice) >= 3 and int(np.max(gap_slice)) < max(3, int(height * 0.04)):
            mask[:, :last_ink + 1] = 0


def _content_mask(img):
    """提取文本/图形墨迹，并剔除页边长条、装订虚线等无关边缘元素。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = mask.shape
    # 虚线有时不会在连通域中连成一条线；对页边”跨越大量行”的窄列直接排除。
    edge_width = max(1, int(width * 0.08))
    edge_col_density = np.count_nonzero(mask, axis=0)
    for x in list(range(edge_width)) + list(range(max(edge_width, width - edge_width), width)):
        if edge_col_density[x] >= height * 0.12:
            mask[:, x] = 0
    # 密封线由竖排文字 + 虚/实线组成，单列墨迹往往低于 12% 阈值因而
    # 逃过上面的逐列清理。对边缘区内”稀疏但连片”的墨迹带做整体切除，
    # 只在该带右侧存在清晰空白沟时生效，避免误伤紧贴边缘的正文题号。
    edge_has_ink = edge_col_density[:edge_width] >= max(2, int(height * 0.02))
    first_ink, last_ink = None, None
    for x in range(edge_width):
        if edge_has_ink[x]:
            if first_ink is None:
                first_ink = x
            last_ink = x
    if first_ink is not None and last_ink is not None:
        _clean_sparse_edge_band(
            mask, edge_col_density, first_ink, last_ink, width, height,
        )
    # 右边缘同样可能存在密封线（右半页的外侧装订边），对称处理。
    right_edge_start = max(edge_width, width - edge_width)
    first_ink_r, last_ink_r = None, None
    for x in range(right_edge_start, width):
        if edge_col_density[x] >= max(2, int(height * 0.02)):
            if first_ink_r is None:
                first_ink_r = x
            last_ink_r = x
    if first_ink_r is not None and last_ink_r is not None:
        _clean_sparse_edge_band(
            mask, edge_col_density, first_ink_r, last_ink_r, width, height,
            right_edge=True,
        )
    # 先把虚线沿纵向连接起来，才能将”密封线”识别为整条边缘元素。
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 20)))
    linked_vertical = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel)
    line_count, _, line_stats, _ = cv2.connectedComponentsWithStats(linked_vertical, connectivity=8)
    edge_line_ranges = []
    for label in range(1, line_count):
        x, y, comp_w, comp_h, _ = line_stats[label]
        is_edge = x <= width * 0.08 or x + comp_w >= width * 0.92
        is_long_thin = comp_h >= height * 0.45 and comp_w <= max(10, width * 0.03)
        if is_edge and is_long_thin:
            edge_line_ranges.append((x, y, x + comp_w, y + comp_h))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Build a label lookup once and apply it to the whole image.  Assigning
    # ``cleaned[labels == label]`` for every glyph scans the full page once per
    # connected component; dense textbook pages can contain thousands of
    # components and spend minutes in this otherwise simple cleanup step.
    kept_labels = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        x, y, comp_w, comp_h, area = stats[label]
        # 这里只生成“内容边界掩码”，不会修改原题图。过滤极小孤立点可避免
        # 4像素扫描脏点把整张题图的上下边界拖出数百像素空白。
        if area < 8:
            continue
        # 靠边、细而贯穿半页以上的成分通常是密封线、装订线或扫描边线。
        is_edge = x <= width * 0.08 or x + comp_w >= width * 0.92
        is_long_thin = comp_h >= height * 0.45 and comp_w <= max(8, width * 0.025)
        if is_edge and is_long_thin:
            continue
        if any(x >= line_x0 and x + comp_w <= line_x1 and y >= line_y0 and y + comp_h <= line_y1
               for line_x0, line_y0, line_x1, line_y1 in edge_line_ranges):
            continue
        kept_labels[label] = 255
    return kept_labels[labels]


def _remove_detached_footer(mask):
    """去掉被大段留白隔开的页脚/页码装饰带，不动正常相邻题目内容。"""
    height, width = mask.shape
    if height < 140:
        return mask
    row_density = np.count_nonzero(mask, axis=1)
    active = (row_density >= max(2, int(width * 0.001))).astype(np.uint8)[:, None] * 255
    # 合并同一内容段内的普通行距，只保留真正的大空档作为段间边界。
    close_height = max(9, int(height * 0.045))
    closed = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_height)),
    )[:, 0]
    indices = np.flatnonzero(closed)
    if not len(indices):
        return mask
    segments = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value > previous + 1:
            segments.append((start, previous + 1))
            start = value
        previous = value
    segments.append((start, previous + 1))
    if len(segments) < 2:
        return mask

    footer_start, footer_end = segments[-1]
    previous_end = segments[-2][1]
    footer_ink = np.flatnonzero(np.count_nonzero(mask[footer_start:footer_end], axis=0))
    footer_span = (
        int(footer_ink[-1] - footer_ink[0] + 1) / width
        if len(footer_ink) else 0.0
    )
    footer_center = (
        float(footer_ink[-1] + footer_ink[0]) / (2 * width)
        if len(footer_ink) else 0.0
    )
    footer_region = mask[footer_start:footer_end]
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        footer_region, connectivity=8,
    )
    meaningful_components = [
        stats for stats in component_stats[1:]
        if stats[cv2.CC_STAT_AREA] >= 5
    ]
    footer_area = int(np.count_nonzero(footer_region))
    # 末段墨迹可能在同一行带混入页边拾噪（如左缘“1”、右缘“2”这类孤立小墨点），
    # 把整段宽度/中心测偏。以面积最大的连通块为锚，只把水平上相邻的块并入
    # “核心块组”，据此重测页脚形状；真·正文宽行（如长公式、横线）仍会保持大跨度而落选。
    proximity = max(4, int(width * 0.18))
    anchor = (
        max(meaningful_components, key=lambda s: s[cv2.CC_STAT_AREA])
        if meaningful_components else None
    )
    core_components = []
    if anchor is not None:
        anchor_left = anchor[cv2.CC_STAT_LEFT]
        anchor_right = anchor[cv2.CC_STAT_LEFT] + anchor[cv2.CC_STAT_WIDTH]
        core_components = [
            stats for stats in meaningful_components
            if max(
                stats[cv2.CC_STAT_LEFT] - anchor_right,
                anchor_left - (stats[cv2.CC_STAT_LEFT] + stats[cv2.CC_STAT_WIDTH]),
                0,
            ) <= proximity
        ]
    if core_components:
        core_left = min(stats[cv2.CC_STAT_LEFT] for stats in core_components)
        core_right = max(
            stats[cv2.CC_STAT_LEFT] + stats[cv2.CC_STAT_WIDTH]
            for stats in core_components
        )
        core_span = (core_right - core_left + 1) / width
        core_center = (core_right + core_left) / (2 * width)
        core_strokes = sum(
            1 for stats in core_components
            if (
                stats[cv2.CC_STAT_WIDTH] >= max(12, int(width * 0.012))
                and stats[cv2.CC_STAT_WIDTH] >= stats[cv2.CC_STAT_HEIGHT] * 4
            )
        )
    else:
        core_span = 0.0
        core_center = 0.0
        core_strokes = 0
    # 常见页码是“— 1 —”：核心块组居中、很窄，由一个数字和两条短横线组成。
    # 它可能位于原始裁块中部（页码下面还有整段扫描白边），不能只看相对页底位置。
    looks_like_page_number = (
        core_components
        and 0.02 <= core_span <= 0.28
        and 0.32 <= core_center <= 0.68
        and 2 <= len(core_components) <= 7
        and core_strokes >= 2
        and footer_end - footer_start <= max(60, int(height * 0.24))
    )
    # Some books print the footer as a very small coloured “- 2 -”.  After
    # binarisation its short dashes are only a few pixels wide, so the normal
    # horizontal-stroke test above deliberately does not see them.  Its tiny,
    # centred, three-part footprint is still strong evidence when isolated
    # beneath real question content.
    compact_page_marker = (
        core_components
        and 0.015 <= core_span <= 0.12
        and 0.38 <= core_center <= 0.62
        and 2 <= len(core_components) <= 5
        and footer_end - footer_start <= max(32, int(height * 0.12))
        and footer_area <= max(360, int(width * 0.40))
    )
    # 页脚文字有时紧贴在最后一行正文之下（隔 ~20~30px），常见阈值 `0.12*height`
    # 会漏掉；只要页脚形状成立、下方还有一段明显留白直达裁块底，就按“终段页脚”处置。
    footer_tail = height - footer_end
    terminal_footer = (
        (looks_like_page_number or compact_page_marker)
        and footer_tail >= max(80, int(height * 0.20))
        and footer_start >= int(height * 0.35)
    )
    detached = footer_start - previous_end >= max(30, int(height * 0.12))
    if compact_page_marker:
        detached = footer_start - previous_end >= max(18, int(height * 0.06))
    if terminal_footer:
        detached = footer_start - previous_end >= max(12, int(height * 0.02))
    looks_marginal = (
        footer_start >= height * 0.62
        # 扫描页通常在页脚装饰带之后还保留一截白边，不能要求墨迹贴住图底。
        and footer_end >= height * 0.75
        and footer_end - footer_start <= height * 0.28
        and footer_span >= 0.72
    ) or looks_like_page_number or compact_page_marker or terminal_footer
    if detached and looks_marginal:
        cleaned = mask.copy()
        # 抗锯齿会在分段起点上方残留一两行淡墨；回退数行一起清掉，
        # 否则极小残迹仍会把内容下边界拖到页码位置。
        erase_from = max(previous_end, footer_start - 8)
        cleaned[erase_from:, :] = 0
        return cleaned
    return mask


def _left_aligned_body_near_sidebar_candidate(mask, boundary_x):
    """Whether a candidate line has genuine horizontal content to its left."""
    height, width = mask.shape
    left_body_width = max(0, int(boundary_x) - max(6, int(width * 0.006)))
    top_height = min(height, max(90, int(height * 0.16)))
    if left_body_width < 40 or top_height <= 0:
        return False
    top_row_ink = np.count_nonzero(mask[:top_height, :left_body_width], axis=1)
    dense_rows = np.count_nonzero(
        top_row_ink >= max(20, int(left_body_width * 0.09))
    )
    return dense_rows >= max(12, int(top_height * 0.10))


def _sidebar_has_clear_inner_gap(mask, boundary_end, cut_x):
    """Whether a suspected marginal line is isolated from the page body.

    Repeated first strokes of left-aligned Chinese/English paragraphs can form
    many short vertical runs at the same x coordinate.  Geometrically those
    runs resemble a dashed seal line, but they remain joined to glyph ink on
    their right.  A real binding/seal strip normally has a blank gutter before
    the question body.  Refuse a destructive trim when that gutter is absent.
    """
    height, width = mask.shape
    start = max(0, min(width, int(boundary_end) + 1))
    stop = max(start, min(width, int(cut_x)))
    if stop - start < 6:
        return False
    gap = mask[:, start:stop]
    if gap.size == 0:
        return False
    ink = int(np.count_nonzero(gap))
    ink_density = ink / max(1, gap.size)
    inked_rows = int(np.count_nonzero(np.count_nonzero(gap, axis=1) > 0))
    row_ratio = inked_rows / max(1, height)
    return ink_density <= 0.035 and row_ratio <= 0.18


def _remove_marginal_sidebar(mask):
    """从题图内容掩码中排除左侧装订/密封栏，返回 ``(mask, cut_x)``。

    这类侧栏不是单一的细线：通常由竖排的“密封线、学校、班级”等
    多个框和两三条平行竖线组成。它被误认为题目内容后，普通的去白边
    会把整条侧栏保留下来。只删除有充分版式证据的左侧窄栏，避免误伤
    题内表格的普通竖线。
    """
    height, width = mask.shape
    if width < 300 or height < 80:
        return mask, None

    linked = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 20))),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
    min_x = int(width * 0.08)
    max_x = int(width * 0.24)
    min_height = int(height * 0.55)
    min_group_width = max(14, int(width * 0.008))
    max_group_width = max(24, int(width * 0.045))
    candidates = []
    for x, y, component_width, component_height, area in stats[1:]:
        if not (
            min_x <= x <= max_x
            and component_height >= min_height
            and min_group_width <= component_width <= max_group_width
            and area >= component_height * 1.5
        ):
            continue
        # 侧栏分界线右侧必须仍有足够大的一片正文；否则很可能只是题内
        # 表格的左边框，不能删除。
        # 给侧栏最右一条虚线/竖排文字留出安全间隔；正文通常在分界线
        # 右侧留有更大的空白，避免清理后仍残留“密/封”等单字。
        cut_x = min(width, int(x + component_width + max(12, width * 0.020)))
        # 大题标题或题内表格有时恰好从裁图左边开始，表格的长竖线会伪装
        # 成密封栏边界。若候选线左侧顶部已经有成行正文，说明这里是有效
        # 内容，不能整列清掉。真正的密封栏顶部通常只有竖线/标签框，不会
        # 形成连续的横向文字笔画。
        if _left_aligned_body_near_sidebar_candidate(mask, x):
            continue
        if not _sidebar_has_clear_inner_gap(mask, x + component_width - 1, cut_x):
            continue
        right_ink = np.count_nonzero(mask[:, cut_x:])
        if right_ink < height * max(5, int(width * 0.012)):
            continue
        candidates.append((cut_x, component_height, component_width, area))
    if candidates:
        # 若有两组侧栏线，最右一组才是“侧栏 -> 正文”的边界。
        cut_x, _, _, _ = max(candidates, key=lambda value: value[0])
        cleaned = mask.copy()
        cleaned[:, :cut_x] = 0
        return cleaned, cut_x

    # 真实试卷的密封栏内边界有时只是一条 3～6 px 细线，它不满足
    # 上面“宽组件”的最小宽度。只在整页/整栏高度下启用这个保守分支：
    #   1) 边界线贯穿至少 72% 页高；
    #   2) 它左侧还有一条相隔较远的长线/竖排标签，证明这是独立密封栏；
    #   3) 右侧有明确空白沟，再往右有大量正文。
    # 三项缺一不裁，避免把表格、坐标轴或正文首笔误当密封线。
    thin_boundary_min_height = int(height * 0.72)
    thin_boundary_max_width = max(8, int(width * 0.008))
    thin_boundaries = []
    for x, y, component_width, component_height, area in stats[1:]:
        if not (
            min_x <= x <= max_x
            and component_height >= thin_boundary_min_height
            and component_width <= thin_boundary_max_width
            and area >= component_height * 1.15
        ):
            continue
        cut_x = min(width, int(x + component_width + max(12, width * 0.020)))
        if _left_aligned_body_near_sidebar_candidate(mask, x):
            continue
        if not _sidebar_has_clear_inner_gap(mask, x + component_width - 1, cut_x):
            continue
        outer_evidence = any(
            other_x < x - max(14, int(width * 0.012))
            and other_x >= max(0, int(width * 0.015))
            and other_height >= int(height * 0.32)
            and other_width <= max(30, int(width * 0.055))
            and other_area >= other_height * 1.05
            for other_x, _other_y, other_width, other_height, other_area in stats[1:]
        )
        if not outer_evidence:
            continue
        right_ink = np.count_nonzero(mask[:, cut_x:])
        if right_ink < height * max(5, int(width * 0.012)):
            continue
        thin_boundaries.append((cut_x, component_height, area))
    if thin_boundaries:
        cut_x, _, _ = max(thin_boundaries, key=lambda value: value[0])
        cleaned = mask.copy()
        cleaned[:, :cut_x] = 0
        return cleaned, cut_x

    # 一些试卷的小题高度只有 150～250 像素，密封栏会被小题切线截断：
    # 标签框无法连成前面的“长组件”，只剩一条稀疏的贯穿虚线。不能仅按
    # 左边有竖线就删除（题内坐标轴会误伤），因此要求同时满足：位于左侧
    # 边缘、细窄、纵向跨越大部分题图、由多段稀疏墨迹组成。
    short_left = max(8, int(width * 0.025))
    short_right = max(short_left + 1, int(width * 0.15))
    column_hits = []
    for x in range(short_left, min(width, short_right)):
        ys = np.flatnonzero(mask[:, x])
        if len(ys) < max(4, int(height * 0.10)):
            continue
        span = int(ys[-1] - ys[0] + 1)
        # 连续实线（题内坐标轴、表格线）不属于密封虚线。
        # 密封虚线纵向跨度大但每列墨迹稀疏；用密度（墨迹像素 / 跨度）
        # 区分实线和虚线，避免把全长密封线误判为坐标轴。
        span_density = len(ys) / max(1, span)
        if span < int(height * 0.32) or span_density > 0.55:
            continue
        column_hits.append(x)

    if column_hits:
        groups = []
        group_start = previous = column_hits[0]
        for x in column_hits[1:]:
            # 同一条虚线有轻微抗锯齿和断列，容许很小间隔。
            if x <= previous + 3:
                previous = x
                continue
            groups.append((group_start, previous))
            group_start = previous = x
        groups.append((group_start, previous))
        for group_start, group_end in reversed(groups):
            group_width = group_end - group_start + 1
            if group_width > max(10, int(width * 0.035)):
                continue
            band_rows = np.count_nonzero(mask[:, group_start:group_end + 1], axis=1) > 0
            runs = 0
            active = False
            for has_ink in band_rows:
                if has_ink and not active:
                    runs += 1
                active = bool(has_ink)
            # 密封虚线至少有三个分段；正文的单个竖笔、坐标轴均不会满足。
            if runs < 3:
                continue
            cut_x = min(width, group_end + 1 + max(12, int(width * 0.020)))
            if _left_aligned_body_near_sidebar_candidate(mask, group_start):
                continue
            if not _sidebar_has_clear_inner_gap(mask, group_end, cut_x):
                continue
            if np.count_nonzero(mask[:, cut_x:]) >= height * max(5, int(width * 0.012)):
                cleaned = mask.copy()
                cleaned[:, :cut_x] = 0
                return cleaned, cut_x

    # 一道短小题目只截到侧栏中部时，原来的多个标签框不再连成“宽组件”，
    # 但仍会留下至少三条等高平行细竖线。只在很矮的题图中启用这个更窄的
    # 规则，避免把正常题内表格误当侧栏。
    if height <= 360:
        thin_lines = []
        for x, y, component_width, component_height, area in stats[1:]:
            if (
                min_x <= x <= max_x
                and component_height >= int(height * 0.65)
                and y <= int(height * 0.20)
                and y + component_height >= int(height * 0.75)
                and component_width <= max(8, int(width * 0.008))
                and area >= component_height * 1.2
            ):
                thin_lines.append((int(x), int(component_width)))
        if len(thin_lines) >= 3:
            thin_lines.sort()
            first_x = thin_lines[0][0]
            last_x, last_width = thin_lines[-1]
            spread = last_x + last_width - first_x
            if int(width * 0.04) <= spread <= int(width * 0.17):
                cut_x = min(width, last_x + last_width + max(12, int(width * 0.020)))
                if _left_aligned_body_near_sidebar_candidate(mask, first_x):
                    return mask, None
                if not _sidebar_has_clear_inner_gap(mask, last_x + last_width - 1, cut_x):
                    return mask, None
                if np.count_nonzero(mask[:, cut_x:]) >= height * max(5, int(width * 0.012)):
                    cleaned = mask.copy()
                    cleaned[:, :cut_x] = 0
                    return cleaned, cut_x
    return mask, None


def _remove_marginal_sidebars(mask):
    """Remove proven binding/seal sidebars from either edge of a crop."""
    height, width = mask.shape
    # A question shorter than two lines of body text risks having its
    # leading "2. 每个…" mistaken for a truncated sidebar.  Still attempt
    # the detection but with a more conservative width cap so that genuine
    # binding/seal strips (which are much narrower than question text) can
    # be removed even from one-line questions.
    if height <= 120 and width / max(1, height) >= 12:
        return mask, None, None
    cleaned, left_cut_x = _remove_marginal_sidebar(mask)
    # Right-page scans commonly place the same seal strip on the outer right
    # edge.  Mirror the conservative left-edge detector rather than adding a
    # second, divergent set of geometric thresholds.
    mirrored, mirrored_cut_x = _remove_marginal_sidebar(np.fliplr(cleaned).copy())
    right_cut_x = None
    if mirrored_cut_x is not None:
        right_cut_x = cleaned.shape[1] - int(mirrored_cut_x)
        cleaned = np.fliplr(mirrored).copy()
    return cleaned, left_cut_x, right_cut_x


def _protect_sidebar_exclusions_with_ocr(exclusions, ocr_items, page_width):
    """Pull a sidebar limit back when it would cross proven horizontal text."""
    if not isinstance(exclusions, dict) or not ocr_items:
        return exclusions
    left = exclusions.get('left')
    right = exclusions.get('right')
    padding = max(4, int(page_width * 0.003))
    for item in ocr_items:
        bounds = _ocr_item_bounds(item)
        if not bounds:
            continue
        x0, _y0, x1, y1 = bounds
        box_height = max(1.0, y1 - _y0)
        box_width = max(0.0, x1 - x0)
        text = re.sub(r'\s+', '', str(item.get('text', '')))
        # Sidebars are predominantly narrow/vertical labels.  Only a credible
        # horizontal text line is strong enough to override geometric removal.
        if len(text) < 2 or box_width < max(55.0, box_height * 1.8):
            continue
        if left is not None and x0 < float(left) < x1:
            left = min(int(left), max(0, int(np.floor(x0)) - padding))
        if right is not None and x0 < float(right) < x1:
            right = max(int(right), min(int(page_width), int(np.ceil(x1)) + padding))
    exclusions['left'] = left
    exclusions['right'] = right
    return exclusions


def detect_page_sidebar_exclusions(page_img, major_blocks, ocr_items=None):
    """Find *outer* page sidebar limits once, before individual question cuts.

    Per-question detection is necessarily weak for a short question: it may
    contain only one label from a seal strip.  A full column sees the complete
    long vertical strip and is far more reliable.  Only the outside edge of a
    page/spread is accepted, so an ordinary centre gutter or inner table border
    can never become a forbidden area.
    """
    if page_img is None or page_img.size == 0 or not major_blocks:
        return {'left': None, 'right': None}
    height, width = page_img.shape[:2]
    bounds = {
        tuple(block.get('_column_bounds', [0, width]))
        for block in major_blocks
    }
    valid_bounds = sorted(
        (max(0, int(x0)), min(width, int(x1)))
        for x0, x1 in bounds
        if int(x1) - int(x0) >= 300
    )
    if not valid_bounds:
        valid_bounds = [(0, width)]
    left_outer = min(x0 for x0, _ in valid_bounds)
    right_outer = max(x1 for _, x1 in valid_bounds)
    exclusions = {'left': None, 'right': None}
    for x0, x1 in valid_bounds:
        column = page_img[:, x0:x1]
        mask = _content_mask(column)
        _, left_cut, right_cut = _remove_marginal_sidebars(mask)
        if x0 == left_outer and left_cut is not None:
            exclusions['left'] = x0 + int(left_cut)
        if x1 == right_outer and right_cut is not None:
            exclusions['right'] = x0 + int(right_cut)
    return _protect_sidebar_exclusions_with_ocr(exclusions, ocr_items, width)


def trim_cuts_to_page_sidebar_exclusions(minor_cuts, page_img, exclusions):
    """Hard-trim source boxes that cross a confirmed page-level sidebar."""
    if page_img is None or page_img.size == 0 or not minor_cuts:
        return minor_cuts
    height, width = page_img.shape[:2]
    left = exclusions.get('left') if isinstance(exclusions, dict) else None
    right = exclusions.get('right') if isinstance(exclusions, dict) else None
    for minor in minor_cuts:
        box = minor.get('_source_box', [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in box)
        new_x0 = max(x0, int(left)) if left is not None and x0 < int(left) < x1 else x0
        new_x1 = min(x1, int(right)) if right is not None and x0 < int(right) < x1 else x1
        x0, x1 = max(0, new_x0), min(width, new_x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 <= x0 or y1 <= y0 or (x0, y0, x1, y1) == tuple(box):
            continue
        minor['_source_box'] = [x0, y0, x1, y1]
        minor['crop_img'] = page_img[y0:y1, x0:x1].copy()
        minor['_page_sidebar_trimmed'] = True
        segments = minor.get('_source_segments')
        if isinstance(segments, list) and len(segments) == 1:
            segments[0]['question_box'] = [x0, y0, x1, y1]
    return minor_cuts


def trim_cuts_to_column_bounds(minor_cuts, page_img, column_bounds):
    """Keep every emitted question box inside its verified reading column.

    Rule-based cuts already start with these bounds, but an optional detector
    refinement can enlarge a box afterwards.  In a multi-column page that
    enlargement must never cross a confirmed gutter and absorb a neighbour.
    """
    if page_img is None or page_img.size == 0 or not minor_cuts:
        return minor_cuts
    if not isinstance(column_bounds, (list, tuple)) or len(column_bounds) != 2:
        return minor_cuts
    height, width = page_img.shape[:2]
    try:
        column_x0 = max(0, min(width, int(column_bounds[0])))
        column_x1 = max(0, min(width, int(column_bounds[1])))
    except (TypeError, ValueError):
        return minor_cuts
    if column_x1 <= column_x0:
        return minor_cuts
    for minor in minor_cuts:
        # A shared-context crop is already composed from several segments and
        # no longer equals the raw ``_source_box`` rectangle on this page;
        # re-cropping it flat would silently strip the attached preamble.
        if minor.get('_shared_context_attached'):
            continue
        box = minor.get('_source_box', [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in box)
        bounded_x0 = max(column_x0, min(width, x0))
        bounded_x1 = min(column_x1, max(0, x1))
        bounded_y0 = max(0, min(height, y0))
        bounded_y1 = max(0, min(height, y1))
        if bounded_x1 <= bounded_x0 or bounded_y1 <= bounded_y0:
            continue
        if (bounded_x0, bounded_y0, bounded_x1, bounded_y1) == (x0, y0, x1, y1):
            continue
        minor['_source_box'] = [bounded_x0, bounded_y0, bounded_x1, bounded_y1]
        minor['crop_img'] = page_img[bounded_y0:bounded_y1, bounded_x0:bounded_x1].copy()
        minor['_column_bounds_trimmed'] = True
        segments = minor.get('_source_segments')
        if isinstance(segments, list) and len(segments) == 1:
            segments[0]['question_box'] = list(minor['_source_box'])
    return minor_cuts


def trim_cuts_to_detached_page_footer(minor_cuts, page_img, bottom_tolerance=2):
    """Remove a proven detached footer from cuts that reach the page bottom.

    The production crop normalizer already strips an isolated page number, but
    the source box shown in the live/manual review used to retain it.  That
    made the last question in a column look much larger than its actual
    content.  Only act when the existing footer detector positively removes a
    detached segment, so a legitimate question that simply finishes near the
    bottom is never shortened by this step.
    """
    if page_img is None or page_img.size == 0 or not minor_cuts:
        return minor_cuts
    page_height, page_width = page_img.shape[:2]
    for minor in minor_cuts:
        box = minor.get('_source_box', [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in box)
        if y1 < page_height - max(0, int(bottom_tolerance)):
            continue
        x0, x1 = max(0, x0), min(page_width, x1)
        y0, y1 = max(0, y0), min(page_height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = _content_mask(page_img[y0:y1, x0:x1])
        without_footer = _remove_detached_footer(mask)
        # _remove_detached_footer returns the original mask when its evidence
        # is insufficient.  Do not turn ordinary lower whitespace into a hard
        # crop boundary.
        if without_footer is mask:
            continue
        row_density = np.count_nonzero(without_footer, axis=1)
        content_rows = np.flatnonzero(
            row_density >= max(2, int((x1 - x0) * 0.0015))
        )
        if not len(content_rows):
            continue
        safe_padding = max(6, int((y1 - y0) * 0.02))
        trimmed_y1 = min(y1, y0 + int(content_rows[-1]) + 1 + safe_padding)
        if trimmed_y1 >= y1 - max(8, safe_padding):
            continue
        minor['_source_box'] = [x0, y0, x1, trimmed_y1]
        minor['crop_img'] = page_img[y0:trimmed_y1, x0:x1].copy()
        minor['_page_footer_trimmed'] = True
        segments = minor.get('_source_segments')
        if isinstance(segments, list) and len(segments) == 1:
            segments[0]['question_box'] = [x0, y0, x1, trimmed_y1]
    return minor_cuts


def normalize_question_crop(crop_img, deskew_angle=None):
    """纠正轻微倾斜并按内容居中裁剪，确保上下留白近似一致。

    返回 (图片, 信息)。内容无法可靠定位时保留原图并记录质量提示。
    """
    if crop_img is None or crop_img.size == 0:
        return crop_img, {'quality_warning': '空裁剪图', 'skew_angle': 0.0}

    if ENABLE_DESKEW:
        corrected, angle = _deskew_image(crop_img, angle=deskew_angle)
    else:
        corrected, angle = crop_img, 0.0
    mask = _remove_detached_footer(_content_mask(corrected))
    mask, sidebar_cut_x, sidebar_cut_right_x = _remove_marginal_sidebars(mask)
    height, width = mask.shape
    row_density = np.count_nonzero(mask, axis=1)
    col_density = np.count_nonzero(mask, axis=0)
    # 边界定位不能被页底两三个像素的扫描脏点牵走，否则会留下巨大单侧空白。
    # 阈值仍远低于最细文字笔画，不影响题干、图形和标点。
    row_threshold = max(4, int(width * 0.0025))
    col_threshold = max(4, int(height * 0.002))
    ys = np.flatnonzero(row_density >= row_threshold)
    xs = np.flatnonzero(col_density >= col_threshold)
    if len(ys) == 0 or len(xs) == 0:
        return corrected, {'quality_warning': '未检测到有效题目内容', 'skew_angle': angle}

    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    hard_x0 = None
    hard_x1 = None
    # 密封线竖排文字/虚线位于页边，与正文之间有宽阔空白。
    # 通过列密度的间隙检测边缘密封条：若 xs 中存在 >15% 图宽的
    # 大跨度间隙，且间隙外侧的窄条贴边，则将其裁掉。
    if len(xs) >= 2:
        xs_diffs = np.diff(xs)
        gap_threshold = max(20, int(width * 0.03))
        large_gaps = np.flatnonzero(xs_diffs >= gap_threshold)
        if len(large_gaps) > 0:
            # 右边缘：最后一个大间隙之后的内容如果是靠右的窄条 → 密封线
            last_gap = large_gaps[-1]
            right_seg_start = int(xs[last_gap + 1])
            right_seg_end = int(xs[-1])
            right_seg_width = right_seg_end - right_seg_start
            if (
                right_seg_width < int(width * 0.12)
                and right_seg_end >= int(width * 0.90)
            ):
                x1 = int(xs[last_gap]) + 1
                hard_x1 = x1
            # 左边缘：第一个大间隙之前的内容如果是靠左的窄条 → 密封线
            first_gap = large_gaps[0]
            left_seg_start = int(xs[0])
            left_seg_end = int(xs[first_gap])
            left_seg_width = left_seg_end - left_seg_start
            if (
                left_seg_width < int(width * 0.08)
                and left_seg_start <= int(width * 0.03)
            ):
                x0 = int(xs[first_gap + 1])
                hard_x0 = x0
    # 墨迹阈值会偶尔漏掉浅色笔画（尤其是题号 2、3 的最左侧弧线）。
    # 在内容框四周保留小的原图安全带，确保“去白边”不会切到字符本身。
    x_guard = max(4, min(48, int((x1 - x0) * 0.035)))
    y_guard = max(3, min(20, int((y1 - y0) * 0.06)))
    x0 = max(0, x0 - x_guard)
    x1 = min(width, x1 + x_guard)
    # A confirmed seal/sidebar boundary is a hard limit.  The ordinary guard
    # band protects faint text strokes, but must never grow back into a strip
    # that has already been classified as a page margin.
    if hard_x0 is not None:
        x0 = max(x0, hard_x0)
    if hard_x1 is not None:
        x1 = min(x1, hard_x1)
    y0, y1 = max(0, y0 - y_guard), min(height, y1 + y_guard)
    content_height = y1 - y0
    content_width = x1 - x0
    # 先取纯内容，再补等量白边；即使原切线紧贴文字，也不会出现“上紧下松”。
    y_margin = min(max(18, int(content_height * 0.10)), max(18, height // 4))
    x_margin = min(max(18, int(content_width * 0.035)), max(18, width // 5))
    content = corrected[y0:y1, x0:x1]
    normalized = cv2.copyMakeBorder(
        content,
        y_margin, y_margin, x_margin, x_margin,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    info = {
        'skew_angle': angle,
        'content_bounds': (x0, y0, x1, y1),
        'balanced_margin': (x_margin, y_margin),
    }
    if sidebar_cut_x is not None:
        info['marginal_sidebar_cut_x'] = sidebar_cut_x
    if sidebar_cut_right_x is not None:
        info['marginal_sidebar_cut_right_x'] = sidebar_cut_right_x
    normalized_height, normalized_width = normalized.shape[:2]
    if normalized_height < 32 or normalized_width < 80:
        info['quality_warning'] = '有效内容区域过小，保留归一化前题图'
        info['content_mask'] = mask
        return corrected, info
    if normalized_width / max(1, normalized_height) > 24 and normalized_height < 70:
        info['quality_warning'] = '裁剪过薄，保留归一化前题图'
        info['content_mask'] = mask
        return corrected, info
    # 质量检查复用同一次内容分析：掩码同步裁剪出内容区并补同样的白边，
    # 尺寸与 normalized 完全一致，crop_quality_issues 可直接复用，不再重算连通域。
    info['content_mask'] = cv2.copyMakeBorder(
        mask[y0:y1, x0:x1],
        y_margin, y_margin, x_margin, x_margin,
        cv2.BORDER_CONSTANT, value=0,
    )
    return normalized, info


def crop_quality_issues(img, content_mask=None):
    """交付前的纯视觉质量门槛，不阻断输出，只触发自动回退。"""
    if img is None or img.size == 0:
        return ['空图']
    height, width = img.shape[:2]
    issues = []
    if height < 32:
        issues.append('高度过小')
    if width < 80:
        issues.append('宽度过小')
    mask = content_mask
    if mask is None or mask.shape[:2] != img.shape[:2]:
        mask = _content_mask(img)
    ink = int(np.count_nonzero(mask))
    if ink < max(18, img.size // 5000):
        issues.append('有效内容过少')
    # 横向细条只有在几乎没有墨迹时才是碎片。密排填空题的一行题干本来就
    # 又宽又扁（2484×79 时宽高比 31），只看宽高比会把每一道正常小题都判成
    # 空白条，进而触发一次无意义的质量回退，把归一化裁切换成未归一化原图。
    if width / max(1, height) > 28 and height < 80 and ink < height * width * 0.02:
        issues.append('疑似空白细条')
    return issues


# ═══════════════════════════════════════════════════════════════
# 双栏检测
# ═══════════════════════════════════════════════════════════════
def detect_columns(img, ocr_items):
    """检测双栏，返回分栏X坐标或None"""
    h, w = img.shape[:2]
    if w < 400:
        return None
    
    x_density = np.zeros(w, dtype=np.int32)
    for item in ocr_items:
        bbox = item['bbox']
        x_min = int(max(0, min(p[0] for p in bbox)))
        x_max = int(min(w, max(p[0] for p in bbox)))
        if x_max > x_min:
            x_density[x_min:x_max] += 1
    
    mid_start = int(w * 0.35)
    mid_end = int(w * 0.65)
    if mid_end <= mid_start:
        return None
    
    mid_density = x_density[mid_start:mid_end]
    if len(mid_density) == 0:
        return None

    # 用滑动平均寻找“连续”的低密度区，而不是只取单个最小像素。
    # 这样中间有虚线/装订线时仍能识别出两侧题目，避免同图混入两栏内容。
    window = max(9, w // 100)
    smooth = np.convolve(mid_density, np.ones(window) / window, mode='same')
    nonzero = smooth[smooth > 0]
    low_threshold = max(1.0, float(np.median(nonzero)) * 0.12) if len(nonzero) else 1.0
    low = smooth <= low_threshold

    best_start = best_end = -1
    start = None
    for index, is_low in enumerate(low):
        if is_low and start is None:
            start = index
        elif not is_low and start is not None:
            if index - start > best_end - best_start:
                best_start, best_end = start, index
            start = None
    if start is not None and len(low) - start > best_end - best_start:
        best_start, best_end = start, len(low)
    if best_start >= 0:
        gap_width = best_end - best_start
        split_x = mid_start + (best_start + best_end) // 2
        left_count = sum(1 for it in ocr_items if max(p[0] for p in it['bbox']) <= split_x)
        right_count = sum(1 for it in ocr_items if min(p[0] for p in it['bbox']) >= split_x)
        if (
            gap_width >= max(8, w // 120)
            and left_count >= 2
            and right_count >= 2
            and w * 0.38 <= split_x <= w * 0.62
        ):
            return split_x

    # 暑假作业、双页扫描常用贯穿大半页的竖向虚线分栏。它不是“空白沟”，
    # 因此单靠密度最低点会漏检；先把虚线纵向连接，再寻找中部细长分隔带。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_u8 = (gray < 210).astype(np.uint8) * 255
    raw_col_density = np.count_nonzero(binary_u8, axis=0)
    dense_columns = raw_col_density >= h * 0.18
    dense_runs = []
    run_start = None
    for index, is_dense in enumerate(dense_columns):
        if is_dense and run_start is None:
            run_start = index
        elif not is_dense and run_start is not None:
            dense_runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        dense_runs.append((run_start, len(dense_columns)))
    centers = [
        (min(point[0] for point in item['bbox']) + max(point[0] for point in item['bbox'])) / 2
        for item in ocr_items
    ]
    strong_dividers = []
    for run_x0, run_x1 in dense_runs:
        candidate_x = run_x0 + int(np.argmax(raw_col_density[run_x0:run_x1]))
        if not (
            run_x1 - run_x0 <= max(14, w * 0.025)
            and raw_col_density[candidate_x] >= h * 0.25
            and w * 0.36 <= candidate_x <= w * 0.64
        ):
            continue
        left_count = sum(center < candidate_x for center in centers)
        right_count = sum(center >= candidate_x for center in centers)
        if left_count >= 3 and right_count >= 3:
            strong_dividers.append((abs(candidate_x - w / 2), candidate_x))
    if strong_dividers:
        return min(strong_dividers)[1]

    divider_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(25, h // 16)),
    )
    linked = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, divider_kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
    divider_candidates = []
    for label in range(1, count):
        x, y, comp_w, comp_h, _ = stats[label]
        candidate_x = x + comp_w // 2
        if not (
            h * 0.55 <= comp_h
            and comp_w <= max(12, w * 0.025)
            and w * 0.36 <= candidate_x <= w * 0.64
        ):
            continue
        left_count = sum(center < candidate_x for center in centers)
        right_count = sum(center >= candidate_x for center in centers)
        if left_count >= 3 and right_count >= 3:
            divider_candidates.append((abs(candidate_x - w / 2), candidate_x))
    if divider_candidates:
        return min(divider_candidates)[1]

    # OCR偶尔会把左右栏同一高度的标题合成一个超长文本框。
    # 此时改用图像中心的连续低墨迹带，并用OCR中心点验证两侧确有内容。
    binary = gray < 210
    density = np.count_nonzero(binary, axis=0).astype(np.float32) / max(1, h)
    image_window = max(15, w // 90)
    smooth_image = np.convolve(
        density, np.ones(image_window, dtype=np.float32) / image_window, mode='same',
    )
    search_start, search_end = int(w * 0.45), int(w * 0.55)
    center_slice = smooth_image[search_start:search_end]
    if not len(center_slice):
        return None
    candidate_x = search_start + int(np.argmin(center_slice))
    gutter_half = max(6, w // 100)
    gutter_density = float(np.mean(smooth_image[
        max(0, candidate_x - gutter_half):min(w, candidate_x + gutter_half)
    ]))
    left_shoulder = smooth_image[max(0, candidate_x - w // 7):max(1, candidate_x - w // 25)]
    right_shoulder = smooth_image[min(w - 1, candidate_x + w // 25):min(w, candidate_x + w // 7)]
    shoulder_values = np.concatenate([left_shoulder, right_shoulder])
    shoulder_density = float(np.median(shoulder_values)) if len(shoulder_values) else 0.0
    centers = [
        (min(point[0] for point in item['bbox']) + max(point[0] for point in item['bbox'])) / 2
        for item in ocr_items
    ]
    center_left = sum(center < candidate_x for center in centers)
    center_right = sum(center >= candidate_x for center in centers)
    if (
        center_left >= 3 and center_right >= 3
        # 双页答案扫描的装订缝经常略偏（本地样本为页面宽度的45.3%）。
        # 下限仍明显高于单栏练习页曾出现的误分位置43.1%，兼顾两类页面。
        and w * 0.445 <= candidate_x <= w * 0.555
        and shoulder_density > 0
        and gutter_density <= shoulder_density * 0.72
    ):
        return candidate_x
    return None


def has_spanning_body_text(ocr_items, split_x, height, width):
    """候选分栏线被多行整行正文穿过时，判定它不是实际栏缝。

    上方两幅并排示意图、两张统计图等，常会在页面中部留下很宽的空白；
    仅靠该空白会把单栏题目的整行题干误判为双栏。本规则只看正文区内
    至少两行、明显横跨候选中线的长文本。真实双栏偶有横跨页眉，但通常
    不会连续出现在正文区；而真实栏缝会由 visual_column_split 提供独立证据。
    """
    if split_x is None or width < 400 or height < 400:
        return False
    side_margin = max(18, int(width * 0.035))
    min_line_width = max(240, int(width * 0.45))
    row_centers = []
    for item in ocr_items:
        bbox = item.get('bbox') or []
        if not bbox:
            continue
        x0 = min(point[0] for point in bbox)
        x1 = max(point[0] for point in bbox)
        y0 = min(point[1] for point in bbox)
        y1 = max(point[1] for point in bbox)
        text = re.sub(r'\s+', '', str(item.get('text', '')))
        if (
            len(text) < 6
            or y0 < height * 0.18
            or y1 > height * 0.92
            or x1 - x0 < min_line_width
            or x0 > split_x - side_margin
            or x1 < split_x + side_margin
        ):
            continue
        center_y = (y0 + y1) / 2
        # 同一行被重复 OCR 时只算一次，避免单行标题触发误判。
        if not any(abs(center_y - existing) <= max(12, (y1 - y0) * 0.7)
                   for existing in row_centers):
            row_centers.append(center_y)
    return len(row_centers) >= 2


def _split_cross_column_items(ocr_items, split_x):
    """拆分横跨中缝且同时含左右大题标题的 OCR 框。"""
    left_items, right_items = [], []
    marker_pattern = re.compile(r'([一二三四五六七八九十百]+)[、．\.]')
    for item in ocr_items:
        x0 = min(point[0] for point in item['bbox'])
        x1 = max(point[0] for point in item['bbox'])
        if x1 <= split_x + 10:
            left_items.append(item)
            continue
        if x0 >= split_x - 10:
            right_items.append(item)
            continue

        text = item.get('text', '')
        markers = list(marker_pattern.finditer(text))
        if x1 <= x0:
            continue
        # OCR sometimes merges a left-page minor item ("2.") with the next
        # right-page major title ("六、...") into one cross-gutter row.  One
        # Chinese major marker is enough to split that specific prefix/suffix;
        # previously it was discarded because there were not two major marks.
        if len(markers) == 1 and markers[0].start() > 0:
            segment_starts = [0, markers[0].start()]
        elif len(markers) >= 2:
            segment_starts = [marker.start() for marker in markers]
        else:
            continue
        segment_starts.append(len(text))
        for index in range(len(segment_starts) - 1):
            char_start, char_end = segment_starts[index], segment_starts[index + 1]
            fragment = text[char_start:char_end].strip()
            if not fragment:
                continue
            fragment_x0 = x0 + (x1 - x0) * char_start / max(1, len(text))
            fragment_x1 = x0 + (x1 - x0) * char_end / max(1, len(text))
            y0 = min(point[1] for point in item['bbox'])
            y1 = max(point[1] for point in item['bbox'])
            virtual = {
                'text': fragment,
                'bbox': [
                    [fragment_x0, y0], [fragment_x1, y0],
                    [fragment_x1, y1], [fragment_x0, y1],
                ],
                'conf': item.get('conf', 1.0),
            }
            if (fragment_x0 + fragment_x1) / 2 < split_x:
                left_items.append(virtual)
            else:
                right_items.append(virtual)
    return left_items, right_items


def _local_column_split_candidates(img, ocr_items):
    """Return verified internal column gutters, ranked by visual/OCR evidence.

    ``detect_columns`` intentionally concentrates on the page centre because it
    originally served two-page scans.  A real three-column exercise page has
    its two gutters around 1/3 and 2/3, so neither one is reliably visible to
    that old centre-only rule.  This helper is deliberately used only as part
    of recursive layout splitting and requires both image and OCR evidence.
    """
    height, width = img.shape[:2]
    if width < 520 or height < 420 or len(ocr_items) < 6:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y0, y1 = int(height * 0.10), int(height * 0.95)
    if y1 - y0 < 100:
        return []
    density = np.count_nonzero(gray[y0:y1] < 210, axis=0).astype(np.float32)
    window = max(11, width // 100)
    smooth = np.convolve(density, np.ones(window, dtype=np.float32) / window, mode='same')
    nonzero = smooth[smooth > 0]
    if not len(nonzero):
        return []
    baseline = float(np.percentile(nonzero, 60))
    low = smooth <= max(1.0, baseline * 0.20)
    min_gutter = max(8, width // 145)
    # After the first split of a three-column page, the remaining two-column
    # region is narrower while its original gutter keeps the same pixel width.
    # A 1/7 cap therefore rejects a perfectly valid second split (especially
    # on worksheets with generous column whitespace).  The OCR-on-both-sides
    # and spanning-text checks below are the safety guard; allow up to 18%.
    max_gutter = max(min_gutter + 1, int(width * 0.18))
    candidates = []
    run_start = None
    for index, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            candidates.append((run_start, index))
            run_start = None
    if run_start is not None:
        candidates.append((run_start, len(low)))

    centers = []
    for item in ocr_items:
        bbox = item.get('bbox') or []
        if not bbox:
            continue
        centers.append((
            min(point[0] for point in bbox), max(point[0] for point in bbox),
            (min(point[0] for point in bbox) + max(point[0] for point in bbox)) / 2,
        ))
    if len(centers) < 6:
        return []
    scored = []
    for start, end in candidates:
        run_width = end - start
        split_x = (start + end) // 2
        if not (min_gutter <= run_width <= max_gutter
                and width * 0.20 <= split_x <= width * 0.80):
            continue
        left_count = sum(center <= split_x - 4 for _x0, _x1, center in centers)
        right_count = sum(center >= split_x + 4 for _x0, _x1, center in centers)
        if left_count < 3 or right_count < 3:
            continue
        # At least one side must be an actual column-sized region.  This
        # rejects internal whitespace in a single wide question/table.
        if min(split_x, width - split_x) < max(250, int(width * 0.22)):
            continue
        if has_spanning_body_text(ocr_items, split_x, height, width):
            continue
        shoulder = np.concatenate((
            smooth[max(0, start - width // 12):start],
            smooth[end:min(width, end + width // 12)],
        ))
        shoulder_density = float(np.median(shoulder[shoulder > 0])) if np.any(shoulder > 0) else 0.0
        if shoulder_density <= 0:
            continue
        balance = min(left_count, right_count) / max(left_count, right_count)
        scored.append({
            'split_x': int(split_x),
            'score': float(run_width * (1.0 + balance)),
            'source': 'whitespace',
        })

    # Some exam sheets use a dotted/solid vertical rule instead of whitespace.
    # Reuse the visual evidence, but accept an off-centre divider only when the
    # OCR items prove that both resulting regions contain real text.
    binary = (gray < 210).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 16)))
    linked = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(linked, connectivity=8)
    divider_candidates = []
    for label in range(1, count):
        x, _y, component_width, component_height, _area = stats[label]
        split_x = x + component_width // 2
        if not (component_height >= height * 0.58
                and component_width <= max(12, width * 0.025)
                and width * 0.20 <= split_x <= width * 0.80
                and min(split_x, width - split_x) >= max(250, int(width * 0.22))):
            continue
        left_count = sum(center < split_x for _x0, _x1, center in centers)
        right_count = sum(center >= split_x for _x0, _x1, center in centers)
        if left_count >= 3 and right_count >= 3 and not has_spanning_body_text(ocr_items, split_x, height, width):
            divider_candidates.append({
                'split_x': int(split_x),
                'bar_start': int(x),
                'bar_end': int(x + component_width),
                'score': float(min(left_count, right_count)),
                'source': 'divider',
            })

    # A whitespace gutter is the stronger signal.  A drawn divider remains a
    # useful fallback, but is never allowed to hide a nearby whitespace gutter.
    # When both describe the *same* gutter (a blank band next to a printed
    # rule), the divider wins: its outer edges mark the true column boundary,
    # while the whitespace centre sits inside the blank band and would pull
    # the neighbouring box left over the rule.
    ranked = (
        sorted(divider_candidates, key=lambda item: item['score'], reverse=True)
        + sorted(scored, key=lambda item: item['score'], reverse=True)
    )
    distinct = []
    minimum_separation = max(12, int(width * 0.018))
    for candidate in ranked:
        if any(abs(candidate['split_x'] - previous['split_x']) < minimum_separation
               for previous in distinct):
            continue
        distinct.append(candidate)
    return distinct


def _local_column_split(img, ocr_items):
    """Return the strongest reliable internal column gutter, if any."""
    candidates = _local_column_split_candidates(img, ocr_items)
    if not candidates:
        return None
    first = candidates[0]
    # A printed divider rule is not part of either column: its right edge is
    # where the right-side column text starts, its left edge where the
    # left-side column text ends.  The centre would leak the rule into both.
    if first.get('source') == 'divider':
        return first['bar_end']
    return first['split_x']


def _verified_three_column_layout(img, ocr_items):
    """Return three globally-consistent reading columns, or ``None``.

    Recursive splitting is intentionally conservative, but a wide local blank
    inside one column can still win the first greedy split.  For a real
    three-column page the two gutters must jointly form three similarly sized
    text-bearing regions.  Selecting that pair before recursion avoids a
    false first split that would make subsequent question boxes cross columns.
    """
    height, width = img.shape[:2]
    # A global three-column decision needs two independent gutters.  Printed
    # divider rules are the physical ground truth of the column boundaries:
    # their outer edges mark exactly where each column's text must stop.  A
    # whitespace blank, by contrast, can be ordinary trailing whitespace
    # inside one column: the left column's text ends early and the large
    # blank band up to the printed rule scores as a huge gutter, which would
    # wrongly pull the cut out of the true gutter line and swallow the
    # neighbour's content.  So when a pair of printed rules verifies it always
    # wins; pure whitespace pairs are only consulted when the page has no
    # verified printed rules.
    all_candidates = _local_column_split_candidates(img, ocr_items)
    whitespace = [
        candidate for candidate in all_candidates
        if candidate.get('source') == 'whitespace'
    ]
    divider = [
        candidate for candidate in all_candidates
        if candidate.get('source') == 'divider'
    ]
    if width < 780 or height < 420:
        return None
    # Pair candidates within the same source only: a stray wide blank inside
    # one column must never pair with a competing divider.  Everything below
    # runs the same verification on each pool, but a verified printed-rule
    # pair takes precedence because the rules physically delimit the columns.
    divider_pair = _pick_three_column_pair(img, ocr_items, divider)
    if divider_pair:
        return divider_pair[1]
    whitespace_pair = _pick_three_column_pair(img, ocr_items, whitespace)
    return whitespace_pair[1] if whitespace_pair else None


def _pick_three_column_pair(img, ocr_items, candidates):
    """Return the best scored ``(score, regions)`` from one same-source pool."""
    height, width = img.shape[:2]
    if len(candidates) < 2:
        return None
    # The pair loop relies on candidate order being spatial: the first gutter
    # in the pair is the left one, the second the right one.  Candidate order
    # from _local_column_split_candidates is score-ranked, so sort here.
    candidates = sorted(candidates, key=lambda candidate: candidate['split_x'])

    best = None
    minimum_width = max(250, int(width * 0.18))
    for left_index, left_gutter in enumerate(candidates):
        for right_gutter in candidates[left_index + 1:]:
            left_split = int(min(left_gutter['split_x'], right_gutter['split_x']))
            right_split = int(max(left_gutter['split_x'], right_gutter['split_x']))
            widths = (left_split, right_split - left_split, width - right_split)
            if min(widths) < minimum_width or max(widths) / max(1, min(widths)) > 1.75:
                continue
            # Three-column gutters normally sit around one third and two
            # thirds of the page.  This rejects a large internal blank in a
            # single wide question while allowing moderately uneven columns.
            if not (width * 0.22 <= left_split <= width * 0.45
                    and width * 0.55 <= right_split <= width * 0.78):
                continue
            if (has_spanning_body_text(ocr_items, left_split, height, width)
                    or has_spanning_body_text(ocr_items, right_split, height, width)):
                continue

            left_items, remaining_items = _split_cross_column_items(ocr_items, left_split)
            middle_items, right_items = _split_cross_column_items(remaining_items, right_split)
            item_counts = (len(left_items), len(middle_items), len(right_items))
            if min(item_counts) < 2:
                continue
            # A column with very few OCR lines (short calculation questions or
            # a diagram legend) is still a genuine column.  Only guard against
            # a single stray fragment accidentally certifying a third slot:
            # sparse columns must carry a question-number anchor or a printed
            # group heading to be trusted.
            if min(item_counts) < 3 and any(
                not any(
                    extract_minor_idx(re.sub(r'\s+', '', it.get('text', ''))) is not None
                    or it.get('text', '').strip().startswith('知识点')
                    for it in group
                )
                for group in (left_items, middle_items, right_items)
                if len(group) < 3
            ):
                continue
            width_balance = min(widths) / max(widths)
            item_balance = min(item_counts) / max(item_counts)
            score = (
                left_gutter['score'] + right_gutter['score']
            ) * (0.75 + width_balance) * (0.75 + item_balance)
            # A printed divider rule belongs to neither column: each neighbour
            # starts/ends at the rule's outer edge so the rule itself is never
            # absorbed into a question box.  Whitespace gutters keep the
            # centred split used by the item-grouping checks above.
            left_start, left_end = (left_split, left_split)
            if left_gutter.get('source') == 'divider':
                left_start, left_end = (
                    left_gutter.get('bar_start', left_split),
                    left_gutter.get('bar_end', left_split),
                )
            right_start, right_end = (right_split, right_split)
            if right_gutter.get('source') == 'divider':
                right_start, right_end = (
                    right_gutter.get('bar_start', right_split),
                    right_gutter.get('bar_end', right_split),
                )
            proposal = (
                score,
                [(0, left_start, left_items),
                 (left_end, right_start, middle_items),
                 (right_end, width, right_items)],
            )
            if best is None or proposal[0] > best[0]:
                best = proposal
    return best


def split_question_layout_columns(img, ocr_items, max_columns=4):
    """Recursively split question pages into 1–4 reading-order columns."""
    _height, width = img.shape[:2]
    # Resolve a complete three-column layout before greedy recursion.  This is
    # more reliable on exam pages where an unusually wide blank inside one
    # column could otherwise be picked as the first split.
    verified_three = _verified_three_column_layout(img, ocr_items)
    if verified_three:
        return verified_three
    regions = [(0, width, list(ocr_items))]
    while len(regions) < max_columns:
        expanded = []
        changed = False
        for x0, x1, items in regions:
            if len(regions) + int(changed) >= max_columns or x1 - x0 < 520:
                expanded.append((x0, x1, items))
                continue
            local_items = rebase_ocr_items(items, x_offset=x0)
            local_split = _local_column_split(img[:, x0:x1], local_items)
            if local_split is None:
                expanded.append((x0, x1, items))
                continue
            split_x = x0 + int(local_split)
            left, right = _split_cross_column_items(items, split_x)
            if len(left) < 3 or len(right) < 3:
                expanded.append((x0, x1, items))
                continue
            expanded.extend(((x0, split_x, left), (split_x, x1, right)))
            changed = True
        regions = expanded
        if not changed:
            break
    return sorted(regions, key=lambda region: region[0])


def _answer_matrix_crosses_split(img, ocr_items, split_x):
    """Whether a proposed column split cuts through a dense answer matrix.

    Answer keys frequently arrange short ``① 35 / ② 9 / ③ 57`` cells in a
    grid.  The gaps between those cells look exactly like reading-column
    gutters to the generic layout splitter, but splitting there produces
    incomplete answer crops.  This guard is deliberately narrow: it only
    rejects a split when at least three OCR rows each contain two compact,
    numbered answer cells on *both* sides of that same split.  Normal prose,
    ordinary two-column answer pages, and sparse layouts retain the existing
    recursive behaviour.
    """
    if img is None or img.size == 0 or split_x is None:
        return False
    height, width = img.shape[:2]
    if not (max(36, int(width * 0.18)) <= split_x <= min(width - 36, int(width * 0.82))):
        return False

    numbered_cell = re.compile(
        r'^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕]'
        r'|\d{1,3}\s*[\.．、])'
    )
    matched_rows = []
    for row in group_ocr_rows(ocr_items):
        cells = []
        for item in row.get('items', []):
            text = str(item.get('text', '')).strip()
            bbox = item.get('bbox') or []
            if not text or not bbox or len(re.sub(r'\s+', '', text)) > 14:
                continue
            if not numbered_cell.match(text):
                continue
            x0 = min(point[0] for point in bbox)
            x1 = max(point[0] for point in bbox)
            # A long OCR paragraph beginning with ``1.`` is a question/answer
            # line, not a matrix cell.  Keep the matrix rule away from it.
            if x1 - x0 > width * 0.28:
                continue
            cells.append(((x0 + x1) / 2, x0, x1))
        left_count = sum(center < split_x for center, _, _ in cells)
        right_count = sum(center > split_x for center, _, _ in cells)
        if left_count >= 2 and right_count >= 2:
            matched_rows.append(row)

    if len(matched_rows) < 3:
        return False
    # Require the pattern to occupy a real vertical run, rather than three
    # OCR fragments accidentally aligned around a decorative heading.
    first_y = min(row['y0'] for row in matched_rows)
    last_y = max(row['y1'] for row in matched_rows)
    typical_height = float(np.median([row['height'] for row in matched_rows]))
    return last_y - first_y >= max(typical_height * 2.0, height * 0.045)


def _has_answer_reading_divider(img, split_x):
    """Detect a long printed rule that proves two answer reading columns.

    A dense matrix can happen to line up on both sides of a true column
    divider.  In that case retaining the divider is safer than preserving the
    matrix as one crop: the rule is physical layout evidence, whereas the
    alignment is only a statistical clue.  Short table borders do not qualify
    because the ink must form one near-continuous line through most of the
    page body.
    """
    if img is None or img.size == 0 or split_x is None:
        return False
    height, width = img.shape[:2]
    if not (max(12, int(width * 0.15)) <= split_x <= min(width - 12, int(width * 0.85))):
        return False
    y0, y1 = int(height * 0.12), int(height * 0.94)
    if y1 - y0 < 80:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ``detect_columns`` returns the centre of a whitespace gutter; a printed
    # rule can sit near either gutter edge.  Search only this small local
    # neighbourhood, never the full page, so ordinary text cannot certify an
    # unrelated split.
    half_width = max(8, min(28, int(width * 0.045)))
    strip = gray[y0:y1, max(0, split_x - half_width):min(width, split_x + half_width + 1)]
    if strip.size == 0:
        return False
    inked = np.any(strip < 205, axis=1).astype(np.uint8)
    # Keep a scanned/rasterised rule continuous across tiny pixel gaps, but do
    # not join the unrelated text lines of a normal answer list.
    bridge = max(2, min(8, (y1 - y0) // 90))
    inked = cv2.morphologyEx(
        inked.reshape(-1, 1), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)),
    ).ravel()
    longest = current = 0
    for value in inked:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest >= int((y1 - y0) * 0.58)


def _answer_split_has_independent_numbered_streams(ocr_items, split_x):
    """Require evidence before recursively splitting an answer-grid gutter.

    A real extra reading column normally has several printed ``1.``/``2.``
    anchors on both sides.  A grid of circled response cells has neither; its
    internal whitespace must not become a third or fourth column merely
    because OCR happened to split the cells into separate text items.
    """
    left_items, right_items = _split_cross_column_items(ocr_items, split_x)

    def numbered_rows(items):
        count = 0
        for row in group_ocr_rows(items):
            matches = [
                match for match in _answer_minor_matches(row.get('text', ''))
                if not _answer_match_is_decimal_literal(row.get('text', ''), *match)
            ]
            if matches:
                count += 1
        return count

    return numbered_rows(left_items) >= 2 and numbered_rows(right_items) >= 2


def split_answer_layout_columns(img, ocr_items, max_columns=4):
    """递归拆分答案页阅读栏，且不拆开横向排列的答案矩阵。"""
    height, width = img.shape[:2]
    del height
    # A long visual page gutter is stronger evidence than OCR whitespace.  It
    # establishes the two physical answer columns first, then recursion may
    # add a column only with its own divider or two independent numbered flows.
    physical_split = _visual_column_split(img)
    if physical_split is not None:
        left, right = _split_cross_column_items(ocr_items, physical_split)
        regions = [(0, physical_split, left), (physical_split, width, right)]
    else:
        regions = [(0, width, list(ocr_items))]
    while len(regions) < max_columns:
        changed = False
        expanded = []
        splits_remaining = max_columns - len(regions)
        for x0, x1, items in regions:
            if splits_remaining <= 0 or x1 - x0 < 420:
                expanded.append((x0, x1, items))
                continue
            local_img = img[:, x0:x1]
            local_items = rebase_ocr_items(items, x_offset=x0)
            local_split = detect_columns(local_img, local_items)
            if local_split is None:
                expanded.append((x0, x1, items))
                continue
            # Preserve the initial split of a clearly horizontal spread: it
            # may be two physical pages.  Once a portrait-like page/column is
            # reached, a repeated two-cells-on-each-side pattern is evidence
            # of one answer matrix rather than separate reading columns.
            is_horizontal_spread = local_img.shape[1] / max(1, local_img.shape[0]) >= 1.28
            has_reading_divider = _has_answer_reading_divider(local_img, local_split)
            if (
                not is_horizontal_spread
                and not has_reading_divider
                and (
                    _answer_matrix_crosses_split(local_img, local_items, local_split)
                    or not _answer_split_has_independent_numbered_streams(local_items, local_split)
                )
            ):
                expanded.append((x0, x1, items))
                continue
            absolute_split = x0 + local_split
            left, right = _split_cross_column_items(items, absolute_split)
            if len(left) < 2 or len(right) < 2:
                expanded.append((x0, x1, items))
                continue
            expanded.extend([
                (x0, absolute_split, left),
                (absolute_split, x1, right),
            ])
            splits_remaining -= 1
            changed = True
        regions = expanded
        if not changed:
            break
    return sorted(regions, key=lambda region: region[0])


# ═══════════════════════════════════════════════════════════════
# 阶段1：按大题边界切分页面
# ═══════════════════════════════════════════════════════════════
def find_major_boundaries(ocr_items):
    """找页面中的大题边界，返回 [{text, y, x, idx}]"""
    boundaries = []
    for item in ocr_items:
        text = item['text'].strip()
        if RE_MAJOR.match(text):
            boundaries.append({
                'text': text,
                'y': int(min(p[1] for p in item['bbox'])),
                'x': int(min(p[0] for p in item['bbox'])),
                'idx': extract_major_idx(text),
            })
    
    # 去重
    seen = set()
    unique = []
    for b in boundaries:
        key = b['y'] // 30
        if key not in seen:
            seen.add(key)
            unique.append(b)

    # 同一行标题被 OCR 拆成两个文字框时，两个框的 y 只差十几到几十像素，
    # 但恰好跨过 30 的整数倍就各占一个桶、都留下来。后续切块会把它们当成
    # 两个相邻大题，产出「高度 9~13px、标题与下一块完全相同」的空壳块
    # （金版卷王四年级 56260587、六年级 15260935 都是这个形态）。
    # 因此再按「同一大题号 + y 足够近」合并，保留更靠上的那个。
    unique.sort(key=lambda value: value['y'])
    merged = []
    for boundary in unique:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and boundary['idx'] is not None
            and boundary['idx'] == previous['idx']
            and boundary['y'] - previous['y'] < SPLIT_MAJOR_MERGE_GAP
            # 对开页左右两页可能印着同一个大题号且在同一水平线，
            # 只有 x 也在同一段（一行标题的宽度范围内）才算同一行被拆开。
            and abs(boundary.get('x', 0) - previous.get('x', 0)) < SPLIT_MAJOR_MERGE_DX
        ):
            continue
        merged.append(boundary)
    return merged


def _is_olympiad_lesson_heading(item, page_height):
    """判断“第三讲/第 3 课”这类页内讲次标题，不能误认页眉书名。"""
    if not item or page_height <= 0:
        return False
    text = re.sub(r'\s+', '', item.get('text', ''))
    if not OLYMPIAD_LESSON_RE.fullmatch(text):
        return False
    bbox = item.get('bbox') or []
    if not bbox:
        return False
    y = min(point[1] for point in bbox)
    # 页码或页眉中的“第X讲”不作为正文讲解的起点。
    return page_height * 0.08 <= y <= page_height * 0.55


def find_knowledge_point_boundaries(ocr_items, page_height):
    """识别“知识点一 有理数”这类节标题，作为大题边界候选。

    周测/练习册常以知识点组织题目，且题目本身不再使用“一、二、三”大题号。
    只认 RE_MAJOR 时这类标题一次都不会命中，整页退回“整页”单块，知识点
    标题被并进相邻小题的裁图。这里接收 OCR 项，返回的边界不限定左右位置
    （知识点标题有左对齐也有居中），但要求短小且形如“知识点+中文数字”。
    """
    boundaries = []
    min_body_y = page_height * 0.04
    max_body_y = page_height * 0.96
    for item in ocr_items:
        text = re.sub(r'\s+', '', item.get('text', ''))
        if not text.startswith('知识点') or len(text) > 20:
            continue
        if not (
            re.match(r'^知识点[一二三四五六七八九十]{1,3}', text)
            or re.match(r'^知识点[0-9]{1,3}', text)
        ):
            continue
        bbox = item.get('bbox') or []
        if not bbox:
            continue
        y = min(point[1] for point in bbox)
        if not (min_body_y <= y <= max_body_y):
            continue
        boundaries.append({
            'text': item.get('text', '').strip(),
            'y': int(y),
            'idx': None,
        })
    # 同一行被 OCR 拆成多个文字框时按 y 去重。
    seen = set()
    unique = []
    for boundary in sorted(boundaries, key=lambda value: value['y']):
        key = boundary['y'] // 30
        if key in seen:
            continue
        seen.add(key)
        unique.append(boundary)
    return unique


def find_olympiad_field_boundaries(ocr_items, page_height):
    """识别奥数书的题目栏目，例如“难题点拨”“想一想做一做”。

    返回的 idx 与普通大题号不重叠；“难题点拨①/②”分别使用稳定编号，
    因而跨页的续题可以继续归入相同栏目。书名中出现“难题点拨”不会命中，
    因为这里只接受短小的完整栏目名。
    """
    if page_height <= 0:
        return []
    boundaries = []
    min_body_y = page_height * 0.10
    # 彩色栏目有时被 OCR 拆成同一行的两三个文字框（如“想一想”+“做一做”）。
    # 只补充很短的行级候选，正文长行仍不会命中下方的完整标题规则。
    candidates = list(ocr_items)
    for row in group_ocr_rows(ocr_items):
        row_items = row.get('items', [])
        if not 2 <= len(row_items) <= 3 or len(re.sub(r'\s+', '', row['text'])) > 16:
            continue
        candidates.append({
            'text': row['text'],
            'bbox': [[row['x0'], row['y0']], [row['x1'], row['y0']],
                     [row['x1'], row['y1']], [row['x0'], row['y1']]],
            'conf': min(item.get('conf', 1.0) for item in row_items),
        })
    for item in candidates:
        if item.get('conf', 1.0) < MIN_OCR_CONFIDENCE:
            continue
        text = re.sub(r'\s+', '', item.get('text', ''))
        bbox = item.get('bbox') or []
        if not text or not bbox:
            continue
        y = int(min(point[1] for point in bbox))
        if y < min_body_y or y >= page_height * 0.94:
            continue
        for pattern, base_idx in OLYMPIAD_FIELD_PATTERNS:
            if not pattern.fullmatch(text):
                continue
            idx = base_idx
            title = '想一想做一做' if base_idx == 2200 else text
            if base_idx == 2000 and text[-1:] in OLYMPIAD_CIRCLED_NUMBERS:
                idx += OLYMPIAD_CIRCLED_NUMBERS.index(text[-1]) + 1
            if any(abs(y - previous['y']) <= 35 for previous in boundaries):
                break
            boundaries.append({
                'y': y,
                'text': title,
                'idx': idx,
                'semantic_field': True,
            })
            break
    return sorted(boundaries, key=lambda boundary: boundary['y'])


def find_circled_number_boundaries(ocr_items, crop_width, crop_height):
    """识别用 ①②③ 编号的整道练习（部分奥数书不用“一、二、三”）。

    规则刻意保守，只认“圈号 + 至少 4 个字的题干”且靠左起行：
    图注、选项标记常常只有一个圈号，正文里的“如图①”也不在行首。
    """
    if crop_height <= 0 or crop_width <= 0:
        return []
    boundaries = []
    for item in ocr_items:
        if item.get('conf', 1.0) < MIN_OCR_CONFIDENCE:
            continue
        match = RE_CIRCLED_STEM.match(item.get('text', '').strip())
        if not match:
            continue
        stem = re.sub(r'\s+', '', match.group(2))
        if len(stem) < 4:
            continue
        bbox = item.get('bbox') or []
        if not bbox:
            continue
        x0 = int(min(point[0] for point in bbox))
        y = int(min(point[1] for point in bbox))
        if x0 > crop_width * 0.22:
            continue
        if not crop_height * 0.05 <= y < crop_height * 0.95:
            continue
        if any(abs(y - known['y']) <= 35 for known in boundaries):
            continue
        boundaries.append({
            'y': y,
            'text': f'{match.group(1)} {stem[:20]}',
            'idx': 5000 + OLYMPIAD_CIRCLED_NUMBERS.index(match.group(1)) + 1,
        })
    return sorted(boundaries, key=lambda boundary: boundary['y'])


def _apply_layout_arbiter(img, multi_regions, ocr_items):
    """用版面模型复核“≥3 栏”的判定，避免把整页正文吞进单块。

    ``split_question_layout_columns`` 的栏缝证据全部来自页内局部几何，
    当某栏右侧只剩一条括号/答案标记列时，栏目内部的白带会被当成第二道
    栏缝，于是页被切成 3 块、右侧整页大题并进同一个“整页”块。

    版面裁判（PP-DocLayout）按区域框判断“这条竖带有没有被正文穿过”，
    只在现有结论为 ≥3 栏时介入；模型缺失或关闭时原样返回，主流程行为不变。
    返回 ``(regions, overridden)``。
    """
    if len(multi_regions) < 3:
        return multi_regions, False
    try:
        import layout_arbiter
    except Exception:  # noqa: BLE001
        return multi_regions, False
    try:
        corrected = layout_arbiter.resolve_regions(
            img, multi_regions, ocr_items, log=print,
        )
    except Exception as error:  # noqa: BLE001
        print(f'[版面裁判] 跳过（{type(error).__name__}: {error}）')
        return multi_regions, False
    if not corrected:
        return multi_regions, False
    same = len(corrected) == len(multi_regions) and all(
        abs(float(a0) - float(b0)) <= 1 and abs(float(a1) - float(b1)) <= 1
        for (a0, a1, _x), (b0, b1, _y) in zip(corrected, multi_regions)
    )
    return corrected, not same


def split_page_into_majors(img, ocr_items, padding=6, profile='general'):
    """将页面按大题边界切分成大题块
    
    返回: [{title, idx, crop_img}]  或  [{title='整页', idx=0, crop_img}]
    """
    h, w = img.shape[:2]

    # A three/four-column exam page cannot be represented by one centre split:
    # after the first cut the remaining two columns would be treated as one
    # wide column and every horizontal minor boundary leaks across it.  Detect
    # a verified multi-column layout first and process each reading column in
    # full before moving to the next one.  Keep the established two-column
    # branch below as a conservative fallback for ordinary spreads.
    multi_regions = split_question_layout_columns(img, ocr_items, max_columns=4)
    multi_regions, arbiter_used = _apply_layout_arbiter(img, multi_regions, ocr_items)
    # 裁判覆盖后即使只剩两栏，也按“逐栏切开”处理：栏缝由裁判给出，
    # 不再让下方基于局部空隙的启发式重新决定栏位（那正是误判来源）。
    if len(multi_regions) >= 3 or (arbiter_used and len(multi_regions) == 2):
        blocks = []
        two_columns = len(multi_regions) == 2
        for column_index, (x0, x1, column_items) in enumerate(multi_regions):
            local_items = rebase_ocr_items(column_items, x_offset=x0)
            column_blocks = _split_single_col_majors(
                img[:, x0:x1], local_items, padding, profile=profile,
            )
            for block in column_blocks:
                # 两栏沿用 left/right 语义：跨页延续逻辑靠它区分左右半页。
                block['_col'] = (
                    ('left' if column_index == 0 else 'right')
                    if two_columns else f'column_{column_index + 1}'
                )
                block['_x_start'] = x0
                block['_column_bounds'] = [x0, x1]
            blocks.extend(column_blocks)
        return blocks
    
    # 双栏检测。OCR 的“低密度沟”有时会落在左页末尾的大片空白中，
    # 而不是扫描双页真正的装订缝；这种情况下右页裁块会把左页尾题带进去。
    # 对宽幅双页扫描，若图像装订缝与 OCR 分栏明显冲突，优先可信的图像缝。
    ocr_split_x = detect_columns(img, ocr_items)
    visual_split_x = _visual_column_split(img)
    # OCR 仅因上方并排图形的空隙推断出双栏、但整行题干横跨中线时，
    # 不能把题目从中间竖切。第十四讲第15题（两幅示意图+整行题干）
    # 即为这一类；真正双栏会有可见栏沟/竖线作为独立视觉证据。
    if (
        ocr_split_x is not None
        and visual_split_x is None
        and has_spanning_body_text(ocr_items, ocr_split_x, h, w)
    ):
        ocr_split_x = None
    split_x = ocr_split_x
    height, width = img.shape[:2]
    if visual_split_x is not None:
        materially_disagrees = (
            ocr_split_x is not None
            and abs(ocr_split_x - visual_split_x) >= max(80, int(width * 0.035))
        )
        is_wide_spread = width / max(1, height) >= 1.24
        if ocr_split_x is None or (is_wide_spread and materially_disagrees):
            split_x = visual_split_x
    
    if split_x is not None:
        # 双栏：左右分别处理再合并
        left_items, right_items = _split_cross_column_items(ocr_items, split_x)
        
        left_blocks = _split_single_col_majors(
            img[:, :split_x], left_items, padding, profile=profile,
        )
        right_blocks = _split_single_col_majors(
            img[:, split_x:], right_items, padding, profile=profile,
        )
        
        # 标记左/右栏
        for b in left_blocks:
            b['_col'] = 'left'
            b['_x_start'] = 0
            b['_column_bounds'] = [0, split_x]
        for b in right_blocks:
            b['_col'] = 'right'
            b['_x_start'] = split_x
            b['_column_bounds'] = [split_x, width]
        
        # 教材双栏和双页扫描的阅读顺序均应先左栏到底，再进入右栏。
        # 按全局 Y 混排会让不同页面的大题/答案相互穿插。
        return left_blocks + right_blocks
    else:
        blocks = _split_single_col_majors(img, ocr_items, padding, profile=profile)
        blocks = _reorder_mixed_blocks(blocks, width)
        for block in blocks:
            block['_x_start'] = 0
            block['_column_bounds'] = [0, width]
        return blocks


def _find_major_section_cut(
    img, current_y, next_y, fallback_padding=6, row_density=None,
):
    """在相邻大题/活动区之间找最长的真实空白带，避开下一标题上方装饰。"""
    gap = int(next_y) - int(current_y)
    fallback = max(int(current_y) + 1, int(next_y) - fallback_padding)
    if gap < 80:
        return fallback
    height = img.shape[0]
    search_start = max(0, int(current_y + gap * 0.52))
    search_end = min(height, int(next_y) - 1)
    if search_end - search_start < 12:
        return fallback

    if row_density is None:
        mask = _content_mask(img)
        density = np.count_nonzero(mask, axis=1).astype(np.float32)
    else:
        density = np.asarray(row_density, dtype=np.float32)
    smooth = np.convolve(density, np.ones(5, dtype=np.float32) / 5, mode='same')
    window = smooth[search_start:search_end]
    threshold = max(1.0, float(np.percentile(window, 22)))
    low = window <= threshold
    runs = []
    run_start = None
    for index, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(low)))
    usable = [run for run in runs if run[1] - run[0] >= 5]
    if not usable:
        return fallback
    best_start, best_end = max(
        usable,
        key=lambda run: (run[1] - run[0], run[1]),
    )
    return search_start + (best_start + best_end) // 2


def _reorder_mixed_blocks(blocks, page_width):
    """校正“左右两栏按 y 混排”的阅读顺序。

    练习册常以一条点划线分开两栏。视觉分栏找的是纯留白或连续实线，虚线两条
    都不满足，于是没被识别成两栏；单栏切分把所有大题按 y 排序，得到
    一、四、五、二、六、三 这种交错顺序。每个块的内容都是对的，但交付文件
    会按数组顺序编号，顺序就跟着错了（全能测控 image0002586A）。

    这里只在证据明确时重排：标题 x 能分成两组、组心相距 >25% 页宽、两组各
    至少 2 块，且按 y 排出来的大题号确实回头。
    """
    if len(blocks) < 4 or page_width <= 0:
        return blocks
    x_values = sorted(
        block['_title_x'] for block in blocks if block.get('_title_x') is not None
    )
    if len(x_values) < 4:
        return blocks
    gaps = [
        (x_values[index + 1] - x_values[index], index)
        for index in range(len(x_values) - 1)
    ]
    biggest_gap, cut_index = max(gaps)
    if biggest_gap < page_width * 0.25:
        return blocks
    threshold = (x_values[cut_index] + x_values[cut_index + 1]) / 2
    left = [
        block for block in blocks
        if block.get('_title_x') is not None and block['_title_x'] < threshold
    ]
    right = [
        block for block in blocks
        if block.get('_title_x') is not None and block['_title_x'] >= threshold
    ]
    if len(left) < 2 or len(right) < 2:
        return blocks
    sequence = [
        block['idx'] for block in blocks
        if isinstance(block.get('idx'), int) and 0 < block['idx'] < 100
    ]
    if len(sequence) < 3:
        return blocks
    if all(later > earlier for earlier, later in zip(sequence, sequence[1:])):
        return blocks
    head = [block for block in blocks if block.get('_title_x') is None]
    left.sort(key=lambda block: block.get('_y_start', 0))
    right.sort(key=lambda block: block.get('_y_start', 0))
    return head + left + right


def _split_single_col_majors(img, ocr_items, padding=6, profile='general'):
    """单栏大题切分"""
    h, w = img.shape[:2]
    strategy = layout_strategy.strategy_for(profile)
    boundaries = find_major_boundaries(ocr_items)

    # 奥数辅导书多以“难题点拨”“想一想做一做”等栏目组织，而没有
    # “一、二、三”大题号。把这些短标题提升为一级分类，避免整讲题目
    # 都落到“未分类题目”。普通正文中的“点拨/拓展”不参与此处切分。
    if strategy.field_headings:
        existing_y = [boundary['y'] for boundary in boundaries]
        for boundary in find_olympiad_field_boundaries(ocr_items, h):
            if any(abs(boundary['y'] - known_y) <= 35 for known_y in existing_y):
                continue
            boundaries.append(boundary)
            existing_y.append(boundary['y'])
        boundaries.sort(key=lambda boundary: boundary['y'])

    # 有些奥数书连“难题点拨”都不印，直接用 ①②③ 给整道练习编号；
    # extract_major_idx 只认中文大题号，页面会整页塌成一块。
    if strategy.circled_major_numbers:
        existing_y = [boundary['y'] for boundary in boundaries]
        for boundary in find_circled_number_boundaries(ocr_items, w, h):
            if any(abs(boundary['y'] - known_y) <= 35 for known_y in existing_y):
                continue
            boundaries.append(boundary)
            existing_y.append(boundary['y'])
        boundaries.sort(key=lambda boundary: boundary['y'])

    # 暑假/寒假作业常把“数学游戏、假期体验”作为独立活动区，却不用中文
    # 大题号。把这些醒目的活动标题提升为独立块，避免统统塞进前一个大题。
    if strategy.activity_headings:
        existing_y = [boundary['y'] for boundary in boundaries]
        activity_index = 0
        for item in ocr_items:
            text = re.sub(r'\s+', '', item.get('text', ''))
            keyword = next(
                (
                    candidate for candidate in ACTIVITY_HEADING_KEYWORDS
                    if candidate in text and len(text) <= len(candidate) + 8
                ),
                None,
            )
            if not keyword:
                continue
            y = int(min(point[1] for point in item['bbox']))
            if any(abs(y - known_y) <= 35 for known_y in existing_y):
                continue
            activity_index += 1
            boundaries.append({
                'y': y,
                'text': keyword,
                'idx': 100 + activity_index,
            })
            existing_y.append(y)
        boundaries.sort(key=lambda boundary: boundary['y'])

    # 周测/练习册常以“知识点一 有理数”为节标题，不再使用中文大题号。
    # 提升为独立大题块，避免知识点标题被并进相邻小题的裁图。
    existing_y = [boundary['y'] for boundary in boundaries]
    knowledge_index = 0
    for boundary in find_knowledge_point_boundaries(ocr_items, h):
        if any(abs(boundary['y'] - known_y) <= 40 for known_y in existing_y):
            continue
        knowledge_index += 1
        boundary['idx'] = 3000 + knowledge_index
        boundaries.append(boundary)
        existing_y.append(boundary['y'])
    boundaries.sort(key=lambda boundary: boundary['y'])
    
    if not boundaries:
        # 讲次首页往往先是概念讲解或示例，随后才出现“难题点拨”。
        # 这类内容也应有可读的字段名，而不是直接被标为未分类。
        if strategy.lesson_opening_fallback and any(
            _is_olympiad_lesson_heading(item, h) for item in ocr_items
        ):
            return [{
                'title': '讲解示例', 'idx': 2500, 'crop_img': img,
                '_y_start': 0, 'semantic_field': True,
            }]
        return [{'title': '整页', 'idx': 0, 'crop_img': img, '_y_start': 0}]
    
    blocks = []
    section_row_density = (
        np.count_nonzero(_content_mask(img), axis=1)
        if strategy.density_section_cut and len(boundaries) > 1
        else None
    )

    # 双页扫描的右半页经常先接着排完上一大题，随后才出现“四、选择题”等
    # 新标题。旧逻辑从第一个大题标题处开始切，会悄悄丢掉标题上方的续题。
    # 当本栏第一个大题号不是“一”且标题上方确有题号/正文时，先保留为延续块；
    # 后续汇总会按阅读顺序把它归入左栏最后一个大题。
    first = boundaries[0]
    prefix_end = max(0, int(first['y']) - padding)
    if (
        (1 < first['idx'] < 100 or first.get('semantic_field') or first['idx'] >= 3000)
        and prefix_end >= max(70, int(h * 0.06))
    ):
        prefix_items = []
        for item in ocr_items:
            center_y = (
                min(point[1] for point in item['bbox'])
                + max(point[1] for point in item['bbox'])
            ) / 2
            if center_y < prefix_end:
                prefix_items.append(item)
        has_numbered_question = bool(
            find_minor_boundaries(prefix_items, w, max(prefix_end, 1))
        )
        meaningful_text = [
            item for item in prefix_items
            if len(re.sub(r'\s+', '', item.get('text', ''))) >= 2
        ]
        if has_numbered_question or len(meaningful_text) >= 3:
            prefix_crop = img[0:prefix_end, 0:w]
            if prefix_crop.size:
                # 第一页的“第X讲”之前/附近是本讲讲解，不应生成未分类目录；
                # 其余语义栏目前的内容则作为上一栏目的跨页续题保留。
                is_lesson_opening = (
                    first.get('semantic_field')
                    and any(_is_olympiad_lesson_heading(item, h) for item in prefix_items)
                )
                blocks.append({
                    'title': '讲解示例' if is_lesson_opening else '整页',
                    'idx': 2500 if is_lesson_opening else 0,
                    'crop_img': prefix_crop,
                    '_y_start': 0,
                    'semantic_field': is_lesson_opening,
                })

    for i, bd in enumerate(boundaries):
        y_start = max(0, bd['y'] - padding)
        if i + 1 < len(boundaries):
            if strategy.density_section_cut:
                y_end = _find_major_section_cut(
                    img, bd['y'], boundaries[i + 1]['y'], padding,
                    row_density=section_row_density,
                )
            else:
                y_end = int(boundaries[i + 1]['y']) - padding
        else:
            y_end = h
        y_end = min(y_end, h)
        if y_end <= y_start:
            continue
        
        crop = img[y_start:y_end, 0:w]
        if crop.size == 0:
            continue
        
        # 提取大题标题文字（如"一、选择题"）
        title = bd['text']
        # 如果OCR丢掉了中文数字（如"、填空题"缺了"一"），补回
        if title and title[0] in '、．.':
            cn_idx = num_to_cn(bd['idx'])
            title = f"{cn_idx}{title}"
        # 截取到合理长度做目录名
        title = safe_dirname(title, 40)
        
        blocks.append({
            'title': title,
            'idx': bd['idx'],
            'crop_img': crop,
            '_y_start': y_start,
            # 标题左缘的 x，仅用于单栏分支里校正“左右两栏按 y 混排”的阅读顺序。
            '_title_x': int(bd.get('x', 0) or 0),
            'semantic_field': bd.get('semantic_field', False),
        })
    
    if not blocks:
        blocks.append({'title': '整页', 'idx': 0, 'crop_img': img, '_y_start': 0})
    
    return blocks


# ═══════════════════════════════════════════════════════════════
# 阶段2：在大题块内按小题切分
# ═══════════════════════════════════════════════════════════════
def _find_clean_horizontal_cut(
    img, current_y, next_y, fallback_padding=4, *, content_row_density=None,
):
    """在相邻题号之间的靠下空白带落刀，避开下一题先出现的括号/图形。"""
    gap = int(next_y) - int(current_y)
    fallback = max(int(current_y) + 1, int(next_y) - fallback_padding)
    if gap < 36:
        return fallback
    height, width = img.shape[:2]
    search_start = max(0, int(current_y + gap * 0.58))
    search_end = min(height, int(next_y) - 1)
    if search_end - search_start < 8:
        return fallback

    density = content_row_density
    if density is None:
        mask = _content_mask(img)
        density = np.count_nonzero(mask, axis=1).astype(np.float32)
    smooth = np.convolve(density, np.ones(3, dtype=np.float32) / 3, mode='same')
    window = smooth[search_start:search_end]
    threshold = max(1.0, float(np.percentile(window, 25)))
    low = window <= threshold
    runs = []
    run_start = None
    for index, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(low)))
    if not runs:
        return fallback
    # 当前题可能有多行选项。选择“最长空白”会误落在C、D选项之间；
    # 应选择最靠近下一题号、且有实际宽度的最后一条空白带。
    usable_runs = [run for run in runs if run[1] - run[0] >= 3]
    if not usable_runs:
        return fallback
    best_start, best_end = max(usable_runs, key=lambda run: (run[1], run[1] - run[0]))
    if best_end - best_start < 3:
        return fallback
    return search_start + (best_start + best_end) // 2


def _recover_missed_minor_boundaries(major_crop, boundaries):
    """找回被结构 OCR 整条漏读的题号，补回缺失的小题边界。

    结构 OCR 是整页降采样后的一次轻量识别，偶会漏掉一整条题干（典型如
    一个小题号独占一行的题干）。`find_minor_boundaries` 因此得到的序列会
    跳号（1→3），相邻两道题被并入同一个裁剪框。本函数对跳号两侧之间的
    条带按原生分辨率单独重识别（条带小，成本远低于整页高精度 OCR），
    找回缺失的题号后插回边界序列。

    只处理“同一编号序列内跳号 ≥2”的确定签名；回到 1 是新分节/知识点，
    不属于漏读。重识别结果必须同时满足：题号样式合法、位于左缘、且编号
    正好填补跳号（前一号 +1 或后一号 -1），否则宁可不拆，保持现状。
    """
    if len(boundaries) < 2 or major_crop is None or major_crop.size == 0:
        return boundaries
    crop_height, crop_width = major_crop.shape[:2]
    if crop_height < 100 or crop_width < 200:
        return boundaries
    recovered = []
    for index in range(len(boundaries) - 1):
        left, right = boundaries[index], boundaries[index + 1]
        left_num = left.get('num')
        right_num = right.get('num')
        if not isinstance(left_num, int) or not isinstance(right_num, int):
            continue
        if right_num == 1 or right_num - left_num < 2:
            continue
        y_start = int(left.get('y', 0)) + 4
        y_end = int(right.get('y', crop_height)) - 4
        if y_end - y_start < 40:
            continue
        # 题号只出现在左缘：x 超出 left_limit 的识别结果下面会被整批丢弃，
        # 所以条带右侧一大半像素是白跑的。裁窄后 OCR 面积降到约 1/3，
        # 对开扫描上实测这一项是小题切分的主要耗时来源。
        # 比左缘判据多留 8% 页宽余量，避免缩进题号的前几个字被切掉、
        # 文本长度掉到 body_min 以下反而漏题。
        strip_width = min(crop_width, max(160, int(crop_width * 0.36)))
        strip = major_crop[y_start:y_end, :strip_width]
        if strip.size == 0 or strip.shape[0] < 30:
            continue
        try:
            strip_items = _run_ocr_cached_strip(strip)
        except Exception:
            continue
        left_limit = max(90, int(crop_width * 0.28))
        candidates = []
        for item in strip_items:
            text = str(item.get('text', '')).strip()
            num, style = extract_minor_anchor(text)
            if num is None or num > 150 or item.get('conf', 1.0) < MIN_OCR_CONFIDENCE:
                continue
            bbox = item.get('bbox') or []
            if not bbox:
                continue
            try:
                x = float(min(point[0] for point in bbox))
                y = float(min(point[1] for point in bbox))
            except (TypeError, ValueError):
                continue
            if x < 0 or x > left_limit or y < 0 or y >= strip.shape[0]:
                continue
            compact = re.sub(r'\s+', '', text)
            body_min = 5 if style == 'primary' else 3
            if len(compact) < body_min:
                continue
            if num not in {left_num + 1, right_num - 1}:
                continue
            candidates.append((y, num, style, x, text, item.get('conf', 1.0)))
        if not candidates:
            continue
        candidates.sort(key=lambda value: (value[0], value[1]))
        chosen = next(
            (candidate for candidate in candidates if candidate[1] == left_num + 1),
            candidates[0],
        )
        y, num, style, x, text, confidence = chosen
        recovered.append({
            'num': num, 'style': style, 'text': text,
            'y': int(y_start + y), 'x': int(x),
            'confidence': float(confidence),
        })

    if not recovered:
        return boundaries
    merged = list(boundaries) + recovered
    merged.sort(key=lambda boundary: boundary['y'])
    min_gap = max(28, int(crop_height * 0.015))
    deduped = []
    for candidate in merged:
        if deduped and candidate['y'] - deduped[-1]['y'] < min_gap:
            continue
        deduped.append(candidate)
    return deduped


def _side_by_side_columns(ocr_items, crop_width, crop_height=0):
    """把横向长条里的文字按左缘聚成阅读列，返回每列的 (x0, x1)。

    只统计"窄"文本块，并剔除顶部横跨整块的大题标题：它本来就不属于任何
    一列，参与聚类会把相邻两列粘成一组。

    列区间允许轻微重叠：题号与左列题干在水平方向交叉时（"有（）个平行四边形"
    的末字与下一题题号同一水平带），宁可两张裁图各带一点重复内容，也不要
    把题干或题号切掉。
    """
    width = max(1, int(crop_width))
    height = max(1, int(crop_height or width))
    narrow_limit = int(width * 0.45)
    title_band = height * 0.2
    boxes = []
    for item in ocr_items:
        if not (item.get('text') or '').strip():
            continue
        xs = [float(point[0]) for point in item['bbox']]
        ys = [float(point[1]) for point in item['bbox']]
        left, right = min(xs), max(xs)
        if right - left > narrow_limit:
            continue
        if min(ys) < title_band and right - left > width * 0.2:
            continue  # 顶部横跨的块级标题
        boxes.append((left, right))
    if len(boxes) < 2:
        return []
    boxes.sort()
    min_gap = max(300, int(width * 0.14))
    groups = [[boxes[0]]]
    for box in boxes[1:]:
        previous_left = min(entry[0] for entry in groups[-1])
        if box[0] - previous_left >= min_gap:
            groups.append([box])
        else:
            groups[-1].append(box)
    if len(groups) < 2 or width / len(groups) < 200:
        return []
    pad = 24
    spans = []
    last = len(groups) - 1
    for index, group in enumerate(groups):
        left = min(entry[0] for entry in group)
        right = max(entry[1] for entry in group)
        x0 = 0 if index == 0 else max(0, int(left) - pad)
        x1 = width if index == last else min(width, int(right) + pad)
        if x1 - x0 < 180:
            return []
        spans.append((x0, x1))
    return spans


def _split_side_by_side_minors(major_crop, ocr_items, boundaries, padding, profile):
    """按 x 切开横向并排的小问；证据不足时返回 None 交给原来的按 y 逻辑。

    只处理"明显是横向长条"的块（高 < 宽的 60%），并且必须有横向并排的
    连续题号作为证据，避免把普通密排页切开。
    """
    if len(boundaries) < 2:
        return None
    if not any(boundary.get('side_by_side') for boundary in boundaries):
        return None
    height, width = major_crop.shape[:2]
    if width < 400 or height > width * 0.6:
        return None
    spans = _side_by_side_columns(ocr_items, width, height)
    if not spans:
        return None
    starts = [span[0] for span in spans]

    def column_of(x_position):
        index = 0
        for position, start in enumerate(starts):
            if x_position >= start:
                index = position
        return index

    # 大题标题横跨整块，只应跟第一列在一起；其余列从标题下沿开始，否则
    # 每列顶部都会带一截标题碎片（例如只剩 ")分）"）。
    title_bottom = 0
    for item in ocr_items:
        if not (item.get('text') or '').strip():
            continue
        xs = [float(point[0]) for point in item['bbox']]
        ys = [float(point[1]) for point in item['bbox']]
        if min(ys) < height * 0.2 and max(xs) - min(xs) > width * 0.2:
            title_bottom = max(title_bottom, int(max(ys)))
    if title_bottom and title_bottom > height * 0.6:
        title_bottom = 0

    pieces = []
    for index, (x0, x1) in enumerate(spans):
        top = 0 if index == 0 else min(title_bottom, height - 1)
        local_items = [
            item for item in ocr_items
            if column_of(min(point[0] for point in item['bbox'])) == index
        ]
        sub_cuts = split_major_into_minors(
            major_crop[top:, x0:x1], rebase_ocr_items(local_items, x_offset=x0),
            padding, profile, _depth=1,
        )
        detected = [
            boundary for boundary in boundaries
            if column_of(boundary['x']) == index
        ]
        for cut in sub_cuts:
            cut['_x_start'] = x0
            cut['_x_end'] = x1
            cut['_side_by_side'] = True
            if detected:
                cut['num'] = detected[0]['num']
                cut['number_detected'] = True
                cut['_boundary_source'] = 'ocr_number'
                cut['_number_confidence'] = float(detected[0].get('confidence', 0.0))
            else:
                cut['num'] = 0
            if top:
                cut['_y_start'] = int(cut.get('_y_start', 0)) + top
                cut['_y_end'] = int(cut.get('_y_end', 0)) + top
                cut['_anchor_y'] = int(cut.get('_anchor_y', 0)) + top
        pieces.extend(sub_cuts)

    # 列数必须与题号跨度吻合，否则说明按 x 聚类并不成立。
    numbers = [piece['num'] for piece in pieces]
    known = [number for number in numbers if number]
    if not known:
        return None
    for index in range(len(numbers) - 1, 0, -1):
        if not numbers[index - 1] and numbers[index]:
            numbers[index - 1] = numbers[index] - 1
    for index in range(1, len(numbers)):
        if not numbers[index] and numbers[index - 1]:
            numbers[index] = numbers[index - 1] + 1
    if any(number <= 0 for number in numbers):
        return None
    if max(numbers) - min(numbers) + 1 != len(numbers):
        return None
    for piece, number in zip(pieces, numbers):
        piece['num'] = number
    return pieces


def split_major_into_minors(major_crop, ocr_items=None, padding=4, profile='general',
                            _depth=0):
    """将大题块内按小题编号切分成单题
    
    参数:
        major_crop: 大题块的裁剪图
        ocr_items: 可选，已OCR的结果（复用避免重复OCR）
    返回: [{num, crop_img}]  每个元素是一道小题
    如果没有小题编号，返回整块作为一道题
    """
    # 如果没有传入OCR结果，则重新OCR
    if ocr_items is None:
        try:
            # 直接传内存中的 BGR 数组，避免写临时文件再读回的磁盘往返。
            ocr_items = _run_ocr(major_crop)
            ocr_items.sort(key=lambda it: (
                min(p[1] for p in it['bbox']),
                min(p[0] for p in it['bbox']),
            ))
        except Exception as e:
            print(f"[警告] OCR大题块失败: {e}")
            ocr_items = []
    
    h, w = major_crop.shape[:2]
    
    # 只接受位置、间距均合理的小题号，避免正文数字碎片被误切为题目。
    boundaries = find_minor_boundaries(ocr_items, w, h)
    if not boundaries:
        boundaries = find_unnumbered_boundaries(ocr_items, w, h, profile)
    else:
        # 结构 OCR 偶尔整条漏读小题号，序列会跳号并把相邻题并入同一框。
        # 对跳号条带做原生分辨率重识别，找回缺失边界（无发现则保持现状）。
        # 一次连漏多个号时（如 1→4 缺 2、3），补回一个后仍会跳号，故迭代
        # 至不再有新边界，上限 3 轮，避免罕见情况下失控反复重识别。
        for _recovery_pass in range(3):
            _recovered = _recover_missed_minor_boundaries(major_crop, boundaries)
            if len(_recovered) == len(boundaries):
                break
            boundaries = _recovered
    
    # 横排小问：题号在同一水平线上并排，按 y 一切就只剩一道题。此时改为
    # 按 x 分列再各自递归切分（XY-Cut 的横切步骤）。只在有并排题号证据时
    # 触发，其余页面走原有逻辑，保证零行为变化。
    if boundaries and _depth == 0:
        side_by_side = _split_side_by_side_minors(
            major_crop, ocr_items, boundaries, padding, profile,
        )
        if side_by_side:
            return side_by_side

    if not boundaries:
        # 未识别到编号不等于包含多题：按一个未编号题正常输出。
        return [{
            'num': 1,
            'crop_img': major_crop,
            'number_detected': False,
            '_y_start': 0,
            '_y_end': h,
            '_boundary_source': 'whole_block',
            '_number_confidence': 0.0,
        }]
    
    cuts = []
    # Every boundary searches the same major block.  Compute its cleaned row
    # density once instead of rebuilding the full connected-component mask for
    # every adjacent pair of question numbers.
    content_row_density = None
    if len(boundaries) > 1:
        content_row_density = np.count_nonzero(
            _content_mask(major_crop), axis=1,
        ).astype(np.float32)
    # 紧凑判断题的最后一行可能只有三十多像素；题号已经通过序列校验，
    # 不应再被过高的固定高度门槛静默删除。
    min_crop_height = max(28, int(h * 0.018))
    for i, bd in enumerate(boundaries):
        # 图表、示意图或题干说明常位于第一个 “1.” 之前，属于第 1 小题；
        # 首题从大题顶部开始，后续归一化会去掉纯空白，避免把关联图形裁掉。
        y_start = 0 if i == 0 else max(0, bd['y'] - padding)
        if i + 1 < len(boundaries):
            y_end = _find_clean_horizontal_cut(
                major_crop, bd['y'], boundaries[i + 1]['y'], padding,
                content_row_density=content_row_density,
            )
        else:
            y_end = h
        y_end = min(y_end, h)
        if y_end - y_start < min_crop_height:
            continue
        
        crop = major_crop[y_start:y_end, 0:w]
        if crop.size == 0:
            continue
        
        cuts.append({
            'num': bd['num'],
            'crop_img': crop,
            'number_detected': bd.get('detected', True),
            '_y_start': y_start,
            '_y_end': y_end,
            '_anchor_y': int(bd.get('y', y_start)),
            '_boundary_source': (
                'ocr_number' if bd.get('detected', True) else 'layout_fallback'
            ),
            '_number_confidence': float(bd.get('confidence', 0.0)),
        })
    
    if not cuts:
        cuts.append({
            'num': 1,
            'crop_img': major_crop,
            'number_detected': False,
            '_y_start': 0,
            '_y_end': h,
            '_boundary_source': 'whole_block',
            '_number_confidence': 0.0,
        })
    
    return cuts


def attach_question_source_metadata(
    minor_cuts, block, source_path, source_size, deskew_angle=0.0,
):
    """把局部小题边界映射回原始页面，供评估、标注和后续训练复用。"""
    source_width, source_height = (int(source_size[0]), int(source_size[1]))
    block_x0 = int(block.get('_x_start', 0))
    block_y0 = int(block.get('_y_start', 0))
    block_height, block_width = block['crop_img'].shape[:2]
    block_box = [
        max(0, block_x0),
        max(0, block_y0),
        min(source_width, block_x0 + int(block_width)),
        min(source_height, block_y0 + int(block_height)),
    ]
    for minor in minor_cuts:
        local_y0 = max(0, int(minor.get('_y_start', 0)))
        local_y1 = min(
            int(block_height),
            int(minor.get('_y_end', local_y0 + minor['crop_img'].shape[0])),
        )
        if local_y1 <= local_y0:
            local_y1 = min(int(block_height), local_y0 + minor['crop_img'].shape[0])
        # 横排小问只占块内一段宽度，_source_box 的 x 范围必须跟着收窄，
        # 否则标注工作台与导出会按整块宽再裁一次，把邻题一起带出来。
        local_x0 = max(0, int(minor.get('_x_start', 0)))
        local_x1 = min(int(block_width), int(minor.get('_x_end', block_width)))
        if local_x1 <= local_x0:
            local_x0, local_x1 = 0, int(block_width)
        minor['_source_path'] = os.path.abspath(source_path)
        minor['_source_size'] = [source_width, source_height]
        minor['_source_box'] = [
            min(source_width, block_x0 + local_x0),
            max(0, block_y0 + local_y0),
            min(source_width, block_x0 + local_x1),
            min(source_height, block_y0 + local_y1),
        ]
        minor['_major_box'] = block_box.copy()
        minor['_page_deskew_angle'] = float(deskew_angle or 0.0)
        minor['_col'] = block.get('_col', '')
    return minor_cuts


def attach_shared_context_to_dependent_minors(minors, page_img, *, gap=0):
    """Attach a common preamble/diagram to dependent short numbered subquestions.

    A large region before the first numbered line of a major block is often a
    chart, table, reading passage or shared instruction.  It belongs to every
    following short subquestion, not just to subquestion 1.  This deliberately
    needs several layout signals rather than a title-specific rule: a sizeable
    preamble, a consecutive numbered sequence, and an unusually short later
    item.  The source box remains the item's own box; the extra source segment
    records exactly which common region was composed into its delivered image.
    """
    stats = {'eligible_groups': 0, 'attached': 0}
    if page_img is None or page_img.size == 0 or len(minors) < 2:
        return minors, stats
    ordered = sorted(
        minors,
        key=lambda item: (
            item.get('_source_box', [0, 0, 0, 0])[1],
            item.get('_source_box', [0, 0, 0, 0])[0],
        ),
    )
    first, second = ordered[0], ordered[1]
    major_box = first.get('_major_box')
    if not isinstance(major_box, (list, tuple)) or len(major_box) != 4:
        return minors, stats
    major_x0, major_y0, major_x1, major_y1 = (int(value) for value in major_box)
    major_height = max(1, major_y1 - major_y0)
    first_anchor = int(first.get('_anchor_y', 0))
    context_y1 = major_y0 + max(0, first_anchor - 6)
    context_height = context_y1 - major_y0
    numbers = [item.get('num') for item in ordered]
    sequential = all(
        isinstance(numbers[index], int) and isinstance(numbers[index + 1], int)
        and numbers[index + 1] == numbers[index] + 1
        for index in range(len(numbers) - 1)
    )
    # A normal numbered exercise starts near its first "1.".  Reuse context
    # only when substantial material appears before it, which prevents regular
    # short calculation lists from being expanded unnecessarily.
    if not sequential or context_height < max(120, int(major_height * 0.20)):
        return minors, stats
    context = page_img[major_y0:context_y1, major_x0:major_x1]
    if context.size == 0:
        return minors, stats
    stats['eligible_groups'] = 1
    short_limit = max(150, int(major_height * 0.16))
    source_path = first.get('_source_path', '')
    source_size = list(first.get('_source_size', []))
    context_segment = {
        'image': source_path,
        'size': source_size,
        'question_box': [major_x0, major_y0, major_x1, context_y1],
        'page_deskew_angle': float(first.get('_page_deskew_angle', 0.0)),
    }
    for minor in ordered[1:]:
        box = minor.get('_source_box')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        item_height = int(box[3]) - int(box[1])
        if item_height > short_limit:
            continue
        own = minor.get('crop_img')
        if own is None or own.size == 0:
            continue
        width = max(context.shape[1], own.shape[1])

        def pad(image):
            if image.shape[1] == width:
                return image
            canvas = np.full((image.shape[0], width, 3), 255, dtype=np.uint8)
            canvas[:, :image.shape[1]] = image
            return canvas

        pieces = [pad(context)]
        if gap > 0:
            pieces.append(np.full((gap, width, 3), 255, dtype=np.uint8))
        pieces.append(pad(own))
        minor['crop_img'] = np.vstack(pieces)
        own_segment = {
            'image': source_path,
            'size': source_size,
            'question_box': list(box),
            'page_deskew_angle': float(minor.get('_page_deskew_angle', 0.0)),
        }
        minor['_source_segments'] = [context_segment, own_segment]
        minor['_shared_context_attached'] = True
        minor['_shared_context_reason'] = 'large_preamble_before_consecutive_short_subquestions'
        minor['_shared_context_box'] = list(context_segment['question_box'])
        stats['attached'] += 1
    return minors, stats


def should_merge_cross_page_question(previous, continuation):
    """判断一个无题号顶部片段是否确为上一题的跨页续文。

    这里刻意只接受无可靠题号的片段。下一页可以在续文之后继续出现第 2、3
    题；调用方会先取走这个顶部片段，再让后续题目保持独立，不能因为同页
    有多个小题而放弃真正的跨页拼接。
    """
    if not previous or not continuation or continuation.get('number_detected', True):
        return False
    previous_path = previous.get('_source_path')
    continuation_path = continuation.get('_source_path')
    if not previous_path or not continuation_path:
        return False
    previous_box = previous.get('_source_box', [])
    previous_size = previous.get('_source_size', [])
    continuation_box = continuation.get('_source_box', [])
    continuation_size = continuation.get('_source_size', [])
    if not (
        len(previous_box) == len(continuation_box) == 4
        and len(previous_size) == len(continuation_size) == 2
        and previous_size[1] > 0 and continuation_size[1] > 0
    ):
        return False
    # 裁切框会受页脚、白边、侧栏清理影响，88%/12% 对真实扫描件过于僵硬。
    # 仍要求明显处在阅读边界，避免把同页中段的无编号材料误并进上一题。
    vertical_continuation = (
        previous_box[3] >= previous_size[1] * 0.80
        and continuation_box[1] <= continuation_size[1] * 0.16
    )
    if not vertical_continuation:
        return False
    if previous_path != continuation_path:
        return True
    # A facing-page scan stores left and right pages in one image.  Accept
    # only its reading-order transition: a question ends at the bottom of the
    # left half and continues at the top of the right half.  Same-page blocks
    # elsewhere must remain separate.
    page_width = max(1, previous_size[0])
    split_x = page_width * 0.50
    if (
        previous_box[2] <= split_x * 1.04
        and continuation_box[0] >= split_x * 0.96
    ):
        return True
    # 多栏试卷页：同一张图内按栏序阅读，前栏末尾题会在下一栏顶部继续。
    # 双页对开扫描的 50% 中线判据不适用于三/四栏页，改用栏序 + 横向不相交。
    # 只有确实来自不同栏的顶部无编号片段才合并，普通同栏同页内容不受影响。
    previous_col = previous.get('_col', '')
    continuation_col = continuation.get('_col', '')
    if not (
        str(previous_col).startswith('column_')
        and str(continuation_col).startswith('column_')
        and previous_col != continuation_col
        and previous_box[2] < continuation_box[0]
    ):
        return False
    # 相邻栏顶部的“空白上边距”不是续文，只是单栏留白。只合并确实存在
    # 墨迹的片段，避免把空白带进上一道题；与“宁可少拼接也不错并题干”
    # 的既有保守取向一致。
    continuation_img = continuation.get('crop_img')
    if continuation_img is None or continuation_img.size == 0:
        return False
    gray = cv2.cvtColor(continuation_img, cv2.COLOR_BGR2GRAY)
    ink = int(np.count_nonzero(gray < 200))
    area = max(1, continuation_img.shape[0] * continuation_img.shape[1])
    return ink >= max(12, int(area * 0.001))


def _minor_source_segments(minor):
    segments = minor.get('_source_segments')
    if segments:
        return [dict(segment) for segment in segments]
    return [{
        'image': minor.get('_source_path', ''),
        'size': list(minor.get('_source_size', [])),
        'question_box': list(minor.get('_source_box', [])),
        'page_deskew_angle': float(minor.get('_page_deskew_angle', 0.0)),
    }]


def merge_cross_page_question(previous, continuation):
    """把经保守判定的跨页题纵向拼接，并完整保留两个来源框。"""
    first = previous['crop_img']
    second = continuation['crop_img']
    target_width = max(first.shape[1], second.shape[1])

    def padded(image):
        if image.shape[1] == target_width:
            return image
        canvas = np.full((image.shape[0], target_width, 3), 255, dtype=np.uint8)
        canvas[:, :image.shape[1]] = image
        return canvas

    previous['crop_img'] = np.vstack([padded(first), padded(second)])
    previous['_source_segments'] = (
        _minor_source_segments(previous) + _minor_source_segments(continuation)
    )
    previous['_boundary_source'] = 'cross_page_merge'
    previous['_cross_page_merge'] = True
    # 两页纠斜角可能不同，拼接后不再用单一角度旋转整张长图。
    previous['_page_deskew_angle'] = 0.0
    return previous


def _self_contained_option_question(ocr_items, box):
    """框内是否自包含一道完整的选择题（题干 + 完整 A/B/C/D 选项）。

    多栏延续页的栏顶无编号片段若是独立选择题，说明它是新题而不是上一题
    的续文（题号常因 OCR 漏读而未成为切分边界）。这里只要求出现至少 3 行
    以 A-D 开头的选项；同一选项被 OCR 拆成多行的重复行不计入。
    """
    option_count = 0
    seen_positions = set()
    for item in ocr_items:
        text = re.sub(r'\s+', '', item.get('text', ''))
        if not re.match(r'^[A-D](?:[\.．、)）]|$)', text):
            continue
        xs = [point[0] for point in item['bbox']]
        ys = [point[1] for point in item['bbox']]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        if not (box[0] - 8 <= center_x <= box[2] + 8
                and box[1] - 8 <= center_y <= box[3] + 8):
            continue
        # OCR 常把同一选项识别成相邻两行（y 与 x 几乎重合），按 (y带, x带)
        # 去重后计数；同一行横向排开的 A/B/C/D 各带 x 不同，不会被误去重。
        position = (int(round(center_y / 12)), int(round(center_x / 20)))
        if position in seen_positions:
            continue
        seen_positions.add(position)
        option_count += 1
        if option_count >= 3:
            return True
    return False


def peel_leading_cross_page_fragment(minor, ocr_items=None, option_guard=True):
    """从“续文后紧跟新题”的首题中剥离顶部无编号续文。

    ``split_major_into_minors`` 为了保留图表/题干，会把首题裁切从块顶部开始。
    对延续页而言，首个可信题号之前的较大区域可能属于上一页最后一题。只在
    首题题号不是 1、题号离顶部足够远且来源坐标完整时剥离；其余情况保持原
    裁切，宁可少拼接也不把新题题干错误并入上一题。
    """
    if not minor or not minor.get('number_detected'):
        return None
    if int(minor.get('num', 0) or 0) <= 1:
        return None
    crop = minor.get('crop_img')
    source_box = minor.get('_source_box', [])
    source_size = minor.get('_source_size', [])
    if (
        crop is None or crop.size == 0 or len(source_box) != 4
        or len(source_size) != 2 or source_box[3] <= source_box[1]
    ):
        return None
    crop_height = crop.shape[0]
    anchor_y = int(minor.get('_anchor_y', 0) or 0)
    # 至少保留一段有实际版面高度的续文；题号贴近页顶通常就是新题开始。
    min_fragment_height = max(60, min(150, int(crop_height * 0.30)))
    cut_y = min(crop_height - 28, max(0, anchor_y - 6))
    if cut_y < min_fragment_height:
        return None

    source_height = int(source_box[3]) - int(source_box[1])
    source_cut = int(round(source_height * cut_y / max(1, crop_height)))
    if source_cut <= 0 or source_cut >= source_height:
        return None
    fragment = dict(minor)
    fragment['num'] = 0
    fragment['number_detected'] = False
    fragment['crop_img'] = crop[:cut_y].copy()
    fragment['_source_box'] = [
        int(source_box[0]), int(source_box[1]), int(source_box[2]),
        int(source_box[1]) + source_cut,
    ]
    fragment['_boundary_source'] = 'cross_page_leading_fragment'
    fragment['_number_confidence'] = 0.0
    fragment['_cross_page_fragment'] = True
    # 剥出的区域若自包含一道完整选择题，说明它是栏顶漏掉题号的新题，而不
    # 是上一题的续文。此时保留原裁切（新题内容留在首题里），不把它错并进
    # 上一题。调用方在副本上操作，返回 None 不会改动真实首题。
    if option_guard and ocr_items and _self_contained_option_question(
        ocr_items, fragment['_source_box'],
    ):
        return None

    # 剩余部分仍是原来的新题，保留题号、编号和后续答案配对关系。
    minor['crop_img'] = crop[cut_y:].copy()
    minor['_source_box'] = [
        int(source_box[0]), int(source_box[1]) + source_cut, int(source_box[2]),
        int(source_box[3]),
    ]
    minor['_y_start'] = int(minor.get('_y_start', 0)) + cut_y
    minor['_anchor_y'] = max(0, anchor_y - cut_y)
    minor['_leading_cross_page_fragment_removed'] = True
    return fragment


def merge_cross_page_continuation_prefix(previous, minor_cuts, ocr_items=None):
    """合并延续页顶部续文，返回 ``(是否合并, 合并方式)``。

    先处理本来就无题号的首片段；若首片段包含了下一道已编号题，则谨慎剥离
    题号前的大块续文。列表其余小题绝不参与合并，因此不会吞掉新题。
    """
    if not previous or not minor_cuts:
        return False, ''
    first = minor_cuts[0]
    if should_merge_cross_page_question(previous, first):
        merge_cross_page_question(previous, first)
        del minor_cuts[0]
        return True, 'unnumbered_prefix'

    # 先在副本上试切。若阅读边界证据不足，必须完整保留原首题，不能因为
    # 一次未通过的跨页判断就丢掉它题号前的内容。
    candidate_first = dict(first)
    crop = first.get('crop_img')
    if crop is not None:
        candidate_first['crop_img'] = crop.copy()
    # 题号连续（上一题号 +1 == 首题号）时，栏顶大都是上一题跨栏排版的选项
    # 或解答（如 7→8 的 C/D 选项、4→5 的选项组），属于真实续文；题号不连续
    # 才可能是中间漏了一道独立选择题。选项防护因此只在跳号时启用。
    sequential = (
        int(previous.get('num', 0) or 0) + 1
        == int(first.get('num', 0) or 0)
    )
    fragment = peel_leading_cross_page_fragment(
        candidate_first, ocr_items=ocr_items, option_guard=not sequential,
    )
    if fragment and should_merge_cross_page_question(previous, fragment):
        merge_cross_page_question(previous, fragment)
        first.clear()
        first.update(candidate_first)
        return True, 'peeled_prefix'
    return False, ''


def continuation_target_major(last_left_midx, last_right_midx, page_left_midx, column):
    """Choose the parent major for an untitled continuation block.

    A scanned spread is processed left page then right page.  An untitled
    right page continues the final major on this spread's left page.  The
    *next* scan's untitled left page instead continues the preceding spread's
    right page, which is the physical reading order of facing-page scans.
    """
    if column == 'right' and page_left_midx is not None:
        return page_left_midx
    primary = last_right_midx if column == 'left' else last_right_midx
    secondary = last_left_midx
    return primary if primary is not None else secondary


def _previous_column_key(col):
    """Return the reading-order column key immediately before a multi-column key.

    ``column_2`` reads after ``column_1``, and so on.  Ordinary two-column or
    single-column blocks use ``left``/``right`` and return ``None`` here, so
    the caller falls back to the spread-oriented continuation choice.
    """
    if not isinstance(col, str):
        return None
    prefix, separator, index_text = col.rpartition('_')
    if separator != '_' or prefix != 'column':
        return None
    try:
        column_index = int(index_text)
    except ValueError:
        return None
    if column_index <= 1:
        return None
    return f'column_{column_index - 1}'


def attach_leading_whole_block_to_first_numbered_minor(material, first_minor, gap=0):
    """Attach a preceding unnumbered material block to the real first item."""
    if not material or not first_minor:
        return False
    if material.get('number_detected', True) or material.get('_boundary_source') != 'whole_block':
        return False
    if not first_minor.get('number_detected') or first_minor.get('num') != 1:
        return False
    first_image = material.get('crop_img')
    second_image = first_minor.get('crop_img')
    if first_image is None or second_image is None or first_image.size == 0 or second_image.size == 0:
        return False
    width = max(first_image.shape[1], second_image.shape[1])

    def padded(image):
        if image.shape[1] == width:
            return image
        canvas = np.full((image.shape[0], width, 3), 255, dtype=np.uint8)
        canvas[:, :image.shape[1]] = image
        return canvas

    pieces = [padded(first_image)]
    if gap > 0:
        pieces.append(np.full((gap, width, 3), 255, dtype=np.uint8))
    pieces.append(padded(second_image))
    first_minor['crop_img'] = np.vstack(pieces)
    first_minor['_source_segments'] = (
        _minor_source_segments(material) + _minor_source_segments(first_minor)
    )
    first_minor['_shared_context_attached'] = True
    first_minor['_shared_context_reason'] = 'preceding_cross_column_whole_block'
    first_minor['_shared_context_box'] = list(material.get('_source_box', []))
    return True


# ═══════════════════════════════════════════════════════════════
# 答案处理
# ═══════════════════════════════════════════════════════════════
def group_ocr_rows(ocr_items):
    """把 OCR 文本框按视觉基线合并成行，保留行坐标和组成项。"""
    if not ocr_items:
        return []
    heights = [
        max(point[1] for point in item['bbox']) - min(point[1] for point in item['bbox'])
        for item in ocr_items
    ]
    typical_height = max(8.0, float(np.median(heights)))
    tolerance = max(8.0, typical_height * 0.55)
    ordered = sorted(
        ocr_items,
        key=lambda item: (
            (min(point[1] for point in item['bbox']) + max(point[1] for point in item['bbox'])) / 2,
            min(point[0] for point in item['bbox']),
        ),
    )
    row_groups = []
    for item in ordered:
        center_y = (
            min(point[1] for point in item['bbox'])
            + max(point[1] for point in item['bbox'])
        ) / 2
        if row_groups and abs(center_y - row_groups[-1]['center_y']) <= tolerance:
            row_groups[-1]['items'].append(item)
            centers = row_groups[-1]['centers']
            centers.append(center_y)
            row_groups[-1]['center_y'] = float(np.median(centers))
        else:
            row_groups.append({'items': [item], 'centers': [center_y], 'center_y': center_y})

    rows = []
    for group in row_groups:
        items = sorted(group['items'], key=lambda item: min(point[0] for point in item['bbox']))
        text = ' '.join(item.get('text', '').strip() for item in items if item.get('text', '').strip())
        if not text:
            continue
        rows.append({
            'text': text,
            'items': items,
            'x0': min(min(point[0] for point in item['bbox']) for item in items),
            'x1': max(max(point[0] for point in item['bbox']) for item in items),
            'y0': min(min(point[1] for point in item['bbox']) for item in items),
            'y1': max(max(point[1] for point in item['bbox']) for item in items),
            'height': max(max(point[1] for point in item['bbox']) for item in items)
                      - min(min(point[1] for point in item['bbox']) for item in items),
        })
    return rows


def _answer_context_title(text, typical_height, row_height, allow_lesson=True):
    compact = re.sub(r'\s+', '', text)
    if RE_UNIT_TITLE.match(compact) or PAPER_START_RE.search(compact):
        return compact
    # Answer keys in workbooks frequently restart their minor numbering below
    # short headings such as "知识点二 数轴".  Treating the whole page as one
    # context makes every later "1." look like a duplicate answer and invites
    # incorrect cross-section pairing.  Keep this deliberately as strict as
    # the question-page heading detector: a short line beginning with
    # "知识点" and an explicit ordinal is structural; prose merely mentioning
    # knowledge points is not.
    if (
        len(compact) <= 20
        and re.match(r'^知识点(?:[一二三四五六七八九十]{1,3}|\d{1,3})', compact)
    ):
        return compact
    lesson = LESSON_TITLE_RE.match(text) if allow_lesson else None
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    digit_count = len(re.findall(r'\d', text))
    looks_like_answer = any(symbol in text for symbol in ('=', '×', '√', '+', '改正'))
    if (
        lesson and chinese_count >= 2 and digit_count <= 2 and not looks_like_answer
        and row_height >= typical_height * 0.65
    ):
        return f"{lesson.group(1)}{lesson.group(2).strip()}"
    return ''


def _answer_minor_matches(body):
    """解析答案行里的“1. 2.”锚点；行首1.6按第1题答案6处理。

    OCR 会把高密度选择题串成 ``1.B2.C3.D``。除了行首和空白后的
    编号外，只把紧随选项字母（A-D）后的编号视为下一题，避免把公式、
    小数中的数字泛化成题号。
    """
    matches = []
    for match in re.finditer(
        r'(?:^|\s|(?<=[A-Da-dＡ-Ｄａ-ｄ]))(\d{1,3})\s*[\.．、]', body,
    ):
        number = int(match.group(1))
        if 1 <= number <= 40:
            start = match.start(1)
            matches.append((number, start))
    return matches


def _answer_match_is_decimal_literal(body, minor_num, char_pos):
    """Whether an apparent ``1.`` marker is actually a decimal literal.

    Answer sheets frequently contain rows such as ``1.42 × 100 = 142``.
    Treating the leading ``1.`` as a question number splits one worked answer
    into several tiny, duplicated frames.  A real printed question marker has
    whitespace, text, or an option after its punctuation; only an immediately
    following digit is rejected here.
    """
    start = max(0, int(char_pos))
    end = start + len(str(int(minor_num)))
    while end < len(body) and body[end].isspace():
        end += 1
    if end >= len(body) or body[end] not in '.．':
        return False
    return end + 1 < len(body) and body[end + 1].isdigit()


def _match_signature(text):
    compact = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text or '')
    compact = re.sub(r'第[一二三四五六七八九十百\d]+(?:课时|单元|试卷)', '', compact)
    for word in ('参考答案', '答案', '达标检测', '培优提升', '测试卷', '检测卷', '练习'):
        compact = compact.replace(word, '')
    return compact


def _context_ordinal(text):
    match = re.search(
        r'第\s*([一二三四五六七八九十百\d]+)\s*(?:课时|天|日练|练习)',
        text or '',
    )
    return cn_to_num(match.group(1)) if match else None


def _exam_context_key(text):
    compact = re.sub(r'\s+', '', text or '')
    unit_match = re.search(r'第([一二三四五六七八九十百\d]+)单元', compact)
    if unit_match:
        ordinal = cn_to_num(unit_match.group(1))
        if '培优' in compact or '提升' in compact:
            variant = '培优'
        elif '达标' in compact or '检测' in compact:
            variant = '达标'
        else:
            variant = ''
        return ('unit', ordinal, variant)
    for label in ('期中', '期末', '月考', '模拟', '专项', '综合'):
        if label in compact:
            return (label, compact, '')
    return None


def context_similarity(left, right):
    """课时/试卷标题相似度：包含关系优先，其他情况比较字符二元组。"""
    a, b = _match_signature(left), _match_signature(right)
    left_exam, right_exam = _exam_context_key(left), _exam_context_key(right)
    if left_exam and right_exam:
        if left_exam[0] != right_exam[0] or left_exam[1] != right_exam[1]:
            return 0.0
        if left_exam[2] and right_exam[2] and left_exam[2] != right_exam[2]:
            return 0.0
        return 1.0
    left_ordinal, right_ordinal = _context_ordinal(left), _context_ordinal(right)
    if left_ordinal is not None and right_ordinal is not None:
        if left_ordinal != right_ordinal:
            return 0.0
        if not a or not b:
            return 1.0
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    a_pairs = {a[index:index + 2] for index in range(max(1, len(a) - 1))}
    b_pairs = {b[index:index + 2] for index in range(max(1, len(b) - 1))}
    union = a_pairs | b_pairs
    return len(a_pairs & b_pairs) / len(union) if union else 0.0


def select_answers_for_unit(records, unit_name):
    """从共享答案页中选出与当前课时/试卷上下文最接近的答案段。"""
    contexts = sorted({record.get('context', '') for record in records if record.get('context')})
    if not contexts:
        return list(records)
    scored = [(context_similarity(unit_name, context), context) for context in contexts]
    best_score, best_context = max(scored, default=(0.0, ''))
    if best_score < 0.18:
        return list(records)
    selected = [record for record in records if record.get('context') == best_context]
    return selected or list(records)


def select_answer_context_by_question_sequence(
    records, major_questions, unit_name, unit_position=None,
):
    """标题证据不足时，用整单元题号序列和题数选择一个答案上下文。"""
    records = list(records or [])
    groups = OrderedDict()
    for record in records:
        groups.setdefault(record.get('context', ''), []).append(record)
    if len(groups) <= 1:
        only_context = next(iter(groups), '')
        return records, only_context

    # “第一课时”这类无课名标题无法与答案标题做语义比较。答案上下文按书中
    # 顺序出现时，优先取同位置段；因为证据仍不充分，调用方会强制进入审核。
    if not _match_signature(unit_name):
        try:
            expected_index = int(unit_position) - 1
        except (TypeError, ValueError):
            expected_index = -1
        ordered_groups = list(groups.items())
        if 0 <= expected_index < len(ordered_groups):
            context, context_records = ordered_groups[expected_index]
            return list(context_records), context

    best_records = records
    best_context = ''
    best_rank = None
    for order, (context, context_records) in enumerate(groups.items()):
        by_major = defaultdict(list)
        for record in context_records:
            by_major[record.get('major_idx', 0)].append(record)
        auto_count = 0
        paired_count = 0
        relevant_answer_count = 0
        question_count = 0
        for major_idx, major in major_questions.items():
            questions = major.get('minors', [])
            answers = by_major.get(major_idx, [])
            pairs, _ = review_queue.align_monotonic(questions, answers)
            question_count += len(questions)
            relevant_answer_count += len(answers)
            auto_count += sum(pair['decision_band'] == 'auto' for pair in pairs)
            paired_count += sum(pair.get('answer') is not None for pair in pairs)
        rank = (
            auto_count,
            paired_count,
            -abs(relevant_answer_count - question_count),
            context_similarity(unit_name, context),
            -order,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_records = context_records
            best_context = context
    return list(best_records), best_context


def _compose_answer_continuation_crop(img, primary_box, continuation_boxes, gap=0):
    """Compose an inline answer with wrapped lines that restart at a new X.

    Compact answer keys often print several numbered answers on one line.  A
    long final answer may then wrap back to the left edge on the next line,
    outside its first-line X range.  A rectangular union would include the
    other answers on the first line, so keep the source snippets separate and
    compose them vertically instead.
    """
    height, width = img.shape[:2]
    boxes = [list(primary_box)] + [list(box) for box in continuation_boxes]
    pieces = []
    segments = []
    for index, raw_box in enumerate(boxes):
        x0, y0, x1, y1 = [int(value) for value in raw_box]
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        piece = img[y0:y1, x0:x1]
        if piece.size == 0:
            continue
        pieces.append(piece)
        segments.append({
            'kind': 'answer_primary' if index == 0 else 'answer_continuation',
            'source_box': [x0, y0, x1, y1],
        })
    if not pieces:
        return img[0:0, 0:0], []
    if len(pieces) == 1:
        return pieces[0], segments
    canvas_height = sum(piece.shape[0] for piece in pieces) + gap * (len(pieces) - 1)
    canvas_width = max(piece.shape[1] for piece in pieces)
    if img.ndim == 2:
        canvas = np.full((canvas_height, canvas_width), 255, dtype=img.dtype)
    else:
        canvas = np.full((canvas_height, canvas_width, img.shape[2]), 255, dtype=img.dtype)
    y = 0
    for piece in pieces:
        piece_height, piece_width = piece.shape[:2]
        canvas[y:y + piece_height, :piece_width] = piece
        y += piece_height + gap
    return canvas, segments


def _inline_answer_continuation_rows(anchor, page_rows, page_anchors):
    """Find unnumbered lines immediately following the last inline answer."""
    same_row = [
        other for other in page_anchors
        if other['column_index'] == anchor['column_index']
        and abs(other['y'] - anchor['y']) <= max(10, anchor['row_y1'] - anchor['y'])
    ]
    if len(same_row) < 2 or anchor['x'] != max(other['x'] for other in same_row):
        return []
    row_height = max(10, int(anchor['row_y1'] - anchor['y']))
    candidates = sorted(
        (
            row for row in page_rows
            if row['column_index'] == anchor['column_index']
            and row['y0'] >= anchor['row_y1']
            and not row.get('has_anchor')
            and not row.get('structural')
        ),
        key=lambda row: row['y0'],
    )
    selected = []
    previous_y1 = anchor['row_y1']
    for row in candidates:
        if row['y0'] - previous_y1 > max(28, row_height * 3):
            break
        selected.append(row)
        previous_y1 = row['y1']
        # Only the directly adjacent line is safe.  A second unnumbered line
        # can already belong to the following major answer block.
        if len(selected) >= 1:
            break
    return selected


def _recover_missed_answer_anchors(page_img, page_anchors, page_rows):
    """用局部原图 OCR 补回答案页中明确跳过的题号。

    高密度答案页的降采样 OCR 偶尔会漏掉整行，例如同一知识点内读成
    ``1、2、4``。旧逻辑会让第 2 题的框一直延伸到第 4 题，造成两题合框。
    这里只处理同栏、同上下文且跳过 1--3 个连续编号的情况；仅当局部 OCR
    找到恰好填补该跳号的编号时才插入锚点，宁可保留原框也不凭空拆分。
    """
    if page_img is None or page_img.size == 0 or len(page_anchors) < 2:
        return page_anchors
    height, _ = page_img.shape[:2]
    recovered = []
    grouped = defaultdict(list)
    for anchor in page_anchors:
        grouped[(
            anchor.get('context', ''), anchor.get('major_idx', 0),
            anchor.get('column_index', 0),
        )].append(anchor)

    for anchors in grouped.values():
        anchors.sort(key=lambda anchor: (anchor['y'], anchor['x']))
        for left, right in zip(anchors, anchors[1:]):
            same_visual_row = abs(right['y'] - left['y']) <= max(
                10, left['row_y1'] - left['y'], right['row_y1'] - right['y'],
            )
            missing_count = right['num'] - left['num'] - 1
            if same_visual_row or right['num'] == 1 or not 1 <= missing_count <= 3:
                continue
            y_start = min(height, max(0, int(left['row_y1']) + 3))
            y_end = min(height, max(y_start, int(right['y']) - 3))
            if y_end - y_start < 36:
                continue
            column_x0, column_x1 = int(left['column_x0']), int(left['column_x1'])
            strip = page_img[y_start:y_end, column_x0:column_x1]
            if strip.size == 0:
                continue
            try:
                strip_items = _run_ocr(strip)
            except Exception:
                continue
            expected_numbers = set(range(left['num'] + 1, right['num']))
            candidates = []
            for row in group_ocr_rows(strip_items):
                row_text = row['text'].strip()
                for minor_num, char_pos in _answer_minor_matches(row_text):
                    if minor_num not in expected_numbers:
                        continue
                    confidence = float(np.median([
                        float(item.get('conf', 1.0)) for item in row.get('items', [])
                    ])) if row.get('items') else 0.0
                    if confidence < MIN_OCR_CONFIDENCE:
                        continue
                    relative = char_pos / max(1, len(row_text))
                    candidates.append((
                        int(row['y0']), int(row['x0'] + (row['x1'] - row['x0']) * relative),
                        minor_num, row, char_pos, confidence,
                    ))
            if not candidates:
                continue
            candidates.sort(key=lambda candidate: (candidate[0], candidate[2], candidate[1]))
            row_y, row_x, minor_num, row, char_pos, confidence = candidates[0]
            anchor_y = y_start + row_y
            if any(
                anchor['num'] == minor_num
                and abs(anchor['y'] - anchor_y) < 16
                and anchor['column_index'] == left['column_index']
                for anchor in page_anchors
            ):
                continue
            row_text = row['text'].strip()
            recovered_anchor = {
                'context': left['context'], 'major_idx': left['major_idx'],
                'num': minor_num, 'column_index': left['column_index'],
                'column_x0': left['column_x0'], 'column_x1': left['column_x1'],
                'x': max(left['column_x0'], column_x0 + row_x), 'y': anchor_y,
                'row_y1': y_start + int(row['y1']), 'row_text': row_text,
                'ocr_confidence': confidence, 'char_pos': char_pos,
                'same_row_next_x': None, 'same_row_next_char': None,
                'recovered_by_strip': True,
            }
            recovered.append(recovered_anchor)
            page_rows.append({
                'column_index': left['column_index'],
                'column_x0': left['column_x0'], 'column_x1': left['column_x1'],
                'x0': column_x0 + int(row['x0']), 'x1': column_x0 + int(row['x1']),
                'y0': anchor_y, 'y1': y_start + int(row['y1']),
                'text': row_text, 'has_anchor': True, 'structural': False,
            })

    if not recovered:
        return page_anchors
    RUN_STATS['answer_anchor_strip_recovered_count'] += len(recovered)
    return [*page_anchors, *recovered]


def _next_answer_structure_start(anchor, page_rows):
    """Return the next confirmed section heading in the anchor's own column.

    The final numbered answer before a new ``知识点``/major heading used to
    extend all the way to the next number (or the page bottom), thereby
    swallowing that heading.  Structural rows are produced by the existing,
    intentionally conservative title/major recognisers, so they are safe
    bounds without inventing new OCR evidence.
    """
    starts = [
        int(row['y0']) for row in page_rows
        if row.get('structural')
        and row.get('column_index') == anchor.get('column_index')
        and int(row.get('y0', 0)) > int(anchor.get('row_y1', anchor.get('y', 0)))
    ]
    return min(starts) if starts else None


def _is_circled_answer_matrix_row(row):
    """Return true for a row containing several compact circled answers."""
    text = re.sub(r'\s+', '', str(row.get('text', '') or ''))
    if len(text) > 180:
        return False
    labels = re.findall(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕]', text)
    return len(labels) >= 2


def _unanchored_answer_matrix_sections(page_rows, page_anchors, page_height):
    """Produce review-only boxes for a dense answer column with no anchors.

    A page can contain one ordinary calculation column and one column composed
    entirely of circled answer cells.  The latter has no decimal ``1.`` anchor
    for the normal parser, so previously it disappeared whenever another
    column did produce records.  These boxes deliberately carry no inferred
    question number: they make the missing content visible to the annotator,
    but are excluded from automatic number matching until a person supplies
    the hierarchy.
    """
    by_column = defaultdict(list)
    for row in page_rows:
        by_column[row.get('column_index')].append(row)

    sections = []
    for column_index, rows in by_column.items():
        rows = sorted(rows, key=lambda row: (int(row['y0']), int(row['x0'])))
        matrix_rows = [row for row in rows if _is_circled_answer_matrix_row(row)]
        if len(matrix_rows) < 3:
            continue
        typical_height = float(np.median([max(1, row['y1'] - row['y0']) for row in matrix_rows]))
        gap_limit = max(34, typical_height * 2.8, page_height * 0.025)
        groups = []
        current = []
        for row in matrix_rows:
            if current and int(row['y0']) - int(current[-1]['y1']) > gap_limit:
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)

        for group_index, group in enumerate(groups):
            first, last = group[0], group[-1]
            y0 = int(first['y0']) - 6
            # Retain one immediately-adjacent short heading, but never reach
            # backward across a preceding answer group.
            preceding = [
                row for row in rows
                if int(row['y1']) <= int(first['y0'])
                and int(first['y0']) - int(row['y1']) <= max(28, typical_height * 2.2)
                and not _is_circled_answer_matrix_row(row)
            ]
            if preceding:
                y0 = min(y0, int(preceding[-1]['y0']) - 4)
            next_start = int(groups[group_index + 1][0]['y0']) - 5 if group_index + 1 < len(groups) else None
            if next_start is not None:
                y1 = next_start
            else:
                # The final matrix group may be followed by calculation-style
                # answers without enough circled labels to make another
                # matrix.  It is still the same otherwise-unanchored column;
                # keep that tail reviewable instead of silently losing it.
                tail_rows = [
                    row for row in rows
                    if int(row['y0']) >= int(last['y1'])
                    and not re.fullmatch(r'[—–-]*\d{1,3}[—–-]*', re.sub(r'\s+', '', str(row.get('text', ''))))
                ]
                tail_bottom = max((int(row['y1']) for row in tail_rows), default=int(last['y1']))
                y1 = max(int(last['y1']) + max(8, int(typical_height * 0.7)), tail_bottom + 8)
            x0, x1 = int(first['column_x0']), int(first['column_x1'])
            y0, y1 = max(0, y0), min(page_height, y1)
            # One column can contain a normal numbered answer block above an
            # unnumbered circled matrix below.  Do not discard the latter just
            # because the *column* has an anchor; only an anchor inside this
            # matrix section proves it is already represented by a normal cut.
            section_has_anchor = any(
                anchor.get('column_index') == column_index
                and y0 - 8 <= int(anchor.get('y', -10_000)) <= y1 + 8
                for anchor in page_anchors
            )
            if section_has_anchor:
                continue
            if x1 > x0 and y1 > y0:
                sections.append({
                    'column_index': column_index,
                    'source_box': [x0, y0, x1, y1],
                    'text': ' '.join(str(row.get('text', '')).strip() for row in group),
                })
    return sections


def process_answer_pages(answer_imgs, profile='general', context_hint='', force_refresh=False):
    """用答案专用的行/上下文解析器处理答案页，返回答案记录列表。"""
    if not answer_imgs:
        return []

    records = []
    current_context = ''
    current_major = None
    answer_strategy = layout_strategy.strategy_for(profile)
    for ans_path in answer_imgs:
        img = cv2_imread(ans_path)
        if img is None:
            continue
        page_record_start = len(records)
        ocr_items = ocr_image(
            ans_path, max_side=ANSWER_OCR_MAX_SIDE, force_refresh=force_refresh,
        )
        if needs_high_precision_answer_ocr(ocr_items):
            # 答案页的错漏会直接影响配对；降采样证据不足时宁可慢一点，也不牺牲结果。
            ocr_items = ocr_image(ans_path, force_refresh=force_refresh)
            RUN_STATS['answer_ocr_high_precision_fallback_count'] += 1
        # VLM only supplies coarse titled answer regions.  Re-read those
        # regions locally at full resolution before the established OCR parser
        # derives major/minor numbers and final answer crops.
        ocr_items = refine_answer_ocr_with_vlm(
            ans_path, img, ocr_items, context_hint=context_hint,
        )
        height, width = img.shape[:2]
        columns = split_answer_layout_columns(img, ocr_items, max_columns=4)

        page_anchors = []
        page_rows = []
        for column_index, (column_x0, column_x1, column_items) in enumerate(columns):
            rows = group_ocr_rows(column_items)
            typical_height = float(np.median([row['height'] for row in rows])) if rows else 20.0
            for row in rows:
                row_text = row['text'].strip()
                page_row = {
                    'column_index': column_index,
                    'column_x0': column_x0,
                    'column_x1': column_x1,
                    'x0': row['x0'], 'x1': row['x1'],
                    'y0': row['y0'], 'y1': row['y1'],
                    'text': row_text,
                    'has_anchor': False,
                    'structural': False,
                }
                page_rows.append(page_row)
                row_confidence = float(np.median([
                    float(item.get('conf', 1.0)) for item in row.get('items', [])
                ])) if row.get('items') else 0.0
                holiday_context_row = False
                if answer_strategy.answer_leading_day_number:
                    # 假期作业答案用栏首方框数字表示“第几天/第几课”。
                    # OCR可能漏掉紧随其后的“一、”，所以主要依靠栏首位置识别。
                    holiday_context = re.match(r'^\s*(\d{1,3})(?:\s+|$)(.*)$', row_text)
                    near_column_edge = (
                        row['x0'] <= column_x0 + (column_x1 - column_x0) * 0.13
                    )
                    if holiday_context and near_column_edge:
                        context_number = int(holiday_context.group(1))
                        current_context = f"第{context_number}课时"
                        current_major = None
                        row_text = holiday_context.group(2).strip()
                        holiday_context_row = True
                        if not row_text:
                            continue
                context_title = _answer_context_title(
                    row_text,
                    typical_height,
                    row['height'],
                    allow_lesson=answer_strategy.answer_lesson_context,
                )
                if context_title:
                    page_row['structural'] = True
                    if current_context:
                        current_context = f"{current_context.split(' / ')[0]} / {context_title}"
                    else:
                        current_context = context_title
                    if RE_UNIT_TITLE.match(re.sub(r'\s+', '', context_title)) or PAPER_START_RE.search(context_title):
                        current_context = context_title
                    current_major = None
                    continue

                major_match = re.match(r'^\s*([一二三四五六七八九十百]+)\s*[、,，．.]', row_text)
                body = row_text
                body_offset = 0
                if major_match:
                    page_row['structural'] = True
                    current_major = cn_to_num(major_match.group(1))
                    body_offset = major_match.end()
                    body = row_text[body_offset:].lstrip()
                    body_offset = len(row_text) - len(body)

                minor_matches = [
                    match for match in _answer_minor_matches(body)
                    if not _answer_match_is_decimal_literal(body, *match)
                ]
                if not minor_matches and major_match and current_major is not None:
                    minor_matches = [(1, 0)]
                elif not minor_matches and holiday_context_row and body:
                    current_major = 1
                    minor_matches = [(1, 0)]
                elif not minor_matches and current_major is not None:
                    stripped = body.lstrip()
                    continuation = re.match(r'^(\d{1,3})\s*[\.．、]', stripped)
                    if continuation and not _answer_match_is_decimal_literal(
                        stripped, int(continuation.group(1)), continuation.start(1),
                    ):
                        minor_matches = [(int(continuation.group(1)), len(body) - len(stripped))]
                page_row['has_anchor'] = bool(minor_matches)

                for match_index, (minor_num, char_pos) in enumerate(minor_matches):
                    absolute_pos = body_offset + char_pos
                    relative = absolute_pos / max(1, len(row_text))
                    anchor_x = row['x0'] + (row['x1'] - row['x0']) * relative
                    page_anchors.append({
                        'context': current_context,
                        'major_idx': current_major if current_major is not None else 0,
                        'num': minor_num,
                        'column_index': column_index,
                        'column_x0': column_x0,
                        'column_x1': column_x1,
                        'x': max(column_x0, anchor_x),
                        'y': row['y0'],
                        'row_y1': row['y1'],
                        'row_text': row_text,
                        'ocr_confidence': row_confidence,
                        'char_pos': absolute_pos,
                        'same_row_next_x': None,
                        'same_row_next_char': None,
                    })
                if len(minor_matches) > 1:
                    same_row = page_anchors[-len(minor_matches):]
                    for left_anchor, right_anchor in zip(same_row, same_row[1:]):
                        left_anchor['same_row_next_x'] = right_anchor['x']

        # 补回明确的纵向跳号，避免相邻题在高密度答案页被合成一个框。
        # 最多三轮可处理 1→4 这类连续漏掉多个编号的情况，且每轮只会在
        # 局部 OCR 给出精确编号证据时增加锚点。
        for _recovery_pass in range(3):
            recovered_anchors = _recover_missed_answer_anchors(
                img, page_anchors, page_rows,
            )
            if len(recovered_anchors) == len(page_anchors):
                break
            page_anchors = recovered_anchors

        page_anchors.sort(key=lambda anchor: (
            anchor['column_index'], anchor['y'], anchor['x'],
        ))
        coherent_anchors = []
        last_number_by_group = {}
        for anchor in page_anchors:
            group_key = (anchor.get('context', ''), anchor['major_idx'])
            previous = last_number_by_group.get(group_key)
            number = anchor['num']
            if previous is not None and number > previous + 6 and number != 1:
                continue
            if previous is None or number >= previous or number == 1:
                last_number_by_group[group_key] = number
            coherent_anchors.append(anchor)
        page_anchors = coherent_anchors
        for anchor in page_anchors:
            anchor['same_row_next_x'] = None
            anchor['same_row_next_char'] = None
        for left_anchor, right_anchor in zip(page_anchors, page_anchors[1:]):
            same_visual_row = (
                left_anchor['column_index'] == right_anchor['column_index']
                and abs(left_anchor['y'] - right_anchor['y'])
                <= max(10, left_anchor['row_y1'] - left_anchor['y'])
            )
            if same_visual_row:
                left_anchor['same_row_next_x'] = right_anchor['x']
                if left_anchor['row_text'] == right_anchor['row_text']:
                    left_anchor['same_row_next_char'] = right_anchor['char_pos']

        fallback_sections = _unanchored_answer_matrix_sections(
            page_rows, page_anchors, height,
        )
        for index, anchor in enumerate(page_anchors):
            next_anchor = page_anchors[index + 1] if index + 1 < len(page_anchors) else None
            x0 = max(anchor['column_x0'], int(anchor['x']) - 8)
            if anchor['same_row_next_x'] is not None:
                x1 = min(anchor['column_x1'], int(anchor['same_row_next_x']) - 5)
                y1 = min(height, int(anchor['row_y1']) + 10)
            else:
                x1 = anchor['column_x1']
                if next_anchor and next_anchor['column_index'] == anchor['column_index']:
                    y1 = max(int(anchor['row_y1']) + 5, int(next_anchor['y']) - 5)
                else:
                    y1 = height
            y0 = max(0, int(anchor['y']) - 7)
            section_start = _next_answer_structure_start(anchor, page_rows)
            if section_start is not None:
                y1 = min(y1, max(y0 + 1, section_start - 5))
            if x1 <= x0 or y1 <= y0:
                continue
            continuation_rows = _inline_answer_continuation_rows(
                anchor, page_rows, page_anchors,
            )
            # The normal answer box may already extend down to the next OCR
            # anchor.  Never compose a row that is already inside it, or it
            # would appear twice in the delivered answer image.
            continuation_rows = [
                row for row in continuation_rows if int(row['y1']) > int(y1) + 2
            ]
            continuation_boxes = [
                [
                    max(anchor['column_x0'], int(row['x0']) - 8),
                    max(0, int(row['y0']) - 5),
                    min(anchor['column_x1'], int(row['x1']) + 8),
                    min(height, int(row['y1']) + 6),
                ]
                for row in continuation_rows
            ]
            crop, source_segments = _compose_answer_continuation_crop(
                img, [x0, y0, x1, y1], continuation_boxes,
            )
            if crop.size == 0:
                continue
            answer_text = anchor['row_text']
            if anchor['same_row_next_char'] is not None:
                answer_text = answer_text[
                    anchor['char_pos']:anchor['same_row_next_char']
                ].strip()
            elif anchor['char_pos'] > 0:
                answer_text = answer_text[anchor['char_pos']:].strip()
            if continuation_rows:
                answer_text = ' '.join([
                    answer_text,
                    *(row['text'] for row in continuation_rows),
                ]).strip()
            records.append({
                'context': anchor['context'],
                'major_idx': anchor['major_idx'],
                'num': anchor['num'],
                'crop_img': crop,
                'text': answer_text or anchor['row_text'],
                'source': ans_path,
                'source_size': [width, height],
                'source_box': [x0, y0, x1, y1],
                'source_segments': source_segments,
                'continuation_attached': bool(continuation_rows),
                'column': anchor['column_index'],
                'y': anchor['y'],
                'ocr_confidence': anchor.get('ocr_confidence', 0.0),
            })
        page_records = records[page_record_start:]
        answer_detections = detect_answer_blocks_with_model(img)
        if page_records and answer_detections:
            _, detector_stats = answer_detector.refine_answer_records(
                page_records, img, answer_detections, min_iou=ANSWER_DETECTOR_ASSIST_IOU,
            )
            RUN_STATS['answer_detector_matched_count'] += detector_stats['matched']
            RUN_STATS['answer_detector_refined_count'] += detector_stats['refined']
        for section in fallback_sections:
            x0, y0, x1, y1 = section['source_box']
            records.append({
                'context': '',
                'major_idx': '',
                'num': '',
                'number_detected': False,
                'crop_img': img[y0:y1, x0:x1].copy(),
                'text': section['text'],
                'source': ans_path,
                'source_size': [width, height],
                'source_box': [x0, y0, x1, y1],
                'source_segments': [],
                'continuation_attached': False,
                'column': section['column_index'],
                'y': y0,
                'ocr_confidence': 0.0,
                'unanchored_matrix_fallback': True,
            })
        page_records = records[page_record_start:]
        if not page_records:
            # OCR can occasionally miss every printed number on a sparse answer
            # page.  It is still more useful to give PaperFlow one visible
            # reviewable answer frame than to silently lose the entire page.
            records.append({
                'context': current_context or context_hint,
                'major_idx': 0,
                'num': 0,
                'crop_img': img,
                'text': '',
                'source': ans_path,
                'source_size': [width, height],
                'source_box': [0, 0, width, height],
                'source_segments': [],
                'continuation_attached': False,
                'column': 0,
                'y': 0,
                'ocr_confidence': 0.0,
                'fallback_full_page': True,
            })
    return records


# ═══════════════════════════════════════════════════════════════
# 单元检测
# ═══════════════════════════════════════════════════════════════
def num_to_cn(n):
    """阿拉伯数字转中文数字，支持1-99"""
    cn_map_rev = {v: k for k, v in CN_NUM.items()}
    if n in cn_map_rev:
        return cn_map_rev[n]
    # 手动构建
    digits = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    if n < 10:
        return digits[n] if n > 0 else '零'
    if n == 10:
        return '十'
    tens = n // 10
    ones = n % 10
    result = ''
    if tens > 1:
        result = digits[tens]
    result += '十'
    if ones > 0:
        result += digits[ones]
    return result


def _unit_name_candidates_from_items(ocr_items, unit_num, include_loose_lesson=True):
    """从一页 OCR 结果中提取课时/单元/试卷名称候选。"""
    candidates = []
    for it in ocr_items[:35]:
        text = it.get('text', '').strip()
        lesson_match = LESSON_TITLE_RE.match(text) if include_loose_lesson else None
        if lesson_match:
            lesson_num = int(lesson_match.group(1))
            lesson_title = lesson_match.group(2).strip()
            chinese_count = len(re.findall(r'[\u4e00-\u9fff]', lesson_title))
            looks_like_formula = bool(re.search(r'[=＋+\-−×÷（）()]', text))
            measurement_only = lesson_title[:2] in {
                '分=', '元=', '角=', '厘米', '米=', '时=',
            }
            if (
                chinese_count >= 2
                and not looks_like_formula
                and not measurement_only
                and len(text) <= 42
            ):
                score = 12 if lesson_num == unit_num else 4
                candidates.append((score, f"第{num_to_cn(lesson_num)}课时{lesson_title}"))
        # 优先匹配“第X单元/试卷”类（比课时更优先）
        m = RE_UNIT_TITLE.match(text)
        if m:
            raw_num = m.group(1)
            num_val = cn_to_num(raw_num)
            if num_val is not None:
                cn_name = num_to_cn(num_val)
                suffix = text[m.end():].strip()
                prefix = f"第{cn_name}{m.group(2)}"
                candidates.append((8 if suffix else 6, f"{prefix}{suffix}"))
        # 匹配试卷名（如”第一单元过关检测卷””专项4式与方程”）
        m2 = RE_PAPER_TITLE.match(text)
        if m2:
            candidates.append((10, text.strip()))
        # 匹配广义卷名（如”第一周达标测试卷””期末冲刺夺分卷”），
        # 评分略低于精确单元标记，高于无标题的默认课时名。
        if PAPER_TITLE_RE.fullmatch(text):
            candidates.append((9, text.strip()))
    return candidates


def detect_unit_name(src_dir, unit_num, unit_path, precomputed_ocr=None):
    """检测单元名，策略：
    1. OCR第一页找"第X单元/课时/章节/试卷"字样
    2. OCR第一页找试卷名（如"第一单元过关检测卷"、"专项4式与方程"）
    3. 找到但只有序号（如"第1课时"），统一用中文数字（"第一课时"）
    4. 找到且带标题（如"第3课时十几减8、7（一）"），保留完整标题
    5. 都找不到，默认用"第X课时"
    """
    q_dir = os.path.join(unit_path, '题目')
    if not os.path.isdir(q_dir):
        q_dir = unit_path
    
    # 优先复用启动阶段已经完成的轻量结构 OCR。这里仅用于文件夹命名，
    # 不应再为同一题页额外跑一次高精度 OCR。轻量 OCR 只采纳带“第X课时/
    # 单元/试卷”的正式页眉，不把题干中的“4. ……”误当成课时名称；没有
    # 正式页眉时使用稳定的“第X课时”默认名。
    precomputed_ocr = precomputed_ocr or {}
    # OCR前两页找单元标题（扫描多页以应对双栏首页无标题的情况）
    img_files = sorted(os.listdir(q_dir), key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 999)
    scanned = 0
    candidates = []
    light_scanned_paths = []
    for f in img_files:
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            img_path = os.path.join(q_dir, f)
            if img_path in precomputed_ocr:
                items = precomputed_ocr[img_path]
                light_scanned_paths.append(img_path)
            else:
                items = ocr_image(img_path)
            candidates.extend(_unit_name_candidates_from_items(
                items,
                unit_num,
                include_loose_lesson=img_path not in precomputed_ocr,
            ))
            scanned += 1
            if scanned >= 2:
                break

    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1]

    # 预扫描过的题页已有完整的裁切证据；若未发现正式页眉，不能拿题干中
    # 的小题号猜测课时名，更不能仅为目录名重新 OCR 整页。保留稳定默认名。
    if light_scanned_paths:
        return f"第{num_to_cn(unit_num)}课时"
    
    # 默认：用"第X课时"
    cn_name = num_to_cn(unit_num)
    return f"第{cn_name}课时"


# ═══════════════════════════════════════════════════════════════
# 目录结构发现
# ═══════════════════════════════════════════════════════════════
def detect_printed_page_number(img, ocr_items):
    """读取页脚中独立的小页码；只看底部，避免把题干数字当页码。"""
    height, width = img.shape[:2]
    candidates = []
    for item in ocr_items:
        text = re.sub(r'\s+', '', item.get('text', ''))
        if not re.fullmatch(r'\d{1,3}', text):
            continue
        x0 = min(point[0] for point in item['bbox'])
        x1 = max(point[0] for point in item['bbox'])
        y0 = min(point[1] for point in item['bbox'])
        y1 = max(point[1] for point in item['bbox'])
        center_x = (x0 + x1) / 2
        if y0 < height * 0.90:
            continue
        is_center = width * 0.38 <= center_x <= width * 0.62
        is_outer_corner = center_x <= width * 0.18 or center_x >= width * 0.82
        if not (is_center or is_outer_corner):
            continue
        value = int(text)
        if 0 < value <= 500:
            candidates.append((y1, -abs(center_x - width / 2), value))
    return max(candidates, default=(0, 0, None))[2]


def partition_flat_document(image_paths, profile, announce=False):
    """逐页分类平铺扫描图，并按新试卷标题分成多个逻辑单元。"""
    strategy = layout_strategy.strategy_for(profile)
    entries = []
    answer_imgs = []
    ignored_imgs = []
    in_answer_section = False

    page_total = len(image_paths)
    emit_progress('page_scan', 0, page_total, '正在识别教材页面')
    for index, image_path in enumerate(image_paths):
        img = cv2_imread(image_path)
        if img is None:
            emit_progress(
                'page_scan', index + 1, page_total,
                f'跳过无法读取的页面 {index + 1}/{page_total}',
            )
            continue
        items = ocr_page_structure(image_path, img, profile=profile)
        info = classify_page(img, items, profile)
        if is_likely_low_text_cover(items, info, index, page_total):
            info['kind'] = 'non_content'
        # 题目页的题号证据不足、或中间页完全无法判断时，才回退既有高精度 OCR。
        # 已识别出的封面、说明页只需跳过；答案页会在后续答案解析时完整 OCR 一次。
        needs_full_page = (
            info['kind'] == 'unknown'
            or (info['kind'] == 'question' and needs_high_precision_page_ocr(items, info))
        )
        if needs_full_page:
            items = ocr_image(image_path)
            info = classify_page(img, items, profile)
            ocr_max_side = OCR_MAX_SIDE
            RUN_STATS['ocr_high_precision_fallback_count'] += 1
        else:
            ocr_max_side = OCR_ANCHOR_MAX_SIDE
        info['path'] = image_path
        info['ocr_items'] = items
        info['ocr_max_side'] = ocr_max_side
        info['page_index'] = index
        info['printed_page'] = detect_printed_page_number(img, items)
        # In a flat scan, an ornamental test-paper title is the only reliable
        # boundary between papers.  Ask the VLM before grouping pages, rather
        # than waiting until after all question crops have already been merged.
        if info['kind'] in {'question', 'unknown'} and vlm_hybrid.enabled():
            vlm_analysis = analyze_page_with_vlm(
                img, info, os.path.basename(image_path),
            )
            if vlm_analysis:
                if vlm_analysis.get('_skip_page'):
                    # At this early grouping stage, a local question-page
                    # classification must never be discarded from a single
                    # cloud label.  The later cutter can still use the VLM's
                    # answer-card regions to suppress individual false cuts.
                    info['vlm_skip_suggested'] = vlm_analysis.get('page_type', '')
                else:
                    info['vlm_paper_title'] = vlm_analysis.get('paper_title', '')
                    if not info.get('title') and info['vlm_paper_title']:
                        info['title'] = info['vlm_paper_title']
        emit_progress(
            'page_scan', index + 1, page_total,
            f'识别页面 {index + 1}/{page_total}',
        )

        if info['kind'] == 'answer':
            in_answer_section = True
            answer_imgs.append(image_path)
            continue
        if in_answer_section:
            if info['kind'] in {'non_content', 'instruction'}:
                ignored_imgs.append(image_path)
                continue
            # “参考答案”标题通常只印在第一页，后续密集编号页仍属于答案。
            answer_imgs.append(image_path)
            continue
        if info['kind'] in {'non_content', 'instruction'}:
            ignored_imgs.append(image_path)
            if announce:
                print(f"  跳过{info['kind']}页: {os.path.basename(image_path)}")
            continue
        entries.append(info)

    if strategy.page_per_unit:
        # 假期作业通常每张扫描页就是一份独立日练；合并整本会让重复大题号和答案串单元。
        groups = [[entry] for entry in entries]
    else:
        groups = []
        current = []
        for entry in entries:
            start_reason = paper_start_reason(entry, profile)
            entry['paper_start_reason'] = start_reason
            if start_reason and current:
                if announce:
                    print(
                        f"  识别新试卷边界（{start_reason}）: "
                        f"{os.path.basename(entry['path'])} - {entry.get('title') or '卷首版式'}"
                    )
                groups.append(current)
                current = []
            current.append(entry)
        if current:
            groups.append(current)

    units = []
    for unit_index, group in enumerate(groups, 1):
        title = next((entry['title'] for entry in group if entry.get('title')), '')
        if strategy.numbered_day_units and group:
            # 答案页的方框编号按“第几份日练”排列，通常与印刷页码相差封面/说明页。
            title = f"第{unit_index}课时"
        units.append({
            'unit_num': unit_index,
            'unit_name': '',
            'unit_name_hint': title,
            'question_imgs': [entry['path'] for entry in group],
            'question_ocr': {
                entry['path']: entry.get('ocr_items', [])
                for entry in group
            },
            'question_ocr_max_side': {
                entry['path']: entry.get('ocr_max_side', OCR_MAX_SIDE)
                for entry in group
            },
            'answer_imgs': list(answer_imgs),
            # 平铺教材的封面、说明等页不进入 question_imgs，仍需随预标注项目
            # 写入 ignored_pages；每个逻辑单元携带一份，最终输出阶段会去重。
            'ignored_img_paths': unique_paths(ignored_imgs),
            'answer_source': '逐页分类',
            'unit_path': os.path.dirname(group[0]['path']) if group else '',
            'profile': profile,
        })
    if announce:
        print(f"  页面分类: 题目{sum(len(group) for group in groups)}张, "
              f"答案{len(answer_imgs)}张, 逻辑单元{len(units)}个")
    return units


def collect_units(src_dir, profile=None):
    """收集所有单元，返回 [{unit_num, unit_name, question_imgs, answer_imgs}]"""
    units = []
    profile = profile or resolve_book_profile(src_dir)
    
    # 尝试结构1: src_dir/子目录/数字/题目/
    book_dir = None
    for f in os.listdir(src_dir):
        full = os.path.join(src_dir, f)
        if os.path.isdir(full) and not f.startswith('_'):
            has_digit = any(
                os.path.isdir(os.path.join(full, sf)) and sf.isdigit()
                for sf in os.listdir(full) if not sf.startswith('_')
            )
            if has_digit:
                book_dir = full
                break
    
    if book_dir:
        for f in sorted(os.listdir(book_dir), key=lambda x: int(x) if x.isdigit() else 999):
            unit_path = os.path.join(book_dir, f)
            if not os.path.isdir(unit_path) or not f.isdigit():
                continue
            unit_data = _collect_unit(unit_path, int(f), profile)
            if unit_data['question_imgs']:
                units.append(unit_data)
        if units:
            return units
        print("  两层目录无题目图片，回退到下一级检测")
    
    # 尝试结构2: src_dir/数字/题目/
    has_digit = any(
        os.path.isdir(os.path.join(src_dir, f)) and f.isdigit()
        for f in os.listdir(src_dir)
    )
    if has_digit:
        for f in sorted(os.listdir(src_dir), key=lambda x: int(x) if x.isdigit() else 999):
            unit_path = os.path.join(src_dir, f)
            if not os.path.isdir(unit_path) or not f.isdigit():
                continue
            unit_data = _collect_unit(unit_path, int(f), profile)
            if unit_data['question_imgs']:
                units.append(unit_data)
        # 如果数字子目录里有有效单元，直接返回；否则回退到平铺模式
        if units:
            return units
        print("  数字子目录无题目图片，回退到平铺模式")
    
    # 结构3: 平铺图片，或一个明确的题目/答案子目录
    root_images = list_images(src_dir)
    question_imgs = []
    answer_imgs = []
    answer_sources = []

    for name in sorted(os.listdir(src_dir)):
        full = os.path.join(src_dir, name)
        if not os.path.isdir(full):
            continue
        if name in ANSWER_DIR_NAMES:
            images = list_images(full)
            if images:
                answer_imgs.extend(images)
                answer_sources.append(f'{name}目录')
        elif name in QUESTION_DIR_NAMES:
            question_imgs.extend(list_images(full))

    # 兼容“教材名/图片”的单层包装目录；只在没有任何题目页时启用，避免混入未知目录。
    if not question_imgs:
        for name in sorted(os.listdir(src_dir)):
            full = os.path.join(src_dir, name)
            if not os.path.isdir(full) or name in ANSWER_DIR_NAMES:
                continue
            question_imgs.extend(list_images(full))

    question_imgs = unique_paths(root_images + question_imgs)
    answer_imgs = unique_paths(answer_imgs)
    if question_imgs:
        refined_profile = refine_profile_from_images(profile, question_imgs)
        if refined_profile != profile:
            print(f"  根据页面内容自动切换场景: {profile} -> {refined_profile}")
            profile = refined_profile
        # 平铺教材必须逐页识别试卷/答案边界，不能把整本书强行并成一个单元。
        flat_units = partition_flat_document(question_imgs, profile, announce=True)
        for unit in flat_units:
            if answer_imgs:
                unit['answer_imgs'] = unique_paths(unit['answer_imgs'] + answer_imgs)
                extra = '、'.join(answer_sources) if answer_sources else '答案目录'
                unit['answer_source'] = f"{unit.get('answer_source', '逐页分类')}、{extra}"
        units.extend(flat_units)
    
    return units


def prepare_structured_unit_ocr(units):
    """为“单元/题目”目录预识别题目页，既复用轻量 OCR，也让界面先显示页数进度。"""
    pending = []
    for unit in units:
        cached_items = unit.get('question_ocr', {})
        for image_path in unit.get('question_imgs', []):
            if image_path not in cached_items:
                pending.append((unit, image_path))
    if not pending:
        return

    total = len(pending)
    emit_progress('page_scan', 0, total, '正在预识别题目页面')
    texts_by_unit = defaultdict(list)
    workers = structure_ocr_worker_count()
    RUN_STATS['structure_scan_workers'] = workers
    scan_started = time.perf_counter()
    results = {}

    if workers == 1 or total == 1:
        for index, (unit, image_path) in enumerate(pending, 1):
            img = cv2_imread(image_path)
            profile = unit.get('profile', 'general')
            results[image_path] = ocr_page_structure(
                image_path, img, profile=profile,
            ) if img is not None else []
            emit_progress('page_scan', index, total, f'预识别题目页面 {index}/{total}')
    else:
        # 每页结构 OCR 完全独立；并行只用于这个预扫描阶段。题目裁切、编号、
        # 输出顺序和答案配对仍在主进程按原顺序执行，因此不会改变交付结果。
        future_paths = {}
        failures = []
        completed = 0
        try:
            with ProcessPoolExecutor(
                max_workers=min(workers, total),
                initializer=_init_structure_ocr_worker,
            ) as executor:
                for unit, image_path in pending:
                    profile = unit.get('profile', 'general')
                    future_paths[executor.submit(
                        _structure_ocr_worker, image_path, profile,
                    )] = image_path
                for future in as_completed(future_paths):
                    image_path = future_paths[future]
                    try:
                        returned_path, items, child_stats = future.result()
                        results[returned_path] = items
                        for key, value in child_stats.items():
                            RUN_STATS[key] += value
                    except Exception as exc:
                        failures.append((image_path, exc))
                    completed += 1
                    emit_progress('page_scan', completed, total, f'预识别题目页面 {completed}/{total}')
        except Exception as exc:
            failures.extend((image_path, exc) for _, image_path in pending if image_path not in results)

        # 个别 worker 初始化失败或单页读取异常时，主进程按旧流程补做，保证
        # 并行只是加速项，不会因环境问题漏页或降低精度。
        if failures:
            failed_paths = {image_path for image_path, _ in failures if image_path not in results}
            print(f'  [性能] 结构 OCR 并行有{len(failed_paths)}页未完成，已自动改为单进程补做。')
            for unit, image_path in pending:
                if image_path not in results:
                    img = cv2_imread(image_path)
                    profile = unit.get('profile', 'general')
                    results[image_path] = ocr_page_structure(
                        image_path, img, profile=profile,
                    ) if img is not None else []

    _add_timing('structure_scan_wall', time.perf_counter() - scan_started)
    # 以原始页序写回，保持后续场景判断、题号和输出顺序完全一致。
    for unit, image_path in pending:
        items = results.get(image_path, [])
        unit.setdefault('question_ocr', {})[image_path] = items
        unit.setdefault('question_ocr_max_side', {})[image_path] = OCR_ANCHOR_MAX_SIDE
        texts_by_unit[id(unit)].extend(item.get('text', '') for item in items)

    # 场景细化仅用刚得到的轻量结果，避免为了命名/策略再做一轮整页 OCR。
    for unit in units:
        profile = unit.get('profile', 'general')
        refined = refine_profile_from_text(profile, '\n'.join(texts_by_unit.get(id(unit), [])))
        if refined != profile:
            unit['profile'] = refined


def _collect_unit(unit_path, unit_num, profile='general'):
    question_dir = next(
        (os.path.join(unit_path, name) for name in ('题目', '试卷', '试题', '练习')
         if os.path.isdir(os.path.join(unit_path, name))),
        None,
    )
    source_imgs = list_images(question_dir or unit_path)
    # 页面结构会在真正切题时用轻量 OCR 识别；这里不再额外触发整页高精度 OCR。
    unit_profile = profile
    question_imgs = list(source_imgs)
    answer_imgs = []
    answer_sources = []

    for name in ANSWER_DIR_NAMES:
        images = list_images(os.path.join(unit_path, name))
        if images:
            answer_imgs.extend(images)
            answer_sources.append(f'{name}目录')

    return {
        'unit_num': unit_num,
        'unit_name': '',
        'unit_name_hint': '',
        'question_imgs': unique_paths(question_imgs),
        'answer_imgs': unique_paths(answer_imgs),
        'answer_source': '、'.join(answer_sources) if answer_sources else '未发现',
        'unit_path': unit_path,
        'profile': unit_profile,
    }


# ═══════════════════════════════════════════════════════════════
# 主处理流程
# ═══════════════════════════════════════════════════════════════
def archive_legacy_flat_practice_output(out_dir, units, profile):
    """为“第X练”新分单元规则保留旧版整书合并结果，避免新旧混在一起。

    仅当奥数/思维类平铺书已明确识别出至少两个“第X练”单元，且旧结果
    正好是旧版回退生成的“第一课时”完整目录时触发。采用同盘移动至备份
    目录，不删除任何旧题图；普通教材和真实的单一课时不会受影响。
    """
    if not layout_strategy.strategy_for(profile).legacy_flat_unit_guard:
        return None
    practice_units = [
        unit for unit in units
        if RE_PRACTICE_TITLE.fullmatch(
            re.sub(r'\s+', '', unit.get('unit_name_hint', ''))
        )
    ]
    if len(practice_units) < 2:
        return None
    legacy_dir = os.path.join(out_dir, '第一课时')
    if not os.path.isdir(legacy_dir) or not is_completed_unit(legacy_dir)[0]:
        return None
    backup_root = os.path.join(out_dir, '_旧版单单元结果备份')
    os.makedirs(backup_root, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    destination = os.path.join(backup_root, f'第一课时_旧规则_{stamp}')
    suffix = 1
    while os.path.exists(destination):
        suffix += 1
        destination = os.path.join(
            backup_root, f'第一课时_旧规则_{stamp}_{suffix}',
        )
    try:
        os.replace(legacy_dir, destination)
    except OSError as exc:
        print(f'  [提示] 旧版“第一课时”结果未能自动归档：{exc}')
        return None
    print(f'  已归档旧版合并结果：{destination}')
    return destination


def process_book(src_dir, out_base, skip_unit_paths=None, export_crops=False):
    """处理一本教材。

    skip_unit_paths 仅由批量恢复机制在本次运行内传入；其中的单元已经连续
    两次失败，会被记录并跳过，避免一页异常图片反复拖垮整批任务。
    """
    book_name = os.path.basename(src_dir)
    paperflow_book_root = resolve_paperflow_book_root(src_dir)
    out_dir = os.path.join(out_base, book_name)
    profile = resolve_book_profile(src_dir, paperflow_book_root)
    source_fingerprint = source_book_fingerprint(src_dir)
    paperflow_source_fingerprint = source_book_fingerprint(paperflow_book_root)
    CURRENT_RUN_CONTEXT.update({
        'book_path': os.path.abspath(src_dir),
        'book_name': book_name,
        'unit_path': None,
        'unit_num': None,
        'unit_name': None,
        'unit_position': None,
    })
    completed, _completion_data = is_completed_book(
        paperflow_book_root if not export_crops else src_dir,
        out_dir,
        source_fingerprint if export_crops else paperflow_source_fingerprint,
        output_mode='crops' if export_crops else 'paperflow',
    )
    if completed:
        print(f'\n教材已完成，启动前整本跳过：{book_name}')
        emit_progress('book_skip', 1, 1, '已跳过已完成教材（未进行页面OCR）')
        return
    if export_crops and adopt_legacy_completed_book(src_dir, out_dir, profile, source_fingerprint):
        print(f'\n教材已完成，已兼容旧版结果并整本跳过：{book_name}')
        emit_progress('book_skip', 1, 1, '已跳过旧版已完成教材（未进行页面OCR）')
        return

    reset_run_stats()
    # PaperFlow 预标注数据收集
    units_pf_data = []
    book_started = time.perf_counter()
    skip_unit_paths = {os.path.abspath(path) for path in (skip_unit_paths or set())}
    if export_crops:
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"教材: {book_name}")
    print(f"输出: {out_dir}")
    
    print(f"场景: {profile}")
    units = collect_units(src_dir, profile)
    if export_crops:
        archive_legacy_flat_practice_output(out_dir, units, profile)
    prepare_structured_unit_ocr(units)
    print(f"发现 {len(units)} 个单元\n")
    emit_progress('units', 0, len(units), '已完成页面识别，准备切题')
    
    # 去重答案页（同一文件只OCR一次）
    answer_cache = {}  # 绝对答案页路径 -> 解析记录
    
    if not units:
        print("未发现可处理的内容")
        return
    
    # 处理每个单元
    total_minors = 0
    total_matched = 0
    skipped_incomplete_unit = False
    used_unit_dir_names = set()
    
    for ui, unit in enumerate(units, 1):
        unit_num = unit['unit_num']
        unit_profile = unit.get('profile', profile)
        unmatched_report = []  # 单元级报告，不能串到下一个单元
        source_unit_path = os.path.abspath(unit['unit_path'])
        CURRENT_RUN_CONTEXT.update({
            'unit_path': source_unit_path,
            'unit_num': unit_num,
            'unit_name': None,
            'unit_position': ui,
        })

        if source_unit_path in skip_unit_paths:
            skipped_incomplete_unit = True
            print(f"[{ui}/{len(units)}] 单元 {unit_num} - [跳过] 本次运行中已连续失败两次")
            emit_progress('unit', ui, len(units), f'跳过异常单元 {ui}/{len(units)}')
            CURRENT_RUN_CONTEXT.update({
                'unit_path': None,
                'unit_num': None,
                'unit_name': None,
                'unit_position': None,
            })
            continue
        
        # 检测单元名
        print(f"[{ui}/{len(units)}] 单元 {unit_num} - 检测单元名称...")
        emit_progress('unit', ui - 1, len(units), f'正在处理第 {ui}/{len(units)} 个单元')
        unit_name = unit.get('unit_name_hint') or detect_unit_name(
            src_dir,
            unit_num,
            unit['unit_path'],
            precomputed_ocr=unit.get('question_ocr'),
        )
        if not unit_name:
            unit_name = f"第{num_to_cn(unit_num)}单元"
        unit['unit_name'] = unit_name
        CURRENT_RUN_CONTEXT['unit_name'] = unit_name
        print(f"  单元名: {unit_name}")
        
        # 单元目录
        unit_dir_name = safe_dirname(unit_name, 50)
        if unit_dir_name in used_unit_dir_names:
            # Flat books frequently repeat generic titles such as “测试卷”.
            # Never let a later paper inherit an earlier paper's completion
            # marker or output folder merely because their printed title is
            # identical.
            unit_dir_name = safe_dirname(f'{unit_name}_{unit_num}', 50)
        used_unit_dir_names.add(unit_dir_name)
        final_unit_dir = os.path.join(out_dir, unit_dir_name)
        
        # 裁图模式才需要单元目录、暂存树和历史完成标记；预标注模式只保留内存结果。
        if export_crops:
            if os.path.isdir(final_unit_dir):
                completed, completion_data = is_completed_unit(final_unit_dir)
                if completed:
                    print(f"  跳过（完成标记：{completion_data['question_count']}道小题，"
                          f"答案配对{completion_data.get('matched_answer_count', 0)}）")
                    emit_progress('unit', ui, len(units), f'已跳过已完成单元 {ui}/{len(units)}')
                    CURRENT_RUN_CONTEXT.update({
                        'unit_path': None,
                        'unit_num': None,
                        'unit_name': None,
                        'unit_position': None,
                    })
                    continue
                existing_jpg = sum(1 for _, _, files in os.walk(final_unit_dir)
                                  for f in files if f.lower().endswith('.jpg'))
                if existing_jpg:
                    print(f"  [提示] 检测到{existing_jpg}张旧版/半成品图片，将生成完整新结果后再替换")

            staging_root = os.path.join(out_dir, '_运行中')
            os.makedirs(staging_root, exist_ok=True)
            staging_name = f'{unit_dir_name}.building-{uuid.uuid4().hex[:8]}'
            unit_dir = os.path.join(staging_root, staging_name)
            os.makedirs(unit_dir, exist_ok=True)
        else:
            unit_dir = None
            staging_root = None

        # PaperFlow 页面分类记录
        pf_pages_classification = {}
        pf_answer_img_set = set()
        pf_ignored_img_set = {
            os.path.abspath(path)
            for path in unit.get('ignored_img_paths', [])
        }
        pf_pairs_data = []

        # ── 处理答案 ──
        print(f"  处理答案页...")
        all_answer_records = []
        for ans_path in unit['answer_imgs']:
            pf_answer_img_set.add(os.path.abspath(ans_path))
            try:
                stat = os.stat(ans_path)
                cache_key = (os.path.abspath(ans_path), stat.st_size, stat.st_mtime_ns)
            except OSError:
                cache_key = (os.path.abspath(ans_path), 0, 0)
            if os.environ.get('PAPERCUTTER_VLM_ANSWER_RESCAN', 'off').strip().lower() == 'on':
                # VLM re-OCR is deliberately scoped to the current paper
                # title, so a shared answer spread cannot reuse another
                # paper's focused records.
                cache_key = (*cache_key, unit_name)
            if cache_key in answer_cache:
                all_answer_records.extend(answer_cache[cache_key])
                continue
            
            img = cv2_imread(ans_path)
            if img is None:
                continue
            
            records = process_answer_pages(
                [ans_path], profile=unit_profile, context_hint=unit_name,
            )
            answer_cache[cache_key] = records
            all_answer_records.extend(records)

        selected_answers = select_answers_for_unit(all_answer_records, unit_name)
        answer_blocks = defaultdict(list)
        for record in selected_answers:
            answer_blocks[record['major_idx']].append(record)
        contexts = sorted({record.get('context', '') for record in selected_answers if record.get('context')})
        print(f"  答案块: 大题{len(answer_blocks)}个, 共{len(selected_answers)}道答案")
        if contexts:
            print(f"  答案上下文: {'；'.join(contexts[:3])}")
        print(f"  答案页: {len(unit['answer_imgs'])}张（来源：{unit.get('answer_source', '未记录')}）")
        
        # ── 处理题目 ──
        print(f"  处理题目页（两阶段切分）...")
        t0 = time.time()
        
        # 按大题号收集所有小题（跨页面合并同一大题）
        major_questions = OrderedDict()  # major_idx -> {'title', 'minors': [{num, crop_img}]}
        # 记录左右栏最后出现的大题号（用于延续页归属）
        last_left_midx = None
        last_right_midx = None

        for qi_path in unit['question_imgs']:
            qi_name = os.path.basename(qi_path)
            read_started = time.perf_counter()
            img = cv2_imread(qi_path)
            _add_timing('page_read', time.perf_counter() - read_started)
            if img is None:
                unmatched_report.append({'type': 'read_fail', 'detail': f'无法读取图片: {qi_name}'})
                continue
            
            h, w = img.shape[:2]
            # 扫描倾斜属于整页属性；逐小题重新做霍夫检测会极慢且结果更不稳定。
            deskew_started = time.perf_counter()
            page_deskew_angle = estimate_page_deskew_angle(img) if ENABLE_DESKEW else 0.0
            _add_timing('deskew', time.perf_counter() - deskew_started)
            question_ocr = unit.setdefault('question_ocr', {})
            question_ocr_side = unit.setdefault('question_ocr_max_side', {})
            fast_page_ocr = question_ocr.get(qi_path)
            fast_page_side = question_ocr_side.get(qi_path, OCR_MAX_SIDE)
            if fast_page_ocr is None:
                # 目录式教材此前直接走整页高精度 OCR，是本机耗时的主要来源。
                # 与平铺教材保持同一策略：先读标题和题号带，不确定才回退整页。
                # profile 必须传下去：exam 场景结构 OCR 覆盖整页宽，否则三栏
                # 试卷页的中栏会被“两侧条带抽样”丢弃，导致栏位检测退回单栏。
                fast_page_ocr = ocr_page_structure(qi_path, img, profile=unit_profile)
                fast_page_side = OCR_ANCHOR_MAX_SIDE
                question_ocr[qi_path] = fast_page_ocr
                question_ocr_side[qi_path] = fast_page_side
            used_fast_ocr = fast_page_side < OCR_MAX_SIDE
            ocr_items = fast_page_ocr if used_fast_ocr else ocr_image(qi_path)
            page_info = classify_page(img, ocr_items, unit_profile)
            pf_pages_classification[os.path.abspath(qi_path)] = page_info['kind']
            if used_fast_ocr and needs_high_precision_page_ocr(ocr_items, page_info):
                # 快速结果只要有一点不确定，就完全沿用原来的高精度 OCR 流程。
                ocr_items = ocr_image(qi_path)
                page_info = classify_page(img, ocr_items, unit_profile)
                pf_pages_classification[os.path.abspath(qi_path)] = page_info['kind']
                RUN_STATS['ocr_high_precision_fallback_count'] += 1
                print(f"    {qi_name}: OCR证据不足，已回退高精度识别")
            # The VLM may recognise an answer card even where dense numbered
            # blanks make local OCR look like a question page.  It is only a
            # high-confidence skip gate; normal questions still use the local
            # two-stage splitter and the original answer matcher below.
            vlm_analysis = analyze_page_with_vlm(img, page_info, qi_name)
            if vlm_analysis and vlm_analysis.get('_skip_page'):
                continue
            if page_info['kind'] in {'non_content', 'instruction', 'answer'}:
                print(f"    {qi_name}: 跳过{page_info['kind']}页")
                continue

            # 可训练检测器只提供高置信度候选框；后续仍须通过既有 OCR/规则校验。
            detector_boxes = detect_question_blocks_with_model(img)
            answer_region_boxes = detect_answer_regions_with_model(img)
            
            # 阶段1：按大题切分
            major_split_started = time.perf_counter()
            major_blocks = split_page_into_majors(
                img, ocr_items, profile=unit_profile,
            )
            page_sidebar_exclusions = detect_page_sidebar_exclusions(
                img, major_blocks, ocr_items=ocr_items,
            )
            _add_timing('major_split', time.perf_counter() - major_split_started)
            # For a two-page spread, an untitled right page continues the
            # final major found on this page's left half.
            page_left_midx = None
            # column_N 是当前页内部的栏序；跨页延续由 last_left/right_midx
            # 与 continuation_target_major 决定，不能沿用上一页的栏记录。
            column_last_midx = {}
            
            for mb in major_blocks:
                midx = mb['idx']
                mtitle = mb['title']
                is_fullpage = (mtitle == '整页')
                col = mb.get('_col', 'left')  # 'left' or 'right'

                # “卷面书写”“答题须知”等是试卷说明，不是需要交付的练习题。
                # 在切小题前整体跳过，防止 OCR 数字碎片被拆成大量伪题。
                if not is_fullpage and is_non_question_section(mtitle):
                    print(f"    跳过非题目版块: {mtitle}")
                    continue
                
                # 阶段2：大题块内按小题切分（复用页面OCR结果，避免重复OCR）
                # 只对该大题块区域内的OCR项传入
                local_ocr_items = ocr_items_for_block(ocr_items, mb)
                minor_split_started = time.perf_counter()
                if not local_ocr_items:
                    # 坐标匹配不到，回退到重新OCR
                    minor_cuts = split_major_into_minors(mb['crop_img'], profile=unit_profile)
                else:
                    minor_cuts = split_major_into_minors(
                        mb['crop_img'], ocr_items=local_ocr_items, profile=unit_profile,
                    )
                attach_question_source_metadata(
                    minor_cuts,
                    mb,
                    qi_path,
                    source_size=(w, h),
                    deskew_angle=page_deskew_angle,
                )
                trim_cuts_to_page_sidebar_exclusions(
                    minor_cuts, img, page_sidebar_exclusions,
                )
                trim_cuts_to_detached_page_footer(minor_cuts, img)
                if detector_boxes:
                    minor_cuts, detector_stats = question_detector.refine_legacy_cuts(
                        minor_cuts,
                        img,
                        detector_boxes,
                        min_iou=QUESTION_DETECTOR_ASSIST_IOU,
                    )
                    RUN_STATS['question_detector_matched_count'] += detector_stats['matched']
                    RUN_STATS['question_detector_refined_count'] += detector_stats['refined']
                    # OCR can miss one item number in a dense two-column page.
                    # Permit the trained detector to fill that single gap only
                    # when its box sits inside the preceding merged crop.
                    minor_cuts, recovery_stats = question_detector.recover_sequence_gap_cuts(
                        minor_cuts,
                        img,
                        detector_boxes,
                    )
                    if recovery_stats['recovered']:
                        RUN_STATS['question_detector_refined_count'] += recovery_stats['recovered']
                        print(
                            f"[模型] 题号序列补框 {recovery_stats['recovered']} 道："
                            f"{qi_name}"
                        )
                    if recovery_stats['unresolved_gaps']:
                        unmatched_report.append({
                            'type': 'question_number_gap_review',
                            'detail': (
                                f'{qi_name} 存在 OCR 题号跳号，模型未能安全补框：'
                                f"{recovery_stats['unresolved_gaps']} 处"
                            ),
                        })
                if answer_region_boxes:
                    # The HiLEx class is an inner answer/option region on a
                    # question page.  It cannot create questions or answer
                    # pairs; it only keeps a proven trailing region inside the
                    # current crop without crossing the next OCR question.
                    minor_cuts, answer_region_stats = answer_region_detector.protect_question_cuts(
                        minor_cuts,
                        img,
                        answer_region_boxes,
                        min_confidence=ANSWER_REGION_DETECTOR_ASSIST_CONFIDENCE,
                    )
                    RUN_STATS['answer_region_detector_candidate_count'] += answer_region_stats['candidates']
                    RUN_STATS['answer_region_detector_protected_count'] += answer_region_stats['protected']
                    RUN_STATS['answer_region_detector_cross_next_skip_count'] += answer_region_stats['skipped_cross_next']
                    if answer_region_stats['protected']:
                        print(
                            f"[模型] 题内答案/选项区域保护 {answer_region_stats['protected']} 道："
                            f"{qi_name}"
                        )
                if vlm_analysis:
                    # VLM coordinates cannot invent a question or alter its
                    # OCR number.  They only remove a crop almost wholly
                    # inside an explicitly recognised answer-card region and
                    # attach a primary-group trace for later review.
                    minor_cuts, vlm_cut_stats = vlm_hybrid.apply_to_rule_cuts(
                        minor_cuts, vlm_analysis,
                    )
                    RUN_STATS['vlm_group_confirmed_cut_count'] += vlm_cut_stats['group_confirmed']
                    RUN_STATS['vlm_outside_group_cut_count'] += vlm_cut_stats['outside_groups']
                    RUN_STATS['vlm_answer_card_suppressed_cut_count'] += vlm_cut_stats['answer_card_suppressed']
                    if vlm_cut_stats['answer_card_suppressed']:
                        print(
                            f"[VLM] 已排除答题卡区域中的伪题 {vlm_cut_stats['answer_card_suppressed']} 道："
                            f"{qi_name}"
                        )
                minor_cuts, shared_context_stats = attach_shared_context_to_dependent_minors(
                    minor_cuts, img,
                )
                RUN_STATS['shared_context_group_count'] += shared_context_stats['eligible_groups']
                RUN_STATS['shared_context_attached_count'] += shared_context_stats['attached']
                if shared_context_stats['attached']:
                    print(
                        f"[逻辑] 为连续短小问附加共享图表/题干 {shared_context_stats['attached']} 道："
                        f"{qi_name}"
                    )
                # Detector/context refinements are allowed to expand a box
                # for question content, but never back into a page-level seal
                # strip that was proven from the complete column.  The column
                # bound guard is separate: it only fires when a verified
                # multi-column layout supplied real column bounds, and it can
                # only shrink a box that escaped its column, never delete it.
                trim_cuts_to_column_bounds(
                    minor_cuts, img, mb.get('_column_bounds'),
                )
                trim_cuts_to_page_sidebar_exclusions(
                    minor_cuts, img, page_sidebar_exclusions,
                )
                trim_cuts_to_detached_page_footer(minor_cuts, img)
                _add_timing('minor_split', time.perf_counter() - minor_split_started)
                
                if not is_fullpage:
                    # 有大题编号的块，正常处理
                    if midx not in major_questions:
                        major_questions[midx] = {
                            'title': mtitle,
                            'minors': [],
                        }
                    else:
                        if len(mtitle) > len(major_questions[midx]['title']):
                            major_questions[midx]['title'] = mtitle
                    
                    for mc in minor_cuts:
                        major_questions[midx]['minors'].append(mc)
                    
                    # 更新左右栏最后大题
                    if col == 'left':
                        last_left_midx = midx
                        page_left_midx = midx
                    else:
                        last_right_midx = midx
                    column_last_midx[col] = midx
                
                else:
                    # 整页块（延续页）：归入对应栏的上一个大题
                    previous_col = _previous_column_key(col)
                    if previous_col and previous_col in column_last_midx:
                        # 多栏试卷页在同一张图内按栏序阅读：本栏顶部可能是
                        # 上一栏末尾题的续文，续块归属上一栏的最后一个大题。
                        target_midx = column_last_midx[previous_col]
                    else:
                        target_midx = None
                    if target_midx is None:
                        target_midx = continuation_target_major(
                            last_left_midx, last_right_midx, page_left_midx, col,
                        )
                    if target_midx is None or target_midx not in major_questions:
                        # 缺少大题标题只影响目录归类，不应阻断正常切题输出。
                        target_midx = -1
                        if target_midx not in major_questions:
                            major_questions[target_midx] = {
                                'title': '未分类题目',
                                'minors': [],
                            }
                        unmatched_report.append({
                            'type': 'unclassified_page',
                            'detail': f'延续页无法归属到大题: {qi_name}'
                        })
                    column_last_midx[col] = target_midx
                    target_minors = major_questions[target_midx]['minors']
                    # The left half of a spread may contain only a chart
                    # or data table while its first numbered question is
                    # printed at the top of the right half.  Do not emit
                    # that material as a duplicate "第1题"; compose it
                    # into the actual numbered item instead.
                    if (
                        target_minors and minor_cuts
                        and attach_leading_whole_block_to_first_numbered_minor(
                            target_minors[-1], minor_cuts[0],
                        )
                    ):
                        target_minors.pop()
                    merged_cross_page, merge_mode = merge_cross_page_continuation_prefix(
                        target_minors[-1] if target_minors else None,
                        minor_cuts,
                        ocr_items=ocr_items,
                    )
                    if merged_cross_page:
                        unmatched_report.append({
                            'type': 'cross_page_merge_review',
                            'detail': (
                                f'跨页题顶部续文已拼接并进入审核'
                                f'（{merge_mode}）: {qi_name}'
                            ),
                        })
                    # 即使顶部续文已合并，后续识别出的新题也必须继续输出。
                    for mc in minor_cuts:
                        target_minors.append(mc)
            
            fp_count = sum(1 for b in major_blocks if b['title'] == '整页')
            tp_count = len(major_blocks) - fp_count
            print(f"    {qi_name}: {tp_count}大题 + {fp_count}延续页")
        
        # ── 保存结果 ──
        selected_answers, sequence_context = select_answer_context_by_question_sequence(
            selected_answers, major_questions, unit_name, unit_position=unit_num,
        )
        answer_context_count = len({
            record.get('context', '') for record in all_answer_records
            if record.get('context', '')
        })
        ambiguous_answer_context = (
            answer_context_count > 1
            and context_similarity(unit_name, sequence_context) < 0.18
        )
        if ambiguous_answer_context:
            for major in major_questions.values():
                for minor in major.get('minors', []):
                    minor['_answer_context_ambiguous'] = True
        answer_blocks = defaultdict(list)
        for record in selected_answers:
            answer_blocks[record['major_idx']].append(record)
        if sequence_context:
            print(
                f"  题号序列复核答案上下文: {sequence_context} "
                f"({len(selected_answers)}道候选)"
            )
        unit_minor_count = 0
        unit_matched_count = 0
        unit_review_items = []
        
        for midx, mq in major_questions.items():
            # 大题目录名
            major_dir_name = safe_dirname(mq['title'], 50)
            major_dir = os.path.join(unit_dir, major_dir_name) if export_crops else None
            if export_crops:
                os.makedirs(major_dir, exist_ok=True)
            
            # 该大题对应的答案
            avail_answers = answer_blocks.get(midx, [])
            # 全序列单调对齐允许题目或答案跳过，避免一次漏识别导致后续整段错位。
            pairings, unused_answer_indices = review_queue.align_monotonic(
                mq['minors'], avail_answers,
            )
            
            # 对该大题下的小题统一重新编号（从1开始），避免同号重复
            seq = 0
            for mc in mq['minors']:
                seq += 1
                unit_minor_count += 1

                # 统一留白并去除边缘杂项；不能确定编号/大题归属时也照常输出。
                normalize_started = time.perf_counter()
                normalized_crop, _crop_info = normalize_question_crop(
                    mc['crop_img'],
                    deskew_angle=mc.get('_page_deskew_angle', 0.0),
                )
                RUN_STATS['normalize_count'] += 1
                RUN_STATS['normalize_seconds'] += time.perf_counter() - normalize_started
                # 如果归一化结果不可用（尺寸异常），退回原图；但优先用
                # content_bounds 裁掉已识别的密封线/白边，避免退回全宽原图。
                cb = _crop_info.get('content_bounds') if _crop_info else None
                safe_fallback = mc['crop_img']
                if cb and len(cb) == 4:
                    x0c, y0c, x1c, y1c = (int(v) for v in cb)
                    h, w = safe_fallback.shape[:2]
                    x0c = max(0, min(x0c, w - 1))
                    y0c = max(0, min(y0c, h - 1))
                    x1c = max(x0c + 1, min(x1c, w))
                    y1c = max(y0c + 1, min(y1c, h))
                    safe_fallback = safe_fallback[y0c:y1c, x0c:x1c]

                if normalized_crop is None or normalized_crop.size == 0:
                    normalized_crop = safe_fallback
                    unmatched_report.append({
                        'type': 'crop_normalize_fallback',
                        'detail': f'大题{midx} 第{seq}题使用原始裁剪图',
                    })
                quality_started = time.perf_counter()
                quality_issues = crop_quality_issues(
                    normalized_crop, content_mask=_crop_info.get('content_mask'),
                )
                if quality_issues:
                    raw_issues = crop_quality_issues(safe_fallback)
                    # 归一化把有效内容切成细条时自动退回原图。
                    # 优先用 safe_fallback（已按 content_bounds 裁边），
                    # 避免把密封线带回最终输出。
                    if len(raw_issues) < len(quality_issues):
                        normalized_crop = safe_fallback
                        quality_issues = raw_issues
                    if quality_issues:
                        unmatched_report.append({
                            'type': 'crop_quality_warning',
                            'detail': (
                                f'大题{midx} 第{seq}题: '
                                + '、'.join(quality_issues)
                            ),
                        })
                _add_timing('quality_check', time.perf_counter() - quality_started)

                orig_num = mc.get('num', seq)
                pairing = pairings[seq - 1]
                answer = pairing.get('answer')
                decision_band = pairing['decision_band']

                # 收集 PaperFlow 配对数据
                if not export_crops:
                    pf_q_id = mc.get('_pf_id') or _generate_paperflow_id('Q')
                    mc['_pf_id'] = pf_q_id
                    if answer:
                        pf_a_id = answer.get('_pf_id') or _generate_paperflow_id('A')
                        answer['_pf_id'] = pf_a_id
                        pf_pairs_data.append({
                            'q_id': pf_q_id,
                            'a_ids': [pf_a_id],
                            'decision_band': decision_band,
                            'score': float(pairing.get('score', 0.0)),
                        })
                
                # 小题目录：第X题（用统一编号）
                minor_dir_name = f"第{seq}题"
                minor_dir = os.path.join(major_dir, minor_dir_name) if export_crops else None
                if export_crops:
                    os.makedirs(minor_dir, exist_ok=True)
                
                # 保存题目
                write_started = time.perf_counter()
                if export_crops:
                    question_written = cv2_imwrite(os.path.join(minor_dir, "题目.jpg"), normalized_crop)
                    _add_timing('final_write', time.perf_counter() - write_started)
                else:
                    _add_timing('final_write', time.perf_counter() - write_started)
                    question_written = True  # PaperFlow 模式跳过裁图
                if not question_written:
                    unmatched_report.append({
                        'type': 'write_fail',
                        'detail': f'大题{midx} 第{seq}题题图写入失败',
                    })
                    continue

                relative_minor_dir = (
                    os.path.relpath(minor_dir, unit_dir).replace('\\', '/')
                    if export_crops else ''
                )
                question_relative_path = f'{relative_minor_dir}/题目.jpg' if export_crops else ''
                answer_source = None
                answer_review_data = None
                if answer is not None:
                    answer_source = {
                        'image': os.path.abspath(answer.get('source', '')),
                        'size': list(answer.get('source_size', [])),
                        'answer_box': list(answer.get('source_box', [])),
                        'segments': list(answer.get('source_segments', [])),
                        'continuation_attached': bool(answer.get('continuation_attached', False)),
                        'answer_number': int(answer.get('num', 0)),
                        'ocr_confidence': float(answer.get('ocr_confidence', 0.0)),
                        'text': answer.get('text', ''),
                        'context': answer.get('context', ''),
                    }

                pairing_metadata = {
                    'decision_band': decision_band,
                    'score': float(pairing.get('score', 0.0)),
                    'reasons': list(pairing.get('reasons', [])),
                    'features': dict(pairing.get('features', {})),
                }

                metadata = build_question_metadata(
                    mc,
                    major_idx=midx,
                    major_title=mq['title'],
                    sequence=seq,
                    crop_info=_crop_info,
                    quality_issues=quality_issues,
                    pairing=pairing_metadata,
                    answer_source=answer_source,
                )
                if export_crops and not write_json_atomic(os.path.join(minor_dir, 'metadata.json'), metadata):
                    unmatched_report.append({
                        'type': 'metadata_write_fail',
                        'detail': f'大题{midx} 第{seq}题来源元数据写入失败',
                    })
                
                # 高置信答案进入正式结果；不确定答案只保存为候选，等待审核。
                if answer:
                    answer_filename = '答案.jpg' if decision_band == 'auto' else '候选答案.jpg'
                    write_started = time.perf_counter()
                    if export_crops:
                        answer_written = cv2_imwrite(
                            os.path.join(minor_dir, answer_filename), answer['crop_img'],
                        )
                    else:
                        answer_written = True
                    _add_timing('final_write', time.perf_counter() - write_started)
                    if answer_written:
                        proposed_path = f'{relative_minor_dir}/{answer_filename}'
                        answer_review_data = {
                            'proposed_image': proposed_path,
                            'accepted_image': f'{relative_minor_dir}/答案.jpg',
                            'number': int(answer.get('num', 0)),
                            'text': answer.get('text', ''),
                            'context': answer.get('context', ''),
                            'source_image': os.path.abspath(answer.get('source', '')),
                            'source_size': list(answer.get('source_size', [])),
                            'source_box': list(answer.get('source_box', [])),
                            'ocr_confidence': float(answer.get('ocr_confidence', 0.0)),
                        }
                    else:
                        unmatched_report.append({
                            'type': 'write_fail',
                            'detail': f'大题{midx} 第{seq}题答案候选写入失败',
                        })
                    if answer_written and decision_band == 'auto':
                        unit_matched_count += 1
                    elif answer_written:
                        unmatched_report.append({
                            'type': 'answer_needs_review',
                            'detail': (
                                f'大题{midx} 第{seq}题答案置信度'
                                f"{pairing.get('score', 0.0):.2f}，已保存为候选"
                            ),
                        })
                elif unit['answer_imgs']:
                    # 未配对答案，记录到报告
                    unmatched_report.append({
                        'type': 'unmatched_question',
                        'detail': f'大题{midx}(idx={midx}) 第{seq}题(原号{orig_num}) 无匹配答案'
                    })

                risk_flags = []
                if quality_issues:
                    risk_flags.extend(f'题图质量：{issue}' for issue in quality_issues)
                if not mc.get('number_detected', True):
                    risk_flags.append('题目题号未可靠识别')
                if mc.get('_inferred_number'):
                    risk_flags.append('题号由题块模型按序列补全，需复核')
                if mc.get('_sequence_gap_before'):
                    missing = '、'.join(str(number) for number in mc['_sequence_gap_before'])
                    risk_flags.append(f'题号序列跳过第{missing}题，需复核')
                if mc.get('_cross_page_merge'):
                    risk_flags.append('跨页拼接题需核对接缝与边界')
                if mc.get('_answer_context_ambiguous'):
                    risk_flags.append('单元标题过于笼统，答案上下文仅按顺序选择')
                if mc.get('_answer_region_assisted'):
                    risk_flags.append('题内答案/选项区域已由模型保守补全，建议抽查')
                if mc.get('_shared_context_attached'):
                    risk_flags.append('已附加共享图表/题干，建议抽查上下文范围')
                if decision_band != 'auto':
                    risk_flags.append('答案配对未自动通过')
                if answer is None:
                    risk_flags.append('无答案候选')
                review_id = review_queue.stable_review_id(
                    unit_name, midx, seq, metadata['source'],
                )
                unit_review_items.append({
                    'review_id': review_id,
                    'status': 'auto_accepted' if decision_band == 'auto' else 'pending',
                    'book_name': book_name,
                    'unit_name': unit_name,
                    'major': {
                        'index': int(midx),
                        'title': mq['title'],
                        'sequence': int(seq),
                    },
                    'question': {
                        'image': question_relative_path,
                        'number': int(orig_num),
                        'number_detected': bool(mc.get('number_detected', True)),
                        'number_confidence': float(mc.get('_number_confidence', 0.0)),
                        'source_image': metadata['source']['image'],
                        'source_size': metadata['source']['size'],
                        'source_box': metadata['source']['question_box'],
                        'source_segments': metadata['source'].get('segments', []),
                        'metadata': f'{relative_minor_dir}/metadata.json',
                    },
                    'answer': answer_review_data,
                    'pairing': pairing_metadata,
                    'risk_flags': risk_flags,
                    'review_priority': (
                        (100 if decision_band == 'no_answer' else 70 if decision_band == 'review' else 0)
                        + (15 if quality_issues else 0)
                        + (10 if not mc.get('number_detected', True) else 0)
                        + (20 if mc.get('_cross_page_merge') else 0)
                        + (20 if mc.get('_answer_context_ambiguous') else 0)
                    ),
                })

            for answer_index in unused_answer_indices:
                unused = avail_answers[answer_index]
                unmatched_report.append({
                    'type': 'unmatched_answer',
                    'detail': (
                        f"大题{midx}(idx={midx}) 答案题号{unused.get('num', '?')}"
                        f" 未配对，来源{os.path.basename(unused.get('source', ''))}"
                    ),
                })

        if unit_review_items:
            review_manifest = review_queue.create_review_queue(
                unit_name, unit_review_items, algorithm_version=ALGORITHM_VERSION,
            )
            if export_crops and not write_json_atomic(os.path.join(unit_dir, 'review_queue.json'), review_manifest):
                unmatched_report.append({
                    'type': 'review_queue_write_fail',
                    'detail': '题目—答案审核清单写入失败',
                })
        
        # 裁图模式保留未配对报告；PaperFlow 项目中的未配对题由标注员直接复核。
        if export_crops and unmatched_report:
            report_path = os.path.join(unit_dir, 'unmatched_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                for i, item in enumerate(unmatched_report, 1):
                    f.write(f"{i}. [{item['type']}] {item['detail']}\n")
            unmatched_count = len(unmatched_report)
            print(
                f"  未配对/异常报告: {unmatched_count}项 -> "
                f"{os.path.join(final_unit_dir, 'unmatched_report.txt')}"
            )

        detected_answer_count = len(selected_answers)
        if unit_minor_count:
            if export_crops:
                write_completion_marker(
                    unit_dir,
                    question_count=unit_minor_count,
                    matched_count=unit_matched_count,
                    answer_count=detected_answer_count,
                )
                publish_started = time.perf_counter()
                backup_dir = publish_staged_unit(unit_dir, final_unit_dir)
                _add_timing('publish', time.perf_counter() - publish_started)
                unit_dir = final_unit_dir
                if backup_dir:
                    print(f"  旧结果已保留: {os.path.basename(backup_dir)}")
                try:
                    os.rmdir(staging_root)
                except OSError:
                    pass
        else:
            # 与旧裁图模式一致：一个单元没有产出题框时，不得把整本预标注
            # 发布成 complete，否则下次会因教材指纹相同而永久跳过这本书。
            skipped_incomplete_unit = True
            print("  [警告] 未生成小题，不写完成标记；下次运行将再次处理此单元")
        
        elapsed = time.time() - t0
        total_minors += unit_minor_count
        total_matched += unit_matched_count
        
        # 收集 PaperFlow 单元数据
        if not export_crops:
            pf_scope_id = 'scope-' + uuid.uuid4().hex[:12]
            pf_unit_data = {
                'scope_id': pf_scope_id,
                'unit_name': unit_name,
                'unit_num': unit_num,
                'unit_profile': unit_profile,
                'pages_classification': pf_pages_classification,
                'major_questions': major_questions,
                'answer_records': selected_answers,
                # Keep every OCR-derived answer record too.  The project writer
                # emits records outside the selected context as unassigned
                # answer frames, never as automatic matches.
                'all_answer_records': all_answer_records,
                'answer_img_paths': list(pf_answer_img_set),
                'ignored_img_paths': list(pf_ignored_img_set),
                'pairs_data': pf_pairs_data,
            }
            units_pf_data.append(pf_unit_data)

        match_pct = f"{unit_matched_count/unit_minor_count:.0%}" if unit_minor_count else "N/A"
        print(f"  完成: {unit_minor_count}道小题, 答案配对{unit_matched_count}({match_pct}), {elapsed:.0f}秒")
        emit_progress('unit', ui, len(units), f'已完成第 {ui}/{len(units)} 个单元')
        CURRENT_RUN_CONTEXT.update({
            'unit_path': None,
            'unit_num': None,
            'unit_name': None,
            'unit_position': None,
        })
        sys.stdout.flush()
    
    # 总结
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  教材: {book_name}")
    print(f"  总小题: {total_minors}")
    print(f"  答案配对: {total_matched} ({total_matched/total_minors:.0%})" if total_minors else "  答案配对: 0")
    print(f"  输出: {out_dir}")
    report_run_stats()
    print(f"  总用时: {time.perf_counter() - book_started:.0f}秒")
    if skipped_incomplete_unit:
        print('  [提示] 本书含已跳过的异常单元，未写入整本完成标记。')
    elif export_crops:
        write_book_completion_marker(out_dir, source_fingerprint, len(units))
        print('  已写入整本完成标记；下次将跳过页面识别和切题。')

    # 仅完整处理整本后才发布项目；否则下次不能把缺单元的半成品误判为完成。
    if not export_crops and skipped_incomplete_unit:
        print('  [提示] 本书未完整处理，未发布 PaperFlow 预标注项目；下次会重新处理。')
    elif not export_crops and units_pf_data:
        project = write_paperflow_project(
            paperflow_book_root,
            units_pf_data,
            profile,
            source_fingerprint=paperflow_source_fingerprint,
        )
        if project is None:
            raise RuntimeError('PaperFlow 预标注项目写入失败，本书不会标记为完成。')

    emit_progress('complete', 1, 1, '处理完成')


def write_batch_failure_report(out_dir, failures):
    """把批量运行中的已恢复/已跳过异常写成可追溯报告。"""
    if not failures:
        return None
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(
        out_dir, f'_批量异常报告_{time.strftime("%Y%m%d_%H%M%S")}.txt',
    )
    try:
        with open(report_path, 'w', encoding='utf-8') as file:
            file.write('教材切题工具：批量异常报告\n')
            file.write('说明：单元首次异常会自动重试；连续两次异常才跳过该单元。\n\n')
            for index, failure in enumerate(failures, 1):
                file.write(f"{index}. 处理：{failure['action']}\n")
                file.write(f"   教材：{failure['book_name']}\n")
                if failure.get('unit_name') or failure.get('unit_num') is not None:
                    file.write(
                        f"   单元：{failure.get('unit_name') or '未命名'}"
                        f"（序号 {failure.get('unit_position') or '?'}，"
                        f"原始编号 {failure.get('unit_num') or '?'}）\n"
                    )
                file.write(f"   异常：{failure['error']}\n")
                file.write('   堆栈：\n')
                file.write(failure['traceback'].rstrip() + '\n\n')
    except OSError as exc:
        print(f'[警告] 无法写入批量异常报告: {exc}')
        return None
    print(f'批量异常报告: {report_path}')
    return report_path


def _record_book_failure(failure_attempts, skipped_units_by_book, failures,
                         book_path, book_dir, context, error_desc, trace_text):
    """记录一次批量失败并决定后续动作。

    返回 'retry'（按完成标记重试整本书）或 'skip_book'（跳过这本教材）。
    """
    unit_path = context.get('unit_path')
    unit_key = os.path.abspath(unit_path) if unit_path else None
    failure_key = (book_path, unit_key or '__book__')
    attempt = failure_attempts[failure_key] + 1
    failure_attempts[failure_key] = attempt

    if unit_key and attempt == 1:
        action = '单元首次异常，自动重试'
        decision = 'retry'
    elif unit_key:
        action = '单元连续异常两次，已跳过并继续本书'
        skipped_units_by_book[book_path].add(unit_key)
        decision = 'retry'
    else:
        action = '教材级异常，已跳过并继续下一本'
        decision = 'skip_book'

    failures.append({
        'action': action,
        'book_name': context.get('book_name') or os.path.basename(book_dir),
        'unit_name': context.get('unit_name'),
        'unit_num': context.get('unit_num'),
        'unit_position': context.get('unit_position'),
        'error': error_desc,
        'traceback': trace_text,
    })
    print(f'\n[批量保护] {action}')
    print(f"  原因: {error_desc}")
    if unit_key:
        print(
            f"  单元: {context.get('unit_name') or '未命名'} "
            f"({context.get('unit_position') or '?'}号)"
        )
        print('  正在从完成标记恢复本书，其余已完成单元不会重做。')
    else:
        print('  无法定位到具体单元，已跳过这本教材并继续下一本。')
    return decision


# 并行 worker 共享的进度回传通道。Queue 只能通过进程继承共享（initializer），
# 不能作为 executor.submit 的任务参数 pickle 传递，故放在模块级全局。
_WORKER_PROGRESS_QUEUE = None


def _init_book_worker(progress_queue):
    """ProcessPoolExecutor 的 worker 初始化：继承主进程创建的进度队列。"""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def _process_book_worker(book_dir, out_dir, skip_unit_paths, log_path, export_crops=False):
    """并行子进程任务：完整处理一本书。

    输出同时追加到本次运行日志，并通过继承的进度队列回传主进程，
    由主进程转发到 GUI 进度通道；异常与单元上下文随返回值交回主进程，
    由主进程决定重试/跳过，不会中断其他同时处理的教材。
    """
    original_stdout = sys.stdout
    log_file = None
    if log_path:
        try:
            log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
            # GUI 场景下父进程 stdout 是管道，spawn 后管道句柄已失效，
            # sys.stdout 可能是 None；此时只写日志文件，不回写无效终端。
            sys.stdout = TeeStream(
                original_stdout if original_stdout is not None else None,
                log_file, progress_queue=_WORKER_PROGRESS_QUEUE,
            )
        except (OSError, ValueError):
            log_file = None
    try:
        if export_crops:
            process_book(
                book_dir, out_dir,
                skip_unit_paths=skip_unit_paths,
                export_crops=True,
            )
        else:
            process_book(book_dir, out_dir, skip_unit_paths=skip_unit_paths)
        return {'ok': True, 'book': os.path.abspath(book_dir), 'error': None,
                'traceback': None, 'context': None}
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return {'ok': False, 'book': os.path.abspath(book_dir),
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
                'context': dict(CURRENT_RUN_CONTEXT)}
    finally:
        if log_file is not None:
            sys.stdout = original_stdout
            try:
                log_file.flush()
            except OSError:
                pass
            try:
                log_file.close()
            except OSError:
                pass


def _drain_progress_queue(progress_q):
    """把并行 worker 回传的进度行转发到本进程 stdout（GUI 进度通道）。"""
    if progress_q is None:
        return
    while True:
        try:
            line = progress_q.get_nowait()
        except Exception:
            break
        try:
            print(line, end='', flush=True)
        except Exception:
            break


def process_year(src_year_dir, out_base, export_crops=False):
    """批量处理一个年份下所有教材，单元异常不会终止整批任务。

    默认同时切 2 本教材（环境变量 PAPERCUTTER_BOOK_WORKERS 可调整，设为 1
    即回到完全顺序执行）。每本书在独立进程里跑完全相同的算法，单本输出
    与顺序模式一致，只是多本书的推理在空闲的 CPU 线程上重叠。
    """
    # 直接输出到 out_base，不再额外创建年份子目录（调用方已指定完整路径）
    out_year_dir = out_base

    books = []
    for f in sorted(os.listdir(src_year_dir)):
        full = os.path.join(src_year_dir, f)
        if os.path.isdir(full) and not f.startswith('_'):
            books.append(full)

    print(f"发现 {len(books)} 本教材")

    skipped_units_by_book = defaultdict(set)
    failure_attempts = defaultdict(int)
    failures = []
    log_path = RUN_LOG_PATH or None

    requested = _positive_int_env('PAPERCUTTER_BOOK_WORKERS', 2)
    workers = _book_parallelism(requested, len(books))
    if workers < requested:
        if len(books) < requested:
            print(f"  并行提示：本目录下只有 {len(books)} 本教材，无法并行 {requested} 本，按 {workers} 本处理。"
                  f"（需要多本并行时，把含多本教材的整个年份目录作为一个输入）")
        else:
            print(f"  硬件限制：请求 {requested} 本并行，实际按 {workers} 本处理（CPU/内存不足自动降低）")
    # 同名教材目录会写入同一输出目录，并行会互相覆盖，退化为顺序处理。
    if len({os.path.basename(book) for book in books}) != len(books):
        print('  警告：存在同名教材目录，为保证结果不互相覆盖，本次按顺序处理。')
        workers = 1
    if workers >= 2:
        print(f"并行切题：同时处理 {workers} 本教材（PAPERCUTTER_BOOK_WORKERS 可调整，设为 1 回到顺序模式）")

    if workers == 1:
        # 顺序路径：与旧行为完全一致。
        for i, book_dir in enumerate(books, 1):
            print(f"\n{'#'*60}")
            print(f"教材 [{i}/{len(books)}]: {os.path.basename(book_dir)}")
            book_path = os.path.abspath(book_dir)
            while True:
                try:
                    if export_crops:
                        process_book(
                            book_dir,
                            out_year_dir,
                            skip_unit_paths=skipped_units_by_book[book_path],
                            export_crops=True,
                        )
                    else:
                        process_book(
                            book_dir,
                            out_year_dir,
                            skip_unit_paths=skipped_units_by_book[book_path],
                        )
                    break
                except KeyboardInterrupt:
                    print('\n用户中断任务，停止批量处理。')
                    raise
                except Exception as exc:
                    context = dict(CURRENT_RUN_CONTEXT)
                    decision = _record_book_failure(
                        failure_attempts, skipped_units_by_book, failures,
                        book_path, book_dir, context,
                        f'{type(exc).__name__}: {exc}', traceback.format_exc(),
                    )
                    if decision == 'retry':
                        continue
                    break
    else:
        # 并行路径：每本书一个子进程，异常在各自的结果里独立处理。
        # worker 的 stdout 在 spawn 后可能是 None，进度通过共享 Queue 回传，
        # 由本进程转发到 GUI 管道，保证界面进度条能实时更新。
        mp_context = multiprocessing.get_context('spawn')
        progress_q = mp_context.Queue(maxsize=10000)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                initializer=_init_book_worker,
                initargs=(progress_q,),
            ) as executor:
                pending = {}
                for i, book in enumerate(books, 1):
                    print(f"\n{'#'*60}")
                    print(f"教材 [{i}/{len(books)}]: {os.path.basename(book)}")
                    future = executor.submit(
                        _process_book_worker, book, out_year_dir,
                        frozenset(skipped_units_by_book[os.path.abspath(book)]),
                        log_path,
                        export_crops,
                    )
                    pending[future] = book
                while pending:
                    done, _ = wait(list(pending), timeout=0.5, return_when=FIRST_COMPLETED)
                    _drain_progress_queue(progress_q)
                    for future in done:
                        book = pending.pop(future)
                        book_path = os.path.abspath(book)
                        try:
                            result = future.result()
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            result = {
                                'ok': False, 'book': book_path,
                                'error': f'{type(exc).__name__}: {exc}',
                                'traceback': traceback.format_exc(),
                                'context': dict(CURRENT_RUN_CONTEXT),
                            }
                        if result.get('ok'):
                            print(f"完成教材: {os.path.basename(book)}")
                            continue
                        decision = _record_book_failure(
                            failure_attempts, skipped_units_by_book, failures,
                            book_path, book, result.get('context') or {},
                            result.get('error') or '未知异常',
                            result.get('traceback') or '',
                        )
                        if decision == 'retry':
                            pending[executor.submit(
                                _process_book_worker, book, out_year_dir,
                                frozenset(skipped_units_by_book[book_path]),
                                log_path,
                                export_crops,
                            )] = book
        except KeyboardInterrupt:
            print('\n用户中断任务，停止批量处理。')
            raise
        finally:
            try:
                progress_q.close()
            except Exception:
                pass

    write_batch_failure_report(out_year_dir, failures)
    if failures:
        print(f'批量任务完成：共记录 {len(failures)} 次异常，详见异常报告。')


# ═══════════════════════════════════════════════════════════════
# PaperFlow 预标注项目输出
# ═══════════════════════════════════════════════════════════════

def _infer_book_metadata_from_name(book_name):
    """从教材目录名推断 language/stage/grade/subject。

    目录名示例：
      必胜课冲刺100分二年级下册数学人教版2013__ISBN_9787541655425
      人教版_英语_三年级上册_期末检测卷
    """
    name = str(book_name or '')

    # 学段
    stage = ''
    for s in ('小学', '初中', '高中'):
        if s in name:
            stage = s
            break

    # 年级
    grade = ''
    for key in ('一年级', '二年级', '三年级', '四年级', '五年级', '六年级',
                '七年级', '八年级', '九年级', '初一', '初二', '初三',
                '高一', '高二', '高三'):
        if key in name:
            grade = key
            break

    # 学科
    subject = ''
    for s in ('数学', '语文', '英语', '物理', '化学', '生物', '历史',
              '地理', '政治', '道德与法治', '科学', '美术', '音乐',
              '体育', 'Python', '编程'):
        if s in name:
            subject = s
            break

    # 语言
    language = 'zh'
    if 'english' in name.lower():
        language = 'en'

    return {
        'language': language,
        'stage': stage,
        'grade': grade,
        'subject': subject,
        'answer_mode': 'textbook',
    }


def _generate_paperflow_id(prefix='Q'):
    """生成 PaperFlow 兼容的稳定 ID。"""
    return f'{prefix}-{uuid.uuid4().hex[:12]}'


def _paperflow_answer_record_key(record):
    """Stable physical identity used to keep fallback answer frames unique."""
    if not isinstance(record, dict):
        return None
    source = str(record.get('source', '') or '').strip()
    box = record.get('source_box', [])
    if not source or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        normalized = tuple(int(round(float(value))) for value in box)
    except (TypeError, ValueError):
        return None
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return os.path.abspath(source), normalized


def _paperflow_scope_regions(records):
    """Represent batch scopes as their actual pre-labelled boxes, never a page flag."""
    regions = []
    seen = set()
    for record in records:
        scope_id = str(record.get('match_scope_id', '') or '').strip()
        box = record.get('bbox', [])
        if not scope_id or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            normalized_box = [int(round(float(value))) for value in box]
        except (TypeError, ValueError, OverflowError):
            continue
        if normalized_box[2] <= normalized_box[0] or normalized_box[3] <= normalized_box[1]:
            continue
        key = (scope_id, tuple(normalized_box))
        if key in seen:
            continue
        seen.add(key)
        record_id = str(record.get('id', '') or '')
        regions.append({
            'id': f'region-prelabel-{record_id or len(regions) + 1}',
            'scope_id': scope_id,
            'bbox': normalized_box,
        })
    return regions


def write_paperflow_project(src_dir, units_pf_data, book_profile, source_fingerprint=None):
    """将旧工具处理结果写成 PaperFlow 标注项目 JSON。

    Args:
        src_dir: 教材源目录（JSON 写到这个目录下）
        units_pf_data: list[dict]，每个单元的数据包
        book_profile: 教材的 profile
        source_fingerprint: 本次处理的教材源文件指纹，用于安全跳过重跑
    """
    from collections import defaultdict
    from datetime import datetime, timezone

    root = os.path.abspath(src_dir)
    book_name = os.path.basename(root)
    metadata = _infer_book_metadata_from_name(book_name)

    all_pages = {}
    all_answer_pages = {}
    all_ignored_pages = {}
    all_ignored_page_profiles = {}
    all_match_scopes = []
    all_pairs = []
    emitted_answer_record_keys = set()
    question_primary_by_scope_number = {}
    selected_answer_record_keys = {
        key
        for unit_data in units_pf_data
        for key in (_paperflow_answer_record_key(record)
                    for record in unit_data.get('answer_records', []))
        if key is not None
    }

    for unit_data in units_pf_data:
        scope_id = unit_data['scope_id']
        unit_name = unit_data['unit_name']
        profile = unit_data.get('unit_profile') or book_profile

        now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
        all_match_scopes.append({
            'id': scope_id,
            'name': unit_name,
            'created_at': now_iso,
        })

        pages_classification = unit_data.get('pages_classification', {})
        major_questions = unit_data.get('major_questions', {})
        answer_records = unit_data.get('answer_records', [])
        for ignored_path in unit_data.get('ignored_img_paths', []):
            page_key = os.path.abspath(ignored_path)
            if page_key:
                all_ignored_page_profiles.setdefault(page_key, profile)

        # 题目框按页面分组
        page_questions_map = defaultdict(list)
        for midx, mq in major_questions.items():
            major_title = mq.get('title', '')
            # 提取大题号
            major_num = ''
            m = re.match(
                r'^[（(]?([一二三四五六七八九十百千〇零两]+)[）)]?[、,，.．]'
                r'|^第([一二三四五六七八九十百千〇零两]+|\d{1,3})(?:大题|题|部分|单元|章|节)'
                r'|^[（(]?(\d{1,3})[）)]?[、,，.．]',
                major_title.strip(),
            )
            if m:
                captured = m.group(1) or m.group(2) or m.group(3)
                # 简单转写中文数字
                cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                          '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
                          '十一': '11', '十二': '12', '十三': '13', '十四': '14', '十五': '15',
                          '十六': '16', '十七': '17', '十八': '18', '十九': '19', '二十': '20'}
                major_num = cn_map.get(captured, captured)
            if not major_num and isinstance(midx, int) and midx > 0:
                major_num = str(midx)

            for mc in mq.get('minors', []):
                page_key = os.path.abspath(mc.get('_source_path', ''))
                if not page_key:
                    continue
                box = mc.get('_source_box', [])
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue

                minor_num = str(mc.get('num', '') or '')
                number_path = [major_num] if major_num else []
                if minor_num:
                    number_path.append(minor_num)

                q_id = mc.get('_pf_id') or _generate_paperflow_id('Q')
                mc['_pf_id'] = q_id

                record = {
                    'id': q_id,
                    'bbox': [int(round(v)) for v in box],
                    'number_path': number_path,
                    'display_number': '-'.join(number_path),
                    'major_number': major_num,
                    'major_title': major_title,
                    'minor_number': minor_num,
                    'subsection_number': '',
                    'subsection_title': '',
                    'region_role': 'question',
                    'origin': 'batch_rule',
                    'boundary_source': str(mc.get('_boundary_source', 'rule')),
                    # Keep the model metadata truthful.  OCR's question-number
                    # confidence is recorded separately in review data; it is
                    # not evidence that the YOLO question-box detector moved
                    # this boundary.
                    'detector_assisted': bool(mc.get('_detector_confidence', 0.0)),
                    'detector_confidence': float(mc.get('_detector_confidence', 0.0)),
                    'page_deskew_angle': float(mc.get('_page_deskew_angle', 0.0)),
                    'match_scope_id': scope_id,
                    'number': minor_num,
                }
                # A same-number whole-block that follows a numbered question is
                # a cross-column/page continuation, not another matchable
                # question.  Keep its box but attach it to the primary frame.
                if number_path:
                    number_key = (scope_id, tuple(number_path))
                    primary = question_primary_by_scope_number.get(number_key)
                    if primary is None:
                        question_primary_by_scope_number[number_key] = record
                    else:
                        record['region_role'] = 'continuation'
                        record['continuation_of'] = primary['id']
                page_questions_map[page_key].append(record)

        # 答案框按页面分组
        page_answers_map = defaultdict(list)
        selected_answer_records = list(answer_records)
        fallback_answer_records = []
        for rec in unit_data.get('all_answer_records', []):
            record_key = _paperflow_answer_record_key(rec)
            if record_key is not None and record_key not in selected_answer_record_keys:
                fallback_answer_records.append(rec)
        for rec, is_unassigned in (
            [(record, False) for record in selected_answer_records]
            + [(record, True) for record in fallback_answer_records]
        ):
            page_key = os.path.abspath(rec.get('source', ''))
            if not page_key:
                continue
            box = rec.get('source_box', [])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            record_key = _paperflow_answer_record_key(rec)
            if record_key is not None:
                if record_key in emitted_answer_record_keys:
                    continue
                emitted_answer_record_keys.add(record_key)

            major_idx_str = str(rec.get('major_idx', '') or '')
            minor_num = str(rec.get('num', '') or '')
            number_path = []
            if major_idx_str and major_idx_str != '0':
                number_path.append(major_idx_str)
            if minor_num:
                number_path.append(minor_num)

            a_id = rec.get('_pf_id') or _generate_paperflow_id('A')
            rec['_pf_id'] = a_id

            answer_text = str(rec.get('text', '') or '')
            record = {
                'id': a_id,
                'bbox': [int(round(v)) for v in box],
                'number_path': number_path,
                'display_number': '-'.join(number_path),
                'major_number': major_idx_str,
                'minor_number': minor_num,
                'region_role': 'question',
                'origin': 'batch_unmatched_answer' if is_unassigned else 'batch_rule',
                'answer': answer_text,
                'text': answer_text,
                'ocr_confidence': float(rec.get('ocr_confidence', 0.0)),
                # An out-of-context answer must remain visible but cannot be
                # auto-matched to this unit merely because it shares a page.
                'match_scope_id': '' if is_unassigned else scope_id,
                'unmatched_answer': bool(is_unassigned),
                'number': minor_num,
            }
            page_answers_map[page_key].append(record)

        # 写入 pages / answer_pages / ignored_pages
        for page_abspath, kind in pages_classification.items():
            try:
                rel_path = os.path.relpath(page_abspath, root).replace(chr(92), '/')
            except ValueError:
                continue

            if kind == 'answer':
                answers = page_answers_map.get(page_abspath, [])
                if page_abspath not in all_answer_pages:
                    all_answer_pages[page_abspath] = {
                        'image': rel_path, 'profile': profile,
                        'status': 'accepted' if answers else 'unlabeled',
                        'answers': [], 'match_scope_id': '', 'match_scope_regions': [],
                    }
                all_answer_pages[page_abspath]['answers'].extend(answers)
                if answers:
                    all_answer_pages[page_abspath]['status'] = 'accepted'
            elif kind in ('non_content', 'instruction'):
                if page_abspath not in all_ignored_pages:
                    all_ignored_pages[page_abspath] = {
                        'image': rel_path, 'profile': profile,
                        'status': 'unlabeled', 'questions': [],
                    }
            elif kind == 'question':
                q_list = page_questions_map.get(page_abspath, [])
                if page_abspath not in all_pages:
                    all_pages[page_abspath] = {
                        'image': rel_path, 'profile': profile,
                        'status': 'accepted' if q_list else 'unlabeled',
                        'questions': [], 'match_scope_id': '', 'match_scope_regions': [],
                    }
                all_pages[page_abspath]['questions'].extend(q_list)
                if q_list:
                    all_pages[page_abspath]['status'] = 'accepted'

        # 处理 formal answer_imgs（答案目录里的页面）
        for ans_path in unit_data.get('answer_img_paths', []):
            page_key = os.path.abspath(ans_path)
            if page_key in pages_classification:
                continue
            try:
                rel_path = os.path.relpath(page_key, root).replace(chr(92), '/')
            except ValueError:
                continue
            answers = page_answers_map.get(page_key, [])
            if page_key not in all_answer_pages:
                all_answer_pages[page_key] = {
                    'image': rel_path, 'profile': profile,
                    'status': 'accepted' if answers else 'unlabeled',
                    'answers': [], 'match_scope_id': '', 'match_scope_regions': [],
                }
            all_answer_pages[page_key]['answers'].extend(answers)
            if answers:
                all_answer_pages[page_key]['status'] = 'accepted'

        # 只把高置信自动结果写成已确认关系；其余候选保持未配对，交给标注员复核。
        for pair_item in unit_data.get('pairs_data', []):
            if pair_item.get('decision_band') != 'auto':
                continue
            q_id = pair_item.get('q_id', '')
            a_ids_list = pair_item.get('a_ids', [])
            if not q_id or not a_ids_list:
                continue
            all_pairs.append({
                'id': 'pair-' + uuid.uuid4().hex,
                'question_id': q_id,
                'answer_ids': list(a_ids_list),
                'delivery_id': _generate_paperflow_id('R'),
                'status': 'accepted',
                'source': 'batch_number_match',
                'score': float(pair_item.get('score', 0.0)),
                'updated_at': now_iso,
            })

    # 平铺教材在分单元前已剔除封面、说明等页面。它们不在 question_imgs 中，
    # 所以必须在这里补回 ignored_pages；题目/答案页优先，防止分类回退时重复入册。
    for page_key, profile in all_ignored_page_profiles.items():
        if page_key in all_pages or page_key in all_answer_pages or page_key in all_ignored_pages:
            continue
        try:
            rel_path = os.path.relpath(page_key, root).replace(chr(92), '/')
        except ValueError:
            continue
        all_ignored_pages[page_key] = {
            'image': rel_path,
            'profile': profile,
            'status': 'unlabeled',
            'questions': [],
        }

    # A batch preannotation scope is defined by the emitted question/answer
    # frames.  Do not set a page-wide default here: doing so makes the main
    # workbench display an answer sheet as one huge group and also lets a later
    # unassigned fallback answer inherit the wrong scope.
    for page in list(all_pages.values()) + list(all_answer_pages.values()):
        records = page.get('answers', page.get('questions', []))
        page['match_scope_regions'] = _paperflow_scope_regions(records)

    # 去重 scope
    seen_ids = set()
    unique_scopes = []
    for s in all_match_scopes:
        if s['id'] not in seen_ids:
            seen_ids.add(s['id'])
            unique_scopes.append(s)

    answer_image_root = root if all_answer_pages else ''
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')

    project = {
        'schema_version': 1,
        'kind': 'papercutter_manual_annotation',
        'preannotation_mode': 'paperflow',
        'status': 'complete',
        'algorithm_version': ALGORITHM_VERSION,
        'source_fingerprint': source_fingerprint or source_book_fingerprint(root),
        'annotation_version': 5,
        'created_at': now_iso,
        'updated_at': now_iso,
        'coordinate_format': 'xyxy_pixels',
        'book_root': root,
        'image_root': root,
        'answer_image_root': answer_image_root,
        'pages': sorted(all_pages.values(), key=lambda p: p['image']),
        'answer_pages': sorted(all_answer_pages.values(), key=lambda p: p['image']),
        'ignored_pages': sorted(all_ignored_pages.values(), key=lambda p: p['image']),
        'match_scopes': unique_scopes,
        'pairs': all_pairs,
        'language': metadata['language'],
        'stage': metadata['stage'],
        'grade': metadata['grade'],
        'subject': metadata['subject'],
        'answer_mode': metadata['answer_mode'],
    }

    # 写入文件
    project_path = os.path.join(root, 'PaperFlow标注项目.json')
    temp_path = project_path + '.tmp'
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, project_path)
        print(f'  PaperFlow预标注项目已写入：{project_path}')
        print(f'    页面: {len(project["pages"])}, 答案页: {len(project["answer_pages"])}, '
              f'忽略页: {len(project["ignored_pages"])}')
        print(f'    分组数: {len(project["match_scopes"])}, 自动配对: {len(project["pairs"])}')
    except (OSError, TypeError, ValueError) as exc:
        print(f'  [警告] PaperFlow项目写入失败: {exc}')
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return None

    return project


# ═══════════════════════════════════════════════════════════════
# 命令行
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='切题脚本 v4 - 按小题切分')
    parser.add_argument('src', help='输入目录')
    parser.add_argument(
        '--out',
        default=r'G:\img-to-formatted-docx',
        help='输出根目录；单本模式会在其下自动建立教材同名目录',
    )
    parser.add_argument('--mode', choices=['book', 'year'], default='book')
    parser.add_argument(
        '--vlm-assist', action='store_true',
        help='启用 Qwen 页面语义辅助：只排除答题卡/非题页并标注题组，不替代 OCR 切题或答案配对。API 密钥只从 DASHSCOPE_API_KEY 读取。',
    )
    parser.add_argument(
        '--vlm-answer-rescan', action='store_true',
        help='实验：让 Qwen 定位参考答案上下文区，再仅对该区域作高分辨率本地 OCR；不会直接采用模型坐标裁答案。',
    )
    parser.add_argument('--vlm-model', help='VLM 模型名；默认 qwen3-vl-plus')
    parser.add_argument(
        '--vlm-thinking', action='store_true',
        help='启用 Qwen3-VL 推理模式；模型会先进行视觉推理，再返回最终结构化判断。',
    )
    parser.add_argument('--export-crops', action='store_true',
                        help='输出裁切图片（默认：输出 PaperFlow 预标注框 JSON）')
    parser.add_argument('--log-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
                        help='运行日志目录（默认：脚本目录下的 logs）')
    args = parser.parse_args()

    if args.vlm_assist:
        os.environ['PAPERCUTTER_VLM_MODE'] = 'assist'
    if args.vlm_answer_rescan:
        os.environ['PAPERCUTTER_VLM_MODE'] = 'assist'
        os.environ['PAPERCUTTER_VLM_ANSWER_RESCAN'] = 'on'
    if args.vlm_model:
        os.environ['PAPERCUTTER_VLM_MODEL'] = args.vlm_model
    if args.vlm_thinking:
        os.environ['PAPERCUTTER_VLM_THINKING'] = 'on'

    log_file, original_stdout, original_stderr = start_run_log(args.log_dir)
    try:
        if args.mode == 'year':
            process_year(args.src, args.out, export_crops=args.export_crops)
        else:
            process_book(args.src, args.out, export_crops=args.export_crops)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_file.close()


if __name__ == '__main__':
    main()
