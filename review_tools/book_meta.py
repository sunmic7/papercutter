# -*- coding: utf-8 -*-
"""书名解析器（G0b）与书级信息版本校验（D6）。

两个职责
--------
1. ``guess_book_metadata(title)``：从书名/目录名猜出 ``language/stage/grade/subject/
   semester/edition``，**只做候选，绝不替人拍板**。猜不出来就把字段留空并写
   ``warnings``，让 UI 强制人工确认 —— 猜错的代价是整本书的元数据一致地错，
   比"留空等人填"贵得多。所以 "专项训练" 这种没有年级信息的名字，``grade``
   必须是空字符串且 ``warnings`` 非空。
2. ``validate_book_metadata_revision(meta, expected_rev)``：算
   ``book_meta_rev = sha256(language|stage|grade|subject|difficulty_scheme)``，
   与工程里存的旧值比对。变了就是 ``dirty`` —— 快照失效、不允许出包（D6）。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# 书级字段值域（交付校验与 UI 下拉必须同源）
STAGES = ('小学', '初中', '高中')
GRADE_NAMES = {
    1: '一年级', 2: '二年级', 3: '三年级', 4: '四年级', 5: '五年级', 6: '六年级',
    7: '七年级', 8: '八年级', 9: '九年级', 10: '高一', 11: '高二', 12: '高三',
}
STAGE_GRADES = {
    '小学': (1, 2, 3, 4, 5, 6),
    '初中': (7, 8, 9),
    '高中': (10, 11, 12),
}
SUBJECTS = (
    '数学', '语文', '英语', '物理', '化学', '生物', '历史', '地理',
    '道德与法治', '科学', '信息技术', '政治',
)
SEMESTERS = ('上册', '下册', '全一册')
EDITIONS = (
    '人教版', '北师大版', '苏教版', '沪教版', '外研版', '浙教版', '冀教版',
    '青岛版', '西师大版', '北京版', '湘教版', '鄂教版', '教科版', '译林版',
    '闽教版', '通用版',
)
DIFFICULTY_SCHEMES = ('1-5', '1-3')
LANGUAGES = ('zh', 'en')

# 非教材类书籍：有年级也能用，但缺年级时不能猜
NON_TEXTBOOK_TYPES = ('专项训练', '试卷', '练习册', '同步练习', '寒暑假作业', '口算题卡')
BOOK_TYPES = ('教材',) + NON_TEXTBOOK_TYPES

META_KEYS = ('language', 'stage', 'grade', 'subject', 'difficulty_scheme')

_CN_GRADE_RE = re.compile(r'(?<![初高])([一二三四五六七八九])\s*年级')
_AR_GRADE_RE = re.compile(r'(?<!\d)([1-9])\s*年级')
_JUNIOR_RE = re.compile(r'初\s*([一二三])')
_SENIOR_RE = re.compile(r'高\s*([一二三])')
_SEMESTER_RE = re.compile(r'(上|下)\s*册|第\s*([一二])\s*学期|(全)\s*一册')
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
_LATIN_RE = re.compile(r'[A-Za-z]')


def _clean_title(title: str) -> str:
    value = unicodedata.normalize('NFKC', str(title or ''))
    value = re.sub(r'\.(pdf|jpg|jpeg|png|zip|epub)$', '', value, flags=re.IGNORECASE)
    value = re.sub(r'[\\/]+', ' ', value)
    return value.strip()


def _first_present(text: str, candidates) -> str:
    for candidate in candidates:
        if candidate and candidate in text:
            return candidate
    return ''


def guess_book_metadata(title: str, *, difficulty_scheme: str = '1-5') -> dict:
    """从书名猜书级信息。返回字段全部可能为空字符串，空即是"需要人确认"。"""
    raw = str(title or '')
    text = _clean_title(raw)
    warnings: list[str] = []

    # 年级 / 学段
    grade_index = 0
    match = _JUNIOR_RE.search(text)
    if match:
        grade_index = {'一': 7, '二': 8, '三': 9}[match.group(1)]
    if not grade_index:
        match = _SENIOR_RE.search(text)
        if match:
            grade_index = {'一': 10, '二': 11, '三': 12}[match.group(1)]
    if not grade_index:
        match = _CN_GRADE_RE.search(text)
        if match:
            grade_index = '一二三四五六七八九'.index(match.group(1)) + 1
    if not grade_index:
        match = _AR_GRADE_RE.search(text)
        if match:
            grade_index = int(match.group(1))

    stage = ''
    for name, grades in STAGE_GRADES.items():
        if grade_index and grade_index in grades:
            stage = name
            break
    if not stage and '必修' in text:
        stage = '高中'          # 高中教材用"必修/选择性必修"，年级留空
        warnings.append('按「必修」判定为高中，但具体年级需人工确认')

    grade = GRADE_NAMES.get(grade_index, '')
    if not grade:
        warnings.append('未识别年级，需人工确认')

    # 学科（长词优先，避免"政治"抢走"道德与法治"）
    subject = ''
    for candidate in sorted(SUBJECTS, key=len, reverse=True):
        if candidate in text:
            subject = candidate
            break
    if not subject:
        warnings.append('未识别学科，需人工确认')

    # 学期
    semester = ''
    match = _SEMESTER_RE.search(text)
    if match:
        if match.group(3):
            semester = '全一册'
        elif match.group(1):
            semester = f'{match.group(1)}册'
        elif match.group(2):
            semester = '上册' if match.group(2) == '一' else '下册'
    if not semester:
        warnings.append('未识别学期（上册/下册），需人工确认')

    edition = _first_present(text, EDITIONS)
    if not edition:
        warnings.append('未识别教材版本，需人工确认')

    book_type = _first_present(text, BOOK_TYPES) or ('教材' if grade else '')
    if book_type in NON_TEXTBOOK_TYPES and not grade:
        warnings.append(f'疑似非教材（{book_type}），年级与学期必须人工指定，不做推测')

    # 语言：有中日韩汉字即视为中文书；纯拉丁书名视为英文书；都没有就留空
    language = ''
    if _CJK_RE.search(text):
        language = 'zh'
    elif _LATIN_RE.search(text):
        language = 'en'
    if not language:
        warnings.append('语言未识别，需人工确认')

    scheme = difficulty_scheme if difficulty_scheme in DIFFICULTY_SCHEMES else '1-5'
    # 置信度只反映"猜到了几个关键字段"，不参与任何放行判定
    detected = [stage, grade, subject, semester]
    confidence = round(sum(1 for value in detected if value) / float(len(detected)), 2)
    if not grade:
        confidence = round(min(confidence, 0.5), 2)

    return {
        'title': raw,
        'language': language,
        'stage': stage,
        'grade': grade,
        'grade_index': grade_index if grade else 0,
        'subject': subject,
        'semester': semester,
        'edition': edition,
        'book_type': book_type,
        'difficulty_scheme': scheme,
        'confidence': confidence,
        'source': 'rule',
        'warnings': warnings,
    }


def book_meta_rev(meta: dict) -> str:
    """``sha256(language|stage|grade|subject|difficulty_scheme)``。

    用固定字段顺序拼接，不按字典序 —— 顺序一变，历史工程的 ``book_meta_rev``
    就全部"凭空变了"，会触发无意义的全书 stale。
    """
    meta = meta if isinstance(meta, dict) else {}
    parts = []
    for key in META_KEYS:
        value = meta.get(key, '')
        if key == 'difficulty_scheme' and not str(value or '').strip():
            value = '1-5'
        parts.append(str(value or '').strip())
    payload = '|'.join(parts)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def validate_book_metadata_revision(meta: dict, expected_rev: str = '', *,
                                    locked: bool = False,
                                    unlock_reason: str = '') -> dict:
    """校验书级字段值域，并判断与工程里存的 ``book_meta_rev`` 是否一致。

    返回 ``{'ok', 'rev', 'dirty', 'locked', 'problems', 'warnings'}``：
    - ``problems`` 非空 → 不允许写库、不允许出包
    - ``dirty`` 为真 → 快照全部失效（§3.5 第 5 条），需要重算书级信息相关字段
    - ``locked`` 为真且没有 ``unlock_reason`` → 拒绝改动（D6）
    """
    meta = meta if isinstance(meta, dict) else {}
    problems: list[str] = []
    warnings: list[str] = []

    for key in ('language', 'stage', 'subject'):
        if not str(meta.get(key) or '').strip():
            problems.append(f'缺书级字段：{key}')

    language = str(meta.get('language') or '').strip()
    if language and language not in LANGUAGES:
        problems.append(f'语言值域不合法：{language}')

    stage = str(meta.get('stage') or '').strip()
    if stage and stage not in STAGES:
        problems.append(f'学段值域不合法：{stage}')

    grade = str(meta.get('grade') or '').strip()
    if grade and grade not in GRADE_NAMES.values():
        problems.append(f'年级值域不合法：{grade}')
    if grade and stage:
        index = next((key for key, value in GRADE_NAMES.items() if value == grade), 0)
        if index and index not in STAGE_GRADES.get(stage, ()):
            problems.append(f'年级 {grade} 与学段 {stage} 不匹配')

    subject = str(meta.get('subject') or '').strip()
    if subject and subject not in SUBJECTS:
        warnings.append(f'学科不在常见值域内：{subject}')

    scheme = str(meta.get('difficulty_scheme') or '1-5').strip()
    if scheme not in DIFFICULTY_SCHEMES:
        problems.append(f'难度口径值域不合法：{scheme}')

    rev = book_meta_rev(meta)
    expected = str(expected_rev or '').strip()
    dirty = bool(expected) and rev != expected
    if dirty:
        warnings.append('书级信息已变更：book_meta_rev 不一致，相关快照需重算')
    if dirty and locked and not str(unlock_reason or '').strip():
        problems.append('书级信息已锁定，改动需要 unlock_reason')

    return {
        'ok': not problems,
        'rev': rev,
        'dirty': dirty,
        'locked': bool(locked),
        'problems': problems,
        'warnings': warnings,
    }
