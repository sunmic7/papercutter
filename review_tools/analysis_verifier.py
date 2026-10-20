# -*- coding: utf-8 -*-
"""解析机检：结构检查 + AST 白名单复算 + 符号闭包（纯函数，无 IO）。

为什么要有它
------------
"AI 复核 AI" 会得到高度相关的错误（同一模型的盲区一致），所以这里只做
**规则 + 数学求值**，与生成侧不同源。它的信噪比是整条链路里最高的：
机器能判出算错的解析，人看一眼也一定觉得别扭。

三条检查链（详见架构文档 §7.2）
------------------------------
链 1 结构检查（全学科兜底）：非空、非占位、非答案复制、无幻觉题图、引用题干。
链 2 数值复算（``subject=数学`` 且 ``question_type ∈ {计算, 填空}``）：
  从题干抽算术表达式 → **AST 白名单求值** → 与答案比对。
链 3 不可复算时降级：符号闭包 → 填空回填 → 单位量纲，全不过才 ``not_checkable``。

安全底线
--------
题干来自 OCR，等于不可信输入，因此**禁用 eval/exec**。求值走
``ast.parse`` + 递归白名单：只允许
``Expression / BinOp(Add,Sub,Mult,Div,Pow) / UnaryOp(USub,UAdd) / Constant(int,float)
/ Call(abs, sqrt)``，其它任何节点（Attribute、Name、Subscript、Lambda…）一律
判 ``not_checkable``，且**一行都不执行**。
"""
from __future__ import annotations

import ast
import math
import re
from fractions import Fraction

VERIFIER_VERSION = 'analysis-verify-v1'

# 链 1 阈值
MIN_ANALYSIS_CHARS = 20
PLACEHOLDER_TOKENS = ('略', '同上', '见解析', '见答案', '答案：见上', '答案:见上')
FIGURE_TOKENS = ('如图', '图中', '图示', '下图', '上图')

# 链 2
NUMERIC_SUBJECTS = ('数学', 'math')
NUMERIC_QUESTION_TYPES = ('计算', '填空')
MAX_EXPRESSION_CHARS = 200
MAX_ABS_VALUE = 10 ** 9          # 求值结果绝对值上限，防止构造天量分数（小学题量级远小于它）
MAX_POW_EXPONENT = 12

# 链 3
SYMBOL_CONSTANTS = ('π', 'e')
SAFE_FUNCTION_NAMES = ('abs', 'sqrt')

# 单位表：长的在前，抽取时优先匹配（平方厘米 先于 厘米 先于 分）
UNIT_WORDS = (
    '立方厘米', '立方分米', '立方米', '平方厘米', '平方分米', '平方千米', '平方米',
    '千米', '厘米', '毫米', '分米', '千克', '毫升', '吨', '克', '升', '米',
    '小时', '分钟', '星期', '元', '角', '分', '个', '只', '本', '支', '张',
    '秒', '时', '天', '周', '月', '年', '度',
)
MULTIPLIER_WORDS = ('万', '亿')

# 表达式字符白名单（先粗提，再精判）
_EXPRESSION_CHARS = re.compile(r'[\d.+\-*/×÷()（）^√]+')
_STEP_MARKER_RE = re.compile(r'[（(]\s*\d{1,2}\s*[）)]')
_STEP_LINE_RE = re.compile(r'(^|[；;。\n])\s*(\d{1,2})\s*[、.．](?!\d)', re.MULTILINE)
_NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')
_FRACTION_RE = re.compile(r'(\d+)\s*/\s*(\d+)')
_MIXED_RE = re.compile(r'(\d+)\s+(\d+)\s*/\s*(\d+)')
_ASCII_LETTER_RE = re.compile(r'[A-Za-z]+')
_BLANK_RE = re.compile(r'_{2,}|（\s*）|\(\s*\)')

_STOPWORDS = {
    '的', '了', '和', '是', '在', '有', '与', '及', '把', '被', '个', '这', '那',
    '一个', '所以', '因为', '因此', '然后', '我们', '可以', '这个', '这样',
}


# ---------------------------------------------------------------- 文本工具
def _text_of(record: dict, snapshot: dict, *keys: str) -> str:
    for source in (snapshot, record):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return ''


def _analysis_text(record: dict, snapshot: dict) -> str:
    return _text_of(record, snapshot, 'analysis', 'explanation', 'solution')


def _question_text(record: dict, snapshot: dict) -> str:
    return _text_of(record, snapshot, 'text', 'question', 'stem')


def _answer_text(record: dict, snapshot: dict) -> str:
    return _text_of(record, snapshot, 'answer', 'answer_text')


def _strip_symbols(text: str) -> str:
    return re.sub(r'[\s\W_]+', '', str(text or ''), flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r'[\u4e00-\u9fff]{2,6}|[A-Za-z]{2,}|\d+', str(text or ''))
        if token not in _STOPWORDS
    }


# ---------------------------------------------------------------- 链 1
def _structure_checks(record: dict, snapshot: dict) -> list[dict]:
    """全学科兜底的四条硬检查 + 一条只记录不判定的引用检查。"""
    record = record if isinstance(record, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    analysis = _analysis_text(record, snapshot)
    answer = _answer_text(record, snapshot)
    question = _question_text(record, snapshot)
    checks: list[dict] = []

    compact = re.sub(r'\s+', '', analysis)
    checks.append({
        'name': 'non_empty',
        'ok': len(compact) >= MIN_ANALYSIS_CHARS,
        'hard': True,
        'detail': f'解析 {len(compact)} 字（下限 {MIN_ANALYSIS_CHARS}）',
    })

    hit = next((token for token in PLACEHOLDER_TOKENS if token in analysis), '')
    checks.append({
        'name': 'not_placeholder',
        'ok': not hit,
        'hard': True,
        'detail': f'命中占位词 {hit!r}' if hit else '无占位词',
    })

    if answer.strip():
        same = _strip_symbols(analysis) == _strip_symbols(answer)
        checks.append({
            'name': 'not_answer_copy',
            'ok': not same,
            'hard': True,
            'detail': '解析与答案去符号后完全相同' if same else '解析独立于答案',
        })
    else:
        checks.append({
            'name': 'not_answer_copy',
            'ok': None,
            'hard': True,
            'detail': '答案为空，不适用',
        })

    mentions_figure = next((token for token in FIGURE_TOKENS if token in analysis), '')
    boxes = snapshot.get('illustration_boxes')
    has_figure = bool(boxes) if isinstance(boxes, (list, tuple)) else bool(boxes)
    if mentions_figure:
        checks.append({
            'name': 'no_phantom_figure',
            'ok': has_figure,
            'hard': True,
            'detail': f'解析提到 {mentions_figure!r}，题图 {"存在" if has_figure else "缺失"}',
        })
    else:
        checks.append({
            'name': 'no_phantom_figure',
            'ok': None,
            'hard': True,
            'detail': '解析未提及图形',
        })

    common = _tokens(analysis) & _tokens(question)
    checks.append({
        'name': 'refer_question',
        'ok': bool(common),
        'hard': False,          # 只记录：解析可能用同义改写，不足以判定
        'detail': f'与题干共有词 {sorted(common)[:5]}',
    })
    return checks


# ---------------------------------------------------------------- 链 2：表达式
def _normalize_expression(source: str) -> str:
    value = str(source or '')
    value = value.replace('×', '*').replace('÷', '/').replace('．', '.')
    value = value.replace('（', '(').replace('）', ')')
    value = value.replace('－', '-').replace('＋', '+')
    # √9 → sqrt(9)；√(3+1) → sqrt(3+1)
    value = re.sub(r'√\s*\(([^()]*)\)', r'sqrt(\1)', value)
    value = re.sub(r'√\s*(\d+(?:\.\d+)?)', r'sqrt(\1)', value)
    value = value.replace('^', '**')
    return value.strip()


def _extract_expressions(text: str) -> list[str]:
    """从题干抽算术表达式：长度 ≥3、含运算符、首字符必须是数字/'('/'√'。"""
    cleaned = _STEP_MARKER_RE.sub(' ', str(text or ''))
    found: list[str] = []
    for raw in _EXPRESSION_CHARS.findall(cleaned):
        piece = raw.strip()
        if len(piece) < 3 or len(piece) > MAX_EXPRESSION_CHARS:
            continue
        if not any(char in piece for char in '+-*/×÷^√'):
            continue
        if piece[0] in '+-*/^':
            piece = piece[1:]
        piece = piece.rstrip('+-*/^.')
        if len(piece) < 3:
            continue
        if not re.search(r'\d', piece):
            continue
        # 纯日期 / 纯小数串（如 "2026.10"）不算表达式
        if not any(char in piece for char in '+-*/×÷^√'):
            continue
        found.append(piece)
    return found


class _Unsafe(Exception):
    """表达式里出现了白名单之外的节点或超限运算。"""


def _constant(node: ast.Constant) -> Fraction:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Unsafe('非常量')
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _eval_node(node):
    """AST 白名单递归求值。任何未列出的节点 → ``_Unsafe``（绝不执行）。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise _Unsafe('一元运算符不在白名单')
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError('除数为 0')
            result = left / right
        elif isinstance(node.op, ast.Pow):
            if not isinstance(right, Fraction) or right.denominator != 1:
                raise _Unsafe('指数必须是整数')
            exponent = int(right)
            if abs(exponent) > MAX_POW_EXPONENT:
                raise _Unsafe('指数过大')
            result = left ** exponent
        else:
            raise _Unsafe('二元运算符不在白名单')
        if abs(result) > MAX_ABS_VALUE:
            raise _Unsafe('结果过大')
        return result
    if isinstance(node, ast.Call):
        if node.keywords:
            raise _Unsafe('不接受关键字参数')
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTION_NAMES:
            raise _Unsafe('函数不在白名单')
        if len(node.args) != 1:
            raise _Unsafe('参数个数不是 1')
        value = _eval_node(node.args[0])
        if node.func.id == 'abs':
            return abs(value)
        if value < 0:
            raise _Unsafe('负数开平方')
        numerator_root = math.isqrt(value.numerator)
        denominator_root = math.isqrt(value.denominator)
        if (numerator_root ** 2 == value.numerator
                and denominator_root ** 2 == value.denominator):
            return Fraction(numerator_root, denominator_root)
        return float(value) ** 0.5
    raise _Unsafe(f'节点不在白名单：{type(node).__name__}')


def safe_eval_expression(expression: str):
    """对单条表达式做白名单求值；不可判定时返回 ``None``（绝不抛给调用方）。"""
    source = _normalize_expression(expression)
    if not source or len(source) > MAX_EXPRESSION_CHARS:
        return None
    try:
        tree = ast.parse(source, mode='eval')
    except (SyntaxError, ValueError, MemoryError):
        return None
    try:
        return _eval_node(tree)
    except (_Unsafe, ZeroDivisionError, OverflowError, ValueError):
        return None


def _extract_values(text: str) -> list[Fraction]:
    """从答案侧抽数值：带分数 → 分数 → 小数/整数（剥单位靠正则天然完成）。"""
    source = str(text or '')
    values: list[Fraction] = []
    consumed: list[tuple[int, int]] = []
    for match in _MIXED_RE.finditer(source):
        whole, numerator, denominator = match.groups()
        try:
            divisor = int(denominator)
            if divisor == 0:
                continue
            values.append(Fraction(int(whole)) + Fraction(int(numerator), divisor))
        except (ValueError, ZeroDivisionError):
            continue
        consumed.append(match.span())
    for match in _FRACTION_RE.finditer(source):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        numerator, denominator = match.groups()
        try:
            divisor = int(denominator)
            if divisor == 0:
                continue
            values.append(Fraction(int(numerator), divisor))
        except (ValueError, ZeroDivisionError):
            continue
        consumed.append(match.span())
    for match in _NUMBER_RE.finditer(source):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        try:
            values.append(Fraction(match.group(0)))
        except (ValueError, ZeroDivisionError):
            continue
    return values


def _same_value(left, right, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return False
    try:
        if isinstance(left, Fraction) and isinstance(right, Fraction):
            if left == right:
                return True
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError, OverflowError):
        return False


def _numeric_recompute(record: dict, snapshot: dict) -> dict:
    """链 2：题型可复算时逐问复算，任一不等即 ``fail``。"""
    record = record if isinstance(record, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    subject = str(record.get('subject') or snapshot.get('subject') or '')
    question_type = str(record.get('question_type') or snapshot.get('question_type') or '')
    base = {'name': 'numeric_recompute', 'hard': True, 'applicable': False}

    if subject not in NUMERIC_SUBJECTS or question_type not in NUMERIC_QUESTION_TYPES:
        return {**base, 'ok': None, 'detail': f'不适用（学科={subject or "未知"}，题型={question_type or "未知"}）'}

    question = _question_text(record, snapshot)
    expressions = _extract_expressions(question)
    if not expressions:
        return {**base, 'ok': None, 'detail': '题干未抽取到算术表达式'}

    answer = _answer_text(record, snapshot)
    expected = _extract_values(answer)
    if not expected:
        return {**base, 'ok': None, 'detail': '答案侧未抽取到数值'}

    computed = []
    for expression in expressions:
        value = safe_eval_expression(expression)
        if value is None:
            return {
                **base,
                'ok': None,
                'detail': f'表达式不可安全求值：{expression}',
            }
        computed.append(value)

    if len(computed) == len(expected):
        pairs = list(zip(computed, expected))
    elif len(expected) == 1:
        pairs = [(value, expected[0]) for value in computed]
    else:
        # 多小问但数量不等：要求每个计算结果都能在答案里找到（不猜对应关系）
        pairs = [(value, None) for value in computed]

    mismatches = []
    for value, target in pairs:
        if target is None:
            if not any(_same_value(value, candidate) for candidate in expected):
                mismatches.append((value, expected))
            continue
        if not _same_value(value, target):
            mismatches.append((value, target))

    if mismatches:
        got, want = mismatches[0]
        detail = f'expected={_pretty(want)} got={_pretty(got)}'
        if len(mismatches) > 1:
            detail += f'（共 {len(mismatches)} 处不等）'
        return {**base, 'ok': False, 'applicable': True, 'detail': detail}

    return {
        **base,
        'ok': True,
        'applicable': True,
        'detail': f'{len(pairs)} 处复算一致：{_pretty([item[0] for item in pairs])}',
    }


def _pretty(value) -> str:
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_pretty(item) for item in value) + ']'
    if value is None:
        return '?'
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f'{value.numerator}/{value.denominator}'
    if isinstance(value, float):
        return f'{value:.10g}'
    return str(value)


# ---------------------------------------------------------------- 链 3
def _analysis_numbers(text: str) -> list[Fraction]:
    cleaned = _STEP_LINE_RE.sub(lambda match: match.group(1), str(text or ''))
    cleaned = _STEP_MARKER_RE.sub(' ', cleaned)
    return _extract_values(cleaned)


def _symbol_subset(record: dict, snapshot: dict) -> dict:
    """解析里出现的数字/字母必须 ⊆ 题面 ∪ 答案 ∪ {π, e}。"""
    record = record if isinstance(record, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    analysis = _analysis_text(record, snapshot)
    question = _question_text(record, snapshot)
    answer = _answer_text(record, snapshot)
    base = {'name': 'symbol_subset', 'hard': True}
    if not analysis:
        return {**base, 'ok': None, 'applicable': False, 'detail': '无解析'}

    allowed_numbers = _analysis_numbers(question) + _analysis_numbers(answer)
    analysis_numbers = _analysis_numbers(analysis)
    extra_numbers = [
        value for value in analysis_numbers
        if not any(_same_value(value, candidate) for candidate in allowed_numbers)
    ]

    allowed_letters = set(_ASCII_LETTER_RE.findall(question)) | set(_ASCII_LETTER_RE.findall(answer))
    allowed_letters |= {name for name in SAFE_FUNCTION_NAMES}
    analysis_letters = set(_ASCII_LETTER_RE.findall(analysis))
    extra_letters = {
        letter for letter in analysis_letters
        if letter not in allowed_letters and letter not in SYMBOL_CONSTANTS
    }

    applicable = bool(analysis_numbers or analysis_letters)
    if not applicable:
        return {**base, 'ok': None, 'applicable': False, 'detail': '解析中无符号'}

    ok = not extra_numbers and not extra_letters
    detail_parts = []
    if extra_numbers:
        detail_parts.append('题面没有的数字 ' + _pretty(extra_numbers[:5]))
    if extra_letters:
        detail_parts.append('题面没有的字母 ' + ','.join(sorted(extra_letters)[:5]))
    return {
        **base,
        'ok': ok,
        'applicable': True,
        'detail': '；'.join(detail_parts) if detail_parts else '符号闭包成立',
    }


def _blank_fill(record: dict, snapshot: dict) -> dict:
    """题干有空格时，答案必须出现在解析里。"""
    record = record if isinstance(record, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    question = _question_text(record, snapshot)
    answer = _answer_text(record, snapshot)
    analysis = _analysis_text(record, snapshot)
    base = {'name': 'blank_fill', 'hard': True}
    if not _BLANK_RE.search(question) or not answer.strip():
        return {**base, 'ok': None, 'applicable': False, 'detail': '题干无空格或答案为空'}

    plain_answer = _strip_symbols(answer)
    plain_analysis = _strip_symbols(analysis)
    hit = bool(plain_answer) and plain_answer in plain_analysis
    if not hit:
        answer_values = _extract_values(answer)
        hit = any(
            any(_same_value(value, candidate) for candidate in _extract_values(analysis))
            for value in answer_values
        )
    return {
        **base,
        'ok': hit,
        'applicable': True,
        'detail': '答案已在解析中回填' if hit else f'答案 {answer.strip()[:20]!r} 未出现在解析中',
    }


def _unit_tokens(text: str) -> set[str]:
    """抽"数量+单位"里的单位。

    只认**紧跟数字之后**的单位（"6元"/"3天"/"9平方米"）。理由是中文里大量
    单位字兼作普通词（"一周""分步""角度""几个"），不设这个前置条件会把
    正常解析判成单位越界 —— 假阳性比漏检贵得多。
    """
    found: set[str] = set()
    source = str(text or '')
    index = 0
    while index < len(source):
        matched = ''
        for unit in UNIT_WORDS:
            if source.startswith(unit, index):
                matched = unit
                break
        if not matched:
            index += 1
            continue
        back = index - 1
        while back >= 0 and source[back] in ' \u3000':
            back -= 1
        preceded_by_number = back >= 0 and (source[back].isdigit() or source[back] in '几多半')
        if preceded_by_number:
            found.add(matched)
        index += len(matched)
    return found


def _unit_consistency(record: dict, snapshot: dict) -> dict:
    """解析中的单位 ⊆ 题面 ∪ 答案的单位（倍数词 万/亿 不单独算单位）。"""
    record = record if isinstance(record, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    analysis = _analysis_text(record, snapshot)
    question = _question_text(record, snapshot)
    answer = _answer_text(record, snapshot)
    base = {'name': 'unit_consistency', 'hard': True}
    analysis_units = _unit_tokens(analysis)
    if not analysis_units:
        return {**base, 'ok': None, 'applicable': False, 'detail': '解析中未出现单位'}
    allowed = _unit_tokens(question) | _unit_tokens(answer)
    extra = analysis_units - allowed
    return {
        **base,
        'ok': not extra,
        'applicable': True,
        'detail': ('多出的单位 ' + '、'.join(sorted(extra))) if extra else '单位一致',
    }


# ---------------------------------------------------------------- 对外接口
def verify(record: dict, snapshot: dict | None = None) -> dict:
    """返回 ``{'status': 'pass'|'fail'|'not_checkable', 'checks': [...], 'verifier_version'}``。

    判定顺序（先硬后软，先准后宽）：
    链 1 任一硬检查失败 → ``fail``；
    链 2 能复算且不等 → ``fail``，能复算且相等 → ``pass``；
    链 3 至少一条适用且全过 → ``pass``（``degraded=True``）；全不适用 → ``not_checkable``。
    """
    record = record if isinstance(record, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    structure_checks = _structure_checks(record, snapshot)
    numeric_check = _numeric_recompute(record, snapshot)
    degraded_checks = [
        _symbol_subset(record, snapshot),
        _blank_fill(record, snapshot),
        _unit_consistency(record, snapshot),
    ]
    checks = list(structure_checks) + [numeric_check] + list(degraded_checks)

    status = _decide_status(structure_checks, numeric_check, degraded_checks)
    return {
        'status': status,
        'checks': checks,
        'verifier_version': VERIFIER_VERSION,
        # 降级通过：结构检查全过、复算不可判定，靠链 3 放行（抽检时必须单独分层）
        'degraded': status == 'pass' and numeric_check.get('ok') is not True,
    }


def _decide_status(structure_checks: list[dict], numeric_check: dict,
                   degraded_checks: list[dict]) -> str:
    if any(check.get('ok') is False for check in structure_checks if check.get('hard')):
        return 'fail'
    if numeric_check.get('ok') is False:
        return 'fail'
    if numeric_check.get('ok') is True:
        return 'pass'

    applicable = [
        check for check in degraded_checks
        if check.get('applicable') and check.get('ok') is not None
    ]
    if not applicable:
        return 'not_checkable'
    if any(check.get('ok') is False for check in applicable):
        return 'fail'
    return 'pass'
