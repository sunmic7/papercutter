"""按教材版式（profile）集中声明切题策略。

以前 ``profile == 'exam'`` 这类字面量散落在 OCR 采样、页分类、分卷、大题切分、
答案解析五个环节共十几处，改一条规则要满文件找；也没法回答“这个版式到底开了
哪些特殊处理”。这里把每个版式的行为收成一张不可变表，实现仍留在
``cut_questions_v4``，调用方只问策略要不要这样做::

    strat = strategy_for(profile)
    if strat.full_width_structure:
        ...

约定：

* 本模块**不 import 项目内任何模块**，避免和 ``cut_questions_v4`` 循环依赖。
* 这里只放开关与阈值，不放算法；新增开关必须同时在 ``DESCRIPTIONS`` 里给
  中文说明，界面要能显示。
* 改任何字段的默认值都等于改产品行为。改完必须用
  ``_experiments/p1_recheck.py`` 跑同一批页面，与 ``reports/strategy_before.jsonl``
  逐字段对比；只允许出现预期的差异。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LayoutStrategy:
    """一个版式下所有“要不要特殊处理”的开关。默认值等于通用版式。"""

    profile: str = 'general'

    # ── 结构 OCR 采样（ocr_page_structure）──────────────────────────
    # 页眉标题带占页高的比例。试卷卷名常印在 13%~25%，通用只留 13%。
    header_ratio: float = 0.13
    # 是否整页参与一次结构 OCR。对开扫描靠边条省时间，但多栏卷会把中栏丢掉。
    full_width_structure: bool = False

    # ── 单页分类（classify_page）───────────────────────────────────
    # 卷名带“大试卷”且无题号 → 判为封面/说明页。
    wide_title_as_cover: bool = False
    # 讲解关键词 ≥2 且无练习关键词 → 判为讲解页，不进切题。
    instructional_without_exercise: bool = False

    # ── 整本分单元 / 分卷（partition_flat_document）────────────────
    # 每张扫描页就是一份独立单元（假期作业一天一页）。
    page_per_unit: bool = False
    # 用页眉“第X练”作为新单元边界。
    practice_title_boundary: bool = False
    # 卷名读不出时，用“卷面书写 + 第一大题”作为卷首后备边界。
    exam_cover_fallback: bool = False
    # 兼容旧版“思维拓展32练”整书合并结果，避免升级后沿用错误缓存。
    legacy_flat_unit_guard: bool = False
    # 单元名按序号改成“第N课时”，而不是用页眉标题（答案页方框编号与页码有偏移）。
    numbered_day_units: bool = False

    # ── 大题切分（_split_single_col_majors / find_unnumbered_boundaries）──
    # “难题点拨”“想一想做一做”等栏目提升为大题。
    field_headings: bool = False
    # 用 ①②③ 这类圈号作为大题编号（部分奥数书不用“一、二、三”）。
    circled_major_numbers: bool = False
    # “数学游戏”“假期体验”等活动标题提升为大题。
    activity_headings: bool = False
    # 一个标题都没有但页面是讲次首页时，整体归为“讲解示例”而不是“未分类”。
    lesson_opening_fallback: bool = False
    # 相邻大题之间按行墨量找切缝，避开下一标题上方的装饰图。
    density_section_cut: bool = False
    # 用活动标题把无编号页面分成若干活动区。
    unnumbered_activity: bool = False
    # 用大段垂直留白恢复读不出编号的题块。
    unnumbered_whitespace: bool = False

    # ── 答案页解析（process_answer_pages）──────────────────────────
    # 栏首方框数字表示“第几天/第几课”，OCR 常漏掉紧随其后的“一、”。
    answer_leading_day_number: bool = False
    # 答案里出现“第X讲”时当作讲次标题，用于给后续答案挂上下文。
    answer_lesson_context: bool = True

    def describe(self) -> list[str]:
        """返回已开启的条目中文说明，供界面“为什么这样切”展示。"""
        return _render_notes(self)

    def to_dict(self) -> dict:
        payload = {'profile': self.profile}
        payload.update({
            field: getattr(self, field) for field in DESCRIPTIONS if field != 'profile'
        })
        return payload

    def with_overrides(self, **overrides) -> 'LayoutStrategy':
        """派生一份改过个别开关的策略，用于单页/单本临时覆盖。"""
        unknown = set(overrides) - set(DESCRIPTIONS)
        if unknown:
            raise ValueError(f'未知策略字段：{sorted(unknown)}')
        return replace(self, **overrides)


# 顺序即界面展示顺序，与 cut_questions_v4.BOOK_PROFILE_CHOICES 保持一致。
GENERAL = LayoutStrategy(profile='general')

EXAM = LayoutStrategy(
    profile='exam',
    header_ratio=0.28,
    full_width_structure=True,
    wide_title_as_cover=True,
    exam_cover_fallback=True,
    answer_lesson_context=False,
)

WORKBOOK = LayoutStrategy(
    profile='workbook',
    practice_title_boundary=True,
)

DRILL = LayoutStrategy(
    profile='drill',
    practice_title_boundary=True,
    answer_lesson_context=False,
)

HOLIDAY = LayoutStrategy(
    profile='holiday',
    page_per_unit=True,
    numbered_day_units=True,
    activity_headings=True,
    density_section_cut=True,
    unnumbered_activity=True,
    answer_leading_day_number=True,
    answer_lesson_context=False,
)

OLYMPIAD = LayoutStrategy(
    profile='olympiad',
    practice_title_boundary=True,
    legacy_flat_unit_guard=True,
    field_headings=True,
    circled_major_numbers=True,
    lesson_opening_fallback=True,
    unnumbered_whitespace=True,
    answer_lesson_context=False,
)

INSTRUCTIONAL = LayoutStrategy(
    profile='instructional',
    instructional_without_exercise=True,
)

STRATEGIES = {
    strategy.profile: strategy
    for strategy in (GENERAL, EXAM, WORKBOOK, DRILL, HOLIDAY, OLYMPIAD, INSTRUCTIONAL)
}

PROFILE_ORDER = tuple(STRATEGIES)

DESCRIPTIONS = {
    'profile': '版式',
    'header_ratio': '页眉标题带按页高 {value:.0%} 采样',
    'full_width_structure': '整页参与结构识别（多栏卷不丢中栏）',
    'wide_title_as_cover': '“大试卷”卷名且无题号判为封面页',
    'instructional_without_exercise': '只有讲解词、没有练习题时判为讲解页',
    'page_per_unit': '每页单独成单元',
    'practice_title_boundary': '页眉“第X练”作为新单元边界',
    'exam_cover_fallback': '卷名漏读时用“卷面书写+第一大题”认卷首',
    'legacy_flat_unit_guard': '兼容旧版整书合并结果',
    'numbered_day_units': '单元名按“第N课时”编号',
    'field_headings': '“难题点拨”等栏目作为大题',
    'circled_major_numbers': '用 ①②③ 圈号作为大题编号',
    'activity_headings': '“数学游戏/假期体验”等活动作为大题',
    'lesson_opening_fallback': '无大题号的讲次首页归为“讲解示例”',
    'density_section_cut': '按行墨量找大题之间的切缝',
    'unnumbered_activity': '用活动标题切分无编号页面',
    'unnumbered_whitespace': '用大段留白恢复读不出编号的题块',
    'answer_leading_day_number': '答案页栏首方框数字识别“第几天”',
    'answer_lesson_context': '答案里“第X讲”作为讲次标题',
}

# 默认开启、某些版式需要关掉的开关，说明写在“关掉”的一面。
DISABLED_DESCRIPTIONS = {
    'answer_lesson_context': '答案页不把“第X讲”当讲次标题',
}

# 需要按数值填入说明的字段，避免 describe() 里出现 {value} 占位符。
_NUMERIC_FIELDS = {'header_ratio'}


def strategy_for(profile: str | LayoutStrategy | None) -> LayoutStrategy:
    """取版式策略；未知或空值按通用处理（与旧版 ``==`` 判断的行为一致）。"""
    if isinstance(profile, LayoutStrategy):
        return profile
    key = str(profile or '').strip().lower()
    return STRATEGIES.get(key, GENERAL)


def describe_profile(profile: str | LayoutStrategy | None) -> dict:
    """界面用：返回版式名、已启用条目和原始开关值。"""
    strat = strategy_for(profile)
    return {
        'profile': strat.profile,
        'notes': _render_notes(strat),
        'options': strat.to_dict(),
    }


def _render_notes(strategy: LayoutStrategy) -> list[str]:
    """只列出与本版式默认值不同的开关，通用版式因此返回空列表。"""
    default = LayoutStrategy(profile=strategy.profile)
    notes = []
    for field, template in DESCRIPTIONS.items():
        if field == 'profile':
            continue
        value = getattr(strategy, field)
        if value == getattr(default, field):
            continue
        if not value:
            if field in DISABLED_DESCRIPTIONS:
                notes.append(DISABLED_DESCRIPTIONS[field])
            continue
        if field in _NUMERIC_FIELDS:
            notes.append(template.format(value=value))
        else:
            notes.append(template)
    return notes


__all__ = [
    'LayoutStrategy', 'STRATEGIES', 'PROFILE_ORDER', 'DESCRIPTIONS',
    'strategy_for', 'describe_profile',
    'GENERAL', 'EXAM', 'WORKBOOK', 'DRILL', 'HOLIDAY', 'OLYMPIAD', 'INSTRUCTIONAL',
]
