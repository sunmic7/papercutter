"""策略层：版式开关表必须覆盖所有版式，且与核心模块的口径一致。"""
from __future__ import annotations

import dataclasses
import unittest

import layout_strategy as ls
import cut_questions_v4 as cutter


class StrategyTableTests(unittest.TestCase):
    def test_every_book_profile_has_a_strategy(self) -> None:
        self.assertEqual(set(ls.STRATEGIES), set(cutter.BOOK_PROFILE_CHOICES))
        self.assertEqual(ls.PROFILE_ORDER, cutter.BOOK_PROFILE_CHOICES)

    def test_unknown_profile_falls_back_to_general(self) -> None:
        # 旧版用 ``profile == 'xxx'`` 判断，未知值等价于通用；这里必须保持一致。
        for value in ('', None, 'exam2', '  ', 'GENERAL'):
            self.assertEqual(ls.strategy_for(value).profile, 'general', value)
        self.assertEqual(ls.strategy_for('exam').profile, 'exam')
        self.assertEqual(ls.strategy_for(' Exam ').profile, 'exam')

    def test_strategy_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ls.EXAM.header_ratio = 0.5  # type: ignore[misc]

    def test_overrides_keep_other_fields(self) -> None:
        derived = ls.EXAM.with_overrides(full_width_structure=False)
        self.assertFalse(derived.full_width_structure)
        self.assertEqual(derived.header_ratio, 0.28)
        self.assertEqual(ls.EXAM.full_width_structure, True)
        with self.assertRaises(ValueError):
            ls.EXAM.with_overrides(not_a_field=True)

    def test_every_field_has_a_description(self) -> None:
        fields = {field.name for field in dataclasses.fields(ls.LayoutStrategy)}
        self.assertEqual(fields, set(ls.DESCRIPTIONS))
        self.assertTrue(set(ls.DISABLED_DESCRIPTIONS) <= fields)

    def test_to_dict_covers_all_fields(self) -> None:
        payload = ls.EXAM.to_dict()
        self.assertEqual(set(payload), set(ls.DESCRIPTIONS))
        self.assertEqual(payload['profile'], 'exam')
        self.assertEqual(payload['header_ratio'], 0.28)


class StrategyContentTests(unittest.TestCase):
    """锁住各版式的关键开关，避免重构时静默改掉产品行为。"""

    def test_general_has_no_special_handling(self) -> None:
        self.assertEqual(ls.GENERAL.describe(), [])
        self.assertEqual(ls.GENERAL.header_ratio, 0.13)
        self.assertFalse(ls.GENERAL.full_width_structure)

    def test_exam_widens_header_and_full_width(self) -> None:
        exam = ls.strategy_for('exam')
        self.assertEqual(exam.header_ratio, 0.28)
        self.assertTrue(exam.full_width_structure)
        self.assertTrue(exam.exam_cover_fallback)
        self.assertTrue(exam.wide_title_as_cover)
        # 试卷答案按序号排列，不把“第X讲”当讲次标题。
        self.assertFalse(exam.answer_lesson_context)

    def test_holiday_uses_per_page_units_and_activity_headings(self) -> None:
        holiday = ls.strategy_for('holiday')
        self.assertTrue(holiday.page_per_unit)
        self.assertTrue(holiday.activity_headings)
        self.assertTrue(holiday.density_section_cut)
        self.assertTrue(holiday.unnumbered_activity)
        self.assertTrue(holiday.answer_leading_day_number)
        self.assertFalse(holiday.unnumbered_whitespace)

    def test_olympiad_uses_field_headings_and_whitespace(self) -> None:
        olympiad = ls.strategy_for('olympiad')
        self.assertTrue(olympiad.field_headings)
        self.assertTrue(olympiad.unnumbered_whitespace)
        self.assertTrue(olympiad.lesson_opening_fallback)
        self.assertTrue(olympiad.legacy_flat_unit_guard)
        self.assertTrue(olympiad.circled_major_numbers)
        self.assertFalse(olympiad.activity_headings)

    def test_practice_title_boundary_only_for_flat_practice_books(self) -> None:
        enabled = {
            profile for profile in ls.PROFILE_ORDER
            if ls.strategy_for(profile).practice_title_boundary
        }
        self.assertEqual(enabled, {'olympiad', 'workbook', 'drill'})

    def test_instructional_only_filters_pure_instruction_pages(self) -> None:
        self.assertTrue(ls.strategy_for('instructional').instructional_without_exercise)
        self.assertFalse(ls.strategy_for('workbook').instructional_without_exercise)


# 重构之后新增的开关，不参与“与旧字面量等价”的对照。
NEW_FIELDS = {'circled_major_numbers'}


class StrategyEquivalenceTests(unittest.TestCase):
    """每个开关都必须与重构前的字面量判断逐个版式等价。

    这是“重构不改行为”的静态证明：不依赖 OCR，也不依赖抽到哪几页。
    """

    CASES = {
        # 字段 -> 重构前写在 cut_questions_v4 里的条件
        'header_ratio': lambda p: 0.28 if p == 'exam' else 0.13,
        'full_width_structure': lambda p: p == 'exam',
        'wide_title_as_cover': lambda p: p == 'exam',
        'instructional_without_exercise': lambda p: p == 'instructional',
        'page_per_unit': lambda p: p == 'holiday',
        'practice_title_boundary': lambda p: p in {'olympiad', 'workbook', 'drill'},
        'exam_cover_fallback': lambda p: p == 'exam',
        'legacy_flat_unit_guard': lambda p: p == 'olympiad',
        'numbered_day_units': lambda p: p == 'holiday',
        'field_headings': lambda p: p == 'olympiad',
        'activity_headings': lambda p: p == 'holiday',
        'lesson_opening_fallback': lambda p: p == 'olympiad',
        'density_section_cut': lambda p: p == 'holiday',
        'unnumbered_activity': lambda p: p == 'holiday',
        'unnumbered_whitespace': lambda p: p == 'olympiad',
        'answer_leading_day_number': lambda p: p == 'holiday',
        'answer_lesson_context': lambda p: p in {'workbook', 'general', 'instructional'},
    }

    def test_fields_match_legacy_literals(self) -> None:
        # 重构之后新增的开关没有“重构前”可对照，单独在内容测试里锁。
        self.assertEqual(
            set(self.CASES) | NEW_FIELDS,
            set(ls.DESCRIPTIONS) - {'profile'},
        )
        for profile in ls.PROFILE_ORDER:
            strategy = ls.strategy_for(profile)
            for field, legacy in self.CASES.items():
                self.assertEqual(
                    getattr(strategy, field), legacy(profile),
                    f'{profile}.{field} 与重构前的判断不一致',
                )


class StrategyIntegrationTests(unittest.TestCase):
    def test_cut_core_reads_the_same_strategy_table(self) -> None:
        # 核心模块只允许通过 strategy_for 取开关，不保留第二份表。
        self.assertIs(cutter.layout_strategy.strategy_for('exam'), ls.EXAM)

    def test_paper_start_reason_matches_strategy(self) -> None:
        page = {'title': '', 'ocr_items': []}
        for profile in ls.PROFILE_ORDER:
            self.assertEqual(
                cutter.paper_start_reason(page, profile), '',
                f'{profile} 不应在空页上认出新试卷',
            )

    def test_describe_profile_shape(self) -> None:
        payload = ls.describe_profile('holiday')
        self.assertEqual(payload['profile'], 'holiday')
        self.assertTrue(payload['notes'])
        self.assertTrue(payload['options']['page_per_unit'])


if __name__ == '__main__':
    unittest.main()
