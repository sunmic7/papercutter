import unittest

import numpy as np

import cut_questions_v4 as cutter
from review_tools.manual_annotation_workspace import (
    _knowledge_point_group,
    _primary_number_boundaries,
    _subsection_boundaries,
    _subsection_segments,
    _trim_explanatory_preamble_from_first_minor,
)


def item(text, x, y, confidence=0.99):
    return {
        "text": text,
        "conf": confidence,
        "bbox": [[x, y], [x + 80, y], [x + 80, y + 20], [x, y + 20]],
    }


class ManualAnnotationWorkspaceRuleTests(unittest.TestCase):
    def test_chinese_parenthesized_reading_groups_are_boundaries(self):
        marks = _subsection_boundaries([
            item("（一）课内阅读", 20, 50),
            item("（二）课外阅读", 20, 500),
            item("（1）第2自然段", 30, 700),
        ], 1000)
        self.assertEqual([(mark["number"], mark["y"]) for mark in marks], [("一", 50), ("二", 500)])

    def test_ocr_dash_inside_heading_is_repaired_to_chinese_one(self):
        marks = _subsection_boundaries([
            item("(—)课内阅读。(8分)", 20, 50),
            item("(二)课外阅读。(14分)", 20, 500),
            item("(—)普通内容", 20, 700),
        ], 1000)
        self.assertEqual([(mark["number"], mark["y"]) for mark in marks], [("一", 50), ("二", 500)])

    def test_missing_first_group_is_recovered_before_late_group_two(self):
        segments = _subsection_segments(cutter, [
            item("1. 第一题", 25, 120),
            item("2. 第二题", 25, 300),
            item("（二）课外阅读", 20, 600),
        ], 1000, 1000)
        self.assertEqual([segment["subsection_number"] for segment in segments], ["一", "二"])
        self.assertTrue(segments[0]["subsection_number_inferred"])
        self.assertEqual((segments[0]["y0"], segments[0]["y1"]), (0, 600))

    def test_group_two_at_page_top_is_not_backfilled(self):
        segments = _subsection_segments(cutter, [
            item("（二）课外阅读", 20, 30),
            item("1. 第一题", 25, 100),
        ], 1000, 1000)
        self.assertEqual([segment["subsection_number"] for segment in segments], ["二"])
        self.assertFalse(segments[0]["subsection_number_inferred"])

    def test_lone_printed_two_is_kept_for_page_continuation_split(self):
        boundaries = _primary_number_boundaries(
            cutter, [item("上一页续文", 30, 20), item("2. 新题", 30, 180)], 1000, 240,
        )
        self.assertEqual([boundary["num"] for boundary in boundaries], [2])

    def test_negative_formula_fragment_is_not_a_subsection_heading(self):
        boundaries = _subsection_boundaries(
            [item("（-5）表示", 20, 120)], 1000,
        )
        self.assertEqual(boundaries, [])

    def test_knowledge_point_headings_create_real_group_boundaries(self):
        boundaries = _subsection_boundaries([
            item("知识点一 有理数的加法", 20, 120),
            item("知识点二 加法运算律", 20, 620),
        ], 1000)
        self.assertEqual(
            [(boundary["number"], boundary["y"], boundary["starts_segment"]) for boundary in boundaries],
            [("一", 120, True), ("二", 620, True)],
        )

    def test_knowledge_point_heading_is_promoted_to_major_not_subsection(self):
        group = _knowledge_point_group("知识点四 绝对值")
        self.assertEqual(group["number"], "四")
        self.assertEqual(group["title"], "知识点四 绝对值")
        self.assertEqual(_knowledge_point_group("知识点12 综合测评")["number"], "十二")
        self.assertIsNone(_knowledge_point_group("（二）课外阅读"))
        self.assertIsNone(_knowledge_point_group("4. 普通小题"))

    def test_later_knowledge_point_keeps_numbered_prefix_in_active_group(self):
        segments = _subsection_segments(
            cutter,
            [item("16. 上一知识点的题", 25, 100), item("知识点二 加法运算律", 20, 600)],
            1000, 1000, active_title="知识点一", active_number="一",
        )
        self.assertEqual(
            [(segment["y0"], segment["y1"], segment["subsection_number"]) for segment in segments],
            [(0, 600, "一"), (600, 1000, "二")],
        )

    def test_explanatory_preamble_is_not_merged_into_first_question(self):
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        minors = [{
            "num": 1,
            "number_detected": True,
            "_y_start": 0,
            "_y_end": 900,
            "_anchor_y": 650,
            "crop_img": page[:900].copy(),
        }]
        ocr_items = [
            item("有理数加法的交换律", 30, 160),
            item("有理数加法的结合律", 30, 360),
            item("1. 下列说法正确的是", 30, 650),
        ]

        trimmed = _trim_explanatory_preamble_from_first_minor(minors, ocr_items, page)

        self.assertTrue(trimmed)
        self.assertEqual(minors[0]["_y_start"], 642)
        self.assertEqual(minors[0]["crop_img"].shape[:2], (258, 700))
        self.assertTrue(minors[0]["_explanatory_preamble_removed"])
        self.assertEqual(minors[0]["_boundary_source"], "ocr_number_after_explanation")


if __name__ == "__main__":
    unittest.main()
