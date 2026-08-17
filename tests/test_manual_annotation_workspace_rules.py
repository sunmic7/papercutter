import unittest

import cut_questions_v4 as cutter
from review_tools.manual_annotation_workspace import (
    _primary_number_boundaries,
    _subsection_boundaries,
    _subsection_segments,
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


if __name__ == "__main__":
    unittest.main()
