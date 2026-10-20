import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cut_questions_v4 as cutter


def item(text, x, y, width=180, height=30, conf=0.98):
    return {
        'text': text,
        'bbox': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        'conf': conf,
    }


class MinorNumberTests(unittest.TestCase):
    def test_decimal_is_not_question_number(self):
        self.assertIsNone(cutter.extract_minor_idx('4.8-1.6='))
        self.assertIsNone(cutter.extract_minor_idx('35.2元+15.7元='))

    def test_ocr_joined_number_after_question_anchor_is_recovered(self):
        items = [
            item('4. 上一题', 20, 100),
            item('5.36个铁圆锥，可以熔铸成多少个圆柱', 20, 260),
        ]
        boundaries = cutter.find_minor_boundaries(items, 700, 500)
        self.assertEqual([boundary['num'] for boundary in boundaries], [4, 5])

    def test_supported_number_styles(self):
        self.assertEqual(cutter.extract_minor_idx('7. 如图'), 7)
        self.assertEqual(cutter.extract_minor_idx('（3）判断'), 3)
        self.assertEqual(cutter.extract_minor_idx('⑩ 解决问题'), 10)

    def test_unit_name_reuses_precomputed_structure_ocr(self):
        with tempfile.TemporaryDirectory() as root:
            question_dir = os.path.join(root, '题目')
            os.makedirs(question_dir)
            page_path = os.path.join(question_dir, '1.jpg')
            open(page_path, 'wb').close()
            with patch.object(cutter, 'ocr_image', side_effect=AssertionError('不应重复高精度 OCR')):
                name = cutter.detect_unit_name(
                    root,
                    4,
                    root,
                    precomputed_ocr={page_path: [item('第4课时身高统计', 20, 20)]},
                )
        self.assertEqual(name, '第四课时身高统计')

    def test_cross_column_and_negative_coordinates_are_rejected(self):
        items = [
            item('1. 左栏题', 20, 100),
            item('2. 左栏题', 24, 260),
            item('3. 另一栏题', 700, 420),
            item('4. 负坐标噪声', -200, 580),
        ]
        boundaries = cutter.find_minor_boundaries(items, 600, 800)
        self.assertEqual([boundary['num'] for boundary in boundaries], [1, 2])

    def test_random_jump_is_filtered(self):
        items = [
            item('1. 第一题', 20, 100),
            item('2. 第二题', 20, 220),
            item('88. 正文误识别', 20, 360),
            item('3. 第三题', 20, 500),
        ]
        boundaries = cutter.find_minor_boundaries(items, 600, 800)
        self.assertEqual([boundary['num'] for boundary in boundaries], [1, 2, 3])

    def test_tightly_spaced_judgment_questions_are_not_merged(self):
        items = [
            item('2. 判断题', 20, 100, height=55),
            item('3. 判断题', 18, 182, height=55),
            item('4. 判断题', 20, 275, height=55),
        ]
        boundaries = cutter.find_minor_boundaries(items, 600, 500)
        self.assertEqual([boundary['num'] for boundary in boundaries], [2, 3, 4])

    def test_short_last_judgment_question_is_emitted(self):
        image = np.full((210, 700, 3), 255, dtype=np.uint8)
        items = [
            item('1. 判断题', 20, 80, height=28),
            item('2. 判断题', 20, 176, height=28),
        ]
        cuts = cutter.split_major_into_minors(image, ocr_items=items)
        self.assertEqual([cut['num'] for cut in cuts], [1, 2])
        self.assertEqual(cuts[0]['_y_start'], 0)
        self.assertGreater(cuts[0]['_y_end'], cuts[0]['_y_start'])
        self.assertEqual(cuts[-1]['_y_end'], image.shape[0])

    def test_multiple_question_boundaries_reuse_one_content_mask(self):
        image = np.full((720, 900, 3), 255, dtype=np.uint8)
        for y in (90, 300, 520):
            cv2.putText(
                image, f'{(y - 90) // 210 + 1}. QUESTION', (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
            )
        items = [
            item('1. 第一题', 20, 70),
            item('2. 第二题', 20, 280),
            item('3. 第三题', 20, 500),
        ]

        with patch.object(cutter, '_content_mask', wraps=cutter._content_mask) as content_mask:
            cuts = cutter.split_major_into_minors(image, ocr_items=items)

        self.assertEqual([cut['num'] for cut in cuts], [1, 2, 3])
        self.assertEqual(content_mask.call_count, 1)

    def test_parenthesized_subparts_do_not_split_primary_question(self):
        items = [
            item('1. 第一题', 20, 100),
            item('（1）子问', 60, 180),
            item('（2）子问', 60, 260),
            item('2. 第二题', 20, 420),
        ]
        boundaries = cutter.find_minor_boundaries(items, 600, 700)
        self.assertEqual([boundary['num'] for boundary in boundaries], [1, 2])

    def test_parenthesized_items_are_used_when_no_primary_number_exists(self):
        items = [
            item('（1）应用题', 30, 100),
            item('（2）应用题', 30, 300),
        ]
        boundaries = cutter.find_minor_boundaries(items, 600, 700)
        self.assertEqual([boundary['num'] for boundary in boundaries], [1, 2])

    def test_circled_object_labels_do_not_split_one_question(self):
        items = [
            item('①', 30, 120, width=30),
            item('④', 30, 360, width=30),
        ]
        self.assertEqual(cutter.find_minor_boundaries(items, 600, 700), [])


class SourceMetadataTests(unittest.TestCase):
    def test_large_shared_preamble_is_attached_to_later_short_subquestion(self):
        page = np.full((900, 600, 3), 255, dtype=np.uint8)
        page[20:500, 30:570] = 180  # Stand-in for a chart/table shared material.
        minors = [
            {
                'num': 1, 'crop_img': page[0:640, :], '_anchor_y': 510,
                '_source_box': [0, 0, 600, 640], '_major_box': [0, 0, 600, 800],
                '_source_path': 'page.jpg', '_source_size': [600, 900],
            },
            {
                'num': 2, 'crop_img': page[700:760, :], '_anchor_y': 700,
                '_source_box': [0, 700, 600, 760], '_major_box': [0, 0, 600, 800],
                '_source_path': 'page.jpg', '_source_size': [600, 900],
            },
        ]
        result, stats = cutter.attach_shared_context_to_dependent_minors(minors, page)
        self.assertEqual(stats['attached'], 1)
        self.assertTrue(result[1]['_shared_context_attached'])
        self.assertEqual(len(result[1]['_source_segments']), 2)
        self.assertGreater(result[1]['crop_img'].shape[0], 500)

    def test_regular_short_list_without_large_preamble_is_unchanged(self):
        page = np.full((500, 400, 3), 255, dtype=np.uint8)
        minors = [
            {'num': 1, 'crop_img': page[0:80], '_anchor_y': 35, '_source_box': [0, 0, 400, 80], '_major_box': [0, 0, 400, 500]},
            {'num': 2, 'crop_img': page[90:150], '_anchor_y': 90, '_source_box': [0, 90, 400, 150], '_major_box': [0, 0, 400, 500]},
        ]
        result, stats = cutter.attach_shared_context_to_dependent_minors(minors, page)
        self.assertEqual(stats['attached'], 0)
        self.assertEqual(result[1]['crop_img'].shape[0], 60)

    def test_cross_column_material_is_attached_to_real_first_numbered_item(self):
        material = {
            'crop_img': np.full((60, 100, 3), 230, dtype=np.uint8),
            'number_detected': False, '_boundary_source': 'whole_block',
            '_source_path': 'page.jpg', '_source_size': [400, 300],
            '_source_box': [0, 120, 200, 280],
        }
        first = {
            'crop_img': np.full((30, 80, 3), 180, dtype=np.uint8),
            'number_detected': True, 'num': 1,
            '_source_path': 'page.jpg', '_source_size': [400, 300],
            '_source_box': [200, 0, 400, 90],
        }
        self.assertTrue(cutter.attach_leading_whole_block_to_first_numbered_minor(material, first))
        self.assertEqual(first['_shared_context_reason'], 'preceding_cross_column_whole_block')
        self.assertEqual(len(first['_source_segments']), 2)
        self.assertEqual(first['crop_img'].shape[0], 90)
        self.assertTrue(np.all(first['crop_img'][59] == 230))
        self.assertTrue(np.all(first['crop_img'][60, :80] == 180))

    def test_untitled_right_spread_continues_current_left_major(self):
        self.assertEqual(cutter.continuation_target_major(5, 3, 5, 'right'), 5)
        self.assertEqual(cutter.continuation_target_major(5, 3, None, 'right'), 3)
        self.assertEqual(cutter.continuation_target_major(3, 5, None, 'left'), 5)

    def test_local_question_box_is_mapped_to_original_page(self):
        block = {
            'crop_img': np.full((300, 400, 3), 255, dtype=np.uint8),
            '_x_start': 500,
            '_y_start': 200,
        }
        minors = [{
            'num': 3,
            'crop_img': np.full((80, 400, 3), 255, dtype=np.uint8),
            '_y_start': 20,
            '_y_end': 100,
            '_boundary_source': 'ocr_number',
        }]
        cutter.attach_question_source_metadata(
            minors,
            block,
            'page.jpg',
            source_size=(1200, 1600),
            deskew_angle=1.25,
        )
        self.assertEqual(minors[0]['_source_box'], [500, 220, 900, 300])
        self.assertEqual(minors[0]['_major_box'], [500, 200, 900, 500])
        self.assertEqual(minors[0]['_page_deskew_angle'], 1.25)

    def test_each_question_keeps_its_own_page_angle_in_metadata(self):
        minor = {
            'num': 2,
            'number_detected': True,
            '_source_path': 'page-2.jpg',
            '_source_size': [1000, 1400],
            '_source_box': [0, 100, 1000, 300],
            '_major_box': [0, 0, 1000, 700],
            '_page_deskew_angle': -0.8,
            '_boundary_source': 'ocr_number',
        }
        metadata = cutter.build_question_metadata(
            minor, 1, '一、填空', 2,
            {'skew_angle': -0.8, 'content_bounds': (10, 12, 900, 180)},
            [],
        )
        self.assertEqual(metadata['source']['page_deskew_angle'], -0.8)
        self.assertEqual(metadata['normalization']['content_bounds'], [10, 12, 900, 180])

    def test_cross_page_merge_requires_bottom_to_top_continuation(self):
        previous = {
            'num': 3, 'number_detected': True,
            'crop_img': np.full((80, 100, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [100, 200],
            '_source_box': [0, 150, 100, 200], '_page_deskew_angle': 0.2,
        }
        continuation = {
            'num': 1, 'number_detected': False,
            'crop_img': np.full((60, 90, 3), 240, dtype=np.uint8),
            '_source_path': 'page-2.jpg', '_source_size': [100, 200],
            '_source_box': [0, 0, 100, 120], '_page_deskew_angle': -0.1,
        }
        self.assertTrue(cutter.should_merge_cross_page_question(previous, continuation))
        merged = cutter.merge_cross_page_question(previous, continuation)
        self.assertEqual(merged['crop_img'].shape[:2], (140, 100))
        self.assertEqual(len(merged['_source_segments']), 2)
        self.assertTrue(merged['_cross_page_merge'])

    def test_mid_page_unnumbered_block_is_not_cross_page_continuation(self):
        previous = {
            'number_detected': True, '_source_path': 'page-1.jpg',
            '_source_size': [100, 200], '_source_box': [0, 30, 100, 120],
        }
        continuation = {
            'number_detected': False, '_source_path': 'page-2.jpg',
            '_source_size': [100, 200], '_source_box': [0, 60, 100, 170],
        }
        self.assertFalse(cutter.should_merge_cross_page_question(previous, continuation))

    def test_facing_pages_in_one_scan_merge_left_bottom_to_right_top(self):
        previous = {
            'number_detected': True, '_source_path': 'spread.jpg',
            '_source_size': [1000, 600], '_source_box': [0, 520, 480, 600],
        }
        continuation = {
            'number_detected': False, '_source_path': 'spread.jpg',
            '_source_size': [1000, 600], '_source_box': [510, 0, 1000, 120],
        }
        self.assertTrue(cutter.should_merge_cross_page_question(previous, continuation))
        continuation['_source_box'] = [120, 0, 480, 120]
        self.assertFalse(cutter.should_merge_cross_page_question(previous, continuation))

    def test_unnumbered_cross_page_prefix_merges_but_later_questions_remain(self):
        previous = {
            'num': 3, 'number_detected': True,
            'crop_img': np.full((60, 100, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [100, 300],
            '_source_box': [0, 240, 100, 300],
        }
        continuation = {
            'num': 0, 'number_detected': False,
            'crop_img': np.full((80, 100, 3), 240, dtype=np.uint8),
            '_source_path': 'page-2.jpg', '_source_size': [100, 300],
            '_source_box': [0, 0, 100, 80],
            '_boundary_source': 'whole_block',
        }
        next_question = {
            'num': 4, 'number_detected': True,
            'crop_img': np.full((80, 100, 3), 220, dtype=np.uint8),
            '_source_path': 'page-2.jpg', '_source_size': [100, 300],
            '_source_box': [0, 85, 100, 165],
        }
        cuts = [continuation, next_question]
        merged, mode = cutter.merge_cross_page_continuation_prefix(previous, cuts)
        self.assertTrue(merged)
        self.assertEqual(mode, 'unnumbered_prefix')
        self.assertEqual([cut['num'] for cut in cuts], [4])
        self.assertTrue(previous['_cross_page_merge'])

    def test_numbered_first_cut_can_peel_large_leading_cross_page_fragment(self):
        previous = {
            'num': 3, 'number_detected': True,
            'crop_img': np.full((60, 100, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [100, 300],
            '_source_box': [0, 240, 100, 300],
        }
        first_new_question = {
            'num': 4, 'number_detected': True,
            'crop_img': np.full((220, 100, 3), 220, dtype=np.uint8),
            '_source_path': 'page-2.jpg', '_source_size': [100, 300],
            '_source_box': [0, 0, 100, 220], '_anchor_y': 120,
            '_y_start': 0, '_boundary_source': 'ocr_number',
        }
        later_question = {
            'num': 5, 'number_detected': True,
            'crop_img': np.full((70, 100, 3), 210, dtype=np.uint8),
            '_source_path': 'page-2.jpg', '_source_size': [100, 300],
            '_source_box': [0, 225, 100, 295],
        }
        cuts = [first_new_question, later_question]
        merged, mode = cutter.merge_cross_page_continuation_prefix(previous, cuts)
        self.assertTrue(merged)
        self.assertEqual(mode, 'peeled_prefix')
        self.assertEqual([cut['num'] for cut in cuts], [4, 5])
        self.assertGreater(first_new_question['_source_box'][1], 0)
        self.assertTrue(previous['_cross_page_merge'])

    def test_first_question_number_one_is_not_peeled_into_previous_page(self):
        minor = {
            'num': 1, 'number_detected': True,
            'crop_img': np.full((220, 100, 3), 220, dtype=np.uint8),
            '_source_size': [100, 300], '_source_box': [0, 0, 100, 220],
            '_anchor_y': 120,
        }
        self.assertIsNone(cutter.peel_leading_cross_page_fragment(minor))

    def test_failed_cross_page_evidence_does_not_trim_first_new_question(self):
        previous = {
            'num': 3, 'number_detected': True,
            'crop_img': np.full((60, 100, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [100, 300],
            '_source_box': [0, 80, 100, 150],
        }
        first_new_question = {
            'num': 4, 'number_detected': True,
            'crop_img': np.full((220, 100, 3), 220, dtype=np.uint8),
            '_source_path': 'page-2.jpg', '_source_size': [100, 300],
            '_source_box': [0, 0, 100, 220], '_anchor_y': 120,
            '_y_start': 0, '_boundary_source': 'ocr_number',
        }
        cuts = [first_new_question]
        merged, _ = cutter.merge_cross_page_continuation_prefix(previous, cuts)
        self.assertFalse(merged)
        self.assertEqual(first_new_question['_source_box'], [0, 0, 100, 220])
        self.assertEqual(first_new_question['crop_img'].shape[0], 220)


class SamePageThreeColumnContinuationTests(unittest.TestCase):
    def _inked(self, height, width, stripe_h=None):
        """A near-white canvas with a dark ink band near its top."""
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        band_h = stripe_h if stripe_h is not None else max(1, height // 3)
        canvas[:band_h, :] = 25
        return canvas

    def _column_previous(self, **overrides):
        previous = {
            'num': 34, 'number_detected': True,
            'crop_img': np.full((280, 320, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [1000, 1400],
            '_source_box': [0, 1120, 320, 1400], '_col': 'column_1',
            '_page_deskew_angle': 0.2,
        }
        previous.update(overrides)
        return previous

    def _column_fragment(self, **overrides):
        fragment = {
            'num': 0, 'number_detected': False,
            'crop_img': self._inked(160, 320),
            '_source_path': 'page-1.jpg', '_source_size': [1000, 1400],
            '_source_box': [340, 0, 660, 160], '_col': 'column_2',
            '_boundary_source': 'whole_block',
        }
        fragment.update(overrides)
        return fragment

    def test_three_column_top_fragment_merges_into_previous_column_last_question(self):
        previous = self._column_previous()
        fragment = self._column_fragment()
        self.assertTrue(
            cutter.should_merge_cross_page_question(previous, fragment)
        )
        merged = cutter.merge_cross_page_question(previous, fragment)
        self.assertEqual(merged['crop_img'].shape[:2], (440, 320))
        self.assertEqual(len(merged['_source_segments']), 2)
        self.assertTrue(merged['_cross_page_merge'])

    def test_three_column_numbered_top_is_not_merged(self):
        previous = self._column_previous()
        numbered = self._column_fragment(num=36, number_detected=True, _col='column_3')
        self.assertFalse(
            cutter.should_merge_cross_page_question(previous, numbered)
        )

    def test_same_column_bottom_to_top_is_not_merged(self):
        previous = self._column_previous(_col='column_2', _source_box=[340, 1120, 660, 1400])
        fragment = self._column_fragment(_col='column_2')
        self.assertFalse(
            cutter.should_merge_cross_page_question(previous, fragment)
        )

    def test_unnumbered_three_column_prefix_merges_but_later_questions_remain(self):
        previous = self._column_previous()
        continuation = self._column_fragment()
        next_question = {
            'num': 35, 'number_detected': True,
            'crop_img': np.full((80, 320, 3), 220, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [1000, 1400],
            '_source_box': [340, 165, 660, 245], '_col': 'column_2',
        }
        cuts = [continuation, next_question]
        merged, mode = cutter.merge_cross_page_continuation_prefix(previous, cuts)
        self.assertTrue(merged)
        self.assertEqual(mode, 'unnumbered_prefix')
        self.assertEqual([cut['num'] for cut in cuts], [35])
        self.assertTrue(previous['_cross_page_merge'])

    def test_blank_column_top_margin_is_not_merged(self):
        previous = self._column_previous()
        blank = self._column_fragment(
            crop_img=np.full((160, 320, 3), 255, dtype=np.uint8),
        )
        self.assertFalse(
            cutter.should_merge_cross_page_question(previous, blank)
        )

    def test_three_column_numbered_first_cut_can_peel_large_leading_fragment(self):
        previous = self._column_previous()
        first_new_question = {
            'num': 35, 'number_detected': True,
            'crop_img': self._inked(220, 320),
            '_source_path': 'page-1.jpg', '_source_size': [1000, 1400],
            '_source_box': [340, 0, 660, 220], '_anchor_y': 120,
            '_y_start': 0, '_col': 'column_2',
            '_boundary_source': 'ocr_number',
        }
        later_question = {
            'num': 36, 'number_detected': True,
            'crop_img': np.full((70, 320, 3), 210, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [1000, 1400],
            '_source_box': [340, 225, 660, 295], '_col': 'column_2',
        }
        cuts = [first_new_question, later_question]
        merged, mode = cutter.merge_cross_page_continuation_prefix(previous, cuts)
        self.assertTrue(merged)
        self.assertEqual(mode, 'peeled_prefix')
        self.assertEqual([cut['num'] for cut in cuts], [35, 36])
        self.assertGreater(first_new_question['_source_box'][1], 0)
        self.assertTrue(previous['_cross_page_merge'])

    def test_previous_column_key_maps_reading_order(self):
        self.assertEqual(cutter._previous_column_key('column_2'), 'column_1')
        self.assertEqual(cutter._previous_column_key('column_3'), 'column_2')
        self.assertIsNone(cutter._previous_column_key('column_1'))
        self.assertIsNone(cutter._previous_column_key('left'))
        self.assertIsNone(cutter._previous_column_key('right'))
        self.assertIsNone(cutter._previous_column_key(''))

    def test_self_contained_option_question_needs_three_option_lines(self):
        ocr = [
            item('x的值为', 340, 20, width=120),
            item('A.2', 340, 300),
            item('B.-2', 460, 300),
            item('C.±2', 580, 300),
            item('D.不确定', 700, 300),
        ]
        box = [340, 0, 900, 350]
        self.assertTrue(cutter._self_contained_option_question(ocr, box))
        self.assertFalse(
            cutter._self_contained_option_question(ocr, [340, 0, 900, 200])
        )
        partial = [item('C.延长线段AB', 340, 40), item('D.画直线AB=5cm', 340, 90)]
        self.assertFalse(
            cutter._self_contained_option_question(partial, box)
        )

    def test_peel_keeps_self_contained_choice_question_in_first_cut(self):
        first_new_question = {
            'num': 6, 'number_detected': True,
            'crop_img': self._inked(220, 320),
            '_source_size': [1000, 1400], '_source_box': [340, 0, 660, 220],
            '_anchor_y': 120, '_y_start': 0, '_col': 'column_2',
        }
        ocr = [
            item('x的值为', 340, 20, width=120),
            item('A.2', 340, 40),
            item('B.-2', 400, 40),
            item('C.±2', 460, 40),
            item('D.不确定', 520, 40),
        ]
        fragment = cutter.peel_leading_cross_page_fragment(
            first_new_question, ocr_items=ocr,
        )
        self.assertIsNone(fragment)
        self.assertEqual(first_new_question['_source_box'], [340, 0, 660, 220])

    def test_peel_still_merges_partial_option_continuation(self):
        first_new_question = {
            'num': 8, 'number_detected': True,
            'crop_img': self._inked(220, 320),
            '_source_size': [1000, 1400], '_source_box': [340, 0, 660, 220],
            '_anchor_y': 120, '_y_start': 0, '_col': 'column_2',
        }
        ocr = [
            item('C.延长线段AB到点C', 340, 40),
            item('D.画直线AB=5cm', 340, 90),
        ]
        fragment = cutter.peel_leading_cross_page_fragment(
            first_new_question, ocr_items=ocr,
        )
        self.assertIsNotNone(fragment)
        self.assertEqual(fragment['_source_box'], [340, 0, 660, 114])

    def test_merge_sequential_questions_ignores_full_option_row(self):
        previous = {
            'num': 7, 'number_detected': True,
            'crop_img': np.full((60, 100, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [300, 300],
            '_source_box': [0, 240, 100, 300], '_col': 'column_1',
        }
        first_new_question = {
            'num': 8, 'number_detected': True,
            'crop_img': self._inked(220, 100),
            '_source_path': 'page-1.jpg', '_source_size': [300, 300],
            '_source_box': [110, 0, 210, 220], '_anchor_y': 120,
            '_y_start': 0, '_col': 'column_2',
            '_boundary_source': 'ocr_number',
        }
        ocr = [
            item('A.50°', 115, 30, width=30),
            item('B.60°', 150, 30, width=30),
            item('C.140°', 185, 30, width=30),
            item('D.150°', 115, 60, width=30),
        ]
        merged, mode = cutter.merge_cross_page_continuation_prefix(
            previous, [first_new_question], ocr_items=ocr,
        )
        self.assertTrue(merged)
        self.assertEqual(mode, 'peeled_prefix')
        self.assertTrue(previous['_cross_page_merge'])

    def test_merge_skips_single_question_when_number_gap_and_full_options(self):
        previous = {
            'num': 4, 'number_detected': True,
            'crop_img': np.full((60, 100, 3), 255, dtype=np.uint8),
            '_source_path': 'page-1.jpg', '_source_size': [300, 300],
            '_source_box': [0, 240, 100, 300], '_col': 'column_1',
        }
        first_new_question = {
            'num': 6, 'number_detected': True,
            'crop_img': self._inked(220, 100),
            '_source_path': 'page-1.jpg', '_source_size': [300, 300],
            '_source_box': [110, 0, 210, 220], '_anchor_y': 120,
            '_y_start': 0, '_col': 'column_2',
            '_boundary_source': 'ocr_number',
        }
        ocr = [
            item('A.2', 115, 30, width=30),
            item('B.-2', 150, 30, width=30),
            item('C.±2', 185, 30, width=30),
            item('D.不确定', 115, 60, width=60),
        ]
        merged, mode = cutter.merge_cross_page_continuation_prefix(
            previous, [first_new_question], ocr_items=ocr,
        )
        self.assertFalse(merged)
        self.assertEqual(first_new_question['_source_box'], [110, 0, 210, 220])
        self.assertNotIn('_cross_page_merge', previous)


class ColumnTests(unittest.TestCase):
    def test_three_column_question_layout_is_split_left_to_right(self):
        """Three independent exercise columns must never share one cut region."""
        page = np.full((1000, 1500, 3), 255, dtype=np.uint8)
        items = []
        for column_index, x in enumerate((35, 535, 1035), start=1):
            for row_index, y in enumerate((130, 300, 470, 640), start=1):
                cv2.rectangle(page, (x, y), (x + 355, y + 22), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=355, height=22,
                ))

        regions = cutter.split_question_layout_columns(page, items)

        self.assertEqual(len(regions), 3)
        self.assertLess(regions[0][1], 560)
        self.assertGreater(regions[1][0], 440)
        self.assertLess(regions[1][1], 1060)
        self.assertGreater(regions[2][0], 940)
        self.assertEqual(
            [len(region[2]) for region in regions], [4, 4, 4],
        )

    def test_verified_three_column_layout_uses_both_global_gutters(self):
        """A balanced three-column page is resolved as one layout, not greedily."""
        page = np.full((960, 1800, 3), 255, dtype=np.uint8)
        items = []
        for column_index, x in enumerate((50, 650, 1250), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 430, y + 24), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=430, height=24,
                ))

        regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        self.assertEqual([len(region[2]) for region in regions], [4, 4, 4])
        self.assertLess(regions[0][1], 650)
        self.assertGreater(regions[1][0], 550)
        self.assertLess(regions[1][1], 1250)
        self.assertGreater(regions[2][0], 1150)

    def test_column_bounds_prevent_refined_box_from_crossing_gutter(self):
        page = np.full((500, 1500, 3), 255, dtype=np.uint8)
        cuts = [{
            '_source_box': [80, 120, 980, 260],
            'crop_img': page[120:260, 80:980].copy(),
        }]

        cutter.trim_cuts_to_column_bounds(cuts, page, [0, 500])

        self.assertEqual(cuts[0]['_source_box'], [80, 120, 500, 260])
        self.assertEqual(cuts[0]['crop_img'].shape[:2], (140, 420))
        self.assertTrue(cuts[0]['_column_bounds_trimmed'])

    def test_cross_gutter_minor_and_next_major_are_split(self):
        merged = item('2. 列式计算。六、下面是调查结果', 180, 240, width=280)
        left, right = cutter._split_cross_column_items([merged], 300)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertTrue(left[0]['text'].startswith('2.'))
        self.assertTrue(right[0]['text'].startswith('六、'))

    def test_spanning_body_text_rejects_false_column_split_from_parallel_diagrams(self):
        """两幅并排图不能让下方横跨整页的一道题被竖切成两题。"""
        page = np.full((2706, 1914, 3), 255, dtype=np.uint8)
        items = [
            item('20', 753, 439, width=56),
            item('20', 1146, 439, width=55),
            item('15一个圆柱体容器内，放有一个长方体铁块，现在打开一个水龙头往容器中', 346, 666, width=1444, height=51),
            item('注水3分钟后，水恰好没过长方体的顶面，又过了18分钟后，水灌满了容器。已知', 257, 732, width=1536, height=56),
            item('容器的高度是50厘米，长方体的高度是20厘米，那么长方体底面积与容器底面积', 257, 800, width=1538, height=57),
            item('的比是多少？', 259, 873, width=245, height=52),
        ]
        with patch.object(cutter, 'detect_columns', return_value=1007), \
             patch.object(cutter, '_visual_column_split', return_value=None):
            blocks = cutter.split_page_into_majors(page, items)
        self.assertEqual([block['_x_start'] for block in blocks], [0])

    def test_wide_spread_prefers_visual_binding_over_bad_ocr_gap(self):
        page = np.full((700, 1200, 3), 255, dtype=np.uint8)
        with patch.object(cutter, 'detect_columns', return_value=470), \
             patch.object(cutter, '_visual_column_split', return_value=600):
            blocks = cutter.split_page_into_majors(page, [], profile='exam')
        self.assertEqual([block['_x_start'] for block in blocks], [0, 600])

    def test_multi_digit_chinese_major_number_is_supported(self):
        self.assertEqual(cutter.extract_major_idx('三十一、数学游戏'), 31)
        page = np.full((700, 600, 3), 255, dtype=np.uint8)
        blocks = cutter._split_single_col_majors(
            page,
            [item('三十一、数学游戏', 25, 100), item('1. 第一题', 30, 230)],
            profile='holiday',
        )
        self.assertEqual(blocks[0]['idx'], 31)

    def test_crossing_title_is_split_between_columns(self):
        crossing = item('一、左栏标题三、右栏标题', 100, 80, width=800)
        left, right = cutter._split_cross_column_items([crossing], 500)
        self.assertEqual(cutter.extract_major_idx(left[0]['text']), 1)
        self.assertEqual(cutter.extract_major_idx(right[0]['text']), 3)

    def test_content_above_later_major_is_kept_as_continuation(self):
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        items = [
            item('2. 上一大题续题', 30, 90),
            item('3. 上一大题续题', 30, 250),
            item('四、对号入座', 25, 470),
            item('1. 本大题第一题', 30, 560),
        ]
        blocks = cutter._split_single_col_majors(page, items)
        self.assertEqual(blocks[0]['title'], '整页')
        self.assertEqual(blocks[1]['idx'], 4)
        self.assertLess(blocks[0]['crop_img'].shape[0], blocks[1]['crop_img'].shape[0])

    def test_page_header_above_first_major_is_not_continuation(self):
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        items = [
            item('第一单元检测卷', 180, 50),
            item('一、认真填空', 25, 180),
            item('1. 第一题', 30, 270),
        ]
        blocks = cutter._split_single_col_majors(page, items)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['idx'], 1)

    def test_holiday_activity_heading_becomes_own_block(self):
        page = np.full((1200, 700, 3), 255, dtype=np.uint8)
        items = [
            item('五、算一算，填一填。', 25, 100),
            item('数学游戏', 180, 520),
            item('一张正方形纸剪去一个角', 35, 640),
            item('假期体验', 180, 880),
        ]
        blocks = cutter._split_single_col_majors(
            page, items, profile='holiday',
        )
        self.assertEqual([block['title'] for block in blocks], [
            '五、算一算，填一填。', '数学游戏', '假期体验',
        ])

    def test_exam_major_keeps_content_until_next_title(self):
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        items = [
            item('一、填空题', 25, 100),
            item('8. 最后一题', 25, 730),
            item('二、判断题', 25, 820),
        ]
        blocks = cutter._split_single_col_majors(page, items, profile='exam')
        self.assertGreater(blocks[0]['crop_img'].shape[0], 700)

    def test_knowledge_point_heading_becomes_major_block(self):
        """周测/练习册以“知识点一 …”为节标题且无中文大题号；标题应成为
        独立大题块，而不是被并进相邻小题的裁图。"""
        page = np.full((1600, 700, 3), 255, dtype=np.uint8)
        items = [
            item('知识点一 有理数的加法', 25, 120),
            item('1. 第一题', 30, 220),
            item('（1）子问', 90, 300),
            item('知识点二 加法运算律', 25, 620),
            item('1. 第二题', 30, 720),
            item('2. 第三题', 30, 880),
        ]
        blocks = cutter._split_single_col_majors(page, items)
        self.assertEqual([block['idx'] for block in blocks], [3001, 3002])
        self.assertEqual(
            [block['title'] for block in blocks],
            ['知识点一 有理数的加法', '知识点二 加法运算律'],
        )
        # 标题行保留在各自大题块的顶部，不流入相邻块。
        self.assertLess(blocks[0]['crop_img'].shape[0], blocks[1]['crop_img'].shape[0])

    def test_knowledge_point_heading_works_alongside_major_numbers(self):
        """同一列既有一、二大题号又有知识点标题时，两类边界并存且去重。"""
        page = np.full((1400, 700, 3), 255, dtype=np.uint8)
        items = [
            item('一、选择题', 25, 100),
            item('1. 第一题', 30, 200),
            item('知识点一 有理数的运算', 25, 500),
            item('1. 第二题', 30, 600),
        ]
        blocks = cutter._split_single_col_majors(page, items)
        self.assertEqual([block['idx'] for block in blocks], [1, 3001])

    def test_inline_knowledge_text_is_not_a_boundary(self):
        """正文里出现“知识点”的长句不应被当作节标题边界。"""
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        items = [
            item('本段讲的知识点是绝对值与相反数', 25, 120),
            item('1. 第一题', 30, 260),
        ]
        blocks = cutter._split_single_col_majors(page, items)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['title'], '整页')


class OlympiadFieldTests(unittest.TestCase):
    def test_colored_olympiad_fields_become_major_blocks(self):
        page = np.full((1600, 800, 3), 255, dtype=np.uint8)
        items = [
            item('从课本到奥数难题点拨·1年级', 80, 45, width=420),
            item('难题点拨①', 160, 300),
            item('1. 第一题', 35, 390),
            item('想一想 做一做', 160, 760),
            item('1. 第二题', 35, 850),
            item('难题点拨②', 160, 1180),
        ]
        blocks = cutter._split_single_col_majors(
            page, items, profile='olympiad',
        )
        self.assertEqual(
            [block['title'] for block in blocks],
            ['难题点拨①', '想一想做一做', '难题点拨②'],
        )
        self.assertEqual(
            [block['idx'] for block in blocks], [2001, 2200, 2002],
        )

    def test_book_header_is_not_mistaken_for_field(self):
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        items = [item('从课本到奥数难题点拨·1年级', 80, 35, width=420)]
        self.assertEqual(
            cutter.find_olympiad_field_boundaries(items, page.shape[0]), [],
        )

    def test_split_ocr_heading_is_joined_before_matching(self):
        fields = cutter.find_olympiad_field_boundaries([
            item('想一想', 160, 350, width=110),
            item('做一做', 285, 350, width=110),
        ], 1000)
        self.assertEqual([(field['text'], field['idx']) for field in fields], [
            ('想一想做一做', 2200),
        ])

    def test_lesson_opening_is_named_instead_of_unclassified(self):
        page = np.full((1300, 700, 3), 255, dtype=np.uint8)
        items = [
            item('第三讲', 250, 250),
            item('几个和第几个', 410, 250),
            item('这里是讲解文字', 50, 460),
            item('难题点拨', 160, 860),
        ]
        blocks = cutter._split_single_col_majors(
            page, items, profile='olympiad',
        )
        self.assertEqual(blocks[0]['title'], '讲解示例')
        self.assertEqual(blocks[0]['idx'], 2500)
        self.assertEqual(blocks[1]['title'], '难题点拨')

    def test_explanation_label_is_not_a_separate_field(self):
        page = np.full((1000, 700, 3), 255, dtype=np.uint8)
        items = [
            item('难题点拨', 150, 280),
            item('点拨', 150, 610),
            item('拓展', 150, 790),
        ]
        blocks = cutter._split_single_col_majors(
            page, items, profile='olympiad',
        )
        self.assertEqual([block['title'] for block in blocks], ['难题点拨'])


class CircledExerciseNumberTests(unittest.TestCase):
    """部分奥数书用 ①②③ 给整道练习编号，不印“一、二、三”。"""

    def build_items(self):
        return [
            item('①把动物按两只脚的、四只脚的和会游泳的分类。', 60, 600, width=520),
            item('②看图数一数，填一填。', 60, 1000, width=420),
            item('③', 900, 1400, width=40),                    # 图注：只有圈号
            item('如图①所示的图形', 60, 1600, width=400),        # 不在行首
            item('④这一条离左边太远', 700, 1750, width=300),     # 不在栏首
        ]

    def test_only_stem_lines_become_boundaries(self):
        fields = cutter.find_circled_number_boundaries(
            self.build_items(), crop_width=800, crop_height=2000,
        )
        self.assertEqual([field['y'] for field in fields], [600, 1000])
        self.assertEqual([field['idx'] for field in fields], [5001, 5002])
        self.assertTrue(fields[0]['text'].startswith('① 把动物'))

    def test_general_profile_keeps_the_page_in_one_block(self):
        items = self.build_items()
        page = np.full((2000, 800, 3), 255, dtype=np.uint8)
        general = cutter._split_single_col_majors(page, items, profile='general')
        self.assertEqual([block['title'] for block in general], ['整页'])
        olympiad = cutter._split_single_col_majors(page, items, profile='olympiad')
        self.assertEqual(len(olympiad), 2)
        self.assertTrue(olympiad[0]['title'].startswith('① 把动物按两只脚的'))
        self.assertTrue(olympiad[1]['title'].startswith('② 看图数一数'))


class SideBySideMinorTests(unittest.TestCase):
    """横排小问：1. 2. 3. 在同一条水平线上并排，只能按 x 切开。"""

    PAGE = np.full((479, 2407, 3), 255, dtype=np.uint8)

    def side_by_side_items(self):
        # 取自 56260570「七、数一数，填一填」的真实坐标：题号 2./3. 的 y
        # 只差 3px，题号 1. 被 OCR 整条漏读。
        return [
            item('七、数一数，填一填。（9分）', 252, 7, width=690),
            item('有()个角', 413, 369, width=377),
            item('2.', 878, 109, width=63),
            item('有( )个平行四边形', 929, 373, width=676),
            item('3.', 1541, 112, width=66),
            item(')个梯形', 1805, 386, width=221),
        ]

    def test_columns_follow_left_edges(self):
        spans = cutter._side_by_side_columns(self.side_by_side_items(), 2407, 479)
        self.assertEqual(len(spans), 3)
        starts = [span[0] for span in spans]
        self.assertEqual(starts, sorted(starts))

    def test_three_side_by_side_questions_are_cut_and_numbered(self):
        minors = cutter.split_major_into_minors(
            self.PAGE, self.side_by_side_items(), profile='exam',
        )
        self.assertEqual([minor['num'] for minor in minors], [1, 2, 3])
        self.assertTrue(all(minor.get('_side_by_side') for minor in minors))
        starts = [minor['_x_start'] for minor in minors]
        self.assertEqual(starts, sorted(starts))
        # 题号 3 落在最后一列里，不能被前两列吞掉。
        last = minors[-1]
        self.assertLessEqual(last['_x_start'], 1541)
        self.assertLess(1541, last['_x_end'])

    def test_vertical_block_is_left_to_the_y_based_splitter(self):
        page = np.full((900, 700, 3), 255, dtype=np.uint8)
        items = [
            item('1. 计算', 40, 60, width=200),
            item('2. 判断', 40, 260, width=200),
            item('3. 选择', 40, 460, width=200),
        ]
        self.assertEqual(cutter._side_by_side_columns(items, 700, 900), [])
        minors = cutter.split_major_into_minors(page, items, profile='exam')
        self.assertFalse(any(minor.get('_side_by_side') for minor in minors))

    def test_source_box_narrows_to_the_column(self):
        block = {
            'crop_img': np.full((400, 1200, 3), 255, dtype=np.uint8),
            '_x_start': 100,
            '_y_start': 200,
        }
        minors = [{
            'num': 3,
            'crop_img': np.full((400, 400, 3), 255, dtype=np.uint8),
            '_x_start': 800,
            '_x_end': 1200,
            '_y_start': 0,
            '_y_end': 400,
        }]
        cutter.attach_question_source_metadata(
            minors, block, 'page.jpg', source_size=(1500, 2000),
        )
        self.assertEqual(minors[0]['_source_box'], [900, 200, 1300, 600])


class BookCompletionTests(unittest.TestCase):
    def test_book_marker_matches_same_source_without_ocr(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'source_book')
            output = os.path.join(root, 'output_book')
            os.makedirs(source)
            os.makedirs(output)
            with open(os.path.join(source, '1.jpg'), 'wb') as file:
                file.write(b'image-a')
            fingerprint = cutter.source_book_fingerprint(source)
            cutter.write_book_completion_marker(output, fingerprint, 3)
            self.assertTrue(cutter.is_completed_book(source, output)[0])
            with open(os.path.join(source, '2.jpg'), 'wb') as file:
                file.write(b'image-b')
            self.assertFalse(cutter.is_completed_book(source, output)[0])

    def test_legacy_book_is_adopted_only_when_every_unit_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'source_book')
            output = os.path.join(root, 'output_book')
            os.makedirs(source)
            os.makedirs(output)
            with open(os.path.join(source, '1.jpg'), 'wb') as file:
                file.write(b'image-a')
            for name in ('第一单元', '第二单元'):
                unit_dir = os.path.join(output, name)
                os.makedirs(unit_dir)
                cutter.write_completion_marker(unit_dir, 5, 0, 0)
            fingerprint = cutter.source_book_fingerprint(source)
            self.assertTrue(cutter.adopt_legacy_completed_book(
                source, output, 'general', fingerprint,
            ))
            self.assertTrue(cutter.is_completed_book(source, output)[0])

    def test_legacy_partial_book_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'source_book')
            output = os.path.join(root, 'output_book')
            os.makedirs(source)
            os.makedirs(output)
            with open(os.path.join(source, '1.jpg'), 'wb') as file:
                file.write(b'image-a')
            complete_unit = os.path.join(output, '第一单元')
            incomplete_unit = os.path.join(output, '第二单元')
            os.makedirs(complete_unit)
            os.makedirs(incomplete_unit)
            cutter.write_completion_marker(complete_unit, 5, 0, 0)
            fingerprint = cutter.source_book_fingerprint(source)
            self.assertFalse(cutter.adopt_legacy_completed_book(
                source, output, 'general', fingerprint,
            ))
            self.assertFalse(cutter.is_completed_book(source, output)[0])

    def test_explicitly_incomplete_paperflow_project_cannot_skip(self):
        """显式未完成的 JSON 不能让下一轮整本跳过。"""
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'source_book')
            output = os.path.join(root, 'output_book')
            os.makedirs(source)
            os.makedirs(output)
            image = os.path.join(source, '1.jpg')
            with open(image, 'wb') as file:
                file.write(b'image-a')
            fingerprint = cutter.source_book_fingerprint(source)
            path = os.path.join(source, 'PaperFlow标注项目.json')
            project = {
                'kind': 'papercutter_manual_annotation',
                'preannotation_mode': 'paperflow',
                'algorithm_version': cutter.ALGORITHM_VERSION,
                'source_fingerprint': fingerprint,
                'status': 'incomplete',
                'pages': [],
                'answer_pages': [],
            }
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(project, file)
            self.assertFalse(cutter.is_completed_book(
                source, output, fingerprint, output_mode='paperflow',
            )[0])
            project['status'] = 'complete'
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(project, file)
            self.assertTrue(cutter.is_completed_book(
                source, output, fingerprint, output_mode='paperflow',
            )[0])


class PaperFlowProjectExportTests(unittest.TestCase):
    def test_flat_document_ignored_pages_are_preserved_in_project(self):
        """封面/说明页虽不参加切题，也必须写入 ignored_pages。"""
        with tempfile.TemporaryDirectory() as root:
            question = os.path.join(root, '002.jpg')
            answer = os.path.join(root, '003.jpg')
            ignored = os.path.join(root, '001.jpg')
            for path in (question, answer, ignored):
                with open(path, 'wb') as file:
                    file.write(b'image')
            unit = {
                'scope_id': 'scope-unit-1',
                'unit_name': '第一单元',
                'unit_profile': 'general',
                'pages_classification': {question: 'question'},
                'ignored_img_paths': [ignored],
                'major_questions': {
                    1: {
                        'title': '一、计算',
                        'minors': [{
                            '_source_path': question,
                            '_source_box': [1, 2, 100, 120],
                            'num': 1,
                            '_pf_id': 'Q-unit-1',
                            '_number_confidence': 0.99,
                            '_detector_confidence': 0.87,
                        }, {
                            # A cross-column whole block with the same number
                            # is a continuation, never a second match target.
                            '_source_path': question,
                            '_source_box': [101, 2, 200, 120],
                            'num': 1,
                            '_pf_id': 'Q-unit-1-continuation',
                            '_boundary_source': 'whole_block',
                        }],
                    },
                },
                'answer_records': [{
                    'source': answer,
                    'source_box': [3, 4, 90, 100],
                    'major_idx': 1,
                    'num': 1,
                    '_pf_id': 'A-unit-1',
                }],
                'all_answer_records': [{
                    'source': answer,
                    'source_box': [3, 4, 90, 100],
                    'major_idx': 1,
                    'num': 1,
                    '_pf_id': 'A-unit-1',
                }, {
                    # This answer belongs to another/unknown context.  It must
                    # still become a visible unassigned answer frame.
                    'source': answer,
                    'source_box': [91, 4, 180, 100],
                    'major_idx': 1,
                    'num': 2,
                }],
                'answer_img_paths': [answer],
                'pairs_data': [{
                    'q_id': 'Q-unit-1',
                    'a_ids': ['A-unit-1'],
                    'decision_band': 'auto',
                    'score': 0.99,
                }],
            }
            project = cutter.write_paperflow_project(root, [unit], 'general')

            self.assertEqual(project['status'], 'complete')
            self.assertEqual([page['image'] for page in project['ignored_pages']], ['001.jpg'])
            self.assertEqual(len(project['pages']), 1)
            self.assertEqual(len(project['answer_pages']), 1)
            self.assertEqual(project['pages'][0]['match_scope_id'], '')
            self.assertEqual(len(project['pages'][0]['match_scope_regions']), 2)
            self.assertEqual(project['answer_pages'][0]['match_scope_id'], '')
            self.assertEqual(len(project['answer_pages'][0]['match_scope_regions']), 1)
            question_records = project['pages'][0]['questions']
            self.assertTrue(question_records[0]['detector_assisted'])
            self.assertEqual(question_records[0]['detector_confidence'], 0.87)
            self.assertEqual(question_records[1]['region_role'], 'continuation')
            self.assertEqual(question_records[1]['continuation_of'], 'Q-unit-1')
            answer_records = project['answer_pages'][0]['answers']
            self.assertEqual(len(answer_records), 2)
            fallback = next(record for record in answer_records if record['number'] == '2')
            self.assertTrue(fallback['unmatched_answer'])
            self.assertEqual(fallback['match_scope_id'], '')
            self.assertEqual(len(project['pairs']), 1)
            self.assertTrue(cutter.is_completed_book(
                root, os.path.join(root, 'unused'), output_mode='paperflow',
            )[0])

    def test_empty_answer_page_gets_a_reviewable_fallback_frame(self):
        image = np.full((160, 240, 3), 255, dtype=np.uint8)
        with patch.object(cutter, 'cv2_imread', return_value=image), \
                patch.object(cutter, 'ocr_image', return_value=[]), \
                patch.object(cutter, 'detect_answer_blocks_with_model', return_value=[]):
            records = cutter.process_answer_pages(['answer.jpg'])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['fallback_full_page'])
        self.assertEqual(records[0]['source_box'], [0, 0, 240, 160])

    def test_force_refresh_bypasses_the_current_process_ocr_cache(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        stale = item('旧缓存', 8, 8)
        fresh = item('本次重识别', 12, 12)
        with patch.object(cutter, '_ocr_cache_location', return_value=('page-key', None, None)), \
                patch.object(cutter, 'cv2_imread', return_value=image), \
                patch.object(cutter, 'get_ocr_engine'), \
                patch.object(cutter, '_run_ocr', return_value=[fresh]) as run_ocr, \
                patch.object(cutter, '_ocr_memory_cache', {'page-key': [stale]}):
            records = cutter.ocr_image('answer.jpg', force_refresh=True)

        run_ocr.assert_called_once()
        self.assertEqual(records[0]['text'], '本次重识别')


class BatchRecoveryTests(unittest.TestCase):
    def test_year_mode_retries_then_skips_bad_unit_and_continues(self):
        """一个坏单元不能让后续教材整批停止。"""
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, '2012')
            output = os.path.join(root, 'output')
            first_book = os.path.join(source, '甲教材')
            second_book = os.path.join(source, '乙教材')
            os.makedirs(first_book)
            os.makedirs(second_book)
            bad_unit = os.path.join(first_book, '第1单元')
            calls = []
            original_context = dict(cutter.CURRENT_RUN_CONTEXT)

            def fake_process_book(book_dir, out_base, skip_unit_paths=None):
                calls.append((os.path.basename(book_dir), set(skip_unit_paths or set())))
                if os.path.abspath(book_dir) != os.path.abspath(first_book):
                    return
                cutter.CURRENT_RUN_CONTEXT.update({
                    'book_path': os.path.abspath(first_book),
                    'book_name': '甲教材',
                    'unit_path': bad_unit,
                    'unit_num': 1,
                    'unit_name': '第1课时',
                    'unit_position': 1,
                })
                if len([call for call in calls if call[0] == '甲教材']) <= 2:
                    raise RuntimeError('模拟坏图片')

            try:
                # 顺序模式（PAPERCUTTER_BOOK_WORKERS=1）下 mock 的 process_book
                # 才能在当前进程生效；并行子进程不会继承测试里的 patch。
                with patch.dict(os.environ, {'PAPERCUTTER_BOOK_WORKERS': '1'}), \
                        patch.object(cutter, 'process_book', side_effect=fake_process_book):
                    cutter.process_year(source, output)
            finally:
                cutter.CURRENT_RUN_CONTEXT.clear()
                cutter.CURRENT_RUN_CONTEXT.update(original_context)

            first_calls = [call for call in calls if call[0] == '甲教材']
            self.assertEqual(len(first_calls), 3)
            self.assertIn(os.path.abspath(bad_unit), first_calls[-1][1])
            self.assertIn('乙教材', [call[0] for call in calls])
            reports = [name for name in os.listdir(output) if name.startswith('_批量异常报告_')]
            self.assertEqual(len(reports), 1)

    def test_worker_wrapper_returns_error_dict_on_exception(self):
        """并行 worker 异常时把错误与单元上下文随结果返回，而不是中断整批。"""
        context = {'book_name': '甲教材', 'unit_path': 'C:/x/第1单元',
                   'unit_num': 1, 'unit_name': '第1课时', 'unit_position': 1}
        with patch.object(cutter, 'process_book', side_effect=RuntimeError('模拟坏图片')):
            with patch.dict(cutter.CURRENT_RUN_CONTEXT, context, clear=False):
                result = cutter._process_book_worker('C:/x/甲教材', 'C:/y', frozenset(), None)
        self.assertFalse(result['ok'])
        self.assertIn('模拟坏图片', result['error'])
        self.assertEqual(result['context']['book_name'], '甲教材')
        self.assertEqual(result['context']['unit_path'], 'C:/x/第1单元')

    def test_record_book_failure_first_failure_retries_then_skips(self):
        """单元首次异常重试整本，连续异常两次后加入跳过集合。"""
        attempts = {}
        skipped = {}
        failures = []
        attempts = cutter.defaultdict(int)
        skipped = cutter.defaultdict(set)
        book_path = os.path.abspath('C:/x/甲教材')
        context = {'book_name': '甲教材', 'unit_path': 'C:/x/第1单元',
                   'unit_name': '第1单元', 'unit_position': 1}
        first = cutter._record_book_failure(
            attempts, skipped, failures, book_path, 'C:/x/甲教材',
            dict(context), '错误一', '堆栈一',
        )
        self.assertEqual(first, 'retry')
        self.assertNotIn(os.path.abspath('C:/x/第1单元'), skipped.get(book_path, set()))
        second = cutter._record_book_failure(
            attempts, skipped, failures, book_path, 'C:/x/甲教材',
            dict(context), '错误二', '堆栈二',
        )
        self.assertEqual(second, 'retry')
        self.assertIn(os.path.abspath('C:/x/第1单元'), skipped[book_path])
        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0]['action'], '单元首次异常，自动重试')
        self.assertEqual(failures[1]['action'], '单元连续异常两次，已跳过并继续本书')


class ParallelismCapTests(unittest.TestCase):
    def test_weak_cpu_forces_sequential(self):
        """8 逻辑线程的机器只能顺序处理，不能并行 2 本。"""
        with patch.object(cutter.os, 'cpu_count', return_value=8), \
             patch.object(cutter, '_available_ram_gb', return_value=32.0):
            self.assertEqual(cutter._book_parallelism(2, 5), 1)

    def test_low_memory_caps_parallelism(self):
        """可用内存不足时，即使 CPU 够强也降低并行度，防止 OOM。"""
        with patch.object(cutter.os, 'cpu_count', return_value=32), \
             patch.object(cutter, '_available_ram_gb', return_value=6.0):
            self.assertEqual(cutter._book_parallelism(4, 6), 1)

    def test_memory_probe_missing_falls_back_to_cpu_cap(self):
        """内存探测失败时退回纯 CPU 上限，至少允许顺序处理。"""
        with patch.object(cutter.os, 'cpu_count', return_value=24), \
             patch.object(cutter, '_available_ram_gb', return_value=None):
            self.assertEqual(cutter._book_parallelism(4, 10), 4)

    def test_requested_one_is_always_sequential(self):
        """用户明确选 1 本时，硬件再强也只顺序处理。"""
        with patch.object(cutter.os, 'cpu_count', return_value=32), \
             patch.object(cutter, '_available_ram_gb', return_value=64.0):
            self.assertEqual(cutter._book_parallelism(1, 8), 1)

    def test_book_count_limits_parallelism(self):
        """并行本数不能超过实际教材数量。"""
        with patch.object(cutter.os, 'cpu_count', return_value=32), \
             patch.object(cutter, '_available_ram_gb', return_value=64.0):
            self.assertEqual(cutter._book_parallelism(4, 2), 2)


class PageClassifierTests(unittest.TestCase):
    def setUp(self):
        self.page = np.full((1200, 800, 3), 255, dtype=np.uint8)

    def test_weekly_measurement_book_uses_exam_profile_before_training_keyword(self):
        self.assertEqual(
            cutter.infer_book_profile('巅峰训练周周测七年级上册数学'),
            'exam',
        )

    def test_answer_page(self):
        result = cutter.classify_page(
            self.page,
            [item('参考答案', 250, 50), item('一、1.A 2.C', 50, 180)],
            'workbook',
        )
        self.assertEqual(result['kind'], 'answer')

    def test_instructional_page(self):
        result = cutter.classify_page(
            self.page,
            [
                item('课前导学', 250, 80),
                item('教材解读', 80, 240),
                item('知识点', 80, 360),
            ],
            'instructional',
        )
        self.assertEqual(result['kind'], 'instruction')

    def test_cover_page(self):
        result = cutter.classify_page(
            self.page,
            [
                item('ISBN 978-7-0000-0000-0', 300, 900),
                item('定价 20.00元', 300, 960),
                item('责任编辑 张三', 80, 120),
            ],
            'general',
        )
        self.assertEqual(result['kind'], 'non_content')

    def test_generic_book_is_refined_from_page_content(self):
        self.assertEqual(
            cutter.refine_profile_from_text('general', '暑假生活 假期体验 一年级数学'),
            'holiday',
        )
        self.assertEqual(
            cutter.refine_profile_from_text('general', '第一单元达标检测卷 时间60分钟 总分100分'),
            'exam',
        )
        self.assertEqual(
            cutter.refine_profile_from_text('workbook', '暑假生活'),
            'workbook',
        )

    def test_profile_detail_reports_why_a_book_was_classified(self):
        profile, word = cutter.infer_book_profile_detail('金版卷王名师面对面大试卷数学')
        self.assertEqual((profile, word), ('exam', '试卷'))
        profile, word = cutter.infer_book_profile_detail('原教材3')
        self.assertEqual((profile, word), ('general', ''))
        profile, reason = cutter.refine_profile_from_text_detail(
            'general', '时间：60分钟 总分：100分 考试',
        )
        self.assertEqual(profile, 'exam')
        self.assertIn('总分', reason)
        self.assertEqual(cutter.refine_profile_from_text_detail('exam', '暑假生活'), ('exam', ''))

    def test_detect_book_profile_skips_ocr_when_the_book_name_is_decisive(self):
        with tempfile.TemporaryDirectory() as temp:
            book = os.path.join(temp, '口算速算天天练四年级')
            os.makedirs(book)
            with patch.object(cutter, 'ocr_page_structure') as ocr:
                result = cutter.detect_book_profile(book, runtime_root=temp)
        self.assertEqual(result['profile'], 'drill')
        self.assertEqual(result['source'], 'name')
        ocr.assert_not_called()

    def test_detect_book_profile_samples_text_and_reuses_the_cached_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            book = os.path.join(temp, '原教材3')
            os.makedirs(book)
            pages = []
            for index in range(6):
                page = os.path.join(book, f'{index}.png')
                with open(page, 'wb') as file:
                    file.write(b'page')
                pages.append(page)
            items = [{'text': '第一单元达标检测卷', 'bbox': [[0, 0], [10, 0], [10, 10], [0, 10]], 'conf': 0.9}]
            with patch.object(cutter, 'cv2_imread', return_value=np.zeros((20, 20, 3), np.uint8)), \
                 patch.object(cutter, 'ocr_page_structure', return_value=items) as ocr:
                first = cutter.detect_book_profile(book, runtime_root=temp)
                calls_after_first = ocr.call_count
                second = cutter.detect_book_profile(book, runtime_root=temp)
                forced = cutter.detect_book_profile(book, runtime_root=temp, force=True)

        self.assertEqual(first['profile'], 'exam')
        self.assertEqual(first['source'], 'text')
        self.assertGreaterEqual(calls_after_first, 1)
        self.assertLessEqual(calls_after_first, cutter.BOOK_PROFILE_SAMPLE_LIMIT)
        self.assertEqual(second['source'], 'cache')
        self.assertTrue(second['cached'])
        # 命中缓存不再跑 OCR；force 才重跑。
        self.assertEqual(ocr.call_count, calls_after_first * 2)
        self.assertEqual(forced['source'], 'text')

    def test_detect_book_profile_ignores_a_cache_from_another_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            book = os.path.join(temp, '原教材7')
            os.makedirs(book)
            page = os.path.join(book, '0.png')
            with open(page, 'wb') as file:
                file.write(b'page')
            payload = {
                'revision': 'old-revision', 'profile': 'exam', 'reason': '旧结论',
                'sample': cutter._profile_sample_signature([page]), 'sampled': ['0.png'],
            }
            cutter._write_book_profile_cache(book, payload, temp)
            self.assertIsNone(cutter.read_book_profile_cache(book, temp))

    def test_exam_title_in_middle_header_is_a_new_paper_boundary(self):
        page_info = cutter.classify_page(
            self.page,
            [
                item('第一单元培优提升卷', 260, 220),
                item('一、卷面书写', 40, 370),
                item('1. 第一题', 45, 490),
            ],
            'exam',
        )
        page_info['ocr_items'] = [
            item('第一单元培优提升卷', 260, 220),
            item('一、卷面书写', 40, 370),
            item('1. 第一题', 45, 490),
        ]
        self.assertEqual(page_info['title'], '第一单元培优提升卷')
        self.assertEqual(cutter.paper_start_reason(page_info, 'exam'), '卷名')

    def test_major_question_about_final_exam_cannot_start_a_new_paper(self):
        page_info = cutter.classify_page(
            self.page,
            [
                item('八、下面是某班数学期末测试成绩单。（18分）', 45, 120),
                item('1. 第一小题', 45, 320),
            ],
            'exam',
        )
        self.assertEqual(page_info['title'], '')
        self.assertEqual(cutter.paper_start_reason(page_info, 'exam'), '')

    def test_strict_paper_title_still_accepts_real_unit_paper(self):
        self.assertTrue(cutter.PAPER_TITLE_RE.fullmatch('第一单元达标检测卷'))
        self.assertFalse(cutter.PAPER_TITLE_RE.fullmatch('八、下面是某班数学期末测试成绩单'))

    def test_two_centered_title_lines_are_merged(self):
        title = cutter.detect_page_title(
            [
                item('第二单元', 300, 90, width=180),
                item('达标检测卷', 280, 130, width=220),
                item('一、填空题', 40, 320),
            ],
            1200,
        )
        self.assertEqual(title, '第二单元达标检测卷')

    def test_title_fragments_do_not_merge_with_side_page_metadata(self):
        title = cutter.detect_page_title(
            [
                item('第二单元', 300, 90, width=180),
                item('时间：60分钟', 40, 130, width=180),
                item('一、填空题', 40, 320),
            ],
            1200,
        )
        # 未把左侧“时间”拼接进去；孤立的“第二单元”仍保留给原有标题规则。
        self.assertEqual(title, '第二单元')

    def test_exam_cover_layout_is_a_conservative_title_ocr_fallback(self):
        page_info = {
            'title': '',
            'ocr_items': [
                item('时间：60分钟', 90, 80),
                item('总分：105分', 450, 80),
                item('一、卷面书写。（5分）', 45, 340),
                item('1. 第一题', 45, 470),
            ],
        }
        self.assertEqual(cutter.paper_start_reason(page_info, 'exam'), '卷首版式')
        self.assertFalse(cutter.is_paper_start(page_info, 'general'))


class PracticePartitionTests(unittest.TestCase):
    def test_flat_olympiad_practice_header_becomes_unit_boundary(self):
        page = np.full((1600, 900, 3), 255, dtype=np.uint8)
        items = [
            item('第6练', 300, 70, width=120),
            item('乘法速算', 440, 70, width=200),
            item('1. 第一题', 55, 380),
        ]
        page_info = cutter.classify_page(page, items, 'olympiad')
        self.assertEqual(page_info['title'], '第6练 乘法速算')
        self.assertEqual(cutter.paper_start_reason(page_info, 'olympiad'), '第X练')
        self.assertEqual(cutter.paper_start_reason(page_info, 'general'), '')

    def test_flat_olympiad_book_is_partitioned_by_practice_headers(self):
        page = np.full((1600, 900, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for index in range(1, 5):
                path = os.path.join(root, f'{index}.jpg')
                open(path, 'wb').close()
                paths.append(path)
            page_items = [
                [item('第1练', 300, 70, width=120), item('数数', 440, 70), item('1. 第一题', 55, 380), item('题干内容', 150, 450)],
                [item('2. 第二题', 55, 380), item('题干内容', 150, 450), item('补充条件', 150, 520), item('继续计算', 150, 590)],
                [item('第2练', 300, 70, width=120), item('找规律', 440, 70), item('1. 第一题', 55, 380), item('题干内容', 150, 450)],
                [item('2. 第二题', 55, 380), item('题干内容', 150, 450), item('补充条件', 150, 520), item('继续计算', 150, 590)],
            ]
            with patch.object(cutter, 'cv2_imread', return_value=page), \
                 patch.object(cutter, 'ocr_page_structure', side_effect=page_items), \
                 patch.object(cutter, 'needs_high_precision_page_ocr', return_value=False):
                units = cutter.partition_flat_document(paths, 'olympiad')
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]['unit_name_hint'], '第1练 数数')
        self.assertEqual(units[1]['unit_name_hint'], '第2练 找规律')

    def test_old_single_unit_output_is_archived_before_practice_rerun(self):
        with tempfile.TemporaryDirectory() as root:
            old_unit = os.path.join(root, '第一课时')
            os.makedirs(old_unit)
            cutter.write_completion_marker(old_unit, 20, 0, 0)
            moved = cutter.archive_legacy_flat_practice_output(root, [
                {'unit_name_hint': '第1练 数数'},
                {'unit_name_hint': '第2练 找规律'},
            ], 'olympiad')
            self.assertIsNotNone(moved)
            self.assertFalse(os.path.exists(old_unit))
            self.assertTrue(os.path.isfile(
                os.path.join(moved, '.cut_questions_complete.json'),
            ))


class ExamPartitionTests(unittest.TestCase):
    def test_flat_exam_is_split_by_title_or_cover_layout(self):
        page = np.full((1200, 800, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for index in range(1, 5):
                path = os.path.join(root, f'{index}.jpg')
                open(path, 'wb').close()
                paths.append(path)
            page_items = [
                [item('第一单元达标检测卷', 250, 210), item('一、卷面书写', 40, 360), item('二、填空题', 40, 430), item('1. 第一题', 45, 480)],
                [item('二、选择题', 40, 180), item('1. 第二页题目', 45, 290), item('2. 第二页题目', 45, 390)],
                [item('时间：60分钟', 80, 80), item('总分：105分', 470, 80), item('一、卷面书写', 40, 350), item('二、填空题', 40, 420), item('1. 新卷第一题', 45, 470)],
                [item('二、判断题', 40, 180), item('1. 新卷第二页题目', 45, 290), item('2. 新卷第二页题目', 45, 390)],
            ]
            with patch.object(cutter, 'cv2_imread', return_value=page), \
                 patch.object(cutter, 'ocr_page_structure', side_effect=page_items):
                units = cutter.partition_flat_document(paths, 'exam')
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]['question_imgs'], paths[:2])
        self.assertEqual(units[1]['question_imgs'], paths[2:])


class AnswerParserTests(unittest.TestCase):
    def test_knowledge_point_heading_is_an_answer_context(self):
        self.assertEqual(
            cutter._answer_context_title('知识点二 数轴', 20, 20),
            '知识点二数轴',
        )
        self.assertEqual(
            cutter._answer_context_title('本节知识点二需要掌握数轴的表示方法', 20, 20),
            '',
        )

    def test_inline_answer_numbers(self):
        self.assertEqual(
            cutter._answer_minor_matches('1. A 2. C 3. 12'),
            [(1, 0), (2, 5), (3, 10)],
        )

    def test_inline_answer_numbers_survive_ocr_compaction(self):
        self.assertEqual(
            cutter._answer_minor_matches('1.B2.C3.D'),
            [(1, 0), (2, 3), (3, 6)],
        )

    def test_formula_like_numbers_are_not_promoted_to_compact_anchors(self):
        self.assertEqual(cutter._answer_minor_matches('x2. y3.'), [])

    def test_answer_matrix_is_not_split_into_reading_columns(self):
        """A four-cell answer grid is one visual answer region, not two columns."""
        page = np.full((800, 600, 3), 255, dtype=np.uint8)
        cells = []
        labels = ('① 35', '② 9', '③ 57', '④ 79')
        for row_index, y in enumerate((120, 190, 260, 330)):
            for column_index, x in enumerate((45, 165, 305, 425)):
                cells.append(item(labels[column_index], x, y, width=55, height=24))

        columns = cutter.split_answer_layout_columns(page, cells)

        self.assertEqual(len(columns), 1)
        self.assertEqual((columns[0][0], columns[0][1]), (0, 600))

    def test_long_rule_keeps_true_answer_columns_separate_from_matrix(self):
        """A full-height rule is stronger layout evidence than aligned cells."""
        page = np.full((800, 600, 3), 255, dtype=np.uint8)
        cv2.line(page, (300, 100), (300, 760), (0, 0, 0), 2)
        cells = []
        labels = ('① 35', '② 9', '③ 57', '④ 79')
        for y in (120, 190, 260, 330):
            for column_index, x in enumerate((45, 165, 325, 445)):
                cells.append(item(labels[column_index], x, y, width=55, height=24))

        columns = cutter.split_answer_layout_columns(page, cells)

        self.assertEqual(len(columns), 2)
        self.assertTrue(all(
            max(point[0] for point in cell['bbox']) < columns[0][1]
            for cell in columns[0][2]
        ))
        self.assertTrue(all(
            min(point[0] for point in cell['bbox']) >= columns[1][0]
            for cell in columns[1][2]
        ))

    def test_regular_two_column_answers_still_split(self):
        """The matrix guard must not disable established prose-column splits."""
        page = np.full((800, 600, 3), 255, dtype=np.uint8)
        answers = [
            item('1. 第一题答案内容较长', 45, 120, width=185),
            item('2. 第二题答案内容较长', 45, 220, width=185),
            item('3. 第三题答案内容较长', 45, 320, width=185),
            item('1. 右栏第一题答案较长', 365, 120, width=185),
            item('2. 右栏第二题答案较长', 365, 220, width=185),
            item('3. 右栏第三题答案较长', 365, 320, width=185),
        ]

        columns = cutter.split_answer_layout_columns(page, answers)

        self.assertEqual(len(columns), 2)
        self.assertLess(columns[0][1], columns[1][0] + 1)

    def test_answer_crop_stops_before_next_structural_heading(self):
        anchor = {'column_index': 0, 'y': 100, 'row_y1': 125}
        rows = [
            {'column_index': 0, 'y0': 160, 'y1': 185, 'structural': False},
            {'column_index': 1, 'y0': 170, 'y1': 195, 'structural': True},
            {'column_index': 0, 'y0': 240, 'y1': 265, 'structural': True},
        ]
        self.assertEqual(cutter._next_answer_structure_start(anchor, rows), 240)

    def test_unanchored_circled_answer_column_is_emitted_for_review(self):
        """A circled-only column must not vanish because another column has anchors."""
        page = np.full((600, 600, 3), 255, dtype=np.uint8)
        left_items = []
        for y in (100, 160, 220):
            for label, x in zip(('① 11', '② 22', '③ 33', '④ 44'), (25, 95, 165, 235)):
                left_items.append(item(label, x, y, width=42, height=22))
        # The same answer column later switches to calculation-style lines
        # without enough circled labels to constitute another matrix.
        left_items.append(item('后半段算式结果', 28, 360, width=180, height=24))
        right_items = [
            item('一、1. 右栏答案', 340, 100, width=180, height=24),
            item('2. 右栏下一题', 340, 240, width=180, height=24),
        ]
        all_items = left_items + right_items
        columns = [(0, 300, left_items), (300, 600, right_items)]
        with patch.object(cutter, 'cv2_imread', return_value=page), \
             patch.object(cutter, 'ocr_image', return_value=all_items), \
             patch.object(cutter, 'needs_high_precision_answer_ocr', return_value=False), \
             patch.object(cutter, 'refine_answer_ocr_with_vlm', return_value=all_items), \
             patch.object(cutter, 'split_answer_layout_columns', return_value=columns), \
             patch.object(cutter, 'detect_answer_blocks_with_model', return_value=[]):
            records = cutter.process_answer_pages(['answer.jpg'])

        fallback = next(record for record in records if record.get('unanchored_matrix_fallback'))
        self.assertEqual(fallback['num'], '')
        self.assertEqual(fallback['source_box'][0], 0)
        self.assertEqual(fallback['source_box'][2], 300)
        self.assertGreaterEqual(fallback['source_box'][3], 392)
        self.assertTrue(any(record.get('num') == 1 for record in records))

    def test_matrix_fallback_is_not_disabled_by_an_unrelated_anchor_in_its_column(self):
        def matrix_row(y):
            return {
                'column_index': 0, 'column_x0': 0, 'column_x1': 300,
                'x0': 20, 'x1': 260, 'y0': y, 'y1': y + 20,
                'text': '① 1 ② 2 ③ 3', 'structural': False,
            }

        rows = [matrix_row(y) for y in (100, 140, 180, 340, 380, 420)]
        sections = cutter._unanchored_answer_matrix_sections(
            rows,
            [{'column_index': 0, 'y': 96}],
            600,
        )

        self.assertEqual(len(sections), 1)
        self.assertGreaterEqual(sections[0]['source_box'][1], 330)

    def test_decimal_formula_rows_do_not_create_repeated_question_anchors(self):
        page = np.full((420, 500, 3), 255, dtype=np.uint8)
        items = [
            item('一、1.42 × 10 = 14.2', 20, 70, width=260, height=24),
            item('1.42 × 100 = 142', 50, 120, width=230, height=24),
            item('1.42 × 1000 = 1420', 50, 170, width=250, height=24),
            item('二、2.5 × 10 = 25', 20, 260, width=230, height=24),
        ]
        with patch.object(cutter, 'cv2_imread', return_value=page), \
             patch.object(cutter, 'ocr_image', return_value=items), \
             patch.object(cutter, 'needs_high_precision_answer_ocr', return_value=False), \
             patch.object(cutter, 'refine_answer_ocr_with_vlm', return_value=items), \
             patch.object(cutter, 'split_answer_layout_columns', return_value=[(0, 500, items)]), \
             patch.object(cutter, 'detect_answer_blocks_with_model', return_value=[]):
            records = cutter.process_answer_pages(['answer.jpg'])

        self.assertEqual(len(records), 2)
        self.assertEqual([record['num'] for record in records], [1, 1])
        self.assertGreaterEqual(records[0]['source_box'][3], 250)

    def test_answer_strip_recovery_fills_a_vertical_number_jump(self):
        page = np.full((700, 600, 3), 255, dtype=np.uint8)
        anchors = [
            {'context': '知识点一', 'major_idx': 0, 'num': 1,
             'column_index': 0, 'column_x0': 0, 'column_x1': 600,
             'x': 30, 'y': 80, 'row_y1': 105, 'row_text': '1. A',
             'ocr_confidence': 0.9, 'char_pos': 0},
            {'context': '知识点一', 'major_idx': 0, 'num': 3,
             'column_index': 0, 'column_x0': 0, 'column_x1': 600,
             'x': 30, 'y': 390, 'row_y1': 415, 'row_text': '3. C',
             'ocr_confidence': 0.9, 'char_pos': 0},
        ]
        page_rows = []
        with patch.object(cutter, '_run_ocr', return_value=[item('2. B', 30, 80)]):
            recovered = cutter._recover_missed_answer_anchors(page, anchors, page_rows)

        self.assertEqual(
            [anchor['num'] for anchor in sorted(recovered, key=lambda anchor: anchor['y'])],
            [1, 2, 3],
        )
        inserted = next(anchor for anchor in recovered if anchor['num'] == 2)
        self.assertTrue(inserted['recovered_by_strip'])
        self.assertEqual(len(page_rows), 1)

    def test_last_inline_answer_collects_immediate_unnumbered_wrap(self):
        anchors = [
            {'column_index': 0, 'x': 100, 'y': 50, 'row_y1': 72},
            {'column_index': 0, 'x': 320, 'y': 50, 'row_y1': 72},
            {'column_index': 0, 'x': 540, 'y': 50, 'row_y1': 72},
        ]
        rows = [
            {'column_index': 0, 'y0': 83, 'y1': 105, 'text': '7-4=3(人)',
             'has_anchor': False, 'structural': False},
            {'column_index': 0, 'y0': 145, 'y1': 168, 'text': '五、下一题',
             'has_anchor': True, 'structural': True},
        ]
        self.assertEqual(
            [row['text'] for row in cutter._inline_answer_continuation_rows(
                anchors[-1], rows, anchors,
            )],
            ['7-4=3(人)'],
        )
        self.assertEqual(
            cutter._inline_answer_continuation_rows(anchors[1], rows, anchors),
            [],
        )

    def test_answer_continuation_is_composed_without_inline_neighbours(self):
        image = np.full((100, 180, 3), 255, dtype=np.uint8)
        image[10:30, 100:160] = 0       # numbered answer at the right
        image[42:62, 10:80] = 0         # wrapped formula restarts at left
        crop, segments = cutter._compose_answer_continuation_crop(
            image,
            [95, 5, 170, 35],
            [[5, 38, 90, 68]],
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1]['kind'], 'answer_continuation')
        self.assertEqual(crop.shape[0], 60)
        self.assertGreater(np.count_nonzero(crop[42:] < 128), 0)

    def test_lesson_context_similarity(self):
        context = '第一单元分数加减法 / 2星期日的安排'
        self.assertEqual(cutter.context_similarity('第二课时星期日的安排', context), 1.0)
        self.assertEqual(cutter.context_similarity('第一课时折纸', context), 0.0)

    def test_select_only_best_answer_context(self):
        records = [
            {'context': '第一单元分数加减法 / 1折纸', 'major_idx': 1, 'num': 1},
            {'context': '第一单元分数加减法 / 2星期日的安排', 'major_idx': 1, 'num': 1},
        ]
        selected = cutter.select_answers_for_unit(records, '第一课时折纸')
        self.assertEqual(len(selected), 1)
        self.assertIn('折纸', selected[0]['context'])

    def test_question_sequence_selects_one_context_when_unit_title_is_generic(self):
        records = [
            {'context': '第一单元 / 1折纸', 'major_idx': 1, 'num': 1, 'ocr_confidence': 0.98},
            {'context': '第一单元 / 1折纸', 'major_idx': 1, 'num': 2, 'ocr_confidence': 0.98},
            {'context': '第一单元 / 2星期日', 'major_idx': 1, 'num': 1, 'ocr_confidence': 0.98},
            {'context': '第一单元 / 2星期日', 'major_idx': 1, 'num': 3, 'ocr_confidence': 0.98},
        ]
        majors = {
            1: {'minors': [
                {'num': 1, 'number_detected': True, '_number_confidence': 0.98},
                {'num': 2, 'number_detected': True, '_number_confidence': 0.98},
            ]},
        }
        selected, context = cutter.select_answer_context_by_question_sequence(
            records, majors, '第一课时', unit_position=1,
        )
        self.assertEqual(context, '第一单元 / 1折纸')
        self.assertEqual([record['num'] for record in selected], [1, 2])

    def test_generic_lesson_name_uses_unit_position_not_best_looking_later_context(self):
        records = [
            {'context': '第一单元 / 1折纸', 'major_idx': 1, 'num': 1},
            {'context': '第二单元 / 1长方体', 'major_idx': 1, 'num': 1},
            {'context': '第二单元 / 1长方体', 'major_idx': 1, 'num': 2},
        ]
        majors = {1: {'minors': [{'num': 1, 'number_detected': True}]}}
        selected, context = cutter.select_answer_context_by_question_sequence(
            records, majors, '第一课时', unit_position=1,
        )
        self.assertEqual(context, '第一单元 / 1折纸')
        self.assertEqual(len(selected), 1)

    def test_numeric_holiday_context(self):
        records = [
            {'context': '第3课时', 'major_idx': 1, 'num': 1},
            {'context': '第4课时', 'major_idx': 1, 'num': 1},
        ]
        selected = cutter.select_answers_for_unit(records, '第3课时')
        self.assertEqual([record['context'] for record in selected], ['第3课时'])


class CropQualityTests(unittest.TestCase):
    def test_short_dotted_sidebar_is_removed_but_text_is_kept(self):
        mask = np.zeros((180, 1000), dtype=np.uint8)
        # 左边密封虚线：短题中只剩断续的窄列。
        for y in range(8, 172, 14):
            mask[y:y + 5, 82:86] = 255
        # 正文起始处的题号与文字，不能被裁掉。
        mask[35:65, 210:230] = 255
        mask[95:130, 330:800] = 255
        cleaned, cut_x = cutter._remove_marginal_sidebar(mask)
        self.assertIsNotNone(cut_x)
        self.assertGreater(cut_x, 85)
        self.assertEqual(int(np.count_nonzero(cleaned[:, :cut_x])), 0)
        self.assertGreater(int(np.count_nonzero(cleaned[:, 210:230])), 0)

    def test_major_cut_stays_before_next_heading_decoration(self):
        image = np.full((1000, 700, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 560), (500, 650), (0, 0, 0), 3)
        cv2.rectangle(image, (40, 760), (260, 790), (0, 0, 0), -1)
        cv2.rectangle(image, (40, 820), (500, 845), (0, 0, 0), -1)
        cut_y = cutter._find_major_section_cut(image, 100, 820)
        self.assertGreater(cut_y, 650)
        self.assertLess(cut_y, 760)

    def test_horizontal_cut_keeps_all_option_rows(self):
        image = np.full((520, 900, 3), 255, dtype=np.uint8)
        for y in (100, 180, 260, 340):
            cv2.rectangle(image, (40, y), (650, y + 22), (0, 0, 0), -1)
        cv2.rectangle(image, (40, 440), (700, 462), (0, 0, 0), -1)
        cut_y = cutter._find_clean_horizontal_cut(image, 100, 440)
        self.assertGreater(cut_y, 365)
        self.assertLess(cut_y, 440)

    def test_tiny_normalized_crop_falls_back(self):
        image = np.full((50, 500, 3), 255, dtype=np.uint8)
        cv2.putText(image, '1.', (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        normalized, info = cutter.normalize_question_crop(image)
        self.assertGreaterEqual(normalized.shape[0], 32)
        self.assertIn('quality_warning', info)

    def test_detached_dash_page_number_is_removed(self):
        image = np.full((300, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(image, '1. QUESTION', (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.line(image, (450, 125), (475, 125), (0, 0, 0), 2)
        cv2.putText(image, '1', (490, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(image, (530, 125), (555, 125), (0, 0, 0), 2)
        normalized, _ = cutter.normalize_question_crop(image)
        self.assertLess(normalized.shape[0], 120)

    def test_isolated_scan_speck_does_not_create_bottom_whitespace(self):
        image = np.full((500, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(image, '1. QUESTION', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        image[440:442, 500:502] = 0
        normalized, _ = cutter.normalize_question_crop(image)
        self.assertLess(normalized.shape[0], 140)

    def test_marginal_sidebar_is_removed_without_cutting_question(self):
        image = np.full((700, 1400, 3), 255, dtype=np.uint8)
        # 模拟“密封线”侧栏：两个平行长线和连接它们的竖排标签框，
        # 合并后形成一个窄而高的侧栏组件。
        cv2.line(image, (150, 20), (150, 680), (0, 0, 0), 3)
        cv2.line(image, (190, 20), (190, 680), (0, 0, 0), 3)
        for y in (80, 250, 420):
            cv2.rectangle(image, (150, y), (190, y + 90), (0, 0, 0), 2)
        cv2.putText(image, '1. QUESTION', (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.rectangle(image, (400, 220), (1100, 560), (0, 0, 0), 3)
        normalized, info = cutter.normalize_question_crop(image)
        self.assertIn('marginal_sidebar_cut_x', info)
        self.assertGreater(info['marginal_sidebar_cut_x'], 180)
        self.assertLess(normalized.shape[1], 1100)

    def test_page_sidebar_limit_trims_every_short_question_in_column(self):
        image = np.full((700, 1400, 3), 255, dtype=np.uint8)
        # A left outer seal strip that every individual short question would
        # only see in part.  It must be detected once from the full column.
        cv2.line(image, (150, 20), (150, 680), (0, 0, 0), 3)
        cv2.line(image, (190, 20), (190, 680), (0, 0, 0), 3)
        for y in (80, 250, 420):
            cv2.rectangle(image, (150, y), (190, y + 90), (0, 0, 0), 2)
        cv2.putText(image, '1. QUESTION', (270, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, '2. QUESTION', (270, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, '3. QUESTION', (270, 590), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.rectangle(image, (400, 220), (1100, 560), (0, 0, 0), 3)
        blocks = [
            {'_column_bounds': [0, 1400]},
        ]
        exclusions = cutter.detect_page_sidebar_exclusions(image, blocks)
        self.assertIsNotNone(exclusions['left'])
        cuts = [
            {'_source_box': [0, y0, 1400, y1], 'crop_img': image[y0:y1, :].copy()}
            for y0, y1 in ((90, 180), (320, 410), (550, 640))
        ]
        cutter.trim_cuts_to_page_sidebar_exclusions(cuts, image, exclusions)
        self.assertTrue(all(cut['_source_box'][0] == exclusions['left'] for cut in cuts))
        self.assertTrue(all(cut.get('_page_sidebar_trimmed') for cut in cuts))

    def test_thin_full_page_seal_boundary_trims_column_questions(self):
        image = np.full((2000, 1452, 3), 255, dtype=np.uint8)
        # 外侧姓名/学校栏是一条较短的长线；密封栏内边界只有
        # 4 px，旧的宽组件规则会漏掉这种真实试卷版式。
        cv2.line(image, (128, 560), (128, 1380), (0, 0, 0), 3)
        cv2.line(image, (159, 105), (159, 1815), (0, 0, 0), 4)
        for y in (600, 1040, 1480):
            cv2.rectangle(image, (120, y), (150, y + 80), (0, 0, 0), 2)
        for y, label in ((520, '1. QUESTION'), (900, '2. QUESTION'), (1320, '3. QUESTION')):
            cv2.putText(image, label, (225, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
            cv2.line(image, (225, y + 55), (1200, y + 55), (0, 0, 0), 2)
            for row_y in range(y + 85, y + 260, 38):
                cv2.putText(image, 'BODY TEXT BODY TEXT', (245, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        exclusions = cutter.detect_page_sidebar_exclusions(
            image, [{'_column_bounds': [0, 1452]}],
        )

        self.assertIsNotNone(exclusions['left'])
        self.assertGreater(exclusions['left'], 159)
        self.assertLess(exclusions['left'], 230)
        cuts = [
            {'_source_box': [0, y0, 1452, y1], 'crop_img': image[y0:y1].copy()}
            for y0, y1 in ((480, 780), (860, 1160), (1280, 1580))
        ]
        cutter.trim_cuts_to_page_sidebar_exclusions(cuts, image, exclusions)
        self.assertTrue(all(cut['_source_box'][0] == exclusions['left'] for cut in cuts))

    def test_lone_long_table_border_is_not_a_thin_seal_boundary(self):
        mask = np.zeros((1600, 1200), dtype=np.uint8)
        # 单独的长表格边框左侧没有外侧密封栏证据，不能裁掉。
        cv2.line(mask, (170, 100), (170, 1500), 255, 4)
        cv2.rectangle(mask, (170, 100), (1050, 1500), 255, 2)
        mask[260:300, 250:900] = 255

        _cleaned, cut_x = cutter._remove_marginal_sidebar(mask)

        self.assertIsNone(cut_x)

    def test_repeated_left_aligned_text_strokes_are_not_a_sidebar(self):
        mask = np.zeros((2000, 1300), dtype=np.uint8)
        # Simulate many paragraph lines whose first glyphs share vertical
        # strokes.  The old sparse-line rule mistook this for a dashed seal.
        for y in range(80, 1920, 40):
            cv2.rectangle(mask, (187, y), (235, y + 6), 255, -1)
            cv2.rectangle(mask, (245, y), (1050, y + 6), 255, -1)
        _cleaned, cut_x = cutter._remove_marginal_sidebar(mask)
        self.assertIsNone(cut_x)

    def test_isolated_dashed_sidebar_still_has_a_clear_body_gap(self):
        mask = np.zeros((1200, 1000), dtype=np.uint8)
        for y in range(20, 1160, 60):
            cv2.rectangle(mask, (120, y), (123, y + 22), 255, -1)
        for y in range(80, 1120, 55):
            cv2.rectangle(mask, (210, y), (900, y + 7), 255, -1)
        _cleaned, cut_x = cutter._remove_marginal_sidebar(mask)
        self.assertIsNotNone(cut_x)
        self.assertLess(cut_x, 180)

    def test_ocr_horizontal_text_pulls_back_a_false_sidebar_limit(self):
        exclusions = {'left': 216, 'right': None}
        items = [{
            'text': '人群又沸腾起来',
            'bbox': [[143, 600], [620, 600], [620, 628], [143, 628]],
        }]
        protected = cutter._protect_sidebar_exclusions_with_ocr(exclusions, items, 1300)
        self.assertLessEqual(protected['left'], 139)

    def test_detached_compact_page_marker_is_removed_from_bottom_question_box(self):
        image = np.full((700, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(image, '1. QUESTION', (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.rectangle(image, (130, 200), (880, 460), (0, 0, 0), 3)
        # A small "- 2 -" footer is detached from the question content.  The
        # page marker itself is intentionally too small for the old rule.
        cv2.line(image, (440, 650), (475, 650), (0, 0, 0), 2)
        cv2.putText(image, '2', (490, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(image, (515, 650), (550, 650), (0, 0, 0), 2)
        cuts = [{
            '_source_box': [80, 60, 920, 700],
            'crop_img': image[60:700, 80:920].copy(),
        }]

        cutter.trim_cuts_to_detached_page_footer(cuts, image)

        self.assertTrue(cuts[0].get('_page_footer_trimmed'))
        self.assertLess(cuts[0]['_source_box'][3], 600)
        self.assertGreater(cuts[0]['_source_box'][3], 440)

    def test_tight_footer_with_margin_speckle_is_still_removed(self):
        # 1043A 真实布局：页码紧贴在末题正文之下（间距仅 ~28px），裁块左缘还有
        # 一个扫描杂点把页脚整体宽度/中心测偏。旧规则两处都通不过，页脚被裁进题框。
        image = np.full((700, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(image, '1. QUESTION', (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.rectangle(image, (130, 200), (880, 380), (0, 0, 0), 3)
        cv2.rectangle(image, (82, 495), (84, 498), (0, 0, 0), -1)
        # "- 7 -" sits tight under the content instead of a big detached gap.
        cv2.line(image, (430, 515), (465, 515), (0, 0, 0), 2)
        cv2.putText(image, '7', (480, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(image, (505, 515), (540, 515), (0, 0, 0), 2)
        cuts = [{
            '_source_box': [80, 60, 920, 700],
            'crop_img': image[60:700, 80:920].copy(),
        }]

        cutter.trim_cuts_to_detached_page_footer(cuts, image)

        self.assertTrue(cuts[0].get('_page_footer_trimmed'))
        self.assertLess(cuts[0]['_source_box'][3], 480)
        self.assertGreater(cuts[0]['_source_box'][3], 385)

    def test_short_question_marginal_sidebar_is_removed(self):
        image = np.full((180, 1400, 3), 255, dtype=np.uint8)
        for x in (130, 190, 250):
            cv2.line(image, (x, 18), (x, 162), (0, 0, 0), 3)
        cv2.putText(image, '3. SHORT QUESTION', (320, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.rectangle(image, (760, 38), (1260, 145), (0, 0, 0), 2)
        normalized, info = cutter.normalize_question_crop(image)
        self.assertIn('marginal_sidebar_cut_x', info)
        self.assertGreater(info['marginal_sidebar_cut_x'], 260)
        self.assertLess(normalized.shape[1], 1200)

    def test_left_aligned_heading_prevents_false_sidebar_removal(self):
        mask = np.zeros((600, 1000), dtype=np.uint8)
        mask[25:82, 80:300] = 255  # heading text starts at the crop edge
        mask[:, 250:254] = 255     # a long table border, not a sidebar
        mask[150:570, 620:940] = 255
        _, cut_x = cutter._remove_marginal_sidebar(mask)
        self.assertIsNone(cut_x)

    def test_right_seal_sidebar_is_removed_by_mirrored_rule(self):
        mask = np.zeros((700, 1400), dtype=np.uint8)
        # Mirror of the usual left seal strip on an outer right page edge.
        for x in (1210, 1250):
            cv2.line(mask, (x, 20), (x, 680), 255, 3)
        for y in (80, 250, 420):
            cv2.rectangle(mask, (1210, y), (1250, y + 90), 255, 2)
        mask[80:610, 180:1080] = 255
        cleaned, left_cut, right_cut = cutter._remove_marginal_sidebars(mask)
        self.assertIsNone(left_cut)
        self.assertIsNotNone(right_cut)
        self.assertEqual(int(np.count_nonzero(cleaned[:, 1210:])), 0)

    def test_extremely_wide_single_line_keeps_leading_question_number(self):
        mask = np.zeros((96, 2400), dtype=np.uint8)
        mask[20:68, 20:420] = 255  # "2. 每个…" at the leading edge
        mask[:, 300:304] = 255     # misleading long vertical table line
        cleaned, left_cut, right_cut = cutter._remove_marginal_sidebars(mask)
        self.assertIsNone(left_cut)
        self.assertIsNone(right_cut)
        self.assertGreater(int(np.count_nonzero(cleaned[:, :100])), 0)


class JsonSerializationTests(unittest.TestCase):
    def test_metadata_writer_normalizes_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'metadata.json')
            self.assertTrue(cutter.write_json_atomic(path, {'box': [np.int32(2), np.float32(3.5)]}))
            with open(path, 'r', encoding='utf-8') as file:
                stored = json.load(file)
        self.assertEqual(stored['box'][0], 2)
        self.assertEqual(stored['box'][1], 3.5)


class StructureOcrProfileTests(unittest.TestCase):
    """Structure OCR must know the unit profile or exam multi-column pages
    fall back to strip sampling and their middle column vanishes, which turns
    a real three-column sheet into one wide column full of cross-gutter cuts."""

    def test_worker_forwards_profile_to_structure_ocr(self):
        with patch.object(cutter, 'ocr_page_structure', return_value=[]) as ocr:
            with patch.object(cutter, 'cv2_imread', return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
                cutter._structure_ocr_worker('fake_page.jpg', profile='exam')
        ocr.assert_called_once()
        self.assertEqual(ocr.call_args.kwargs.get('profile'), 'exam')

    def test_prepare_structured_unit_ocr_passes_each_unit_profile(self):
        image_path = os.path.join(tempfile.gettempdir(), 'paperflow_unit_page.jpg')
        unit = {
            'profile': 'exam',
            'question_imgs': [image_path],
            'question_ocr': {},
        }
        with patch.object(cutter, 'structure_ocr_worker_count', return_value=1):
            with patch.object(cutter, 'ocr_page_structure', return_value=[]) as ocr:
                with patch.object(cutter, 'cv2_imread', return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
                    cutter.prepare_structured_unit_ocr([unit])
        ocr.assert_called_once()
        self.assertEqual(ocr.call_args.kwargs.get('profile'), 'exam')
        self.assertIn(image_path, unit['question_ocr'])

    def test_verified_three_column_layout_accepts_printed_divider_gutters(self):
        """A page split only by two printed vertical rules is still 3 columns.

        The global three-column check must fall back to divider gutters when
        the page has no whitespace gutters at all; otherwise greedy recursion
        divides the already-correct middle column and each question box breaks
        against the spurious extra gutter.  The printed rules themselves must
        stay outside every column: the middle column starts at the left rule's
        right edge and ends at the right rule's left edge."""
        page = np.full((960, 1800, 3), 255, dtype=np.uint8)
        cv2.line(page, (600, 80), (600, 880), (0, 0, 0), 2)
        cv2.line(page, (1200, 80), (1200, 880), (0, 0, 0), 2)
        items = []
        for column_index, x in enumerate((40, 640, 1240), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 520, y + 24), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=520, height=24,
                ))
        divider_candidates = [
            {'split_x': 600, 'bar_start': 598, 'bar_end': 602,
             'score': 12.0, 'source': 'divider'},
            {'split_x': 1200, 'bar_start': 1198, 'bar_end': 1202,
             'score': 12.0, 'source': 'divider'},
        ]
        with patch.object(cutter, '_local_column_split_candidates',
                          return_value=divider_candidates):
            regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        self.assertEqual([len(region[2]) for region in regions], [4, 4, 4])
        self.assertEqual(regions[0][1], 598)
        self.assertEqual(regions[1][0], 602)
        self.assertEqual(regions[1][1], 1198)
        self.assertEqual(regions[2][0], 1202)

    def test_verified_three_column_layout_accepts_sparse_middle_column(self):
        """A real middle column can contain only 1–2 short calculation lines.

        The old ``min(item_counts) < 3`` gate silently merged that column with
        its right neighbour, so later question boxes crossed the gutter."""
        page = np.full((1000, 1800, 3), 255, dtype=np.uint8)
        items = []
        for column_index, x in enumerate((50, 650, 1250), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 430, y + 22), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=430, height=22,
                ))
        # Strip most of the middle column so only two short lines remain.
        items = [it for it in items if not 650 <= min(p[0] for p in it['bbox']) < 1250]
        items.extend([item('1. 短中栏小题', 700, 150, width=430, height=22),
                      item('2. 短中栏小题', 700, 400, width=430, height=22)])

        regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        self.assertEqual([len(region[2]) for region in regions[1:]], [2, 4])

    def test_verified_three_column_layout_sorts_divider_pairs_spatially(self):
        """Candidate order is score-ranked, not spatial; the pair loop must not
        assign the right rule's edges to the left column region.

        A right divider with a higher score would be picked as ``left_gutter``
        and the min/max ``left_split``/``right_split`` would cross the regions
        (left column ending at the right rule, middle column inverted).  Real
        page 1042A produced ``[0,3061][3067,1586][1592,4732]`` from this."""
        page = np.full((960, 1800, 3), 255, dtype=np.uint8)
        cv2.line(page, (600, 80), (600, 880), (0, 0, 0), 2)
        cv2.line(page, (1200, 80), (1200, 880), (0, 0, 0), 2)
        items = []
        for column_index, x in enumerate((40, 640, 1240), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 520, y + 24), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=520, height=24,
                ))
        # The right rule scores higher, so score-rank order is right-first.
        divider_candidates = [
            {'split_x': 1200, 'bar_start': 1198, 'bar_end': 1202,
             'score': 99.0, 'source': 'divider'},
            {'split_x': 600, 'bar_start': 598, 'bar_end': 602,
             'score': 12.0, 'source': 'divider'},
        ]
        with patch.object(cutter, '_local_column_split_candidates',
                          return_value=divider_candidates):
            regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0][1], 598)
        self.assertEqual(regions[1][0], 602)
        self.assertEqual(regions[1][1], 1198)
        self.assertEqual(regions[2][0], 1202)
        self.assertTrue(all(r[0] < r[1] for r in regions))

    def test_verified_three_column_layout_never_pairs_blank_with_divider(self):
        """A stray wide blank inside a column must not pair with a divider.

        Real page 1045B has a whitespace blank at x=1421 (inside the left
        column) plus two printed rules.  The old code appended dividers to the
        whitespace candidates and let that internal blank pair with the right
        rule, so the middle column started inside the left column and swallowed
        its text.  Only same-source pairs are allowed."""
        page = np.full((960, 1800, 3), 255, dtype=np.uint8)
        cv2.line(page, (600, 80), (600, 880), (0, 0, 0), 2)
        cv2.line(page, (1200, 80), (1200, 880), (0, 0, 0), 2)
        items = []
        for column_index, x in enumerate((40, 640, 1240), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 520, y + 24), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=520, height=24,
                ))
        # whitespace at x=480 sits inside the left column; two real rules.
        candidates = [
            {'split_x': 480, 'score': 99.0, 'source': 'whitespace'},
            {'split_x': 1200, 'bar_start': 1198, 'bar_end': 1202,
             'score': 12.0, 'source': 'divider'},
            {'split_x': 600, 'bar_start': 598, 'bar_end': 602,
             'score': 12.0, 'source': 'divider'},
        ]
        with patch.object(cutter, '_local_column_split_candidates',
                          return_value=candidates):
            regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0][1], 598)
        self.assertEqual(regions[1][0], 602)
        self.assertEqual([len(region[2]) for region in regions], [4, 4, 4])

    def test_verified_three_column_layout_prefers_printed_rules_over_intra_column_blanks(self):
        """A whitespace pool that verifies must not beat a verified printed
        divider pair.

        Real pages 1053B/1055B have wide trailing blanks *inside* one column
        (left-column text ends early and the blank band up to the printed rule
        scores as an enormous gutter).  The whitespace pair earlier won with
        score ~1800 against the rules' ~39, so the middle column started/ended
        at an internal blank and swallowed or clipped its text.  Printed rules
        are the physical truth; any same-source whitespace pair that passes the
        same verification must yield to them."""
        page = np.full((960, 1800, 3), 255, dtype=np.uint8)
        cv2.line(page, (600, 80), (600, 880), (0, 0, 0), 2)
        cv2.line(page, (1200, 80), (1200, 880), (0, 0, 0), 2)
        items = []
        for column_index, x in enumerate((40, 640, 1240), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 520, y + 24), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=520, height=24,
                ))
        # Two whitespace blanks inside the left and right columns that also
        # pass three-column verification, plus the two real printed rules.
        candidates = [
            {'split_x': 480, 'score': 900.0, 'source': 'whitespace'},
            {'split_x': 1000, 'score': 900.0, 'source': 'whitespace'},
            {'split_x': 1200, 'bar_start': 1198, 'bar_end': 1202,
             'score': 12.0, 'source': 'divider'},
            {'split_x': 600, 'bar_start': 598, 'bar_end': 602,
             'score': 12.0, 'source': 'divider'},
        ]
        with patch.object(cutter, '_local_column_split_candidates',
                          return_value=candidates):
            regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        # The printed rules win: left column ends at the left rule's left edge,
        # middle column between the two rules' inner edges.
        self.assertEqual(regions[0][1], 598)
        self.assertEqual(regions[1][0], 602)
        self.assertEqual(regions[1][1], 1198)
        self.assertEqual(regions[2][0], 1202)
        self.assertEqual([len(region[2]) for region in regions], [4, 4, 4])

    def test_verified_three_column_layout_keeps_whitespace_when_no_rules(self):
        """Without printed rules the verified whitespace pair is still used."""
        page = np.full((960, 1800, 3), 255, dtype=np.uint8)
        items = []
        for column_index, x in enumerate((50, 650, 1250), start=1):
            for row_index, y in enumerate((120, 290, 460, 630), start=1):
                cv2.rectangle(page, (x, y), (x + 430, y + 24), (0, 0, 0), -1)
                items.append(item(
                    f'{row_index}. COLUMN {column_index} EXERCISE', x, y,
                    width=430, height=24,
                ))
        candidates = [
            {'split_x': 565, 'score': 400.0, 'source': 'whitespace'},
            {'split_x': 1165, 'score': 400.0, 'source': 'whitespace'},
        ]
        with patch.object(cutter, '_local_column_split_candidates',
                          return_value=candidates):
            regions = cutter._verified_three_column_layout(page, items)

        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0][1], 565)
        self.assertEqual(regions[1][0], 565)
        self.assertEqual([len(region[2]) for region in regions], [4, 4, 4])

    def test_batch_column_trim_skips_shared_context_composition(self):
        """A shared-context crop is taller than its raw source box; the column
        guard must not flat re-crop it and strip the attached preamble."""
        page = np.full((900, 1500, 3), 255, dtype=np.uint8)
        minor = {
            '_source_box': [0, 100, 500, 200],
            'crop_img': page[0:300, 0:500].copy(),  # 3x taller: preamble + own box
            '_shared_context_attached': True,
            '_source_segments': [{'question_box': [0, 0, 500, 100]}, {'question_box': [0, 100, 500, 200]}],
        }
        cutter.trim_cuts_to_column_bounds([minor], page, [0, 500])
        self.assertEqual(minor['_source_box'], [0, 100, 500, 200])
        self.assertEqual(minor['crop_img'].shape, (300, 500, 3))
        self.assertNotIn('_column_bounds_trimmed', minor)


class MissedMinorRecoveryTests(unittest.TestCase):
    def setUp(self):
        # 条带重识别有内容寻址缓存。本类用例靠 patch 换掉 OCR 结果，
        # 用的是同一张合成图，缓存会跨用例命中上一个用例的假结果，
        # 测出来的行为与生产无关。这里关掉缓存再测。
        patcher = patch.object(cutter, '_STRIP_OCR_CACHE_ENABLED', False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missed_minor_number_is_recovered_by_strip_reocr(self):
        """Structure OCR drops question 2's line; a native-res strip re-OCR
        restores it, so adjacent questions are not merged into one box."""
        image = np.full((700, 600, 3), 255, dtype=np.uint8)
        ocr_items = [
            item('1. 第一题', 20, 100),
            item('3. 第三题', 20, 480),
        ]
        # 只识别到 1、3（跳号说明中间第 2 题漏读），条带重识别找回第 2 题。
        with patch.object(cutter, '_run_ocr',
                          return_value=[item('2. 第二题', 20, 180)]):
            cuts = cutter.split_major_into_minors(image, ocr_items=ocr_items)

        self.assertEqual([cut['num'] for cut in cuts], [1, 2, 3])
        self.assertEqual(cuts[1]['_boundary_source'], 'ocr_number')

    def test_strip_reocr_rejects_candidate_not_filling_the_jump(self):
        """A strip OCR fragment whose number does not fill the jump must not
        create a false split (e.g. body text misread as a minor number)."""
        image = np.full((700, 600, 3), 255, dtype=np.uint8)
        ocr_items = [
            item('1. 第一题', 20, 100),
            item('3. 第三题', 20, 480),
        ]
        with patch.object(cutter, '_run_ocr',
                          return_value=[item('88. 乱码噪声', 20, 180)]):
            cuts = cutter.split_major_into_minors(image, ocr_items=ocr_items)

        self.assertEqual([cut['num'] for cut in cuts], [1, 3])

    def test_strip_reocr_skips_restart_to_one(self):
        """A numbering restart back to 1 is a new 知识点 section, not a missed
        number; no strip re-OCR should be triggered for it."""
        image = np.full((700, 600, 3), 255, dtype=np.uint8)
        ocr_items = [
            item('8. 前节末题', 20, 100),
            item('1. 新节首题', 20, 420),
        ]
        with patch.object(cutter, '_run_ocr',
                          return_value=[item('1. 不应被调用', 20, 180)]) as reocr:
            cuts = cutter.split_major_into_minors(image, ocr_items=ocr_items)

        reocr.assert_not_called()
        self.assertEqual([cut['num'] for cut in cuts], [8, 1])

    def test_strip_reocr_skips_contiguous_sequence(self):
        """A complete contiguous numbering sequence has no jump, so the strip
        re-OCR must not run at all."""
        image = np.full((700, 600, 3), 255, dtype=np.uint8)
        ocr_items = [
            item('1. 第一题', 20, 100),
            item('2. 第二题', 20, 300),
            item('3. 第三题', 20, 500),
        ]
        with patch.object(cutter, '_run_ocr',
                          return_value=[item('9. 不应被调用', 20, 180)]) as reocr:
            cuts = cutter.split_major_into_minors(image, ocr_items=ocr_items)

        reocr.assert_not_called()
        self.assertEqual([cut['num'] for cut in cuts], [1, 2, 3])

    def test_multiple_missed_numbers_recover_iteratively(self):
        """A jump of three (1→4) hides two questions; the recovery iterates so
        both missing boundaries are restored."""
        image = np.full((700, 600, 3), 255, dtype=np.uint8)
        ocr_items = [
            item('1. 第一题', 20, 100),
            item('4. 第四题', 20, 560),
        ]
        strip_results = [
            [item('2. 第二题', 20, 180)],
            [item('3. 第三题', 20, 180)],
        ]
        with patch.object(cutter, '_run_ocr',
                          side_effect=strip_results):
            cuts = cutter.split_major_into_minors(image, ocr_items=ocr_items)

        self.assertEqual([cut['num'] for cut in cuts], [1, 2, 3, 4])
        self.assertEqual(cuts[1]['num'], 2)
        self.assertEqual(cuts[2]['num'], 3)


if __name__ == '__main__':
    unittest.main()
