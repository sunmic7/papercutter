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
    def test_inline_answer_numbers(self):
        self.assertEqual(
            cutter._answer_minor_matches('1. A 2. C 3. 12'),
            [(1, 0), (2, 5), (3, 10)],
        )

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


if __name__ == '__main__':
    unittest.main()
