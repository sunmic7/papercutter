# -*- coding: utf-8 -*-
"""页级一致性（漏题检测）单测。

漏题是唯一"没有错误信号"的失效：数据只是变少，逐条校验永远合格。
所以这里验证两条结构必然性：大题号连续、页题量不偏离同书均值。
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import page_consistency as consistency


def question(index, major='一', region_role='question'):
    return {
        'id': f'q-{index}',
        'region_role': region_role,
        'major_number': major,
        'major_title': f'{major}、题目',
        'minor_number': str(index),
        'bbox': [10, 10 + index * 20, 300, 26 + index * 20],
    }


def make_page(major='一', count=4, image='p.jpg'):
    return {
        'image': image,
        'status': 'accepted',
        'questions': [question(index, major=major) for index in range(count)],
    }


def make_project(page_specs):
    """page_specs: [(major, count)] —— 按阅读顺序给出题页。"""
    return {
        'pages': [make_page(major, count, image=f'p{index}.jpg')
                  for index, (major, count) in enumerate(page_specs)],
        'answer_pages': [],
        'ignored_pages': [],
    }


def check_by_name(result, name):
    return next((check for check in result['checks'] if check['name'] == name), {})


class NumberConversionTests(unittest.TestCase):
    def test_chinese_and_arabic_numbers(self):
        self.assertEqual(consistency.cn_to_number('三'), 3)
        self.assertEqual(consistency.cn_to_number('十二'), 12)
        self.assertEqual(consistency.cn_to_number('二十三'), 23)
        self.assertEqual(consistency.cn_to_number('7'), 7)
        self.assertIsNone(consistency.cn_to_number('甲'))

    def test_major_index_from_labels(self):
        self.assertEqual(consistency.major_index('三、'), 3)
        self.assertEqual(consistency.major_index('三'), 3)
        self.assertEqual(consistency.major_index('2.'), 2)
        self.assertEqual(consistency.major_index('第二大题'), 2)
        self.assertIsNone(consistency.major_index(''))


class ExtractMajorsTests(unittest.TestCase):
    def test_dedupes_and_keeps_order(self):
        records = [question(0, '一'), question(1, '一'), question(2, '二')]
        self.assertEqual(consistency.extract_printed_majors(records), ['一', '二'])

    def test_falls_back_to_major_title(self):
        records = [{'major_number': '', 'major_title': '三、解决问题'},
                   {'major_number': '四'}]
        self.assertEqual(consistency.extract_printed_majors(records), ['三', '四'])

    def test_empty_records(self):
        self.assertEqual(consistency.extract_printed_majors([]), [])


class MajorSequenceTests(unittest.TestCase):
    def test_continuous_majors_do_not_alert(self):
        project = make_project([('一', 3), ('二', 3)])
        result = consistency.check_page(project, 1)
        self.assertTrue(check_by_name(result, 'major_sequence')['ok'])
        self.assertTrue(result['ok'])

    def test_same_major_continues_across_pages(self):
        project = make_project([('一', 3), ('一', 2)])
        result = consistency.check_page(project, 1)
        self.assertTrue(check_by_name(result, 'major_sequence')['ok'])

    def test_skipped_major_alerts(self):
        project = make_project([('一', 3), ('三', 3)])
        result = consistency.check_page(project, 1)
        check = check_by_name(result, 'major_sequence')
        self.assertFalse(check['ok'])
        self.assertIn('漏', check['detail'])
        self.assertFalse(result['ok'])

    def test_new_unit_restart_does_not_alert(self):
        project = make_project([('四', 3), ('一', 3)])
        result = consistency.check_page(project, 1)
        self.assertTrue(check_by_name(result, 'major_sequence')['ok'])

    def test_first_page_is_never_a_gap(self):
        project = make_project([('二', 3)])
        result = consistency.check_page(project, 0)
        self.assertTrue(check_by_name(result, 'major_sequence')['ok'])

    def test_page_without_majors_is_skipped(self):
        project = make_project([('一', 3), ('二', 3)])
        project['pages'][1]['questions'] = []
        result = consistency.check_page(project, 1)
        self.assertTrue(check_by_name(result, 'major_sequence')['skipped'])


class DensityTests(unittest.TestCase):
    def test_density_outlier_alerts(self):
        # 10 页各 8 题 + 1 页 1 题：μ≈7.36，σ≈2.01，偏离 6.36 > 2σ
        specs = [('一', 8) for _ in range(10)] + [('二', 1)]
        project = make_project(specs)
        result = consistency.check_page(project, 10)
        check = check_by_name(result, 'question_density')
        self.assertFalse(check['ok'])
        self.assertIn('同书均值', check['detail'])
        self.assertFalse(result['ok'])

    def test_balanced_pages_do_not_alert(self):
        specs = [('一', 4), ('二', 4), ('三', 4)]
        project = make_project(specs)
        for index in range(3):
            self.assertTrue(check_by_name(consistency.check_page(project, index),
                                         'question_density')['ok'])

    def test_zero_question_pages_are_excluded_from_mean(self):
        project = make_project([('一', 4), ('二', 4)])
        project['pages'].append({'image': 'cover.jpg', 'status': 'unlabeled', 'questions': []})
        stats = consistency.book_density_stats(project)
        self.assertEqual(stats['pages'], 2)
        self.assertAlmostEqual(stats['mean'], 4.0, places=4)
        self.assertEqual(stats['std'], 0.0)

    def test_answer_and_ignored_pages_are_excluded(self):
        project = make_project([('一', 4), ('二', 4)])
        project['answer_pages'] = [{'image': 'a1.jpg', 'answers': [question(0)] * 9}]
        project['ignored_pages'] = [{'image': 'p1.jpg'}]
        stats = consistency.book_density_stats(project)
        self.assertEqual(stats['pages'], 1)
        self.assertAlmostEqual(stats['mean'], 4.0, places=4)
        self.assertIn(1, stats['skipped_pages'])

    def test_single_page_has_no_sigma_and_never_alerts(self):
        project = make_project([('一', 1)])
        stats = consistency.book_density_stats(project)
        self.assertEqual(stats['pages'], 1)
        result = consistency.check_page(project, 0)
        self.assertTrue(check_by_name(result, 'question_density')['skipped'])

    def test_non_question_roles_are_not_counted(self):
        page = make_page('一', 0)
        page['questions'] = [question(0, '一', region_role='group'),
                             question(1, '一', region_role='question')]
        project = {'pages': [page], 'answer_pages': [], 'ignored_pages': []}
        stats = consistency.book_density_stats(project)
        self.assertEqual(stats['counts'], [1])


class ContractTests(unittest.TestCase):
    def test_page_index_out_of_range_is_tolerated(self):
        project = make_project([('一', 4)])
        result = consistency.check_page(project, 99)
        self.assertTrue(result['ok'])
        self.assertTrue(result['warnings'])

    def test_result_shape(self):
        project = make_project([('一', 4), ('二', 4)])
        result = consistency.check_page(project, 0)
        self.assertEqual(set(result) >= {'ok', 'checks', 'majors', 'question_density'}, True)
        names = {check['name'] for check in result['checks']}
        self.assertEqual(names, {'major_sequence', 'question_density'})


if __name__ == '__main__':
    unittest.main()
