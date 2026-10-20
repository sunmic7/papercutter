import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import cut_quality_audit as audit


def item(text, x=100, y=200, width=220, height=34):
    return {
        'text': text,
        'bbox': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        'score': 0.99,
    }


class PrintedMinorNumberTests(unittest.TestCase):
    def test_reads_arabic_numbering(self):
        items = [item('1. 一个数由3个万组成', y=100), item('2．计算下面各题', y=200)]
        self.assertEqual(audit.printed_minor_numbers(items), [1, 2])

    def test_ignores_decimal_point(self):
        # "3.14" 是小数不是题号，RE_MINOR 的 (?!\d) 必须挡掉
        items = [item('3.14 是圆周率的近似值', y=100)]
        self.assertEqual(audit.printed_minor_numbers(items), [])

    def test_keeps_sequence_order_and_duplicates(self):
        items = [item('2. 第二题', y=300), item('1. 第一题', y=100), item('1. 另一栏重新开始', y=500)]
        self.assertEqual(audit.printed_minor_numbers(items), [1, 2, 1])

    def test_ignores_number_above_threshold(self):
        items = [item('120. 超出教材题号范围', y=100)]
        self.assertEqual(audit.printed_minor_numbers(items), [])


class PrintedMajorTitleTests(unittest.TestCase):
    def test_dedupes_same_major_and_sorts_by_top(self):
        items = [
            item('二、填空', y=400),
            item('一、卷面书写', y=100),
            item('一、卷面书写。（5分）', y=100, x=300),
        ]
        found = audit.printed_major_titles(items)
        self.assertEqual([entry[0] for entry in found], [1, 2])
        self.assertEqual(found[0][2], 100)

    def test_rejects_long_body_line_starting_with_major_marker(self):
        items = [item('三、' + '很长的正文' * 10, y=100)]
        self.assertEqual(audit.printed_major_titles(items), [])


class SummarizeTests(unittest.TestCase):
    def _page(self, applicable=True, **overrides):
        record = {
            'book': 'demo',
            'file': 'p1.jpg',
            'applicable': applicable,
            'block_count': 3,
            'major_printed': 2,
            'major_captured': 2,
            'minor_printed': 8,
            'minor_cuts': 8,
            'seconds': 1.0,
            'error_samples': [],
        }
        for name in (
            'untitled_with_evidence', 'merged_blocks', 'duplicate_title_blocks',
            'collapsed_blocks', 'empty_blocks', 'undetected', 'tiny_cuts',
            'giant_cuts', 'bad_quality', 'seq_breaks', 'continuation_blocks',
        ):
            record[name] = 0
        record.update(overrides)
        return record

    def test_ignores_non_applicable_pages_in_ratios(self):
        report = audit.summarize([
            self._page(),
            self._page(applicable=False, major_printed=0, major_captured=0),
        ])
        total = report['total']
        self.assertEqual(total['pages'], 2)
        self.assertEqual(total['applicable_pages'], 1)
        self.assertEqual(total['major_recall'], 1.0)

    def test_error_rate_only_counts_applicable_pages(self):
        report = audit.summarize([
            self._page(merged_blocks=1),
            self._page(applicable=False, merged_blocks=0),
        ])
        self.assertEqual(report['total']['pages_with_error'], 1)
        self.assertEqual(report['total']['error_rate'], 1.0)

    def test_collects_error_type_distribution(self):
        page = self._page()
        page['error_samples'] = [{'type': '空壳块（高度异常小）', 'block': 0}]
        report = audit.summarize([page])
        self.assertEqual(report['error_types'], {'空壳块（高度异常小）': 1})
        self.assertEqual(len(report['page_errors']), 1)

    def test_empty_input_is_safe(self):
        report = audit.summarize([])
        self.assertEqual(report['total'], {'pages': 0})


class SampleSelectionTests(unittest.TestCase):
    def test_sample_skips_cover_and_spreads_pages(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            for index in range(20):
                open(os.path.join(directory, f'{index:04d}.jpg'), 'wb').close()
            picked = audit.iter_page_paths(directory, sample=5)
        self.assertEqual(len(picked), 5)
        self.assertNotIn('0000.jpg', [os.path.basename(path) for path in picked])
        self.assertIn('0019.jpg', [os.path.basename(path) for path in picked])


class PagePlanTests(unittest.TestCase):
    """固定清单是复跑可比的前提，不能再靠 --sample 猜。"""

    def _plan(self, directory, rows):
        import json
        path = os.path.join(directory, 'plan.jsonl')
        with open(path, 'w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return path

    def test_load_page_plan_keeps_profile_and_book(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            plan = self._plan(directory, [
                {'path': os.path.join(directory, 'a.jpg'), 'book': '书A', 'profile': 'exam'},
                {'path': os.path.join(directory, 'b.jpg'), 'book': '书A', 'profile': 'exam'},
            ])
            rows = audit.load_page_plan(plan)
        self.assertEqual([row['profile'] for row in rows], ['exam', 'exam'])
        self.assertEqual(rows[0]['book'], '书A')

    def test_run_page_plan_skips_already_done_pages(self):
        import json
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, 'out.jsonl')
            first = os.path.join(directory, 'a.jpg')
            second = os.path.join(directory, 'b.jpg')
            plan = self._plan(directory, [
                {'path': first, 'book': '书A', 'profile': 'exam'},
                {'path': second, 'book': '书A', 'profile': 'exam'},
            ])
            with open(out, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps({'path': first, 'file': 'a.jpg'}) + '\n')

            calls = []

            def fake_audit(path, profile='general'):
                calls.append((path, profile))
                return {'path': path, 'file': os.path.basename(path), 'block_count': 1}

            with mock.patch.object(audit, 'audit_page', side_effect=fake_audit), \
                    mock.patch.object(audit, 'prewarm_models', return_value=None):
                audit.run_page_plan(plan, out, log=lambda *args: None)

            records = [json.loads(line) for line in open(out, encoding='utf-8')]

        self.assertEqual(calls, [(second, 'exam')])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]['book'], '书A')


if __name__ == '__main__':
    unittest.main()
