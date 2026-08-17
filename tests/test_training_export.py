import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import export_training_data
import review_queue


class TrainingExportTests(unittest.TestCase):
    def test_same_book_always_uses_same_split(self):
        first = export_training_data.dataset_split('教材甲')
        second = export_training_data.dataset_split('教材甲')
        self.assertEqual(first, second)

    def test_only_reviewed_items_export_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            unit = os.path.join(root, '教材甲', '第一单元')
            os.makedirs(unit)
            items = [
                {
                    'review_id': 'review-1', 'book_name': '教材甲',
                    'unit_name': '第一单元', 'major': {},
                    'question': {'image': 'q1.jpg', 'source_image': 's1.jpg', 'source_box': [0, 0, 10, 10]},
                    'answer': None,
                    'pairing': {'decision_band': 'review', 'score': 0.7},
                    'risk_flags': [],
                },
                {
                    'review_id': 'review-2', 'book_name': '教材甲',
                    'unit_name': '第一单元', 'major': {},
                    'question': {'image': 'q2.jpg', 'source_image': 's2.jpg', 'source_box': [0, 0, 10, 10]},
                    'answer': None,
                    'pairing': {'decision_band': 'auto', 'score': 0.98},
                    'risk_flags': [],
                },
            ]
            queue = review_queue.create_review_queue('第一单元', items)
            review_queue.write_json_atomic(os.path.join(unit, 'review_queue.json'), queue)
            decisions = review_queue.new_decision_document(queue)
            decisions['decisions'] = [{
                'review_id': 'review-1', 'decision': 'question_bad',
                'corrected_question_box': [1, 1, 9, 9], 'notes': '裁多',
            }]
            review_queue.write_json_atomic(os.path.join(unit, 'review_decisions.json'), decisions)
            records, invalid = export_training_data.collect_training_records(root)
        self.assertEqual(invalid, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['label'], 'question_bad')
        self.assertEqual(records[0]['question']['target_box'], [1, 1, 9, 9])


if __name__ == '__main__':
    unittest.main()
