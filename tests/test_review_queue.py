import json
import os
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import review_queue


def question(number, detected=True):
    return {'num': number, 'number_detected': detected}


def answer(number, confidence=1.0):
    return {'num': number, 'ocr_confidence': confidence}


def queue_item(review_id='review-1', band='review', answer_data=None):
    return {
        'review_id': review_id,
        'question': {'image': '一/第1题/题目.jpg'},
        'answer': answer_data,
        'pairing': {'decision_band': band, 'score': 0.7},
    }


class MonotonicAlignmentTests(unittest.TestCase):
    def test_extra_answer_is_skipped_without_shifting_later_pairs(self):
        pairs, unused = review_queue.align_monotonic(
            [question(1), question(2)],
            [answer(1), answer(99), answer(2)],
        )
        self.assertEqual([pair['answer']['num'] for pair in pairs], [1, 2])
        self.assertEqual(unused, [1])
        self.assertTrue(all(pair['decision_band'] == 'auto' for pair in pairs))

    def test_missing_question_number_never_auto_accepts(self):
        pairs, unused = review_queue.align_monotonic(
            [question(1, detected=False)], [answer(1)],
        )
        self.assertEqual(unused, [])
        self.assertEqual(pairs[0]['decision_band'], 'review')

    def test_reliable_number_conflict_is_not_forced(self):
        pairs, unused = review_queue.align_monotonic([question(2)], [answer(7)])
        self.assertIsNone(pairs[0]['answer'])
        self.assertEqual(pairs[0]['decision_band'], 'no_answer')
        self.assertEqual(unused, [0])

    def test_low_ocr_confidence_requires_review_even_when_numbers_match(self):
        uncertain_question = {
            'num': 1, 'number_detected': True, '_number_confidence': 0.60,
        }
        pairs, _ = review_queue.align_monotonic(
            [uncertain_question], [answer(1, confidence=0.95)],
        )
        self.assertEqual(pairs[0]['decision_band'], 'review')

    def test_cross_page_pair_requires_review(self):
        cross_page_question = {
            'num': 1, 'number_detected': True, '_number_confidence': 0.99,
            '_cross_page_merge': True,
        }
        pairs, _ = review_queue.align_monotonic(
            [cross_page_question], [answer(1, confidence=0.99)],
        )
        self.assertEqual(pairs[0]['decision_band'], 'review')

    def test_ambiguous_answer_context_requires_review(self):
        ambiguous_question = {
            'num': 1, 'number_detected': True, '_number_confidence': 0.99,
            '_answer_context_ambiguous': True,
        }
        pairs, _ = review_queue.align_monotonic(
            [ambiguous_question], [answer(1, confidence=0.99)],
        )
        self.assertEqual(pairs[0]['decision_band'], 'review')


class ReviewProtocolTests(unittest.TestCase):
    def test_numpy_scalars_are_normalized_for_queue_and_fingerprint(self):
        item = queue_item()
        item['question']['source_box'] = [np.int32(1), np.int32(2), np.int32(30), np.int32(40)]
        item['pairing']['score'] = np.float32(0.7)
        queue = review_queue.create_review_queue('第一单元', [item])
        self.assertEqual(queue['items'][0]['question']['source_box'], [1, 2, 30, 40])
        self.assertIsInstance(queue['items'][0]['pairing']['score'], float)
        self.assertIsInstance(queue['queue_fingerprint'], str)
        review_queue.validate_review_queue(queue)
        json.dumps(queue, ensure_ascii=False)

    def test_tampered_queue_fingerprint_is_rejected(self):
        queue = review_queue.create_review_queue('第一单元', [queue_item()])
        queue['items'][0]['pairing']['score'] = 0.2
        with self.assertRaises(ValueError):
            review_queue.validate_review_queue(queue)

    def test_decision_from_another_queue_is_rejected(self):
        first = review_queue.create_review_queue('第一单元', [queue_item('review-1')])
        second = review_queue.create_review_queue('第二单元', [queue_item('review-2')])
        document = review_queue.new_decision_document(first)
        with self.assertRaises(ValueError):
            review_queue.validate_decisions(second, document)

    def test_answerless_item_cannot_be_accepted(self):
        queue = review_queue.create_review_queue('第一单元', [queue_item(answer_data=None)])
        document = review_queue.new_decision_document(queue)
        document['decisions'] = [{
            'review_id': 'review-1', 'decision': 'accept', 'notes': '',
        }]
        with self.assertRaises(ValueError):
            review_queue.validate_decisions(queue, document)

    def test_accept_copies_candidate_to_official_answer(self):
        with tempfile.TemporaryDirectory() as root:
            question_dir = os.path.join(root, '一', '第1题')
            os.makedirs(question_dir)
            candidate = os.path.join(question_dir, '候选答案.jpg')
            with open(candidate, 'wb') as file:
                file.write(b'candidate-image')
            answer_data = {
                'proposed_image': '一/第1题/候选答案.jpg',
                'accepted_image': '一/第1题/答案.jpg',
            }
            queue = review_queue.create_review_queue(
                '第一单元', [queue_item(answer_data=answer_data)],
            )
            document = review_queue.new_decision_document(queue)
            document['decisions'] = [{
                'review_id': 'review-1', 'decision': 'accept', 'notes': '',
            }]
            result = review_queue.apply_decisions(root, queue, document)
            with open(os.path.join(question_dir, '答案.jpg'), 'rb') as file:
                content = file.read()
        self.assertEqual(content, b'candidate-image')
        self.assertEqual(result['accepted'], 1)


if __name__ == '__main__':
    unittest.main()
