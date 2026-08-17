import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import evaluate_cutter as evaluator
from review_tools import annotate_cuts


class GroundTruthMetricTests(unittest.TestCase):
    def test_exact_boxes_have_perfect_scores(self):
        boxes = [[0, 0, 500, 200], [0, 200, 500, 480]]
        result = evaluator.compare_page_boxes(boxes, boxes)
        self.assertEqual(result['precision'], 1.0)
        self.assertEqual(result['recall'], 1.0)
        self.assertEqual(result['mean_iou'], 1.0)
        self.assertTrue(result['exact_count_page'])

    def test_one_prediction_covering_two_questions_is_under_cut(self):
        result = evaluator.compare_page_boxes(
            [[0, 0, 500, 400]],
            [[0, 0, 500, 200], [0, 200, 500, 400]],
        )
        self.assertEqual(result['under_cut_count'], 1)
        self.assertEqual(result['over_cut_count'], 0)
        self.assertEqual(result['merged_prediction_count'], 1)

    def test_two_predictions_inside_one_question_are_over_cut(self):
        result = evaluator.compare_page_boxes(
            [[0, 0, 500, 200], [0, 200, 500, 400]],
            [[0, 0, 500, 400]],
        )
        self.assertEqual(result['over_cut_count'], 1)
        self.assertEqual(result['under_cut_count'], 0)
        self.assertEqual(result['split_truth_count'], 1)

    def test_tight_truth_width_can_match_full_column_prediction(self):
        result = evaluator.compare_page_boxes(
            [[0, 100, 1000, 300]],
            [[120, 100, 880, 300]],
        )
        self.assertEqual(result['matched_count'], 1)
        self.assertLess(result['mean_iou'], 1.0)

    def test_under_crop_is_separate_from_missing_question(self):
        result = evaluator.compare_page_boxes(
            [[0, 20, 500, 200]], [[0, 0, 500, 200]],
        )
        self.assertEqual(result['missed_truth_count'], 0)
        self.assertEqual(result['under_crop_count'], 1)
        self.assertEqual(result['over_crop_count'], 0)

    def test_over_crop_is_separate_from_extra_prediction(self):
        result = evaluator.compare_page_boxes(
            [[0, 0, 500, 240]], [[0, 20, 500, 200]],
        )
        self.assertEqual(result['extra_prediction_count'], 0)
        self.assertEqual(result['over_crop_count'], 1)


class AnnotationManifestTests(unittest.TestCase):
    def test_annotator_can_create_manifest_from_raw_image_directory(self):
        with tempfile.TemporaryDirectory() as root:
            cv2.imwrite(os.path.join(root, '10.jpg'), np.full((10, 10, 3), 255, dtype=np.uint8))
            cv2.imwrite(os.path.join(root, '2.jpg'), np.full((10, 10, 3), 255, dtype=np.uint8))
            os.makedirs(os.path.join(root, '_ignored'))
            cv2.imwrite(os.path.join(root, '_ignored', '1.jpg'), np.full((10, 10, 3), 255, dtype=np.uint8))
            manifest = annotate_cuts.create_annotation_manifest(root)
        self.assertEqual([page['image'] for page in manifest['pages']], ['2.jpg', '10.jpg'])
        self.assertTrue(all(page['status'] == 'unlabeled' for page in manifest['pages']))

    def test_manifest_starts_pages_as_unlabeled(self):
        with tempfile.TemporaryDirectory() as root:
            image = np.full((100, 120, 3), 255, dtype=np.uint8)
            cv2.imwrite(os.path.join(root, '1.jpg'), image)
            os.makedirs(os.path.join(root, '_ignored'))
            cv2.imwrite(os.path.join(root, '_ignored', '2.jpg'), image)
            manifest = evaluator.create_annotation_manifest(root, profile='exam')
        self.assertEqual(len(manifest['pages']), 1)
        self.assertEqual(manifest['pages'][0]['status'], 'unlabeled')
        self.assertEqual(manifest['pages'][0]['profile'], 'exam')

    def test_existing_output_metadata_can_be_imported_as_unverified_suggestions(self):
        with tempfile.TemporaryDirectory() as root:
            image_root = os.path.join(root, 'pages')
            output_root = os.path.join(root, 'output', '一、填空', '第1题')
            os.makedirs(image_root)
            os.makedirs(output_root)
            image_path = os.path.join(image_root, '1.jpg')
            cv2.imwrite(image_path, np.full((100, 120, 3), 255, dtype=np.uint8))
            manifest_path = os.path.join(root, 'annotations.json')
            manifest = evaluator.create_annotation_manifest(image_root)
            with open(manifest_path, 'w', encoding='utf-8') as file:
                json.dump(manifest, file)
            with open(os.path.join(output_root, 'metadata.json'), 'w', encoding='utf-8') as file:
                json.dump({
                    'source': {
                        'image': image_path,
                        'question_box': [10, 20, 110, 90],
                    },
                }, file)
            result = annotate_cuts.import_prediction_metadata(
                manifest, manifest_path, os.path.join(root, 'output'),
            )
        self.assertEqual(result['boxes'], 1)
        self.assertEqual(manifest['pages'][0]['questions'][0]['bbox'], [10, 20, 110, 90])
        self.assertTrue(manifest['pages'][0]['questions'][0]['suggested'])
        self.assertEqual(manifest['pages'][0]['status'], 'unlabeled')


if __name__ == '__main__':
    unittest.main()
