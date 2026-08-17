from __future__ import annotations

import unittest

from question_detector import Detection
from training_tools.evaluate_question_detector_acceptance import evaluate_record
from training_tools.validate_question_detector_contract import validate_contract_metadata


class QuestionDetectorContractTests(unittest.TestCase):
    def test_current_contract_metadata_is_accepted(self):
        errors = validate_contract_metadata(
            {
                "schema_version": 2, "model_name": "question_detector", "classes": {"0": "question"},
                "input": {"shape": [1, 3, 640, 640]},
                "output": {"format": "ultralytics_nms", "nms_in_model": True, "shape": [1, 300, 6]},
                "artifact": {"filename": "question_detector.onnx", "sha256": "test"},
            },
            [1, 3, 640, 640], "tensor(float)", [1, 300, 6], "tensor(float)",
        )
        self.assertEqual(errors, [])

    def test_raw_yolo_output_and_wrong_nms_are_rejected(self):
        errors = validate_contract_metadata(
            {
                "schema_version": 2, "model_name": "question_detector", "classes": {"0": "question"},
                "input": {"shape": [1, 3, 640, 640]},
                "output": {"format": "ultralytics_nms", "nms_in_model": False, "shape": [1, 84, 8400]},
                "artifact": {"filename": "question_detector.onnx", "sha256": "test"},
            },
            [1, 3, 640, 640], "tensor(float)", [1, 84, 8400], "tensor(float)",
        )
        self.assertTrue(any("NMS" in error for error in errors))
        self.assertTrue(any("nms_in_model" in error for error in errors))

    def test_acceptance_record_uses_count_and_iou_not_exact_coordinates(self):
        result = evaluate_record(
            [Detection((11, 11, 99, 99), 0.9)],
            {"image": "sample.jpg", "expected_count": 1, "expected_boxes": [[10, 10, 100, 100]], "minimum_iou": 0.85},
        )
        self.assertTrue(result["ok"])
