from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import answer_detector
import answer_region_detector
from training_tools import evaluate_pairing_feedback
from question_detector import Detection
from training.prepare_reviewed_answers import prepare_reviewed_answers
from training.prepare_hilex_answer_blocks import prepare_hilex_answer_blocks


class ReviewedAnswerPreparationTests(unittest.TestCase):
    def test_only_accepted_answer_pairs_become_yolo_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "answer-page.jpg"
            cv2.imwrite(str(source), np.zeros((100, 200, 3), dtype=np.uint8))
            manifest = root / "training_manifest.jsonl"
            records = [
                {
                    "dataset_split": "train", "pair_valid": True, "answer_valid": True,
                    "answer": {"source_image": str(source), "target_box": [20, 10, 120, 60]},
                },
                {
                    "dataset_split": "train", "pair_valid": False, "answer_valid": True,
                    "answer": {"source_image": str(source), "target_box": [0, 0, 10, 10]},
                },
            ]
            manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            summary = prepare_reviewed_answers(manifest, root / "dataset")
            self.assertEqual(summary["splits"]["train"]["images"], 1)
            self.assertEqual(summary["splits"]["train"]["answer_boxes"], 1)
            label = next((root / "dataset" / "train" / "labels").glob("*.txt")).read_text(encoding="utf-8")
            self.assertEqual(label, "0 0.3500000 0.3500000 0.5000000 0.5000000\n")

    def test_hilex_answer_projection_keeps_answer_block_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for split in ("train", "valid", "test"):
                (source / split / "images").mkdir(parents=True)
                (source / split / "labels").mkdir(parents=True)
                (source / split / "images" / "page.jpg").write_bytes(b"image")
                (source / split / "labels" / "page.txt").write_text(
                    "0 0.5 0.5 0.4 0.2\n3 0.5 0.5 0.8 0.5\n", encoding="utf-8"
                )
            summary = prepare_hilex_answer_blocks(source, root / "output")
            label = (root / "output" / "train" / "labels" / "page.txt").read_text(encoding="utf-8")
            self.assertEqual(label, "0 0.5 0.5 0.4 0.2\n")
            self.assertEqual(summary["splits"]["train"]["answer_boxes"], 1)


class AnswerDetectorFusionTests(unittest.TestCase):
    def test_matching_answer_detection_refines_only_the_matching_record(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        records = [{"crop_img": image[10:60, 20:120], "source_box": [20, 10, 120, 60]}]
        refined, stats = answer_detector.refine_answer_records(
            records, image, [Detection((22, 8, 118, 62), 0.91)], min_iou=0.55,
        )
        self.assertEqual(refined[0]["source_box"], [22, 8, 118, 62])
        self.assertTrue(refined[0]["detector_assisted"])
        self.assertEqual(stats["refined"], 1)


class AnswerRegionFusionTests(unittest.TestCase):
    def test_internal_region_can_extend_cut_without_reaching_next_question(self):
        image = np.zeros((240, 220, 3), dtype=np.uint8)
        cuts = [
            {"crop_img": image[20:120, :], "_source_box": [0, 20, 220, 120]},
            {"crop_img": image[150:220, :], "_source_box": [0, 150, 220, 220]},
        ]
        protected, stats = answer_region_detector.protect_question_cuts(
            cuts, image, [Detection((5, 105, 210, 135), 0.91)], padding=8,
        )
        self.assertEqual(stats["protected"], 1)
        self.assertEqual(protected[0]["_source_box"], [0, 20, 220, 142])
        self.assertLess(protected[0]["_source_box"][3], protected[1]["_source_box"][1])
        self.assertTrue(protected[0]["_answer_region_assisted"])

    def test_region_that_reaches_next_question_is_rejected(self):
        image = np.zeros((240, 220, 3), dtype=np.uint8)
        cuts = [
            {"crop_img": image[20:120, :], "_source_box": [0, 20, 220, 120]},
            {"crop_img": image[150:220, :], "_source_box": [0, 150, 220, 220]},
        ]
        protected, stats = answer_region_detector.protect_question_cuts(
            cuts, image, [Detection((5, 105, 210, 160), 0.91)], padding=8,
        )
        self.assertEqual(stats["protected"], 0)
        self.assertEqual(stats["skipped_cross_next"], 1)
        self.assertEqual(protected[0]["_source_box"], [0, 20, 220, 120])


class PairingEvaluationTests(unittest.TestCase):
    def test_pairing_report_uses_only_reviewed_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "training_manifest.jsonl"
            records = [
                {"label_provenance": "reviewed", "pair_valid": True, "pairing": {"decision_band": "auto"}},
                {"label_provenance": "reviewed", "pair_valid": False, "pairing": {"decision_band": "auto"}},
                {"label_provenance": "reviewed", "pair_valid": True, "pairing": {"decision_band": "review"}},
                {"label_provenance": "auto", "pair_valid": True, "pairing": {"decision_band": "auto"}},
            ]
            manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            report = evaluate_pairing_feedback.evaluate(manifest)
            self.assertEqual(report["reviewed_pairs"], 3)
            self.assertEqual(report["automatic_precision"], 0.5)
            self.assertEqual(report["automatic_accepted_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
