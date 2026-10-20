from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import question_detector
from training.prepare_hilex import prepare_hilex
from training.prepare_exam_questions import prepare_exam_questions


class HiLExPreparationTests(unittest.TestCase):
    def test_only_complete_question_boxes_are_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for split in ("train", "valid", "test"):
                images = source / split / "images"
                labels = source / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / "page.jpg").write_bytes(b"not-a-real-image")
                (labels / "page.txt").write_text(
                    "0 0.5 0.5 1.0 1.0\n"
                    "3 0.5 0.5 0.8 0.3\n"
                    "4 0.5 0.5 0.6 0.2\n",
                    encoding="utf-8",
                )
            summary = prepare_hilex(source, root / "output")
            label = (root / "output" / "train" / "labels" / "page.txt").read_text(encoding="utf-8")
            self.assertEqual(label, "0 0.5 0.5 0.8 0.3\n")
            self.assertEqual(summary["splits"]["train"]["question_boxes"], 1)
            self.assertIn("0: question", (root / "output" / "data.yaml").read_text(encoding="utf-8"))


class ExamQuestionPreparationTests(unittest.TestCase):
    def test_both_exam_question_variants_become_one_question_class(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for split in ("train", "valid", "test"):
                images = source / split / "images"
                labels = source / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / "page.jpg").write_bytes(b"not-a-real-image")
                (labels / "page.txt").write_text(
                    "0 0.2 0.2 0.3 0.2\n"
                    "1 0.7 0.7 0.3 0.2\n"
                    "9 0.5 0.5 1.0 1.0\n",
                    encoding="utf-8",
                )
            summary = prepare_exam_questions(source, root / "output")
            label = (root / "output" / "train" / "labels" / "page.txt").read_text(encoding="utf-8")
            self.assertEqual(label, "0 0.2 0.2 0.3 0.2\n0 0.7 0.7 0.3 0.2\n")
            self.assertEqual(summary["splits"]["train"]["question_boxes"], 2)


class DetectorFusionTests(unittest.TestCase):
    def test_matching_detection_refines_crop_and_source_box(self):
        image = np.zeros((140, 160, 3), dtype=np.uint8)
        cuts = [{
            "crop_img": image[20:100, 10:140],
            "_source_box": [10, 20, 140, 100],
            "_boundary_source": "ocr_number",
        }]
        detections = [question_detector.Detection((12, 18, 138, 102), 0.91)]
        refined, stats = question_detector.refine_legacy_cuts(cuts, image, detections)
        self.assertEqual(refined[0]["_source_box"], [12, 18, 138, 102])
        self.assertEqual(refined[0]["crop_img"].shape[:2], (84, 126))
        self.assertEqual(refined[0]["_boundary_source"], "detector_assisted")
        self.assertEqual(stats["refined"], 1)

    def test_unmatched_detection_cannot_change_legacy_cut(self):
        image = np.zeros((140, 160, 3), dtype=np.uint8)
        cuts = [{"crop_img": image[20:100, 10:140], "_source_box": [10, 20, 140, 100]}]
        refined, stats = question_detector.refine_legacy_cuts(
            cuts, image, [question_detector.Detection((0, 105, 100, 135), 0.95)],
        )
        self.assertEqual(refined[0]["_source_box"], [10, 20, 140, 100])
        self.assertEqual(stats["refined"], 0)

    def test_sequence_gap_recovers_one_high_confidence_question_box(self):
        image = np.zeros((320, 240, 3), dtype=np.uint8)
        cuts = [
            {
                "num": 6, "crop_img": image[20:220, :220],
                "_source_box": [0, 20, 220, 220], "_major_box": [0, 0, 220, 300],
                "_anchor_y": 20, "_source_path": "page.jpg", "_source_size": [240, 320],
            },
            {
                "num": 8, "crop_img": image[230:300, :220],
                "_source_box": [0, 230, 220, 300], "_major_box": [0, 0, 220, 300],
            },
        ]
        recovered, stats = question_detector.recover_sequence_gap_cuts(
            cuts, image, [question_detector.Detection((8, 130, 214, 215), 0.95)],
        )
        self.assertEqual([cut["num"] for cut in recovered], [6, 7, 8])
        self.assertEqual(recovered[0]["_source_box"], [0, 20, 220, 130])
        self.assertEqual(recovered[1]["_source_box"], [8, 130, 214, 215])
        self.assertTrue(recovered[1]["_inferred_number"])
        self.assertEqual(stats["recovered"], 1)

    def test_sequence_gap_does_not_accept_low_confidence_model_box(self):
        image = np.zeros((320, 240, 3), dtype=np.uint8)
        cuts = [
            {"num": 6, "crop_img": image[20:220, :220], "_source_box": [0, 20, 220, 220]},
            {"num": 8, "crop_img": image[230:300, :220], "_source_box": [0, 230, 220, 300]},
        ]
        recovered, stats = question_detector.recover_sequence_gap_cuts(
            cuts, image, [question_detector.Detection((8, 130, 214, 215), 0.80)],
        )
        self.assertEqual([cut["num"] for cut in recovered], [6, 8])
        self.assertEqual(stats["recovered"], 0)
        self.assertEqual(recovered[1]["_sequence_gap_before"], [7])


if __name__ == "__main__":
    unittest.main()
