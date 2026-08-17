from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import cut_questions_v4 as cutter
import live_review_protocol as protocol


class LiveReviewProtocolTests(unittest.TestCase):
    def test_decision_is_clamped_and_only_preserves_known_source_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {
                "PAPERCUTTER_LIVE_REVIEW_MODE": "gate",
                "PAPERCUTTER_LIVE_REVIEW_DIR": temporary,
            }, clear=False):
                request_path, request = protocol.create_request(
                    image_path="page.jpg",
                    image_size=(100, 80),
                    candidates=[{"id": "cut-1", "box": [10, 12, 60, 52], "number": 3}],
                    book_name="示例教材",
                )
                self.assertTrue(request_path.is_file())
                self.assertEqual(request["mode"], "gate")
                decision_path = protocol.write_decision(request, {
                    "candidates": [
                        {"id": "changed", "source_id": "cut-1", "box": [-2, 1, 120, 90]},
                        {"id": "manual", "source_id": "unknown", "box": [4, 5, 20, 25]},
                    ],
                })
                self.assertTrue(decision_path.is_file())
                decision = protocol.wait_for_decision(request, poll_seconds=0.01)
                self.assertEqual(decision["candidates"][0]["box"], [0, 1, 100, 80])
                self.assertEqual(decision["candidates"][0]["source_id"], "cut-1")
                self.assertEqual(decision["candidates"][1]["source_id"], "")

    def test_invalid_boxes_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"PAPERCUTTER_LIVE_REVIEW_DIR": temporary}, clear=False):
                _, request = protocol.create_request(
                    image_path=str(Path(temporary) / "page.jpg"), image_size=(50, 50),
                    candidates=[{"box": [20, 20, 20, 30]}, {"box": [1, 2, 30, 40]}],
                )
                self.assertEqual(len(request["candidates"]), 1)

    def test_gate_edit_replaces_crop_before_final_writer_uses_it(self):
        image = np.zeros((90, 120, 3), dtype=np.uint8)
        original = {
            "num": 4, "number_detected": True, "_source_path": "page.jpg",
            "_source_size": [120, 90], "_source_box": [10, 10, 40, 35],
            "_major_box": [0, 0, 120, 90], "_page_deskew_angle": 0.0,
            "crop_img": image[10:35, 10:40].copy(),
        }
        decision = {
            "candidates": [
                {"id": "cut-1", "source_id": "cut-1", "box": [5, 8, 70, 45], "number": 4},
                {"id": "manual", "source_id": "", "box": [10, 50, 75, 80]},
            ],
        }
        reviewed = cutter._apply_live_review_decision(image, "page.jpg", [original], decision)
        self.assertEqual([item["_source_box"] for item in reviewed], [[5, 8, 70, 45], [10, 50, 75, 80]])
        self.assertEqual(reviewed[0]["crop_img"].shape[:2], (37, 65))
        self.assertTrue(reviewed[0]["_live_reviewed"])
        self.assertTrue(reviewed[1]["_review_added"])
        self.assertEqual(reviewed[1]["_source_path"], os.path.abspath("page.jpg"))


if __name__ == "__main__":
    unittest.main()
