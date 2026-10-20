from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools.photo_qa_import import build_batch, parse_named_images, write_batch


class PhotoQaImportTests(unittest.TestCase):
    @staticmethod
    def _image(path: Path, width: int = 900, height: int = 800) -> None:
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (width - 40, height - 40), (0, 0, 0), 4)
        cv2.putText(image, "QA", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("cannot write fixture image")
        encoded.tofile(str(path))

    @staticmethod
    def _ocr(path: Path) -> str:
        return {"5217.png": "题目（1）（2）", "5217答案.png": "解答第一页", "5217答案1.png": "解答续页"}.get(path.name, "识别文本")

    @staticmethod
    def _good_review(_question: Path, candidates: list[Path]) -> dict:
        return {
            "question_text": "设系统满足条件，求（1）（2）。",
            "answer": "（1）a；（2）b。",
            "cot": "由题设列方程，分别求解两个小问。",
            "image_domain": "物理", "QA_diff": "大学", "overall_confidence": 0.96,
            "candidates": [
                {"filename": candidates[0].name, "role": "answer", "confidence": 0.97},
                {"filename": candidates[1].name, "role": "continuation", "confidence": 0.96},
            ],
        }

    def test_named_question_with_answer_continuation_exports_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "大学物理10.24"
            root.mkdir()
            for name in ("5217.png", "5217答案.png", "5217答案1.png"):
                self._image(root / name)
            batch = build_batch(root, semantic_reviewer=self._good_review, ocr_runner=self._ocr)
            self.assertEqual(len(batch["dataset"]), 1)
            record = batch["dataset"][0]
            self.assertEqual(set(record), {"MD5", "image_domain", "image_size", "QA_diff", "question", "opinion", "answer", "COT"})
            self.assertEqual(record["image_domain"], "物理")
            self.assertEqual(record["QA_diff"], "大学")
            self.assertEqual(record["image_size"], {"height": 800, "width": 900})
            self.assertRegex(record["MD5"], r"^[0-9a-f]{32}$")
            self.assertEqual(batch["items"][0]["accepted_answer_images"], ["5217答案.png", "5217答案1.png"])

    def test_answer_figure_is_kept_as_support_image_not_a_second_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "大学物理10.24"; root.mkdir()
            for name in ("5003.png", "5003答案.png", "5003答案1.png"):
                self._image(root / name)
            def reviewer(_question: Path, candidates: list[Path]) -> dict:
                result = self._good_review(_question, candidates)
                result["candidates"][1] = {"filename": "5003答案1.png", "role": "answer_figure", "confidence": 0.99}
                return result
            batch = build_batch(root, semantic_reviewer=reviewer, ocr_runner=self._ocr)
            item = batch["items"][0]
            self.assertEqual(item["accepted_answer_images"], ["5003答案.png"])
            self.assertEqual(item["support_images"], ["5003答案1.png"])
            self.assertEqual(len(batch["dataset"]), 1)

    def test_unrelated_same_prefix_file_is_held_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "大学物理10.24"; root.mkdir()
            for name in ("5010.png", "5010答案.png", "5010答案1.png"):
                self._image(root / name)
            def reviewer(_question: Path, candidates: list[Path]) -> dict:
                result = self._good_review(_question, candidates)
                result["candidates"][1] = {"filename": "5010答案1.png", "role": "unrelated", "confidence": 0.98}
                return result
            batch = build_batch(root, semantic_reviewer=reviewer, ocr_runner=self._ocr)
            self.assertEqual(batch["dataset"], [])
            self.assertIn("candidate_assignment_unresolved", batch["items"][0]["quality_flags"])

    def test_short_edge_never_enters_final_dataset_and_output_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "大学物理10.24"; root.mkdir()
            self._image(root / "5003.png", width=1000, height=600)
            self._image(root / "5003答案.png")
            batch = build_batch(root, semantic_reviewer=lambda q, c: {
                "question_text": "题干", "answer": "答案", "cot": "解析", "overall_confidence": 0.99,
                "candidates": [{"filename": c[0].name, "role": "answer", "confidence": 0.99}],
            }, ocr_runner=self._ocr)
            self.assertEqual(batch["dataset"], [])
            self.assertIn("short_edge_below_720", batch["items"][0]["quality_flags"])
            report = write_batch(batch, Path(directory) / "out")
            self.assertEqual(report["eligible_count"], 0)
            self.assertEqual((Path(report["dataset_file"])).read_text(encoding="utf-8"), "")
            self.assertEqual(len(json.loads(Path(report["review_file"]).read_text(encoding="utf-8"))), 1)

    def test_parse_keeps_orphan_answer_and_unknown_name_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self._image(root / "6001答案.png"); self._image(root / "备注图.png")
            parsed = parse_named_images(root)
            self.assertEqual(parsed["questions"], {})
            self.assertEqual([path.name for path in parsed["candidates"]["6001"]], ["6001答案.png"])
            self.assertEqual([path.name for path in parsed["unrecognised"]], ["备注图.png"])


if __name__ == "__main__":
    unittest.main()
