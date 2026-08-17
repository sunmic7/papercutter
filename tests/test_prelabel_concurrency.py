from __future__ import annotations

import importlib.util
import threading
import time
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "desktop_annotation" / "paperflow_annotation_api.py"
SPEC = importlib.util.spec_from_file_location("paperflow_annotation_api_concurrency", API_PATH)
api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(api)


class PrelabelConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        self.first = self.root / "first.png"
        self.second = self.root / "second.png"
        cv2.imwrite(str(self.first), image)
        cv2.imwrite(str(self.second), image)
        with api.PRELABEL_INFLIGHT_LOCK:
            api.PRELABEL_INFLIGHT.clear()

    def tearDown(self) -> None:
        with api.PRELABEL_INFLIGHT_LOCK:
            api.PRELABEL_INFLIGHT.clear()
        self.temp.cleanup()

    @staticmethod
    def _question_payload(path: Path, page_index: int = 0) -> dict:
        return {
            "kind": "question", "image_path": str(path),
            "page_index": page_index, "profile": "exam",
        }

    def test_identical_inflight_page_is_inferred_only_once(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        call_count = 0
        counter_lock = threading.Lock()

        def suggestions(*_args, **_kwargs):
            nonlocal call_count
            with counter_lock:
                call_count += 1
            entered.set()
            self.assertTrue(release.wait(2))
            return [{"id": "q-one", "bbox": [0, 0, 20, 20], "major_number": "一"}]

        with mock.patch.object(api, "_question_suggestions", side_effect=suggestions):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(api._prelabel, self._question_payload(self.first))
                self.assertTrue(entered.wait(2))
                second = executor.submit(api._prelabel, self._question_payload(self.first))
                time.sleep(0.05)
                release.set()
                first_result = first.result(timeout=2)
                second_result = second.result(timeout=2)

        self.assertEqual(call_count, 1)
        self.assertEqual(first_result, second_result)
        self.assertIsNot(first_result, second_result)

    def test_different_pages_still_serialize_predictor_access(self) -> None:
        active = 0
        maximum_active = 0
        call_count = 0
        counter_lock = threading.Lock()
        start = threading.Barrier(2)

        def suggestions(*_args, **_kwargs):
            nonlocal active, maximum_active, call_count
            with counter_lock:
                active += 1
                call_count += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.08)
            with counter_lock:
                active -= 1
            return []

        def invoke(path: Path, page_index: int):
            start.wait(timeout=2)
            return api._prelabel(self._question_payload(path, page_index))

        with mock.patch.object(api, "_question_suggestions", side_effect=suggestions):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(invoke, self.first, 0),
                    executor.submit(invoke, self.second, 1),
                ]
                for future in futures:
                    future.result(timeout=3)

        self.assertEqual(call_count, 2)
        self.assertEqual(maximum_active, 1)

    def test_dense_answer_candidates_are_flagged_without_truncation(self) -> None:
        records = [
            {
                "id": f"a-{index}", "bbox": [0, index, 20, index + 1],
                "major_number": "一", "minor_number": str(index + 1),
                "context": "单元一",
            }
            for index in range(api.ANSWER_PRELABEL_REVIEW_THRESHOLD)
        ]
        with mock.patch.object(api, "_answer_suggestions", return_value=records):
            result = api._prelabel({
                "kind": "answer", "image_path": str(self.first),
                "page_index": 0, "profile": "exam",
            })

        self.assertEqual(len(result["records"]), len(records))
        self.assertTrue(result["diagnostics"]["requires_review"])
        self.assertFalse(result["diagnostics"]["bulk_accept_safe"])
        self.assertIn("dense_answer_candidates", result["diagnostics"]["warning_codes"])


if __name__ == "__main__":
    unittest.main()
