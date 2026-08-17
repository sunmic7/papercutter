import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import vlm_benchmark as benchmark


class VlmBenchmarkContractTests(unittest.TestCase):
    def test_normalized_boxes_map_to_original_pixels(self):
        self.assertEqual(benchmark.normalized_box_to_pixels([100, 200, 900, 800], 2000, 1000), [200, 200, 1800, 800])

    def test_prediction_rejects_unknown_shared_region(self):
        payload = {
            "coordinate_system": "normalized_0_1000",
            "questions": [{"question_id": "1", "bbox": [0, 0, 500, 500], "shared_region_ids": ["missing"]}],
            "shared_regions": [],
        }
        with self.assertRaisesRegex(ValueError, "不存在"):
            benchmark.normalize_prediction(payload, 1000, 1000)

    def test_prediction_preserves_shared_region_relationship(self):
        payload = {
            "coordinate_system": "normalized_0_1000",
            "questions": [{"question_id": "1", "bbox": [0, 200, 1000, 900], "shared_region_ids": ["material_01"], "contains_figure": True}],
            "shared_regions": [{"id": "material_01", "type": "shared_material", "bbox": [0, 0, 1000, 200]}],
        }
        result = benchmark.normalize_prediction(payload, 1200, 2000)
        self.assertEqual(result["questions"][0]["bbox"], [0, 400, 1200, 1800])
        self.assertEqual(result["shared_regions"][0]["bbox"], [0, 0, 1200, 400])

    def test_strict_metric_distinguishes_under_crop(self):
        metrics = benchmark._page_metrics(
            [{"bbox": [0, 0, 100, 90]}],
            [{"bbox": [0, 0, 100, 100]}],
        )
        self.assertEqual(metrics["matched_count"], 1)
        self.assertEqual(metrics["strict_complete_count"], 0)
        self.assertFalse(metrics["perfect_page"])

    def test_json_fence_is_accepted_for_non_schema_fallback(self):
        self.assertEqual(
            benchmark._extract_json("```json\n{\"coordinate_system\": \"normalized_0_1000\"}\n```"),
            {"coordinate_system": "normalized_0_1000"},
        )

    def test_dry_run_writes_a_reproducible_report_without_a_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "page.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), np.full((80, 120, 3), 255, dtype=np.uint8)))
            manifest_path = root / "annotations.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "image_root": str(root),
                "pages": [{"image": "page.jpg", "status": "labeled", "questions": [{"bbox": [0, 0, 120, 80]}]}],
            }), encoding="utf-8")
            report = benchmark.run_benchmark(argparse.Namespace(
                annotations=str(manifest_path), output=str(root / "run"), include_unlabeled=False,
                max_pages=None, api_key_env="TEST_MISSING_KEY", dry_run=True,
                provider="test", model="test-model", base_url="https://example.invalid",
                structured_mode="json_schema", max_long_side=2200, repeats=1, timeout=1,
                input_price=1.0, output_price=10.0, write_previews=True,
            ))
            self.assertEqual(report["summary"]["dry_run_page_count"], 1)
            self.assertTrue((root / "run" / "config.json").is_file())
            self.assertTrue((root / "run" / "report.json").is_file())

    def test_prepare_image_accepts_a_chinese_windows_style_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "中文目录"
            folder.mkdir()
            image_path = folder / "页面.jpg"
            ok, encoded = cv2.imencode(".jpg", np.full((80, 120, 3), 255, dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(str(image_path))
            data_uri, dimensions = benchmark.prepare_image(image_path, 2200)
            self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
            self.assertEqual(dimensions["original_width"], 120)

    def test_manifest_reader_accepts_windows_utf8_bom(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "bom.json"
            manifest.write_text(json.dumps({"schema_version": 1, "pages": []}), encoding="utf-8-sig")
            _, pages, _ = benchmark._load_manifest(manifest, include_unlabeled=False)
            self.assertEqual(pages, [])


if __name__ == "__main__":
    unittest.main()
