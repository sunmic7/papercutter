"""正式交付版本重提时只裁改动过的框，其余裁图跨版本复用。

服务器每次生成都落在一个全新的版本目录里，目录内没有 ``manifest.json``，
所以旧的增量签名比对在正式出包这条路径上等于失效，每次都会把整本书重裁。
``reuse_root`` 把上一版作为只读基线：签名一致的图硬链接到新版本，只有真的
改过的框才重新解码整页扫描图。这些用例锁住「复用」与「该重裁时必须重裁」。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_k12_delivery_contract import build_project  # noqa: E402

from online_review_beta.central_library import CentralLibrary  # noqa: E402
from review_tools import delivery_workspace as dw  # noqa: E402
from review_tools import export_manual_annotations as exp  # noqa: E402
from review_tools import manual_annotation_data as data  # noqa: E402
from review_tools import online_review_sync as snap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CENTRAL_LIBRARY_PY = ROOT / "online_review_beta" / "central_library.py"


class CropReuseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.project_path = build_project(
            self.root / "book",
            texts=["第 1 题：求 1+1 的值。", "第 2 题：求 1+2 的值。"],
            output_root=self.root / "delivery")
        exp._decoded_page.cache_clear()
        snap._PAGE_SIZE_CACHE.clear()
        self.addCleanup(exp._decoded_page.cache_clear)
        self.addCleanup(snap._PAGE_SIZE_CACHE.clear)

    def generate(self, name: str, reuse: Path | None = None) -> dict:
        project = data.load_project(self.project_path)
        project["delivery_output_root"] = str(self.root / name)
        data.save_project(project, self.project_path, increment_revision=False)
        return dw.sync_delivery_workspace(
            self.project_path, self.root / "runtime",
            materialize_all=True, reuse_root=reuse)

    @staticmethod
    def k12_images(output: Path) -> list[Path]:
        return sorted((Path(output) / "k12_dataset" / "images").glob("*.jpg"))

    def test_unchanged_resubmission_reuses_every_crop(self):
        first = self.generate("v1")
        self.assertEqual(first["crop_updates"], 4)  # 2 题 + 2 答案
        second = self.generate("v2", reuse=Path(first["output_path"]))
        self.assertEqual(second["crop_updates"], 0)
        self.assertEqual(second["crop_reused"], 4)
        self.assertTrue((Path(second["output_path"]) / "k12_dataset" / "dataset.jsonl").is_file())

        signatures = json.loads((Path(second["output_path"]) / "k12_dataset"
                                 / "image_signatures.json").read_text(encoding="utf-8"))
        self.assertTrue(signatures)
        old_images = self.k12_images(first["output_path"])
        self.assertTrue(old_images)
        for old in old_images:
            new = Path(second["output_path"]) / "k12_dataset" / "images" / old.name
            self.assertTrue(new.is_file())
            # 硬链接而不是重新解码编码（同卷 inode 相同）。
            self.assertEqual(old.stat().st_ino, new.stat().st_ino)

    def test_pruning_previous_version_keeps_reused_files(self):
        """版本清理会删掉上一版目录；硬链接不能把新版的图一起带走。"""
        import shutil

        first = self.generate("v1")
        second = self.generate("v2", reuse=Path(first["output_path"]))
        kept = self.k12_images(second["output_path"])
        self.assertTrue(kept)
        before = [(item.name, item.read_bytes()) for item in kept]
        shutil.rmtree(first["output_path"])
        for name, payload in before:
            survivor = Path(second["output_path"]) / "k12_dataset" / "images" / name
            self.assertEqual(survivor.read_bytes(), payload)

    def test_changed_box_recrops_only_that_record(self):
        first = self.generate("v1")
        project = data.load_project(self.project_path)
        project["pages"][0]["questions"][0]["bbox"] = [10, 10, 290, 230]
        data.save_project(project, self.project_path, increment_revision=False)
        second = self.generate("v2", reuse=Path(first["output_path"]))
        self.assertEqual(second["crop_updates"], 1)
        self.assertEqual(second["crop_reused"], 3)

    def test_source_image_change_never_reuses_stale_pixels(self):
        first = self.generate("v1")
        question_image = self.root / "book" / "questions" / "1.jpg"
        payload = cv2.imencode(".jpg", np.full((500, 500, 3), 128, dtype=np.uint8))[1]
        question_image.write_bytes(payload.tobytes())
        second = self.generate("v2", reuse=Path(first["output_path"]))
        # 题目页两张裁图必须重裁，答案页两张不受影响仍可复用。
        self.assertEqual(second["crop_updates"], 2)
        self.assertEqual(second["crop_reused"], 2)
        for old in self.k12_images(first["output_path"]):
            new = Path(second["output_path"]) / "k12_dataset" / "images" / old.name
            self.assertNotEqual(old.stat().st_ino, new.stat().st_ino)

    def test_generation_without_reuse_root_still_falls_back_to_full_crop(self):
        """老调用方不传 reuse_root 时行为不变（本地样例题库等增量路径）。"""
        first = self.generate("v1")
        second = self.generate("v2")
        self.assertEqual(first["crop_updates"], 4)
        self.assertEqual(second["crop_updates"], 4)
        self.assertEqual(second["crop_reused"], 0)


class DecodeCacheTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.project_path = build_project(
            self.root / "book", texts=["第 1 题：求 1+1 的值。"],
            output_root=self.root / "delivery")
        self.image = self.root / "book" / "questions" / "1.jpg"
        exp._decoded_page.cache_clear()
        snap._PAGE_SIZE_CACHE.clear()
        self.addCleanup(exp._decoded_page.cache_clear)
        self.addCleanup(snap._PAGE_SIZE_CACHE.clear)

    def test_same_page_is_decoded_once(self):
        first = exp._page_image(str(self.image))
        second = exp._page_image(str(self.image))
        self.assertIs(first, second)
        self.assertEqual(exp._decoded_page.cache_info().hits, 1)

    def test_page_cache_key_tracks_source_mtime(self):
        first = exp._page_image(str(self.image))
        stat = self.image.stat()
        os.utime(self.image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        second = exp._page_image(str(self.image))
        self.assertIsNot(first, second)

    def test_snapshot_page_size_cache_avoids_second_decode(self):
        raw = self.image.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        size = snap._page_size(digest, raw)
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(size, (decoded.shape[0], decoded.shape[1]))
        with patch("cv2.imdecode", side_effect=AssertionError("缓存命中不该再解码")):
            self.assertEqual(snap._page_size(digest, raw), size)

    def test_answer_signature_has_a_bumpable_crop_version(self):
        # 跨版本复用靠签名比对，答案裁图算法变了必须升版本，否则会一直复用旧图。
        self.assertIn("answer_crop_v1", inspect.getsource(dw._source_signature))
        project = data.load_project(self.project_path)
        page = project["answer_pages"][0]
        record = page["answers"][0]
        before = dw._source_signature(project, page, record, True)
        record["bbox"] = [5, 5, 295, 235]
        self.assertNotEqual(before, dw._source_signature(project, page, record, True))


class ServerWiringTests(unittest.TestCase):
    def test_published_version_dir_is_safe_and_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            version = root / "versions" / "v1"
            version.mkdir(parents=True)
            (version / "manifest.json").write_text('{"items": []}', encoding="utf-8")
            current = root / "current.json"
            current.write_text(json.dumps({"path": "versions/v1"}), encoding="utf-8")
            self.assertEqual(CentralLibrary._published_version_dir(root), version.resolve())

            current.write_text(json.dumps({"path": "../../escape"}), encoding="utf-8")
            self.assertIsNone(CentralLibrary._published_version_dir(root))

            current.write_text(json.dumps({"path": "versions/v1"}), encoding="utf-8")
            (version / "manifest.json").unlink()
            self.assertIsNone(CentralLibrary._published_version_dir(root))

            current.unlink()
            self.assertIsNone(CentralLibrary._published_version_dir(root))

    def test_generate_delivery_passes_reuse_root(self):
        source = CENTRAL_LIBRARY_PY.read_text(encoding="utf-8")
        body = source[source.index("def generate_delivery"):
                      source.index("def delivery_page_coverage")]
        self.assertIn("reuse_root = self._published_version_dir(final_root)", body)
        self.assertIn("reuse_root=reuse_root)", body)
        self.assertIn('"crop_reuse"', body)
        engine = (ROOT / "review_tools" / "delivery_workspace.py").read_text(encoding="utf-8")
        self.assertIn("reuse_root: str | Path | None = None", engine)
        self.assertIn("image_signatures.json", engine)


if __name__ == "__main__":
    unittest.main()
