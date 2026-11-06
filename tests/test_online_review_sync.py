import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import online_review_sync


class OnlineReviewSyncTests(unittest.TestCase):
    def test_title_is_preserved_across_generated_storage_folders(self):
        project = {"delivery_book_id": "B0123456789ABCDEF0123456789ABCDEF",
            "book_root": "J:/central/B0123456789ABCDEF0123456789ABCDEF",
            "source_book_name": "数学六年级下册（北师大版）"}
        self.assertEqual(online_review_sync.project_book_name(project, "cache/project.json"), "数学六年级下册（北师大版）")
        project.pop("source_book_name")
        project["book_root"] = r"F:\测试\原版教材"
        self.assertEqual(online_review_sync.project_book_name(project, "cache/project.json"), "原版教材")

    def test_project_sync_document_removes_local_textbook_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            project_path = Path(directory) / "PaperFlow标注项目.json"
            project_path.write_text(
                json.dumps({
                    "delivery_book_id": "BACCOUNT", "book_root": r"G:\教材\私有路径",
                    "image_root": r"G:\教材\私有路径", "answer_image_root": r"G:\答案\私有路径",
                    "book_metadata": {"book_name": "账号归档教材"}, "pages": [], "answer_pages": [],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            document = online_review_sync.build_project_sync_snapshot(project_path)
            self.assertEqual(document["book"], {"id": "BACCOUNT", "name": "账号归档教材"})
            self.assertNotIn("book_root", document["project"])
            self.assertNotIn("image_root", document["project"])
            self.assertNotIn("answer_image_root", document["project"])

    def test_snapshot_reads_a_page_in_a_chinese_windows_style_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "测试教材"
            root.mkdir()
            page_path = root / "第1页.jpg"
            image = np.full((48, 64, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imencode(".jpg", image)[0])
            encoded = cv2.imencode(".jpg", image)[1]
            encoded.tofile(str(page_path))
            project_path = root / "PaperFlow标注项目.json"
            project_path.write_text(
                '{"image_root": "' + str(root).replace("\\", "\\\\") + '", '
                '"pages": [{"image": "第1页.jpg", "status": "accepted", '
                '"questions": [{"id": "q-1", "bbox": [1, 2, 30, 40]}]}], '
                '"answer_pages": []}',
                encoding="utf-8",
            )

            snapshot = online_review_sync.build_review_snapshot(project_path)

            self.assertEqual(snapshot["pages"][0]["width"], 64)
            self.assertEqual(snapshot["pages"][0]["height"], 48)
            self.assertTrue(snapshot["pages"][0]["preview"].startswith("data:image/jpeg;base64,"))
