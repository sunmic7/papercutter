"""The server can only crop what an employee workstation actually uploaded."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import online_review_sync


class SourceImageCollectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.book = self.root / "scans"
        (self.book / "rotated").mkdir(parents=True)
        image = np.full((500, 500, 3), 255, np.uint8)
        cv2.rectangle(image, (20, 20), (220, 220), (0, 0, 0), 2)
        self.original_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        self.rotated_bytes = cv2.imencode(".jpg", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))[1].tobytes()
        (self.book / "1.jpg").write_bytes(self.original_bytes)
        (self.book / "rotated" / "1.jpg").write_bytes(self.rotated_bytes)
        self.project_path = self.root / "PaperFlow标注项目.json"
        self.project_path.write_text(json.dumps({
            "kind": "papercutter_manual_annotation",
            "delivery_book_id": "BOOK-SYNC",
            "image_root": str(self.book),
            "book_root": str(self.book),
            "pages": [{
                "image": "1.jpg", "rotated_image": "rotated/1.jpg", "rotation": 90,
                "status": "accepted",
                "questions": [{"id": "q1", "bbox": [10, 10, 200, 200], "region_role": "question"}],
            }],
            "answer_pages": [], "ignored_pages": [],
        }, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_rotated_page_views_are_uploaded_with_the_original(self):
        pairs = online_review_sync.collect_project_source_images(self.project_path)
        relative = [name for name, _path in pairs]
        self.assertEqual(relative, ["1.jpg", "rotated/1.jpg"])
        self.assertEqual([path.read_bytes() for _name, path in pairs],
                         [self.original_bytes, self.rotated_bytes])

    def test_project_sync_snapshot_strips_local_roots(self):
        document = online_review_sync.build_project_sync_snapshot(self.project_path)
        project = document["project"]
        for key in ("image_root", "book_root", "answer_image_root"):
            self.assertNotIn(key, project)
        # load_project() normalises the book id into an opaque hash.
        self.assertEqual(document["book"]["id"], project["delivery_book_id"])
        self.assertTrue(document["book"]["id"])

    def test_review_snapshot_notice_points_at_server_generation(self):
        notice = online_review_sync.build_review_snapshot(
            self.project_path, include_previews=False)["notice"]
        self.assertIn("服务器", notice)
        self.assertNotIn("必须在本地生成", notice)


if __name__ == "__main__":
    unittest.main()
