from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from review_tools import manual_annotation_data as data
from desktop_annotation import paperflow_annotation_api as api


class BookMetadataApiTests(unittest.TestCase):
    def test_save_page_does_not_rebuild_the_whole_sqlite_read_model(self):
        source = Path(api.__file__).read_text(encoding="utf-8")
        start = source.index("def _save_page(")
        end = source.index("def _match_candidates", start)
        self.assertNotIn("sync_project(", source[start:end])

    def test_open_book_does_not_search_legacy_or_other_paths(self):
        source = Path(api.__file__).read_text(encoding="utf-8")
        start = source.index("def _open_book(")
        end = source.index("def _load_project", start)
        method = source[start:end]
        self.assertNotIn("legacy_book_project_path", method)
        self.assertNotIn("_recover_moved_book_project", method)
        existing_branch = method[method.index("if project_path.is_file()") : method.index("else:")]
        self.assertNotIn("reconcile_book_pages", existing_branch)
        self.assertNotIn("create_book_project", existing_branch)

    def test_opening_a_copied_book_recovers_matching_legacy_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_book, new_book = root / "old", root / "new"
            runtime = root / "runtime" / "manual_annotation_projects"
            old_book.mkdir(); new_book.mkdir(); runtime.mkdir(parents=True)
            image = np.full((30, 40, 3), 255, dtype=np.uint8)
            for book in (old_book, new_book):
                cv2.imencode(".jpg", image)[1].tofile(str(book / "1.jpg"))
            old_project = data.create_book_project(str(old_book))
            old_project["pages"][0]["questions"] = [data.new_record([1, 1, 20, 20], major_number="一")]
            data.save_project(old_project, runtime / "legacy.json")
            empty_project = data.create_book_project(str(new_book))

            with patch.object(api, "ROOT", root):
                recovered = api._recover_moved_book_project(new_book, empty_project)
            self.assertIsNotNone(recovered)
            self.assertEqual(api._annotation_count(recovered), 1)
            self.assertEqual(recovered["image_root"], str(new_book))

    def test_content_match_tolerates_generated_pages_in_legacy_project(self):
        clean = ((("1.jpg", 10), ("2.jpg", 20)), ())
        polluted = ((("1.jpg", 10), ("delivery/Q.jpg", 5), ("2.jpg", 20)), ())
        self.assertTrue(api._same_book_content(clean, polluted))
        self.assertFalse(api._same_book_content(clean, ((("other.jpg", 10),), ())))

    def test_metadata_is_persisted_at_project_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            project = data.create_project(str(images))
            project_path = root / "project.json"
            data.save_project(project, project_path)
            payload = {
                "project_path": str(project_path),
                "metadata": {"language": "zh", "stage": "小学", "grade": "六年级", "subject": "数学",
                             "answer_mode": "generate_later"},
            }
            with patch.object(api, "_load_project", return_value=(project_path, data.load_project(project_path))):
                result = api._set_book_metadata(payload)
            saved = data.load_project(project_path)
            self.assertEqual(saved["language"], "zh")
            self.assertEqual(saved["stage"], "小学")
            self.assertEqual(saved["grade"], "六年级")
            self.assertEqual(saved["subject"], "数学")
            self.assertEqual(saved["answer_mode"], "generate_later")
            self.assertIsNone(result["delivery"])
            self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
