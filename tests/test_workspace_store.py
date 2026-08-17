from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from review_tools.workspace_store import database_path, page_snapshot, sync_project


class WorkspaceStoreTests(unittest.TestCase):
    def test_same_named_projects_in_different_books_use_distinct_databases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            first = root / "book-one" / "PaperFlow标注项目.json"
            second = root / "book-two" / "PaperFlow标注项目.json"

            first_database = database_path(first, runtime)
            second_database = database_path(second, runtime)

            self.assertNotEqual(first_database, second_database)
            self.assertEqual(first_database.parent, second_database.parent)

    def test_project_is_mirrored_transactionally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project_path = root / "project.json"
            project = {
                "book_root": str(root / "book"), "annotation_version": 5, "updated_at": "now",
                "pages": [{"image": "1.png", "source_path": "C:/book/1.png", "status": "accepted", "questions": [
                    {"id": "Q1", "bbox": [1, 2, 30, 40], "display_number": "一-1", "region_role": "question", "text": "题目"}
                ]}],
                "answer_pages": [], "pairs": [{"question_id": "Q1", "answer_ids": ["A1"], "status": "accepted"}],
            }
            db_path = sync_project(project_path, project, root)
            self.assertTrue(db_path.is_file())
            with closing(sqlite3.connect(db_path)) as connection:
                self.assertEqual(connection.execute("SELECT record_count FROM page").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT text FROM region WHERE region_id='Q1'").fetchone()[0], "题目")
                self.assertEqual(connection.execute("SELECT status FROM relation WHERE question_id='Q1'").fetchone()[0], "accepted")
            snapshot = page_snapshot(project_path, project, root, "question", 0)
            self.assertEqual(snapshot["regions"][0]["bbox"], [1, 2, 30, 40])


if __name__ == "__main__":
    unittest.main()
