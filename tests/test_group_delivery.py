from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import manual_annotation_data as data
from review_tools.group_delivery import export_scope, preflight_scope


class GroupDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.questions = self.root / "questions"
        self.answers = self.root / "answers"
        self.questions.mkdir()
        self.answers.mkdir()
        image = np.full((120, 180, 3), 255, np.uint8)
        cv2.putText(image, "7-1", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(str(self.questions / "q.png"), image)
        cv2.imwrite(str(self.answers / "a.png"), image)
        self.project = data.create_project(str(self.questions), str(self.answers))
        self.project["match_scopes"] = [{"id": "scope-1", "name": "第一单元"}]
        self.project["pages"][0].update({"status": "accepted", "match_scope_id": "scope-1"})
        self.project["answer_pages"][0].update({"status": "accepted", "match_scope_id": "scope-1"})
        self.project["pages"][0]["questions"] = [{
            "id": "q-1", "bbox": [5, 5, 170, 110], "major_number": "7", "minor_number": "1",
            "region_role": "question", "match_scope_id": "scope-1", "text": "题目",
        }]
        self.project["answer_pages"][0]["answers"] = [{
            "id": "a-1", "bbox": [5, 5, 170, 110], "major_number": "7", "minor_number": "1",
            "region_role": "question", "match_scope_id": "scope-1", "answer": "答案",
        }]
        data.set_pair(self.project, "q-1", ["a-1"], "accepted")
        self.project_path = self.root / "project.json"
        data.save_project(self.project, self.project_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_matched_question_and_answer_are_packaged_together(self) -> None:
        result = export_scope(self.project_path, self.root / "runtime", "scope-1")
        output = Path(result["output_path"])
        self.assertEqual(output.name, "第一单元")
        rows = [json.loads(line) for line in (output / "json" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer_status"], "matched")
        self.assertTrue((output / rows[0]["question"]["image_file"]).is_file())
        self.assertTrue((output / rows[0]["answers"][0]["image_file"]).is_file())

    def test_unmatched_question_can_be_deferred_for_answer_generation(self) -> None:
        project = data.load_project(self.project_path)
        project["pairs"] = []
        project["answer_mode"] = "generate_later"
        data.save_project(project, self.project_path)
        report = preflight_scope(data.load_project(self.project_path), "scope-1")
        self.assertEqual(report["unmatched_question_count"], 1)
        result = export_scope(self.project_path, self.root / "runtime", "scope-1", include_pending=True)
        row = json.loads(Path(result["output_path"], "json", "dataset.jsonl").read_text(encoding="utf-8").strip())
        self.assertEqual(row["answer_status"], "pending_generation")
        self.assertEqual(row["answers"], [])

    def test_rebuild_replaces_stale_files(self) -> None:
        first = export_scope(self.project_path, self.root / "runtime", "scope-1")
        stale = Path(first["output_path"]) / "stale.txt"
        stale.write_text("old", encoding="utf-8")
        second = export_scope(self.project_path, self.root / "runtime", "scope-1")
        self.assertEqual(first["output_path"], second["output_path"])
        self.assertFalse(stale.exists())

    def test_images_and_json_use_separate_folders(self) -> None:
        result = export_scope(self.project_path, self.root / "runtime", "scope-1")
        output = Path(result["output_path"])
        self.assertTrue((output / "images" / "questions").is_dir())
        self.assertTrue((output / "images" / "answers").is_dir())
        self.assertTrue((output / "json" / "items").is_dir())
        self.assertTrue((output / "json" / "manifest.json").is_file())
        self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
