from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import manual_annotation_data as data
from review_tools.export_ocr_delivery import export_ocr_delivery
from review_tools.delivery_workspace import sync_delivery_workspace


class OCRDeliveryExportTests(unittest.TestCase):
    @staticmethod
    def _complete_k12_record(number: str, text: str, y_offset: int = 0) -> dict:
        return data.new_record(
            [0, y_offset, 300, y_offset + 260],
            number=number,
            major_number="一",
            text=text,
            question_type="填空题",
            difficulty="中等",
            knowledge_points=["测试"],
            illustrations=[{
                "bbox": [20, y_offset + 20, 220, y_offset + 220],
                "image_kind": "question_image",
            }],
        )

    @staticmethod
    def _complete_answer(number: str, y_offset: int = 0) -> dict:
        answer = data.new_record(
            [0, y_offset, 300, y_offset + 260],
            answer=True,
            number=number,
            major_number="一",
            analysis="详细解析",
        )
        answer["answer"] = "标准答案"
        return answer

    def test_k12_jsonl_is_ordered_and_contains_only_meaningful_question_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((500, 500, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (220, 220), (0, 0, 0), 2)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            project.update(language="zh", stage="小学", grade="六年级", subject="数学")
            common = {
                "difficulty": "中等", "knowledge_points": ["几何"], "question_type": "选择题",
                "options": ["A. 1", "B. 2"],
                "illustrations": [{"bbox": [20, 20, 220, 220], "image_kind": "question_image"}],
            }
            later = data.new_record([0, 250, 500, 480], major_number="二", text="第二题", **common)
            earlier = data.new_record([0, 0, 500, 240], major_number="一", text="第一题", **common)
            answer_later = data.new_record([0, 250, 500, 480], answer=True, major_number="二", analysis="第二题解析")
            answer_later["answer"] = "B"
            answer_earlier = data.new_record([0, 0, 500, 240], answer=True, major_number="一", analysis="第一题解析")
            answer_earlier["answer"] = "A"
            project["pages"][0].update(status="accepted", questions=[later, earlier])
            project["answer_pages"][0].update(status="accepted", answers=[answer_later, answer_earlier])
            data.set_pair(project, later["id"], [answer_later["id"]])
            data.set_pair(project, earlier["id"], [answer_earlier["id"]])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            result = sync_delivery_workspace(project_path, root / "runtime", materialize_all=True)
            delivery = Path(result["output_path"])
            lines = [json.loads(line) for line in (delivery / "k12_dataset" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["data"]["question"] for item in lines], ["第一题", "第二题"])
            self.assertEqual(lines[0]["id"], "0000001")
            self.assertEqual(lines[0]["data"]["question_images"], ["images/0000001_img_01.jpg"])
            # 答案与解析合成一段；analysis 字段保留但不再重复一份。
            self.assertEqual(lines[0]["data"]["answer"], "A\n第一题解析")
            self.assertEqual(lines[0]["data"]["analysis"], "")
            self.assertTrue((delivery / "k12_dataset" / "images" / "0000001_img_01.jpg").is_file())
            self.assertTrue((delivery / "k12_dataset" / "pages" / "P0001" / "dataset.jsonl").is_file())
            self.assertFalse((delivery / "k12_dataset" / "pages" / "P0001" / "images" / "answer_1.jpg").exists())
            self.assertEqual(result["k12"]["eligible_count"], 2)

    def test_k12_jsonl_exports_final_unicode_math_not_latex_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((300, 300, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            project.update(language="zh", stage="小学", grade="六年级", subject="数学")
            question = data.new_record(
                [0, 0, 300, 260], major_number="一", text="求 $\\frac{1}{x}$ 与 $\\sqrt{y}$",
                question_type="填空题", difficulty="基础", knowledge_points=["分式"],
                illustrations=[{"bbox": [20, 20, 220, 220], "image_kind": "question_image"}],
            )
            answer = data.new_record([0, 0, 300, 260], answer=True, major_number="一", analysis="$x^2$")
            answer["answer"] = "$\\frac{1}{2}$"
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            result = sync_delivery_workspace(project_path, root / "runtime", materialize_all=True)
            line = (Path(result["output_path"]) / "k12_dataset" / "dataset.jsonl").read_text(encoding="utf-8").strip()
            payload = json.loads(line)
            self.assertEqual(payload["data"]["question"], "求 1⁄x 与 √(y)")
            self.assertEqual(payload["data"]["answer"], "1⁄2\nx²")
            self.assertEqual(payload["data"]["analysis"], "")
            self.assertNotIn("\\frac", line)
            self.assertNotIn("$", line)
            # Per-frame delivery JSON must use the same current editable result
            # as K12 JSONL, never the local LaTeX source kept for rendering.
            document_path = next((Path(result["output_path"]) / "json" / "questions" / "P0001").glob("*.json"))
            document = json.loads(document_path.read_text(encoding="utf-8"))
            self.assertEqual(document["text"], "求 1⁄x 与 √(y)")
            self.assertNotIn("\\frac", document["text"])

    def test_k12_derives_choices_from_unchanged_ocr_text_only_at_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((300, 300, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            project.update(language="zh", stage="小学", grade="六年级", subject="数学")
            question = data.new_record(
                [0, 0, 300, 260], major_number="一",
                text="下列说法正确的是（）。\nA. 第一项\nB．第二项\nC、第三项\nD: 第四项",
                question_type="填空题", difficulty="基础", knowledge_points=["测试"],
                illustrations=[{"bbox": [20, 20, 220, 220], "image_kind": "question_image"}],
            )
            answer = data.new_record([0, 0, 300, 260], answer=True, major_number="一", analysis="详细解析")
            answer["answer"] = "A"
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            result = sync_delivery_workspace(project_path, root / "runtime", materialize_all=True)
            payload = json.loads((Path(result["output_path"]) / "k12_dataset" / "dataset.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(payload["metainfo"]["question_type"], "选择")
            self.assertEqual(payload["data"]["question"], "下列说法正确的是（）。")
            self.assertEqual(payload["data"]["options"], ["A. 第一项", "B. 第二项", "C. 第三项", "D. 第四项"])

    def test_k12_rejects_incomplete_samples_without_leaving_formal_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root = root / "questions"
            question_root.mkdir()
            image = np.full((300, 300, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            project = data.create_project(str(question_root))
            question = data.new_record(
                [0, 0, 300, 300], major_number="一", text="缺少答案和元数据",
                illustrations=[{"bbox": [20, 20, 220, 220], "image_kind": "question_image"}],
            )
            project["pages"][0].update(status="accepted", questions=[question])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            result = sync_delivery_workspace(project_path, root / "runtime", materialize_all=True)
            delivery = Path(result["output_path"]) / "k12_dataset"
            self.assertEqual((delivery / "dataset.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual(list((delivery / "pages").glob("P*/images/*.jpg")), [])
            report = json.loads((delivery / "quality_report.json").read_text(encoding="utf-8"))
            self.assertIn("缺少标准答案", report["items"][0]["errors"])
            self.assertIn("题目尚未匹配答案", report["items"][0]["errors"])

    def test_delivery_ids_images_json_and_relations_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((80, 120, 3), 255, dtype=np.uint8)
            cv2.putText(image, "1 TEST", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))

            project = data.create_project(str(question_root), str(answer_root))
            question = data.new_record([0, 0, 120, 80], number="1", major_number="一")
            answer = data.new_record([0, 0, 120, 80], answer=True, number="1", major_number="一")
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            calls = []
            def fake_ocr(path: str) -> dict:
                calls.append(path)
                return {"text": "识别文字", "confidence": 0.98, "lines": [], "engine": "test"}

            first = export_ocr_delivery(str(project_path), str(root / "out1"), ocr_runner=fake_ocr)
            saved = data.load_project(project_path)
            question_id = saved["pages"][0]["questions"][0]["delivery_id"]
            answer_id = saved["answer_pages"][0]["answers"][0]["delivery_id"]
            self.assertRegex(question_id, r"^[A-Za-z0-9]+$")
            self.assertRegex(answer_id, r"^[A-Za-z0-9]+$")
            self.assertTrue((root / "out1" / "images" / "questions" / "P0001" / f"{question_id}.jpg").is_file())
            self.assertTrue((root / "out1" / "json" / "questions" / "P0001" / f"{question_id}.json").is_file())
            database = json.loads((root / "out1" / "database_import.json").read_text(encoding="utf-8"))
            self.assertEqual(database["questions"][0]["answer_ids"], [answer_id])
            self.assertEqual(database["answers"][0]["question_ids"], [question_id])
            self.assertEqual(database["questions"][0]["text"], "识别文字")
            self.assertEqual(first["failed_ocr_count"], 0)
            self.assertEqual(
                first["items"][0]["question_json"],
                f"json/questions/P0001/{question_id}.json",
            )

            first_relation_id = saved["pairs"][0]["delivery_id"]
            data.set_pair(saved, question["id"], [answer["id"]])
            data.save_project(saved, project_path)
            export_ocr_delivery(str(project_path), str(root / "out2"), ocr_runner=fake_ocr)
            saved_again = data.load_project(project_path)
            self.assertEqual(saved_again["pages"][0]["questions"][0]["delivery_id"], question_id)
            self.assertEqual(saved_again["answer_pages"][0]["answers"][0]["delivery_id"], answer_id)
            self.assertEqual(saved_again["pairs"][0]["delivery_id"], first_relation_id)
            self.assertEqual(len(calls), 2, "second export should reuse cached OCR for unchanged crops")

    def test_incremental_workspace_keeps_unanswered_questions_in_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((80, 120, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            question = data.new_record([0, 0, 120, 80], number="1", major_number="一", text="题目")
            answer = data.new_record([0, 0, 120, 80], answer=True, number="1", major_number="一", text="答案")
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="draft", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            first = sync_delivery_workspace(project_path, root / "runtime", changed_kind="question", changed_page_index=0)
            saved = data.load_project(project_path)
            qid = saved["pages"][0]["questions"][0]["delivery_id"]
            aid = saved["answer_pages"][0]["answers"][0]["delivery_id"]
            delivery = Path(first["output_path"])
            self.assertTrue((delivery / "json" / "questions" / "P0001" / f"{qid}.json").is_file())
            self.assertTrue((delivery / "images" / "questions" / "P0001" / f"{qid}.jpg").is_file())
            self.assertFalse((delivery / "images" / "answers" / "P0001" / f"{aid}.jpg").is_file())

            database = json.loads((delivery / "database_import.json").read_text(encoding="utf-8"))
            self.assertEqual(len(database["questions"]), 1)
            self.assertEqual(database["questions"][0]["answer_status"], "pending_generation")
            self.assertEqual(database["questions"][0]["answer_ids"], [])

            saved["answer_pages"][0]["status"] = "accepted"
            data.save_project(saved, project_path)
            second = sync_delivery_workspace(project_path, root / "runtime", changed_kind="answer", changed_page_index=0)
            database = json.loads((delivery / "database_import.json").read_text(encoding="utf-8"))
            self.assertEqual(second["ready_question_count"], 1)
            self.assertEqual(database["questions"][0]["id"], qid)
            self.assertEqual(database["questions"][0]["answer_ids"], [aid])
            self.assertEqual(database["answers"][0]["question_ids"], [qid])
            self.assertTrue((delivery / "images" / "answers" / "P0001" / f"{aid}.jpg").is_file())

            third = sync_delivery_workspace(project_path, root / "runtime", changed_kind="answer", changed_page_index=0)
            self.assertEqual(third["crop_updates"], 0)
            saved_again = data.load_project(project_path)
            self.assertEqual(saved_again["pages"][0]["questions"][0]["delivery_id"], qid)
            manifest = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))
            question_item = next(item for item in manifest["items"] if item["id"] == qid)
            self.assertEqual(question_item["image_file"], f"images/questions/P0001/{qid}.jpg")
            self.assertEqual(question_item["json_file"], f"json/questions/P0001/{qid}.json")

            data.set_pair(saved_again, question["id"], [])
            data.save_project(saved_again, project_path)
            sync_delivery_workspace(
                project_path, root / "runtime",
                changed_record_ids=[question["id"], answer["id"]],
            )
            answer_json = json.loads(
                (delivery / "json" / "answers" / "P0001" / f"{aid}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(answer_json["question_ids"], [])
            database = json.loads((delivery / "database_import.json").read_text(encoding="utf-8"))
            self.assertEqual(len(database["questions"]), 1)
            self.assertEqual(database["questions"][0]["answer_status"], "pending_generation")
            self.assertEqual(database["questions"][0]["answer_ids"], [])

    def test_incremental_confirmation_only_creates_the_selected_page_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "questions"
            images.mkdir()
            image = np.full((80, 120, 3), 255, dtype=np.uint8)
            for name in ("1.jpg", "2.jpg"):
                cv2.imencode(".jpg", image)[1].tofile(str(images / name))
            project = data.create_project(str(images))
            project["pages"][0]["status"] = "accepted"
            project_path = root / "project.json"
            data.save_project(project, project_path)
            result = sync_delivery_workspace(
                project_path, root / "runtime", changed_kind="question", changed_page_index=0
            )
            delivery = Path(result["output_path"])
            self.assertTrue((delivery / "images" / "questions" / "P0001").is_dir())
            self.assertTrue((delivery / "json" / "questions" / "P0001").is_dir())
            self.assertFalse((delivery / "images" / "questions" / "P0002").exists())
            self.assertFalse((delivery / "json" / "questions" / "P0002").exists())

    def test_incremental_pairing_materializes_every_referenced_k12_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((320, 320, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            project.update(language="zh", stage="小学", grade="六年级", subject="数学")
            question = self._complete_k12_record("1", "完整题干")
            answer = self._complete_answer("1")
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            first = sync_delivery_workspace(
                project_path, root / "runtime", changed_kind="question", changed_page_index=0,
            )
            delivery = Path(first["output_path"])
            self.assertEqual(first["k12"]["eligible_count"], 0)

            project = data.load_project(project_path)
            data.set_pair(project, question["id"], [answer["id"]])
            data.save_project(project, project_path)
            paired = sync_delivery_workspace(
                project_path, root / "runtime",
                changed_record_ids=[question["id"], answer["id"]],
            )

            dataset = delivery / "k12_dataset" / "dataset.jsonl"
            samples = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(paired["k12"]["eligible_count"], 1)
            self.assertEqual(len(samples), 1)
            for relative in samples[0]["data"]["question_images"] + samples[0]["data"]["option_images"]:
                self.assertTrue((delivery / "k12_dataset" / relative).is_file(), relative)

    def test_incremental_sync_does_not_delete_other_eligible_pages_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root, answer_root = root / "questions", root / "answers"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((320, 320, 3), 255, dtype=np.uint8)
            for name in ("1.jpg", "2.jpg"):
                cv2.imencode(".jpg", image)[1].tofile(str(question_root / name))
                cv2.imencode(".jpg", image)[1].tofile(str(answer_root / name))
            project = data.create_project(str(question_root), str(answer_root))
            project.update(language="zh", stage="小学", grade="六年级", subject="数学")
            first_question = self._complete_k12_record("1", "第一题")
            second_question = self._complete_k12_record("2", "第二题")
            first_answer = self._complete_answer("1")
            second_answer = self._complete_answer("2")
            project["pages"][0].update(status="accepted", questions=[first_question])
            project["answer_pages"][0].update(status="accepted", answers=[first_answer])
            project["pages"][1].update(status="draft", questions=[second_question])
            project["answer_pages"][1].update(status="draft", answers=[second_answer])
            data.set_pair(project, first_question["id"], [first_answer["id"]])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            first = sync_delivery_workspace(
                project_path, root / "runtime", changed_kind="question", changed_page_index=0,
            )
            delivery = Path(first["output_path"])
            first_image = delivery / "k12_dataset" / "images" / "0000001_img_01.jpg"
            self.assertTrue(first_image.is_file())

            project = data.load_project(project_path)
            project["pages"][1]["status"] = "accepted"
            project["answer_pages"][1]["status"] = "accepted"
            data.set_pair(project, second_question["id"], [second_answer["id"]])
            data.save_project(project, project_path)
            second = sync_delivery_workspace(
                project_path, root / "runtime", changed_kind="question", changed_page_index=1,
            )

            second_image = delivery / "k12_dataset" / "images" / "0000002_img_01.jpg"
            self.assertEqual(second["k12"]["eligible_count"], 2)
            self.assertTrue(first_image.is_file())
            self.assertTrue(second_image.is_file())

    def test_untrusted_manifest_paths_cannot_delete_outside_delivery_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root = root / "questions"
            question_root.mkdir()
            project = data.create_project(str(question_root))
            project_path = root / "project.json"
            data.save_project(project, project_path)
            saved = data.load_project(project_path)
            delivery = question_root / saved["delivery_book_id"]
            legitimate = delivery / "json" / "questions" / "P0001" / "QSTALE.json"
            legitimate.parent.mkdir(parents=True)
            legitimate.write_text("{}", encoding="utf-8")
            traversal_victim = root / "traversal-victim.json"
            absolute_victim = root / "absolute-victim.jpg"
            traversal_victim.write_text("keep", encoding="utf-8")
            absolute_victim.write_text("keep", encoding="utf-8")
            (delivery / "manifest.json").write_text(json.dumps({
                "items": [
                    {"id": "traversal", "json_file": "../../../traversal-victim.json"},
                    {"id": "absolute", "image_file": str(absolute_victim)},
                    {"id": "legitimate", "json_file": "json/questions/P0001/QSTALE.json"},
                ],
            }), encoding="utf-8")

            sync_delivery_workspace(project_path, root / "runtime")

            self.assertTrue(traversal_victim.is_file())
            self.assertTrue(absolute_victim.is_file())
            self.assertFalse(legitimate.exists())


if __name__ == "__main__":
    unittest.main()
