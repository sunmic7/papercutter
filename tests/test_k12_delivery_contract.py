"""甲方 K12 交付契约闸：词表、PII、噪声、整页图、去重、跨批次身份。

依据 ``数据需求明细-供应商.xlsx`` / sheet「需求1」/ 第 3 行：
N3 备注给出 13 个核心字段与六个字段的枚举词表，H3 是 15 条数据要求。
这些用例锁的是"违规样本进不了交付"的行为，不是字段本身。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import manual_annotation_data as data
from review_tools.delivery_workspace import (
    _k12_text,
    _mask_pii,
    _strip_noise,
    sync_delivery_workspace,
)


def build_project(root: Path, *, texts: list[str], difficulty: str = "中等",
                  question_type: str = "填空题", box: tuple[int, int, int, int] = (20, 20, 220, 220),
                  output_root: Path | None = None, subject: str = "数学") -> Path:
    """Create a complete, otherwise-valid project with one question per text."""
    question_root, answer_root = root / "questions", root / "answers"
    question_root.mkdir(parents=True, exist_ok=True)
    answer_root.mkdir(parents=True, exist_ok=True)
    image = np.full((500, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (220, 220), (0, 0, 0), 2)
    cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
    cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
    project = data.create_project(str(question_root), str(answer_root))
    project.update(language="zh", stage="小学", grade="六年级", subject=subject)
    if output_root is not None:
        project["delivery_output_root"] = str(output_root)
    questions, answers = [], []
    for index, text in enumerate(texts):
        y = index * 250
        record = data.new_record(
            [0, y, 300, y + 240], major_number=f"一{index + 1}", text=text,
            question_type=question_type, difficulty=difficulty, knowledge_points=["测试"],
            illustrations=[{"bbox": list(box), "image_kind": "question_image"}],
        )
        answer = data.new_record([0, y, 300, y + 240], answer=True,
                                 major_number=f"一{index + 1}", analysis="详细解析过程")
        answer["answer"] = "42"
        questions.append(record)
        answers.append(answer)
    project["pages"][0].update(status="accepted", questions=questions)
    project["answer_pages"][0].update(status="accepted", answers=answers)
    for question, answer in zip(questions, answers):
        data.set_pair(project, question["id"], [answer["id"]])
    path = root / "project.json"
    data.save_project(project, path)
    return path


def deliver(project_path: Path, runtime: Path) -> tuple[list[dict], Path]:
    result = sync_delivery_workspace(project_path, runtime, materialize_all=True)
    delivery = Path(result["output_path"])
    text = (delivery / "k12_dataset" / "dataset.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()], delivery


class TextPipelineTests(unittest.TestCase):
    def test_phone_email_id_and_social_accounts_become_lowercase_x(self):
        text = "联系 13800138000 或 teacher@example.com，身份证 110101199003071234，QQ：123456789"
        masked = _mask_pii(text)
        for secret in ("13800138000", "teacher@example.com", "110101199003071234", "123456789"):
            self.assertNotIn(secret, masked)
        self.assertIn("x", masked)

    def test_normal_math_text_is_not_masked(self):
        self.assertEqual(_mask_pii("当 x=3 时求 2x+1，答：7"), "当 x=3 时求 2x+1，答：7")

    def test_html_ads_urls_and_emoji_are_stripped(self):
        raw = '<p>求面积。</p>版权所有 侵权必究 http://example.com/a 扫码关注公众号 😀😀'
        cleaned = _strip_noise(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("版权所有", cleaned)
        self.assertNotIn("http://", cleaned)
        self.assertNotIn("😀", cleaned)
        self.assertIn("求面积。", cleaned)

    def test_k12_text_runs_math_then_noise_then_pii(self):
        self.assertEqual(_k12_text("$\\frac{1}{2}$ <b>请拨打 13800138000</b>"), "1⁄2 请拨打 x")


class DeliveryGateTests(unittest.TestCase):
    def test_off_list_difficulty_and_question_type_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for difficulty, question_type in (("3", "填空题"), ("中等", "火星题")):
                case_root = root / f"{difficulty}-{question_type}"
                case_root.mkdir()
                path = build_project(case_root, texts=["求面积。"],
                                     difficulty=difficulty, question_type=question_type)
                lines, delivery = deliver(path, case_root / "runtime")
                self.assertEqual(lines, [], f"{difficulty}/{question_type} 不应交付")
                report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                    .read_text(encoding="utf-8"))
                self.assertEqual(report["eligible_count"], 0)
                self.assertTrue(any("词表" in error for error in report["items"][0]["errors"]))

    def test_off_list_subject_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"], subject="火星学")
            lines, _ = deliver(path, root / "runtime")
            self.assertEqual(lines, [])

    def test_pii_never_reaches_the_delivered_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["请拨打 13800138000 或邮件 teacher@example.com 咨询。"])
            lines, delivery = deliver(path, root / "runtime")
            self.assertEqual(len(lines), 1)
            raw = (delivery / "k12_dataset" / "dataset.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("13800138000", raw)
            self.assertNotIn("teacher@example.com", raw)

    def test_full_page_screenshot_is_not_delivered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"], box=(0, 0, 500, 500))
            lines, delivery = deliver(path, root / "runtime")
            self.assertEqual(lines, [])
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            joined = "；".join(report["items"][0]["errors"])
            self.assertIn("整页截图", joined)

    def test_duplicate_questions_collapse_to_one_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"], )
            project = data.load_project(path)
            clone = json.loads(json.dumps(project["pages"][0]["questions"][0]))
            clone["id"] = "clone-question"
            clone["delivery_id"] = "clone-question-delivery"
            clone["bbox"] = [0, 260, 300, 500]
            answer_clone = json.loads(json.dumps(project["answer_pages"][0]["answers"][0]))
            answer_clone["id"] = "clone-answer"
            answer_clone["delivery_id"] = "clone-answer-delivery"
            project["pages"][0]["questions"].append(clone)
            project["answer_pages"][0]["answers"].append(answer_clone)
            data.set_pair(project, clone["id"], [answer_clone["id"]])
            data.save_project(project, path)
            lines, _ = deliver(path, root / "runtime")
            self.assertEqual(len(lines), 1)

    def test_garble_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面\ufffd积。"])
            lines, delivery = deliver(path, root / "runtime")
            self.assertEqual(lines, [])
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertTrue(any("乱码" in error for error in report["items"][0]["errors"]))

    def test_coverage_report_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"])
            _lines, delivery = deliver(path, root / "runtime")
            coverage = json.loads((delivery / "k12_dataset" / "coverage_report.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(coverage["samples"], 1)
            self.assertEqual(coverage["distribution"]["subject"], {"数学": 1})


class CrossBatchIdentityTests(unittest.TestCase):
    def test_two_batches_get_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "packages"
            first, _ = deliver(build_project(root / "a", texts=["求面积。"], output_root=shared),
                               root / "runtime-a")
            second, _ = deliver(build_project(root / "b", texts=["计算各式的值。"], output_root=shared),
                                root / "runtime-b")
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertNotEqual(first[0]["id"], second[0]["id"])

    def test_same_content_in_another_book_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "packages"
            marker = "已知 x=3，求 x²+1。"
            first, _ = deliver(build_project(root / "a", texts=[marker], output_root=shared),
                               root / "runtime-a")
            second, delivery = deliver(build_project(root / "b", texts=[marker], output_root=shared),
                                       root / "runtime-b")
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertTrue(any("内容重复" in error for error in report["items"][0]["errors"]))

    def test_regenerating_the_same_book_keeps_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"])
            first, _ = deliver(path, root / "runtime")
            second, _ = deliver(path, root / "runtime")
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["id"], second[0]["id"])


if __name__ == "__main__":
    unittest.main()
