"""甲方 K12 交付契约闸：词表、PII、噪声、整页图、去重、跨批次身份。

依据 ``数据需求明细-供应商.xlsx`` / sheet「需求1」/ 第 3 行：
N3 备注给出 13 个核心字段与六个字段的枚举词表，H3 是 15 条数据要求。
这些用例锁的是"违规样本进不了交付"的行为，不是字段本身。
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import manual_annotation_data as data
from review_tools.delivery_workspace import (
    _DeliveryIdentity,
    _answer_body_text,
    _k12_text,
    _mask_pii,
    _merged_answer_text,
    _strip_noise,
    _strip_printed_minor_prefix,
    analyze_delivery,
    sync_delivery_workspace,
)


def build_project(root: Path, *, texts: list[str], difficulty: str = "中等",
                  question_type: str = "填空题", box: tuple[int, int, int, int] = (20, 20, 220, 220),
                  output_root: Path | None = None, subject: str = "数学",
                  illustrations: bool = True) -> Path:
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
            illustrations=([{"bbox": list(box), "image_kind": "question_image"}]
                           if illustrations else []),
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


def deliver(project_path: Path, runtime: Path, **kwargs) -> tuple[list[dict], Path]:
    result = sync_delivery_workspace(project_path, runtime, materialize_all=True, **kwargs)
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


class PrintedMinorPrefixTests(unittest.TestCase):
    """答案框常把印刷小题号一起框进切片，OCR 忠实读出"8. πr²h"。

    交付前要剥掉与该答案小题号一致的题号前缀；剥完只剩纯数字时不动，
    那更可能是答案本身的小数（小题 8 的答案 "8.5"）。
    """

    def test_strips_prefix_matching_minor_number(self):
        self.assertEqual(
            _strip_printed_minor_prefix("8. $\\pi r^{2}h$", 8), "$\\pi r^{2}h$")
        self.assertEqual(_strip_printed_minor_prefix("8、一个圆锥的体积", 8), "一个圆锥的体积")
        self.assertEqual(_strip_printed_minor_prefix("8．50.24", 8), "50.24")

    def test_two_dots_mean_number_then_value(self):
        # "7." 是题号，"6.0288" 是答案；整串只有两个点，不存在小数歧义。
        self.assertEqual(_strip_printed_minor_prefix("7.6.0288", 7), "6.0288")

    def test_bare_decimal_answer_is_kept(self):
        self.assertEqual(_strip_printed_minor_prefix("8.5", 8), "8.5")
        self.assertEqual(_strip_printed_minor_prefix("8.5 平方分米", 8), "8.5 平方分米")

    def test_prefix_of_other_questions_is_untouched(self):
        self.assertEqual(_strip_printed_minor_prefix("9. πr²h", 8), "9. πr²h")
        self.assertEqual(_strip_printed_minor_prefix("12.56", 8), "12.56")

    def test_answer_body_text_uses_record_minor_number(self):
        record = {"answer": "8. $\\pi r^{2}h$", "text": "", "minor_number": 8}
        self.assertEqual(_answer_body_text(record), "$\\pi r^{2}h$")
        record["minor_number"] = None
        self.assertEqual(_answer_body_text(record), "8. $\\pi r^{2}h$")

    def test_punctuated_prefix_from_real_case(self):
        # 真实案例 BADA11D3FD6：答案框 OCR 把答案行序号点读歪成 "、1. "。
        self.assertEqual(
            _strip_printed_minor_prefix("、1. 12.56    37.68    62.8", 1),
            "12.56    37.68    62.8")
        self.assertEqual(_strip_printed_minor_prefix("，、1. 12.56", 1), "12.56")


class StaleAnalysisFragmentTests(unittest.TestCase):
    """答案框重新 OCR 后 text 会刷新，但旧"答案/解析拆分"写进 analysis 的
    纯数字碎片会残留（如 "12.\\n56\\n68\\n37\\n62.\\n1."）。工作台不重刷
    旧字段，交付出口识别这种碎片并丢弃；真正的文字解析不受影响。
    """

    def test_numeric_fragment_of_the_answer_is_dropped(self):
        merged = _merged_answer_text(
            ["12.56 37.68 62.8"], ["12.\n56\n68\n37\n62.\n1."])
        self.assertEqual(merged, "12.56 37.68 62.8")

    def test_prose_analysis_is_kept_even_with_overlapping_digits(self):
        merged = _merged_answer_text(
            ["12.56"], ["先求底面周长 3.14×4=12.56，再求侧面积。"])
        self.assertIn("先求底面周长", merged)

    def test_numbers_absent_from_answer_are_kept(self):
        # 数字不在答案里时不能丢——那可能是另一来源的真实内容。
        merged = _merged_answer_text(["25"], ["3.14 100"])
        self.assertEqual(merged, "25\n3.14 100")

    def test_pure_punctuation_is_dropped(self):
        self.assertEqual(_merged_answer_text(["12.56"], ["。、．"]), "12.56")


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


class DeliveryDiagnosisTests(unittest.TestCase):
    """生成前体检：不进队列也能看到缺什么，而且绝不留下文件。"""

    def test_healthy_book_is_deliverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = analyze_delivery(build_project(root, texts=["求面积。"]), root / "runtime")
            self.assertTrue(report["deliverable"])
            self.assertEqual(report["eligible_count"], 1)
            self.assertEqual(report["reason_summary"], [])
            self.assertEqual(report["book"]["errors"], [])
            self.assertEqual(sorted(report["sample_preview"][0]),
                             ["data", "id", "metainfo", "source"])

    def test_missing_question_field_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"], difficulty="")
            report = analyze_delivery(path, root / "runtime")
            self.assertFalse(report["deliverable"])
            reasons = {row["reason"]: row for row in report["reason_summary"]}
            self.assertIn("缺少难度", reasons)
            self.assertEqual(reasons["缺少难度"]["count"], 1)
            self.assertTrue(reasons["缺少难度"]["questions"])

    def test_reason_examples_show_the_question_number_only(self):
        """示例列只放题号：页码是扫描页序，和书上印的页码对不上。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"], difficulty="")
            report = analyze_delivery(path, root / "runtime")
            example = report["reason_summary"][0]["questions"][0]
            self.assertEqual(example, report["items"][0]["question_number"])
            self.assertNotIn("页", example)

    def test_missing_book_meta_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"])
            project = data.load_project(path)
            project["grade"] = ""
            data.save_project(project, path, increment_revision=False)
            report = analyze_delivery(path, root / "runtime")
            self.assertIn("缺少年级", report["book"]["errors"])
            self.assertFalse(report["deliverable"])

    def test_diagnosis_writes_nothing_and_matches_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。", "求周长。"])
            runtime = root / "runtime"
            report = analyze_delivery(path, runtime)
            self.assertFalse((runtime / "delivery_packages").exists(),
                             "体检不得创建交付目录")
            samples, _delivery = deliver(path, runtime)
            self.assertEqual(report["eligible_count"], len(samples))

    def test_quality_report_carries_reason_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["求面积。"], question_type="火星题")
            _samples, delivery = deliver(path, root / "runtime")
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertTrue(report["reason_summary"])
            self.assertEqual(report["reason_summary"][0]["count"], 1)


class IdentityOverwriteTests(unittest.TestCase):
    """同一道题重出就顶掉旧的：一个题目只对应一条样本、一个 id。"""

    def test_same_question_updates_in_place_and_retires_old_content(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = _DeliveryIdentity(Path(directory) / "identity.sqlite3")
            try:
                first = identity.claim("B1#Q1", "B1", "hash-a")
                self.assertEqual(first["status"], "new")
                self.assertEqual(identity.claim("B1#Q1", "B1", "hash-a"),
                                 {"status": "kept", "id": first["id"]})
                changed = identity.claim("B1#Q1", "B1", "hash-b")
                self.assertEqual(changed["status"], "updated")
                self.assertEqual(changed["id"], first["id"])
                # 旧内容已经被顶掉，别的书再用同一段内容不算重复；
                # 新内容已经属于 B1，别的书撞上才算跨批次重复。
                self.assertEqual(identity.claim("B2#Q1", "B2", "hash-a")["status"], "new")
                self.assertEqual(identity.claim("B2#Q2", "B2", "hash-b")["status"], "conflict")
            finally:
                identity.close()

    def test_regenerating_after_editing_the_answer_keeps_one_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["同一道题"])
            rows, delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows), 1)
            # 原教材给了讲解就并进答案，不再单列一份。
            self.assertEqual(rows[0]["data"]["answer"], "42\n详细解析过程")
            self.assertEqual(rows[0]["data"]["analysis"], "")
            project = data.load_project(path)
            project["answer_pages"][0]["answers"][0]["answer"] = "43"
            data.save_project(project, path, increment_revision=False)
            rows2, _delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows2), 1, "改答案后不能变成两条样本")
            self.assertEqual(rows2[0]["id"], rows[0]["id"], "同一道题必须沿用同一个 id")
            self.assertEqual(rows2[0]["data"]["answer"], "43\n详细解析过程")
            database = next(delivery.parent.glob("delivery_identity.sqlite3"))
            connection = sqlite3.connect(str(database))
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0], 1)
            finally:
                connection.close()


class DeliveryScopeTests(unittest.TestCase):
    """进包口径：默认按甲方字段判定，审核只否决"退回/需精修"。"""

    def test_fields_are_enough_without_any_review(self):
        """没审过的页也能出包——抽检流程不能被全量审核卡住。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            rows, _delivery = deliver(path, root / "runtime", allowed_pages=set())
            self.assertEqual(len(rows), 1)

    def test_reviewed_only_mode_needs_both_lists_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            self.assertEqual(
                deliver(path, root / "runtime", allowed_pages=set(),
                        only_reviewed=True)[0], [])
            # 只放行题目页：答案还没复核，严格模式下这一题不能进包。
            self.assertEqual(
                deliver(path, root / "runtime",
                        allowed_pages={("question", 0)}, only_reviewed=True)[0], [])
            rows, _delivery = deliver(
                path, root / "runtime",
                allowed_pages={("question", 0), ("answer", 0)}, only_reviewed=True)
            self.assertEqual(len(rows), 1)

    def test_review_unapproved_page_is_counted_out_of_scope(self):
        """严格模式下没放行的页是"还没审"，不是"缺字段"。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            _rows, delivery = deliver(path, root / "runtime", allowed_pages=set(),
                                      only_reviewed=True)
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(report["total_questions"], 0)
            self.assertEqual(report["scope"]["skipped_unreviewed_questions"], 1)
            self.assertEqual(report["scope"]["skipped_unreviewed_pages"], 1)
            self.assertEqual(report["reason_summary"], [])

    def test_answer_page_not_approved_blocks_only_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            _rows, delivery = deliver(path, root / "runtime",
                                      allowed_pages={("question", 0)},
                                      only_reviewed=True)
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            reasons = {row["reason"] for row in report["reason_summary"]}
            self.assertIn("答案页尚未审核放行", reasons)

    def test_returned_page_never_reaches_the_package(self):
        """审核员明确退回的页：字段再齐也不收，且要说清原因。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            rows, delivery = deliver(
                path, root / "runtime",
                rejected_pages={("question", 0)})
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(rows, [])
            self.assertEqual(report["scope"]["blocked_by_review_questions"], 1)
            self.assertEqual(report["scope"]["blocked_by_review_pages"], 1)

    def test_answer_text_falls_back_to_ocr_when_answer_is_blank(self):
        """答案框只做了 OCR、没人工填 answer：不能判成缺少标准答案。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            project = json.loads(path.read_text(encoding="utf-8"))
            for page in project["answer_pages"]:
                for record in page["answers"]:
                    record["text"] = "12. 56 68 37 62."
                    record["answer"] = ""
            path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            rows, delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows), 1)
            self.assertIn("56", rows[0]["data"]["answer"])
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(report["reason_summary"], [])

    def test_missing_illustration_is_only_a_hint(self):
        """题图不是必填：question_images 字段存在且为空数组就合规。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"], illustrations=False)
            rows, delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data"]["question_images"], [])
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(report["reason_summary"], [])
            self.assertEqual(report["hint_summary"][0]["reason"], "本题没有题图")

    def test_answer_and_analysis_share_one_field(self):
        """答案和解析放一块：有讲解就接在答案后面，analysis 不再单独写一份。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            rows, _delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data"]["answer"], "42\n详细解析过程")
            self.assertIn("analysis", rows[0]["data"], "字段要保留，值可以为空")
            self.assertEqual(rows[0]["data"]["analysis"], "")

    def test_question_without_analysis_still_ships(self):
        """原教材没写解析：不拦，这一题照样进包。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            project = json.loads(path.read_text(encoding="utf-8"))
            for page in project["answer_pages"]:
                for record in page["answers"]:
                    record["analysis"] = ""
            path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            rows, delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data"]["answer"], "42")
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(report["reason_summary"], [])

    def test_answer_key_alone_is_enough_when_only_analysis_was_written(self):
        """答案框里只填了解析：答案字段拿得到内容，不判缺少标准答案。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_project(root, texts=["题一"])
            project = json.loads(path.read_text(encoding="utf-8"))
            for page in project["answer_pages"]:
                for record in page["answers"]:
                    record["answer"] = ""
                    record["text"] = ""
                    record["analysis"] = "先通分再相加，得 42"
            path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            rows, delivery = deliver(path, root / "runtime")
            self.assertEqual(len(rows), 1)
            self.assertIn("42", rows[0]["data"]["answer"])
            report = json.loads((delivery / "k12_dataset" / "quality_report.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual(report["reason_summary"], [])


if __name__ == "__main__":
    unittest.main()
