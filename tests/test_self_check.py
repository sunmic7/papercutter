"""员工自检（本地测试版）：OCR/LaTeX 乱码 + 交付字段缺项 + 硬闸护栏。

自检规则与生成闸共用 ``delivery_workspace.k12_field_issues``，这里的用例
同时锁住「界面看到的问题 = 生成会拒收的问题」，防止两套规则各说各话。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from review_tools import delivery_workspace as dw
from review_tools import self_check

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "desktop_annotation" / "PaperFlow-Annotation" / "resources" / "app" / "js"
BRIDGE_PY = ROOT / "desktop_annotation" / "paperflow_annotation_api.py"


def valid_meta() -> dict:
    return {
        "language": "zh", "stage": "小学", "grade": "六年级", "subject": "数学",
        "question_type": "填空", "difficulty": "中等", "knowledge_points": ["分数"],
    }


def valid_body() -> dict:
    return {"question": "求 1⁄2 的值", "answer": "1⁄2", "options": []}


class LatexSyntaxTests(unittest.TestCase):
    def test_legitimate_formulas_are_not_flagged(self):
        for text in (r"$4\frac{1}{2}$", r"$\alpha+\beta$",
                     r"$\begin{aligned}x\end{aligned}$",
                     r"$\pi r^{2}h$", r"$\sqrt{3}$"):
            with self.subTest(text=text):
                self.assertEqual(dw.latex_syntax_errors(text), [])

    def test_plain_chinese_text_is_never_flagged(self):
        self.assertEqual(dw.latex_syntax_errors("普通中文题干，没有公式"), [])
        self.assertEqual(dw.latex_syntax_errors(""), [])

    def test_ocr_misspelled_commands_are_flagged(self):
        for text, command in ((r"$\frc{1}{2}$", "frc"),
                              (r"$\timse 3$", "timse"),
                              (r"$\sqr{3}$", "sqr")):
            with self.subTest(text=text):
                errors = dw.latex_syntax_errors(text)
                self.assertTrue(any(command in item for item in errors), errors)

    def test_unclosed_delimiter_and_braces(self):
        errors = dw.latex_syntax_errors(r"$4\frac{1}{2")
        self.assertIn("公式定界符 $ 未闭合", errors)
        self.assertIn("公式花括号不配对", errors)

    def test_command_without_argument(self):
        self.assertIn("公式命令缺少参数", dw.latex_syntax_errors(r"$\frac$"))

    def test_control_and_replacement_characters(self):
        self.assertIn("含乱码字符", dw.latex_syntax_errors("$\ufffd$"))
        self.assertIn("含控制字符", dw.latex_syntax_errors("$\x01$"))

    def test_cjk_directly_after_backslash(self):
        self.assertIn("公式命令后混入中文字符", dw.latex_syntax_errors(r"$\中$"))


class FieldIssueTests(unittest.TestCase):
    def test_valid_sample_has_no_errors(self):
        self.assertEqual(dw.k12_field_errors(valid_meta(), valid_body()), [])

    def test_missing_and_invalid_fields(self):
        meta = valid_meta()
        meta.update({"language": "", "difficulty": "", "question_type": "乱填",
                     "knowledge_points": []})
        errors = dw.k12_field_errors(meta, {"question": "", "answer": "", "options": []})
        for expected in ("缺少语言", "缺少难度", "题型不在规范词表内：乱填",
                         "缺少知识点", "缺少题目文本", "缺少标准答案"):
            self.assertIn(expected, errors)

    def test_raw_latex_is_checked_even_after_cleaning(self):
        raw = r"求 $\frc{1}{2}$ 的值"
        cleaned = dw._k12_text(raw)
        self.assertNotIn(r"\frc", cleaned)  # 清洗会把反斜杠吃掉，所以必须看原文
        errors = dw.k12_field_errors(
            valid_meta(), {"question": cleaned, "answer": "1", "options": []},
            raw_texts={"question": raw, "answer": "1", "options": []})
        self.assertTrue(any("疑似 OCR 拼错" in item for item in errors), errors)

    def test_skip_fields_controls_scope(self):
        issues = dw.k12_field_issues(
            {"language": "", "stage": "小学", "grade": "六年级", "subject": "数学",
             "question_type": "填空", "difficulty": "", "knowledge_points": []},
            {"question": "题干", "answer": "", "options": []},
            skip_fields={"language", "difficulty", "knowledge_points", "answer"})
        self.assertEqual(issues, [])

    def test_generation_gate_uses_the_same_field_rules(self):
        sample = {"metainfo": valid_meta(), "data": valid_body()}
        errors, _hints = dw._k12_validation(sample, [])
        self.assertEqual(errors, dw.k12_field_errors(valid_meta(), valid_body()))
        raw = r"$\frc{1}{2}$"
        sample["data"]["question"] = dw._k12_text(raw)
        errors, _ = dw._k12_validation(sample, [], raw_texts={"question": raw})
        self.assertTrue(any("疑似 OCR 拼错" in item for item in errors), errors)


def synthetic_project(*, language: str = "zh", difficulty: str = "中等",
                      knowledge_points: list | None = None,
                      question_text: str = "求 1⁄2 的值",
                      matched_answer: bool = True) -> dict:
    if knowledge_points is None:
        knowledge_points = ["分数"]
    answer = {"id": "a1", "answer": "1⁄2", "text": "1⁄2", "status": "accepted"}
    return {
        "language": language, "stage": "小学", "grade": "六年级", "subject": "数学",
        "pages": [{"status": "accepted", "questions": [{
            "id": "q1", "text": question_text, "question_type": "填空题",
            "difficulty": difficulty, "knowledge_points": knowledge_points,
        }]}],
        "answer_pages": [{"status": "accepted", "answers": [answer]}] if matched_answer else [],
        "pairs": [{"question_id": "q1", "answer_ids": ["a1"], "status": "accepted"}]
        if matched_answer else [],
    }


class SelfCheckProjectTests(unittest.TestCase):
    def test_clean_project_passes(self):
        report = self_check.check_project(synthetic_project())
        self.assertTrue(report["ok"], report["summary"])
        self.assertFalse(report["blocking"])
        self.assertEqual(report["counts"]["total"], 0)
        self.assertEqual(report["issues"], [])

    def test_book_level_issue_reported_only_once(self):
        project = synthetic_project(language="")
        project["pages"].append({"status": "accepted", "questions": [{
            "id": "q2", "text": "另一题", "question_type": "填空题",
            "difficulty": "中等", "knowledge_points": ["分数"]}]})
        report = self_check.check_project(project)
        language_issues = [item for item in report["issues"]
                           if item["field"] == "language"]
        self.assertEqual(len(language_issues), 1)
        self.assertEqual(language_issues[0]["scope"], "book")

    def test_missing_fields_are_blocking_with_location(self):
        report = self_check.check_project(
            synthetic_project(difficulty="", knowledge_points=[]))
        self.assertTrue(report["blocking"])
        fields = {item["field"] for item in report["issues"]}
        self.assertIn("difficulty", fields)
        self.assertIn("knowledge_points", fields)
        for item in report["issues"]:
            if item["scope"] == "question":
                self.assertEqual(item["page_number"], 1)
                self.assertEqual(item["record_id"], "q1")
        self.assertIn("自检未通过", report["summary"])

    def test_latex_garble_is_blocking(self):
        report = self_check.check_project(
            synthetic_project(question_text=r"求 $\frc{1}{2}$ 的值"))
        self.assertTrue(report["blocking"])
        self.assertEqual(report["counts"]["garbled"], 1)
        self.assertIn("疑似 OCR 拼错", report["summary"])

    def test_unmatched_answer_is_not_reported_as_missing_field(self):
        """还没做答案匹配属于流程状态，不该被当成「字段没填」卡住提交。"""
        report = self_check.check_project(synthetic_project(matched_answer=False))
        messages = [item["message"] for item in report["issues"]]
        self.assertNotIn("缺少标准答案", messages)
        self.assertTrue(report["ok"], report["summary"])

    def test_check_project_file(self):
        project = synthetic_project(difficulty="")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.json"
            path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            report = self_check.check_project_file(path)
        self.assertTrue(report["blocking"])
        self.assertIn("缺少难度", report["summary"])


class BridgeGateGuardTests(unittest.TestCase):
    def test_bridge_exposes_self_check_route_and_hard_gate(self):
        source = BRIDGE_PY.read_text(encoding="utf-8")
        self.assertIn('"/api/project/self-check": _self_check_route', source)
        self.assertIn("def _ensure_self_check_clear", source)
        self.assertIn("已阻止生成和上传", source)
        # 自检链式加载 cv2/numpy，顶层导入会破坏工作台冷启动（快速启动护栏）。
        head = source[:source.index("def _run_self_check(")]
        self.assertNotIn("from review_tools import self_check", head)
        gate_body = source[source.index("def _run_self_check("):source.index("def _ensure_self_check_clear")]
        self.assertIn("from review_tools import self_check", gate_body)
        for entry in ("_export_project", "_sync_delivery", "_export_match_scope",
                      "_review_submit"):
            start = source.index(f"def {entry}(")
            end = source.find("\ndef ", start + 5)
            self.assertIn("_ensure_self_check_clear", source[start:end], entry)

    def test_front_end_shows_panel_and_blocks_submit(self):
        panel = (APP_JS / "self-check.js").read_text(encoding="utf-8")
        review = (APP_JS / "company-review.js").read_text(encoding="utf-8")
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        self.assertIn("/api/project/self-check", panel)
        self.assertIn("ensureClear()", panel)
        self.assertIn("server://", panel)
        # 服务器教材同样自检：走本地内联入口，不再显示「由审核员处理」。
        self.assertIn("{ project: inline }", panel)
        self.assertNotIn("由审核员处理", panel)
        self.assertNotIn("不进行本地员工自检", panel)
        self.assertNotIn("skipped: true", panel)
        self.assertIn('js/self-check.js', html)
        self.assertLess(html.index('js/main.js'), html.index('js/self-check.js'))
        submit = review[review.index("async action(mode)"):review.index("async refresh()")]
        self.assertIn("paperFlowSelfCheck.ensureClear()", submit)
        self.assertIn("saveCurrentPaperFlowProject()", submit)
        self.assertLess(submit.index("saveCurrentPaperFlowProject()"),
                        submit.index("paperFlowSelfCheck.ensureClear()"))

    def test_bridge_self_check_accepts_inline_server_project(self):
        source = BRIDGE_PY.read_text(encoding="utf-8")
        body = source[source.index("def _self_check_route("):]
        body = body[:body.index("\ndef ", 5)]
        self.assertIn('payload.get("project")', body)
        self.assertIn("isinstance(inline, dict)", body)
        # 内联工程不经过路径校验（不落盘、不转发），但仍走同一份规则。
        self.assertIn("_run_self_check(project)", body)
        self.assertLess(body.index("if isinstance(inline, dict):"),
                        body.index("_load_project(payload)"))
        # 关键：check_project 的 ok（自检通过与否）不能覆盖响应的 ok（请求成功）。
        # 否则前端会把「自检未通过」当请求失败抛掉，面板永远看不到问题清单。
        self.assertIn('report.pop("ok", None)', body)
        self.assertIn('"self_check_ok"', body)
        self.assertIn('return {"ok": True, **report}', body)


class DetailReportTests(unittest.TestCase):
    """详细面板的数据契约：缺项汇总照审核网站，乱码要能指出原文位置。"""

    def test_latex_details_and_errors_share_one_rule_source(self):
        for text in (r"$4\frac{1}{2}$", r"$\frc{1}{2}$", r"$4\frac{1}{2",
                     "$\ufffd$", r"$\frac$"):
            with self.subTest(text=text):
                self.assertEqual(dw.latex_syntax_errors(text),
                                 [item["message"] for item in dw.latex_syntax_details(text)])

    def test_garbled_issue_carries_raw_excerpt_and_offset(self):
        meta = valid_meta()
        raw = r"求 $\frc{1}{2}$ 的值"
        issues = dw.k12_field_issues(
            meta, {"question": dw._k12_text(raw), "answer": "1", "options": []},
            raw_texts={"question": raw, "answer": "1", "options": []})
        garbled = [item for item in issues if item["code"] == "garbled"]
        self.assertEqual(len(garbled), 1)
        item = garbled[0]
        self.assertIn(r"\frc", item["raw_excerpt"])
        self.assertEqual(item["raw_offset"], raw.index(r"\frc"))
        self.assertEqual(item["raw_excerpt"][item["excerpt_offset"]], "\\")
        self.assertEqual(item["raw_length"], len(raw))

    def test_replacement_char_is_located_even_when_cleaning_removes_it(self):
        raw = "求 $\ufffd$ 的值"
        issues = dw.k12_field_issues(
            valid_meta(), {"question": dw._k12_text(raw), "answer": "1", "options": []},
            raw_texts={"question": raw, "answer": "1", "options": []})
        garbled = [item for item in issues if item["code"] == "garbled"]
        self.assertEqual([item["message"] for item in garbled], ["题目文本含乱码字符"])
        self.assertEqual(garbled[0]["raw_offset"], raw.index("\ufffd"))

    def test_report_has_book_errors_and_reason_summary(self):
        project = synthetic_project(language="", difficulty="", knowledge_points=[])
        project["pages"].append({"status": "accepted", "questions": [{
            "id": "q2", "text": "另一题", "question_type": "填空题",
            "difficulty": "", "knowledge_points": []}]})
        report = self_check.check_project(project)
        self.assertEqual(report["schema_version"], 2)
        self.assertIn("缺少语言", report["book_errors"])
        self.assertNotIn("缺少难度", report["book_errors"])
        reasons = {row["reason"]: row["count"] for row in report["reason_summary"]}
        self.assertEqual(reasons.get("缺少难度"), 2)
        self.assertEqual(reasons.get("缺少知识点"), 2)
        for row in report["reason_summary"]:
            self.assertNotIn("疑似 OCR 拼错", row["reason"])
        for item in report["issues"]:
            self.assertIn("reason", item)
            self.assertIn("field_label", item)

    def test_clean_report_has_empty_summaries(self):
        report = self_check.check_project(synthetic_project())
        self.assertEqual(report["book_errors"], [])
        self.assertEqual(report["reason_summary"], [])


class ToolbarReportGuardTests(unittest.TestCase):
    def test_toolbar_button_opens_structured_report(self):
        panel = (APP_JS / "self-check.js").read_text(encoding="utf-8")
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="openSelfCheck"', html)
        self.assertIn("class=\"pfsc-count\"", html)
        self.assertIn("getElementById('openSelfCheck')", panel)
        self.assertIn("showModal()", panel)
        # 三块结构：缺什么（审核网站同款汇总表）、乱码明细（原文定位）、逐题明细。
        for marker in ("renderMissing()", "renderGarbled()", "renderItems()",
                       "data-pfsc-reason", "data-pfsc-goto", "reason_summary",
                       "raw_excerpt", "excerpt_offset", "book_errors"):
            self.assertIn(marker, panel, marker)
        self.assertLess(html.index('id="openSelfCheck"'), html.index('id="submitPaperFlowReview"'))


if __name__ == "__main__":
    unittest.main()
