"""员工自检（本地测试版）：OCR / LaTeX 乱码 + 交付字段缺项。

设计约束：

* **规则单点**：逐题调用 ``delivery_workspace.k12_field_issues``，与生成时
  ``_k12_validation`` 拒收用的是同一个函数。界面提示和生成闸不会各说各话。
* **轻量**：不读页面图片、不裁图、不碰身份库、不写文件、不看审核状态。
  标注时每次字段变更都要跑，不能像 ``analyze_delivery`` 那样去解码扫描图。
* **只管两类**：乱码（garbled）与缺项（missing_field / invalid_vocabulary）。
  「题目页尚未确认」「尚未匹配答案」属于流程状态，不在这里报。
  题目还没匹配答案时跳过答案缺项检查，避免把没做完的匹配误判成字段没填。
* **全部阻断**：任何 issue 都算 blocking，桥服务据此拦住生成与提交审核。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from review_tools import manual_annotation_data as data
from review_tools import delivery_workspace as dw

BOOK_FIELD_KEYS = ("language", "stage", "grade", "subject")
SELF_CHECK_SCHEMA_VERSION = 2
_SUMMARY_LIMIT = 8

# 只给前端表格做列名展示；判定口径仍完全来自 delivery_workspace.k12_field_issues。
FIELD_LABELS = {
    "language": "语言", "stage": "学段", "grade": "年级", "subject": "学科",
    "question_type": "题型", "difficulty": "难度", "knowledge_points": "知识点",
    "question": "题目文本", "answer": "标准答案", "options": "选项",
}


def _book_issues(project: dict) -> list[dict]:
    """书级字段（教材信息对话框填的那四项）只报一次，不逐题重复。"""
    meta = {key: dw._clean_text(project.get(key)) for key in BOOK_FIELD_KEYS}
    # 题级字段给合法占位，保证这次调用只暴露书级问题。
    meta.update({"question_type": "填空", "difficulty": "基础",
                 "knowledge_points": ["__self_check_placeholder__"]})
    body = {"question": "__self_check_placeholder__",
            "answer": "__self_check_placeholder__", "options": []}
    issues = [dict(item, reason=item["message"],
                   field_label=FIELD_LABELS.get(item["field"], item["field"]),
                   scope="book", page_index=None, page_number=None,
                   question_number=None, record_id=None)
              for item in dw.k12_field_issues(meta, body)
              if item["field"] in BOOK_FIELD_KEYS]
    return issues


def _answer_records(project: dict) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """答案记录索引 + 续框链（与交付的 prefer_text 口径一致）。"""
    by_id: dict[str, dict] = {}
    children: dict[str, list[dict]] = {}
    for _index, _page, record in data.all_answers(project):
        if not isinstance(record, dict) or not record.get("id"):
            continue
        record_id = str(record["id"])
        by_id[record_id] = record
        parent = str(record.get("continuation_of", "") or "")
        if parent:
            children.setdefault(parent, []).append(record)
    return by_id, children


def _answer_text(project: dict, question_id: str, by_id: dict[str, dict],
                 children: dict[str, list[dict]]) -> str | None:
    """返回该题当前匹配答案的合并原文；没匹配到就返回 None（不判缺项）。"""
    chain: list[dict] = []
    for answer_id in data.paired_answer_ids(project, question_id):
        record = by_id.get(str(answer_id))
        if not record:
            continue
        chain.append(record)
        chain.extend(children.get(str(record.get("id", "")), []))
    if not chain:
        return None
    bodies, analyses = [], []
    for record in chain:
        prefer_text = bool(children.get(str(record.get("id", ""))))
        bodies.append(dw._answer_body_text(record, prefer_text))
        analyses.append(dw._clean_text(record.get("analysis")))
    return dw._merged_answer_text(bodies, analyses)


def check_project(project: dict) -> dict[str, Any]:
    """只读自检，返回结构化 issues。任何 issue 都会阻断生成/提交。"""
    if not isinstance(project, dict):
        raise ValueError("自检需要工程字典")
    issues: list[dict] = list(_book_issues(project))
    by_id, children = _answer_records(project)
    seen: set[tuple] = set()

    for page_index, _page, question in data.all_questions(project):
        if not isinstance(question, dict):
            continue
        if str(question.get("region_role", "question")) != "question":
            continue
        question_id = str(question.get("id", ""))
        ocr_question, inferred_options = dw._split_ocr_choices(question.get("text", ""))
        options = dw._string_list(question.get("options", [])) or inferred_options
        raw_type = dw._clean_text(question.get("question_type", ""))
        effective_type = "选择题" if inferred_options else raw_type
        final_type = dw.K12_QUESTION_TYPES.get(effective_type, effective_type)
        merged_answer = _answer_text(project, question_id, by_id, children)

        meta = {key: dw._clean_text(project.get(key)) for key in BOOK_FIELD_KEYS}
        meta.update({
            "question_type": final_type,
            "difficulty": dw._clean_text(question.get("difficulty", "")),
            "knowledge_points": dw._string_list(question.get("knowledge_points", [])),
        })
        body = {
            "question": dw._k12_text(ocr_question),
            "answer": dw._k12_text(merged_answer) if merged_answer is not None else "",
            "options": ([dw._k12_text(value) for value in options]
                        if final_type == "选择" else []),
        }
        raw_texts = {"question": ocr_question,
                     "answer": merged_answer or "", "options": options}
        skip = set(BOOK_FIELD_KEYS)
        if merged_answer is None:
            skip.add("answer")

        location = {"scope": "question", "page_index": int(page_index),
                    "page_number": int(page_index) + 1,
                    "question_number": data.display_number(question),
                    "record_id": question_id or None}
        for item in dw.k12_field_issues(meta, body, raw_texts=raw_texts,
                                        skip_fields=skip):
            key = (location["page_index"], question_id,
                   item["code"], item["field"], item["message"])
            if key in seen:
                continue
            seen.add(key)
            issues.append(dict(item, reason=item["message"],
                               field_label=FIELD_LABELS.get(item["field"], item["field"]),
                               **location))

    counts = {"garbled": 0, "missing_field": 0, "invalid_vocabulary": 0}
    for item in issues:
        counts[item["code"]] = counts.get(item["code"], 0) + 1
    counts["total"] = len(issues)
    blocking = bool(issues)

    # 「缺什么」页照审核网站的结构：整本教材字段单独列，逐题缺项按原因汇总，
    # 点某一行就能筛出缺这一项的题。乱码有独立明细页，不混进缺项汇总。
    book_errors = [item["message"] for item in issues
                   if item.get("scope") == "book" and item["code"] != "garbled"]
    reason_counts: dict[str, int] = {}
    for item in issues:
        if item.get("scope") == "book" or item["code"] == "garbled":
            continue
        reason_counts[item["message"]] = reason_counts.get(item["message"], 0) + 1
    reason_summary = [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items(),
                                    key=lambda pair: (-pair[1], pair[0]))
    ]
    return {
        "schema_version": SELF_CHECK_SCHEMA_VERSION,
        "ok": not blocking,
        "blocking": blocking,
        "counts": counts,
        "issues": issues,
        "book_errors": book_errors,
        "reason_summary": reason_summary,
        "summary": format_summary(issues),
    }


def check_project_file(project_path: str | Path) -> dict[str, Any]:
    """便捷入口：从工程文件加载后自检（不落盘）。"""
    return check_project(data.load_project(Path(project_path)))


def format_summary(issues: list[dict], limit: int = _SUMMARY_LIMIT) -> str:
    """给桥服务抛错/前端弹窗用的一句话摘要。"""
    if not issues:
        return "自检通过"
    lines = []
    for item in issues[:limit]:
        if item.get("scope") == "book":
            prefix = "教材信息"
        else:
            prefix = f"第{item.get('page_number')}页 {item.get('question_number') or ''}".strip()
        lines.append(f"{prefix}：{item['message']}")
    rest = len(issues) - len(lines)
    if rest > 0:
        lines.append(f"……另有 {rest} 条")
    return f"自检未通过（{len(issues)} 条）：" + "；".join(lines)
