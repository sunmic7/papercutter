"""Build one self-contained question/answer package for a matching scope."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from review_tools import manual_annotation_data as data
from review_tools.delivery_workspace import delivery_root
from review_tools.export_manual_annotations import _crop, _question_crop, _write_image


def _scope_id(record: dict, page: dict) -> str:
    return str(record.get("match_scope_id") or page.get("match_scope_id") or "").strip()


def _scope(project: dict, scope_id: str) -> dict:
    scope = next((item for item in project.get("match_scopes", [])
                  if str(item.get("id", "")) == scope_id), None)
    if scope is None:
        raise ValueError("所选匹配分组不存在。")
    return scope


def _indexes(project: dict) -> tuple[dict, dict, dict]:
    questions = {
        str(record.get("id", "")): (page_index, page, record)
        for page_index, page, record in data.all_questions(project)
        if record.get("region_role", "question") == "question"
    }
    answers = {
        str(record.get("id", "")): (page_index, page, record)
        for page_index, page, record in data.all_answers(project)
        if record.get("region_role", "question") == "question"
    }
    pairs = {str(pair.get("question_id", "")): pair for pair in project.get("pairs", [])}
    return questions, answers, pairs


def preflight_scope(project: dict, scope_id: str) -> dict:
    scope = _scope(project, scope_id)
    questions, answers, pairs = _indexes(project)
    selected = [row for row in questions.values() if _scope_id(row[2], row[1]) == scope_id]
    confirmed = [row for row in selected if str(row[1].get("status", "")) == "accepted"]
    matched = []
    unmatched = []
    for row in confirmed:
        question = row[2]
        pair = pairs.get(str(question.get("id", "")))
        answer_ids = [str(value) for value in (pair or {}).get("answer_ids", [])]
        answer_rows = [answers.get(answer_id) for answer_id in answer_ids]
        ready = bool(
            pair and pair.get("status") == "accepted" and answer_ids
            and all(answer_rows)
            and all(str(item[1].get("status", "")) == "accepted" for item in answer_rows if item)
            and all(_scope_id(item[2], item[1]) == scope_id for item in answer_rows if item)
        )
        (matched if ready else unmatched).append(row)
    return {
        "scope": {"id": scope_id, "name": str(scope.get("name", scope_id))},
        "answer_mode": str(project.get("answer_mode", "textbook")),
        "question_count": len(selected),
        "confirmed_question_count": len(confirmed),
        "unconfirmed_question_count": len(selected) - len(confirmed),
        "matched_ready_count": len(matched),
        "unmatched_question_count": len(unmatched),
        "can_generate": bool(matched or unmatched),
    }


def _safe_scope_folder(scope_name: str) -> str:
    """Keep the staff-visible group name, replacing only invalid path characters."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(scope_name)).strip(" .")[:80]
    if not cleaned:
        return "未命名分组"
    if cleaned.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                            *(f"LPT{i}" for i in range(1, 10))}:
        cleaned += "_分组"
    return cleaned


def _remove_previous_scope_folders(root: Path, target: Path, scope_id: str) -> None:
    """Remove an older ID-named/renamed package only after the replacement succeeds."""
    for candidate in root.iterdir():
        if not candidate.is_dir() or candidate == target or ".tmp-" in candidate.name or ".old-" in candidate.name:
            continue
        manifest_path = candidate / "json" / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = candidate / "manifest.json"  # v1 layout
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str((manifest.get("scope") or {}).get("id", "")) == scope_id:
            shutil.rmtree(candidate)


def _record_payload(record: dict, page: dict, page_index: int, image_file: str) -> dict:
    return {
        "id": record.get("id"),
        "delivery_id": record.get("delivery_id"),
        "display_number": data.display_number(record),
        "number_path": data.number_path(record),
        "text": record.get("text", ""),
        "question_type": record.get("question_type", ""),
        "difficulty": record.get("difficulty", ""),
        "knowledge_points": record.get("knowledge_points", []),
        "source": {"page": page_index + 1, "image": page.get("image", ""), "bbox": record.get("bbox", [])},
        "image_file": image_file,
    }


def _answer_segments(project: dict, answer_page_index: int, answer_page: dict, answer: dict):
    yield answer_page_index, answer_page, answer
    for page_index, page, continuation in data.all_answers(project):
        if (continuation.get("region_role") == "continuation"
                and continuation.get("continuation_of") == answer.get("id")):
            yield page_index, page, continuation


def export_scope(project_path: str | Path, runtime_root: str | Path, scope_id: str) -> dict:
    project_path = Path(project_path).resolve()
    project = data.load_project(project_path)
    report = preflight_scope(project, scope_id)
    questions, answers, pairs = _indexes(project)
    scope = report["scope"]
    selected = [row for row in questions.values()
                if _scope_id(row[2], row[1]) == scope_id
                and str(row[1].get("status", "")) == "accepted"]

    root = delivery_root(project, runtime_root) / "groups"
    root.mkdir(parents=True, exist_ok=True)
    target = root / _safe_scope_folder(scope["name"])
    temporary = root / (target.name + ".tmp-" + uuid.uuid4().hex)
    backup = root / (target.name + ".old-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    dataset = []
    try:
        for page_index, page, question in selected:
            pair = pairs.get(str(question.get("id", "")))
            answer_ids = [str(value) for value in (pair or {}).get("answer_ids", [])]
            answer_rows = [answers.get(answer_id) for answer_id in answer_ids]
            matched = bool(
                pair and pair.get("status") == "accepted" and answer_ids
                and all(answer_rows)
                and all(str(item[1].get("status", "")) == "accepted" for item in answer_rows if item)
                and all(_scope_id(item[2], item[1]) == scope_id for item in answer_rows if item)
            )
            delivery_id = str(question.get("delivery_id") or question.get("id"))
            question_file = f"images/questions/{delivery_id}.jpg"
            (temporary / "images" / "questions").mkdir(parents=True, exist_ok=True)
            _write_image(temporary / question_file, _question_crop(project, page, question))
            answer_payloads = []
            if matched:
                sequence = 0
                for answer_row in answer_rows:
                    answer_page_index, answer_page, answer = answer_row
                    for segment_page_index, segment_page, segment in _answer_segments(
                            project, answer_page_index, answer_page, answer):
                        sequence += 1
                        answer_file = f"images/answers/{delivery_id}/answer_{sequence:02d}.jpg"
                        (temporary / "images" / "answers" / delivery_id).mkdir(parents=True, exist_ok=True)
                        _write_image(
                            temporary / answer_file,
                            _crop(data.source_path(project, segment_page, answer=True), segment["bbox"]),
                        )
                        answer_payloads.append(_record_payload(
                            segment, segment_page, segment_page_index, answer_file,
                        ))
            item = {
                "schema_version": 1,
                "book": {key: project.get(key, "") for key in
                         ("delivery_book_id", "language", "stage", "grade", "subject", "answer_mode")},
                "scope": scope,
                "question": _record_payload(question, page, page_index, question_file),
                "answer_status": "matched" if matched else "pending_generation",
                "answers": answer_payloads,
            }
            item_json = temporary / "json" / "items" / f"{delivery_id}.json"
            item_json.parent.mkdir(parents=True, exist_ok=True)
            item_json.write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            dataset.append(item)
        json_root = temporary / "json"
        json_root.mkdir(parents=True, exist_ok=True)
        with (json_root / "dataset.jsonl").open("w", encoding="utf-8") as handle:
            for item in dataset:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": scope,
            "answer_mode": project.get("answer_mode", "textbook"),
            "item_count": len(dataset),
            "matched_count": sum(item["answer_status"] == "matched" for item in dataset),
            "pending_generation_count": sum(item["answer_status"] == "pending_generation" for item in dataset),
            "skipped_unmatched_count": 0,
        }
        (json_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if target.exists():
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup.exists():
            shutil.rmtree(backup)
        _remove_previous_scope_folders(root, target, scope_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    return {**manifest, "output_path": str(target), "preflight": report}
