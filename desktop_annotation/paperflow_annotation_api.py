"""Small, local-only bridge between the Electron annotation desk and PaperFlow.

The Electron application owns the user interface and workspace files.  This
server deliberately exposes only a single page-at-a-time pre-annotation API;
it never uploads an image or starts a cloud model.  Keeping the bridge in the
standard library avoids adding a Flask dependency to every staff computer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_tools.manual_annotation_workspace import _answer_suggestions, _question_suggestions
from review_tools import manual_annotation_data as annotation_data
from review_tools.export_manual_annotations import export_project
from review_tools.delivery_workspace import delivery_root, sync_delivery_workspace
from review_tools.group_delivery import export_scope, preflight_scope
from review_tools.workspace_store import database_path, page_snapshot, sync_project
from cut_questions_v4 import infer_book_profile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
PROJECT_WRITE_LOCK = threading.RLock()
# PaddleOCR/PaddleX predictors are process-wide singletons.  The HTTP bridge is
# threaded, but a predictor invocation must not overlap another invocation.
PRELABEL_INFERENCE_LOCK = threading.Lock()
PRELABEL_INFLIGHT_LOCK = threading.Lock()
PRELABEL_INFLIGHT: dict[tuple, "_PrelabelFlight"] = {}
ANSWER_PRELABEL_REVIEW_THRESHOLD = 120
MUTATING_ROUTES = {
    "/api/project/open-book", "/api/project/save-page", "/api/project/set-pair",
    "/api/project/sync-delivery", "/api/project/export", "/api/project/set-delivery-output",
    "/api/project/set-book-metadata",
    "/api/project/reclassify-pages",
    "/api/project/rotate-page",
    "/api/project/accept-confident-matches",
    "/api/project/upsert-match-scope",
    "/api/project/assign-match-scope",
    "/api/project/remove-match-scope-pages",
    "/api/project/auto-match-scopes",
    "/api/project/save-cache",
    "/api/project/export-match-scope",
}


class _PrelabelFlight:
    """Result shared by simultaneous requests for the same source page."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: dict | None = None
        self.error: tuple[type[BaseException], str] | None = None


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _run_prelabel_once(key: tuple, operation) -> dict:
    """Coalesce identical in-flight work while serialising all model calls."""
    with PRELABEL_INFLIGHT_LOCK:
        flight = PRELABEL_INFLIGHT.get(key)
        owner = flight is None
        if owner:
            flight = _PrelabelFlight()
            PRELABEL_INFLIGHT[key] = flight

    assert flight is not None
    if owner:
        try:
            with PRELABEL_INFERENCE_LOCK:
                flight.result = _json_copy(operation())
        except Exception as exc:  # publish failures so waiters never hang
            flight.error = (type(exc), str(exc))
        finally:
            flight.event.set()
            with PRELABEL_INFLIGHT_LOCK:
                if PRELABEL_INFLIGHT.get(key) is flight:
                    PRELABEL_INFLIGHT.pop(key, None)
    else:
        flight.event.wait()

    if flight.error:
        error_type, message = flight.error
        if issubclass(error_type, ValueError):
            raise ValueError(message)
        raise RuntimeError(message)
    if flight.result is None:
        raise RuntimeError("预标注任务未返回结果。")
    return _json_copy(flight.result)


def _prelabel_diagnostics(records: list[dict], kind: str) -> dict:
    """Flag risky answer pages without deleting or silently merging candidates."""
    contexts: set[str] = set()
    numbered: dict[tuple[str, tuple[str, ...]], int] = {}
    missing_number_count = 0
    for record in records:
        context = str(record.get("context", "") or "").strip()
        if context:
            contexts.add(context)
        path = tuple(annotation_data.number_path(record))
        if not path:
            missing_number_count += 1
            continue
        key = (context, path)
        numbered[key] = numbered.get(key, 0) + 1
    duplicate_number_path_count = sum(max(0, count - 1) for count in numbered.values())

    warning_codes: list[str] = []
    warnings: list[str] = []
    if kind == "answer" and len(records) >= ANSWER_PRELABEL_REVIEW_THRESHOLD:
        warning_codes.append("dense_answer_candidates")
        warnings.append(
            f"本页识别到 {len(records)} 个答案候选，属于高密度答案页，请分段核对后再确认。"
        )
    if kind == "answer" and missing_number_count:
        warning_codes.append("missing_answer_number")
        warnings.append(f"有 {missing_number_count} 个答案候选缺少可配对题号。")
    if kind == "answer" and duplicate_number_path_count:
        warning_codes.append("duplicate_answer_number_path")
        warnings.append(f"有 {duplicate_number_path_count} 个答案候选与同上下文题号重复。")
    return {
        "record_count": len(records),
        "context_count": len(contexts),
        "missing_number_count": missing_number_count,
        "duplicate_number_path_count": duplicate_number_path_count,
        "requires_review": bool(warning_codes),
        "bulk_accept_safe": not warning_codes,
        "warning_codes": warning_codes,
        "warnings": warnings,
    }


def _remove_parent_question_boxes(project: dict) -> bool:
    """Flatten obsolete group/shared-material boxes without losing real questions."""
    changed = False
    for pages_key, records_key in (("pages", "questions"), ("answer_pages", "answers")):
        for page in project.get(pages_key, []):
            records = [record for record in page.get(records_key, []) if isinstance(record, dict)]
            removed_ids = {
                str(record.get("id", "")) for record in records
                if str(record.get("region_role", "question")) in {"group", "shared_material"}
            }
            kept = []
            for record in records:
                if str(record.get("region_role", "question")) in {"group", "shared_material"}:
                    changed = True
                    continue
                if str(record.get("parent_id", "")) in removed_ids:
                    record["parent_id"] = ""
                    changed = True
                kept.append(record)
            if len(kept) != len(records):
                page[records_key] = kept
    return changed


def _prelabel(payload: dict) -> dict:
    image_path = Path(str(payload.get("image_path", ""))).expanduser()
    if not image_path.is_absolute() or not image_path.is_file():
        raise ValueError("当前页图片不存在，无法生成预标注框。")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("当前页不是支持的图片格式。")

    profile = str(payload.get("profile", "") or "").strip()
    kind = str(payload.get("kind", "question") or "question")
    page_index = int(payload.get("page_index", -1))
    project = None
    if payload.get("project_path"):
        _project_path, project = _load_project(payload)
        pages_key = "answer_pages" if kind == "answer" else "pages"
        pages = project.get(pages_key, [])
        if page_index < 0 or page_index >= len(pages):
            raise ValueError("预标注页面序号无效。")
        expected = Path(annotation_data.source_path(project, pages[page_index], kind == "answer")).resolve()
        if image_path.resolve() != expected:
            raise ValueError("预标注图片与当前教材页面不匹配。")
        requested_auto_profile = not profile or profile == "auto"
        if requested_auto_profile:
            page_profile = str(pages[page_index].get("profile", "") or "").strip()
            # Answer parsing is unusually sensitive to this legacy setting;
            # preserve a page's recorded profile unless it is empty.  Question
            # pages may safely infer a more specific scene from a general
            # placeholder (for example a book titled "大试卷").
            profile = page_profile if (kind == "answer" or page_profile != "general") else ""
        if not profile:
            profile = infer_book_profile(
                str(project.get("book_root") or project.get("image_root") or image_path.parent)
            )
    if not profile or profile == "auto":
        profile = infer_book_profile(str(image_path.parent))
    if kind not in {"question", "answer"}:
        raise ValueError("未知的预标注类型。")

    previous_question = None
    if kind == "question" and project is not None and page_index >= 0:
        pages = project.get("pages", [])
        for previous_index in range(min(page_index, len(pages)) - 1, max(-1, page_index - 2), -1):
            candidates = [
                record for record in pages[previous_index].get("questions", [])
                if annotation_data.normalize_record(record).get("region_role") == "question"
            ]
            if candidates:
                previous_question = candidates[-1]
                break

    source_stat = image_path.stat()
    previous_signature = tuple(
        str(previous_question.get(field, "")) if previous_question else ""
        for field in ("id", "major_number", "subsection_number", "minor_number", "bbox")
    )
    request_key = (
        str(image_path.resolve()).casefold(), source_stat.st_size, source_stat.st_mtime_ns,
        kind, profile, page_index, previous_signature,
    )

    def infer() -> dict:
        if kind == "answer":
            records = _answer_suggestions(str(image_path), profile)
        else:
            records = _question_suggestions(
                str(image_path), profile, page_index=page_index,
                previous_question=previous_question,
            )
        return {
            "ok": True,
            "records": records,
            "engine": "paperflow-local",
            "profile": profile,
            "diagnostics": _prelabel_diagnostics(records, kind),
        }

    return _run_prelabel_once(request_key, infer)


def _project_response(project_path: Path, project: dict) -> dict:
    snapshot = json.loads(json.dumps(project, ensure_ascii=False))
    for page in snapshot.get("pages", []):
        page["source_path"] = annotation_data.source_path(project, page)
    for page in snapshot.get("answer_pages", []):
        page["source_path"] = annotation_data.source_path(project, page, True)
    for page in snapshot.get("ignored_pages", []):
        page["source_path"] = str(Path(str(project.get("book_root") or project.get("image_root", ""))) / page.get("image", ""))
    return {"ok": True, "project_path": str(project_path), "project": snapshot}


def _annotation_count(project: dict) -> int:
    return sum(len(page.get(field, [])) for pages, field in (
        (project.get("pages", []), "questions"),
        (project.get("answer_pages", []), "answers"),
    ) for page in pages)


def _book_content_signature(project: dict) -> tuple:
    """Identify a copied/moved textbook without relying on its absolute path."""
    result = []
    for pages_key, answer in (("pages", False), ("answer_pages", True)):
        pages = project.get(pages_key, [])
        entries = []
        for page in pages:
            relative = str(page.get("image", "")).replace("\\", "/")
            source = Path(annotation_data.source_path(project, page, answer))
            try:
                size = source.stat().st_size
            except OSError:
                size = -1
            entries.append((relative, size))
        result.append(tuple(entries))
    return tuple(result)


def _ordered_subsequence(needle: tuple, haystack: tuple) -> bool:
    if not needle:
        return not haystack
    position = 0
    for item in haystack:
        if item == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False


def _same_book_content(left: tuple, right: tuple) -> bool:
    return all(a == b or _ordered_subsequence(a, b) or _ordered_subsequence(b, a)
               for a, b in zip(left, right))


def _recover_moved_book_project(book_root: Path, current: dict) -> dict | None:
    """Recover annotations when the same textbook folder was copied or renamed."""
    if _annotation_count(current):
        return None
    signature = _book_content_signature(current)
    candidates = []
    central_root = ROOT / "runtime" / "manual_annotation_projects"
    for candidate_path in central_root.glob("*.json"):
        try:
            candidate = annotation_data.load_project(candidate_path)
            count = _annotation_count(candidate)
            if count and _same_book_content(_book_content_signature(candidate), signature):
                candidates.append((count, candidate_path.stat().st_mtime_ns, candidate))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    recovered = max(candidates, key=lambda item: (item[0], item[1]))[2]
    recovered["book_root"] = str(book_root)
    recovered["image_root"] = str(book_root)
    if recovered.get("answer_pages"):
        recovered["answer_image_root"] = str(book_root)
    return recovered


def _open_book(payload: dict) -> dict:
    book_root = Path(str(payload.get("book_root", ""))).expanduser().resolve()
    if not book_root.is_dir():
        raise ValueError("教材目录不存在。")
    project_path = annotation_data.default_book_project_path(book_root, ROOT / "runtime")
    if project_path.is_file():
        project = annotation_data.load_project(project_path)
        changed = bool(annotation_data.remove_cached_delivery_pages(project, book_root))
        changed = _remove_parent_question_boxes(project) or changed
        if changed:
            annotation_data.save_project(project, project_path)
    else:
        project = annotation_data.create_book_project(str(book_root))
        annotation_data.save_project(project, project_path)
    # Opening should be proportional to the source pages, not to every field in
    # every historical record. The SQLite read model is refreshed lazily only
    # when its workspace endpoint is requested.
    return _project_response(project_path, project)


def _load_project(payload: dict) -> tuple[Path, dict]:
    project_path = Path(str(payload.get("project_path", ""))).expanduser().resolve()
    if not project_path.is_file():
        raise ValueError("标注项目不存在。")
    project = annotation_data.load_project(project_path)
    book_root = Path(str(project.get("book_root") or project.get("image_root") or "")).resolve()
    central_root = (ROOT / "runtime" / "manual_annotation_projects").resolve()
    is_central = project_path.parent == central_root
    is_beside_book = project_path.name == "PaperFlow标注项目.json" and project_path.parent == book_root
    if not is_central and not is_beside_book:
        raise ValueError("标注项目路径与教材目录不匹配。")
    return project_path, project


def _save_page(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    kind = str(payload.get("kind", "question"))
    pages_key, records_key = ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
    page_index = int(payload.get("page_index", -1))
    pages = project.get(pages_key, [])
    if page_index < 0 or page_index >= len(pages):
        raise ValueError("页面序号无效。")
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("标注框数据无效。")
    drawing_tool_active = payload.get("drawing_tool_active", False)
    if not isinstance(drawing_tool_active, bool):
        raise ValueError("绘制状态无效。")
    existing_records = {
        str(record.get("id", "")): record
        for record in pages[page_index].get(records_key, [])
        if isinstance(record, dict) and record.get("id")
    }
    page_scope_id = str(pages[page_index].get("match_scope_id", "") or "").strip()
    scope_regions = pages[page_index].get("match_scope_regions", [])

    def inherited_scope(item: dict, box: list[int]) -> str:
        explicit = str(item.get("match_scope_id", "") or "").strip()
        if explicit:
            return explicit
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        for region in reversed(scope_regions if isinstance(scope_regions, list) else []):
            region_box = region.get("bbox", []) if isinstance(region, dict) else []
            if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
                continue
            try:
                rx0, ry0, rx1, ry1 = (float(value) for value in region_box)
            except (TypeError, ValueError):
                continue
            if rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1:
                return str(region.get("scope_id", "") or "").strip()
        return page_scope_id
    cleaned = []
    for record_index, item in enumerate(records, 1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {record_index} 个标注框数据无效，页面未保存。")
        if str(item.get("region_role", "question")) in {"group", "shared_material"}:
            continue
        box = item.get("bbox", [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"第 {record_index} 个标注框坐标无效，页面未保存。")
        try:
            x0, y0, x1, y1 = (int(round(float(value))) for value in box)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"第 {record_index} 个标注框坐标无效，页面未保存。") from error
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"第 {record_index} 个标注框范围无效，页面未保存。")
        minor_number = str(item.get("minor_number", item.get("number", "")))
        major_title = str(item.get("major_title", item.get("major", "")))
        shared_context_box = item.get("shared_context_bbox", [])
        if not isinstance(shared_context_box, list) or len(shared_context_box) != 4:
            shared_context_box = []
        shared_context_segments = item.get("shared_context_segments", [])
        record = annotation_data.new_record(
            [x0, y0, x1, y1], answer=kind == "answer",
            number=minor_number,
            minor_number=minor_number,
            major=major_title,
            major_title=major_title,
            major_number=str(item.get("major_number", "")),
            subsection_title=str(item.get("subsection_title", "")),
            subsection_number=str(item.get("subsection_number", "")),
            number_path=item.get("number_path", []),
            parent_id=str(item.get("parent_id", "")),
            region_role=str(item.get("region_role", "question")),
            continuation_of=str(item.get("continuation_of", "")),
            minor_number_inferred=bool(item.get("minor_number_inferred", False)),
            shared_context_bbox=[int(round(float(value))) for value in shared_context_box],
            shared_context_segments=shared_context_segments,
            use_shared_context=bool(item.get("use_shared_context", False)),
            context=str(item.get("context", "")),
            text=str(item.get("text", "")),
            question_type=str(item.get("question_type", "")),
            difficulty=str(item.get("difficulty", "")),
            knowledge_points=item.get("knowledge_points", []),
            options=item.get("options", []),
            analysis=str(item.get("analysis", "")),
            illustrations=item.get("illustrations", []),
            origin=str(item.get("origin", "manual")),
            answer_split_status=str(item.get("answer_split_status", "")),
            boundary_source=str(item.get("boundary_source", "")),
            detector_assisted=bool(item.get("detector_assisted", False)),
            detector_confidence=float(item.get("detector_confidence", 0.0) or 0.0),
            match_scope_id=inherited_scope(item, [x0, y0, x1, y1]),
        )
        record["answer"] = str(item.get("answer", ""))
        if item.get("id"):
            record["id"] = str(item["id"])
        # OCR and public delivery identity belong to the logical record, not
        # the editable form. Preserve them while bbox/number fields are saved.
        previous = existing_records.get(record.get("id", ""), {})
        for persistent_key in ("delivery_id",):
            if persistent_key in previous:
                record[persistent_key] = previous[persistent_key]
        if isinstance(item.get("ocr"), dict):
            record["ocr"] = dict(item["ocr"])
        elif "ocr" in previous:
            record["ocr"] = previous["ocr"]
        cleaned.append(record)
    cleaned_ids = [str(record.get("id", "")) for record in cleaned]
    if any(not value for value in cleaned_ids) or len(cleaned_ids) != len(set(cleaned_ids)):
        raise ValueError("标注框内部 ID 为空或重复，页面未保存。")
    other_ids: set[str] = set()
    for other_pages_key, other_records_key in (
        ("pages", "questions"), ("answer_pages", "answers"),
        ("ignored_pages", "questions"), ("ignored_pages", "answers"),
    ):
        for other_page_index, other_page in enumerate(project.get(other_pages_key, [])):
            is_current_page = (
                other_pages_key == pages_key and other_page_index == page_index
            )
            if is_current_page:
                continue
            for other_record in other_page.get(other_records_key, []):
                if isinstance(other_record, dict) and other_record.get("id"):
                    other_ids.add(str(other_record["id"]))
    duplicate_ids = sorted(set(cleaned_ids).intersection(other_ids))
    if duplicate_ids:
        raise ValueError("标注框内部 ID 与教材其他页面重复，页面未保存。")
    previous_ids = set(existing_records)
    current_ids = {str(record.get("id", "")) for record in cleaned if record.get("id")}
    removed_ids = previous_ids - current_ids
    if payload.get("accepted") and kind == "question" and not cleaned:
        raise ValueError("题目页没有题框，不能确认；请先绘制题框或重新归类页面。")
    old_pairs = list(project.get("pairs", []))
    if removed_ids:
        if kind == "question":
            project["pairs"] = [
                pair for pair in old_pairs
                if str(pair.get("question_id", "")) not in removed_ids
            ]
        else:
            # A multi-part answer relation can remain valid after one answer
            # box is removed.  Remove only the missing IDs and drop the
            # relation if it no longer points to any answer.
            retained_pairs = []
            for pair in old_pairs:
                answer_ids = [
                    str(value) for value in pair.get("answer_ids", [])
                    if str(value) not in removed_ids
                ]
                if not answer_ids and pair.get("answer_ids"):
                    continue
                pair["answer_ids"] = answer_ids
                retained_pairs.append(pair)
            project["pairs"] = retained_pairs
    pages[page_index][records_key] = cleaned
    pages[page_index]["status"] = "accepted" if payload.get("accepted") else "draft"
    annotation_data.save_project(project, project_path)
    warnings = annotation_data.hierarchy_warnings(cleaned)
    if payload.get("accepted") and kind == "answer":
        missing_answer_count = sum(
            not str(record.get("analysis") or record.get("text") or "").strip()
            if str(record.get("region_role", "question")) == "continuation"
            else not str(record.get("answer", "")).strip()
            for record in cleaned
        )
        if missing_answer_count:
            warnings.append(
                f"本页已确认，但有 {missing_answer_count} 个答案框尚无答案文字；请在后续标注中补充，并在交付前复核。"
            )
    response = {
        "ok": True,
        "saved": len(cleaned),
        "status": pages[page_index]["status"],
        "project_revision": project["project_revision"],
        "warnings": warnings,
        "removed_pairs": len(old_pairs) - len(project.get("pairs", [])),
        # The draw button is renderer state only.  A save can contain it for
        # diagnostics, but the project always persists completed records only.
        "transient_drawing_tool_active": drawing_tool_active,
    }
    return response


def _match_candidates(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    question_id = str(payload.get("question_id", ""))
    question = next((record for _, _, record in annotation_data.all_questions(project)
                     if record.get("id") == question_id), None)
    if not question:
        raise ValueError("题目不存在。")
    paired_ids = annotation_data.paired_answer_ids(project, question_id)
    answer_count = sum(
        1 for _page_index, page, answer in annotation_data.all_answers(project)
        if str(page.get("status", "")) == "accepted"
        and str(answer.get("region_role", "question")) == "question"
    )
    all_ranked = annotation_data.candidates(
        project, question, limit=max(100, answer_count), accepted_only=True,
    )
    ranked = [
        item for index, item in enumerate(all_ranked)
        if index < 100 or str(item[3].get("id", "")) in paired_ids
    ]
    candidates = []
    for index, (score, page_index, page, answer) in enumerate(ranked):
        second_score = ranked[index + 1][0] if index + 1 < len(ranked) else float("-inf")
        candidates.append({
            "score": score,
            "page_index": page_index,
            "source_path": annotation_data.source_path(project, page, True),
            "answer": answer,
            "match_reason": annotation_data.candidate_reason(question, answer),
            "recommended": bool(
                index == 0 and annotation_data.number_path_matches(question, answer)
                and score >= 180 and score - second_score >= 80
            ),
        })
    return {
        "ok": True,
        "question": question,
        "paired_answer_ids": paired_ids,
        "candidates": candidates,
    }


def _reclassify_pages(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    source_kind = str(payload.get("source_kind", "question"))
    target_kind = str(payload.get("target_kind", "answer"))
    page_indices = payload.get("page_indices", [])
    if not isinstance(page_indices, list):
        raise ValueError("页面序号格式无效。")
    result = annotation_data.reclassify_pages(
        project, source_kind, [int(value) for value in page_indices], target_kind,
    )
    annotation_data.save_project(project, project_path)
    return {"ok": True, **result, **_project_response(project_path, project)}


def _rotate_page(payload: dict) -> dict:
    """Rotate one project page clockwise without modifying its source image."""
    import cv2

    project_path, project = _load_project(payload)
    kind = "answer" if str(payload.get("kind", "question")) == "answer" else "question"
    pages_key, records_key = (
        ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
    )
    page_index = int(payload.get("page_index", -1))
    pages = project.get(pages_key, [])
    if page_index < 0 or page_index >= len(pages):
        raise ValueError("页面序号无效。")
    page = pages[page_index]
    original_path = Path(annotation_data.original_source_path(
        project, page, answer=(kind == "answer"),
    ))
    source = cv2.imdecode(np.fromfile(str(original_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("无法读取当前教材页。")

    old_rotation = int(page.get("rotation", 0) or 0) % 360
    if page.get(records_key):
        raise ValueError("当前页已有标注框，请先清空当前页再旋转。")
    if kind == "question":
        referenced = any(
            int(segment.get("page_index", -1)) == page_index
            for question_page in project.get("pages", [])
            for record in question_page.get("questions", [])
            for segment in record.get("shared_context_segments", [])
            if isinstance(segment, dict)
        )
        if referenced:
            raise ValueError("当前页已被其他题目作为共享材料引用，不能旋转。")
    new_rotation = (old_rotation + 90) % 360
    rotated = source
    for _ in range(new_rotation // 90):
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)

    book_root = Path(str(project.get("book_root") or project.get("image_root", ""))).resolve()
    token = hashlib.sha1(str(page.get("image", "")).encode("utf-8")).hexdigest()[:16]
    old_relative_cache = Path(str(page.get("rotated_image", "") or ""))
    relative_cache = Path("_paperflow_page_views") / f"{token}-r{new_rotation}.png"
    cache_path = (book_root / relative_cache).resolve()
    cache_path.relative_to(book_root)
    if new_rotation:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", rotated)
        if not ok:
            raise ValueError("旋转页面编码失败。")
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            encoded.tofile(str(temporary))
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        page["rotation"] = new_rotation
        page["rotated_image"] = relative_cache.as_posix()
    else:
        page.pop("rotation", None)
        page.pop("rotated_image", None)

    annotation_data.save_project(project, project_path)
    if old_relative_cache.parts and old_relative_cache != relative_cache:
        old_cache_path = (book_root / old_relative_cache).resolve()
        try:
            old_cache_path.relative_to((book_root / "_paperflow_page_views").resolve())
        except ValueError:
            pass
        else:
            old_cache_path.unlink(missing_ok=True)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True, "rotation": new_rotation, "delivery": None,
        "warnings": [],
        **_project_response(project_path, project),
    }


def _set_pair(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    question_id = str(payload.get("question_id", ""))
    answer_ids = [str(value) for value in payload.get("answer_ids", [])]
    accepted_question_ids = {
        str(question.get("id", ""))
        for _page_index, page, question in annotation_data.all_questions(project)
        if str(page.get("status", "")) == "accepted"
        and str(question.get("region_role", "question")) == "question"
    }
    if question_id not in accepted_question_ids:
        raise ValueError("只能为已经确认的题目页建立答案匹配。")
    accepted_answer_ids = {
        str(answer.get("id", ""))
        for _page_index, page, answer in annotation_data.all_answers(project)
        if str(page.get("status", "")) == "accepted"
        and str(answer.get("region_role", "question")) == "question"
    }
    invalid_answer_ids = [value for value in answer_ids if value not in accepted_answer_ids]
    if invalid_answer_ids:
        raise ValueError("只能匹配已经确认的答案页，请先完成答案页审核。")
    status = "accepted" if answer_ids else "no_answer"
    annotation_data.set_pair(project, question_id, answer_ids, status)
    annotation_data.save_project(project, project_path)
    sync_project(project_path, project, ROOT / "runtime")
    saved_pair = next(
        pair for pair in project.get("pairs", [])
        if str(pair.get("question_id", "")) == question_id
    )
    return {
        "ok": True, "status": status, "answer_ids": answer_ids,
        "pair": saved_pair, "project_revision": project["project_revision"],
        "delivery": None,
    }


def _accept_confident_matches(payload: dict) -> dict:
    """Conservatively accept only unambiguous, confirmed-answer suggestions."""
    project_path, project = _load_project(payload)
    existing = {str(pair.get("question_id", "")): pair for pair in project.get("pairs", [])}
    used_answers = {
        str(answer_id)
        for pair in project.get("pairs", []) if pair.get("status") == "accepted"
        for answer_id in pair.get("answer_ids", [])
    }
    accepted_questions = [
        question for _page_index, page, question in annotation_data.all_questions(project)
        if str(page.get("status", "")) == "accepted"
        and str(question.get("region_role", "question")) == "question"
    ]
    proposals = []
    reserved_answers: set[str] = set()
    for question in accepted_questions:
        question_id = str(question.get("id", ""))
        if question_id in existing:
            continue
        ranked = annotation_data.candidates(project, question, limit=2, accepted_only=True)
        if not ranked:
            continue
        best_score, _page_index, _page, best_answer = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
        answer_id = str(best_answer.get("id", ""))
        # Exact hierarchy matches normally score well above 180. The margin
        # prevents repeated printed numbers in different units being accepted.
        if (not annotation_data.number_path_matches(question, best_answer)
                or best_score < 180 or best_score - second_score < 80
                or answer_id in used_answers or answer_id in reserved_answers):
            continue
        proposals.append((question_id, answer_id, float(best_score)))
        reserved_answers.add(answer_id)
    saved_pairs: dict[str, dict] = {}
    for question_id, answer_id, _score in proposals:
        annotation_data.set_pair(project, question_id, [answer_id], "accepted")
        saved_pairs[question_id] = next(
            pair for pair in project.get("pairs", [])
            if str(pair.get("question_id", "")) == question_id
        )
    if proposals:
        annotation_data.save_project(project, project_path)
        sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "accepted_count": len(proposals),
        "matches": [
            {"question_id": question_id, "answer_ids": [answer_id], "score": score,
             "pair": saved_pairs[question_id]}
            for question_id, answer_id, score in proposals
        ],
        "project_revision": project.get("project_revision", 0),
        "delivery": None,
    }


def _upsert_match_scope(payload: dict) -> dict:
    """Create or rename a book-wide matching scope."""
    project_path, project = _load_project(payload)
    name = str(payload.get("name", "") or "").strip()
    if not name:
        raise ValueError("匹配分组名称不能为空。")
    if len(name) > 80:
        raise ValueError("匹配分组名称不能超过 80 个字符。")
    scope_id = str(payload.get("scope_id", "") or "").strip()
    scopes = project.setdefault("match_scopes", [])
    duplicate = next(
        (item for item in scopes if str(item.get("name", "")).casefold() == name.casefold()
         and str(item.get("id", "")) != scope_id),
        None,
    )
    if duplicate:
        raise ValueError("已经存在同名匹配分组。")
    scope = next((item for item in scopes if str(item.get("id", "")) == scope_id), None)
    if scope is None:
        scope = {"id": "scope-" + uuid.uuid4().hex, "name": name}
        scopes.append(scope)
    else:
        scope["name"] = name
    annotation_data.save_project(project, project_path)
    return {"ok": True, "scope": scope,
            "project": _project_response(project_path, project)["project"]}


def _assign_match_scope(payload: dict) -> dict:
    """Assign a shared scope to a rectangle, current page, or page selection."""
    project_path, project = _load_project(payload)
    kind = "answer" if str(payload.get("kind", "question")) == "answer" else "question"
    pages_key, records_key = ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
    pages = project.get(pages_key, [])
    scope_id = str(payload.get("scope_id", "") or "").strip()
    valid_scope_ids = {str(item.get("id", "")) for item in project.get("match_scopes", [])}
    if scope_id not in valid_scope_ids:
        raise ValueError("请选择有效的匹配分组。")
    raw_indices = payload.get("page_indices", [])
    if not isinstance(raw_indices, list):
        raise ValueError("页面范围无效。")
    try:
        page_indices = sorted(set(int(value) for value in raw_indices))
    except (TypeError, ValueError) as error:
        raise ValueError("页面范围无效。") from error
    if not page_indices or any(index < 0 or index >= len(pages) for index in page_indices):
        raise ValueError("请选择有效页面。")
    record_ids = {str(value) for value in payload.get("record_ids", []) if str(value)}
    set_page_default = bool(payload.get("set_page_default", False))
    region_box = payload.get("region_bbox")
    normalized_region = None
    if region_box is not None:
        if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
            raise ValueError("框选区域坐标无效。")
        try:
            x0, y0, x1, y1 = (int(round(float(value))) for value in region_box)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("框选区域坐标无效。") from error
        if x1 <= x0 or y1 <= y0:
            raise ValueError("框选区域范围无效。")
        if len(page_indices) != 1:
            raise ValueError("框选区域只能应用到当前一页。")
        normalized_region = [x0, y0, x1, y1]

    assigned_ids: list[str] = []
    for page_index in page_indices:
        page = pages[page_index]
        if set_page_default:
            page["match_scope_id"] = scope_id
        if normalized_region is not None:
            page.setdefault("match_scope_regions", []).append({
                "id": "region-" + uuid.uuid4().hex,
                "scope_id": scope_id,
                "bbox": normalized_region,
            })
        for record in page.get(records_key, []):
            record_id = str(record.get("id", ""))
            selected = set_page_default or record_id in record_ids
            if normalized_region is not None:
                box = record.get("bbox", [])
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    center_x = (float(box[0]) + float(box[2])) / 2
                    center_y = (float(box[1]) + float(box[3])) / 2
                    selected = (normalized_region[0] <= center_x <= normalized_region[2]
                                and normalized_region[1] <= center_y <= normalized_region[3])
            if selected:
                record["match_scope_id"] = scope_id
                if record_id:
                    assigned_ids.append(record_id)
    annotation_data.save_project(project, project_path)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "assigned_count": len(set(assigned_ids)),
        "page_count": len(page_indices),
        "project": _project_response(project_path, project)["project"],
    }


def _remove_match_scope_pages(payload: dict) -> dict:
    """Remove selected pages from one scope without deleting pages or records."""
    project_path, project = _load_project(payload)
    scope_id = str(payload.get("scope_id", "") or "").strip()
    valid_scope_ids = {str(item.get("id", "")) for item in project.get("match_scopes", [])}
    if scope_id not in valid_scope_ids:
        raise ValueError("请选择有效的匹配分组。")
    selections = payload.get("pages", [])
    if not isinstance(selections, list) or not selections:
        raise ValueError("请选择要移出分组的页面。")
    normalized = set()
    for selection in selections:
        if not isinstance(selection, dict):
            raise ValueError("页面选择无效。")
        kind = "answer" if str(selection.get("kind", "question")) == "answer" else "question"
        try:
            page_index = int(selection.get("page_index", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("页面选择无效。") from error
        pages = project.get("answer_pages" if kind == "answer" else "pages", [])
        if page_index < 0 or page_index >= len(pages):
            raise ValueError("页面选择无效。")
        normalized.add((kind, page_index))

    removed_regions = 0
    cleared_records = 0
    for kind, page_index in sorted(normalized):
        pages_key, records_key = ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
        page = project[pages_key][page_index]
        if str(page.get("match_scope_id", "")) == scope_id:
            page["match_scope_id"] = ""
        regions = page.get("match_scope_regions", [])
        kept_regions = [
            region for region in regions
            if str(region.get("scope_id", "")) != scope_id
        ]
        removed_regions += len(regions) - len(kept_regions)
        page["match_scope_regions"] = kept_regions
        for record in page.get(records_key, []):
            if str(record.get("match_scope_id", "")) == scope_id:
                record["match_scope_id"] = ""
                cleared_records += 1
    annotation_data.save_project(project, project_path)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "removed_page_count": len(normalized),
        "removed_region_count": removed_regions,
        "cleared_record_count": cleared_records,
        "project": _project_response(project_path, project)["project"],
    }


def _auto_match_scopes(payload: dict) -> dict:
    """Match only unique confirmed Q/A records sharing scope and canonical number."""
    project_path, project = _load_project(payload)
    existing_questions = {
        str(pair.get("question_id", "")) for pair in project.get("pairs", [])
    }
    used_answers = {
        str(answer_id) for pair in project.get("pairs", []) if pair.get("status") == "accepted"
        for answer_id in pair.get("answer_ids", [])
    }
    question_map: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    answer_map: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for _index, page, record in annotation_data.all_questions(project):
        if str(page.get("status", "")) != "accepted" or record.get("region_role", "question") != "question":
            continue
        scope_id = str(record.get("match_scope_id") or page.get("match_scope_id") or "").strip()
        path = annotation_data.canonical_number_path(record)
        if scope_id and path:
            question_map.setdefault((scope_id, path), []).append(record)
    for _index, page, record in annotation_data.all_answers(project):
        if str(page.get("status", "")) != "accepted" or record.get("region_role", "question") != "question":
            continue
        scope_id = str(record.get("match_scope_id") or page.get("match_scope_id") or "").strip()
        path = annotation_data.canonical_number_path(record)
        if scope_id and path:
            answer_map.setdefault((scope_id, path), []).append(record)

    proposals: list[tuple[str, str, str]] = []
    conflicts = []
    all_keys = sorted(set(question_map).union(answer_map), key=lambda item: (item[0], item[1]))
    for scope_id, path in all_keys:
        questions = question_map.get((scope_id, path), [])
        answers = answer_map.get((scope_id, path), [])
        available_questions = [item for item in questions if str(item.get("id", "")) not in existing_questions]
        available_answers = [item for item in answers if str(item.get("id", "")) not in used_answers]
        if len(questions) == 1 and len(answers) == 1 and len(available_questions) == 1 and len(available_answers) == 1:
            proposals.append((str(questions[0]["id"]), str(answers[0]["id"]), scope_id))
        elif questions or answers:
            conflicts.append({
                "scope_id": scope_id,
                "number": "-".join(path),
                "question_count": len(questions),
                "answer_count": len(answers),
                "reason": "ambiguous_or_already_matched",
            })
    saved_pairs = []
    for question_id, answer_id, scope_id in proposals:
        annotation_data.set_pair(
            project, question_id, [answer_id], "accepted", source="scope_number_auto"
        )
        saved_pairs.append(next(
            pair for pair in project.get("pairs", [])
            if str(pair.get("question_id", "")) == question_id
        ))
    if proposals:
        annotation_data.save_project(project, project_path)
        sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "accepted_count": len(proposals),
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:100],
        "pairs": saved_pairs,
        "project_revision": project.get("project_revision", 0),
        "delivery": None,
    }


def _set_delivery_output(payload: dict) -> dict:
    """Set a user-selected delivery root without eagerly materialising pages."""
    project_path, project = _load_project(payload)
    selected_root = Path(str(payload.get("output_root", ""))).expanduser()
    if not selected_root.is_absolute():
        raise ValueError("交付输出位置必须是本机文件夹。")
    selected_root = selected_root.resolve()
    if not selected_root.is_dir():
        raise ValueError("选择的交付输出文件夹不存在。")
    project["delivery_output_root"] = str(selected_root)
    annotation_data.save_project(project, project_path)
    output_path = delivery_root(project, ROOT / "runtime")
    return {
        "ok": True,
        "output_root": str(selected_root),
        "output_path": str(output_path),
        "project": _project_response(project_path, project)["project"],
        "delivery": {"book_id": project["delivery_book_id"], "output_path": str(output_path)},
    }


def _set_book_metadata(payload: dict) -> dict:
    """Persist fields shared by every question in one textbook."""
    project_path, project = _load_project(payload)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("教材信息格式无效。")
    allowed = {
        "language": {"zh", "en"},
        "stage": {"小学", "初中", "高中"},
        "answer_mode": annotation_data.ANSWER_MODES,
    }
    for key in ("language", "stage", "grade", "subject"):
        value = str(metadata.get(key, "") or "").strip()
        if key in allowed and value not in allowed[key]:
            raise ValueError(f"教材{key}填写无效。")
        if not value:
            raise ValueError("语言、学段、年级、学科均不能为空。")
        project[key] = value
    answer_mode = str(metadata.get("answer_mode", project.get("answer_mode", "textbook")) or "").strip()
    if answer_mode not in annotation_data.ANSWER_MODES:
        raise ValueError("教材答案模式填写无效。")
    project["answer_mode"] = answer_mode
    annotation_data.save_project(project, project_path)
    return {"ok": True, "project": _project_response(project_path, project)["project"],
            "delivery": None}


def _save_project_cache(payload: dict) -> dict:
    """Atomically persist the renderer's complete in-memory textbook cache."""
    project_path, existing = _load_project(payload)
    incoming = payload.get("project")
    if not isinstance(incoming, dict):
        raise ValueError("教材缓存格式无效。")
    try:
        base_revision = int(payload.get("base_revision", incoming.get("project_revision", 0)))
        current_revision = int(existing.get("project_revision", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("教材缓存修订号无效。") from error
    if base_revision != current_revision:
        raise ValueError("教材缓存已在其他操作中更新，请重新打开教材后再保存。")
    expected_root = Path(str(existing.get("book_root") or existing.get("image_root") or "")).resolve()
    incoming_root = Path(str(incoming.get("book_root") or incoming.get("image_root") or "")).resolve()
    if incoming_root != expected_root:
        raise ValueError("教材缓存路径与当前教材不匹配。")
    for page in incoming.get("pages", []):
        page.pop("source_path", None)
    for page in incoming.get("answer_pages", []):
        page.pop("source_path", None)
    for page in incoming.get("ignored_pages", []):
        page.pop("source_path", None)
    annotation_data.merge_persistent_delivery_state(incoming, existing)
    annotation_data.save_project(incoming, project_path)
    return {"ok": True, "project_path": str(project_path),
            "project_revision": incoming["project_revision"],
            "question_pages": len(incoming.get("pages", [])),
            "answer_pages": len(incoming.get("answer_pages", []))}


def _delivery_location(payload: dict) -> dict:
    _project_path, project = _load_project(payload)
    return {"ok": True, "output_path": str(delivery_root(project, ROOT / "runtime"))}


def _workspace_page(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    kind = "answer" if str(payload.get("kind", "question")) == "answer" else "question"
    page_index = int(payload.get("page_index", 0))
    snapshot = page_snapshot(project_path, project, ROOT / "runtime", kind, page_index)
    return {"ok": True, **snapshot}


def _export_project(payload: dict) -> dict:
    project_path, _project = _load_project(payload)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = ROOT / "runtime" / "manual_annotation_exports" / project_path.stem / stamp
    report = export_project(str(project_path), str(target))
    return {"ok": True, "output_path": str(target), "report": report}


def _sync_delivery(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    result = sync_delivery_workspace(project_path, ROOT / "runtime", materialize_all=True)
    return {"ok": True, "project_revision": project.get("project_revision", 0), **result}


def _group_export_preflight(payload: dict) -> dict:
    _project_path, project = _load_project(payload)
    scope_id = str(payload.get("scope_id", "") or "").strip()
    if not scope_id:
        raise ValueError("请选择要生成的匹配分组。")
    return {"ok": True, **preflight_scope(project, scope_id)}


def _export_match_scope(payload: dict) -> dict:
    project_path, _project = _load_project(payload)
    scope_id = str(payload.get("scope_id", "") or "").strip()
    if not scope_id:
        raise ValueError("请选择要生成的匹配分组。")
    result = export_scope(
        project_path, ROOT / "runtime", scope_id,
        include_pending=bool(payload.get("include_pending", False)),
    )
    return {"ok": True, **result}


ROUTES = {
    "/api/prelabel": _prelabel,
    "/api/project/open-book": _open_book,
    "/api/project/save-page": _save_page,
    "/api/project/match-candidates": _match_candidates,
    "/api/project/reclassify-pages": _reclassify_pages,
    "/api/project/rotate-page": _rotate_page,
    "/api/project/set-pair": _set_pair,
    "/api/project/accept-confident-matches": _accept_confident_matches,
    "/api/project/upsert-match-scope": _upsert_match_scope,
    "/api/project/assign-match-scope": _assign_match_scope,
    "/api/project/remove-match-scope-pages": _remove_match_scope_pages,
    "/api/project/auto-match-scopes": _auto_match_scopes,
    "/api/project/set-delivery-output": _set_delivery_output,
    "/api/project/set-book-metadata": _set_book_metadata,
    "/api/project/save-cache": _save_project_cache,
    "/api/project/delivery-location": _delivery_location,
    "/api/project/workspace-page": _workspace_page,
    "/api/project/export": _export_project,
    "/api/project/sync-delivery": _sync_delivery,
    "/api/project/group-export-preflight": _group_export_preflight,
    "/api/project/export-match-scope": _export_match_scope,
}


class PaperFlowHandler(BaseHTTPRequestHandler):
    server_version = "PaperFlowLocal/1.0"

    def log_message(self, _format: str, *_args) -> None:
        # Electron owns the visible log.  Avoid writing a line for each browser
        # request into the console of a staff-facing application.
        return

    def _respond(self, status: HTTPStatus, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "null")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib request handler API
        self._respond(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802 - stdlib request handler API
        if urlparse(self.path).path == "/api/health":
            self._respond(HTTPStatus.OK, {"ok": True, "service": "paperflow-local"})
            return
        self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib request handler API
        route_path = urlparse(self.path).path
        route = ROUTES.get(route_path)
        if route is None:
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BODY_BYTES:
                raise ValueError("请求数据无效。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求数据格式错误。")
            if route_path in MUTATING_ROUTES:
                with PROJECT_WRITE_LOCK:
                    result = route(payload)
            else:
                result = route(payload)
            self._respond(HTTPStatus.OK, result)
        except ValueError as error:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
        except Exception as error:  # Keep a local worker failure actionable in UI.
            traceback.print_exc()
            self._respond(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "ok": False,
                "error": f"本地预标注失败：{error}",
            })


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8601), PaperFlowHandler)
    print("PaperFlow local annotation bridge listening on 127.0.0.1:8601", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
