"""Incrementally maintain database-ready question/answer delivery files."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from review_tools import manual_annotation_data as data
from review_tools.export_manual_annotations import _crop, _question_crop


K12_QUESTION_TYPES = {
    "选择题": "选择", "填空题": "填空", "判断题": "判断",
    "计算题": "计算", "解答题": "计算", "应用题": "综合应用",
    "证明题": "证明", "实验题": "实验探究", "实验探究题": "实验探究",
    "阅读理解": "阅读理解", "材料分析": "材料分析", "综合应用": "综合应用",
}


def _wide_path(path: Path) -> str:
    """Windows delivery trees are deep; opt into the extended-length path form.

    A server generation writes under
    ``<root>/exports/<version>/staging/<book_id>/images/questions/P0001/...``.
    On the data machine that can exceed the classic 260-character MAX_PATH,
    where directory creation still succeeds but opening the final file fails
    with a bare FileNotFoundError.
    """
    text = str(path.resolve())
    if os.name == "nt" and len(text) >= 240 and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(_wide_path(temporary), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(_wide_path(temporary), _wide_path(path))


def _atomic_image(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("交付图片编码失败")
    temporary = path.with_suffix(path.suffix + ".tmp")
    # ndarray.tofile() goes through the C runtime's fopen, which cannot use
    # long paths and mishandles non-ASCII folders.  Python IO honours the
    # wide-path prefix, matching how PaperFlow reads scans (fromfile/imdecode).
    with open(_wide_path(temporary), "wb") as handle:
        handle.write(encoded.tobytes())
    os.replace(_wide_path(temporary), _wide_path(path))


def _delivery_artifact_path(root: Path, value: object) -> Path | None:
    """Resolve a manifest path only when it names a managed delivery artifact."""
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute():
        return None
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        managed_relative = candidate.relative_to(resolved_root)
    except ValueError:
        return None
    if (not managed_relative.parts
            or managed_relative.parts[0].lower() not in {"images", "json"}
            or candidate.suffix.lower() not in {".jpg", ".json"}):
        return None
    return candidate


def _number_sort_key(record: dict) -> tuple:
    parts = []
    for value in data.number_path(record):
        number = data.chinese_number_to_int(value)
        normalized = re.sub(r"\s+", "", str(value)).lower()
        parts.append((0, number, "") if number is not None else (1, 0, normalized))
    return tuple(parts)


def _record_position(row: tuple[int, dict, dict]) -> tuple:
    """Stable human order: page, printed hierarchy, box position, persistent id."""
    page_index, _page, record = row
    box = record.get("bbox", [0, 0, 0, 0])
    hierarchy = _number_sort_key(record)
    return (page_index, 0 if hierarchy else 1, hierarchy,
            int(box[1]), int(box[0]), str(record.get("delivery_id", "")))


def _clean_text(value: object) -> str:
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip()
             for line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [_clean_text(item) for item in re.split(r"[\n,，;；]+", str(value or "")) if _clean_text(item)]


_CHOICE_MARKER = re.compile(r"(^|[\n\s])([A-DＡ-Ｄ])\s*[\.．、:：]\s*")


def _split_ocr_choices(value: object) -> tuple[str, list[str]]:
    """Extract A-D choices only when OCR text contains at least two markers."""
    source = _clean_text(value)
    matches = list(_CHOICE_MARKER.finditer(source))
    if len(matches) < 2:
        return source, []
    first = matches[0]
    marker_offset = next((index for index, char in enumerate(first.group(0))
                          if char in "ABCDＡＢＣＤ"), 0)
    question = source[:first.start() + marker_offset].strip()
    options: list[str] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[match.end():body_end].strip()
        label = match.group(2)
        if "Ａ" <= label <= "Ｄ":
            label = chr(ord(label) - 0xFEE0)
        if body:
            options.append(f"{label}. {body}")
    return (question, options) if len(options) >= 2 else (source, [])


_LATEX_SYMBOLS = {
    "times": "×", "div": "÷", "cdot": "·", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "neq": "≠", "pm": "±", "mp": "∓",
    "infty": "∞", "approx": "≈", "sim": "∼", "angle": "∠",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ",
    "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ", "omega": "ω",
    "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Omega": "Ω",
}
_SUPERSCRIPTS = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPTS = str.maketrans("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

def _latex_group(source: str, start: int) -> tuple[str, int]:
    """Read one braced LaTeX group; malformed input is kept visible."""
    if start >= len(source) or source[start] != "{":
        return "", start
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index], index + 1
    return source[start + 1:], len(source)


def _unicode_math_text(value: object) -> str:
    """Turn OCR LaTeX into customer-facing text without leaking source syntax.

    The UI keeps its richer, vertically typeset DOM preview. JSON is text-only,
    so fractions use U+2044 and unsupported scripts stay readable in brackets.
    """
    source = _clean_text(value).replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")

    def convert(text: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == "$":
                index += 1
                continue
            if char == "\\":
                match = re.match(r"\\([A-Za-z]+|.)", text[index:])
                if not match:
                    index += 1
                    continue
                command = match.group(1)
                index += len(match.group(0))
                if command == "frac":
                    numerator, index = _latex_group(text, index)
                    denominator, index = _latex_group(text, index)
                    output.append(f"{convert(numerator)}⁄{convert(denominator)}")
                elif command == "sqrt":
                    if index < len(text) and text[index] == "[":
                        close = text.find("]", index + 1)
                        index = len(text) if close < 0 else close + 1
                    radicand, index = _latex_group(text, index)
                    output.append(f"√({convert(radicand)})")
                elif command in {"left", "right", "displaystyle", "mathrm", "text", "quad", "qquad", ",", "!", " "}:
                    if command in {"mathrm", "text"}:
                        group, index = _latex_group(text, index)
                        output.append(convert(group))
                else:
                    output.append(_LATEX_SYMBOLS.get(command, command))
                continue
            if char in {"^", "_"}:
                script, next_index = _latex_group(text, index + 1)
                if not script and index + 1 < len(text):
                    script, next_index = text[index + 1], index + 2
                converted = convert(script)
                table = _SUPERSCRIPTS if char == "^" else _SUBSCRIPTS
                styled = converted.translate(table)
                output.append(styled if styled != converted else f"{'⁽' if char == '^' else '₍'}{converted}{'⁾' if char == '^' else '₎'}")
                index = next_index
                continue
            output.append(char)
            index += 1
        return "".join(output)

    return _clean_text(convert(source)).replace("{", "(").replace("}", ")")


def _k12_validation(sample: dict, image_sizes: list[tuple[int, int]]) -> list[str]:
    meta, body = sample["metainfo"], sample["data"]
    errors = []
    required_meta = {
        "language": "语言", "stage": "学段", "grade": "年级", "subject": "学科",
        "question_type": "题型", "difficulty": "难度",
    }
    for key, label in required_meta.items():
        if not str(meta.get(key, "")).strip():
            errors.append(f"缺少{label}")
    if not meta.get("knowledge_points"):
        errors.append("缺少知识点")
    for key, label in (("question", "题目文本"), ("answer", "标准答案"), ("analysis", "详细解析")):
        if not str(body.get(key, "")).strip():
            errors.append(f"缺少{label}")
    if not body.get("question_images") and not any(body.get("option_images", [])):
        errors.append("缺少有效题图")
    if any(min(width, height) < 128 for width, height in image_sizes):
        errors.append("图片最短边低于128像素")
    if meta.get("question_type") == "选择" and len(body.get("options", [])) < 2:
        errors.append("选择题选项不完整")
    return errors


def _write_k12_delivery(root: Path, project: dict, question_rows: list, answer_rows: list,
                        question_relations: dict[str, dict], *, changed_page_index: int | None,
                        materialize_all: bool) -> dict:
    """Build the customer-facing UTF-8 JSONL and meaningful illustration crops.

    Internal Q/A crops remain available for audit, but answer screenshots and whole-question
    screenshots are deliberately not referenced by this formal dataset.
    """
    k12_root = root / "k12_dataset"
    pages_root = k12_root / "pages"
    images_root = k12_root / "images"
    pages_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    answers_by_internal = {record["id"]: (index, page, record)
                           for index, page, record in answer_rows}
    continuations_by_parent: dict[str, list[tuple]] = {}
    for row in answer_rows:
        record = row[2]
        if record.get("region_role") == "continuation" and record.get("continuation_of"):
            continuations_by_parent.setdefault(str(record["continuation_of"]), []).append(row)

    def answer_chain(row: tuple) -> list[tuple]:
        chain, queue, seen = [], [row], set()
        while queue:
            current = queue.pop(0)
            internal_id = str(current[2].get("id", ""))
            if internal_id in seen:
                continue
            seen.add(internal_id)
            chain.append(current)
            queue.extend(sorted(continuations_by_parent.get(internal_id, []), key=_record_position))
        return chain
    ordered_questions = sorted(question_rows, key=_record_position)
    lines, report_items, active_images, page_lines = [], [], set(), {}
    page_orders = {}
    for global_order, (page_index, page, question) in enumerate(ordered_questions, 1):
        sample_id = f"{global_order:07d}"
        page_number = page_index + 1
        page_folder = f"P{page_number:04d}"
        page_orders[page_index] = page_orders.get(page_index, 0) + 1
        page_order = page_orders[page_index]
        image_root = images_root
        materialize_page = materialize_all or page_index == changed_page_index
        relation = question_relations.get(str(question.get("id", "")), {})
        primary_answer_infos = [answers_by_internal.get(str(value)) for value in relation.get("answer_ids", [])]
        primary_answer_infos = sorted((item for item in primary_answer_infos if item), key=_record_position)
        answer_infos = [part for item in primary_answer_infos for part in answer_chain(item)]
        answers = [_clean_text(item[2].get("answer")) for item in answer_infos]
        analyses = [_clean_text(item[2].get("analysis")) for item in answer_infos]
        question_images, option_images, image_sizes, image_targets = [], [], [], []
        illustrations = sorted(
            (item for item in question.get("illustrations", []) if isinstance(item, dict)),
            key=lambda item: (int(item.get("bbox", [0, 0, 0, 0])[1]), int(item.get("bbox", [0, 0, 0, 0])[0])),
        )
        q_index = 0
        option_index = 0
        for illustration in illustrations:
            kind = str(illustration.get("image_kind", "question_image"))
            if kind == "ignore":
                continue
            box = illustration.get("bbox", [])
            width = max(0, int(box[2]) - int(box[0])) if len(box) == 4 else 0
            height = max(0, int(box[3]) - int(box[1])) if len(box) == 4 else 0
            image_sizes.append((width, height))
            if kind == "option_image":
                label = re.sub(r"[^A-Za-z0-9]", "", str(illustration.get("option_label", ""))).upper() or "X"
                option_index += 1
                filename = f"{sample_id}_option_{label}_{option_index:02d}.jpg"
                option_images.append(f"images/{filename}")
            else:
                q_index += 1
                filename = f"{sample_id}_img_{q_index:02d}.jpg"
                question_images.append(f"images/{filename}")
            target_image = (image_root / filename).resolve()
            image_targets.append((target_image, box))
        ocr_question, inferred_options = _split_ocr_choices(question.get("text", ""))
        explicit_options = _string_list(question.get("options", []))
        options = explicit_options or inferred_options
        raw_type = _clean_text(question.get("question_type", ""))
        effective_type = "选择题" if inferred_options else raw_type
        sample = {
            "id": sample_id,
            "metainfo": {
                "language": _clean_text(project.get("language")),
                "stage": _clean_text(project.get("stage")),
                "grade": _clean_text(project.get("grade")),
                "subject": _clean_text(project.get("subject")),
                "question_type": K12_QUESTION_TYPES.get(effective_type, effective_type),
                "difficulty": _clean_text(question.get("difficulty", "")),
                "knowledge_points": _string_list(question.get("knowledge_points", [])),
            },
            "data": {
                "question_images": question_images,
                "option_images": option_images,
                "question": _unicode_math_text(ocr_question),
                "options": ([_unicode_math_text(value) for value in options]
                            if K12_QUESTION_TYPES.get(effective_type, effective_type) == "选择" else []),
                "answer": _unicode_math_text("\n".join(value for value in answers if value)),
                "analysis": _unicode_math_text("\n".join(value for value in analyses if value)),
            },
            "source": {
                "book_id": project["delivery_book_id"], "question_page": page_number,
                "page_order": page_order,
                "question_number": data.display_number(question),
                "internal_question_id": question.get("delivery_id", ""),
                "internal_answer_ids": [item[2].get("delivery_id", "") for item in answer_infos],
            },
        }
        errors = _k12_validation(sample, image_sizes)
        if str(page.get("status", "")) != "accepted":
            errors.append("题目页尚未确认")
        if relation.get("status") != "accepted" or not answer_infos:
            errors.append("题目尚未匹配答案")
        elif any(str(item[1].get("status", "")) != "accepted" for item in answer_infos):
            errors.append("答案页尚未确认")
        eligible = (str(page.get("status", "")) == "accepted"
                    and relation.get("status") == "accepted"
                    and bool(answer_infos) and all(str(item[1].get("status", "")) == "accepted" for item in answer_infos)
                    and not errors)
        report_items.append({"id": sample_id, "page": page_index + 1,
                             "question_number": data.display_number(question),
                             "eligible": eligible, "errors": errors})
        if eligible:
            for target, box in image_targets:
                if materialize_page or not target.is_file():
                    crop = _crop(data.source_path(project, page), box)
                    _atomic_image(target, crop)
            encoded = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
            lines.append(encoded)
            page_lines.setdefault(page_index, []).append(encoded)
            active_images.update(target for target, _box in image_targets)
        else:
            for target, _box in image_targets:
                if target.is_file():
                    target.unlink()
    for stale in images_root.glob("*.jpg"):
        if stale.resolve() not in active_images:
            stale.unlink()
    # Remove images emitted by the pre-contract page-folder layout as well.
    for stale in pages_root.glob("P*/images/*.jpg"):
        if stale.resolve() not in active_images:
            stale.unlink()
    # One folder per source page, including pages without accepted samples, so
    # operators can audit the book page-by-page without gaps.
    active_page_indexes = ({index for index in range(len(project.get("pages", [])))}
                           if materialize_all else ({changed_page_index} if changed_page_index is not None else set()))
    active_page_indexes.update(
        int(path.name[1:]) - 1 for path in pages_root.glob("P[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    active_page_indexes.update(page_lines)
    for page_index in active_page_indexes:
        page_dir = pages_root / f"P{page_index + 1:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_jsonl = page_dir / "dataset.jsonl"
        payload = "\n".join(page_lines.get(page_index, []))
        temporary_page = page_jsonl.with_suffix(".jsonl.tmp")
        temporary_page.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        os.replace(temporary_page, page_jsonl)
    dataset_path = k12_root / "dataset.jsonl"
    temporary = dataset_path.with_suffix(".jsonl.tmp")
    temporary.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    os.replace(temporary, dataset_path)
    report = {"schema_version": 1, "total_questions": len(ordered_questions),
              "eligible_count": len(lines), "rejected_count": len(ordered_questions) - len(lines),
              "items": report_items}
    _atomic_json(k12_root / "quality_report.json", report)
    return {"dataset_file": str(dataset_path), "eligible_count": len(lines),
            "rejected_count": report["rejected_count"]}


def delivery_root(project: dict, runtime_root: str | Path) -> Path:
    data.ensure_delivery_ids(project)
    configured_root = str(project.get("delivery_output_root", "") or "").strip()
    if configured_root:
        candidate = Path(configured_root).expanduser()
        if candidate.is_absolute():
            # Keep each book self-contained even when staff choose one shared
            # output folder for multiple textbooks.
            return candidate.resolve() / project["delivery_book_id"]
    # Batch preannotation projects live beside their original textbook pages.
    # Keep the default generated images/JSON there as well, while still using
    # a dedicated book-ID folder so source scans never confuse delivery crops
    # with new textbook pages.
    book_root = Path(str(project.get("book_root") or project.get("image_root") or "")).expanduser()
    if book_root.is_absolute() and book_root.is_dir():
        return book_root.resolve() / project["delivery_book_id"]
    return Path(runtime_root).resolve() / "delivery_packages" / project["delivery_book_id"]


def _source_signature(project: dict, page: dict, record: dict, answer: bool) -> str:
    source = Path(data.source_path(project, page, answer)).resolve()
    stat = source.stat()
    payload = {
        "source": os.path.normcase(str(source)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "bbox": record.get("bbox", []),
        "segments": record.get("shared_context_segments", []),
        "shared": record.get("shared_context_bbox", []),
        "use_shared": bool(record.get("use_shared_context")),
        "continuation": record.get("continuation_of", ""),
    }
    if not answer:
        # Bump whenever the composed question-image pixel layout changes so
        # an incremental sync rewrites already-materialized crops.
        payload["composition_version"] = "edge_to_edge_v1"
        continuation_sources = []
        for _index, continuation_page, continuation in data.all_questions(project):
            if (continuation.get("region_role") != "continuation"
                    or continuation.get("continuation_of") != record.get("id")):
                continue
            continuation_path = Path(data.source_path(project, continuation_page)).resolve()
            continuation_stat = continuation_path.stat()
            continuation_sources.append({
                "source": os.path.normcase(str(continuation_path)),
                "size": continuation_stat.st_size,
                "mtime_ns": continuation_stat.st_mtime_ns,
                "bbox": continuation.get("bbox", []),
            })
        payload["continuation_sources"] = continuation_sources
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _source(page_index: int, page: dict, record: dict) -> dict:
    return {
        "page_index": page_index,
        "page_image": str(page.get("image", "")),
        "annotation_id": str(record.get("id", "")),
        "bbox": list(record.get("bbox", [])),
    }


def _ocr(record: dict) -> dict:
    value = record.get("ocr", {})
    if not isinstance(value, dict):
        value = {}
    return {
        "status": str(value.get("status", "not_run")),
        "provider": str(value.get("provider", "")),
        "engine": str(value.get("engine", "")),
        "model": str(value.get("model", "")),
        "confidence": float(value.get("confidence", 0.0) or 0.0),
        "updated_at": str(value.get("updated_at", "")),
    }


def _record_document(project: dict, page_index: int, page: dict, record: dict,
                     kind: str, relations: dict[str, dict]) -> dict:
    public_id = record["delivery_id"]
    page_folder = f"P{page_index + 1:04d}"
    relation = relations.get(str(record.get("id", "")), {})
    document = {
        "schema_version": 2,
        "id": public_id,
        "kind": kind,
        "book_id": project["delivery_book_id"],
        "review_status": str(page.get("status", "unlabeled")),
        "number_path": data.number_path(record),
        "display_number": data.display_number(record),
        # The editable OCR field is the authoritative delivery text. Keep the
        # local project source (including LaTeX where useful for re-rendering),
        # but never leak that source representation into a generated JSON.
        "text": _unicode_math_text(record.get("text", "")),
        "question_type": str(record.get("question_type", "")),
        "ocr": _ocr(record),
        "image_file": f"images/{kind}s/{page_folder}/{public_id}.jpg",
        "source": _source(page_index, page, record),
    }
    if kind == "question":
        matched = (
            str(relation.get("status", "")) == "accepted"
            and bool(relation.get("public_answer_ids", []))
        )
        document.update({
            "relation_id": str(relation.get("delivery_id", "")) if matched else "",
            "relation_status": "accepted" if matched else "pending_generation",
            "answer_status": "matched" if matched else "pending_generation",
            "answer_ids": list(relation.get("public_answer_ids", [])) if matched else [],
        })
    else:
        document["question_ids"] = list(relation.get("public_question_ids", []))
    return document


def sync_delivery_workspace(project_path: str | Path, runtime_root: str | Path,
                            *, changed_kind: str | None = None,
                            changed_page_index: int | None = None,
                            materialize_all: bool = False,
                            changed_record_ids: list[str] | None = None) -> dict:
    """Sync edited delivery artifacts and rebuild the small authoritative indexes.

    Crops are rewritten only when their source signature changes. JSON is cheap and is
    refreshed for the changed page plus records touched by relations.
    """
    project_file = Path(project_path).resolve()
    project = data.load_project(project_file)
    data.ensure_delivery_ids(project)
    root = delivery_root(project, runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    old_manifest_path = root / "manifest.json"
    try:
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        old_manifest = {}
    old_items = {item.get("id"): item for item in old_manifest.get("items", []) if isinstance(item, dict)}
    targeted_internal_ids = {str(value) for value in (changed_record_ids or []) if str(value)}

    question_rows = sorted(
        [row for row in data.all_questions(project) if row[2].get("region_role", "question") == "question"],
        key=_record_position,
    )
    answer_rows = sorted(
        [row for row in data.all_answers(project)
         if row[2].get("region_role", "question") in {"question", "continuation"}],
        key=_record_position,
    )
    questions_by_internal = {record["id"]: (index, page, record) for index, page, record in question_rows}
    answers_by_internal = {record["id"]: (index, page, record) for index, page, record in answer_rows}

    question_relations: dict[str, dict] = {}
    answer_relations: dict[str, dict] = {}
    for pair in project.get("pairs", []):
        if str(pair.get("status", "")) != "accepted" or not pair.get("answer_ids"):
            continue
        question = questions_by_internal.get(str(pair.get("question_id", "")))
        answers = [answers_by_internal.get(str(value)) for value in pair.get("answer_ids", [])]
        if (
            not question
            or str(question[1].get("status", "")) != "accepted"
            or not answers
            or any(row is None or str(row[1].get("status", "")) != "accepted" for row in answers)
        ):
            continue
        public_answers = [row[2]["delivery_id"] for row in answers if row]
        question_relations[question[2]["id"]] = {**pair, "public_answer_ids": public_answers}
        for answer in (row for row in answers if row):
            relation = answer_relations.setdefault(answer[2]["id"], {"public_question_ids": []})
            relation["public_question_ids"].append(question[2]["delivery_id"])

    selected_kind = "answer" if changed_kind == "answer" else "question"
    changed_ids: set[str] = set()
    if changed_page_index is not None:
        rows = answer_rows if selected_kind == "answer" else question_rows
        changed_ids = {
            record["delivery_id"] for index, _page, record in rows
            if index == int(changed_page_index)
        }

    documents: dict[str, dict] = {}
    manifest_items = []
    crop_updates = 0
    for kind, rows, relations in (
        ("question", question_rows, question_relations),
        ("answer", answer_rows, answer_relations),
    ):
        if changed_page_index is not None and kind == selected_kind:
            changed_folder = f"P{int(changed_page_index) + 1:04d}"
            (root / "images" / f"{kind}s" / changed_folder).mkdir(parents=True, exist_ok=True)
            (root / "json" / f"{kind}s" / changed_folder).mkdir(parents=True, exist_ok=True)
        for page_index, page, record in rows:
            public_id = record["delivery_id"]
            page_folder = f"P{page_index + 1:04d}"
            signature = _source_signature(project, page, record, kind == "answer")
            image_path = root / "images" / f"{kind}s" / page_folder / f"{public_id}.jpg"
            json_path = root / "json" / f"{kind}s" / page_folder / f"{public_id}.json"
            old = old_items.get(public_id, {})
            current_image_file = f"images/{kind}s/{page_folder}/{public_id}.jpg"
            current_json_file = f"json/{kind}s/{page_folder}/{public_id}.json"
            for old_key, current_file in (("image_file", current_image_file), ("json_file", current_json_file)):
                old_file = str(old.get(old_key, ""))
                if old_file and old_file != current_file:
                    old_target = _delivery_artifact_path(root, old_file)
                    if old_target is not None and old_target.is_file():
                        old_target.unlink()
            on_changed_page = (
                changed_page_index is not None
                and kind == selected_kind and page_index == int(changed_page_index)
            )
            explicitly_changed = str(record.get("id", "")) in targeted_internal_ids
            should_materialize = (
                (materialize_all and str(page.get("status", "")) == "accepted")
                or on_changed_page
                or explicitly_changed
                or image_path.is_file()
            )
            if should_materialize and (old.get("source_signature") != signature or not image_path.is_file()):
                crop = (_crop(data.source_path(project, page, True), record["bbox"])
                        if kind == "answer" else _question_crop(project, page, record))
                _atomic_image(image_path, crop)
                crop_updates += 1
            document = _record_document(project, page_index, page, record, kind, relations)
            documents[public_id] = document
            should_write_json = (
                materialize_all or on_changed_page or explicitly_changed
                or public_id in changed_ids
            )
            if should_write_json:
                _atomic_json(json_path, document)
            manifest_items.append({
                "id": public_id, "kind": kind, "page_index": page_index,
                "review_status": document["review_status"],
                "json_file": current_json_file,
                "image_file": document["image_file"], "source_signature": signature,
                "materialized": bool(image_path.is_file() and json_path.is_file()),
            })

    current_ids = set(documents)
    removed = 0
    for stale_id, item in old_items.items():
        if stale_id in current_ids:
            continue
        for key in ("json_file", "image_file"):
            target = _delivery_artifact_path(root, item.get(key, ""))
            if target is not None and target.is_file():
                target.unlink()
        removed += 1

    ready_questions = []
    ready_answer_ids: set[str] = set()
    for _page_index, page, question in question_rows:
        question_document = documents[question["delivery_id"]]
        if question_document["review_status"] != "accepted":
            continue
        ready_questions.append(question_document)
        if question_document["answer_status"] != "matched":
            continue
        relation = question_relations.get(question["id"], {})
        answer_rows_for_pair = [answers_by_internal.get(str(value)) for value in relation.get("answer_ids", [])]
        if any(row is None or str(row[1].get("status", "")) != "accepted" for row in answer_rows_for_pair):
            question_document.update({
                "relation_id": "", "relation_status": "pending_generation",
                "answer_status": "pending_generation", "answer_ids": [],
            })
            continue
        ready_answer_ids.update(question_document["answer_ids"])
    ready_answers = [documents[value] for value in sorted(ready_answer_ids) if value in documents]
    database = {
        "schema_version": 2,
        "kind": "papercutter_database_delivery",
        "book": {
            "id": project["delivery_book_id"],
            "source_root": str(project.get("book_root", project.get("image_root", ""))),
            "isbn": str(project.get("isbn", "")),
            "subject": str(project.get("subject", "")),
            "grade": str(project.get("grade", "")),
        },
        "questions": ready_questions,
        "answers": ready_answers,
    }
    _atomic_json(root / "database_import.json", database)
    manifest = {
        "schema_version": 2,
        "book_id": project["delivery_book_id"],
        "project_path": str(project_file),
        "question_count": len(question_rows),
        "answer_count": len(answer_rows),
        "ready_question_count": len(ready_questions),
        "items": manifest_items,
        "database_file": "database_import.json",
    }
    _atomic_json(old_manifest_path, manifest)
    k12 = _write_k12_delivery(
        root, project, question_rows, answer_rows, question_relations,
        changed_page_index=(changed_page_index if selected_kind == "question" else None),
        materialize_all=materialize_all,
    )
    # Delivery generation may assign missing public IDs, but it is part of the
    # caller's logical mutation and must not advance the renderer CAS revision
    # a second time.
    data.save_project(project, project_file, increment_revision=False)
    return {
        "output_path": str(root), "book_id": project["delivery_book_id"],
        "question_count": len(question_rows), "answer_count": len(answer_rows),
        "ready_question_count": len(ready_questions), "crop_updates": crop_updates,
        "removed_count": removed, "k12": k12,
    }
